import json
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor

from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationExperimentConfig,
    DirectionalDeviationModelVariantConfig,
    load_experiment_config,
)
from sgnsae.experiments.directional_deviation.data import (
    build_feature_directions_from_decoder,
    export_data_summary,
    generate_anomaly_test_data,
    generate_normal_data,
    set_random_seed,
)
from sgnsae.experiments.directional_deviation.evaluation import (
    DirectionalDeviationEvaluationInputs,
    build_comparison_summary,
    evaluate_and_persist_directional_deviation,
    persist_comparison_summary,
)
from sgnsae.experiments.directional_deviation.scoring import (
    BaselineEvidenceThresholds,
    ScoreThresholds,
    SignAwareLatentThresholds,
    assign_baseline_latent_directions,
    build_threshold_metadata,
    calibrate_baseline_evidence_thresholds,
    calibrate_score_thresholds,
    calibrate_sign_aware_latent_thresholds,
    calculate_baseline_directional_scores,
    calculate_latent_argmax_channel,
    calculate_sign_aware_directional_scores,
    extract_detection_signal,
    extract_one_sided_evidence,
    persist_baseline_threshold_artifacts,
    persist_sign_aware_threshold_artifacts,
)
from sgnsae.experiments.directional_deviation.training import (
    persist_model_artifacts,
    run_training_for_variant,
)
from sgnsae.experiments.directional_deviation.plotting import generate_plots_for_run
from sgnsae.experiments.directional_deviation.decoder_alignment import (
    persist_decoder_alignment_diagnostics,
)


_DEFAULT_BATCH_SIZE = 4096


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def _prepare_run_directory(
    config: DirectionalDeviationExperimentConfig,
) -> tuple[str, str]:
    run_id = config.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    normalized_output = os.path.normpath(config.output_dir)
    run_parent = (
        normalized_output
        if os.path.basename(normalized_output) == "directional_deviation"
        else os.path.join(normalized_output, "directional_deviation")
    )
    run_dir = os.path.join(run_parent, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


def _persist_run_config(
    config: DirectionalDeviationExperimentConfig, run_dir: str
) -> None:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        sanitized_config = asdict(config)
        sanitized_config["wandb_api_key"] = None
        json.dump(sanitized_config, f, indent=2)


def _configure_determinism(enabled: bool) -> None:
    torch.use_deterministic_algorithms(enabled)
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _iterate_tensor_batches(x: Tensor, batch_size: int) -> Iterator[Tensor]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; received {batch_size}.")
    for start in range(0, x.shape[0], batch_size):
        yield x[start : start + batch_size]


def _stable_model_seed(base_seed: int, model_name: str) -> int:
    digest = sha256(model_name.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((base_seed + derived) % (2**63))


def _determine_scoring_mode(
    model_config: DirectionalDeviationModelVariantConfig,
) -> Literal["sign_aware", "baseline"]:
    return "sign_aware" if model_config.sign_aware else "baseline"


def _extract_validation_signal_cpu(
    model: torch.nn.Module,
    scoring_mode: Literal["sign_aware", "baseline"],
    x_val: Tensor,
    batch_size: int,
) -> Tensor:
    model.eval()
    outputs: List[Tensor] = []
    with torch.no_grad():
        for batch in _iterate_tensor_batches(x_val, batch_size=batch_size):
            if scoring_mode == "sign_aware":
                signal = extract_detection_signal(model, batch)
            else:
                signal = extract_one_sided_evidence(model, batch)
            outputs.append(signal.detach().to("cpu"))
    if not outputs:
        raise ValueError("Validation tensor produced zero batches.")
    return torch.cat(outputs, dim=0)


def _calibrate_and_persist_thresholds(
    experiment_config: DirectionalDeviationExperimentConfig,
    model: torch.nn.Module,
    scoring_mode: Literal["sign_aware", "baseline"],
    anomaly_manifest: Dict[str, Any],
    x_val: Tensor,
    model_dir: str,
    batch_size: int,
) -> Tuple[
    ScoreThresholds,
    Optional[SignAwareLatentThresholds],
    Optional[BaselineEvidenceThresholds],
]:
    thresholds_dir = os.path.join(model_dir, "thresholds")
    latent_quantile = float(experiment_config.thresholds.latent_quantile)
    min_validation_samples = int(experiment_config.thresholds.min_validation_samples)
    score_fpr = float(experiment_config.thresholds.score_fpr)

    shift_mode = anomaly_manifest.get("shift_mode")
    if shift_mode not in {"single_channel", "subset", "feature_direction"}:
        raise ValueError(f"Unsupported shift_mode in anomaly_manifest: {shift_mode}.")

    baseline_shift_mode: Literal["single_channel", "subset"] = (
        "single_channel" if shift_mode == "feature_direction" else shift_mode
    )

    subset_indices_raw = anomaly_manifest.get("subset_indices")
    subset_indices = (
        None
        if subset_indices_raw is None
        else (
            [int(value) for value in subset_indices_raw]
            if isinstance(subset_indices_raw, list)
            else None
        )
    )

    pooling = experiment_config.scoring.pooling
    topk_k = experiment_config.scoring.topk_k
    targets: Optional[Tensor] = None
    latent_argmax: Optional[Tensor] = None
    if experiment_config.scoring.target_restricted_scoring:
        channels_to_evaluate = anomaly_manifest.get("channels_to_evaluate")
        if not isinstance(channels_to_evaluate, list) or not channels_to_evaluate:
            raise ValueError(
                "channels_to_evaluate must be present in anomaly_manifest when target_restricted_scoring=True."
            )
        channels_to_evaluate_ints = [int(value) for value in channels_to_evaluate]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(experiment_config.data.seed + 11)
        permutation = torch.randperm(
            len(channels_to_evaluate_ints), generator=generator
        )
        shuffled = torch.tensor(channels_to_evaluate_ints, dtype=torch.int64)[
            permutation
        ]
        targets = shuffled[
            torch.arange(int(x_val.shape[0]), dtype=torch.int64) % shuffled.shape[0]
        ]
        latent_argmax = calculate_latent_argmax_channel(model).to("cpu")

    if scoring_mode == "sign_aware":
        pi_val = _extract_validation_signal_cpu(
            model=model,
            scoring_mode=scoring_mode,
            x_val=x_val,
            batch_size=batch_size,
        )
        latent_thresholds_cpu = calibrate_sign_aware_latent_thresholds(
            pi_validation=pi_val,
            latent_quantile=latent_quantile,
            min_validation_samples=min_validation_samples,
        )
        score_high_val, score_low_val = calculate_sign_aware_directional_scores(
            pi=pi_val,
            latent_thresholds=latent_thresholds_cpu,
            pooling=pooling,
            topk_k=topk_k,
            targets=targets,
            latent_argmax_channel=latent_argmax,
        )
        score_thresholds = calibrate_score_thresholds(
            score_high_validation=score_high_val,
            score_low_validation=score_low_val,
            score_fpr=score_fpr,
            min_validation_samples=min_validation_samples,
        )
        metadata = build_threshold_metadata(
            model=model,
            scoring_mode=scoring_mode,
            latent_quantile=latent_quantile,
            score_thresholds=score_thresholds,
            pooling=pooling,
            topk_k=topk_k,
            target_restricted_scoring=experiment_config.scoring.target_restricted_scoring,
            shift_mode=baseline_shift_mode,
            subset_indices=subset_indices,
        )
        persist_sign_aware_threshold_artifacts(
            thresholds_dir=thresholds_dir,
            latent_thresholds=latent_thresholds_cpu,
            score_thresholds=score_thresholds,
            metadata=metadata,
        )
        latent_thresholds = SignAwareLatentThresholds(
            tau_plus=latent_thresholds_cpu.tau_plus.to(device=x_val.device),
            tau_minus=latent_thresholds_cpu.tau_minus.to(device=x_val.device),
        )
        return score_thresholds, latent_thresholds, None

    evidence_val = _extract_validation_signal_cpu(
        model=model,
        scoring_mode=scoring_mode,
        x_val=x_val,
        batch_size=batch_size,
    )
    latent_directions = assign_baseline_latent_directions(
        model=model,
        shift_mode=baseline_shift_mode,
        subset_indices=subset_indices,
    ).to("cpu")
    baseline_thresholds_cpu = calibrate_baseline_evidence_thresholds(
        evidence_validation=evidence_val,
        latent_directions=latent_directions,
        latent_quantile=latent_quantile,
        min_validation_samples=min_validation_samples,
    )
    score_high_val, score_low_val = calculate_baseline_directional_scores(
        one_sided_evidence=evidence_val,
        baseline_thresholds=baseline_thresholds_cpu,
        pooling=pooling,
        topk_k=topk_k,
        targets=targets,
        latent_argmax_channel=latent_argmax,
    )
    score_thresholds = calibrate_score_thresholds(
        score_high_validation=score_high_val,
        score_low_validation=score_low_val,
        score_fpr=score_fpr,
        min_validation_samples=min_validation_samples,
    )
    metadata = build_threshold_metadata(
        model=model,
        scoring_mode=scoring_mode,
        latent_quantile=latent_quantile,
        score_thresholds=score_thresholds,
        pooling=pooling,
        topk_k=topk_k,
        target_restricted_scoring=experiment_config.scoring.target_restricted_scoring,
        shift_mode=baseline_shift_mode,
        subset_indices=subset_indices,
    )
    persist_baseline_threshold_artifacts(
        thresholds_dir=thresholds_dir,
        baseline_thresholds=baseline_thresholds_cpu,
        score_thresholds=score_thresholds,
        metadata=metadata,
    )
    baseline_thresholds = BaselineEvidenceThresholds(
        evidence_tau=baseline_thresholds_cpu.evidence_tau.to(device=x_val.device),
        latent_directions=baseline_thresholds_cpu.latent_directions.to(
            device=x_val.device
        ),
    )
    return score_thresholds, None, baseline_thresholds


def _persist_failed_evaluation(
    model_dir: str,
    model_name: str,
    scoring_mode: Literal["sign_aware", "baseline"],
    error: Exception,
    decoder_alignment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evaluation_path = os.path.join(model_dir, "evaluation.json")
    evaluation = {
        "model_name": model_name,
        "model_type": None,
        "scoring_mode": scoring_mode,
        "metrics": None,
        "per_channel": None,
        "decoder_alignment": decoder_alignment,
        "error": f"{type(error).__name__}: {error}",
    }
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    return evaluation


def _run_single_model(
    experiment_config: DirectionalDeviationExperimentConfig,
    run_dir: str,
    model_config: DirectionalDeviationModelVariantConfig,
    normal_data: Dict[str, Tensor],
    anomaly_data: Dict[str, Any],
    batch_size: int,
    training_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_dir = os.path.join(run_dir, "models", model_config.name)
    os.makedirs(model_dir, exist_ok=True)

    scoring_mode = _determine_scoring_mode(model_config)
    decoder_alignment: Optional[Dict[str, Any]] = None
    try:
        if training_output is None:
            training_output = run_training_for_variant(
                experiment_config=experiment_config,
                data=normal_data,
                model_config=model_config,
                device=experiment_config.device,
            )
        model = training_output["model"]
        persist_model_artifacts(
            model=model,
            training_metrics=training_output["training_metrics"],
            model_config=model_config,
            model_dir=model_dir,
        )
        decoder_alignment = persist_decoder_alignment_diagnostics(
            model=model,
            model_dir=model_dir,
        )

        score_thresholds, latent_thresholds, baseline_thresholds = (
            _calibrate_and_persist_thresholds(
                experiment_config=experiment_config,
                model=model,
                scoring_mode=scoring_mode,
                anomaly_manifest=anomaly_data["anomaly_manifest"],
                x_val=normal_data["x_val"],
                model_dir=model_dir,
                batch_size=batch_size,
            )
        )
        evaluation_inputs = DirectionalDeviationEvaluationInputs(
            model=model,
            model_name=model_config.name,
            scoring_mode=scoring_mode,
            anomaly_data=anomaly_data,
            x_test_normal=normal_data["x_test_normal"],
            score_thresholds=score_thresholds,
            evaluation_batch_size=batch_size,
            coverage_recall_target=float(experiment_config.coverage_recall_target),
            pooling=experiment_config.scoring.pooling,
            topk_k=experiment_config.scoring.topk_k,
            target_restricted_scoring=experiment_config.scoring.target_restricted_scoring,
            decoder_alignment=decoder_alignment,
            latent_thresholds=latent_thresholds,
            baseline_thresholds=baseline_thresholds,
        )
        return evaluate_and_persist_directional_deviation(
            model_dir=model_dir,
            inputs=evaluation_inputs,
        )
    except Exception as exc:
        return _persist_failed_evaluation(
            model_dir=model_dir,
            model_name=model_config.name,
            scoring_mode=scoring_mode,
            error=exc,
            decoder_alignment=decoder_alignment,
        )


def _run_leaf_experiment(
    config: DirectionalDeviationExperimentConfig,
    run_dir: str,
    batch_size: int,
) -> Dict[str, Any]:
    _configure_determinism(config.thresholds.deterministic_algorithms)

    set_random_seed(config.data.seed)
    normal_data = generate_normal_data(config.data, config.device)

    evaluations_by_model_name: Dict[str, Dict[str, Any]] = {}
    if (
        config.anomaly.shift_mode == "feature_direction"
        and config.anomaly.direction_source == "per_model_decoder"
    ):
        raise ValueError(
            "direction_source='per_model_decoder' is not supported by the runner. "
            "Use direction_source='reference_model_decoder' or 'fixed_random'."
        )
    if (
        config.anomaly.shift_mode == "feature_direction"
        and config.anomaly.direction_source == "reference_model_decoder"
    ):
        reference_model_name = config.anomaly.reference_model_name
        reference_model_config = next(
            (
                model_config
                for model_config in config.models
                if model_config.name == reference_model_name
            ),
            None,
        )
        if reference_model_config is None:
            raise ValueError(
                "reference_model_name must match a model name in config.models; "
                f"received reference_model_name='{reference_model_name}'."
            )

        set_random_seed(_stable_model_seed(config.data.seed, reference_model_name))
        reference_training_output = run_training_for_variant(
            experiment_config=config,
            data=normal_data,
            model_config=reference_model_config,
            device=config.device,
        )
        direction_seed = config.data.seed + 6
        anomaly_directions, direction_metadata = build_feature_directions_from_decoder(
            model=reference_training_output["model"],
            anomaly_config=config.anomaly,
            direction_seed=direction_seed,
        )
        anomaly_data = generate_anomaly_test_data(
            data_config=config.data,
            anomaly_config=config.anomaly,
            device=config.device,
            batch_size=batch_size,
            anomaly_directions=anomaly_directions,
            direction_metadata=direction_metadata,
        )
        export_data_summary(
            run_dir=run_dir,
            data_config=config.data,
            anomaly_config=config.anomaly,
            normal_data=normal_data,
            anomaly_data=anomaly_data,
        )

        for model_config in config.models:
            set_random_seed(_stable_model_seed(config.data.seed, model_config.name))
            evaluations_by_model_name[model_config.name] = _run_single_model(
                experiment_config=config,
                run_dir=run_dir,
                model_config=model_config,
                normal_data=normal_data,
                anomaly_data=anomaly_data,
                batch_size=batch_size,
                training_output=(
                    reference_training_output
                    if model_config.name == reference_model_name
                    else None
                ),
            )
    else:
        anomaly_data = generate_anomaly_test_data(
            data_config=config.data,
            anomaly_config=config.anomaly,
            device=config.device,
            batch_size=batch_size,
        )
        export_data_summary(
            run_dir=run_dir,
            data_config=config.data,
            anomaly_config=config.anomaly,
            normal_data=normal_data,
            anomaly_data=anomaly_data,
        )

        for model_config in config.models:
            set_random_seed(_stable_model_seed(config.data.seed, model_config.name))
            evaluations_by_model_name[model_config.name] = _run_single_model(
                experiment_config=config,
                run_dir=run_dir,
                model_config=model_config,
                normal_data=normal_data,
                anomaly_data=anomaly_data,
                batch_size=batch_size,
            )
    persist_comparison_summary(run_dir, evaluations_by_model_name)
    try:
        generate_plots_for_run(run_dir)
    except Exception as exc:
        if config.verbose:
            print(f"Plot generation failed for run_dir={run_dir}: {exc}")
    return build_comparison_summary(evaluations_by_model_name)


def _aggregate_seed_sweep(
    per_seed_summaries: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    metrics: Sequence[str] = (
        "recall_high",
        "recall_low",
        "fpr_high",
        "fpr_low",
        "covered_channels_fraction",
        "covered_channels_count",
        "unique_channels_claimed",
        "purity_mean",
        "purity_median",
        "purity_p10",
        "purity_p50",
        "purity_p90",
    )
    per_model_values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed_summary in per_seed_summaries.values():
        models = seed_summary.get("models")
        if not isinstance(models, dict):
            continue
        for model_name, model_metrics in models.items():
            if not isinstance(model_metrics, dict):
                continue
            for metric_name in metrics:
                value = model_metrics.get(metric_name)
                if value is None:
                    continue
                per_model_values[model_name][metric_name].append(float(value))

    aggregate_models: Dict[str, Any] = {}
    for model_name, metrics_by_name in per_model_values.items():
        aggregate_models[model_name] = {}
        for metric_name in metrics:
            values = metrics_by_name.get(metric_name, [])
            aggregate_models[model_name][f"{metric_name}_mean"] = (
                None if not values else statistics.mean(values)
            )
            aggregate_models[model_name][f"{metric_name}_std"] = (
                None if not values else statistics.pstdev(values)
            )
            aggregate_models[model_name][f"{metric_name}_n"] = len(values)

    return {"models": aggregate_models, "std_ddof": 0}


def _override_models_for_width(
    models: List[DirectionalDeviationModelVariantConfig], width: int
) -> List[DirectionalDeviationModelVariantConfig]:
    return [replace(model_config, d_hidden=width) for model_config in models]


def _run_seed_sweep(
    config: DirectionalDeviationExperimentConfig,
    run_dir: str,
    batch_size: int,
) -> Dict[str, Any]:
    if config.data.seeds is None:
        raise ValueError("Expected config.data.seeds to be set for seed sweep.")

    seed_summaries: Dict[int, Dict[str, Any]] = {}
    for seed in config.data.seeds:
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        seed_config = replace(
            config,
            data=replace(config.data, seed=int(seed), seeds=None),
        )
        _persist_run_config(seed_config, seed_dir)
        seed_summaries[int(seed)] = _run_leaf_experiment(
            config=seed_config,
            run_dir=seed_dir,
            batch_size=batch_size,
        )

    summary = {
        "sweep": {"type": "seed", "seeds": [int(value) for value in config.data.seeds]},
        "per_seed": {str(seed): seed_summaries[seed] for seed in seed_summaries},
        "aggregate_over_seeds": _aggregate_seed_sweep(seed_summaries),
    }
    summary_path = os.path.join(run_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    try:
        generate_plots_for_run(run_dir)
    except Exception as exc:
        if config.verbose:
            print(f"Plot generation failed for seed sweep run_dir={run_dir}: {exc}")
    return summary


def _run_width_sweep(
    config: DirectionalDeviationExperimentConfig,
    run_dir: str,
    batch_size: int,
) -> Dict[str, Any]:
    if config.width_sweep is None:
        raise ValueError("Expected config.width_sweep to be set for width sweep.")

    width_summaries: Dict[int, Dict[str, Any]] = {}
    for width in config.width_sweep:
        width_dir = os.path.join(run_dir, f"width_{width}")
        os.makedirs(width_dir, exist_ok=True)
        width_config = replace(
            config,
            width_sweep=None,
            models=_override_models_for_width(config.models, int(width)),
        )
        _persist_run_config(width_config, width_dir)
        width_summary = _run_experiment_dispatch(
            config=width_config,
            run_dir=width_dir,
            batch_size=batch_size,
        )
        width_summaries[int(width)] = width_summary
        width_metrics_path = os.path.join(width_dir, "metrics.json")
        with open(width_metrics_path, "w") as f:
            json.dump(
                {
                    "width": int(width),
                    "models": _extract_models_metrics_for_metrics_json(width_summary),
                },
                f,
                indent=2,
            )

    summary = {
        "sweep": {
            "type": "width",
            "widths": [int(value) for value in config.width_sweep],
        },
        "per_width": {str(width): width_summaries[width] for width in width_summaries},
    }
    summary_path = os.path.join(run_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _persist_per_width_metrics(run_dir=run_dir, width_summaries=width_summaries)
    try:
        generate_plots_for_run(run_dir)
    except Exception as exc:
        if config.verbose:
            print(f"Plot generation failed for width sweep run_dir={run_dir}: {exc}")
    return summary


def _extract_models_metrics_for_metrics_json(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    models = summary.get("models") if isinstance(summary, dict) else None
    if isinstance(models, dict):
        return models
    aggregate = (
        summary.get("aggregate_over_seeds") if isinstance(summary, dict) else None
    )
    if isinstance(aggregate, dict) and isinstance(aggregate.get("models"), dict):
        return aggregate["models"]
    return {}


def _persist_per_width_metrics(
    run_dir: str,
    width_summaries: Dict[int, Dict[str, Any]],
) -> None:
    per_width_models: Dict[str, Any] = {}
    widths_sorted = sorted(width_summaries.keys())
    for width in widths_sorted:
        per_width_models[str(width)] = _extract_models_metrics_for_metrics_json(
            width_summaries[width]
        )

    output_path = os.path.join(run_dir, "per-width metrics.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "widths": widths_sorted,
                "per_width": per_width_models,
            },
            f,
            indent=2,
        )


def _run_experiment_dispatch(
    config: DirectionalDeviationExperimentConfig,
    run_dir: str,
    batch_size: int,
) -> Dict[str, Any]:
    if config.width_sweep is not None:
        return _run_width_sweep(config=config, run_dir=run_dir, batch_size=batch_size)
    if config.data.seeds is not None:
        return _run_seed_sweep(config=config, run_dir=run_dir, batch_size=batch_size)
    return _run_leaf_experiment(config=config, run_dir=run_dir, batch_size=batch_size)


def run_directional_deviation_experiment(config_path: str) -> str:
    raw_config = load_experiment_config(config_path)
    device = _resolve_device(raw_config.device)
    config = replace(raw_config, device=device)
    run_dir, _ = _prepare_run_directory(config)
    _persist_run_config(config, run_dir)
    _run_experiment_dispatch(
        config=config, run_dir=run_dir, batch_size=_DEFAULT_BATCH_SIZE
    )
    return run_dir
