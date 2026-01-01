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
    run_dir = os.path.join(config.output_dir, "directional_deviation", run_id)
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
    if shift_mode not in {"single_channel", "subset"}:
        raise ValueError(f"Unsupported shift_mode in anomaly_manifest: {shift_mode}.")

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
            shift_mode=shift_mode,
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
        shift_mode=shift_mode,
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
        shift_mode=shift_mode,
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
) -> Dict[str, Any]:
    evaluation_path = os.path.join(model_dir, "evaluation.json")
    evaluation = {
        "model_name": model_name,
        "model_type": None,
        "scoring_mode": scoring_mode,
        "metrics": None,
        "per_channel": None,
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
) -> Dict[str, Any]:
    model_dir = os.path.join(run_dir, "models", model_config.name)
    os.makedirs(model_dir, exist_ok=True)

    scoring_mode = _determine_scoring_mode(model_config)
    try:
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
        )


def _run_leaf_experiment(
    config: DirectionalDeviationExperimentConfig,
    run_dir: str,
    batch_size: int,
) -> Dict[str, Any]:
    _configure_determinism(config.thresholds.deterministic_algorithms)

    set_random_seed(config.data.seed)
    normal_data = generate_normal_data(config.data, config.device)
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

    evaluations_by_model_name: Dict[str, Dict[str, Any]] = {}
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
        width_summaries[int(width)] = _run_experiment_dispatch(
            config=width_config,
            run_dir=width_dir,
            batch_size=batch_size,
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
    return summary


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
