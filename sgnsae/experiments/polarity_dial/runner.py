import json
import os
import shutil
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, List, Optional

import torch

from sgnsae.experiments.polarity_dial.config import (
    PolarityDialExperimentConfig,
    PolarityDialModelVariantConfig,
    load_experiment_config,
)
from sgnsae.experiments.polarity_dial.data import (
    export_data_summary,
    generate_polarity_dial_dataset,
    set_random_seed,
)
from sgnsae.experiments.polarity_dial.environment import persist_environment_metadata
from sgnsae.experiments.polarity_dial.evaluation import (
    align_decoder_and_features,
    compute_alignment_and_calibration,
    persist_alignment_and_calibration,
)
from sgnsae.experiments.polarity_dial.plotting import (
    plot_asymmetry_sweep_calibration,
    plot_calibration_scatter,
    plot_gamma_comparison,
    plot_gamma_histograms,
    plot_gamma_strip,
    plot_multi_sparsity_loss_summaries,
    plot_mse_vs_l0_tradeoff,
    plot_training_loss,
)
from sgnsae.experiments.polarity_dial.sweeps import (
    run_asymmetry_sweep,
    run_sparsity_sweep,
)
from sgnsae.experiments.polarity_dial.training import (
    persist_model_artifacts,
    run_training_for_variant,
    slice_trained_model_instance,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def _stable_model_seed(base_seed: int, model_name: str) -> int:
    payload = f"{base_seed}:{model_name}".encode("utf-8")
    digest = sha256(payload).digest()
    derived = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int(derived % (2**31))


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def _generate_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def _prepare_run_directory(config: PolarityDialExperimentConfig) -> tuple[str, str]:
    run_id = config.run_id or _generate_run_id()
    run_dir = os.path.join(config.output_dir, "polarity_dial", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


def _persist_run_config(config: PolarityDialExperimentConfig, run_dir: str) -> None:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        sanitized_config = asdict(config)
        sanitized_config["wandb_api_key"] = None
        json.dump(sanitized_config, f, indent=2)


def _maybe_init_wandb(
    experiment_config: PolarityDialExperimentConfig,
    model_name: str,
    run_id: str,
) -> Optional[Any]:
    if (
        experiment_config.wandb_project is None
        or experiment_config.wandb_api_key is None
    ):
        return None
    if wandb is None:
        return None
    wandb.login(key=experiment_config.wandb_api_key, relogin=True)
    return wandb.init(
        project=experiment_config.wandb_project,
        name=f"{model_name}_{run_id}",
        config=asdict(experiment_config),
        group=run_id,
    )


def _log_wandb_artifacts(
    wandb_run: Any,
    training_metrics: Dict[str, Any],
    calibration: Dict[str, Any],
) -> None:
    if wandb_run is None:
        return
    wandb_run.log(
        {
            "training_loss_total": (
                training_metrics["loss_total"][-1]
                if training_metrics["loss_total"]
                else None
            ),
            "training_loss_reconstruction": (
                training_metrics["loss_reconstruction"][-1]
                if training_metrics["loss_reconstruction"]
                else None
            ),
            "training_loss_sparsity": (
                training_metrics["loss_sparsity"][-1]
                if training_metrics["loss_sparsity"]
                else None
            ),
            "calibration_summary": calibration["summary"],
        }
    )
    wandb_run.finish()


def _model_uses_list_sparsity(model_config: PolarityDialModelVariantConfig) -> bool:
    return isinstance(model_config.sparsity_coeff, list)


def _model_sparsity_length(model_config: PolarityDialModelVariantConfig) -> int:
    if isinstance(model_config.sparsity_coeff, list):
        return len(model_config.sparsity_coeff)
    return 1


def _resolve_sparsity_coeff_for_index(
    model_config: PolarityDialModelVariantConfig,
    coeff_index: int,
) -> Optional[float]:
    if isinstance(model_config.sparsity_coeff, list):
        if coeff_index >= len(model_config.sparsity_coeff):
            return None
        return float(model_config.sparsity_coeff[coeff_index])
    return float(model_config.sparsity_coeff)


def _resolve_subrun_count(models: List[PolarityDialModelVariantConfig]) -> int:
    if not models:
        raise ValueError("At least one model configuration must be provided.")
    return max(_model_sparsity_length(model_config) for model_config in models)


def _copy_seed_data_exports(seed_run_dir: str, subrun_dir: str) -> None:
    source_dir = os.path.join(seed_run_dir, "data")
    target_dir = os.path.join(subrun_dir, "data")
    if not os.path.isdir(source_dir):
        return
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def _steps_per_epoch(num_train: int, batch_size: int) -> int:
    return max(1, int((num_train + batch_size - 1) // batch_size))


def _training_metrics_for_instance(
    training_metrics: Dict[str, Any],
    instance_index: int,
) -> Dict[str, Any]:
    if "loss_total_by_instance" not in training_metrics:
        return training_metrics

    metrics_for_instance: Dict[str, Any] = {
        "epoch": list(training_metrics.get("epoch") or []),
        "loss_total": list(training_metrics["loss_total_by_instance"][instance_index]),
        "loss_reconstruction": list(
            training_metrics["loss_reconstruction_by_instance"][instance_index]
        ),
        "loss_sparsity": list(
            training_metrics["loss_sparsity_by_instance"][instance_index]
        ),
    }
    if "loss_aux_by_instance" in training_metrics:
        metrics_for_instance["loss_aux"] = list(
            training_metrics["loss_aux_by_instance"][instance_index]
        )
    if "terminated_early" in training_metrics:
        metrics_for_instance["terminated_early"] = True
    if "model_seed" in training_metrics:
        metrics_for_instance["model_seed"] = training_metrics["model_seed"]
    return metrics_for_instance


def _evaluate_trained_model(
    experiment_config: PolarityDialExperimentConfig,
    run_dir: str,
    run_id: str,
    model_config: PolarityDialModelVariantConfig,
    model: torch.nn.Module,
    training_metrics: Dict[str, Any],
    data: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    model_dir = os.path.join(run_dir, "models", model_config.name)
    wandb_run = _maybe_init_wandb(experiment_config, model_config.name, run_id)
    persist_model_artifacts(model, training_metrics, model_config, model_dir)

    alignment, axis_alignment, _ = align_decoder_and_features(
        model,
        data["feature_directions"],
        experiment_config.evaluation,
    )
    calibration, scatter_data = compute_alignment_and_calibration(
        model,
        data["x_test"],
        data["coefficients_test"],
        alignment,
        axis_alignment,
        experiment_config.evaluation,
    )
    persist_alignment_and_calibration(model_dir, alignment, axis_alignment, calibration)

    plot_calibration_scatter(scatter_data, model_config.name, model_dir)
    plot_gamma_histograms(calibration, model_config.name, model_dir)
    plot_gamma_strip(calibration, model_config.name, model_dir)
    plot_training_loss(training_metrics, model_config.name, model_dir)
    _log_wandb_artifacts(wandb_run, training_metrics, calibration)

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    d_hidden_effective = (
        int(getattr(model, "d_hidden"))
        if hasattr(model, "d_hidden")
        else int(model_config.d_hidden)
    )
    model_seed = training_metrics.get("model_seed")
    return {
        "model_dir": model_dir,
        "calibration": calibration,
        "metadata": {
            "model_seed": model_seed,
            "d_hidden": int(model_config.d_hidden),
            "d_hidden_effective": d_hidden_effective,
            "parameter_count": parameter_count,
            "coverage_summary": axis_alignment.get("coverage_summary"),
        },
    }


def _run_single_model(
    experiment_config: PolarityDialExperimentConfig,
    run_dir: str,
    run_id: str,
    model_config: PolarityDialModelVariantConfig,
    data: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, Any]:
    model_dir = os.path.join(run_dir, "models", model_config.name)
    wandb_run = _maybe_init_wandb(experiment_config, model_config.name, run_id)
    model_seed = _stable_model_seed(experiment_config.data.seed, model_config.name)
    set_random_seed(model_seed)
    dataloader_generator = torch.Generator().manual_seed(model_seed)
    training_output = run_training_for_variant(
        experiment_config,
        data,
        model_config,
        device,
        dataloader_generator=dataloader_generator,
    )
    training_output["training_metrics"]["model_seed"] = model_seed
    persist_model_artifacts(
        training_output["model"],
        training_output["training_metrics"],
        model_config,
        model_dir,
    )

    alignment, axis_alignment, _ = align_decoder_and_features(
        training_output["model"],
        data["feature_directions"],
        experiment_config.evaluation,
    )
    calibration, scatter_data = compute_alignment_and_calibration(
        training_output["model"],
        data["x_test"],
        data["coefficients_test"],
        alignment,
        axis_alignment,
        experiment_config.evaluation,
    )
    persist_alignment_and_calibration(model_dir, alignment, axis_alignment, calibration)

    plot_calibration_scatter(scatter_data, model_config.name, model_dir)
    plot_gamma_histograms(calibration, model_config.name, model_dir)
    plot_gamma_strip(calibration, model_config.name, model_dir)
    plot_training_loss(
        training_output["training_metrics"], model_config.name, model_dir
    )
    _log_wandb_artifacts(wandb_run, training_output["training_metrics"], calibration)

    parameter_count = int(
        sum(parameter.numel() for parameter in training_output["model"].parameters())
    )
    d_hidden_effective = (
        int(getattr(training_output["model"], "d_hidden"))
        if hasattr(training_output["model"], "d_hidden")
        else int(model_config.d_hidden)
    )
    return {
        "model_dir": model_dir,
        "calibration": calibration,
        "metadata": {
            "model_seed": model_seed,
            "d_hidden": int(model_config.d_hidden),
            "d_hidden_effective": d_hidden_effective,
            "parameter_count": parameter_count,
            "coverage_summary": axis_alignment.get("coverage_summary"),
        },
    }


def _run_multi_sparsity_subruns(
    experiment_config: PolarityDialExperimentConfig,
    seed_run_dir: str,
    seed_run_id: str,
    data: Dict[str, torch.Tensor],
    device: str,
) -> None:
    subrun_count = _resolve_subrun_count(experiment_config.models)
    trained_by_model_name: Dict[str, Dict[str, Any]] = {}
    for model_config in experiment_config.models:
        model_seed = _stable_model_seed(
            int(experiment_config.data.seed), model_config.name
        )
        set_random_seed(model_seed)
        dataloader_generator = torch.Generator().manual_seed(model_seed)
        training_output = run_training_for_variant(
            experiment_config,
            data,
            model_config,
            device,
            dataloader_generator=dataloader_generator,
        )
        training_output["training_metrics"]["model_seed"] = model_seed
        trained_by_model_name[model_config.name] = training_output

    num_train = int(data["x_train"].shape[0])
    input_dim = int(data["x_train"].shape[1])
    for model_config in experiment_config.models:
        if not _model_uses_list_sparsity(model_config):
            continue
        training_metrics = trained_by_model_name[model_config.name]["training_metrics"]
        plot_multi_sparsity_loss_summaries(
            training_metrics=training_metrics,
            model_name=model_config.name,
            seed_run_dir=seed_run_dir,
            sparsity_coeff_list=[float(value) for value in model_config.sparsity_coeff],
        )

    for coeff_index in range(subrun_count):
        subrun_dir = os.path.join(seed_run_dir, str(coeff_index))
        os.makedirs(subrun_dir, exist_ok=True)
        persist_environment_metadata(subrun_dir)
        _copy_seed_data_exports(seed_run_dir, subrun_dir)

        subrun_models: List[PolarityDialModelVariantConfig] = []
        base_model_configs_by_name = {
            model_config.name: model_config for model_config in experiment_config.models
        }
        for model_config in experiment_config.models:
            sparsity_coeff = _resolve_sparsity_coeff_for_index(
                model_config, coeff_index
            )
            if sparsity_coeff is None:
                continue
            subrun_models.append(replace(model_config, sparsity_coeff=sparsity_coeff))

        subrun_config = replace(
            experiment_config,
            models=subrun_models,
            run_id=f"{seed_run_id}_idx{coeff_index}",
        )
        _persist_run_config(subrun_config, subrun_dir)

        calibration_by_model: Dict[str, Dict[str, Any]] = {}
        metadata_by_model: Dict[str, Dict[str, Any]] = {}
        for model_config in subrun_models:
            base_model_config = base_model_configs_by_name[model_config.name]
            trained = trained_by_model_name[model_config.name]
            training_metrics = trained["training_metrics"]
            if _model_uses_list_sparsity(base_model_config):
                steps_per_epoch = _steps_per_epoch(
                    num_train, base_model_config.batch_size
                )
                model = slice_trained_model_instance(
                    trained["model"],
                    base_model_config,
                    coeff_index,
                    input_dim=input_dim,
                    steps_per_epoch=steps_per_epoch,
                    device=device,
                )
                training_metrics = _training_metrics_for_instance(
                    training_metrics, instance_index=coeff_index
                )
            else:
                model = trained["model"]

            model_result = _evaluate_trained_model(
                subrun_config,
                subrun_dir,
                subrun_config.run_id or f"{seed_run_id}_idx{coeff_index}",
                model_config,
                model,
                training_metrics,
                data,
            )
            calibration_by_model[model_config.name] = model_result["calibration"]
            metadata_by_model[model_config.name] = model_result["metadata"]

        comparison_summary = _build_comparison_summary(
            calibration_by_model, metadata_by_model
        )
        comparison_path = os.path.join(subrun_dir, "comparison_summary.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison_summary, f, indent=2)
        plot_gamma_comparison(
            {
                name: calibration["summary"]
                for name, calibration in calibration_by_model.items()
            },
            os.path.join(subrun_dir, "gamma_diff_bar_comparison.png"),
        )


def _build_comparison_summary(
    calibration_by_model: Dict[str, Dict[str, Any]],
    model_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    summaries = {
        model_name: calibration["summary"]
        for model_name, calibration in calibration_by_model.items()
    }
    baseline = next(iter(calibration_by_model.keys()))
    baseline_summary = calibration_by_model[baseline]["summary"]
    relative_to_baseline = {}
    for model_name, calibration in calibration_by_model.items():
        summary = calibration["summary"]
        relative_to_baseline[model_name] = {
            "delta_mean_rel_error_plus": (
                None
                if baseline_summary["mean_rel_error_plus"] is None
                or summary["mean_rel_error_plus"] is None
                else summary["mean_rel_error_plus"]
                - baseline_summary["mean_rel_error_plus"]
            ),
            "delta_mean_rel_error_minus": (
                None
                if baseline_summary["mean_rel_error_minus"] is None
                or summary["mean_rel_error_minus"] is None
                else summary["mean_rel_error_minus"]
                - baseline_summary["mean_rel_error_minus"]
            ),
        }
    return {
        "models": list(calibration_by_model.keys()),
        "summaries": summaries,
        "baseline": baseline,
        "relative_to_baseline": relative_to_baseline,
        "model_metadata": model_metadata or {},
    }


def run_polarity_dial(
    experiment_config: PolarityDialExperimentConfig,
) -> str | List[str]:
    seed_value = experiment_config.data.seed
    if isinstance(seed_value, list):
        seeds = [int(seed) for seed in seed_value]
        base_run_id = experiment_config.run_id or _generate_run_id()
        run_dirs: List[str] = []
        for seed in seeds:
            per_seed_config = replace(
                experiment_config,
                data=replace(experiment_config.data, seed=seed),
                run_id=f"{base_run_id}_seed{seed}",
            )
            run_dirs.append(run_polarity_dial(per_seed_config))
        return run_dirs

    device = _resolve_device(experiment_config.device)
    config = replace(experiment_config, device=device)
    run_dir, run_id = _prepare_run_directory(config)
    persist_environment_metadata(run_dir)
    _persist_run_config(config, run_dir)

    set_random_seed(int(config.data.seed))
    data = generate_polarity_dial_dataset(config.data, device)
    export_data_summary(run_dir, config.data, data, config.sample_preview_count)
    set_random_seed(int(config.data.seed))

    uses_list_sparsity = any(
        _model_uses_list_sparsity(model_config) for model_config in config.models
    )
    if uses_list_sparsity:
        _run_multi_sparsity_subruns(config, run_dir, run_id, data, device)
    else:
        calibration_by_model: Dict[str, Dict[str, Any]] = {}
        metadata_by_model: Dict[str, Dict[str, Any]] = {}
        for model_config in config.models:
            model_result = _run_single_model(
                config, run_dir, run_id, model_config, data, device
            )
            calibration_by_model[model_config.name] = model_result["calibration"]
            metadata_by_model[model_config.name] = model_result["metadata"]

        comparison_summary = _build_comparison_summary(
            calibration_by_model, metadata_by_model
        )
        comparison_path = os.path.join(run_dir, "comparison_summary.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison_summary, f, indent=2)
        plot_gamma_comparison(
            {
                name: calibration["summary"]
                for name, calibration in calibration_by_model.items()
            },
            os.path.join(run_dir, "gamma_diff_bar_comparison.png"),
        )

    if config.sweeps.sweep_mode in {"sparsity", "both"}:
        sparsity_summary = run_sparsity_sweep(config, run_dir, data, device)
        tradeoff_path = os.path.join(run_dir, "mse_vs_l0_tradeoff.json")
        with open(tradeoff_path, "w") as f:
            json.dump(sparsity_summary, f, indent=2)
        plot_mse_vs_l0_tradeoff(
            sparsity_summary, os.path.join(run_dir, "mse_vs_l0_tradeoff.png")
        )

    if config.sweeps.sweep_mode in {"asymmetry", "both"}:
        asymmetry_summary = run_asymmetry_sweep(config, run_dir, device)
        sweep_path = os.path.join(run_dir, "asymmetry_sweep_summary.json")
        with open(sweep_path, "w") as f:
            json.dump(asymmetry_summary, f, indent=2)
        plot_asymmetry_sweep_calibration(
            asymmetry_summary, os.path.join(run_dir, "asymmetry_sweep_calibration.png")
        )
    return run_dir


def run_polarity_dial_experiment(config_path: str) -> str | List[str]:
    raw_config = load_experiment_config(config_path)
    return run_polarity_dial(raw_config)
