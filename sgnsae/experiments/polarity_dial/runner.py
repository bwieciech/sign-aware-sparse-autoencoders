import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, Optional

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


def _prepare_run_directory(config: PolarityDialExperimentConfig) -> tuple[str, str]:
    run_id = config.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
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
            "d_hidden": int(model_config.d_hidden),
            "d_hidden_effective": d_hidden_effective,
            "parameter_count": parameter_count,
            "coverage_summary": axis_alignment.get("coverage_summary"),
        },
    }


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


def run_polarity_dial_experiment(config_path: str) -> str:
    raw_config = load_experiment_config(config_path)
    device = _resolve_device(raw_config.device)
    config = replace(raw_config, device=device)
    run_dir, run_id = _prepare_run_directory(config)
    _persist_run_config(config, run_dir)

    set_random_seed(config.data.seed)
    data = generate_polarity_dial_dataset(config.data, device)
    export_data_summary(run_dir, config.data, data, config.sample_preview_count)
    set_random_seed(config.data.seed)

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
