import json
import os
import warnings
from hashlib import sha256
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch

from sgnsae.experiments.polarity_dial.environment import persist_environment_metadata
from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DExperimentConfig,
    load_experiment_config,
)
from sgnsae.experiments.toy_model_geometry_validation.data import (
    export_toy2d_dataset,
    generate_toy2d_dataset,
    set_random_seed,
)
from sgnsae.experiments.toy_model_geometry_validation.evaluation import (
    extract_signed_pair_coefficients,
    evaluate_toy2d_model,
    persist_evaluation,
)
from sgnsae.experiments.toy_model_geometry_validation.plotting import (
    plot_bijump_histogram,
    plot_bijump_histogram_logy,
    plot_latent_vs_coeff_scatter,
    plot_pair_consolidation_sweep,
    plot_unit_circle,
)
from sgnsae.experiments.toy_model_geometry_validation.training import (
    persist_model_artifacts,
    run_training_for_variant,
)


_TORCH_SEED_MODULUS = 2**63


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def _resolve_run_parent_dir(output_dir: str) -> str:
    normalized_output = os.path.normpath(output_dir)
    if os.path.basename(normalized_output) == "toy_model_geometry_validation":
        return normalized_output
    return os.path.join(normalized_output, "toy_model_geometry_validation")


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _prepare_run_directory(config: Toy2DExperimentConfig) -> Tuple[str, str]:
    run_id = config.run_id or _generate_run_id()
    run_parent = _resolve_run_parent_dir(config.output_dir)
    run_dir = os.path.join(run_parent, run_id)

    if os.path.exists(run_dir):
        if config.run_id is not None:
            raise ValueError(
                "Requested run directory already exists. Choose a different run_id or "
                f"remove the existing directory: {run_dir}"
            )
        suffix = 1
        while True:
            candidate_run_id = f"{run_id}_{suffix}"
            candidate_run_dir = os.path.join(run_parent, candidate_run_id)
            if not os.path.exists(candidate_run_dir):
                run_id = candidate_run_id
                run_dir = candidate_run_dir
                break
            suffix += 1

    os.makedirs(run_dir, exist_ok=False)
    return run_dir, run_id


def _persist_run_config(config: Toy2DExperimentConfig, run_dir: str) -> None:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)


def _normalize_torch_seed(seed: int) -> int:
    return int(seed % _TORCH_SEED_MODULUS)


def _stable_model_seed(base_seed: int, model_name: str) -> int:
    payload = f"{base_seed}:{model_name}".encode("utf-8")
    digest = sha256(payload).digest()
    derived = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return _normalize_torch_seed(int(derived))


def _format_pair_correlation(value: float) -> str:
    normalized = 0.0 if abs(value) < 1e-12 else float(value)
    return str(normalized)


def _warn_if_width_name_mismatch(model_name: str, d_hidden: int, num_pairs: int) -> None:
    """
    Protocol C uses naming conventions like *_k and *_2k to communicate intended width.
    This check prevents accidental mismatches (e.g. num_pairs=8 but d_hidden=64).
    """
    expected = None
    if model_name.endswith("_k"):
        expected = num_pairs
    elif model_name.endswith("_2k"):
        expected = 2 * num_pairs
    if expected is not None and d_hidden != expected:
        warnings.warn(
            f"[toy_model_geometry_validation] Model '{model_name}' implies d_hidden={expected} "
            f"(num_pairs={num_pairs}), but config has d_hidden={d_hidden}. "
            "This can invalidate 'k vs 2k' Protocol C comparisons.",
            stacklevel=2,
        )


def _run_data_generation(
    experiment_config: Toy2DExperimentConfig,
    run_dir: str,
) -> Dict[str, torch.Tensor]:
    data = generate_toy2d_dataset(experiment_config.data, experiment_config.device)
    export_toy2d_dataset(
        run_dir=run_dir,
        config=experiment_config.data,
        data=data,
        sample_preview_count=experiment_config.sample_preview_count,
    )
    return data


def _extract_final_metric(values: Any) -> Optional[float]:
    if not isinstance(values, list) or not values:
        return None
    last_value = values[-1]
    if last_value is None:
        return None
    return float(last_value)


def _run_single_model(
    experiment_config: Toy2DExperimentConfig,
    run_dir: str,
    data: Dict[str, torch.Tensor],
    model_config: Any,
) -> Dict[str, Any]:
    model_dir = os.path.join(run_dir, "models", model_config.name)
    os.makedirs(model_dir, exist_ok=True)

    model_seed = _stable_model_seed(experiment_config.data.seed, model_config.name)
    set_random_seed(model_seed)
    dataloader_generator = torch.Generator(device="cpu").manual_seed(
        _normalize_torch_seed(model_seed)
    )

    training_output = run_training_for_variant(
        experiment_config=experiment_config,
        data=data,
        model_config=model_config,
        device=experiment_config.device,
        dataloader_generator=dataloader_generator,
    )
    training_metrics = training_output["training_metrics"]
    training_metrics["model_seed"] = model_seed
    persist_model_artifacts(
        model=training_output["model"],
        training_metrics=training_metrics,
        model_config=model_config,
        model_dir=model_dir,
    )

    parameter_count = int(
        sum(parameter.numel() for parameter in training_output["model"].parameters())
    )
    training_summary = {
        "model_seed": model_seed,
        "terminated_early": bool(training_metrics.get("terminated_early", False)),
        "parameter_count": parameter_count,
        "final_loss_total": _extract_final_metric(training_metrics.get("loss_total")),
        "final_loss_reconstruction": _extract_final_metric(
            training_metrics.get("loss_reconstruction")
        ),
        "final_loss_sparsity": _extract_final_metric(
            training_metrics.get("loss_sparsity")
        ),
        "final_loss_aux": _extract_final_metric(training_metrics.get("loss_aux")),
    }

    evaluation = evaluate_toy2d_model(
        model=training_output["model"],
        data=data,
        evaluation_config=experiment_config.evaluation,
        model_name=model_config.name,
        sign_aware=bool(model_config.sign_aware),
        device=experiment_config.device,
    )
    persist_evaluation(evaluation=evaluation, model_dir=model_dir)
    plot_unit_circle(
        model=training_output["model"],
        model_name=model_config.name,
        feature_directions=data["feature_directions"],
        model_dir=model_dir,
        x_test=data.get("x_test"),
    )
    if bool(model_config.sign_aware) and evaluation.get("histogram_latent") is not None:
        histogram_info = evaluation["histogram_latent"]
        pair_index = int(histogram_info.get("pair_index", 0))
        orientation_sign = int(histogram_info.get("orientation_sign", 1))
        axis_abs_cos = (
            None
            if histogram_info.get("axis_abs_cos") is None
            else float(histogram_info["axis_abs_cos"])
        )
        plot_bijump_histogram(
            model=training_output["model"],
            model_name=model_config.name,
            x_test=data["x_test"],
            latent_index=int(histogram_info["latent_index"]),
            hist_num_bins=int(experiment_config.evaluation.hist_num_bins),
            model_dir=model_dir,
            device=experiment_config.device,
            axis_abs_cos=axis_abs_cos,
        )
        training_output["model"].eval()
        with torch.no_grad():
            output = training_output["model"](
                data["x_test"].to(device=experiment_config.device)
            )
        latents_tensor = output.get("latents")
        if latents_tensor is not None and latents_tensor.ndim == 3:
            latents = latents_tensor[0].detach().cpu().to(dtype=torch.float32)
            latent_index = int(histogram_info["latent_index"])
            oriented_latent = (
                latents[:, latent_index] * float(orientation_sign)
            ).numpy()

            signed_coeffs = extract_signed_pair_coefficients(
                data["coeffs_test"].detach().cpu().to(dtype=torch.float32)
            )
            signed_coeff = signed_coeffs[:, pair_index].numpy()
            eligible_mask = (
                data["pair_active_mask_test"][:, pair_index]
                .detach()
                .cpu()
                .to(dtype=torch.bool)
                .numpy()
            )
            corr_mask = eligible_mask & (signed_coeff != 0.0)
            if bool(corr_mask.any()):
                plots_dir = os.path.join(model_dir, "plots")
                title = f"Bi-Jump Histogram - {model_config.name} latent {latent_index}"
                if axis_abs_cos is not None:
                    title = f"{title} (axis abs cos {axis_abs_cos:.3f})"
                plot_bijump_histogram_logy(
                    values=oriented_latent[corr_mask],
                    hist_num_bins=int(experiment_config.evaluation.hist_num_bins),
                    title=title,
                    output_path=os.path.join(
                        plots_dir,
                        f"bi_jump_hist_{model_config.name}_latent{latent_index}_logy.png",
                    ),
                )
                plot_latent_vs_coeff_scatter(
                    latent=oriented_latent[corr_mask],
                    signed_coeff=signed_coeff[corr_mask],
                    title=(
                        f"Latent vs coeff - {model_config.name} "
                        f"pair {pair_index} latent {latent_index}"
                    ),
                    output_path=os.path.join(
                        plots_dir,
                        f"latent_vs_coeff_scatter_{model_config.name}"
                        f"_pair{pair_index}_latent{latent_index}.png",
                    ),
                )

    training_summary.update(
        {
            "pair_consolidation_rate": evaluation.get("pair_consolidation_rate"),
            "pair_consolidation_rate_over_evaluable_pairs": evaluation.get(
                "pair_consolidation_rate_over_evaluable_pairs"
            ),
            "rate_over_evaluable_pairs": evaluation.get("rate_over_evaluable_pairs"),
            "evaluable_pair_fraction": evaluation.get("evaluable_pair_fraction"),
            "num_pairs_consolidated": evaluation.get("num_pairs_consolidated"),
            "num_pairs_evaluable": evaluation.get("num_pairs_evaluable"),
        }
    )
    return {
        "model_dir": model_dir,
        "training_summary": training_summary,
        "evaluation_summary": {
            "pair_consolidation_rate": evaluation.get("pair_consolidation_rate"),
            "pair_consolidation_rate_over_evaluable_pairs": evaluation.get(
                "pair_consolidation_rate_over_evaluable_pairs"
            ),
            "rate_over_evaluable_pairs": evaluation.get("rate_over_evaluable_pairs"),
            "evaluable_pair_fraction": evaluation.get("evaluable_pair_fraction"),
            "num_pairs_consolidated": evaluation.get("num_pairs_consolidated"),
            "num_pairs_evaluable": evaluation.get("num_pairs_evaluable"),
        },
    }


def _run_leaf_experiment(
    experiment_config: Toy2DExperimentConfig,
    run_dir: str,
) -> Dict[str, Any]:
    data = _run_data_generation(experiment_config, run_dir)

    per_model: Dict[str, Any] = {}
    num_pairs = experiment_config.data.num_pairs
    for model_config in experiment_config.models:
        _warn_if_width_name_mismatch(
            model_config.name,
            int(model_config.d_hidden),
            int(num_pairs),
        )
        per_model[model_config.name] = _run_single_model(
            experiment_config=experiment_config,
            run_dir=run_dir,
            data=data,
            model_config=model_config,
        )

    comparison_summary = {
        "models": list(per_model.keys()),
        "per_model": {name: per_model[name]["training_summary"] for name in per_model},
        "data": {
            "seed": int(experiment_config.data.seed),
            "num_pairs": int(experiment_config.data.num_pairs),
            "pair_feature_prob": float(experiment_config.data.pair_feature_prob),
            "pair_axis_active_prob": float(
                experiment_config.data.pair_axis_active_prob
            ),
            "pair_correlation": (
                None
                if experiment_config.data.pair_correlation is None
                else float(experiment_config.data.pair_correlation)
            ),
        },
    }
    comparison_path = os.path.join(run_dir, "comparison_summary.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    return comparison_summary


def _run_sweep_experiment(
    experiment_config: Toy2DExperimentConfig,
    run_dir: str,
) -> List[str]:
    grid = experiment_config.data.pair_correlation_grid
    if grid is None:
        raise ValueError("pair_correlation_grid must be set to run a sweep.")

    leaf_dirs: List[str] = []
    per_correlation: Dict[str, Any] = {}
    for correlation_value in grid:
        leaf_dir = os.path.join(
            run_dir, f"corr_{_format_pair_correlation(correlation_value)}"
        )
        os.makedirs(leaf_dir, exist_ok=False)
        leaf_config = replace(
            experiment_config,
            data=replace(
                experiment_config.data,
                pair_correlation=float(correlation_value),
                pair_correlation_grid=None,
            ),
        )
        _persist_run_config(leaf_config, leaf_dir)
        per_correlation[_format_pair_correlation(correlation_value)] = (
            _run_leaf_experiment(leaf_config, leaf_dir)
        )
        leaf_dirs.append(leaf_dir)

    sweep_summary = {
        "sweep": {
            "type": "pair_correlation_grid",
            "pair_correlation_grid": [float(value) for value in grid],
        },
        "per_correlation": per_correlation,
    }

    sweep_points: List[Dict[str, Any]] = []
    model_names = [model.name for model in experiment_config.models]
    series_by_model: Dict[str, List[Optional[float]]] = {
        model_name: [] for model_name in model_names
    }
    for correlation_value in grid:
        key = _format_pair_correlation(correlation_value)
        leaf_summary = per_correlation[key]
        per_model_metrics: Dict[str, Any] = {}
        for model_name in model_names:
            model_summary = leaf_summary["per_model"][model_name]
            per_model_metrics[model_name] = {
                "pair_consolidation_rate": model_summary.get("pair_consolidation_rate"),
                "pair_consolidation_rate_over_evaluable_pairs": model_summary.get(
                    "pair_consolidation_rate_over_evaluable_pairs"
                ),
                "rate_over_evaluable_pairs": model_summary.get("rate_over_evaluable_pairs"),
                "evaluable_pair_fraction": model_summary.get("evaluable_pair_fraction"),
                "num_pairs_evaluable": model_summary.get("num_pairs_evaluable"),
                "num_pairs_consolidated": model_summary.get("num_pairs_consolidated"),
            }
            series_by_model[model_name].append(
                model_summary.get("pair_consolidation_rate_over_evaluable_pairs")
            )

        sweep_points.append(
            {
                "pair_correlation": float(correlation_value),
                "per_model": per_model_metrics,
            }
        )

    with open(os.path.join(run_dir, "pair_consolidation_sweep.json"), "w") as f:
        json.dump(
            {
                "sweep": sweep_summary["sweep"],
                "points": sweep_points,
            },
            f,
            indent=2,
        )
    plot_pair_consolidation_sweep(
        pair_correlation_grid=[float(value) for value in grid],
        series_by_model=series_by_model,
        output_path=os.path.join(run_dir, "plots", "pair_consolidation_sweep.png"),
    )

    with open(os.path.join(run_dir, "comparison_summary.json"), "w") as f:
        json.dump(sweep_summary, f, indent=2)
    return leaf_dirs


def run_toy_model_geometry_validation(experiment_config: Toy2DExperimentConfig) -> str:
    device = _resolve_device(experiment_config.device)
    config = replace(experiment_config, device=device)
    run_dir, run_id = _prepare_run_directory(config)
    config_with_run_id = replace(config, run_id=run_id)
    persist_environment_metadata(run_dir)
    _persist_run_config(config_with_run_id, run_dir)
    if config_with_run_id.data.pair_correlation_grid is not None:
        _run_sweep_experiment(config_with_run_id, run_dir)
    else:
        _run_leaf_experiment(config_with_run_id, run_dir)
    return run_dir


def run_toy_model_geometry_validation_experiment(config_path: str) -> str:
    raw_config = load_experiment_config(config_path)
    return run_toy_model_geometry_validation(raw_config)
