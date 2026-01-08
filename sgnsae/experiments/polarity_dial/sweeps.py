import hashlib
import json
import os
from dataclasses import asdict, replace
from typing import Any, Dict, List, Tuple

from torch import Tensor

from sgnsae.experiments.polarity_dial.config import (
    PolarityDialDataConfig,
    PolarityDialExperimentConfig,
    PolarityDialModelVariantConfig,
)
from sgnsae.experiments.polarity_dial.data import (
    generate_polarity_dial_dataset,
    set_random_seed,
)
from sgnsae.experiments.polarity_dial.evaluation import (
    align_decoder_and_features,
    compute_alignment_and_calibration,
)
from sgnsae.experiments.polarity_dial.training import run_training_for_variant


def _format_float_for_path(value: float) -> str:
    return f"{value:.6g}"


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _resolve_tail_grid(
    experiment_config: PolarityDialExperimentConfig,
) -> Tuple[str, List[float]]:
    sweeps = experiment_config.sweeps
    if sweeps.lognormal_sigma_grid:
        return "lognormal_sigma", list(sweeps.lognormal_sigma_grid)
    if sweeps.exponential_lambda_grid:
        return "exponential_lambda", list(sweeps.exponential_lambda_grid)
    raise ValueError("Asymmetry sweep requires a tail-heaviness grid.")


def _seed_for_model(base_seed: int, model_name: str) -> int:
    digest = hashlib.sha256(model_name.encode("utf-8")).digest()
    name_hash = int.from_bytes(digest[:4], "big")
    return int((base_seed + name_hash) % 2_147_483_647)


def run_sparsity_sweep(
    experiment_config: PolarityDialExperimentConfig,
    run_dir: str,
    data: Dict[str, Tensor],
    device: str,
) -> Dict[str, Any]:
    if experiment_config.sweeps.sparsity_coeff_grid is None:
        raise ValueError("sparsity_coeff_grid must be provided for sparsity sweep.")

    sweep_dir = os.path.join(run_dir, "sweeps", "sparsity")
    os.makedirs(sweep_dir, exist_ok=True)
    results_by_model: Dict[str, List[Dict[str, Any]]] = {
        model_config.name: [] for model_config in experiment_config.models
    }

    for sparsity_coeff in experiment_config.sweeps.sparsity_coeff_grid:
        point_dir = os.path.join(
            sweep_dir, f"sparsity_coeff={_format_float_for_path(sparsity_coeff)}"
        )
        os.makedirs(point_dir, exist_ok=True)
        point_results: Dict[str, Any] = {
            "sparsity_coeff": float(sparsity_coeff),
            "models": {},
        }

        for model_config in experiment_config.models:
            set_random_seed(
                _seed_for_model(experiment_config.data.seed, model_config.name)
            )
            model_config_for_point: PolarityDialModelVariantConfig = replace(
                model_config, sparsity_coeff=float(sparsity_coeff)
            )
            training_output = run_training_for_variant(
                experiment_config, data, model_config_for_point, device
            )
            alignment, axis_alignment, _ = align_decoder_and_features(
                training_output["model"],
                data["feature_directions"],
                experiment_config.evaluation,
            )
            calibration, _ = compute_alignment_and_calibration(
                training_output["model"],
                data["x_test"],
                data["coefficients_test"],
                alignment,
                axis_alignment,
                experiment_config.evaluation,
            )
            summary = calibration["summary"]
            model_result = {
                "sparsity_coeff": float(sparsity_coeff),
                "test_reconstruction_mse": summary["test_reconstruction_mse"],
                "test_l0": summary["test_l0"],
                "mean_rel_error_plus": summary["mean_rel_error_plus"],
                "mean_rel_error_minus": summary["mean_rel_error_minus"],
            }
            point_results["models"][model_config.name] = model_result
            results_by_model[model_config.name].append(model_result)

        _write_json(
            os.path.join(point_dir, "point_config.json"),
            {"sparsity_coeff": float(sparsity_coeff)},
        )
        _write_json(os.path.join(point_dir, "point_summary.json"), point_results)

    return {
        "sweep_type": "sparsity",
        "sparsity_coeff_grid": list(experiment_config.sweeps.sparsity_coeff_grid),
        "models": results_by_model,
    }


def run_asymmetry_sweep(
    experiment_config: PolarityDialExperimentConfig,
    run_dir: str,
    device: str,
) -> Dict[str, Any]:
    sweeps = experiment_config.sweeps
    if sweeps.p_plus_grid is None:
        raise ValueError("p_plus_grid must be provided for asymmetry sweep.")

    tail_key, tail_grid = _resolve_tail_grid(experiment_config)
    sweep_dir = os.path.join(run_dir, "sweeps", "asymmetry")
    os.makedirs(sweep_dir, exist_ok=True)
    points: List[Dict[str, Any]] = []

    for tail_value in tail_grid:
        for p_plus in sweeps.p_plus_grid:
            point_name = (
                f"p_plus={_format_float_for_path(float(p_plus))}__"
                f"{tail_key}={_format_float_for_path(float(tail_value))}"
            )
            point_dir = os.path.join(sweep_dir, point_name)
            os.makedirs(point_dir, exist_ok=True)

            data_config_kwargs: Dict[str, Any] = {"positive_sign_prob": float(p_plus)}
            if tail_key == "lognormal_sigma":
                data_config_kwargs["lognormal_sigma"] = float(tail_value)
            if tail_key == "exponential_lambda":
                data_config_kwargs["exponential_lambda"] = float(tail_value)
            data_config: PolarityDialDataConfig = replace(
                experiment_config.data, **data_config_kwargs
            )
            data = generate_polarity_dial_dataset(data_config, device)

            point_summary: Dict[str, Any] = {
                "positive_sign_prob": float(p_plus),
                tail_key: float(tail_value),
                "models": {},
            }
            for model_config in experiment_config.models:
                set_random_seed(
                    _seed_for_model(experiment_config.data.seed, model_config.name)
                )
                training_output = run_training_for_variant(
                    experiment_config, data, model_config, device
                )
                alignment, axis_alignment, _ = align_decoder_and_features(
                    training_output["model"],
                    data["feature_directions"],
                    experiment_config.evaluation,
                )
                calibration, _ = compute_alignment_and_calibration(
                    training_output["model"],
                    data["x_test"],
                    data["coefficients_test"],
                    alignment,
                    axis_alignment,
                    experiment_config.evaluation,
                )
                summary = calibration["summary"]
                point_summary["models"][model_config.name] = {
                    "test_reconstruction_mse": summary["test_reconstruction_mse"],
                    "test_l0": summary["test_l0"],
                    "mean_rel_error_plus": summary["mean_rel_error_plus"],
                    "mean_rel_error_minus": summary["mean_rel_error_minus"],
                }
            _write_json(
                os.path.join(point_dir, "generator_config.json"), asdict(data_config)
            )
            _write_json(os.path.join(point_dir, "point_summary.json"), point_summary)
            points.append(point_summary)

    return {
        "sweep_type": "asymmetry",
        "p_plus_grid": list(sweeps.p_plus_grid),
        f"{tail_key}_grid": tail_grid,
        "points": points,
        "models": [model_config.name for model_config in experiment_config.models],
    }
