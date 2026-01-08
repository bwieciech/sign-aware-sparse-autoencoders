import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import torch

from sgnsae.experiments.polarity_dial.environment import persist_environment_metadata
from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DExperimentConfig,
    load_experiment_config,
)
from sgnsae.experiments.toy_model_geometry_validation.data import (
    export_toy2d_dataset,
    generate_toy2d_dataset,
)


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


def _format_pair_correlation(value: float) -> str:
    normalized = 0.0 if abs(value) < 1e-12 else float(value)
    return str(normalized)


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


def _run_sweep_data_generation(
    experiment_config: Toy2DExperimentConfig,
    run_dir: str,
) -> List[str]:
    grid = experiment_config.data.pair_correlation_grid
    if grid is None:
        raise ValueError("pair_correlation_grid must be set to run a sweep.")

    leaf_dirs: List[str] = []
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
        _run_data_generation(leaf_config, leaf_dir)
        leaf_dirs.append(leaf_dir)
    return leaf_dirs


def run_toy_model_geometry_validation_experiment(config_path: str) -> str:
    raw_config = load_experiment_config(config_path)
    device = _resolve_device(raw_config.device)
    config = replace(raw_config, device=device)
    run_dir, run_id = _prepare_run_directory(config)
    config_with_run_id = replace(config, run_id=run_id)
    persist_environment_metadata(run_dir)
    _persist_run_config(config_with_run_id, run_dir)
    if config_with_run_id.data.pair_correlation_grid is not None:
        _run_sweep_data_generation(config_with_run_id, run_dir)
    else:
        _run_data_generation(config_with_run_id, run_dir)
    return run_dir
