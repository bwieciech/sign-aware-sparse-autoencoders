import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Tuple

import torch

from sgnsae.experiments.polarity_dial.environment import persist_environment_metadata
from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DExperimentConfig,
    load_experiment_config,
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


def run_toy_model_geometry_validation_experiment(config_path: str) -> str:
    raw_config = load_experiment_config(config_path)
    device = _resolve_device(raw_config.device)
    config = replace(raw_config, device=device)
    run_dir, run_id = _prepare_run_directory(config)
    persist_environment_metadata(run_dir)
    _persist_run_config(replace(config, run_id=run_id), run_dir)
    return run_dir
