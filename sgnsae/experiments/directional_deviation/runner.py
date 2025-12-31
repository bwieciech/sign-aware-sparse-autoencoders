import json
import os
from dataclasses import asdict, replace
from datetime import datetime

import torch

from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationExperimentConfig,
    load_experiment_config,
)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def _prepare_run_directory(config: DirectionalDeviationExperimentConfig) -> tuple[str, str]:
    run_id = config.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = os.path.join(config.output_dir, "directional_deviation", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


def _persist_run_config(config: DirectionalDeviationExperimentConfig, run_dir: str) -> None:
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        sanitized_config = asdict(config)
        sanitized_config["wandb_api_key"] = None
        json.dump(sanitized_config, f, indent=2)


def run_directional_deviation_experiment(config_path: str) -> str:
    raw_config = load_experiment_config(config_path)
    device = _resolve_device(raw_config.device)
    config = replace(raw_config, device=device)
    run_dir, _ = _prepare_run_directory(config)
    _persist_run_config(config, run_dir)
    return run_dir

