import json
import os
import random
from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from sgnsae.experiments.polarity_dial.config import PolarityDialDataConfig


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_lognormal(
    shape: Tuple[int, ...],
    mu: float,
    sigma: float,
    device: str,
    generator: torch.Generator,
) -> Tensor:
    normal = torch.randn(shape, device=device, generator=generator)
    return (normal * sigma + mu).exp()


def _sample_exponential(
    shape: Tuple[int, ...],
    rate: float,
    device: str,
    generator: torch.Generator,
) -> Tensor:
    uniforms = torch.rand(shape, device=device, generator=generator)
    return -uniforms.clamp(min=1e-12).log() / rate


def _sample_coefficients(
    num_samples: int,
    config: PolarityDialDataConfig,
    device: str,
    generator: torch.Generator,
) -> Tensor:
    active_mask = torch.bernoulli(
        torch.full(
            (num_samples, config.num_features),
            config.activation_prob,
            device=device,
            dtype=torch.float32,
        ),
        generator=generator,
    ).bool()
    positive_mask = torch.bernoulli(
        torch.full(
            (num_samples, config.num_features),
            config.positive_sign_prob,
            device=device,
            dtype=torch.float32,
        ),
        generator=generator,
    ).bool()

    coefficients = torch.zeros(
        (num_samples, config.num_features), device=device, dtype=torch.float32
    )
    positive_values = _sample_lognormal(
        (num_samples, config.num_features),
        config.lognormal_mu,
        config.lognormal_sigma,
        device,
        generator,
    )
    negative_values = _sample_exponential(
        (num_samples, config.num_features),
        config.exponential_lambda,
        device,
        generator,
    )

    positive_indices = active_mask & positive_mask
    negative_indices = active_mask & (~positive_mask)
    if positive_indices.any():
        coefficients = coefficients + positive_values * positive_indices
    if negative_indices.any():
        coefficients = coefficients - negative_values * negative_indices

    return coefficients


def _generate_split(
    num_samples: int,
    feature_directions: Tensor,
    config: PolarityDialDataConfig,
    device: str,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor]:
    coefficients = _sample_coefficients(num_samples, config, device, generator)
    x_clean = coefficients @ feature_directions
    noise = torch.randn(
        (num_samples, config.input_dim), device=device, generator=generator
    )
    x = x_clean + config.noise_std * noise
    return x, coefficients


def generate_polarity_dial_dataset(
    config: PolarityDialDataConfig,
    device: str,
) -> Dict[str, Tensor]:
    set_random_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    feature_directions = torch.randn(
        (config.num_features, config.input_dim), device=device, generator=generator
    )
    feature_directions = feature_directions / (
        feature_directions.norm(dim=1, keepdim=True) + 1e-8
    )

    x_train, coefficients_train = _generate_split(
        config.num_train, feature_directions, config, device, generator
    )
    x_val, coefficients_val = _generate_split(
        config.num_val, feature_directions, config, device, generator
    )
    x_test, coefficients_test = _generate_split(
        config.num_test, feature_directions, config, device, generator
    )

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "feature_directions": feature_directions,
        "coefficients_train": coefficients_train,
        "coefficients_val": coefficients_val,
        "coefficients_test": coefficients_test,
    }


def _tensor_stats(values: Tensor) -> Dict[str, Any]:
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _collect_coefficients_stats(coefficients: Tensor) -> Dict[str, Any]:
    flattened = coefficients.reshape(-1, coefficients.shape[-1])
    positive_values = flattened[flattened > 0]
    negative_values = -flattened[flattened < 0]
    sparsity = float((flattened == 0).float().mean().item())
    per_feature_stats = []
    for feature_idx in range(flattened.shape[1]):
        feature_values = flattened[:, feature_idx]
        feature_positive = feature_values[feature_values > 0]
        feature_negative = -feature_values[feature_values < 0]
        per_feature_stats.append(
            {
                "feature_index": feature_idx,
                "positive": _tensor_stats(feature_positive),
                "negative": _tensor_stats(feature_negative),
                "sparsity": float((feature_values == 0).float().mean().item()),
            }
        )
    return {
        "overall": {
            "positive": _tensor_stats(positive_values),
            "negative": _tensor_stats(negative_values),
            "sparsity": sparsity,
        },
        "per_feature": per_feature_stats,
    }


def export_data_summary(
    run_dir: str,
    config: PolarityDialDataConfig,
    data: Dict[str, Tensor],
    sample_preview_count: int,
) -> None:
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    generator_config_path = os.path.join(data_dir, "generator_config.json")
    with open(generator_config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    feature_directions_path = os.path.join(data_dir, "feature_directions.pt")
    torch.save(data["feature_directions"].cpu(), feature_directions_path)

    coefficients = torch.cat(
        (data["coefficients_train"], data["coefficients_val"], data["coefficients_test"]),
        dim=0,
    )
    coefficients_stats_path = os.path.join(data_dir, "coefficients_stats.json")
    with open(coefficients_stats_path, "w") as f:
        json.dump(_collect_coefficients_stats(coefficients.cpu()), f, indent=2)

    sample_count = min(sample_preview_count, data["x_train"].shape[0])
    sampled_data_path = os.path.join(data_dir, "sampled_data.pt")
    torch.save(
        {
            "x": data["x_train"][:sample_count].cpu(),
            "coefficients": data["coefficients_train"][:sample_count].cpu(),
        },
        sampled_data_path,
    )
