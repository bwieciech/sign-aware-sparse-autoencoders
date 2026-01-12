import json
import math
import os
import random
from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from sgnsae.experiments.toy_model_geometry_validation import config as config_module
from sgnsae.experiments.toy_model_geometry_validation.config import Toy2DDataConfig


_TORCH_SEED_MODULUS = 2**63
_NUMPY_SEED_MODULUS = 2**32
_DIRECTION_EPSILON = 1e-8


def set_random_seed(seed: int) -> None:
    normalized_torch_seed = int(seed % _TORCH_SEED_MODULUS)
    normalized_numpy_seed = int(seed % _NUMPY_SEED_MODULUS)
    random.seed(normalized_torch_seed)
    np.random.seed(normalized_numpy_seed)
    torch.manual_seed(normalized_torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_torch_seed)


def _create_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed % _TORCH_SEED_MODULUS))
    return generator


def _generate_pair_axis_angles(
    config: Toy2DDataConfig,
    generator: torch.Generator,
) -> Tensor:
    if config.direction_mode == "uniform":
        indices = torch.arange(config.num_pairs, dtype=torch.float32, device="cpu")
        return indices * (math.pi / float(config.num_pairs))
    if config.direction_mode == "random":
        uniforms = torch.rand(
            (config.num_pairs,), generator=generator, device="cpu", dtype=torch.float32
        )
        return uniforms * math.pi
    raise ValueError(f"Unsupported direction_mode={config.direction_mode}.")


def _construct_pair_directions(pair_axis_angles: Tensor) -> Tuple[Tensor, Tensor]:
    if pair_axis_angles.ndim != 1:
        raise ValueError(
            "pair_axis_angles must be a 1D tensor; "
            f"received shape={tuple(pair_axis_angles.shape)}."
        )
    axis = torch.stack(
        (torch.cos(pair_axis_angles), torch.sin(pair_axis_angles)), dim=1
    ).to(dtype=torch.float32)
    axis = axis / (torch.linalg.norm(axis, dim=1, keepdim=True) + _DIRECTION_EPSILON)

    feature_directions = torch.empty(
        (2 * axis.shape[0], 2), device="cpu", dtype=torch.float32
    )
    feature_directions[0::2] = axis
    feature_directions[1::2] = -axis
    return axis, feature_directions


def _sample_pair_active_mask(
    num_samples: int,
    config: Toy2DDataConfig,
    generator: torch.Generator,
) -> Tensor:
    if not 0.0 <= config.pair_axis_active_prob <= 1.0:
        raise ValueError(
            "pair_axis_active_prob must be within [0, 1]; "
            f"received {config.pair_axis_active_prob}."
        )
    rho = float(config.pair_axis_active_prob)
    k = int(config.num_pairs)
    if config.active_pairs_per_sample is None:
        g = torch.bernoulli(torch.full((num_samples, k), rho, device="cpu"))
        return g.to(dtype=torch.bool)
    n = int(config.active_pairs_per_sample)
    if not (0 <= n <= k):
        raise ValueError(
            f"active_pairs_per_sample must be in [0,k]; got {n} for k={k}."
        )
    g = torch.zeros((num_samples, k), device="cpu", dtype=torch.bool)
    # Sample exactly n pairs without replacement per sample
    for i in range(num_samples):
        idx = torch.randperm(k, device="cpu")[:n]
        g[i, idx] = True
    return g


def _calculate_within_pair_joint_probabilities(
    pair_feature_prob: float,
    pair_correlation: float,
) -> Tuple[float, float, float, float]:
    p = float(pair_feature_prob)
    r = float(pair_correlation)
    if not -1.0 <= r <= 1.0:
        raise ValueError(
            "pair_correlation must be within [-1, 1]; "
            f"received pair_correlation={pair_correlation}."
        )

    r_min, r_max = config_module._calculate_pair_correlation_feasible_interval(p)
    if r < r_min or r > r_max:
        raise ValueError(
            f"pair_correlation={pair_correlation} is infeasible for pair_feature_prob={pair_feature_prob}; "
            f"feasible interval is [{r_min}, {r_max}]."
        )

    q = p * p + r * p * (1.0 - p)

    p11 = q
    p10 = p - q
    p01 = p - q
    p00 = 1.0 - 2.0 * p + q

    return float(p00), float(p01), float(p10), float(p11)


def _sample_within_pair_presence(
    num_samples: int,
    config: Toy2DDataConfig,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    if config.pair_correlation is None:
        raise ValueError("pair_correlation must be provided to sample toy data.")
    if config.pair_correlation_grid is not None:
        raise ValueError(
            "pair_correlation_grid must be resolved by the runner before calling the "
            "data generator."
        )

    p00, p01, p10, p11 = _calculate_within_pair_joint_probabilities(
        pair_feature_prob=config.pair_feature_prob,
        pair_correlation=float(config.pair_correlation),
    )

    tolerance = 1e-12
    if min(p00, p01, p10, p11) < -tolerance:
        raise ValueError(
            "Within-pair joint probabilities must be non-negative; received "
            f"p00={p00}, p01={p01}, p10={p10}, p11={p11}."
        )

    p00 = max(0.0, p00)
    p01 = max(0.0, p01)
    p10 = max(0.0, p10)
    p11 = max(0.0, p11)
    total = p00 + p01 + p10 + p11
    if total <= 0.0:
        raise ValueError("Within-pair joint probabilities are degenerate.")
    p00, p01, p10, p11 = (p00 / total, p01 / total, p10 / total, p11 / total)

    uniforms = torch.rand(
        (num_samples, config.num_pairs),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    threshold_00 = p00
    threshold_01 = threshold_00 + p01
    threshold_10 = threshold_01 + p10

    is_01 = (uniforms >= threshold_00) & (uniforms < threshold_01)
    is_10 = (uniforms >= threshold_01) & (uniforms < threshold_10)
    is_11 = uniforms >= threshold_10

    plus_presence = is_10 | is_11
    minus_presence = is_01 | is_11
    return (
        plus_presence,
        minus_presence,
        {"p00": p00, "p01": p01, "p10": p10, "p11": p11},
    )


def _sample_magnitudes(
    shape: Tuple[int, ...],
    config: Toy2DDataConfig,
    generator: torch.Generator,
) -> Tensor:
    if config.magnitude_dist == "constant":
        return torch.full(
            shape,
            float(config.constant_magnitude),
            device="cpu",
            dtype=torch.float32,
        )
    if config.magnitude_dist == "lognormal":
        normal = torch.randn(
            shape, device="cpu", dtype=torch.float32, generator=generator
        )
        return (
            normal * float(config.lognormal_sigma) + float(config.lognormal_mu)
        ).exp()
    raise ValueError(f"Unsupported magnitude_dist={config.magnitude_dist}.")


def _generate_split(
    num_samples: int,
    feature_directions: Tensor,
    config: Toy2DDataConfig,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
    pair_active_mask = _sample_pair_active_mask(num_samples, config, generator)
    plus_presence, minus_presence, joint_probs = _sample_within_pair_presence(
        num_samples, config, generator
    )

    plus_active = pair_active_mask & plus_presence
    minus_active = pair_active_mask & minus_presence

    magnitudes = _sample_magnitudes(
        (num_samples, config.num_pairs, 2), config, generator
    )
    plus_magnitudes = magnitudes[:, :, 0]
    minus_magnitudes = magnitudes[:, :, 1]

    plus_coeffs = plus_active.to(dtype=torch.float32) * plus_magnitudes
    minus_coeffs = minus_active.to(dtype=torch.float32) * minus_magnitudes

    coeffs = torch.zeros(
        (num_samples, 2 * config.num_pairs), device="cpu", dtype=torch.float32
    )
    coeffs[:, 0::2] = plus_coeffs
    coeffs[:, 1::2] = minus_coeffs

    x_clean = coeffs @ feature_directions
    if float(config.noise_std) > 0.0:
        noise = torch.randn(
            (num_samples, 2), device="cpu", dtype=torch.float32, generator=generator
        )
        x = x_clean + float(config.noise_std) * noise
    else:
        x = x_clean

    return x, coeffs, pair_active_mask, joint_probs


def generate_toy2d_dataset(
    config: Toy2DDataConfig,
    device: str,
) -> Dict[str, Tensor]:
    if isinstance(config.seed, list):
        raise ValueError(
            "seed must be a single integer when generating toy data; "
            "resolve multi-seed configurations in the runner."
        )
    seed = int(config.seed)
    set_random_seed(seed)
    generator = _create_cpu_generator(seed)
    pair_axis_angles = _generate_pair_axis_angles(config, generator)
    pair_axis_directions, feature_directions = _construct_pair_directions(
        pair_axis_angles
    )

    x_train, coeffs_train, pair_active_mask_train, joint_probs = _generate_split(
        config.num_train, feature_directions, config, generator
    )
    x_val, coeffs_val, pair_active_mask_val, _ = _generate_split(
        config.num_val, feature_directions, config, generator
    )
    x_test, coeffs_test, pair_active_mask_test, _ = _generate_split(
        config.num_test, feature_directions, config, generator
    )

    return {
        "pair_axis_angles_rad": pair_axis_angles.to(device=device),
        "pair_axis_directions": pair_axis_directions.to(device=device),
        "feature_directions": feature_directions.to(device=device),
        "x_train": x_train.to(device=device),
        "x_val": x_val.to(device=device),
        "x_test": x_test.to(device=device),
        "coeffs_train": coeffs_train.to(device=device),
        "coeffs_val": coeffs_val.to(device=device),
        "coeffs_test": coeffs_test.to(device=device),
        "pair_active_mask_train": pair_active_mask_train.to(device=device),
        "pair_active_mask_val": pair_active_mask_val.to(device=device),
        "pair_active_mask_test": pair_active_mask_test.to(device=device),
        "within_pair_joint_probabilities": torch.tensor(
            [
                joint_probs["p00"],
                joint_probs["p01"],
                joint_probs["p10"],
                joint_probs["p11"],
            ],
            device=device,
            dtype=torch.float32,
        ),
    }


def export_toy2d_dataset(
    run_dir: str,
    config: Toy2DDataConfig,
    data: Dict[str, Tensor],
    sample_preview_count: int,
) -> None:
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    pair_angles = data.get("pair_axis_angles_rad")
    if pair_angles is None:
        raise ValueError("data is missing required key 'pair_axis_angles_rad'.")

    joint_probs = data.get("within_pair_joint_probabilities")
    if joint_probs is None:
        raise ValueError(
            "data is missing required key 'within_pair_joint_probabilities'."
        )

    generator_config_path = os.path.join(data_dir, "generator_config.json")
    feasible_r_min, feasible_r_max = (
        config_module._calculate_pair_correlation_feasible_interval(
            float(config.pair_feature_prob)
        )
    )
    q_value = None
    if config.pair_correlation is not None:
        q_value = float(config.pair_feature_prob) ** 2 + float(
            config.pair_correlation
        ) * float(config.pair_feature_prob) * (1.0 - float(config.pair_feature_prob))
    generator_payload: Dict[str, Any] = {
        **asdict(config),
        "pair_axis_angles_rad": [
            float(value) for value in pair_angles.detach().cpu().tolist()
        ],
        "pair_correlation_feasible_interval": {
            "r_min": float(feasible_r_min),
            "r_max": float(feasible_r_max),
        },
        "within_pair_joint_probability_q": q_value,
        "within_pair_joint_probabilities": {
            "p00": float(joint_probs[0].detach().cpu().item()),
            "p01": float(joint_probs[1].detach().cpu().item()),
            "p10": float(joint_probs[2].detach().cpu().item()),
            "p11": float(joint_probs[3].detach().cpu().item()),
        },
    }
    with open(generator_config_path, "w") as f:
        json.dump(generator_payload, f, indent=2)

    torch.save(
        data["feature_directions"].detach().cpu(),
        os.path.join(data_dir, "feature_directions.pt"),
    )

    for split in ("train", "val", "test"):
        torch.save(
            data[f"pair_active_mask_{split}"].detach().cpu(),
            os.path.join(data_dir, f"pair_active_mask_{split}.pt"),
        )
        torch.save(
            data[f"coeffs_{split}"].detach().cpu(),
            os.path.join(data_dir, f"coeffs_{split}.pt"),
        )
        torch.save(
            data[f"x_{split}"].detach().cpu(),
            os.path.join(data_dir, f"x_{split}.pt"),
        )

    preview_payload: Dict[str, Tensor] = {}
    for split in ("train", "val", "test"):
        split_count = int(data[f"x_{split}"].shape[0])
        sample_count = min(int(sample_preview_count), split_count)
        preview_payload[f"x_{split}"] = data[f"x_{split}"][:sample_count].detach().cpu()
        preview_payload[f"coeffs_{split}"] = (
            data[f"coeffs_{split}"][:sample_count].detach().cpu()
        )
        preview_payload[f"pair_active_mask_{split}"] = (
            data[f"pair_active_mask_{split}"][:sample_count].detach().cpu()
        )

    torch.save(preview_payload, os.path.join(data_dir, "sampled_data.pt"))
