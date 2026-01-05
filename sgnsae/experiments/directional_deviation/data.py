import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationAnomalyConfig,
    DirectionalDeviationDataConfig,
)
from sgnsae.experiments.directional_deviation.scoring import extract_detection_signal


LABEL_TYPE_NORMAL = 0
LABEL_TYPE_HIGH = 1
LABEL_TYPE_LOW = 2

_TORCH_SEED_MODULUS = 2**63
_NUMPY_SEED_MODULUS = 2**32
_UNIFORM_EPSILON = 1e-6
_DIRECTION_EPSILON = 1e-8


def set_random_seed(seed: int) -> None:
    normalized_torch_seed = int(seed % _TORCH_SEED_MODULUS)
    normalized_numpy_seed = int(seed % _NUMPY_SEED_MODULUS)
    random.seed(normalized_torch_seed)
    np.random.seed(normalized_numpy_seed)
    torch.manual_seed(normalized_torch_seed)


def _normalize_torch_seed(seed: int) -> int:
    return int(seed % _TORCH_SEED_MODULUS)


def _create_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_normalize_torch_seed(seed))
    return generator


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}.")


def _validate_feature_directions(
    directions: Tensor,
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
) -> None:
    if anomaly_config.num_directions is None:
        raise ValueError(
            "num_directions must be provided when shift_mode='feature_direction'."
        )
    _validate_positive_int(int(anomaly_config.num_directions), "num_directions")
    if directions.ndim != 2:
        raise ValueError(f"Expected directions.ndim == 2, got {directions.ndim}.")
    if directions.shape[0] != int(anomaly_config.num_directions):
        raise ValueError(
            "Expected directions.shape[0] == num_directions, got "
            f"{directions.shape[0]} vs {anomaly_config.num_directions}."
        )
    if directions.shape[1] != int(data_config.num_channels):
        raise ValueError(
            "Expected directions.shape[1] == num_channels, got "
            f"{directions.shape[1]} vs {data_config.num_channels}."
        )


def _extract_decoder_matrix(model: torch.nn.Module) -> Tensor:
    if not hasattr(model, "W_dec"):
        raise ValueError("Model is missing required attribute W_dec.")
    decoder = getattr(model, "W_dec")
    if not isinstance(decoder, Tensor):
        raise ValueError("Model attribute W_dec must be a torch.Tensor.")
    if decoder.ndim != 3:
        raise ValueError(f"Expected W_dec.ndim == 3, got {decoder.ndim}.")
    if not hasattr(model, "n_instances") or int(getattr(model, "n_instances")) != 1:
        raise ValueError(
            "Feature-direction extraction requires model.n_instances == 1."
        )
    return decoder[0].detach().to(device="cpu")


def _calculate_latent_purity(decoder: Tensor) -> Tensor:
    if decoder.ndim != 2:
        raise ValueError(f"Expected decoder.ndim == 2, got {decoder.ndim}.")
    max_abs = decoder.abs().max(dim=0).values
    norms = torch.linalg.norm(decoder, dim=0)
    return max_abs / (norms + _DIRECTION_EPSILON)


def _select_latent_indices_top_purity(purity: Tensor, num_directions: int) -> List[int]:
    purity_values = [float(value) for value in purity.cpu().tolist()]
    indices = list(range(len(purity_values)))
    indices.sort(key=lambda idx: (-purity_values[idx], idx))
    return [int(idx) for idx in indices[:num_directions]]


def _select_latent_indices_random(
    num_latents: int,
    num_directions: int,
    generator: torch.Generator,
) -> List[int]:
    if num_directions > num_latents:
        raise ValueError(
            "num_directions must be <= the number of available latents; received "
            f"num_directions={num_directions} and num_latents={num_latents}."
        )
    perm = torch.randperm(num_latents, generator=generator, device="cpu")
    return [int(value) for value in perm[:num_directions].tolist()]


def _select_latent_indices_top_weight_norm(
    weight_norms: Tensor, num_directions: int
) -> List[int]:
    norm_values = [float(value) for value in weight_norms.cpu().tolist()]
    indices = list(range(len(norm_values)))
    indices.sort(key=lambda idx: (-norm_values[idx], idx))
    return [int(idx) for idx in indices[:num_directions]]


def _calculate_latent_pi_std(
    model: torch.nn.Module,
    base_samples: Tensor,
    batch_size: int,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; received {batch_size}.")
    if base_samples.ndim != 2:
        raise ValueError(f"Expected base_samples.ndim == 2, got {base_samples.ndim}.")
    model.eval()
    sums: Optional[Tensor] = None
    sumsq: Optional[Tensor] = None
    total_count = 0
    with torch.no_grad():
        for start in range(0, int(base_samples.shape[0]), batch_size):
            batch = base_samples[start : start + batch_size]
            signal = extract_detection_signal(model, batch)
            signal_cpu = signal.detach().to(device="cpu", dtype=torch.float64)
            if sums is None:
                sums = torch.zeros(
                    (signal_cpu.shape[1],), device="cpu", dtype=torch.float64
                )
                sumsq = torch.zeros_like(sums)
            sums += signal_cpu.sum(dim=0)
            sumsq += (signal_cpu * signal_cpu).sum(dim=0)
            total_count += int(signal_cpu.shape[0])
    if sums is None or sumsq is None or total_count == 0:
        raise ValueError("base_samples must contain at least one sample.")
    mean = sums / float(total_count)
    variance = sumsq / float(total_count) - mean * mean
    variance = torch.clamp(variance, min=0.0)
    return torch.sqrt(variance).to(dtype=torch.float32)


def build_feature_directions_from_decoder(
    model: torch.nn.Module,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    direction_seed: int,
    base_samples: Optional[Tensor] = None,
    batch_size: int = 4096,
) -> Tuple[Tensor, Dict[str, Any]]:
    if anomaly_config.shift_mode != "feature_direction":
        raise ValueError(
            "build_feature_directions_from_decoder requires shift_mode='feature_direction'."
        )
    if anomaly_config.num_directions is None:
        raise ValueError("num_directions must be provided.")

    decoder = _extract_decoder_matrix(model)
    purity = _calculate_latent_purity(decoder)
    weight_norms = torch.linalg.norm(decoder, dim=0)
    num_directions = int(anomaly_config.num_directions)
    num_latents = int(decoder.shape[1])

    selection = anomaly_config.direction_selection
    if selection is None:
        raise ValueError(
            "direction_selection must be provided when shift_mode='feature_direction'."
        )
    if selection == "top_purity":
        latent_indices = _select_latent_indices_top_purity(purity, num_directions)
        selected_pi_std = None
        selected_weight_norms = None
    elif selection == "top_pi_std":
        if base_samples is None:
            raise ValueError(
                "base_samples must be provided when direction_selection='top_pi_std'."
            )
        pi_std = _calculate_latent_pi_std(
            model=model,
            base_samples=base_samples,
            batch_size=batch_size,
        )
        latent_indices = _select_latent_indices_top_weight_norm(pi_std, num_directions)
        selected_pi_std = [float(value) for value in pi_std[latent_indices].tolist()]
        selected_weight_norms = None
    elif selection == "top_weight_norm":
        latent_indices = _select_latent_indices_top_weight_norm(
            weight_norms, num_directions
        )
        selected_pi_std = None
        selected_weight_norms = [
            float(value) for value in weight_norms[latent_indices].tolist()
        ]
    elif selection == "random":
        generator = _create_cpu_generator(direction_seed)
        latent_indices = _select_latent_indices_random(
            num_latents=num_latents,
            num_directions=num_directions,
            generator=generator,
        )
        selected_pi_std = None
        selected_weight_norms = None
    elif selection == "explicit":
        if anomaly_config.direction_latent_indices is None:
            raise ValueError(
                "direction_latent_indices must be provided when direction_selection='explicit'."
            )
        latent_indices = [
            int(value) for value in anomaly_config.direction_latent_indices
        ]
        selected_pi_std = None
        selected_weight_norms = None
    else:
        raise ValueError(f"Unsupported direction_selection={selection}.")

    if any(index >= num_latents for index in latent_indices):
        raise ValueError(
            "direction_latent_indices entries must be < the decoder width; received "
            f"indices={latent_indices} and decoder_width={num_latents}."
        )

    directions = decoder[:, latent_indices].T.to(dtype=torch.float32)
    selected_weight_norms_raw = weight_norms[latent_indices].to(dtype=torch.float32)
    if anomaly_config.normalize_directions:
        norms = torch.linalg.norm(directions, dim=1, keepdim=True)
        directions = directions / (norms + _DIRECTION_EPSILON)

    selected_purity = [float(value) for value in purity[latent_indices].tolist()]
    direction_delta_mode = anomaly_config.direction_delta_mode
    if direction_delta_mode is None:
        raise ValueError(
            "direction_delta_mode must be provided when shift_mode='feature_direction'."
        )
    direction_input_deltas: Optional[List[float]] = None
    clipped_indices: Optional[List[int]] = None
    max_direction_input_delta = anomaly_config.max_direction_input_delta
    if direction_delta_mode == "fixed_pi_shift":
        log_alpha_gate = getattr(model, "log_alpha_gate", None)
        if isinstance(log_alpha_gate, Tensor):
            log_alpha_gate_cpu = log_alpha_gate.detach().to(
                device="cpu", dtype=torch.float32
            )
            if log_alpha_gate_cpu.ndim != 2 or log_alpha_gate_cpu.shape[0] != 1:
                raise ValueError(
                    "log_alpha_gate must have shape (1, d_hidden) when n_instances==1."
                )
            exp_clamp = float(getattr(model, "_exp_clamp", 20.0))
            log_alpha_gate_cpu = log_alpha_gate_cpu.clamp(min=-exp_clamp, max=exp_clamp)
            alpha = torch.exp(log_alpha_gate_cpu)[0]
            selected_alpha = alpha[latent_indices].to(dtype=torch.float32)
            input_deltas = float(anomaly_config.delta) / (
                selected_alpha + _DIRECTION_EPSILON
            )
        else:
            input_deltas = float(anomaly_config.delta) / (
                selected_weight_norms_raw + _DIRECTION_EPSILON
            )
        clipped_mask = torch.zeros_like(input_deltas, dtype=torch.bool)
        if max_direction_input_delta is not None:
            cap = float(max_direction_input_delta)
            clipped_mask = input_deltas > cap
            input_deltas = torch.clamp(input_deltas, max=cap)
        direction_input_deltas = [float(value) for value in input_deltas.tolist()]
        clipped_indices = [
            int(index) for index in clipped_mask.nonzero(as_tuple=False).flatten()
        ]

    metadata: Dict[str, Any] = {
        "direction_source": anomaly_config.direction_source,
        "direction_selection": selection,
        "direction_latent_indices": latent_indices,
        "direction_latent_purity": selected_purity,
        "direction_latent_pi_std": selected_pi_std,
        "direction_latent_weight_norm": selected_weight_norms,
        "direction_delta_mode": direction_delta_mode,
        "direction_input_deltas": direction_input_deltas,
        "max_direction_input_delta": max_direction_input_delta,
        "direction_input_deltas_clipped_indices": clipped_indices,
        "normalize_directions": bool(anomaly_config.normalize_directions),
        "seed_plan": {"direction_seed": _normalize_torch_seed(direction_seed)},
    }
    return directions, metadata


def build_fixed_random_directions(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    direction_seed: int,
) -> Tuple[Tensor, Dict[str, Any]]:
    if anomaly_config.shift_mode != "feature_direction":
        raise ValueError(
            "build_fixed_random_directions requires shift_mode='feature_direction'."
        )
    if anomaly_config.num_directions is None:
        raise ValueError("num_directions must be provided.")
    generator = _create_cpu_generator(direction_seed)
    num_directions = int(anomaly_config.num_directions)
    raw_directions = torch.randn(
        (num_directions, data_config.num_channels),
        device="cpu",
        dtype=torch.float32,
        generator=generator,
    )
    weight_norms_raw = torch.linalg.norm(raw_directions, dim=1)
    directions = raw_directions
    if anomaly_config.normalize_directions:
        norms = torch.linalg.norm(directions, dim=1, keepdim=True)
        directions = directions / (norms + _DIRECTION_EPSILON)

    direction_delta_mode = anomaly_config.direction_delta_mode
    if direction_delta_mode is None:
        raise ValueError(
            "direction_delta_mode must be provided when shift_mode='feature_direction'."
        )
    direction_input_deltas: Optional[List[float]] = None
    clipped_indices: Optional[List[int]] = None
    max_direction_input_delta = anomaly_config.max_direction_input_delta
    if direction_delta_mode == "fixed_pi_shift":
        input_deltas = float(anomaly_config.delta) / (
            weight_norms_raw.to(dtype=torch.float32) + _DIRECTION_EPSILON
        )
        clipped_mask = torch.zeros_like(input_deltas, dtype=torch.bool)
        if max_direction_input_delta is not None:
            cap = float(max_direction_input_delta)
            clipped_mask = input_deltas > cap
            input_deltas = torch.clamp(input_deltas, max=cap)
        direction_input_deltas = [float(value) for value in input_deltas.tolist()]
        clipped_indices = [
            int(index) for index in clipped_mask.nonzero(as_tuple=False).flatten()
        ]

    metadata: Dict[str, Any] = {
        "direction_source": "fixed_random",
        "direction_selection": "fixed_random",
        "direction_delta_mode": direction_delta_mode,
        "direction_input_deltas": direction_input_deltas,
        "max_direction_input_delta": max_direction_input_delta,
        "direction_input_deltas_clipped_indices": clipped_indices,
        "normalize_directions": bool(anomaly_config.normalize_directions),
        "seed_plan": {"direction_seed": _normalize_torch_seed(direction_seed)},
    }
    return directions, metadata


def _sample_iid_non_gaussian_split(
    num_samples: int,
    config: DirectionalDeviationDataConfig,
    generator: torch.Generator,
) -> Tensor:
    if config.non_gaussian_dist is None:
        raise ValueError(
            f"non_gaussian_dist must be provided when normal_mode='{config.normal_mode}'."
        )

    if config.non_gaussian_dist == "laplace":
        scale = 1.0 / math.sqrt(2.0)
        u = (
            torch.rand(
                (num_samples, config.num_channels),
                device="cpu",
                dtype=torch.float32,
                generator=generator,
            )
            .clamp(min=_UNIFORM_EPSILON)
            .sub(0.5)
        )
        return -scale * torch.sign(u) * torch.log1p(-2.0 * torch.abs(u))

    if config.non_gaussian_dist == "logistic":
        scale = math.sqrt(3.0) / math.pi
        u = torch.rand(
            (num_samples, config.num_channels),
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        ).clamp(min=_UNIFORM_EPSILON, max=1.0 - _UNIFORM_EPSILON)
        return scale * torch.logit(u)

    if config.non_gaussian_dist == "student_t":
        if config.student_t_df is None:
            raise ValueError(
                "student_t_df must be provided when non_gaussian_dist='student_t'."
            )
        df = float(config.student_t_df)
        z = torch.randn(
            (num_samples, config.num_channels),
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        )
        gamma_samples = torch._standard_gamma(
            torch.full(
                (num_samples, config.num_channels),
                df / 2.0,
                device="cpu",
                dtype=torch.float32,
            ),
            generator=generator,
        )
        chi_squared = 2.0 * gamma_samples
        student_t_samples = z / torch.sqrt(chi_squared / df)
        unit_variance_scale = math.sqrt((df - 2.0) / df)
        return unit_variance_scale * student_t_samples

    raise ValueError(f"Unsupported non_gaussian_dist={config.non_gaussian_dist}.")


def _sample_normal_split(
    num_samples: int,
    config: DirectionalDeviationDataConfig,
    generator: torch.Generator,
) -> Tensor:
    if config.normal_mode in {"standard_normal", "heteroskedastic_normal"}:
        samples = torch.randn(
            (num_samples, config.num_channels),
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        )
    elif config.normal_mode in {"iid_non_gaussian", "non_gaussian_heteroskedastic"}:
        samples = _sample_iid_non_gaussian_split(num_samples, config, generator)
    else:
        raise ValueError(f"Unsupported normal_mode={config.normal_mode}.")

    if config.normal_mode in {
        "heteroskedastic_normal",
        "non_gaussian_heteroskedastic",
    }:
        if config.channel_stds is None:
            raise ValueError(
                f"channel_stds must be provided when normal_mode='{config.normal_mode}'."
            )
        channel_stds = torch.tensor(
            config.channel_stds, device="cpu", dtype=torch.float32
        ).unsqueeze(0)
        samples = samples * channel_stds
    if config.noise_std > 0:
        noise = torch.randn(
            (num_samples, config.num_channels),
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        )
        samples = samples + config.noise_std * noise
    return samples


def generate_normal_data(
    config: DirectionalDeviationDataConfig,
    device: str,
) -> Dict[str, Tensor]:
    set_random_seed(config.seed)

    train_generator = _create_cpu_generator(config.seed + 0)
    val_generator = _create_cpu_generator(config.seed + 1)
    test_generator = _create_cpu_generator(config.seed + 2)

    x_train = _sample_normal_split(config.num_train, config, train_generator).to(device)
    x_val = _sample_normal_split(config.num_val, config, val_generator).to(device)
    x_test_normal = _sample_normal_split(
        config.num_test_normal, config, test_generator
    ).to(device)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test_normal": x_test_normal,
    }


def _resolve_channels_to_evaluate(
    num_channels: int,
    channels_to_evaluate: Optional[List[int]],
) -> List[int]:
    if channels_to_evaluate is None:
        return list(range(num_channels))
    return list(channels_to_evaluate)


def _select_subset_indices(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    seed: int,
) -> List[int]:
    if anomaly_config.shift_mode != "subset":
        raise ValueError("subset indices are only defined for shift_mode='subset'.")
    if anomaly_config.subset_size is None:
        raise ValueError("subset_size must be provided when shift_mode='subset'.")

    candidate_channels = _resolve_channels_to_evaluate(
        data_config.num_channels, anomaly_config.channels_to_evaluate
    )
    if anomaly_config.subset_size > len(candidate_channels):
        raise ValueError(
            "subset_size must not exceed the number of candidate channels; received "
            f"subset_size={anomaly_config.subset_size} and num_candidates={len(candidate_channels)}."
        )

    generator = _create_cpu_generator(seed)
    candidate_tensor = torch.tensor(candidate_channels, dtype=torch.int64, device="cpu")
    perm = torch.randperm(candidate_tensor.numel(), generator=generator, device="cpu")
    chosen = candidate_tensor[perm[: anomaly_config.subset_size]].tolist()
    return sorted(int(value) for value in chosen)


def _build_anomaly_manifest(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    channels_to_evaluate: List[int],
    subset_indices: Optional[List[int]],
    subset_seed: int,
    sample_seed: int,
) -> Dict[str, Any]:
    if anomaly_config.shift_mode == "single_channel":
        total_high = len(channels_to_evaluate) * data_config.num_test_high_per_channel
        total_low = len(channels_to_evaluate) * data_config.num_test_low_per_channel
    else:
        total_high = data_config.num_test_high_per_channel
        total_low = data_config.num_test_low_per_channel

    return {
        "delta": anomaly_config.delta,
        "shift_mode": anomaly_config.shift_mode,
        "num_channels": data_config.num_channels,
        "channels_to_evaluate": channels_to_evaluate,
        "subset_size": anomaly_config.subset_size,
        "subset_indices": subset_indices,
        "num_test_high_per_channel": data_config.num_test_high_per_channel,
        "num_test_low_per_channel": data_config.num_test_low_per_channel,
        "counts": {
            "high": total_high,
            "low": total_low,
            "total": total_high + total_low,
        },
        "label_mapping": {
            "normal": LABEL_TYPE_NORMAL,
            "high": LABEL_TYPE_HIGH,
            "low": LABEL_TYPE_LOW,
        },
        "seed_plan": {
            "subset_seed": _normalize_torch_seed(subset_seed),
            "sample_seed": _normalize_torch_seed(sample_seed),
        },
    }


def _build_feature_direction_anomaly_manifest(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    direction_metadata: Dict[str, Any],
    direction_seed: int,
    sample_seed: int,
) -> Dict[str, Any]:
    if anomaly_config.shift_mode != "feature_direction":
        raise ValueError(
            "_build_feature_direction_anomaly_manifest requires shift_mode='feature_direction'."
        )
    if anomaly_config.num_directions is None:
        raise ValueError("num_directions must be provided.")
    num_directions = int(anomaly_config.num_directions)

    num_high = (
        int(anomaly_config.num_test_high_per_direction)
        if anomaly_config.num_test_high_per_direction is not None
        else int(data_config.num_test_high_per_channel)
    )
    num_low = (
        int(anomaly_config.num_test_low_per_direction)
        if anomaly_config.num_test_low_per_direction is not None
        else int(data_config.num_test_low_per_channel)
    )
    total_high = num_directions * num_high
    total_low = num_directions * num_low

    return {
        "delta": anomaly_config.delta,
        "shift_mode": anomaly_config.shift_mode,
        "target_kind": "direction",
        "num_channels": data_config.num_channels,
        "num_directions": num_directions,
        "base_samples": anomaly_config.base_samples,
        "direction_source": anomaly_config.direction_source,
        "reference_model_name": (
            anomaly_config.reference_model_name
            if anomaly_config.direction_source == "reference_model_decoder"
            else None
        ),
        "direction_selection": direction_metadata.get(
            "direction_selection",
            (
                "fixed_random"
                if anomaly_config.direction_source == "fixed_random"
                else anomaly_config.direction_selection
            ),
        ),
        "direction_latent_indices": direction_metadata.get("direction_latent_indices"),
        "direction_latent_purity": direction_metadata.get("direction_latent_purity"),
        "direction_latent_pi_std": direction_metadata.get("direction_latent_pi_std"),
        "direction_latent_weight_norm": direction_metadata.get(
            "direction_latent_weight_norm"
        ),
        "direction_delta_mode": direction_metadata.get(
            "direction_delta_mode", anomaly_config.direction_delta_mode
        ),
        "max_direction_input_delta": direction_metadata.get(
            "max_direction_input_delta", anomaly_config.max_direction_input_delta
        ),
        "direction_input_deltas": direction_metadata.get("direction_input_deltas"),
        "direction_input_deltas_clipped_indices": direction_metadata.get(
            "direction_input_deltas_clipped_indices"
        ),
        "normalize_directions": bool(anomaly_config.normalize_directions),
        "direction_indices_to_evaluate": list(range(num_directions)),
        "num_test_high_per_direction": num_high,
        "num_test_low_per_direction": num_low,
        "counts": {
            "high": total_high,
            "low": total_low,
            "total": total_high + total_low,
        },
        "label_mapping": {
            "normal": LABEL_TYPE_NORMAL,
            "high": LABEL_TYPE_HIGH,
            "low": LABEL_TYPE_LOW,
        },
        "seed_plan": {
            "direction_seed": _normalize_torch_seed(direction_seed),
            "sample_seed": _normalize_torch_seed(sample_seed),
        },
    }


@dataclass(frozen=True)
class DirectionalDeviationAnomalyBatch:
    x: Tensor
    labels_type: Tensor
    labels_channel: Tensor
    labels_mask: Optional[Tensor]


@dataclass(frozen=True)
class DirectionalDeviationAnomalyStream:
    data_config: DirectionalDeviationDataConfig
    anomaly_config: DirectionalDeviationAnomalyConfig
    device: str
    batch_size: int
    channels_to_evaluate: List[int]
    subset_indices: Optional[List[int]]
    sample_seed: int

    def __iter__(self) -> Iterator[DirectionalDeviationAnomalyBatch]:
        generator = _create_cpu_generator(self.sample_seed)
        delta = float(self.anomaly_config.delta)

        if self.anomaly_config.shift_mode == "single_channel":
            yield from self._iter_single_channel(generator, delta)
            return

        if self.subset_indices is None:
            raise ValueError(
                "subset_indices must be provided when shift_mode='subset'."
            )

        subset_tensor = torch.tensor(
            self.subset_indices, dtype=torch.int64, device="cpu"
        )
        num_high = self.data_config.num_test_high_per_channel
        num_low = self.data_config.num_test_low_per_channel
        yield from self._iter_subset(
            generator,
            subset_tensor=subset_tensor,
            delta=delta,
            label_type=LABEL_TYPE_HIGH,
            num_samples=num_high,
        )
        yield from self._iter_subset(
            generator,
            subset_tensor=subset_tensor,
            delta=-delta,
            label_type=LABEL_TYPE_LOW,
            num_samples=num_low,
        )

    def _iter_single_channel(
        self,
        generator: torch.Generator,
        delta: float,
    ) -> Iterator[DirectionalDeviationAnomalyBatch]:
        num_high = self.data_config.num_test_high_per_channel
        num_low = self.data_config.num_test_low_per_channel
        for channel in self.channels_to_evaluate:
            yield from self._iter_single_channel_direction(
                generator=generator,
                channel=channel,
                label_type=LABEL_TYPE_HIGH,
                delta=delta,
                num_samples=num_high,
            )
            yield from self._iter_single_channel_direction(
                generator=generator,
                channel=channel,
                label_type=LABEL_TYPE_LOW,
                delta=-delta,
                num_samples=num_low,
            )

    def _iter_single_channel_direction(
        self,
        generator: torch.Generator,
        channel: int,
        label_type: int,
        delta: float,
        num_samples: int,
    ) -> Iterator[DirectionalDeviationAnomalyBatch]:
        for batch_count in _iterate_batch_counts(num_samples, self.batch_size):
            batch = _sample_normal_split(batch_count, self.data_config, generator)
            batch[:, channel] = batch[:, channel] + delta
            yield DirectionalDeviationAnomalyBatch(
                x=batch.to(self.device),
                labels_type=torch.full(
                    (batch_count,),
                    label_type,
                    dtype=torch.int64,
                    device=self.device,
                ),
                labels_channel=torch.full(
                    (batch_count,),
                    channel,
                    dtype=torch.int64,
                    device=self.device,
                ),
                labels_mask=None,
            )

    def _iter_subset(
        self,
        generator: torch.Generator,
        subset_tensor: Tensor,
        delta: float,
        label_type: int,
        num_samples: int,
    ) -> Iterator[DirectionalDeviationAnomalyBatch]:
        for batch_count in _iterate_batch_counts(num_samples, self.batch_size):
            batch = _sample_normal_split(batch_count, self.data_config, generator)
            batch[:, subset_tensor] = batch[:, subset_tensor] + delta
            labels_mask = (
                subset_tensor.unsqueeze(0).repeat(batch_count, 1).to(self.device)
            )
            yield DirectionalDeviationAnomalyBatch(
                x=batch.to(self.device),
                labels_type=torch.full(
                    (batch_count,),
                    label_type,
                    dtype=torch.int64,
                    device=self.device,
                ),
                labels_channel=torch.full(
                    (batch_count,),
                    -1,
                    dtype=torch.int64,
                    device=self.device,
                ),
                labels_mask=labels_mask,
            )


def _iterate_batch_counts(total_count: int, batch_size: int) -> Iterator[int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; received {batch_size}.")
    remaining = total_count
    while remaining > 0:
        current = min(batch_size, remaining)
        yield current
        remaining -= current


def generate_anomaly_test_data(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    device: str,
    batch_size: int = 4096,
    anomaly_directions: Optional[Tensor] = None,
    direction_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    set_random_seed(data_config.seed)

    if anomaly_config.shift_mode == "feature_direction":
        direction_seed = data_config.seed + 6
        if anomaly_config.base_samples == "test_normal":
            sample_seed = data_config.seed + 2
        elif anomaly_config.base_samples == "val_normal":
            sample_seed = data_config.seed + 1
        else:
            raise ValueError(
                "base_samples must be one of 'test_normal' or 'val_normal'; "
                f"received '{anomaly_config.base_samples}'."
            )

        directions: Tensor
        resolved_metadata: Dict[str, Any]
        if anomaly_directions is None:
            if anomaly_config.direction_source != "fixed_random":
                raise ValueError(
                    "anomaly_directions must be provided when shift_mode='feature_direction' "
                    "unless direction_source='fixed_random'."
                )
            directions, resolved_metadata = build_fixed_random_directions(
                data_config=data_config,
                anomaly_config=anomaly_config,
                direction_seed=direction_seed,
            )
        else:
            directions = anomaly_directions.detach().to(
                device="cpu", dtype=torch.float32
            )
            resolved_metadata = (
                {} if direction_metadata is None else dict(direction_metadata)
            )

        _validate_feature_directions(
            directions=directions,
            data_config=data_config,
            anomaly_config=anomaly_config,
        )
        direction_input_deltas_tensor: Optional[Tensor] = None
        if anomaly_config.direction_delta_mode == "fixed_pi_shift":
            raw_deltas = resolved_metadata.get("direction_input_deltas")
            if raw_deltas is None:
                raise ValueError(
                    "direction_input_deltas must be provided when direction_delta_mode='fixed_pi_shift'."
                )
            if not isinstance(raw_deltas, list):
                raise ValueError(
                    "direction_input_deltas must be a list when direction_delta_mode='fixed_pi_shift'."
                )
            deltas_list = [float(value) for value in raw_deltas]
            if len(deltas_list) != int(directions.shape[0]):
                raise ValueError(
                    "direction_input_deltas must have length num_directions; received "
                    f"{len(deltas_list)} vs {directions.shape[0]}."
                )
            if any(not (delta > 0 and delta < float("inf")) for delta in deltas_list):
                raise ValueError(
                    "direction_input_deltas must contain only finite positive values."
                )
            direction_input_deltas_tensor = torch.tensor(
                deltas_list, device="cpu", dtype=torch.float32
            )
        anomaly_stream = DirectionalDeviationFeatureDirectionAnomalyStream(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=device,
            batch_size=batch_size,
            directions=directions,
            direction_input_deltas=direction_input_deltas_tensor,
            sample_seed=sample_seed,
        )
        manifest = _build_feature_direction_anomaly_manifest(
            data_config=data_config,
            anomaly_config=anomaly_config,
            direction_metadata=resolved_metadata,
            direction_seed=direction_seed,
            sample_seed=sample_seed,
        )

        if data_config.anomaly_eval_mode == "stream":
            return {
                "anomaly_stream": anomaly_stream,
                "anomaly_manifest": manifest,
                "anomaly_directions": directions,
            }

        materialized = _materialize_anomaly_stream(anomaly_stream)
        return {
            **materialized,
            "anomaly_manifest": manifest,
            "anomaly_stream": None,
            "anomaly_directions": directions,
        }

    channels_to_evaluate = _resolve_channels_to_evaluate(
        data_config.num_channels, anomaly_config.channels_to_evaluate
    )
    subset_seed = data_config.seed + 3
    sample_seed = data_config.seed + 4
    subset_indices = (
        _select_subset_indices(data_config, anomaly_config, subset_seed)
        if anomaly_config.shift_mode == "subset"
        else None
    )
    anomaly_stream = DirectionalDeviationAnomalyStream(
        data_config=data_config,
        anomaly_config=anomaly_config,
        device=device,
        batch_size=batch_size,
        channels_to_evaluate=channels_to_evaluate,
        subset_indices=subset_indices,
        sample_seed=sample_seed,
    )
    manifest = _build_anomaly_manifest(
        data_config=data_config,
        anomaly_config=anomaly_config,
        channels_to_evaluate=channels_to_evaluate,
        subset_indices=subset_indices,
        subset_seed=subset_seed,
        sample_seed=sample_seed,
    )

    if data_config.anomaly_eval_mode == "stream":
        return {"anomaly_stream": anomaly_stream, "anomaly_manifest": manifest}

    materialized = _materialize_anomaly_stream(anomaly_stream)
    return {**materialized, "anomaly_manifest": manifest, "anomaly_stream": None}


@dataclass(frozen=True)
class DirectionalDeviationFeatureDirectionAnomalyStream:
    data_config: DirectionalDeviationDataConfig
    anomaly_config: DirectionalDeviationAnomalyConfig
    device: str
    batch_size: int
    directions: Tensor
    direction_input_deltas: Optional[Tensor]
    sample_seed: int

    def __iter__(self) -> Iterator[DirectionalDeviationAnomalyBatch]:
        if self.anomaly_config.shift_mode != "feature_direction":
            raise ValueError(
                "DirectionalDeviationFeatureDirectionAnomalyStream requires shift_mode='feature_direction'."
            )
        _validate_feature_directions(
            directions=self.directions,
            data_config=self.data_config,
            anomaly_config=self.anomaly_config,
        )
        generator = _create_cpu_generator(self.sample_seed)

        num_high = (
            int(self.anomaly_config.num_test_high_per_direction)
            if self.anomaly_config.num_test_high_per_direction is not None
            else int(self.data_config.num_test_high_per_channel)
        )
        num_low = (
            int(self.anomaly_config.num_test_low_per_direction)
            if self.anomaly_config.num_test_low_per_direction is not None
            else int(self.data_config.num_test_low_per_channel)
        )
        directions_cpu = self.directions.to(device="cpu", dtype=torch.float32)
        input_deltas = (
            self.direction_input_deltas.to(device="cpu", dtype=torch.float32)
            if self.direction_input_deltas is not None
            else None
        )
        if input_deltas is not None:
            if input_deltas.ndim != 1:
                raise ValueError(
                    f"Expected direction_input_deltas.ndim == 1, got {input_deltas.ndim}."
                )
            if int(input_deltas.shape[0]) != int(directions_cpu.shape[0]):
                raise ValueError(
                    "Expected direction_input_deltas.shape[0] == directions.shape[0], got "
                    f"{input_deltas.shape[0]} vs {directions_cpu.shape[0]}."
                )
            if not torch.all(torch.isfinite(input_deltas)).item():
                raise ValueError(
                    "direction_input_deltas must contain only finite values."
                )
            if torch.any(input_deltas <= 0).item():
                raise ValueError("direction_input_deltas must be positive.")

        for direction_index in range(int(directions_cpu.shape[0])):
            direction_vector = directions_cpu[direction_index]
            base_delta = (
                float(input_deltas[direction_index])
                if input_deltas is not None
                else float(self.anomaly_config.delta)
            )
            yield from self._iter_direction(
                generator=generator,
                direction_index=int(direction_index),
                direction_vector=direction_vector,
                label_type=LABEL_TYPE_HIGH,
                delta=base_delta,
                num_samples=num_high,
            )
            yield from self._iter_direction(
                generator=generator,
                direction_index=int(direction_index),
                direction_vector=direction_vector,
                label_type=LABEL_TYPE_LOW,
                delta=-base_delta,
                num_samples=num_low,
            )

    def _iter_direction(
        self,
        generator: torch.Generator,
        direction_index: int,
        direction_vector: Tensor,
        label_type: int,
        delta: float,
        num_samples: int,
    ) -> Iterator[DirectionalDeviationAnomalyBatch]:
        for batch_count in _iterate_batch_counts(num_samples, self.batch_size):
            batch = _sample_normal_split(batch_count, self.data_config, generator)
            shifted = batch + delta * direction_vector.unsqueeze(0)
            yield DirectionalDeviationAnomalyBatch(
                x=shifted.to(self.device),
                labels_type=torch.full(
                    (batch_count,),
                    label_type,
                    dtype=torch.int64,
                    device=self.device,
                ),
                labels_channel=torch.full(
                    (batch_count,),
                    direction_index,
                    dtype=torch.int64,
                    device=self.device,
                ),
                labels_mask=None,
            )


def _materialize_anomaly_stream(
    anomaly_stream: Iterator[DirectionalDeviationAnomalyBatch],
) -> Dict[str, Tensor]:
    x_batches: List[Tensor] = []
    labels_type_batches: List[Tensor] = []
    labels_channel_batches: List[Tensor] = []
    labels_mask_batches: List[Tensor] = []
    has_labels_mask = False

    for batch in anomaly_stream:
        x_batches.append(batch.x)
        labels_type_batches.append(batch.labels_type)
        labels_channel_batches.append(batch.labels_channel)
        if batch.labels_mask is not None:
            has_labels_mask = True
            labels_mask_batches.append(batch.labels_mask)

    labels_mask = torch.cat(labels_mask_batches, dim=0) if has_labels_mask else None
    output: Dict[str, Tensor] = {
        "x_test_anomaly": torch.cat(x_batches, dim=0),
        "labels_type": torch.cat(labels_type_batches, dim=0),
        "labels_channel": torch.cat(labels_channel_batches, dim=0),
    }
    if labels_mask is not None:
        output["labels_mask"] = labels_mask
    return output


def export_data_summary(
    run_dir: str,
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    normal_data: Dict[str, Tensor],
    anomaly_data: Dict[str, Any],
    sample_preview_count: int = 1024,
    anomaly_preview_count: int = 256,
) -> None:
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    generator_config = asdict(data_config)
    if data_config.normal_mode in {"iid_non_gaussian", "non_gaussian_heteroskedastic"}:
        derived_config: Dict[str, Any] = {}
        if data_config.non_gaussian_dist == "laplace":
            derived_config["laplace_scale"] = 1.0 / math.sqrt(2.0)
        elif data_config.non_gaussian_dist == "logistic":
            derived_config["logistic_scale"] = math.sqrt(3.0) / math.pi
        elif data_config.non_gaussian_dist == "student_t":
            if data_config.student_t_df is not None:
                df = float(data_config.student_t_df)
                derived_config["student_t_unit_variance_scale"] = math.sqrt(
                    (df - 2.0) / df
                )
        generator_config["non_gaussian_derived"] = derived_config
    generator_config["seed_plan"] = {
        "train_seed": _normalize_torch_seed(data_config.seed + 0),
        "val_seed": _normalize_torch_seed(data_config.seed + 1),
        "test_normal_seed": _normalize_torch_seed(data_config.seed + 2),
        "subset_seed": _normalize_torch_seed(data_config.seed + 3),
        "anomaly_seed": _normalize_torch_seed(
            (
                data_config.seed + 2
                if (
                    anomaly_config.shift_mode == "feature_direction"
                    and anomaly_config.base_samples == "test_normal"
                )
                else (
                    data_config.seed + 1
                    if (
                        anomaly_config.shift_mode == "feature_direction"
                        and anomaly_config.base_samples == "val_normal"
                    )
                    else data_config.seed + 4
                )
            )
        ),
        "direction_seed": _normalize_torch_seed(data_config.seed + 6),
    }
    generator_config_path = os.path.join(data_dir, "generator_config.json")
    with open(generator_config_path, "w") as f:
        json.dump(generator_config, f, indent=2)

    anomaly_manifest_path = os.path.join(data_dir, "anomaly_manifest.json")
    with open(anomaly_manifest_path, "w") as f:
        json.dump(anomaly_data["anomaly_manifest"], f, indent=2)

    anomaly_directions = anomaly_data.get("anomaly_directions")
    if isinstance(anomaly_directions, Tensor):
        torch.save(
            anomaly_directions.detach().to(device="cpu"),
            os.path.join(data_dir, "anomaly_directions.pt"),
        )

    preview = _collect_sampled_preview(
        normal_data=normal_data,
        anomaly_data=anomaly_data,
        sample_preview_count=sample_preview_count,
        anomaly_preview_count=anomaly_preview_count,
    )
    sampled_data_path = os.path.join(data_dir, "sampled_data.pt")
    torch.save(preview, sampled_data_path)


def _collect_sampled_preview(
    normal_data: Dict[str, Tensor],
    anomaly_data: Dict[str, Any],
    sample_preview_count: int,
    anomaly_preview_count: int,
) -> Dict[str, Tensor]:
    x_train = normal_data["x_train"]
    x_val = normal_data["x_val"]
    x_test_normal = normal_data["x_test_normal"]

    preview: Dict[str, Tensor] = {
        "x_train": x_train[: min(sample_preview_count, x_train.shape[0])].cpu(),
        "x_val": x_val[: min(sample_preview_count, x_val.shape[0])].cpu(),
        "x_test_normal": x_test_normal[
            : min(sample_preview_count, x_test_normal.shape[0])
        ].cpu(),
    }

    anomaly_preview = _collect_anomaly_preview(anomaly_data, anomaly_preview_count)
    preview.update(anomaly_preview)
    return preview


def _collect_anomaly_preview(
    anomaly_data: Dict[str, Any],
    anomaly_preview_count: int,
) -> Dict[str, Tensor]:
    if "x_test_anomaly" in anomaly_data:
        x_anomaly = anomaly_data["x_test_anomaly"][:anomaly_preview_count].cpu()
        labels_type = anomaly_data["labels_type"][:anomaly_preview_count].cpu()
        labels_channel = anomaly_data["labels_channel"][:anomaly_preview_count].cpu()
        output = {
            "x_test_anomaly": x_anomaly,
            "labels_type": labels_type,
            "labels_channel": labels_channel,
        }
        if "labels_mask" in anomaly_data:
            output["labels_mask"] = anomaly_data["labels_mask"][
                :anomaly_preview_count
            ].cpu()
        return output

    if "anomaly_stream" not in anomaly_data:
        raise ValueError(
            "Expected anomaly_data to contain either x_test_anomaly or anomaly_stream."
        )

    anomaly_stream = anomaly_data["anomaly_stream"]
    if not hasattr(anomaly_stream, "data_config"):
        raise ValueError("anomaly_stream is missing required attribute data_config.")
    num_channels = int(getattr(anomaly_stream, "data_config").num_channels)
    subset_indices = getattr(anomaly_stream, "subset_indices", None)
    subset_size = len(subset_indices) if subset_indices else 0
    x_batches: List[Tensor] = []
    labels_type_batches: List[Tensor] = []
    labels_channel_batches: List[Tensor] = []
    labels_mask_batches: List[Tensor] = []
    remaining = anomaly_preview_count
    has_labels_mask = False

    for batch in anomaly_stream:
        if remaining <= 0:
            break
        take = min(remaining, batch.x.shape[0])
        x_batches.append(batch.x[:take].cpu())
        labels_type_batches.append(batch.labels_type[:take].cpu())
        labels_channel_batches.append(batch.labels_channel[:take].cpu())
        if batch.labels_mask is not None:
            has_labels_mask = True
            labels_mask_batches.append(batch.labels_mask[:take].cpu())
        remaining -= take

    output = {
        "x_test_anomaly": (
            torch.cat(x_batches, dim=0)
            if x_batches
            else torch.empty((0, num_channels), dtype=torch.float32)
        ),
        "labels_type": (
            torch.cat(labels_type_batches, dim=0)
            if labels_type_batches
            else torch.empty((0,), dtype=torch.int64)
        ),
        "labels_channel": (
            torch.cat(labels_channel_batches, dim=0)
            if labels_channel_batches
            else torch.empty((0,), dtype=torch.int64)
        ),
    }
    if has_labels_mask:
        output["labels_mask"] = (
            torch.cat(labels_mask_batches, dim=0)
            if labels_mask_batches
            else torch.empty((0, subset_size), dtype=torch.int64)
        )
    return output
