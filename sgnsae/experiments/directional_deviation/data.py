import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch import Tensor

from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationAnomalyConfig,
    DirectionalDeviationDataConfig,
)


LABEL_TYPE_NORMAL = 0
LABEL_TYPE_HIGH = 1
LABEL_TYPE_LOW = 2

_TORCH_SEED_MODULUS = 2**63
_NUMPY_SEED_MODULUS = 2**32


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


def _sample_normal_split(
    num_samples: int,
    config: DirectionalDeviationDataConfig,
    generator: torch.Generator,
) -> Tensor:
    samples = torch.randn(
        (num_samples, config.num_channels),
        device="cpu",
        dtype=torch.float32,
        generator=generator,
    )
    if config.normal_mode == "heteroskedastic_normal":
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
) -> Dict[str, Any]:
    set_random_seed(data_config.seed)

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


def _materialize_anomaly_stream(
    anomaly_stream: DirectionalDeviationAnomalyStream,
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
    generator_config["seed_plan"] = {
        "train_seed": _normalize_torch_seed(data_config.seed + 0),
        "val_seed": _normalize_torch_seed(data_config.seed + 1),
        "test_normal_seed": _normalize_torch_seed(data_config.seed + 2),
        "subset_seed": _normalize_torch_seed(data_config.seed + 3),
        "anomaly_seed": _normalize_torch_seed(data_config.seed + 4),
    }
    generator_config_path = os.path.join(data_dir, "generator_config.json")
    with open(generator_config_path, "w") as f:
        json.dump(generator_config, f, indent=2)

    anomaly_manifest_path = os.path.join(data_dir, "anomaly_manifest.json")
    with open(anomaly_manifest_path, "w") as f:
        json.dump(anomaly_data["anomaly_manifest"], f, indent=2)

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

    anomaly_stream: DirectionalDeviationAnomalyStream = anomaly_data["anomaly_stream"]
    num_channels = anomaly_stream.data_config.num_channels
    subset_size = (
        len(anomaly_stream.subset_indices) if anomaly_stream.subset_indices else 0
    )
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
