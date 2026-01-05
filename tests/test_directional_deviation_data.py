import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationAnomalyConfig,
    DirectionalDeviationDataConfig,
)
from sgnsae.experiments.directional_deviation.data import (
    LABEL_TYPE_HIGH,
    LABEL_TYPE_LOW,
    DirectionalDeviationAnomalyStream,
    build_feature_directions_from_decoder,
    export_data_summary,
    generate_anomaly_test_data,
    generate_normal_data,
)
from sgnsae.experiments.directional_deviation import data as data_module


def _iterate_batch_counts(total_count: int, batch_size: int) -> Iterator[int]:
    remaining = total_count
    while remaining > 0:
        current = min(batch_size, remaining)
        yield current
        remaining -= current


def _sample_expected_normal_split(
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

    output: Dict[str, Tensor] = {
        "x_test_anomaly": torch.cat(x_batches, dim=0),
        "labels_type": torch.cat(labels_type_batches, dim=0),
        "labels_channel": torch.cat(labels_channel_batches, dim=0),
    }
    if has_labels_mask:
        output["labels_mask"] = torch.cat(labels_mask_batches, dim=0)
    return output


def _build_expected_single_channel_anomalies(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    channels_to_evaluate: List[int],
    sample_seed: int,
    batch_size: int,
    device: str,
) -> Dict[str, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sample_seed)
    delta = float(anomaly_config.delta)

    x_batches: List[Tensor] = []
    labels_type_batches: List[Tensor] = []
    labels_channel_batches: List[Tensor] = []
    for channel in channels_to_evaluate:
        for label_type, shift, count in (
            (LABEL_TYPE_HIGH, delta, data_config.num_test_high_per_channel),
            (LABEL_TYPE_LOW, -delta, data_config.num_test_low_per_channel),
        ):
            for batch_count in _iterate_batch_counts(count, batch_size):
                batch = _sample_expected_normal_split(
                    batch_count, data_config, generator
                )
                batch[:, channel] = batch[:, channel] + shift
                x_batches.append(batch.to(device))
                labels_type_batches.append(
                    torch.full(
                        (batch_count,),
                        label_type,
                        dtype=torch.int64,
                        device=device,
                    )
                )
                labels_channel_batches.append(
                    torch.full(
                        (batch_count,),
                        channel,
                        dtype=torch.int64,
                        device=device,
                    )
                )

    return {
        "x_test_anomaly": torch.cat(x_batches, dim=0),
        "labels_type": torch.cat(labels_type_batches, dim=0),
        "labels_channel": torch.cat(labels_channel_batches, dim=0),
    }


def _build_expected_subset_anomalies(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    subset_indices: List[int],
    sample_seed: int,
    batch_size: int,
    device: str,
) -> Dict[str, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sample_seed)
    delta = float(anomaly_config.delta)
    subset_tensor = torch.tensor(subset_indices, dtype=torch.int64, device="cpu")

    x_batches: List[Tensor] = []
    labels_type_batches: List[Tensor] = []
    labels_channel_batches: List[Tensor] = []
    labels_mask_batches: List[Tensor] = []
    for label_type, shift, count in (
        (LABEL_TYPE_HIGH, delta, data_config.num_test_high_per_channel),
        (LABEL_TYPE_LOW, -delta, data_config.num_test_low_per_channel),
    ):
        for batch_count in _iterate_batch_counts(count, batch_size):
            batch = _sample_expected_normal_split(batch_count, data_config, generator)
            batch[:, subset_tensor] = batch[:, subset_tensor] + shift
            x_batches.append(batch.to(device))
            labels_type_batches.append(
                torch.full((batch_count,), label_type, dtype=torch.int64, device=device)
            )
            labels_channel_batches.append(
                torch.full((batch_count,), -1, dtype=torch.int64, device=device)
            )
            labels_mask_batches.append(
                subset_tensor.unsqueeze(0).repeat(batch_count, 1).to(device)
            )

    return {
        "x_test_anomaly": torch.cat(x_batches, dim=0),
        "labels_type": torch.cat(labels_type_batches, dim=0),
        "labels_channel": torch.cat(labels_channel_batches, dim=0),
        "labels_mask": torch.cat(labels_mask_batches, dim=0),
    }


def _build_expected_feature_direction_anomalies(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    directions: Tensor,
    direction_input_deltas: Optional[Tensor],
    sample_seed: int,
    batch_size: int,
    device: str,
) -> Dict[str, Tensor]:
    generator = data_module._create_cpu_generator(sample_seed)
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
    base_delta = float(anomaly_config.delta)
    input_deltas = (
        direction_input_deltas.to(device="cpu", dtype=torch.float32)
        if direction_input_deltas is not None
        else None
    )

    x_batches: List[Tensor] = []
    labels_type_batches: List[Tensor] = []
    labels_channel_batches: List[Tensor] = []

    for direction_index in range(int(directions.shape[0])):
        direction_vector = directions[direction_index].to(device="cpu")
        delta = (
            float(input_deltas[direction_index])
            if input_deltas is not None
            else base_delta
        )
        for label_type, shift, count in (
            (LABEL_TYPE_HIGH, delta, num_high),
            (LABEL_TYPE_LOW, -delta, num_low),
        ):
            for batch_count in _iterate_batch_counts(count, batch_size):
                base = data_module._sample_normal_split(
                    batch_count, data_config, generator
                )
                shifted = base + shift * direction_vector.unsqueeze(0)
                x_batches.append(shifted.to(device))
                labels_type_batches.append(
                    torch.full(
                        (batch_count,), label_type, dtype=torch.int64, device=device
                    )
                )
                labels_channel_batches.append(
                    torch.full(
                        (batch_count,),
                        direction_index,
                        dtype=torch.int64,
                        device=device,
                    )
                )

    return {
        "x_test_anomaly": torch.cat(x_batches, dim=0),
        "labels_type": torch.cat(labels_type_batches, dim=0),
        "labels_channel": torch.cat(labels_channel_batches, dim=0),
    }


class TestDirectionalDeviationDataGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"

    def test_calculate_latent_pi_std_rejects_invalid_inputs(self) -> None:
        # given
        from sgnsae.models import SAE

        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        base_samples = torch.zeros((3, 2), dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._calculate_latent_pi_std(
                model=model,
                base_samples=base_samples,
                batch_size=0,
            )
        with self.assertRaises(ValueError):
            _ = data_module._calculate_latent_pi_std(
                model=model,
                base_samples=torch.zeros((1, 1, 2), dtype=torch.float32),
                batch_size=1,
            )
        with self.assertRaises(ValueError):
            _ = data_module._calculate_latent_pi_std(
                model=model,
                base_samples=torch.zeros((0, 2), dtype=torch.float32),
                batch_size=1,
            )

    def test_build_feature_directions_from_decoder_top_pi_std_requires_base_samples(
        self,
    ) -> None:
        # given
        from sgnsae.models import SAE

        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=1,
            direction_selection="top_pi_std",
            normalize_directions=True,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=model,
                anomaly_config=anomaly_config,
                direction_seed=123,
            )

    def test_build_feature_directions_from_decoder_requires_direction_delta_mode(
        self,
    ) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self, decoder: Tensor) -> None:
                super().__init__()
                self.W_dec = decoder
                self.n_instances = 1

        decoder = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        model = DummyModel(decoder)
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=1,
            direction_selection="explicit",
            direction_latent_indices=[0],
            normalize_directions=True,
        )
        anomaly_config.direction_delta_mode = None  # type: ignore[assignment]

        # when & then
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=model,
                anomaly_config=anomaly_config,
                direction_seed=123,
            )

    def test_build_fixed_random_directions_requires_direction_delta_mode(self) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=1,
            normalize_directions=True,
        )
        anomaly_config.direction_delta_mode = None  # type: ignore[assignment]

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module.build_fixed_random_directions(
                data_config=data_config,
                anomaly_config=anomaly_config,
                direction_seed=0,
            )

    def test_generate_anomaly_test_data_feature_direction_rejects_invalid_base_samples(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=1,
            normalize_directions=True,
        )
        anomaly_config.base_samples = "invalid"  # type: ignore[assignment]

        # when & then
        with self.assertRaises(ValueError):
            _ = generate_anomaly_test_data(
                data_config=data_config,
                anomaly_config=anomaly_config,
                device=self.device,
                batch_size=2,
            )

    def test_generate_anomaly_test_data_fixed_pi_shift_rejects_missing_input_deltas(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=2,
            direction_selection="explicit",
            direction_latent_indices=[0, 1],
            normalize_directions=True,
            direction_delta_mode="fixed_pi_shift",
        )
        directions = torch.eye(2, dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = generate_anomaly_test_data(
                data_config=data_config,
                anomaly_config=anomaly_config,
                device=self.device,
                batch_size=2,
                anomaly_directions=directions,
                direction_metadata={},
            )
        with self.assertRaises(ValueError):
            _ = generate_anomaly_test_data(
                data_config=data_config,
                anomaly_config=anomaly_config,
                device=self.device,
                batch_size=2,
                anomaly_directions=directions,
                direction_metadata={"direction_input_deltas": "not-a-list"},
            )
        with self.assertRaises(ValueError):
            _ = generate_anomaly_test_data(
                data_config=data_config,
                anomaly_config=anomaly_config,
                device=self.device,
                batch_size=2,
                anomaly_directions=directions,
                direction_metadata={"direction_input_deltas": [1.0]},
            )
        with self.assertRaises(ValueError):
            _ = generate_anomaly_test_data(
                data_config=data_config,
                anomaly_config=anomaly_config,
                device=self.device,
                batch_size=2,
                anomaly_directions=directions,
                direction_metadata={"direction_input_deltas": [1.0, float("inf")]},
            )

    def test_generate_anomaly_test_data_feature_direction_supports_val_normal_base_samples(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=10,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=1,
            base_samples="val_normal",
            normalize_directions=True,
        )

        # when
        anomaly_data = generate_anomaly_test_data(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=2,
        )

        # then
        manifest = anomaly_data["anomaly_manifest"]
        self.assertEqual(manifest["base_samples"], "val_normal")
        self.assertEqual(manifest["seed_plan"]["sample_seed"], data_config.seed + 1)

    def test_feature_direction_anomaly_stream_rejects_invalid_input_deltas(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=2,
            normalize_directions=True,
        )
        directions = torch.eye(2, dtype=torch.float32)

        # when & then
        invalid_ndim = data_module.DirectionalDeviationFeatureDirectionAnomalyStream(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=2,
            directions=directions,
            direction_input_deltas=torch.ones((2, 1), dtype=torch.float32),
            sample_seed=0,
        )
        with self.assertRaises(ValueError):
            _ = list(iter(invalid_ndim))

        invalid_length = data_module.DirectionalDeviationFeatureDirectionAnomalyStream(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=2,
            directions=directions,
            direction_input_deltas=torch.ones((3,), dtype=torch.float32),
            sample_seed=0,
        )
        with self.assertRaises(ValueError):
            _ = list(iter(invalid_length))

        invalid_finite = data_module.DirectionalDeviationFeatureDirectionAnomalyStream(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=2,
            directions=directions,
            direction_input_deltas=torch.tensor(
                [1.0, float("inf")], dtype=torch.float32
            ),
            sample_seed=0,
        )
        with self.assertRaises(ValueError):
            _ = list(iter(invalid_finite))

        invalid_positive = (
            data_module.DirectionalDeviationFeatureDirectionAnomalyStream(
                data_config=data_config,
                anomaly_config=anomaly_config,
                device=self.device,
                batch_size=2,
                directions=directions,
                direction_input_deltas=torch.tensor([1.0, 0.0], dtype=torch.float32),
                sample_seed=0,
            )
        )
        with self.assertRaises(ValueError):
            _ = list(iter(invalid_positive))

    def test_generate_normal_data_shapes_and_reproducible_with_seed(self) -> None:
        # given
        config = DirectionalDeviationDataConfig(
            seed=7,
            num_train=13,
            num_val=11,
            num_test_normal=5,
            num_test_high_per_channel=2,
            num_test_low_per_channel=3,
            num_channels=4,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )

        # when
        first = generate_normal_data(config, self.device)
        second = generate_normal_data(config, self.device)

        # then
        self.assertEqual(first["x_train"].shape, (13, 4))
        self.assertEqual(first["x_val"].shape, (11, 4))
        self.assertEqual(first["x_test_normal"].shape, (5, 4))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))

    def test_generate_normal_data_heteroskedastic_and_noise_matches_expected(
        self,
    ) -> None:
        # given
        config = DirectionalDeviationDataConfig(
            seed=123,
            num_train=4,
            num_val=3,
            num_test_normal=2,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=3,
            noise_std=0.25,
            normal_mode="heteroskedastic_normal",
            channel_stds=[0.5, 1.0, 2.0],
            anomaly_eval_mode="stream",
        )

        # when
        normal_data = generate_normal_data(config, self.device)

        # then
        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(config.seed + 0)
        expected_train = _sample_expected_normal_split(
            config.num_train, config, train_generator
        ).to(self.device)

        val_generator = torch.Generator(device="cpu")
        val_generator.manual_seed(config.seed + 1)
        expected_val = _sample_expected_normal_split(
            config.num_val, config, val_generator
        ).to(self.device)

        test_generator = torch.Generator(device="cpu")
        test_generator.manual_seed(config.seed + 2)
        expected_test_normal = _sample_expected_normal_split(
            config.num_test_normal, config, test_generator
        ).to(self.device)

        self.assertTrue(torch.equal(normal_data["x_train"], expected_train))
        self.assertTrue(torch.equal(normal_data["x_val"], expected_val))
        self.assertTrue(torch.equal(normal_data["x_test_normal"], expected_test_normal))

    def test_generate_normal_data_non_gaussian_modes_are_deterministic_with_seed(
        self,
    ) -> None:
        # given
        base_kwargs: Dict[str, Any] = {
            "seed": 9,
            "num_train": 17,
            "num_val": 11,
            "num_test_normal": 5,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 4,
            "noise_std": 0.0,
            "anomaly_eval_mode": "stream",
        }

        # when & then
        for normal_mode, extra_kwargs in (
            ("iid_non_gaussian", {"non_gaussian_dist": "laplace"}),
            ("iid_non_gaussian", {"non_gaussian_dist": "logistic"}),
            (
                "iid_non_gaussian",
                {"non_gaussian_dist": "student_t", "student_t_df": 7.0},
            ),
            (
                "non_gaussian_heteroskedastic",
                {
                    "non_gaussian_dist": "laplace",
                    "channel_stds": [0.5, 1.0, 2.0, 3.0],
                },
            ),
        ):
            with self.subTest(normal_mode=normal_mode, extra_kwargs=extra_kwargs):
                config = DirectionalDeviationDataConfig(
                    **base_kwargs,
                    normal_mode=normal_mode,  # type: ignore[arg-type]
                    **extra_kwargs,
                )

                first = generate_normal_data(config, self.device)
                second = generate_normal_data(config, self.device)

                for key in first:
                    self.assertTrue(torch.equal(first[key], second[key]))

    def test_generate_normal_data_non_gaussian_modes_have_expected_variance(
        self,
    ) -> None:
        # given
        sample_count = 50_000
        base_kwargs: Dict[str, Any] = {
            "seed": 21,
            "num_train": sample_count,
            "num_val": 1,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 3,
            "noise_std": 0.0,
            "anomaly_eval_mode": "stream",
        }

        # when & then
        for normal_mode, extra_kwargs, expected_variances in (
            ("iid_non_gaussian", {"non_gaussian_dist": "laplace"}, [1.0, 1.0, 1.0]),
            ("iid_non_gaussian", {"non_gaussian_dist": "logistic"}, [1.0, 1.0, 1.0]),
            (
                "iid_non_gaussian",
                {"non_gaussian_dist": "student_t", "student_t_df": 10.0},
                [1.0, 1.0, 1.0],
            ),
            (
                "non_gaussian_heteroskedastic",
                {
                    "non_gaussian_dist": "laplace",
                    "channel_stds": [0.5, 1.5, 2.0],
                },
                [0.25, 2.25, 4.0],
            ),
        ):
            with self.subTest(normal_mode=normal_mode, extra_kwargs=extra_kwargs):
                config = DirectionalDeviationDataConfig(
                    **base_kwargs,
                    normal_mode=normal_mode,  # type: ignore[arg-type]
                    **extra_kwargs,
                )

                normal_data = generate_normal_data(config, self.device)
                x_train = normal_data["x_train"].double()
                means = x_train.mean(dim=0)
                variances = x_train.var(dim=0, unbiased=False)

                for channel_index, expected_variance in enumerate(expected_variances):
                    self.assertLess(abs(float(means[channel_index])), 0.05)
                    relative_error = (
                        abs(float(variances[channel_index] - expected_variance))
                        / expected_variance
                    )
                    self.assertLess(relative_error, 0.1)

    def test_generate_normal_data_gaussian_modes_have_expected_variance(
        self,
    ) -> None:
        # given
        sample_count = 50_000
        base_kwargs: Dict[str, Any] = {
            "seed": 22,
            "num_train": sample_count,
            "num_val": 1,
            "num_test_normal": 1,
            "num_test_high_per_channel": 1,
            "num_test_low_per_channel": 1,
            "num_channels": 3,
            "noise_std": 0.0,
            "anomaly_eval_mode": "stream",
        }

        # when & then
        for normal_mode, extra_kwargs, expected_variances in (
            ("standard_normal", {}, [1.0, 1.0, 1.0]),
            (
                "heteroskedastic_normal",
                {"channel_stds": [0.5, 1.5, 2.0]},
                [0.25, 2.25, 4.0],
            ),
        ):
            with self.subTest(normal_mode=normal_mode, extra_kwargs=extra_kwargs):
                config = DirectionalDeviationDataConfig(
                    **base_kwargs,
                    normal_mode=normal_mode,  # type: ignore[arg-type]
                    **extra_kwargs,
                )
                normal_data = generate_normal_data(config, self.device)
                x_train = normal_data["x_train"].double()
                means = x_train.mean(dim=0)
                variances = x_train.var(dim=0, unbiased=False)

                for channel_index, expected_variance in enumerate(expected_variances):
                    self.assertLess(abs(float(means[channel_index])), 0.05)
                    relative_error = (
                        abs(float(variances[channel_index] - expected_variance))
                        / expected_variance
                    )
                    self.assertLess(relative_error, 0.1)

    def test_sample_normal_split_raises_for_invalid_config_state(self) -> None:
        # given
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)

        # when & then
        config_missing_dist = DirectionalDeviationDataConfig(
            seed=1,
            num_train=1,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
            normal_mode="iid_non_gaussian",
            non_gaussian_dist="laplace",
            anomaly_eval_mode="stream",
        )
        config_missing_dist.non_gaussian_dist = None
        with self.assertRaises(ValueError):
            _ = data_module._sample_normal_split(1, config_missing_dist, generator)

        config_missing_df = DirectionalDeviationDataConfig(
            seed=1,
            num_train=1,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
            normal_mode="iid_non_gaussian",
            non_gaussian_dist="student_t",
            student_t_df=7.0,
            anomaly_eval_mode="stream",
        )
        config_missing_df.student_t_df = None
        with self.assertRaises(ValueError):
            _ = data_module._sample_normal_split(1, config_missing_df, generator)

        config_unsupported_dist = DirectionalDeviationDataConfig(
            seed=1,
            num_train=1,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
            normal_mode="iid_non_gaussian",
            non_gaussian_dist="laplace",
            anomaly_eval_mode="stream",
        )
        config_unsupported_dist.non_gaussian_dist = "unsupported"  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = data_module._sample_normal_split(1, config_unsupported_dist, generator)

        config_unsupported_mode = DirectionalDeviationDataConfig(
            seed=1,
            num_train=1,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        config_unsupported_mode.normal_mode = "unsupported"  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = data_module._sample_normal_split(1, config_unsupported_mode, generator)

        config_missing_channel_stds = DirectionalDeviationDataConfig(
            seed=1,
            num_train=1,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=2,
            normal_mode="heteroskedastic_normal",
            channel_stds=[1.0, 2.0],
            anomaly_eval_mode="stream",
        )
        config_missing_channel_stds.channel_stds = None
        with self.assertRaises(ValueError):
            _ = data_module._sample_normal_split(
                1, config_missing_channel_stds, generator
            )

    def test_feature_direction_validation_helpers_raise_for_invalid_inputs(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=1,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=3,
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=1,
        )
        directions = torch.zeros((1, 3), dtype=torch.float32)

        # when & then
        anomaly_config.num_directions = 0  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            data_module._validate_feature_directions(
                directions, data_config, anomaly_config
            )

        anomaly_config.num_directions = None  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            data_module._validate_feature_directions(
                directions, data_config, anomaly_config
            )

        anomaly_config.num_directions = 1  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            data_module._validate_feature_directions(
                torch.zeros((3,), dtype=torch.float32), data_config, anomaly_config
            )
        with self.assertRaises(ValueError):
            data_module._validate_feature_directions(
                torch.zeros((2, 3), dtype=torch.float32), data_config, anomaly_config
            )
        with self.assertRaises(ValueError):
            data_module._validate_feature_directions(
                torch.zeros((1, 4), dtype=torch.float32), data_config, anomaly_config
            )

    def test_feature_direction_helpers_raise_for_invalid_models_and_configs(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=1,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=3,
            anomaly_eval_mode="stream",
        )
        base_anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=1,
            direction_selection="top_purity",
        )
        directions = torch.zeros((1, 3), dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=torch.nn.Identity(),
                anomaly_config=base_anomaly_config,
                direction_seed=1,
            )

        class DummyModel(torch.nn.Module):
            def __init__(self, w_dec: Any, n_instances: int = 1) -> None:
                super().__init__()
                self.W_dec = w_dec
                self.n_instances = n_instances

        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=DummyModel(w_dec="not_a_tensor"),
                anomaly_config=base_anomaly_config,
                direction_seed=1,
            )
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=DummyModel(w_dec=torch.zeros((3, 2), dtype=torch.float32)),
                anomaly_config=base_anomaly_config,
                direction_seed=1,
            )
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=DummyModel(
                    w_dec=torch.zeros((1, 3, 2), dtype=torch.float32), n_instances=2
                ),
                anomaly_config=base_anomaly_config,
                direction_seed=1,
            )

        with self.assertRaises(ValueError):
            _ = data_module._calculate_latent_purity(torch.zeros((1, 2, 3)))

        generator = data_module._create_cpu_generator(0)
        with self.assertRaises(ValueError):
            _ = data_module._select_latent_indices_random(
                num_latents=1, num_directions=2, generator=generator
            )

        wrong_shift_mode_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=DummyModel(w_dec=torch.zeros((1, 3, 2), dtype=torch.float32)),
                anomaly_config=wrong_shift_mode_config,
                direction_seed=1,
            )

        mutated_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=1,
            direction_selection="top_purity",
        )
        mutated_config.num_directions = None  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=DummyModel(w_dec=torch.zeros((1, 3, 2), dtype=torch.float32)),
                anomaly_config=mutated_config,
                direction_seed=1,
            )

        mutated_config.num_directions = 1  # type: ignore[assignment]
        mutated_config.direction_selection = None  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=DummyModel(w_dec=torch.zeros((1, 3, 2), dtype=torch.float32)),
                anomaly_config=mutated_config,
                direction_seed=1,
            )

        explicit_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=1,
            direction_selection="explicit",
            direction_latent_indices=[0],
        )
        model = DummyModel(w_dec=torch.eye(3, 2, dtype=torch.float32).unsqueeze(0))
        directions_explicit, metadata_explicit = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=explicit_config,
            direction_seed=1,
        )
        self.assertEqual(metadata_explicit["direction_latent_indices"], [0])
        self.assertTrue(
            torch.allclose(
                directions_explicit,
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            )
        )

        explicit_config.direction_latent_indices = None  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=model,
                anomaly_config=explicit_config,
                direction_seed=1,
            )

        explicit_config.direction_selection = "unsupported"  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=model,
                anomaly_config=explicit_config,
                direction_seed=1,
            )

        out_of_range_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=1,
            direction_selection="explicit",
            direction_latent_indices=[0],
        )
        out_of_range_config.direction_latent_indices = [10]  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = build_feature_directions_from_decoder(
                model=model,
                anomaly_config=out_of_range_config,
                direction_seed=1,
            )

        with self.assertRaises(ValueError):
            _ = data_module.build_fixed_random_directions(
                data_config=data_config,
                anomaly_config=wrong_shift_mode_config,
                direction_seed=0,
            )

        invalid_fixed_random_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=1,
        )
        invalid_fixed_random_config.num_directions = None  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = data_module.build_fixed_random_directions(
                data_config=data_config,
                anomaly_config=invalid_fixed_random_config,
                direction_seed=0,
            )

        with self.assertRaises(ValueError):
            _ = data_module._build_feature_direction_anomaly_manifest(
                data_config=data_config,
                anomaly_config=wrong_shift_mode_config,
                direction_metadata={},
                direction_seed=0,
                sample_seed=0,
            )

        invalid_manifest_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=1,
        )
        invalid_manifest_config.num_directions = None  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            _ = data_module._build_feature_direction_anomaly_manifest(
                data_config=data_config,
                anomaly_config=invalid_manifest_config,
                direction_metadata={},
                direction_seed=0,
                sample_seed=0,
            )

        missing_directions_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=1,
        )
        with self.assertRaises(ValueError):
            _ = generate_anomaly_test_data(
                data_config=data_config,
                anomaly_config=missing_directions_config,
                device=self.device,
                batch_size=2,
            )

        invalid_stream = data_module.DirectionalDeviationFeatureDirectionAnomalyStream(
            data_config=data_config,
            anomaly_config=wrong_shift_mode_config,
            device=self.device,
            batch_size=1,
            directions=directions,
            direction_input_deltas=None,
            sample_seed=0,
        )
        with self.assertRaises(ValueError):
            _ = list(iter(invalid_stream))

        with self.assertRaises(ValueError):
            _ = data_module._collect_anomaly_preview(
                {"anomaly_stream": object()}, anomaly_preview_count=1
            )

    def test_generate_anomaly_test_data_single_channel_materialize_matches_expected(
        self,
    ) -> None:
        # given
        batch_size = 2
        data_config = DirectionalDeviationDataConfig(
            seed=5,
            num_train=3,
            num_val=3,
            num_test_normal=3,
            num_test_high_per_channel=3,
            num_test_low_per_channel=2,
            num_channels=6,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="materialize",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=5.0,
            shift_mode="single_channel",
            channels_to_evaluate=[1, 4],
        )

        # when
        anomaly_data = generate_anomaly_test_data(
            data_config, anomaly_config, self.device, batch_size=batch_size
        )

        # then
        expected_count = 2 * (
            data_config.num_test_high_per_channel + data_config.num_test_low_per_channel
        )
        self.assertEqual(anomaly_data["x_test_anomaly"].shape, (expected_count, 6))
        self.assertEqual(anomaly_data["labels_type"].shape, (expected_count,))
        self.assertEqual(anomaly_data["labels_channel"].shape, (expected_count,))
        self.assertNotIn("labels_mask", anomaly_data)

        manifest = anomaly_data["anomaly_manifest"]
        self.assertEqual(
            manifest["counts"]["high"], 2 * data_config.num_test_high_per_channel
        )
        self.assertEqual(
            manifest["counts"]["low"], 2 * data_config.num_test_low_per_channel
        )

        sample_seed = int(manifest["seed_plan"]["sample_seed"])
        channels_to_evaluate = list(manifest["channels_to_evaluate"])
        expected = _build_expected_single_channel_anomalies(
            data_config=data_config,
            anomaly_config=anomaly_config,
            channels_to_evaluate=channels_to_evaluate,
            sample_seed=sample_seed,
            batch_size=batch_size,
            device=self.device,
        )

        self.assertTrue(
            torch.equal(anomaly_data["x_test_anomaly"], expected["x_test_anomaly"])
        )
        self.assertTrue(
            torch.equal(anomaly_data["labels_type"], expected["labels_type"])
        )
        self.assertTrue(
            torch.equal(anomaly_data["labels_channel"], expected["labels_channel"])
        )

    def test_generate_anomaly_test_data_stream_materialization_matches_materialize(
        self,
    ) -> None:
        # given
        batch_size = 3
        base_data_kwargs: Dict[str, Any] = {
            "seed": 11,
            "num_train": 3,
            "num_val": 3,
            "num_test_normal": 3,
            "num_test_high_per_channel": 4,
            "num_test_low_per_channel": 5,
            "num_channels": 4,
            "noise_std": 0.0,
            "normal_mode": "standard_normal",
        }
        stream_data_config = DirectionalDeviationDataConfig(
            **base_data_kwargs,
            anomaly_eval_mode="stream",
        )
        materialize_data_config = DirectionalDeviationDataConfig(
            **base_data_kwargs,
            anomaly_eval_mode="materialize",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.25, shift_mode="single_channel"
        )

        # when
        stream_data = generate_anomaly_test_data(
            stream_data_config, anomaly_config, self.device, batch_size=batch_size
        )
        materialize_data = generate_anomaly_test_data(
            materialize_data_config, anomaly_config, self.device, batch_size=batch_size
        )
        materialized_stream = _materialize_anomaly_stream(stream_data["anomaly_stream"])

        # then
        self.assertTrue(
            torch.equal(
                materialized_stream["x_test_anomaly"],
                materialize_data["x_test_anomaly"],
            )
        )
        self.assertTrue(
            torch.equal(
                materialized_stream["labels_type"], materialize_data["labels_type"]
            )
        )
        self.assertTrue(
            torch.equal(
                materialized_stream["labels_channel"],
                materialize_data["labels_channel"],
            )
        )
        self.assertEqual(
            stream_data["anomaly_manifest"], materialize_data["anomaly_manifest"]
        )

    def test_generate_anomaly_test_data_subset_stream_and_materialize_match_expected(
        self,
    ) -> None:
        # given
        batch_size = 3
        base_data_kwargs: Dict[str, Any] = {
            "seed": 101,
            "num_train": 2,
            "num_val": 2,
            "num_test_normal": 2,
            "num_test_high_per_channel": 5,
            "num_test_low_per_channel": 4,
            "num_channels": 8,
            "noise_std": 0.0,
            "normal_mode": "standard_normal",
        }
        stream_data_config = DirectionalDeviationDataConfig(
            **base_data_kwargs,
            anomaly_eval_mode="stream",
        )
        materialize_data_config = DirectionalDeviationDataConfig(
            **base_data_kwargs,
            anomaly_eval_mode="materialize",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=2.5,
            shift_mode="subset",
            subset_size=3,
            channels_to_evaluate=[0, 3, 5, 7],
        )

        # when
        materialize_data = generate_anomaly_test_data(
            materialize_data_config, anomaly_config, self.device, batch_size=batch_size
        )
        stream_data = generate_anomaly_test_data(
            stream_data_config, anomaly_config, self.device, batch_size=batch_size
        )
        materialized_stream = _materialize_anomaly_stream(stream_data["anomaly_stream"])

        # then
        expected_count = (
            materialize_data_config.num_test_high_per_channel
            + materialize_data_config.num_test_low_per_channel
        )
        self.assertEqual(materialize_data["x_test_anomaly"].shape, (expected_count, 8))
        self.assertEqual(materialize_data["labels_type"].shape, (expected_count,))
        self.assertEqual(materialize_data["labels_channel"].shape, (expected_count,))
        self.assertEqual(materialize_data["labels_mask"].shape, (expected_count, 3))
        self.assertTrue(bool((materialize_data["labels_channel"] == -1).all()))

        subset_indices = list(materialize_data["anomaly_manifest"]["subset_indices"])
        self.assertEqual(len(subset_indices), 3)
        self.assertEqual(subset_indices, sorted(subset_indices))
        for index in subset_indices:
            self.assertIn(index, anomaly_config.channels_to_evaluate)
        self.assertTrue(
            torch.equal(
                materialize_data["labels_mask"],
                torch.tensor(subset_indices, dtype=torch.int64)
                .unsqueeze(0)
                .repeat(expected_count, 1),
            )
        )

        expected = _build_expected_subset_anomalies(
            data_config=materialize_data_config,
            anomaly_config=anomaly_config,
            subset_indices=subset_indices,
            sample_seed=int(
                materialize_data["anomaly_manifest"]["seed_plan"]["sample_seed"]
            ),
            batch_size=batch_size,
            device=self.device,
        )

        self.assertTrue(
            torch.equal(materialize_data["x_test_anomaly"], expected["x_test_anomaly"])
        )
        self.assertTrue(
            torch.equal(materialize_data["labels_type"], expected["labels_type"])
        )
        self.assertTrue(
            torch.equal(materialize_data["labels_channel"], expected["labels_channel"])
        )
        self.assertTrue(
            torch.equal(materialize_data["labels_mask"], expected["labels_mask"])
        )

        self.assertTrue(
            torch.equal(
                materialize_data["x_test_anomaly"],
                materialized_stream["x_test_anomaly"],
            )
        )
        self.assertTrue(
            torch.equal(
                materialize_data["labels_type"], materialized_stream["labels_type"]
            )
        )
        self.assertTrue(
            torch.equal(
                materialize_data["labels_channel"],
                materialized_stream["labels_channel"],
            )
        )
        self.assertTrue(
            torch.equal(
                materialize_data["labels_mask"], materialized_stream["labels_mask"]
            )
        )
        self.assertEqual(
            stream_data["anomaly_manifest"], materialize_data["anomaly_manifest"]
        )

    def test_build_feature_directions_from_decoder_is_deterministic_and_selects_top_purity(
        self,
    ) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self, decoder: Tensor) -> None:
                super().__init__()
                self.W_dec = decoder
                self.n_instances = 1

        decoder = torch.zeros((1, 3, 4), dtype=torch.float32)
        decoder[0, :, 0] = torch.tensor([1.0, 0.0, 0.0])
        decoder[0, :, 1] = torch.tensor([1.0, 1.0, 0.0])
        decoder[0, :, 2] = torch.tensor([0.5, 0.5, 0.5])
        decoder[0, :, 3] = torch.tensor([0.0, 0.0, 1.0])
        model = DummyModel(decoder)

        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=2,
            direction_selection="top_purity",
            normalize_directions=True,
        )

        # when
        directions_a, metadata_a = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=123,
        )
        directions_b, metadata_b = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=123,
        )

        # then
        self.assertTrue(torch.equal(directions_a, directions_b))
        self.assertEqual(metadata_a["direction_latent_indices"], [0, 3])
        self.assertEqual(
            metadata_a["direction_latent_indices"],
            metadata_b["direction_latent_indices"],
        )

    def test_build_feature_directions_from_decoder_top_pi_std_selection_is_deterministic(
        self,
    ) -> None:
        # given
        from sgnsae.models import SAE

        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=3,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        model.W_dec.data = torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 2.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        model.b_dec.data.zero_()
        model.b_enc.data.zero_()
        base_samples = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32
        )

        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=2,
            direction_selection="top_pi_std",
            normalize_directions=True,
        )

        # when
        directions_a, metadata_a = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=123,
            base_samples=base_samples,
            batch_size=2,
        )
        directions_b, metadata_b = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=123,
            base_samples=base_samples,
            batch_size=2,
        )

        # then
        self.assertTrue(torch.equal(directions_a, directions_b))
        self.assertEqual(metadata_a["direction_latent_indices"], [0, 2])
        self.assertEqual(
            metadata_a["direction_latent_indices"],
            metadata_b["direction_latent_indices"],
        )

        pi_std = metadata_a["direction_latent_pi_std"]
        self.assertIsInstance(pi_std, list)
        self.assertEqual(len(pi_std), 2)

        expected_std_0 = math.sqrt(1.25)
        expected_std_2 = expected_std_0 / math.sqrt(2.0)
        self.assertAlmostEqual(pi_std[0], expected_std_0, places=6)
        self.assertAlmostEqual(pi_std[1], expected_std_2, places=6)

    def test_build_feature_directions_from_decoder_top_weight_norm_selection_is_deterministic(
        self,
    ) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self, decoder: Tensor) -> None:
                super().__init__()
                self.W_dec = decoder
                self.n_instances = 1

        decoder = torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 2.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        model = DummyModel(decoder)

        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=2,
            direction_selection="top_weight_norm",
            normalize_directions=True,
        )

        # when
        directions_a, metadata_a = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=123,
        )
        directions_b, metadata_b = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=999,
        )

        # then
        self.assertTrue(torch.equal(directions_a, directions_b))
        self.assertEqual(metadata_a["direction_latent_indices"], [1, 2])
        self.assertEqual(
            metadata_a["direction_latent_indices"],
            metadata_b["direction_latent_indices"],
        )

        weight_norms = metadata_a["direction_latent_weight_norm"]
        self.assertIsInstance(weight_norms, list)
        self.assertEqual(len(weight_norms), 2)
        self.assertAlmostEqual(weight_norms[0], 2.0, places=6)
        self.assertAlmostEqual(weight_norms[1], math.sqrt(2.0), places=6)

    def test_build_feature_directions_from_decoder_random_selection_is_deterministic(
        self,
    ) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self, decoder: Tensor) -> None:
                super().__init__()
                self.W_dec = decoder
                self.n_instances = 1

        decoder = torch.arange(1, 1 + 1 * 3 * 6, dtype=torch.float32).reshape(1, 3, 6)
        model = DummyModel(decoder)

        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=3,
            direction_selection="random",
            normalize_directions=False,
        )

        # when
        directions_a, metadata_a = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=999,
        )
        directions_b, metadata_b = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=999,
        )

        # then
        self.assertTrue(torch.equal(directions_a, directions_b))
        self.assertEqual(
            metadata_a["direction_latent_indices"],
            metadata_b["direction_latent_indices"],
        )

    def test_build_feature_directions_from_decoder_fixed_pi_shift_clips_large_deltas(
        self,
    ) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self, decoder: Tensor) -> None:
                super().__init__()
                self.W_dec = decoder
                self.n_instances = 1

        decoder = torch.tensor(
            [[[1e-9, 1.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32
        )
        model = DummyModel(decoder)

        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=2,
            direction_selection="explicit",
            direction_latent_indices=[0, 1],
            normalize_directions=True,
            direction_delta_mode="fixed_pi_shift",
            max_direction_input_delta=2.0,
        )

        # when
        _, metadata = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=123,
        )

        # then
        self.assertEqual(metadata["direction_delta_mode"], "fixed_pi_shift")
        self.assertEqual(metadata["direction_input_deltas_clipped_indices"], [0])
        input_deltas = metadata["direction_input_deltas"]
        self.assertIsInstance(input_deltas, list)
        self.assertEqual(len(input_deltas), 2)
        self.assertAlmostEqual(input_deltas[0], 2.0, places=6)
        self.assertAlmostEqual(
            input_deltas[1],
            1.0 / (1.0 + float(data_module._DIRECTION_EPSILON)),
            places=6,
        )

    def test_generate_anomaly_test_data_feature_direction_materialize_matches_expected(
        self,
    ) -> None:
        # given
        batch_size = 2
        data_config = DirectionalDeviationDataConfig(
            seed=5,
            num_train=3,
            num_val=3,
            num_test_normal=3,
            num_test_high_per_channel=2,
            num_test_low_per_channel=1,
            num_channels=4,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="materialize",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=2.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=2,
            num_test_high_per_direction=3,
            num_test_low_per_direction=2,
            normalize_directions=True,
        )
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.6, 0.8, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        # when
        anomaly_data = generate_anomaly_test_data(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=batch_size,
            anomaly_directions=directions,
            direction_metadata={
                "direction_selection": "explicit",
                "direction_latent_indices": [0, 1],
            },
        )

        # then
        num_high = int(anomaly_config.num_test_high_per_direction)
        num_low = int(anomaly_config.num_test_low_per_direction)
        expected_count = int(anomaly_config.num_directions) * (num_high + num_low)
        self.assertEqual(anomaly_data["x_test_anomaly"].shape, (expected_count, 4))
        self.assertEqual(anomaly_data["labels_type"].shape, (expected_count,))
        self.assertEqual(anomaly_data["labels_channel"].shape, (expected_count,))

        manifest = anomaly_data["anomaly_manifest"]
        self.assertEqual(manifest["shift_mode"], "feature_direction")
        self.assertEqual(
            manifest["counts"]["high"], int(anomaly_config.num_directions) * num_high
        )
        self.assertEqual(
            manifest["counts"]["low"], int(anomaly_config.num_directions) * num_low
        )

        expected = _build_expected_feature_direction_anomalies(
            data_config=data_config,
            anomaly_config=anomaly_config,
            directions=directions,
            direction_input_deltas=None,
            sample_seed=int(manifest["seed_plan"]["sample_seed"]),
            batch_size=batch_size,
            device=self.device,
        )
        self.assertTrue(
            torch.equal(anomaly_data["x_test_anomaly"], expected["x_test_anomaly"])
        )
        self.assertTrue(
            torch.equal(anomaly_data["labels_type"], expected["labels_type"])
        )
        self.assertTrue(
            torch.equal(anomaly_data["labels_channel"], expected["labels_channel"])
        )

    def test_generate_anomaly_test_data_feature_direction_fixed_pi_shift_matches_expected(
        self,
    ) -> None:
        # given
        batch_size = 2
        data_config = DirectionalDeviationDataConfig(
            seed=5,
            num_train=3,
            num_val=3,
            num_test_normal=3,
            num_test_high_per_channel=2,
            num_test_low_per_channel=1,
            num_channels=4,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="materialize",
        )

        class DummyModel(torch.nn.Module):
            def __init__(self, decoder: Tensor) -> None:
                super().__init__()
                self.W_dec = decoder
                self.n_instances = 1

        decoder = torch.tensor(
            [[[1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32
        )
        model = DummyModel(decoder)

        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=2.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=2,
            direction_selection="explicit",
            direction_latent_indices=[0, 1],
            num_test_high_per_direction=3,
            num_test_low_per_direction=2,
            normalize_directions=True,
            direction_delta_mode="fixed_pi_shift",
        )
        directions, metadata = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=123,
        )

        # when
        anomaly_data = generate_anomaly_test_data(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=batch_size,
            anomaly_directions=directions,
            direction_metadata=metadata,
        )

        # then
        manifest = anomaly_data["anomaly_manifest"]
        self.assertEqual(manifest["direction_delta_mode"], "fixed_pi_shift")
        self.assertEqual(manifest["direction_input_deltas_clipped_indices"], [])
        direction_input_deltas = torch.tensor(
            manifest["direction_input_deltas"], dtype=torch.float32
        )
        self.assertEqual(tuple(direction_input_deltas.shape), (2,))
        self.assertAlmostEqual(float(direction_input_deltas[0]), 2.0, places=6)
        self.assertAlmostEqual(float(direction_input_deltas[1]), 1.0, places=6)

        expected = _build_expected_feature_direction_anomalies(
            data_config=data_config,
            anomaly_config=anomaly_config,
            directions=directions,
            direction_input_deltas=direction_input_deltas,
            sample_seed=int(manifest["seed_plan"]["sample_seed"]),
            batch_size=batch_size,
            device=self.device,
        )
        self.assertTrue(
            torch.equal(anomaly_data["x_test_anomaly"], expected["x_test_anomaly"])
        )

    def test_feature_direction_fixed_pi_shift_produces_expected_pi_shifts(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=5,
            num_train=3,
            num_val=3,
            num_test_normal=3,
            num_test_high_per_channel=2,
            num_test_low_per_channel=2,
            num_channels=4,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="materialize",
        )

        class DummyModel(torch.nn.Module):
            def __init__(self, decoder: Tensor) -> None:
                super().__init__()
                self.W_dec = decoder
                self.n_instances = 1

        decoder = torch.tensor(
            [[[1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32
        )
        model = DummyModel(decoder)

        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=2.0,
            shift_mode="feature_direction",
            direction_source="reference_model_decoder",
            num_directions=2,
            direction_selection="explicit",
            direction_latent_indices=[0, 1],
            normalize_directions=True,
            direction_delta_mode="fixed_pi_shift",
        )

        directions, metadata = build_feature_directions_from_decoder(
            model=model,
            anomaly_config=anomaly_config,
            direction_seed=123,
        )
        anomaly_data = generate_anomaly_test_data(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=4,
            anomaly_directions=directions,
            direction_metadata=metadata,
        )

        # when
        x_anom = anomaly_data["x_test_anomaly"].to(device="cpu")
        labels_type = anomaly_data["labels_type"].to(device="cpu")
        labels_target = anomaly_data["labels_channel"].to(device="cpu")
        direction_input_deltas = torch.tensor(
            anomaly_data["anomaly_manifest"]["direction_input_deltas"],
            dtype=torch.float32,
        )

        # then
        latent_indices = metadata["direction_latent_indices"]
        for index in range(int(x_anom.shape[0])):
            direction_index = int(labels_target[index])
            latent_index = int(latent_indices[direction_index])
            raw_direction = decoder[0, :, latent_index].to(device="cpu")
            raw_norm = float(torch.linalg.norm(raw_direction))
            input_delta = float(direction_input_deltas[direction_index])
            direction_vector = directions[direction_index].to(device="cpu")
            sign = 1.0 if int(labels_type[index]) == LABEL_TYPE_HIGH else -1.0

            x_base = x_anom[index] - sign * input_delta * direction_vector
            pi_anom = float(torch.dot(x_anom[index], raw_direction))
            pi_base = float(torch.dot(x_base, raw_direction))
            observed_shift = pi_anom - pi_base
            expected_shift = (
                sign
                * float(anomaly_config.delta)
                * raw_norm
                / (raw_norm + float(data_module._DIRECTION_EPSILON))
            )
            self.assertAlmostEqual(observed_shift, expected_shift, places=5)

    def test_generate_anomaly_test_data_feature_direction_fixed_random_fixed_pi_shift_includes_input_deltas(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=9,
            num_train=3,
            num_val=3,
            num_test_normal=3,
            num_test_high_per_channel=2,
            num_test_low_per_channel=1,
            num_channels=1,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=3,
            normalize_directions=True,
            direction_delta_mode="fixed_pi_shift",
            max_direction_input_delta=0.01,
        )

        # when
        anomaly_data = generate_anomaly_test_data(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=4,
        )

        # then
        manifest = anomaly_data["anomaly_manifest"]
        self.assertEqual(manifest["direction_source"], "fixed_random")
        self.assertEqual(manifest["direction_delta_mode"], "fixed_pi_shift")
        self.assertEqual(len(manifest["direction_input_deltas"]), 3)
        self.assertEqual(manifest["direction_input_deltas_clipped_indices"], [0, 1, 2])

    def test_export_data_summary_writes_expected_files_for_stream_anomalies(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=0,
            num_train=10,
            num_val=11,
            num_test_normal=12,
            num_test_high_per_channel=2,
            num_test_low_per_channel=3,
            num_channels=5,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=4.0, shift_mode="single_channel"
        )
        normal_data = generate_normal_data(data_config, self.device)
        anomaly_data = generate_anomaly_test_data(
            data_config, anomaly_config, self.device, batch_size=4
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            export_data_summary(
                tmpdir,
                data_config,
                anomaly_config,
                normal_data,
                anomaly_data,
                sample_preview_count=4,
                anomaly_preview_count=5,
            )

            # then
            data_dir = os.path.join(tmpdir, "data")
            self.assertTrue(
                os.path.exists(os.path.join(data_dir, "generator_config.json"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(data_dir, "anomaly_manifest.json"))
            )
            self.assertTrue(os.path.exists(os.path.join(data_dir, "sampled_data.pt")))

            with open(os.path.join(data_dir, "generator_config.json"), "r") as f:
                generator_config = json.load(f)
            self.assertIn("seed_plan", generator_config)
            self.assertEqual(generator_config["seed"], data_config.seed)

            with open(os.path.join(data_dir, "anomaly_manifest.json"), "r") as f:
                anomaly_manifest = json.load(f)
            self.assertEqual(anomaly_manifest["shift_mode"], "single_channel")
            self.assertEqual(anomaly_manifest["counts"]["total"], 25)

            sampled = torch.load(os.path.join(data_dir, "sampled_data.pt"))
            self.assertEqual(sampled["x_train"].shape, (4, 5))
            self.assertEqual(sampled["x_val"].shape, (4, 5))
            self.assertEqual(sampled["x_test_normal"].shape, (4, 5))
            self.assertEqual(sampled["x_test_anomaly"].shape, (5, 5))
            self.assertEqual(sampled["labels_type"].shape, (5,))
            self.assertEqual(sampled["labels_channel"].shape, (5,))

    def test_export_data_summary_writes_anomaly_directions_for_feature_direction(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=0,
            num_train=10,
            num_val=11,
            num_test_normal=12,
            num_test_high_per_channel=2,
            num_test_low_per_channel=3,
            num_channels=5,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=4.0,
            shift_mode="feature_direction",
            direction_source="fixed_random",
            num_directions=3,
            normalize_directions=True,
        )
        normal_data = generate_normal_data(data_config, self.device)
        anomaly_data = generate_anomaly_test_data(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            export_data_summary(
                tmpdir,
                data_config,
                anomaly_config,
                normal_data,
                anomaly_data,
                sample_preview_count=4,
                anomaly_preview_count=5,
            )

            # then
            data_dir = os.path.join(tmpdir, "data")
            directions_path = os.path.join(data_dir, "anomaly_directions.pt")
            self.assertTrue(os.path.isfile(directions_path))
            directions = torch.load(directions_path)
            self.assertEqual(tuple(directions.shape), (3, 5))

            manifest_path = os.path.join(data_dir, "anomaly_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["direction_source"], "fixed_random")
            self.assertEqual(manifest["direction_selection"], "fixed_random")

    def test_export_data_summary_records_non_gaussian_derived_parameters(
        self,
    ) -> None:
        # given
        base_kwargs: Dict[str, Any] = {
            "seed": 2,
            "num_train": 10,
            "num_val": 11,
            "num_test_normal": 12,
            "num_test_high_per_channel": 2,
            "num_test_low_per_channel": 3,
            "num_channels": 5,
            "noise_std": 0.0,
            "normal_mode": "iid_non_gaussian",
            "anomaly_eval_mode": "stream",
        }
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=4.0, shift_mode="single_channel"
        )

        # when & then
        for non_gaussian_dist, extra_kwargs, expected_key, expected_value in (
            (
                "laplace",
                {},
                "laplace_scale",
                1.0 / math.sqrt(2.0),
            ),
            (
                "logistic",
                {},
                "logistic_scale",
                math.sqrt(3.0) / math.pi,
            ),
            (
                "student_t",
                {"student_t_df": 7.0},
                "student_t_unit_variance_scale",
                math.sqrt((7.0 - 2.0) / 7.0),
            ),
        ):
            with self.subTest(non_gaussian_dist=non_gaussian_dist):
                data_config = DirectionalDeviationDataConfig(
                    **base_kwargs,
                    non_gaussian_dist=non_gaussian_dist,  # type: ignore[arg-type]
                    **extra_kwargs,
                )
                normal_data = generate_normal_data(data_config, self.device)
                anomaly_data = generate_anomaly_test_data(
                    data_config, anomaly_config, self.device, batch_size=4
                )

                with tempfile.TemporaryDirectory() as tmpdir:
                    export_data_summary(
                        tmpdir,
                        data_config,
                        anomaly_config,
                        normal_data,
                        anomaly_data,
                        sample_preview_count=4,
                        anomaly_preview_count=5,
                    )

                    with open(
                        os.path.join(tmpdir, "data", "generator_config.json"), "r"
                    ) as f:
                        generator_config = json.load(f)

                self.assertEqual(generator_config["normal_mode"], "iid_non_gaussian")
                self.assertEqual(
                    generator_config["non_gaussian_dist"], non_gaussian_dist
                )
                self.assertIn("non_gaussian_derived", generator_config)
                self.assertIn(expected_key, generator_config["non_gaussian_derived"])
                self.assertAlmostEqual(
                    generator_config["non_gaussian_derived"][expected_key],
                    expected_value,
                    places=7,
                )

    def test_export_data_summary_writes_expected_files_for_materialized_anomalies(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=2,
            num_train=10,
            num_val=11,
            num_test_normal=12,
            num_test_high_per_channel=4,
            num_test_low_per_channel=5,
            num_channels=6,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="materialize",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=3.0,
            shift_mode="subset",
            subset_size=2,
            channels_to_evaluate=[1, 2, 4],
        )
        normal_data = generate_normal_data(data_config, self.device)
        anomaly_data = generate_anomaly_test_data(
            data_config, anomaly_config, self.device, batch_size=8
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            export_data_summary(
                tmpdir,
                data_config,
                anomaly_config,
                normal_data,
                anomaly_data,
                sample_preview_count=4,
                anomaly_preview_count=7,
            )

            # then
            data_dir = os.path.join(tmpdir, "data")
            sampled = torch.load(os.path.join(data_dir, "sampled_data.pt"))
            self.assertEqual(sampled["x_train"].shape, (4, 6))
            self.assertEqual(sampled["x_val"].shape, (4, 6))
            self.assertEqual(sampled["x_test_normal"].shape, (4, 6))
            self.assertEqual(sampled["x_test_anomaly"].shape, (7, 6))
            self.assertEqual(sampled["labels_type"].shape, (7,))
            self.assertEqual(sampled["labels_channel"].shape, (7,))
            self.assertEqual(sampled["labels_mask"].shape, (7, 2))

    def test_export_data_summary_includes_labels_mask_for_stream_subset_anomalies(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=3,
            num_train=10,
            num_val=11,
            num_test_normal=12,
            num_test_high_per_channel=4,
            num_test_low_per_channel=5,
            num_channels=6,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=3.0,
            shift_mode="subset",
            subset_size=2,
            channels_to_evaluate=[1, 2, 4],
        )
        normal_data = generate_normal_data(data_config, self.device)
        anomaly_data = generate_anomaly_test_data(
            data_config, anomaly_config, self.device, batch_size=8
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            export_data_summary(
                tmpdir,
                data_config,
                anomaly_config,
                normal_data,
                anomaly_data,
                sample_preview_count=4,
                anomaly_preview_count=7,
            )

            # then
            data_dir = os.path.join(tmpdir, "data")
            sampled = torch.load(os.path.join(data_dir, "sampled_data.pt"))
            self.assertEqual(sampled["labels_mask"].shape, (7, 2))

    def test_select_subset_indices_raises_for_non_subset_shift_mode(self) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=0,
            num_train=10,
            num_val=11,
            num_test_normal=12,
            num_test_high_per_channel=4,
            num_test_low_per_channel=5,
            num_channels=6,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._select_subset_indices(data_config, anomaly_config, seed=0)

    def test_select_subset_indices_raises_for_missing_subset_size(self) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=0,
            num_train=10,
            num_val=11,
            num_test_normal=12,
            num_test_high_per_channel=4,
            num_test_low_per_channel=5,
            num_channels=6,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        anomaly_config = SimpleNamespace(
            shift_mode="subset", subset_size=None, channels_to_evaluate=None
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._select_subset_indices(data_config, anomaly_config, seed=0)

    def test_select_subset_indices_raises_when_subset_size_exceeds_candidates(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=0,
            num_train=10,
            num_val=11,
            num_test_normal=12,
            num_test_high_per_channel=4,
            num_test_low_per_channel=5,
            num_channels=6,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="subset",
            subset_size=3,
            channels_to_evaluate=[0, 1],
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._select_subset_indices(data_config, anomaly_config, seed=0)

    def test_anomaly_stream_raises_when_subset_indices_missing(self) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=0,
            num_train=10,
            num_val=11,
            num_test_normal=12,
            num_test_high_per_channel=4,
            num_test_low_per_channel=5,
            num_channels=6,
            noise_std=0.0,
            normal_mode="standard_normal",
            anomaly_eval_mode="stream",
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="subset", subset_size=2
        )
        stream = data_module.DirectionalDeviationAnomalyStream(
            data_config=data_config,
            anomaly_config=anomaly_config,
            device=self.device,
            batch_size=2,
            channels_to_evaluate=list(range(data_config.num_channels)),
            subset_indices=None,
            sample_seed=0,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = next(iter(stream))

    def test_iterate_batch_counts_raises_for_non_positive_batch_size(self) -> None:
        # given
        # when & then
        with self.assertRaises(ValueError):
            _ = list(data_module._iterate_batch_counts(total_count=1, batch_size=0))

    def test_collect_anomaly_preview_raises_for_invalid_anomaly_data(self) -> None:
        # given
        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._collect_anomaly_preview({}, anomaly_preview_count=1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
