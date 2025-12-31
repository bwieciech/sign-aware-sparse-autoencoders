import json
import os
import tempfile
import unittest
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
    export_data_summary,
    generate_anomaly_test_data,
    generate_normal_data,
)


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
                batch = _sample_expected_normal_split(batch_count, data_config, generator)
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


class TestDirectionalDeviationDataGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"

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
        expected_count = 2 * (data_config.num_test_high_per_channel + data_config.num_test_low_per_channel)
        self.assertEqual(anomaly_data["x_test_anomaly"].shape, (expected_count, 6))
        self.assertEqual(anomaly_data["labels_type"].shape, (expected_count,))
        self.assertEqual(anomaly_data["labels_channel"].shape, (expected_count,))
        self.assertNotIn("labels_mask", anomaly_data)

        manifest = anomaly_data["anomaly_manifest"]
        self.assertEqual(manifest["counts"]["high"], 2 * data_config.num_test_high_per_channel)
        self.assertEqual(manifest["counts"]["low"], 2 * data_config.num_test_low_per_channel)

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

        self.assertTrue(torch.equal(anomaly_data["x_test_anomaly"], expected["x_test_anomaly"]))
        self.assertTrue(torch.equal(anomaly_data["labels_type"], expected["labels_type"]))
        self.assertTrue(torch.equal(anomaly_data["labels_channel"], expected["labels_channel"]))

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
            torch.equal(materialized_stream["x_test_anomaly"], materialize_data["x_test_anomaly"])
        )
        self.assertTrue(
            torch.equal(materialized_stream["labels_type"], materialize_data["labels_type"])
        )
        self.assertTrue(
            torch.equal(materialized_stream["labels_channel"], materialize_data["labels_channel"])
        )
        self.assertEqual(stream_data["anomaly_manifest"], materialize_data["anomaly_manifest"])

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
            sample_seed=int(materialize_data["anomaly_manifest"]["seed_plan"]["sample_seed"]),
            batch_size=batch_size,
            device=self.device,
        )

        self.assertTrue(torch.equal(materialize_data["x_test_anomaly"], expected["x_test_anomaly"]))
        self.assertTrue(torch.equal(materialize_data["labels_type"], expected["labels_type"]))
        self.assertTrue(torch.equal(materialize_data["labels_channel"], expected["labels_channel"]))
        self.assertTrue(torch.equal(materialize_data["labels_mask"], expected["labels_mask"]))

        self.assertTrue(
            torch.equal(
                materialize_data["x_test_anomaly"], materialized_stream["x_test_anomaly"]
            )
        )
        self.assertTrue(
            torch.equal(materialize_data["labels_type"], materialized_stream["labels_type"])
        )
        self.assertTrue(
            torch.equal(
                materialize_data["labels_channel"], materialized_stream["labels_channel"]
            )
        )
        self.assertTrue(
            torch.equal(materialize_data["labels_mask"], materialized_stream["labels_mask"])
        )
        self.assertEqual(stream_data["anomaly_manifest"], materialize_data["anomaly_manifest"])

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
        anomaly_config = DirectionalDeviationAnomalyConfig(delta=4.0, shift_mode="single_channel")
        normal_data = generate_normal_data(data_config, self.device)
        anomaly_data = generate_anomaly_test_data(data_config, anomaly_config, self.device, batch_size=4)

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
            self.assertTrue(os.path.exists(os.path.join(data_dir, "generator_config.json")))
            self.assertTrue(os.path.exists(os.path.join(data_dir, "anomaly_manifest.json")))
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
        anomaly_data = generate_anomaly_test_data(data_config, anomaly_config, self.device, batch_size=8)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

