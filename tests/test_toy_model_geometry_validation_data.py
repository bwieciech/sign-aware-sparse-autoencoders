import json
import math
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

import torch
from torch import Tensor

from sgnsae.experiments.toy_model_geometry_validation.config import Toy2DDataConfig
from sgnsae.experiments.toy_model_geometry_validation.data import (
    export_toy2d_dataset,
    generate_toy2d_dataset,
)
from sgnsae.experiments.toy_model_geometry_validation import data as data_module


def _calculate_pearson_correlation(x: Tensor, y: Tensor) -> float:
    x_values = x.to(dtype=torch.float64)
    y_values = y.to(dtype=torch.float64)
    x_mean = x_values.mean()
    y_mean = y_values.mean()
    covariance = ((x_values - x_mean) * (y_values - y_mean)).mean()
    x_std = x_values.std(unbiased=False)
    y_std = y_values.std(unbiased=False)
    return float((covariance / (x_std * y_std + 1e-12)).item())


class TestToyModelGeometryValidationDataGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"

    def _build_config(self, overrides: Dict[str, Any] | None = None) -> Toy2DDataConfig:
        base: Dict[str, Any] = {
            "seed": 123,
            "num_train": 32,
            "num_val": 16,
            "num_test": 8,
            "num_pairs": 4,
            "direction_mode": "uniform",
            "pair_axis_active_prob": 0.5,
            "pair_feature_prob": 0.5,
            "pair_correlation": -1.0,
            "magnitude_dist": "constant",
            "constant_magnitude": 1.0,
            "noise_std": 0.0,
        }
        if overrides:
            base.update(overrides)
        return Toy2DDataConfig(**base)

    def test_generate_dataset_shapes_and_antipodal_unit_norms(self) -> None:
        # given
        config = self._build_config()

        # when
        data = generate_toy2d_dataset(config, self.device)

        # then
        self.assertEqual(data["pair_axis_angles_rad"].shape, (4,))
        self.assertEqual(data["pair_axis_directions"].shape, (4, 2))
        self.assertEqual(data["feature_directions"].shape, (8, 2))
        self.assertEqual(data["x_train"].shape, (32, 2))
        self.assertEqual(data["x_val"].shape, (16, 2))
        self.assertEqual(data["x_test"].shape, (8, 2))
        self.assertEqual(data["coeffs_train"].shape, (32, 8))
        self.assertEqual(data["coeffs_val"].shape, (16, 8))
        self.assertEqual(data["coeffs_test"].shape, (8, 8))
        self.assertEqual(data["pair_active_mask_train"].shape, (32, 4))
        self.assertEqual(data["pair_active_mask_val"].shape, (16, 4))
        self.assertEqual(data["pair_active_mask_test"].shape, (8, 4))

        self.assertTrue(
            bool(
                torch.allclose(
                    data["feature_directions"][1::2],
                    -data["feature_directions"][0::2],
                )
            )
        )
        norms = data["feature_directions"].norm(dim=1)
        self.assertTrue(bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5)))

        expected_angles = torch.tensor(
            [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.assertTrue(
            bool(
                torch.allclose(data["pair_axis_angles_rad"], expected_angles, atol=1e-7)
            )
        )

        reconstructed = data["coeffs_train"] @ data["feature_directions"]
        self.assertTrue(bool(torch.allclose(reconstructed, data["x_train"], atol=1e-6)))

    def test_generate_dataset_reproducible_with_seed(self) -> None:
        # given
        config = self._build_config(
            {
                "seed": 99,
                "direction_mode": "random",
                "magnitude_dist": "lognormal",
                "lognormal_mu": -0.1,
                "lognormal_sigma": 0.2,
                "noise_std": 0.05,
            }
        )

        # when
        data_first = generate_toy2d_dataset(config, self.device)
        data_second = generate_toy2d_dataset(config, self.device)

        # then
        for key in data_first:
            self.assertTrue(torch.equal(data_first[key], data_second[key]))

    def test_generate_dataset_raises_when_seed_is_list(self) -> None:
        # given
        config = self._build_config({"seed": [1, 2]})

        # when & then
        with self.assertRaises(ValueError):
            _ = generate_toy2d_dataset(config, self.device)

    def test_pair_active_mask_zeroes_out_coefficients_for_inactive_pairs(self) -> None:
        # given
        config = self._build_config({"seed": 7, "num_train": 256, "num_pairs": 3})

        # when
        data = generate_toy2d_dataset(config, self.device)

        # then
        pair_active_mask = data["pair_active_mask_train"]
        coeffs = data["coeffs_train"]
        plus_coeffs = coeffs[:, 0::2]
        minus_coeffs = coeffs[:, 1::2]
        inactive_mask = ~pair_active_mask
        self.assertTrue(bool((plus_coeffs[inactive_mask] == 0.0).all()))
        self.assertTrue(bool((minus_coeffs[inactive_mask] == 0.0).all()))

    def test_active_pairs_per_sample_selects_exactly_n_pairs_per_sample(self) -> None:
        # given
        config = self._build_config(
            {
                "seed": 19,
                "num_train": 256,
                "num_val": 1,
                "num_test": 1,
                "num_pairs": 4,
                "active_pairs_per_sample": 2,
                "pair_axis_active_prob": 0.0,
            }
        )

        # when
        data = generate_toy2d_dataset(config, self.device)

        # then
        pair_active_mask = data["pair_active_mask_train"].to(dtype=torch.int64)
        active_counts = pair_active_mask.sum(dim=1)
        self.assertTrue(bool((active_counts == 2).all()))

    def test_active_pairs_per_sample_raises_when_out_of_range(self) -> None:
        # given
        config = self._build_config({"num_pairs": 3, "active_pairs_per_sample": 4})

        # when & then
        with self.assertRaises(ValueError):
            _ = generate_toy2d_dataset(config, self.device)

    def test_within_pair_correlation_matches_expected_values(self) -> None:
        # given
        correlations_to_test = [-1.0, -0.5, 0.0]
        base_overrides = {
            "seed": 11,
            "num_train": 20000,
            "num_val": 1,
            "num_test": 1,
            "num_pairs": 1,
            "pair_axis_active_prob": 1.0,
            "pair_feature_prob": 0.5,
            "noise_std": 0.0,
        }

        for expected_corr in correlations_to_test:
            config = self._build_config(
                {**base_overrides, "pair_correlation": expected_corr}
            )

            # when
            data = generate_toy2d_dataset(config, self.device)

            # then
            eligible_mask = data["pair_active_mask_train"][:, 0]
            plus_active = (data["coeffs_train"][eligible_mask, 0] > 0).to(
                dtype=torch.float32
            )
            minus_active = (data["coeffs_train"][eligible_mask, 1] > 0).to(
                dtype=torch.float32
            )
            self.assertGreater(int(eligible_mask.sum().item()), 1000)
            self.assertGreater(float(plus_active.std(unbiased=False).item()), 0.0)
            self.assertGreater(float(minus_active.std(unbiased=False).item()), 0.0)

            observed_corr = _calculate_pearson_correlation(plus_active, minus_active)

            if expected_corr == -1.0:
                self.assertLess(observed_corr, -0.999)
            else:
                self.assertTrue(
                    math.isclose(observed_corr, expected_corr, abs_tol=0.05),
                    msg=f"Expected corr {expected_corr}, got {observed_corr}.",
                )

    def test_export_dataset_writes_expected_artifacts(self) -> None:
        # given
        config = self._build_config(
            {
                "seed": 5,
                "num_train": 10,
                "num_val": 4,
                "num_test": 3,
                "num_pairs": 4,
                "pair_axis_active_prob": 1.0,
                "pair_correlation": -0.5,
            }
        )
        data = generate_toy2d_dataset(config, self.device)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir, exist_ok=True)

            # when
            export_toy2d_dataset(
                run_dir=run_dir,
                config=config,
                data=data,
                sample_preview_count=5,
            )

            # then
            data_dir = os.path.join(run_dir, "data")
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "generator_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "feature_directions.pt"))
            )
            self.assertTrue(os.path.isfile(os.path.join(data_dir, "sampled_data.pt")))

            for split in ("train", "val", "test"):
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(data_dir, f"pair_active_mask_{split}.pt")
                    )
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(data_dir, f"coeffs_{split}.pt"))
                )

            with open(os.path.join(data_dir, "generator_config.json"), "r") as f:
                persisted = json.load(f)
            self.assertIn("pair_axis_angles_rad", persisted)
            self.assertEqual(len(persisted["pair_axis_angles_rad"]), 4)
            self.assertIn("pair_correlation_feasible_interval", persisted)
            self.assertEqual(
                persisted["pair_correlation_feasible_interval"],
                {"r_min": -1.0, "r_max": 1.0},
            )
            self.assertIn("within_pair_joint_probability_q", persisted)
            self.assertAlmostEqual(
                persisted["within_pair_joint_probability_q"],
                0.125,
                places=6,
            )
            self.assertIn("within_pair_joint_probabilities", persisted)
            self.assertEqual(
                set(persisted["within_pair_joint_probabilities"].keys()),
                {"p00", "p01", "p10", "p11"},
            )

            angles = torch.tensor(
                persisted["pair_axis_angles_rad"], dtype=torch.float32, device="cpu"
            )
            expected_angles = torch.tensor(
                [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0],
                dtype=torch.float32,
                device="cpu",
            )
            self.assertTrue(bool(torch.allclose(angles, expected_angles, atol=1e-7)))

            sampled = torch.load(os.path.join(data_dir, "sampled_data.pt"))
            for split in ("train", "val", "test"):
                self.assertIn(f"x_{split}", sampled)
                self.assertIn(f"coeffs_{split}", sampled)
                self.assertIn(f"pair_active_mask_{split}", sampled)
                self.assertEqual(sampled[f"x_{split}"].shape[1], 2)
                self.assertEqual(sampled[f"coeffs_{split}"].shape[1], 8)
                self.assertEqual(sampled[f"pair_active_mask_{split}"].shape[1], 4)

    def test_set_random_seed_seeds_cuda_when_available(self) -> None:
        # given
        seed = 2**63 + 5

        # when
        with patch.object(data_module.torch.cuda, "is_available", return_value=True):
            with patch.object(data_module.torch, "manual_seed") as manual_seed:
                with patch.object(
                    data_module.torch.cuda, "manual_seed_all"
                ) as manual_seed_all:
                    data_module.set_random_seed(seed)

        # then
        manual_seed.assert_called_once_with(5)
        manual_seed_all.assert_called_once_with(5)

    def test_internal_helpers_reject_invalid_inputs(self) -> None:
        # given
        generator = data_module._create_cpu_generator(0)
        config = self._build_config()

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._construct_pair_directions(torch.zeros((2, 2)))

        config.direction_mode = "not_a_mode"
        with self.assertRaises(ValueError):
            _ = data_module._generate_pair_axis_angles(config, generator)

    def test_sample_pair_active_mask_rejects_out_of_range_prob(self) -> None:
        # given
        generator = data_module._create_cpu_generator(0)
        config = self._build_config()
        config.pair_axis_active_prob = 1.1

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._sample_pair_active_mask(2, config, generator)

    def test_sample_within_pair_presence_rejects_missing_or_grid_correlation(
        self,
    ) -> None:
        # given
        generator = data_module._create_cpu_generator(0)
        config = self._build_config()
        config.pair_correlation = None

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._sample_within_pair_presence(2, config, generator)

        config = self._build_config()
        config.pair_correlation_grid = [-1.0]
        with self.assertRaises(ValueError):
            _ = data_module._sample_within_pair_presence(2, config, generator)

    def test_sample_within_pair_presence_rejects_degenerate_probabilities(self) -> None:
        # given
        generator = data_module._create_cpu_generator(0)
        config = self._build_config()

        # when & then
        with patch.object(
            data_module,
            "_calculate_within_pair_joint_probabilities",
            return_value=(0.0, 0.0, 0.0, 0.0),
        ):
            with self.assertRaises(ValueError):
                _ = data_module._sample_within_pair_presence(2, config, generator)

    def test_sample_within_pair_presence_rejects_negative_joint_probabilities(
        self,
    ) -> None:
        # given
        generator = data_module._create_cpu_generator(0)
        config = self._build_config()

        # when & then
        with patch.object(
            data_module,
            "_calculate_within_pair_joint_probabilities",
            return_value=(-0.25, 0.25, 0.25, 0.75),
        ):
            with self.assertRaises(ValueError):
                _ = data_module._sample_within_pair_presence(2, config, generator)

    def test_calculate_within_pair_joint_probabilities_rejects_invalid_correlation(
        self,
    ) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = data_module._calculate_within_pair_joint_probabilities(0.5, 2.0)

        with self.assertRaises(ValueError) as context:
            _ = data_module._calculate_within_pair_joint_probabilities(0.1, -1.0)
        self.assertIn("feasible interval is [", str(context.exception))

    def test_sample_magnitudes_rejects_unknown_distribution(self) -> None:
        # given
        generator = data_module._create_cpu_generator(0)
        config = self._build_config()
        config.magnitude_dist = "not_a_dist"

        # when & then
        with self.assertRaises(ValueError):
            _ = data_module._sample_magnitudes((2, 1), config, generator)

    def test_export_dataset_rejects_missing_required_keys(self) -> None:
        # given
        config = self._build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir, exist_ok=True)

            # when & then
            with self.assertRaises(ValueError):
                export_toy2d_dataset(run_dir, config, {}, sample_preview_count=1)

            with self.assertRaises(ValueError):
                export_toy2d_dataset(
                    run_dir,
                    config,
                    {"pair_axis_angles_rad": torch.zeros((config.num_pairs,))},
                    sample_preview_count=1,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
