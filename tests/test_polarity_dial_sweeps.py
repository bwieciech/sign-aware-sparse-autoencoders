import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import torch

from sgnsae.experiments.polarity_dial import config as config_module
from sgnsae.experiments.polarity_dial import plotting as plotting_module
from sgnsae.experiments.polarity_dial import runner as runner_module
from sgnsae.experiments.polarity_dial import sweeps as sweeps_module


class TestPolarityDialSweepConfig(unittest.TestCase):
    def test_load_config_maps_top_level_sparsity_grid_into_sweeps(self) -> None:
        # given
        config_dict = {
            "output_dir": "/tmp/output",
            "device": "cpu",
            "data": {
                "seed": 1,
                "num_train": 2,
                "num_val": 1,
                "num_test": 1,
                "input_dim": 3,
                "num_features": 2,
                "activation_prob": 0.5,
                "positive_sign_prob": 0.6,
                "lognormal_mu": 0.0,
                "lognormal_sigma": 0.4,
                "exponential_lambda": 1.2,
                "noise_std": 0.0,
            },
            "models": [
                {
                    "name": "relu",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "single_activation_offset": False,
                    "reset_activation_offsets_on_resample": False,
                    "d_hidden": 2,
                    "sparsity_coeff": 0.01,
                    "learning_rate": 0.1,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "adam_betas": [0.9, 0.999],
                }
            ],
            "sparsity_coeff_grid": [0.0, 0.01],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            # when
            loaded = config_module.load_experiment_config(config_path)

            # then
            self.assertEqual(loaded.sweeps.sparsity_coeff_grid, [0.0, 0.01])
            self.assertEqual(loaded.sweeps.sweep_mode, "sparsity")

    def test_load_config_maps_top_level_p_plus_grid_into_sweeps(self) -> None:
        # given
        config_dict = {
            "output_dir": "/tmp/output",
            "device": "cpu",
            "data": {
                "seed": 1,
                "num_train": 2,
                "num_val": 1,
                "num_test": 1,
                "input_dim": 3,
                "num_features": 2,
                "activation_prob": 0.5,
                "positive_sign_prob": 0.6,
                "lognormal_mu": 0.0,
                "lognormal_sigma": 0.4,
                "exponential_lambda": 1.2,
                "noise_std": 0.0,
            },
            "models": [
                {
                    "name": "relu",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "single_activation_offset": False,
                    "reset_activation_offsets_on_resample": False,
                    "d_hidden": 2,
                    "sparsity_coeff": 0.01,
                    "learning_rate": 0.1,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "adam_betas": [0.9, 0.999],
                }
            ],
            "p_plus_grid": [0.1, 0.9],
            "sweeps": {"exponential_lambda_grid": [1.0]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            # when
            loaded = config_module.load_experiment_config(config_path)

            # then
            self.assertEqual(loaded.sweeps.p_plus_grid, [0.1, 0.9])
            self.assertEqual(loaded.sweeps.exponential_lambda_grid, [1.0])
            self.assertEqual(loaded.sweeps.sweep_mode, "asymmetry")

    def test_load_config_infers_both_sweep_mode(self) -> None:
        # given
        config_dict = {
            "output_dir": "/tmp/output",
            "device": "cpu",
            "data": {
                "seed": 1,
                "num_train": 2,
                "num_val": 1,
                "num_test": 1,
                "input_dim": 3,
                "num_features": 2,
                "activation_prob": 0.5,
                "positive_sign_prob": 0.6,
                "lognormal_mu": 0.0,
                "lognormal_sigma": 0.4,
                "exponential_lambda": 1.2,
                "noise_std": 0.0,
            },
            "models": [
                {
                    "name": "relu",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "single_activation_offset": False,
                    "reset_activation_offsets_on_resample": False,
                    "d_hidden": 2,
                    "sparsity_coeff": 0.01,
                    "learning_rate": 0.1,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "adam_betas": [0.9, 0.999],
                }
            ],
            "sparsity_coeff_grid": [0.0],
            "p_plus_grid": [0.4],
            "sweeps": {"exponential_lambda_grid": [1.0]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            # when
            loaded = config_module.load_experiment_config(config_path)

            # then
            self.assertEqual(loaded.sweeps.sweep_mode, "both")

    def test_sweep_config_rejects_both_tail_grids(self) -> None:
        # given
        sweeps = {
            "sweep_mode": "asymmetry",
            "p_plus_grid": [0.5],
            "lognormal_sigma_grid": [0.4],
            "exponential_lambda_grid": [1.0],
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.PolarityDialSweepConfig(**sweeps)

    def test_sweep_config_rejects_missing_tail_grid_when_asymmetry_enabled(
        self,
    ) -> None:
        # given
        sweeps = {"sweep_mode": "asymmetry", "p_plus_grid": [0.5]}

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.PolarityDialSweepConfig(**sweeps)

    def test_sweep_config_rejects_missing_sparsity_grid_when_enabled(self) -> None:
        # given
        sweeps = {"sweep_mode": "sparsity"}

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.PolarityDialSweepConfig(**sweeps)

    def test_sweep_config_rejects_negative_sparsity_coeff(self) -> None:
        # given
        sweeps = {"sweep_mode": "sparsity", "sparsity_coeff_grid": [-0.1]}

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.PolarityDialSweepConfig(**sweeps)

    def test_sweep_config_rejects_missing_p_plus_grid_when_enabled(self) -> None:
        # given
        sweeps = {"sweep_mode": "asymmetry", "exponential_lambda_grid": [1.0]}

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.PolarityDialSweepConfig(**sweeps)

    def test_sweep_config_rejects_invalid_lognormal_sigma_grid_value(self) -> None:
        # given
        sweeps = {
            "sweep_mode": "asymmetry",
            "p_plus_grid": [0.5],
            "lognormal_sigma_grid": [0.0],
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.PolarityDialSweepConfig(**sweeps)

    def test_sweep_config_rejects_invalid_exponential_lambda_grid_value(self) -> None:
        # given
        sweeps = {
            "sweep_mode": "asymmetry",
            "p_plus_grid": [0.5],
            "exponential_lambda_grid": [0.0],
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.PolarityDialSweepConfig(**sweeps)

    def test_sweep_config_rejects_invalid_sweep_mode(self) -> None:
        # given
        sweeps = {"sweep_mode": "invalid"}

        # when & then
        with self.assertRaises(ValueError):
            _ = config_module.PolarityDialSweepConfig(**sweeps)

    def test_load_config_rejects_non_object_sweeps(self) -> None:
        # given
        config_dict = {
            "output_dir": "/tmp/output",
            "device": "cpu",
            "data": {
                "seed": 1,
                "num_train": 2,
                "num_val": 1,
                "num_test": 1,
                "input_dim": 3,
                "num_features": 2,
                "activation_prob": 0.5,
                "positive_sign_prob": 0.6,
                "lognormal_mu": 0.0,
                "lognormal_sigma": 0.4,
                "exponential_lambda": 1.2,
                "noise_std": 0.0,
            },
            "models": [
                {
                    "name": "relu",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "single_activation_offset": False,
                    "reset_activation_offsets_on_resample": False,
                    "d_hidden": 2,
                    "sparsity_coeff": 0.01,
                    "learning_rate": 0.1,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "adam_betas": [0.9, 0.999],
                }
            ],
            "sweeps": ["not", "a", "dict"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = config_module.load_experiment_config(config_path)


class TestPolarityDialSweepRunners(unittest.TestCase):
    def test_format_float_for_path_returns_short_stable_string(self) -> None:
        # given
        value = 0.123456789

        # when
        formatted = sweeps_module._format_float_for_path(value)

        # then
        self.assertEqual(formatted, "0.123457")

    def test_write_json_creates_file_without_directory_component(self) -> None:
        # given
        payload = {"ok": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # when
                sweeps_module._write_json("file.json", payload)
            finally:
                os.chdir(original_cwd)

            # then
            with open(os.path.join(tmpdir, "file.json"), "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, payload)

    def test_resolve_tail_grid_raises_when_missing(self) -> None:
        # given
        data_config = config_module.PolarityDialDataConfig(
            seed=1,
            num_train=1,
            num_val=1,
            num_test=1,
            input_dim=2,
            num_features=1,
            activation_prob=0.5,
            positive_sign_prob=0.5,
            lognormal_mu=0.0,
            lognormal_sigma=0.4,
            exponential_lambda=1.2,
            noise_std=0.0,
        )
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="/tmp/output",
            device="cpu",
            data=data_config,
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(
                min_regime_samples=1, max_points_per_regime_scatter=1
            ),
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = sweeps_module._resolve_tail_grid(experiment_config)

    def test_run_sparsity_sweep_writes_point_summaries(self) -> None:
        # given
        data_config = config_module.PolarityDialDataConfig(
            seed=2,
            num_train=1,
            num_val=1,
            num_test=1,
            input_dim=2,
            num_features=1,
            activation_prob=0.5,
            positive_sign_prob=0.5,
            lognormal_mu=0.0,
            lognormal_sigma=0.4,
            exponential_lambda=1.2,
            noise_std=0.0,
        )
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="/tmp/output",
            device="cpu",
            data=data_config,
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(
                min_regime_samples=1, max_points_per_regime_scatter=1
            ),
            sweeps=config_module.PolarityDialSweepConfig(
                sweep_mode="sparsity", sparsity_coeff_grid=[0.0, 0.1]
            ),
        )
        data = {
            "x_test": torch.zeros(1, 2),
            "coefficients_test": torch.zeros(1, 1),
            "feature_directions": torch.zeros(1, 2),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleNamespace(sparsity_coeff=None)

            def fake_run_training_for_variant(
                _experiment_config: Any,
                _data: Any,
                model_config_for_point: Any,
                _device: Any,
            ) -> Dict[str, Any]:
                model.sparsity_coeff = model_config_for_point.sparsity_coeff
                return {"model": model, "training_metrics": {}}

            def fake_calibration(
                trained_model: Any, *_args: Any, **_kwargs: Any
            ) -> Any:
                return (
                    {
                        "summary": {
                            "test_reconstruction_mse": float(
                                trained_model.sparsity_coeff
                            ),
                            "test_l0": float(trained_model.sparsity_coeff) * 10.0,
                            "mean_rel_error_plus": 0.1,
                            "mean_rel_error_minus": 0.2,
                        },
                        "per_pair": [],
                    },
                    {},
                )

            with patch(
                "sgnsae.experiments.polarity_dial.sweeps.run_training_for_variant",
                side_effect=fake_run_training_for_variant,
            ), patch(
                "sgnsae.experiments.polarity_dial.sweeps.align_decoder_and_features",
                return_value=({}, {}, torch.zeros(1, 1)),
            ), patch(
                "sgnsae.experiments.polarity_dial.sweeps.compute_alignment_and_calibration",
                side_effect=fake_calibration,
            ):
                # when
                summary = sweeps_module.run_sparsity_sweep(
                    experiment_config, tmpdir, data, device="cpu"
                )

            # then
            self.assertEqual(summary["sweep_type"], "sparsity")
            self.assertIn("relu", summary["models"])
            self.assertEqual(len(summary["models"]["relu"]), 2)
            sweep_dir = os.path.join(tmpdir, "sweeps", "sparsity")
            for sparsity_coeff in [0.0, 0.1]:
                point_dir = os.path.join(
                    sweep_dir,
                    f"sparsity_coeff={sweeps_module._format_float_for_path(sparsity_coeff)}",
                )
                with open(os.path.join(point_dir, "point_summary.json"), "r") as f:
                    point_summary = json.load(f)
                self.assertIn("relu", point_summary["models"])

    def test_run_sparsity_sweep_raises_when_grid_missing(self) -> None:
        # given
        data_config = config_module.PolarityDialDataConfig(
            seed=2,
            num_train=1,
            num_val=1,
            num_test=1,
            input_dim=2,
            num_features=1,
            activation_prob=0.5,
            positive_sign_prob=0.5,
            lognormal_mu=0.0,
            lognormal_sigma=0.4,
            exponential_lambda=1.2,
            noise_std=0.0,
        )
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="/tmp/output",
            device="cpu",
            data=data_config,
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(
                min_regime_samples=1, max_points_per_regime_scatter=1
            ),
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = sweeps_module.run_sparsity_sweep(
                experiment_config, "/tmp/output", {}, device="cpu"
            )

    def test_run_asymmetry_sweep_writes_exponential_tail_points(self) -> None:
        # given
        data_config = config_module.PolarityDialDataConfig(
            seed=3,
            num_train=1,
            num_val=1,
            num_test=1,
            input_dim=2,
            num_features=1,
            activation_prob=0.5,
            positive_sign_prob=0.5,
            lognormal_mu=0.0,
            lognormal_sigma=0.4,
            exponential_lambda=1.2,
            noise_std=0.0,
        )
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="/tmp/output",
            device="cpu",
            data=data_config,
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(
                min_regime_samples=1, max_points_per_regime_scatter=1
            ),
            sweeps=config_module.PolarityDialSweepConfig(
                sweep_mode="asymmetry",
                p_plus_grid=[0.3, 0.7],
                exponential_lambda_grid=[1.5],
            ),
        )
        data = {
            "x_test": torch.zeros(1, 2),
            "coefficients_test": torch.zeros(1, 1),
            "feature_directions": torch.zeros(1, 2),
        }
        data_generator_calls: List[Dict[str, Any]] = []

        def fake_generate_dataset(
            cfg: config_module.PolarityDialDataConfig, _device: str
        ) -> Dict[str, Any]:
            data_generator_calls.append(
                {
                    "positive_sign_prob": cfg.positive_sign_prob,
                    "exponential_lambda": cfg.exponential_lambda,
                }
            )
            return data

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sgnsae.experiments.polarity_dial.sweeps.generate_polarity_dial_dataset",
                side_effect=fake_generate_dataset,
            ), patch(
                "sgnsae.experiments.polarity_dial.sweeps.run_training_for_variant",
                return_value={"model": MagicMock(), "training_metrics": {}},
            ), patch(
                "sgnsae.experiments.polarity_dial.sweeps.align_decoder_and_features",
                return_value=({}, {}, torch.zeros(1, 1)),
            ), patch(
                "sgnsae.experiments.polarity_dial.sweeps.compute_alignment_and_calibration",
                return_value=(
                    {
                        "summary": {
                            "test_reconstruction_mse": 0.0,
                            "test_l0": 0.0,
                            "mean_rel_error_plus": 0.1,
                            "mean_rel_error_minus": 0.2,
                        },
                        "per_pair": [],
                    },
                    {},
                ),
            ):
                # when
                summary = sweeps_module.run_asymmetry_sweep(
                    experiment_config, tmpdir, device="cpu"
                )

            # then
            self.assertEqual(summary["sweep_type"], "asymmetry")
            self.assertIn("exponential_lambda_grid", summary)
            self.assertEqual(len(summary["points"]), 2)
            self.assertEqual(len(data_generator_calls), 2)
            point_dir = os.path.join(
                tmpdir,
                "sweeps",
                "asymmetry",
                "p_plus=0.3__exponential_lambda=1.5",
            )
            with open(os.path.join(point_dir, "generator_config.json"), "r") as f:
                generator_config = json.load(f)
            self.assertEqual(generator_config["positive_sign_prob"], 0.3)
            self.assertEqual(generator_config["exponential_lambda"], 1.5)

    def test_run_asymmetry_sweep_writes_lognormal_tail_points(self) -> None:
        # given
        data_config = config_module.PolarityDialDataConfig(
            seed=3,
            num_train=1,
            num_val=1,
            num_test=1,
            input_dim=2,
            num_features=1,
            activation_prob=0.5,
            positive_sign_prob=0.5,
            lognormal_mu=0.0,
            lognormal_sigma=0.4,
            exponential_lambda=1.2,
            noise_std=0.0,
        )
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="/tmp/output",
            device="cpu",
            data=data_config,
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(
                min_regime_samples=1, max_points_per_regime_scatter=1
            ),
            sweeps=config_module.PolarityDialSweepConfig(
                sweep_mode="asymmetry",
                p_plus_grid=[0.3],
                lognormal_sigma_grid=[0.9],
            ),
        )
        data = {
            "x_test": torch.zeros(1, 2),
            "coefficients_test": torch.zeros(1, 1),
            "feature_directions": torch.zeros(1, 2),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sgnsae.experiments.polarity_dial.sweeps.generate_polarity_dial_dataset",
                return_value=data,
            ), patch(
                "sgnsae.experiments.polarity_dial.sweeps.run_training_for_variant",
                return_value={"model": MagicMock(), "training_metrics": {}},
            ), patch(
                "sgnsae.experiments.polarity_dial.sweeps.align_decoder_and_features",
                return_value=({}, {}, torch.zeros(1, 1)),
            ), patch(
                "sgnsae.experiments.polarity_dial.sweeps.compute_alignment_and_calibration",
                return_value=(
                    {
                        "summary": {
                            "test_reconstruction_mse": 0.0,
                            "test_l0": 0.0,
                            "mean_rel_error_plus": 0.1,
                            "mean_rel_error_minus": 0.2,
                        },
                        "per_pair": [],
                    },
                    {},
                ),
            ):
                # when
                summary = sweeps_module.run_asymmetry_sweep(
                    experiment_config, tmpdir, device="cpu"
                )

            # then
            self.assertIn("lognormal_sigma_grid", summary)
            point_dir = os.path.join(
                tmpdir,
                "sweeps",
                "asymmetry",
                "p_plus=0.3__lognormal_sigma=0.9",
            )
            self.assertTrue(
                os.path.isfile(os.path.join(point_dir, "point_summary.json"))
            )

    def test_run_asymmetry_sweep_raises_when_grid_missing(self) -> None:
        # given
        data_config = config_module.PolarityDialDataConfig(
            seed=3,
            num_train=1,
            num_val=1,
            num_test=1,
            input_dim=2,
            num_features=1,
            activation_prob=0.5,
            positive_sign_prob=0.5,
            lognormal_mu=0.0,
            lognormal_sigma=0.4,
            exponential_lambda=1.2,
            noise_std=0.0,
        )
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="/tmp/output",
            device="cpu",
            data=data_config,
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(
                min_regime_samples=1, max_points_per_regime_scatter=1
            ),
            sweeps=config_module.PolarityDialSweepConfig(sweep_mode="none"),
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = sweeps_module.run_asymmetry_sweep(
                experiment_config, "/tmp/output", device="cpu"
            )


class TestPolarityDialSweepPlotting(unittest.TestCase):
    def test_plot_mse_vs_l0_tradeoff_writes_file(self) -> None:
        # given
        tradeoff_summary = {
            "models": {
                "model_a": [
                    {
                        "sparsity_coeff": 0.01,
                        "test_l0": 2.0,
                        "test_reconstruction_mse": 0.1,
                    },
                    {
                        "sparsity_coeff": None,
                        "test_l0": 1.0,
                        "test_reconstruction_mse": 0.2,
                    },
                ],
                "model_b": [
                    {
                        "sparsity_coeff": 0.1,
                        "test_l0": None,
                        "test_reconstruction_mse": 0.1,
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "mse_vs_l0_tradeoff.png")

            # when
            plotting_module.plot_mse_vs_l0_tradeoff(tradeoff_summary, output_path)

            # then
            self.assertTrue(os.path.isfile(output_path))

    def test_plot_asymmetry_sweep_calibration_writes_file_and_infers_models(
        self,
    ) -> None:
        # given
        asymmetry_summary = {
            "p_plus_grid": [0.3, 0.7],
            "exponential_lambda_grid": [1.5],
            "points": [
                {
                    "positive_sign_prob": 0.3,
                    "exponential_lambda": 1.5,
                    "models": {
                        "model_a": {
                            "mean_rel_error_plus": 0.1,
                            "mean_rel_error_minus": 0.2,
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "asymmetry_sweep_calibration.png")

            # when
            plotting_module.plot_asymmetry_sweep_calibration(
                asymmetry_summary, output_path
            )

            # then
            self.assertTrue(os.path.isfile(output_path))

    def test_plot_asymmetry_sweep_calibration_raises_for_missing_p_plus_grid(
        self,
    ) -> None:
        # given
        asymmetry_summary = {"points": []}

        # when & then
        with self.assertRaises(ValueError):
            plotting_module.plot_asymmetry_sweep_calibration(
                asymmetry_summary, "/tmp/x.png"
            )

    def test_plot_asymmetry_sweep_calibration_raises_for_missing_models(self) -> None:
        # given
        asymmetry_summary = {"p_plus_grid": [0.3], "points": []}

        # when & then
        with self.assertRaises(ValueError):
            plotting_module.plot_asymmetry_sweep_calibration(
                asymmetry_summary, "/tmp/x.png"
            )

    def test_plot_asymmetry_sweep_calibration_raises_for_missing_tail_grid(
        self,
    ) -> None:
        # given
        asymmetry_summary = {"p_plus_grid": [0.3], "models": ["model_a"], "points": []}

        # when & then
        with self.assertRaises(ValueError):
            plotting_module.plot_asymmetry_sweep_calibration(
                asymmetry_summary, "/tmp/x.png"
            )

    def test_plot_asymmetry_sweep_calibration_raises_for_empty_tail_grid(self) -> None:
        # given
        asymmetry_summary = {
            "p_plus_grid": [0.3],
            "models": ["model_a"],
            "lognormal_sigma_grid": [],
            "points": [],
        }

        # when & then
        with self.assertRaises(ValueError):
            plotting_module.plot_asymmetry_sweep_calibration(
                asymmetry_summary, "/tmp/x.png"
            )


class TestPolarityDialRunnerSweepIntegration(unittest.TestCase):
    def test_run_polarity_dial_experiment_writes_sweep_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config = {
                "output_dir": tmpdir,
                "device": "cpu",
                "data": {
                    "seed": 11,
                    "num_train": 4,
                    "num_val": 2,
                    "num_test": 2,
                    "input_dim": 6,
                    "num_features": 3,
                    "activation_prob": 0.5,
                    "positive_sign_prob": 0.6,
                    "lognormal_mu": 0.0,
                    "lognormal_sigma": 0.4,
                    "exponential_lambda": 1.2,
                    "noise_std": 0.05,
                },
                "models": [
                    {
                        "name": "relu_sae_smoke",
                        "sae_class": "SAE",
                        "sign_aware": False,
                        "single_activation_offset": False,
                        "reset_activation_offsets_on_resample": False,
                        "d_hidden": 4,
                        "sparsity_coeff": 0.0005,
                        "learning_rate": 0.001,
                        "num_epochs": 1,
                        "batch_size": 2,
                        "adam_betas": [0.9, 0.999],
                    }
                ],
                "verbose": False,
                "sample_preview_count": 2,
                "max_scatter_points_per_regime": 50,
                "sparsity_coeff_grid": [0.001, 0.01],
                "p_plus_grid": [0.4, 0.6],
                "sweeps": {"exponential_lambda_grid": [1.5]},
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            def fake_run_single_model(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
                return {
                    "model_dir": os.path.join(tmpdir, "models", "relu_sae_smoke"),
                    "calibration": {
                        "summary": {
                            "mean_rel_error_plus": 0.0,
                            "mean_rel_error_minus": 0.0,
                        }
                    },
                    "metadata": {
                        "d_hidden": 4,
                        "d_hidden_effective": 4,
                        "parameter_count": 1,
                    },
                }

            def fake_plot_to_path(_payload: Any, output_path: str) -> None:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(b"ok")

            with patch(
                "sgnsae.experiments.polarity_dial.runner._run_single_model",
                side_effect=fake_run_single_model,
            ), patch(
                "sgnsae.experiments.polarity_dial.runner.plot_gamma_comparison",
                side_effect=fake_plot_to_path,
            ), patch(
                "sgnsae.experiments.polarity_dial.runner.run_sparsity_sweep",
                return_value={"models": {}},
            ), patch(
                "sgnsae.experiments.polarity_dial.runner.plot_mse_vs_l0_tradeoff",
                side_effect=fake_plot_to_path,
            ), patch(
                "sgnsae.experiments.polarity_dial.runner.run_asymmetry_sweep",
                return_value={
                    "p_plus_grid": [0.4, 0.6],
                    "exponential_lambda_grid": [1.5],
                    "points": [],
                    "models": [],
                },
            ), patch(
                "sgnsae.experiments.polarity_dial.runner.plot_asymmetry_sweep_calibration",
                side_effect=fake_plot_to_path,
            ):
                # when
                run_dir = runner_module.run_polarity_dial_experiment(config_path)

            # then
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "mse_vs_l0_tradeoff.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "mse_vs_l0_tradeoff.png"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "asymmetry_sweep_summary.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "asymmetry_sweep_calibration.png"))
            )


if __name__ == "__main__":
    unittest.main()
