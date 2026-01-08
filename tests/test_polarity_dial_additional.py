import io
import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import torch

from sgnsae.experiments.polarity_dial import config as config_module
from sgnsae.experiments.polarity_dial import data as data_module
from sgnsae.experiments.polarity_dial import evaluation as evaluation_module
from sgnsae.experiments.polarity_dial import plotting as plotting_module
from sgnsae.experiments.polarity_dial import runner as runner_module
from sgnsae.experiments.polarity_dial import training as training_module
from sgnsae.models import GatedSAE, SAE, SAEBase, SignedSoftThresholdSAE


class TestPolarityDialConfigValidation(unittest.TestCase):
    def test_validate_probability_raises_for_out_of_range(self) -> None:
        # given
        value = 1.5

        # when & then
        with self.assertRaises(ValueError):
            config_module._validate_probability(value, "activation_prob")

    def test_data_config_rejects_invalid_scales(self) -> None:
        # given
        base_kwargs = {
            "seed": 1,
            "num_train": 2,
            "num_val": 1,
            "num_test": 1,
            "input_dim": 2,
            "num_features": 1,
            "activation_prob": 0.5,
            "positive_sign_prob": 0.5,
            "lognormal_mu": 0.0,
            "lognormal_sigma": 0.5,
            "exponential_lambda": 1.0,
            "noise_std": 0.0,
        }

        # when & then
        with self.assertRaises(ValueError):
            config_module.PolarityDialDataConfig(
                **{**base_kwargs, "lognormal_sigma": 0.0}
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialDataConfig(
                **{**base_kwargs, "exponential_lambda": 0.0}
            )

    def test_validate_positive_int_raises_for_non_positive(self) -> None:
        # given
        value = 0

        # when & then
        with self.assertRaises(ValueError):
            config_module._validate_positive_int(value, "num_train")

    def test_validate_non_negative_raises_for_negative(self) -> None:
        # given
        value = -0.1

        # when & then
        with self.assertRaises(ValueError):
            config_module._validate_non_negative(value, "noise_std")

    def test_validate_adam_betas_raises_for_invalid(self) -> None:
        # given
        bad_length = (0.9,)
        bad_value = (0.0, 0.5)

        # when & then
        with self.assertRaises(ValueError):
            config_module._validate_adam_betas(bad_length)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            config_module._validate_adam_betas(bad_value)

    def test_load_config_from_yaml_and_save_round_trip(self) -> None:
        # given
        yaml_text = "\n".join(
            [
                "output_dir: /tmp/output",
                "device: cpu",
                "data:",
                "  seed: 1",
                "  num_train: 2",
                "  num_val: 1",
                "  num_test: 1",
                "  input_dim: 3",
                "  num_features: 2",
                "  activation_prob: 0.5",
                "  positive_sign_prob: 0.6",
                "  lognormal_mu: 0.0",
                "  lognormal_sigma: 0.4",
                "  exponential_lambda: 1.2",
                "  noise_std: 0.0",
                "models:",
                "  - name: relu",
                "    sae_class: SAE",
                "    sign_aware: false",
                "    single_activation_offset: false",
                "    reset_activation_offsets_on_resample: false",
                "    d_hidden: 2",
                "    sparsity_coeff: 0.01",
                "    learning_rate: 0.1",
                "    num_epochs: 1",
                "    batch_size: 2",
                "    adam_betas: [0.9, 0.999]",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            output_path = os.path.join(tmpdir, "saved.json")
            with open(config_path, "w") as f:
                f.write(yaml_text)

            # when
            loaded = config_module.load_experiment_config(config_path)
            config_module.save_experiment_config(loaded, output_path)

            # then
            self.assertEqual(loaded.device, "cpu")
            self.assertEqual(loaded.data.num_features, 2)
            with open(output_path, "r") as f:
                serialized = json.load(f)
            self.assertEqual(serialized["output_dir"], "/tmp/output")
            self.assertEqual(serialized["data"]["activation_prob"], 0.5)

    def test_load_config_rejects_missing_sections(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"output_dir": "x", "models": []}, f)

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(config_path)

    def test_load_config_requires_yaml_dependency(self) -> None:
        # given
        yaml_text = "\n".join(
            [
                "output_dir: /tmp/output",
                "device: cpu",
                "data: {}",
                "models: []",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write(yaml_text)

            # when & then
            with patch.object(config_module, "yaml", None):
                with self.assertRaises(ValueError):
                    config_module.load_experiment_config(config_path)

    def test_load_config_rejects_missing_output_dir_and_models_type(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            path_missing_output = os.path.join(tmpdir, "missing_output.json")
            with open(path_missing_output, "w") as f:
                json.dump({"data": {}, "models": []}, f)
            path_models_not_list = os.path.join(tmpdir, "models_not_list.json")
            with open(path_models_not_list, "w") as f:
                json.dump({"output_dir": "x", "data": {}, "models": {}}, f)

            # when & then
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(path_missing_output)
            with self.assertRaises(ValueError):
                config_module.load_experiment_config(path_models_not_list)

    def test_model_config_validation_errors(self) -> None:
        # given
        base_kwargs = {
            "sae_class": "SAE",
            "sign_aware": False,
            "single_activation_offset": False,
            "reset_activation_offsets_on_resample": False,
            "d_hidden": 2,
            "sparsity_coeff": 0.0,
            "learning_rate": 0.01,
            "num_epochs": 1,
            "batch_size": 2,
            "adam_betas": (0.9, 0.999),
        }

        # when & then
        with self.assertRaises(ValueError):
            config_module.PolarityDialModelVariantConfig(name="", **base_kwargs)
        with self.assertRaises(ValueError):
            config_module.PolarityDialModelVariantConfig(
                name="model", **{**base_kwargs, "d_hidden": 0}
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialModelVariantConfig(
                name="model", **{**base_kwargs, "num_epochs": 0}
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialModelVariantConfig(
                name="model", **{**base_kwargs, "batch_size": 0}
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialModelVariantConfig(
                name="model", **{**base_kwargs, "sparsity_coeff": -0.1}
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialModelVariantConfig(
                name="model", **{**base_kwargs, "learning_rate": 0.0}
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialModelVariantConfig(
                name="model", **{**base_kwargs, "resample_latents_epochs": 0.0}
            )

    def test_model_config_rejects_symmetric_magnitude_without_sign_aware(self) -> None:
        # given
        base_kwargs = {
            "sae_class": "SAE",
            "sign_aware": False,
            "single_activation_offset": False,
            "reset_activation_offsets_on_resample": False,
            "d_hidden": 2,
            "sparsity_coeff": 0.0,
            "learning_rate": 0.01,
            "num_epochs": 1,
            "batch_size": 2,
            "adam_betas": (0.9, 0.999),
            "symmetric_magnitude": True,
        }

        # when & then
        with self.assertRaises(ValueError):
            config_module.PolarityDialModelVariantConfig(name="model", **base_kwargs)

    def test_evaluation_config_rejects_unknown_alignment_mode(self) -> None:
        # given
        alignment_mode = "unsupported"

        # when & then
        with self.assertRaises(ValueError):
            config_module.PolarityDialEvaluationConfig(alignment_mode=alignment_mode)

    def test_load_config_rejects_non_object_evaluation(self) -> None:
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
            "evaluation": ["not", "a", "dict"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = config_module.load_experiment_config(config_path)

    def test_experiment_config_validation_errors(self) -> None:
        # given
        data_config = config_module.PolarityDialDataConfig(
            seed=1,
            num_train=2,
            num_val=1,
            num_test=1,
            input_dim=2,
            num_features=1,
            activation_prob=0.5,
            positive_sign_prob=0.5,
            lognormal_mu=0.0,
            lognormal_sigma=0.5,
            exponential_lambda=1.0,
            noise_std=0.0,
        )
        model_config = config_module.PolarityDialModelVariantConfig(
            name="model",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )

        # when & then
        with self.assertRaises(ValueError):
            config_module.PolarityDialExperimentConfig(
                output_dir="",
                device="cpu",
                data=data_config,
                models=[model_config],
                evaluation=config_module.PolarityDialEvaluationConfig(),
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialExperimentConfig(
                output_dir="out",
                device="invalid",
                data=data_config,
                models=[model_config],
                evaluation=config_module.PolarityDialEvaluationConfig(),
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialExperimentConfig(
                output_dir="out",
                device="cpu",
                data=data_config,
                models=[],
                evaluation=config_module.PolarityDialEvaluationConfig(),
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialExperimentConfig(
                output_dir="out",
                device="cpu",
                data=data_config,
                models=[model_config],
                evaluation=config_module.PolarityDialEvaluationConfig(),
                log_every_n_steps=0,
            )
        with self.assertRaises(ValueError):
            config_module.PolarityDialExperimentConfig(
                output_dir="out",
                device="cpu",
                data=data_config,
                models=[model_config],
                evaluation=config_module.PolarityDialEvaluationConfig(),
                log_every_n_epochs=0,
            )


class TestPolarityDialDataStats(unittest.TestCase):
    def test_tensor_stats_handles_empty_and_values(self) -> None:
        # given
        empty = torch.tensor([])
        values = torch.tensor([1.0, 3.0, 5.0])

        # when
        empty_result = data_module._tensor_stats(empty)
        values_result = data_module._tensor_stats(values)

        # then
        self.assertEqual(empty_result["count"], 0)
        self.assertIsNone(empty_result["mean"])
        self.assertEqual(values_result["count"], 3)
        self.assertAlmostEqual(values_result["mean"], 3.0)
        self.assertAlmostEqual(values_result["min"], 1.0)
        self.assertAlmostEqual(values_result["max"], 5.0)

    def test_collect_coefficients_stats_numeric(self) -> None:
        # given
        coefficients = torch.tensor(
            [
                [1.0, 0.0],
                [-2.0, 0.0],
                [0.0, 3.0],
            ]
        )

        # when
        stats = data_module._collect_coefficients_stats(coefficients)

        # then
        overall = stats["overall"]
        self.assertAlmostEqual(overall["positive"]["mean"], 2.0)
        self.assertAlmostEqual(overall["negative"]["mean"], 2.0)
        self.assertAlmostEqual(overall["sparsity"], 0.5)


class DummyModelNoDecoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"latents": x}


class TestPolarityDialEvaluationUtilities(unittest.TestCase):
    def test_get_decoder_directions_requires_decoder(self) -> None:
        # given
        model = DummyModelNoDecoder()

        # when & then
        model.forward(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            evaluation_module._get_decoder_directions(model, normalize=True)

    def test_compute_gamma_handles_empty_and_zero_denominator(self) -> None:
        # given
        true_values = torch.tensor([1.0, 2.0])
        predicted_zero = torch.zeros(2)
        predicted = torch.tensor([1.0, 2.0])

        # when
        gamma_empty = evaluation_module._compute_gamma(
            torch.tensor([]), torch.tensor([]), eps=1e-8, min_count=1
        )
        gamma_zero = evaluation_module._compute_gamma(
            true_values, predicted_zero, eps=1e-8, min_count=1
        )
        gamma_value = evaluation_module._compute_gamma(
            true_values, predicted, eps=1e-8, min_count=1
        )

        # then
        self.assertIsNone(gamma_empty)
        self.assertAlmostEqual(gamma_zero, 0.0)
        self.assertAlmostEqual(gamma_value, 1.0)

        # when
        gamma_zero_denominator = evaluation_module._compute_gamma(
            true_values, predicted_zero, eps=0.0, min_count=1
        )

        # then
        self.assertIsNone(gamma_zero_denominator)

    def test_sample_points_budget_zero(self) -> None:
        # given
        values = torch.tensor([1.0, 2.0])

        # when
        true_values, predicted_values = evaluation_module._sample_points(
            values, values, remaining_budget=0
        )

        # then
        self.assertEqual(true_values, [])
        self.assertEqual(predicted_values, [])

    def test_safe_mean_and_std(self) -> None:
        # given
        values = [1.0, None, 3.0]
        single = [2.0]

        # when
        mean_value = evaluation_module._safe_mean(values)
        std_none = evaluation_module._safe_std(single)
        std_value = evaluation_module._safe_std(values)

        # then
        self.assertAlmostEqual(mean_value, 2.0)
        self.assertIsNone(std_none)
        self.assertAlmostEqual(std_value, 1.41421356237, places=5)

    def test_sample_points_truncates_to_budget(self) -> None:
        # given
        torch.manual_seed(0)
        values = torch.arange(10, dtype=torch.float32)

        # when
        true_values, predicted_values = evaluation_module._sample_points(
            values, values + 1.0, remaining_budget=3
        )

        # then
        self.assertEqual(len(true_values), 3)
        self.assertEqual(len(predicted_values), 3)
        for original, predicted in zip(true_values, predicted_values):
            self.assertAlmostEqual(predicted, original + 1.0)

    def test_safe_mean_returns_none_when_empty(self) -> None:
        # given
        values: list[float] = []

        # when
        mean_value = evaluation_module._safe_mean(values)

        # then
        self.assertIsNone(mean_value)

    def test_calculate_per_pair_metrics_numeric(self) -> None:
        # given
        true_values = torch.tensor([2.0, -2.0, 0.0])
        predicted = torch.tensor([2.0, -2.0, 1.0])

        # when
        metrics = evaluation_module._calculate_per_axis_metrics(
            true_values,
            predicted,
            eps=1e-8,
            min_regime_samples=1,
        )

        # then
        self.assertAlmostEqual(metrics["gamma_plus"], 1.0)
        self.assertAlmostEqual(metrics["gamma_minus"], 1.0)
        self.assertEqual(metrics["n_pos"], 1)
        self.assertEqual(metrics["n_neg"], 1)

    def test_compute_pearson_r_returns_none_when_denominator_zero(self) -> None:
        # given
        true_values = torch.tensor([1.0, 1.0])
        predicted_values = torch.tensor([2.0, 2.0])

        # when
        pearson_r = evaluation_module._compute_pearson_r(
            true_values, predicted_values, eps=0.0
        )

        # then
        self.assertIsNone(pearson_r)

    def test_compute_alignment_and_calibration_raises_without_reconstruction(
        self,
    ) -> None:
        # given
        evaluation_config = config_module.PolarityDialEvaluationConfig(
            min_regime_samples=1,
            max_points_per_regime_scatter=10,
        )

        class ModelWithoutReconstruction(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": torch.zeros(1, x.shape[0], 1)}

        model = ModelWithoutReconstruction()
        x_test = torch.zeros(2, 3)
        coefficients_test = torch.zeros(2, 1)

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluation_module.compute_alignment_and_calibration(
                model,
                x_test,
                coefficients_test,
                alignment={"matched_pairs": []},
                axis_alignment={"axes": []},
                evaluation_config=evaluation_config,
            )


class TestPolarityDialPlottingUtilities(unittest.TestCase):
    def test_plot_gamma_strip_handles_missing_values(self) -> None:
        # given
        calibration = {
            "per_pair": [
                {"gamma_plus": None, "gamma_minus": None},
                {"gamma_plus": 1.0, "gamma_minus": None},
                {"gamma_plus": 1.2, "gamma_minus": 0.8},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")

            # when
            plotting_module.plot_gamma_strip(calibration, "model", model_dir)

            # then
            plot_path = os.path.join(
                model_dir, "plots", "gamma_plus_minus_strip_model.png"
            )
            self.assertTrue(os.path.isfile(plot_path))

    def test_plotting_outputs_are_created(self) -> None:
        # given
        scatter_data = {
            "positive_pred": [1.0],
            "positive_true": [1.0],
            "negative_pred": [-1.0],
            "negative_true": [-1.0],
        }
        calibration = {
            "per_pair": [
                {"gamma_plus": 1.0, "gamma_minus": 1.0},
            ],
            "summary": {
                "mean_rel_error_plus": 0.1,
                "mean_rel_error_minus": 0.2,
                "std_rel_error_plus": 0.05,
                "std_rel_error_minus": 0.04,
            },
        }
        training_metrics = {
            "epoch": [1, 2],
            "loss_total": [1.0, 0.5],
            "loss_reconstruction": [0.8, 0.4],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            comparison_path = os.path.join(tmpdir, "comparison.png")

            # when
            plotting_module.plot_calibration_scatter(scatter_data, "model", model_dir)
            plotting_module.plot_gamma_histograms(calibration, "model", model_dir)
            plotting_module.plot_training_loss(training_metrics, "model", model_dir)
            plotting_module.plot_gamma_comparison(
                {"model": calibration["summary"]}, comparison_path
            )

            # then
            plots_dir = os.path.join(model_dir, "plots")
            self.assertTrue(
                os.path.isfile(os.path.join(plots_dir, "calibration_scatter_model.png"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(plots_dir, "gamma_hist_model.png"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(plots_dir, "training_loss_model.png"))
            )
            self.assertTrue(os.path.isfile(comparison_path))


class TestPolarityDialTrainingUtilities(unittest.TestCase):
    def test_instantiate_model_rejects_unknown_class(self) -> None:
        # given
        config = config_module.PolarityDialModelVariantConfig(
            name="unknown",
            sae_class="NotAClass",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )

        # when & then
        with self.assertRaises(ValueError):
            training_module._instantiate_model(
                config, input_dim=2, steps_per_epoch=1, device="cpu"
            )

    def test_instantiate_model_supports_signed_soft_threshold(self) -> None:
        # given
        config = config_module.PolarityDialModelVariantConfig(
            name="signed_soft_threshold",
            sae_class="SignedSoftThresholdSAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )

        # when
        model = training_module._instantiate_model(
            config, input_dim=2, steps_per_epoch=2, device="cpu"
        )

        # then
        self.assertIsInstance(model, SignedSoftThresholdSAE)

    def test_log_helpers_emit_expected_text(self) -> None:
        # given
        losses = {
            "losses": torch.tensor([1.0]),
            "reconstruction_losses": torch.tensor([0.5]),
            "sparsity_losses": torch.tensor([0.25]),
        }
        epoch_losses = {
            "losses": 1.0,
            "reconstruction_losses": 0.5,
            "sparsity_losses": 0.25,
            "aux_loss": 0.1,
        }
        model_config = config_module.PolarityDialModelVariantConfig(
            name="model",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        buffer = io.StringIO()

        # when
        with patch("sys.stdout", buffer):
            training_module._log_step_progress(
                epoch=0,
                num_epochs=1,
                step=0,
                total_steps=1,
                losses_dict=losses,
                mean_active_latents=1.0,
                model_name="model",
            )
        with patch("builtins.print") as print_mock:
            training_module._log_epoch_summary(0, epoch_losses, model_config)

        # then
        self.assertIn("Loss 1.000000", buffer.getvalue())
        print_mock.assert_called()

    def test_training_terminates_on_nan_loss(self) -> None:
        # given
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="tmp",
            device="cpu",
            data=config_module.PolarityDialDataConfig(
                seed=1,
                num_train=2,
                num_val=1,
                num_test=1,
                input_dim=2,
                num_features=1,
                activation_prob=0.5,
                positive_sign_prob=0.5,
                lognormal_mu=0.0,
                lognormal_sigma=0.5,
                exponential_lambda=1.0,
                noise_std=0.0,
            ),
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(),
            verbose=False,
        )
        data = {"x_train": torch.ones(2, 2)}

        class NaNModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": torch.zeros(1, x.shape[0], 1)}

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                return {
                    "losses": torch.tensor([float("nan")]),
                    "reconstruction_losses": torch.tensor([0.0]),
                    "sparsity_losses": torch.tensor([0.0]),
                }

            def resample_dead_latents(
                self, inactive_latents_count: torch.Tensor
            ) -> torch.Tensor:
                return torch.zeros_like(inactive_latents_count, dtype=torch.bool)

        # when
        with patch.object(
            training_module, "_instantiate_model", return_value=NaNModel()
        ):
            result = training_module.run_training_for_variant(
                experiment_config, data, model_config, device="cpu"
            )

        # then
        self.assertTrue(result["training_metrics"]["terminated_early"])
        NaNModel().resample_dead_latents(torch.zeros(1, 1))

    def test_persist_model_artifacts_writes_files(self) -> None:
        # given
        model = torch.nn.Linear(2, 2)
        training_metrics = {
            "epoch": [1],
            "loss_total": [1.0],
            "loss_reconstruction": [1.0],
            "loss_sparsity": [0.0],
        }
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")

            # when
            training_module.persist_model_artifacts(
                model, training_metrics, model_config, model_dir
            )

            # then
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "model.pt")))
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "model_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "training_metrics.json"))
            )

    def test_instantiate_model_builds_gated_sae(self) -> None:
        # given
        config = config_module.PolarityDialModelVariantConfig(
            name="gated",
            sae_class="GatedSAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )

        # when
        model = training_module._instantiate_model(
            config, input_dim=2, steps_per_epoch=2, device="cpu"
        )

        # then
        self.assertIsInstance(model, GatedSAE)

    def test_run_training_for_variant_updates_metrics(self) -> None:
        # given
        model_config = config_module.PolarityDialModelVariantConfig(
            name="relu",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="tmp",
            device="cpu",
            data=config_module.PolarityDialDataConfig(
                seed=1,
                num_train=4,
                num_val=1,
                num_test=1,
                input_dim=2,
                num_features=1,
                activation_prob=0.5,
                positive_sign_prob=0.5,
                lognormal_mu=0.0,
                lognormal_sigma=0.5,
                exponential_lambda=1.0,
                noise_std=0.0,
            ),
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(),
            log_every_n_steps=1,
            log_every_n_epochs=1,
            verbose=True,
        )
        data = {
            "x_train": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]])
        }

        # when
        with patch.object(training_module, "_log_step_progress") as log_step_mock:
            result = training_module.run_training_for_variant(
                experiment_config, data, model_config, device="cpu"
            )

        # then
        self.assertEqual(len(result["training_metrics"]["epoch"]), 1)
        self.assertTrue(log_step_mock.called)

    def test_training_records_aux_loss_for_gated_sae(self) -> None:
        # given
        model_config = config_module.PolarityDialModelVariantConfig(
            name="gated",
            sae_class="GatedSAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="tmp",
            device="cpu",
            data=config_module.PolarityDialDataConfig(
                seed=1,
                num_train=4,
                num_val=1,
                num_test=1,
                input_dim=2,
                num_features=1,
                activation_prob=0.5,
                positive_sign_prob=0.5,
                lognormal_mu=0.0,
                lognormal_sigma=0.5,
                exponential_lambda=1.0,
                noise_std=0.0,
            ),
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(),
            log_every_n_steps=1,
            log_every_n_epochs=1,
            verbose=False,
        )
        data = {"x_train": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}

        # when
        result = training_module.run_training_for_variant(
            experiment_config, data, model_config, device="cpu"
        )

        # then
        self.assertIn("loss_aux", result["training_metrics"])


class TestSAEBaseResample(unittest.TestCase):
    def test_resample_dead_latents_updates_weights_and_offsets(self) -> None:
        # given
        torch.manual_seed(0)
        model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )
        inactive_latents_count = torch.tensor([[2, 0]])
        torch.manual_seed(123)
        expected = torch.rand(1, model.d_in)
        expected = 1e-2 * expected / (expected.norm(dim=1, keepdim=True) + 1e-8)
        torch.manual_seed(123)

        # when
        mask = model.resample_dead_latents(inactive_latents_count)

        # then
        self.assertTrue(mask[0, 0])
        torch.testing.assert_close(
            model.W_dec[:, :, 0].squeeze(0), expected[0], atol=0, rtol=0
        )
        torch.testing.assert_close(model.b_enc[0, 0], torch.tensor(0.0), atol=0, rtol=0)
        torch.testing.assert_close(model.act_offset[0, 0], torch.tensor(1e-3))

    def test_resample_no_op_when_no_latents(self) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        inactive_latents_count = torch.zeros(1, 2)

        # when
        mask = model.resample_dead_latents(inactive_latents_count)

        # then
        self.assertFalse(mask.any())

    def test_sign_aware_requires_offsets(self) -> None:
        # given
        with self.assertRaises(ValueError):
            SAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=True,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            SAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                device="cpu",
            )

    def test_resample_resets_dual_offsets(self) -> None:
        # given
        torch.manual_seed(0)
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=1,
            resample_latents_threshold=1,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device="cpu",
        )
        inactive_latents_count = torch.tensor([[2]])
        model.act_offset_left.data.fill_(0.5)
        model.act_offset_right.data.fill_(0.6)

        # when
        model.resample_dead_latents(inactive_latents_count)

        # then
        self.assertAlmostEqual(model.act_offset_left.item(), 1e-3)
        self.assertAlmostEqual(model.act_offset_right.item(), 1e-3)

    def test_base_abstract_methods_raise(self) -> None:
        # given
        class DummyBase(SAEBase):
            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return super().forward(x)

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                return super().calculate_losses(y_true, sparsity_coeffs, output)

        model = DummyBase(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        output = {"y_pred": torch.zeros(1, 1, 2), "latents": torch.zeros(1, 1, 2)}

        # when & then
        with self.assertRaises(NotImplementedError):
            model.forward(torch.zeros(1, 2))
        with self.assertRaises(NotImplementedError):
            model.calculate_losses(torch.zeros(1, 1, 2), torch.tensor([0.0]), output)


class TestGatedSAEBehavior(unittest.TestCase):
    def test_forward_and_losses_are_deterministic(self) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            )  # shared weights => identity dictionary
            model.b_enc.data = torch.zeros(1, 2)
            model.b_dec.data = torch.zeros(1, 2)
            model.log_alpha_gate.data.zero_()
            model.r_mag.data = torch.zeros(1, 2)
            model.r_mag_neg.data.zero_()
            model.b_mag.data = torch.tensor([[0.1, 0.2]])
            model.act_offset.data = torch.tensor([[0.05, 0.05]])
        x = torch.tensor([[2.0, -3.0]])

        # when
        output = model(x)
        losses = model.calculate_losses(x, torch.tensor([0.5]), output)

        # then
        expected_latents = torch.tensor([[[2.1, -3.2]]])
        torch.testing.assert_close(output["latents"], expected_latents, rtol=0, atol=0)
        reconstruction = output["x_hat"]
        torch.testing.assert_close(reconstruction, expected_latents, rtol=0, atol=0)
        expected_recon_loss = (reconstruction - x).pow(2).mean(dim=(-1, -2))
        torch.testing.assert_close(losses["reconstruction_losses"], expected_recon_loss)
        pi_gate = output["pi_gate"]
        delta_positive = torch.relu(model.act_offset).unsqueeze(1)
        delta_negative = delta_positive
        expected_sparsity_source = torch.relu(pi_gate - delta_positive) + torch.relu(
            -pi_gate - delta_negative
        )
        expected_sparsity = expected_sparsity_source.sum(dim=-1).mean(-1)
        torch.testing.assert_close(losses["sparsity_losses"], expected_sparsity)

    def test_forward_rejects_invalid_dimensions(self) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        bad_input = torch.zeros(1, 1, 1, 1)

        # when & then
        with self.assertRaises(ValueError):
            model(bad_input)

    def test_losses_use_relu_when_not_sign_aware(self) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        output = {
            "x_hat": torch.zeros(1, 1, 2),
            "pi_gate": torch.tensor([[[-1.0, 2.0]]]),
            "x_hat_frozen": torch.zeros(1, 1, 2),
        }
        y_true = torch.zeros(1, 1, 2)

        # when
        losses = model.calculate_losses(y_true, torch.tensor([1.0]), output)

        # then
        self.assertAlmostEqual(losses["sparsity_losses"].item(), 2.0)

    def test_sae_forward_rejects_invalid_dimensions(self) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        bad_input = torch.zeros(1, 1, 1, 1)

        # when & then
        with self.assertRaises(ValueError):
            model(bad_input)


class TestPolarityDialRunnerUtilities(unittest.TestCase):
    def test_resolve_device_paths(self) -> None:
        # given
        with patch.object(runner_module.torch.cuda, "is_available", return_value=False):
            buffer = io.StringIO()

            # when
            auto_device = runner_module._resolve_device("auto")
            with patch("builtins.print") as print_mock:
                cuda_device = runner_module._resolve_device("cuda")

            # then
            self.assertEqual(auto_device, "cpu")
            self.assertEqual(cuda_device, "cpu")
            print_mock.assert_called()

    def test_build_comparison_summary_handles_none(self) -> None:
        # given
        calibration_by_model = {
            "a": {
                "summary": {"mean_rel_error_plus": None, "mean_rel_error_minus": None}
            },
            "b": {"summary": {"mean_rel_error_plus": 0.2, "mean_rel_error_minus": 0.1}},
        }

        # when
        summary = runner_module._build_comparison_summary(calibration_by_model)

        # then
        self.assertIsNone(
            summary["relative_to_baseline"]["a"]["delta_mean_rel_error_plus"]
        )
        self.assertIsNone(
            summary["relative_to_baseline"]["b"]["delta_mean_rel_error_minus"]
        )

    def test_log_wandb_artifacts_handles_none(self) -> None:
        # given
        training_metrics = {
            "loss_total": [],
            "loss_reconstruction": [],
            "loss_sparsity": [],
        }
        calibration = {"summary": {"mean_gamma_plus": 1.0}}

        # when & then
        runner_module._log_wandb_artifacts(None, training_metrics, calibration)

    def test_maybe_init_wandb_calls_login_and_init(self) -> None:
        # given
        model_config = config_module.PolarityDialModelVariantConfig(
            name="model",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="out",
            device="cpu",
            data=config_module.PolarityDialDataConfig(
                seed=1,
                num_train=2,
                num_val=1,
                num_test=1,
                input_dim=2,
                num_features=1,
                activation_prob=0.5,
                positive_sign_prob=0.5,
                lognormal_mu=0.0,
                lognormal_sigma=0.5,
                exponential_lambda=1.0,
                noise_std=0.0,
            ),
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(),
            wandb_project="project",
            wandb_api_key="key",
        )
        fake_wandb = MagicMock()
        fake_run = MagicMock()
        fake_wandb.init.return_value = fake_run

        # when
        with patch.object(runner_module, "wandb", fake_wandb):
            run = runner_module._maybe_init_wandb(experiment_config, "model", "run")

        # then
        self.assertEqual(run, fake_run)
        fake_wandb.login.assert_called_with(key="key", relogin=True)

    def test_maybe_init_wandb_returns_none_when_missing_dependency(self) -> None:
        # given
        model_config = config_module.PolarityDialModelVariantConfig(
            name="model",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = config_module.PolarityDialExperimentConfig(
            output_dir="out",
            device="cpu",
            data=config_module.PolarityDialDataConfig(
                seed=1,
                num_train=2,
                num_val=1,
                num_test=1,
                input_dim=2,
                num_features=1,
                activation_prob=0.5,
                positive_sign_prob=0.5,
                lognormal_mu=0.0,
                lognormal_sigma=0.5,
                exponential_lambda=1.0,
                noise_std=0.0,
            ),
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(),
            wandb_project="project",
            wandb_api_key="key",
        )

        # when
        with patch.object(runner_module, "wandb", None):
            run = runner_module._maybe_init_wandb(experiment_config, "model", "run")

        # then
        self.assertIsNone(run)

    def test_persist_run_config_redacts_api_key(self) -> None:
        # given
        model_config = config_module.PolarityDialModelVariantConfig(
            name="model",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        config = config_module.PolarityDialExperimentConfig(
            output_dir="out",
            device="cpu",
            data=config_module.PolarityDialDataConfig(
                seed=1,
                num_train=2,
                num_val=1,
                num_test=1,
                input_dim=2,
                num_features=1,
                activation_prob=0.5,
                positive_sign_prob=0.5,
                lognormal_mu=0.0,
                lognormal_sigma=0.5,
                exponential_lambda=1.0,
                noise_std=0.0,
            ),
            models=[model_config],
            evaluation=config_module.PolarityDialEvaluationConfig(),
            wandb_api_key="secret",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir, exist_ok=True)

            # when
            runner_module._persist_run_config(config, run_dir)

            # then
            with open(os.path.join(run_dir, "config.json"), "r") as f:
                saved = json.load(f)
            self.assertIsNone(saved["wandb_api_key"])

    def test_log_wandb_artifacts_logs_and_finishes(self) -> None:
        # given
        wandb_run = MagicMock()
        training_metrics = {
            "loss_total": [1.0],
            "loss_reconstruction": [0.5],
            "loss_sparsity": [0.1],
        }
        calibration = {"summary": {"mean_gamma_plus": 1.0}}

        # when
        runner_module._log_wandb_artifacts(wandb_run, training_metrics, calibration)

        # then
        wandb_run.log.assert_called()
        wandb_run.finish.assert_called()

    def test_prepare_run_directory_uses_run_id(self) -> None:
        # given
        model_config = config_module.PolarityDialModelVariantConfig(
            name="model",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        data_config = config_module.PolarityDialDataConfig(
            seed=1,
            num_train=2,
            num_val=1,
            num_test=1,
            input_dim=2,
            num_features=1,
            activation_prob=0.5,
            positive_sign_prob=0.5,
            lognormal_mu=0.0,
            lognormal_sigma=0.5,
            exponential_lambda=1.0,
            noise_std=0.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = config_module.PolarityDialExperimentConfig(
                output_dir=tmpdir,
                device="cpu",
                data=data_config,
                models=[model_config],
                evaluation=config_module.PolarityDialEvaluationConfig(),
                run_id="fixed",
            )

            # when
            run_dir, run_id = runner_module._prepare_run_directory(config)

            # then
            self.assertEqual(run_id, "fixed")
            self.assertTrue(run_dir.endswith(os.path.join("polarity_dial", "fixed")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
