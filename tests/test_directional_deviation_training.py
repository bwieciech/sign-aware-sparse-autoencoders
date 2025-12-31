import io
import json
import os
import tempfile
import unittest
from typing import Dict
from unittest.mock import patch

import torch

from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationAnomalyConfig,
    DirectionalDeviationDataConfig,
    DirectionalDeviationExperimentConfig,
    DirectionalDeviationModelVariantConfig,
    DirectionalDeviationThresholdConfig,
)
from sgnsae.experiments.directional_deviation.data import generate_normal_data
from sgnsae.experiments.directional_deviation.training import (
    _instantiate_model,
    _log_epoch_summary,
    _log_step_progress,
    _prepare_training_tensor,
    _select_subset_indices,
    persist_model_artifacts,
    run_training_for_variant,
)
from sgnsae.models import GatedSAE, SAE


class TestDirectionalDeviationTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"

    def _build_experiment_config(
        self,
        data_config: DirectionalDeviationDataConfig,
        anomaly_config: DirectionalDeviationAnomalyConfig,
        model_config: DirectionalDeviationModelVariantConfig,
    ) -> DirectionalDeviationExperimentConfig:
        threshold_config = DirectionalDeviationThresholdConfig(
            latent_quantile=0.99,
            score_fpr=0.01,
            min_validation_samples=1,
            deterministic_algorithms=False,
        )
        return DirectionalDeviationExperimentConfig(
            output_dir="/tmp/output",
            device="cpu",  # type: ignore[arg-type]
            data=data_config,
            anomaly=anomaly_config,
            thresholds=threshold_config,
            models=[model_config],
            verbose=False,
            log_every_n_steps=1,
            log_every_n_epochs=1,
        )

    def test_instantiate_model_builds_expected_model_class(self) -> None:
        # given
        sae_config = DirectionalDeviationModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=3,
            sparsity_coeff=0.01,
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        gated_config = DirectionalDeviationModelVariantConfig(
            name="gated_sae",
            sae_class="GatedSAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=3,
            sparsity_coeff=0.01,
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )

        # when
        sae_model = _instantiate_model(
            sae_config, input_dim=5, steps_per_epoch=7, device=self.device
        )
        gated_model = _instantiate_model(
            gated_config, input_dim=5, steps_per_epoch=7, device=self.device
        )

        # then
        self.assertIsInstance(sae_model, SAE)
        self.assertIsInstance(gated_model, GatedSAE)
        self.assertEqual(sae_model.d_hidden, 3)
        self.assertEqual(gated_model.d_hidden, 3)

    def test_instantiate_model_rejects_unknown_class(self) -> None:
        # given
        model_config = DirectionalDeviationModelVariantConfig(
            name="unknown",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.01,
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        model_config.sae_class = "OtherModel"

        # when & then
        with self.assertRaises(ValueError):
            _ = _instantiate_model(
                model_config, input_dim=5, steps_per_epoch=1, device=self.device
            )

    def test_prepare_training_tensor_noop_when_fraction_zero(self) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=6,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
            train_anomaly_fraction=0.0,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=3.0, shift_mode="single_channel"
        )
        x_train = generate_normal_data(data_config, self.device)["x_train"]

        # when
        mixed, anomaly_count, anomaly_seed = _prepare_training_tensor(
            x_train=x_train,
            data_config=data_config,
            anomaly_config=anomaly_config,
        )

        # then
        self.assertTrue(torch.equal(mixed, x_train))
        self.assertEqual(anomaly_count, 0)
        self.assertIsInstance(anomaly_seed, int)

    def test_prepare_training_tensor_single_channel_injects_one_coordinate_per_anomaly_row(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=10,
            num_train=8,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=5,
            train_anomaly_fraction=0.5,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=2.0, shift_mode="single_channel", channels_to_evaluate=[1, 3]
        )
        x_train = generate_normal_data(data_config, self.device)["x_train"]

        # when
        mixed, anomaly_count, _ = _prepare_training_tensor(
            x_train=x_train, data_config=data_config, anomaly_config=anomaly_config
        )

        # then
        expected_anomaly_count = int(
            round(data_config.num_train * data_config.train_anomaly_fraction)
        )
        self.assertEqual(anomaly_count, expected_anomaly_count)

        diff = mixed - x_train
        changed_rows = (diff.abs() > 1e-6).any(dim=1)
        self.assertEqual(int(changed_rows.sum().item()), anomaly_count)

        changed_diff = diff[changed_rows]
        non_zero_counts = (changed_diff.abs() > 1e-6).sum(dim=1)
        self.assertTrue(bool((non_zero_counts == 1).all()))
        magnitudes = changed_diff.abs().sum(dim=1)
        self.assertTrue(
            bool(
                torch.allclose(
                    magnitudes, torch.full_like(magnitudes, anomaly_config.delta)
                )
            )
        )

    def test_prepare_training_tensor_subset_injects_only_subset_coordinates(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=22,
            num_train=10,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=7,
            train_anomaly_fraction=0.4,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.5,
            shift_mode="subset",
            subset_size=2,
            channels_to_evaluate=[0, 2, 4, 6],
        )
        x_train = generate_normal_data(data_config, self.device)["x_train"]
        expected_subset = _select_subset_indices(
            data_config, anomaly_config, data_config.seed + 3
        )
        subset_tensor = torch.tensor(expected_subset, dtype=torch.int64)

        # when
        mixed, anomaly_count, _ = _prepare_training_tensor(
            x_train=x_train, data_config=data_config, anomaly_config=anomaly_config
        )

        # then
        diff = mixed - x_train
        changed_rows = (diff.abs() > 1e-6).any(dim=1)
        self.assertEqual(int(changed_rows.sum().item()), anomaly_count)

        changed_diff = diff[changed_rows]
        non_zero_counts = (changed_diff.abs() > 1e-6).sum(dim=1)
        self.assertTrue(bool((non_zero_counts == anomaly_config.subset_size).all()))

        other_mask = torch.ones((data_config.num_channels,), dtype=torch.bool)
        other_mask[subset_tensor] = False
        self.assertTrue(bool((changed_diff[:, other_mask] == 0.0).all()))

        subset_diff_abs = changed_diff[:, subset_tensor].abs()
        self.assertTrue(
            bool(
                torch.allclose(
                    subset_diff_abs,
                    torch.full_like(subset_diff_abs, anomaly_config.delta),
                )
            )
        )

    def test_select_subset_indices_rejects_wrong_shift_mode(self) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = _select_subset_indices(data_config, anomaly_config, seed=0)

    def test_select_subset_indices_rejects_missing_subset_size(self) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="subset", subset_size=1
        )
        anomaly_config.subset_size = None

        # when & then
        with self.assertRaises(ValueError):
            _ = _select_subset_indices(data_config, anomaly_config, seed=0)

    def test_select_subset_indices_rejects_subset_size_greater_than_candidates(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=6,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0,
            shift_mode="subset",
            subset_size=4,
            channels_to_evaluate=[0, 1],
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = _select_subset_indices(data_config, anomaly_config, seed=0)

    def test_prepare_training_tensor_rounds_small_fraction_to_zero_anomalies(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=10,
            num_train=1,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=3,
            train_anomaly_fraction=0.1,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=2.0, shift_mode="single_channel"
        )
        x_train = generate_normal_data(data_config, self.device)["x_train"]

        # when
        mixed, anomaly_count, _ = _prepare_training_tensor(
            x_train=x_train, data_config=data_config, anomaly_config=anomaly_config
        )

        # then
        self.assertTrue(torch.equal(mixed, x_train))
        self.assertEqual(anomaly_count, 0)

    def test_log_step_progress_writes_expected_output(self) -> None:
        # given
        losses_dict = {
            "losses": torch.tensor([1.0]),
            "reconstruction_losses": torch.tensor([0.5]),
            "sparsity_losses": torch.tensor([0.25]),
        }
        stdout = io.StringIO()

        # when
        with patch(
            "sgnsae.experiments.directional_deviation.training.sys.stdout", new=stdout
        ):
            _log_step_progress(
                epoch=0,
                num_epochs=2,
                step=0,
                total_steps=3,
                losses_dict=losses_dict,
                mean_active_latents=1.5,
                model_name="model",
            )

        # then
        output = stdout.getvalue()
        self.assertIn("[model] Epoch 1/2", output)
        self.assertIn("Step 1/3", output)

    def test_log_epoch_summary_includes_aux_when_present(self) -> None:
        # given
        model_config = DirectionalDeviationModelVariantConfig(
            name="model",
            sae_class="GatedSAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        losses_without_aux = {
            "losses": 1.0,
            "reconstruction_losses": 0.5,
            "sparsity_losses": 0.25,
        }
        losses_with_aux = {**losses_without_aux, "aux_loss": 0.125}

        # when & then
        with patch("builtins.print") as mock_print:
            _log_epoch_summary(0, losses_without_aux, model_config)
            _log_epoch_summary(0, losses_with_aux, model_config)
            printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
            self.assertIn("Loss 1.000000", printed)
            self.assertIn("Aux 0.125000", printed)

    def test_run_training_for_variant_returns_metrics_and_persists_artifacts(
        self,
    ) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=16,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=6,
            train_anomaly_fraction=0.25,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=2.0, shift_mode="single_channel"
        )
        model_config = DirectionalDeviationModelVariantConfig(
            name="gated_sae",
            sae_class="GatedSAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=3,
            sparsity_coeff=0.01,
            learning_rate=1e-3,
            num_epochs=2,
            batch_size=4,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = self._build_experiment_config(
            data_config, anomaly_config, model_config
        )
        normal_data = generate_normal_data(data_config, self.device)

        # when
        output = run_training_for_variant(
            experiment_config, normal_data, model_config, self.device
        )

        # then
        training_metrics = output["training_metrics"]
        self.assertIn("loss_total", training_metrics)
        self.assertIn("loss_reconstruction", training_metrics)
        self.assertIn("loss_sparsity", training_metrics)
        self.assertIn("mean_active_latents", training_metrics)
        self.assertIn("loss_aux", training_metrics)

        self.assertEqual(len(training_metrics["loss_total"]), model_config.num_epochs)
        self.assertEqual(len(training_metrics["loss_aux"]), model_config.num_epochs)
        self.assertEqual(
            training_metrics["train_anomaly_count"],
            int(round(data_config.num_train * data_config.train_anomaly_fraction)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_model_artifacts(
                model=output["model"],
                training_metrics=training_metrics,
                model_config=model_config,
                model_dir=tmpdir,
            )

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "model.pt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "model_config.json")))
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "training_metrics.json"))
            )

            with open(os.path.join(tmpdir, "model_config.json"), "r") as f:
                saved_config = json.load(f)
            self.assertEqual(saved_config["name"], "gated_sae")

            with open(os.path.join(tmpdir, "training_metrics.json"), "r") as f:
                saved_metrics = json.load(f)
            self.assertEqual(len(saved_metrics["loss_total"]), model_config.num_epochs)

    def test_run_training_for_variant_sets_terminated_early_on_nan_losses(self) -> None:
        # given
        class NaNLossModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {}

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                return {
                    "losses": torch.tensor([float("nan")], dtype=torch.float32),
                    "reconstruction_losses": torch.tensor([0.0], dtype=torch.float32),
                    "sparsity_losses": torch.tensor([0.0], dtype=torch.float32),
                }

        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=4,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )
        model_config = DirectionalDeviationModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = self._build_experiment_config(
            data_config, anomaly_config, model_config
        )
        normal_data = generate_normal_data(data_config, self.device)

        # when
        with patch(
            "sgnsae.experiments.directional_deviation.training._instantiate_model",
            return_value=NaNLossModel(),
        ), patch("builtins.print"):
            output = run_training_for_variant(
                experiment_config, normal_data, model_config, self.device
            )

        # then
        self.assertTrue(output["training_metrics"]["terminated_early"])

    def test_run_training_for_variant_verbose_calls_logging_helpers(self) -> None:
        # given
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=4,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )
        model_config = DirectionalDeviationModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=4,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = self._build_experiment_config(
            data_config, anomaly_config, model_config
        )
        experiment_config.verbose = True
        normal_data = generate_normal_data(data_config, self.device)

        # when
        with patch(
            "sgnsae.experiments.directional_deviation.training._log_step_progress"
        ) as mock_step, patch(
            "sgnsae.experiments.directional_deviation.training._log_epoch_summary"
        ) as mock_epoch:
            _ = run_training_for_variant(
                experiment_config, normal_data, model_config, self.device
            )

        # then
        self.assertTrue(mock_step.called)
        self.assertTrue(mock_epoch.called)

    def test_run_training_for_variant_raises_when_training_tensor_is_empty(
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
            num_channels=4,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )
        model_config = DirectionalDeviationModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = self._build_experiment_config(
            data_config, anomaly_config, model_config
        )
        empty_data = {
            "x_train": torch.empty((0, data_config.num_channels), dtype=torch.float32),
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = run_training_for_variant(
                experiment_config, empty_data, model_config, self.device
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
