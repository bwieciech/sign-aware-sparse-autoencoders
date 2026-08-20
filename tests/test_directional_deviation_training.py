import io
import json
import os
import tempfile
import unittest
from typing import Dict
from unittest.mock import patch

import torch

from sgnsae.experiments.directional_deviation import training as training_module
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

    def test_run_optimizer_warmup_steps_returns_true_on_non_finite_loss(self) -> None:
        # given
        class _NaNWarmupModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {
                    "latents": torch.zeros((1, int(x.shape[0]), 1), device=x.device)
                }

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del y_true
                del sparsity_coeffs
                del output
                return {
                    "losses": torch.tensor([float("nan")], device=self.weight.device),
                    "reconstruction_losses": torch.tensor(
                        [0.0], device=self.weight.device
                    ),
                    "sparsity_losses": torch.tensor([0.0], device=self.weight.device),
                }

        model = _NaNWarmupModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        dataloader = training_module._build_training_dataloader(
            x_train=torch.ones((2, 4), dtype=torch.float32),
            batch_size=1,
            dataloader_seed=0,
        )

        # when
        terminated_early = training_module._run_optimizer_warmup_steps(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            warmup_steps=1,
            sparsity_coeffs=torch.tensor([0.0], dtype=torch.float32),
        )

        # then
        self.assertTrue(terminated_early)
        self.assertEqual(float(optimizer.param_groups[0]["lr"]), 0.1)

    def test_run_optimizer_warmup_steps_raises_when_dataloader_has_no_batches(
        self,
    ) -> None:
        # given
        class _FiniteWarmupModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {
                    "latents": torch.zeros((1, int(x.shape[0]), 1), device=x.device)
                }

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del y_true
                del sparsity_coeffs
                del output
                loss = (self.weight - 1.0) ** 2
                return {
                    "losses": loss.reshape(1),
                    "reconstruction_losses": loss.reshape(1),
                    "sparsity_losses": torch.zeros(1, device=self.weight.device),
                }

        model = _FiniteWarmupModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros((0, 4), dtype=torch.float32)),
            batch_size=1,
            shuffle=False,
        )

        # when / then
        with self.assertRaises(ValueError):
            _ = training_module._run_optimizer_warmup_steps(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                warmup_steps=1,
                sparsity_coeffs=torch.tensor([0.0], dtype=torch.float32),
            )

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

    def test_run_training_for_variant_sets_terminated_early_on_non_finite_losses(
        self,
    ) -> None:
        # given
        class NonFiniteLossModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))
                self.resample_calls = 0
                self.loss_value = float("nan")

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {}

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del y_true
                del sparsity_coeffs
                del output
                return {
                    "losses": torch.tensor([self.loss_value], dtype=torch.float32),
                    "reconstruction_losses": torch.tensor([0.0], dtype=torch.float32),
                    "sparsity_losses": torch.tensor([0.0], dtype=torch.float32),
                }

            def resample_dead_latents(
                self, inactive_latents_count: torch.Tensor
            ) -> torch.Tensor:
                self.resample_calls += 1
                return torch.zeros_like(inactive_latents_count, dtype=torch.bool)

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

        # when / then
        for loss_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(loss_value=loss_value):
                model = NonFiniteLossModel()
                model.loss_value = loss_value
                with patch(
                    "sgnsae.experiments.directional_deviation.training._instantiate_model",
                    return_value=model,
                ), patch("builtins.print"):
                    output = run_training_for_variant(
                        experiment_config, normal_data, model_config, self.device
                    )

                self.assertTrue(output["training_metrics"]["terminated_early"])
                self.assertEqual(output["training_metrics"]["epoch"], [])
                self.assertEqual(model.resample_calls, 0)

    def test_run_training_for_variant_restarts_training_after_warmup(self) -> None:
        # given
        class RecordingWarmupModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.seen_batches = []
                self.seen_weights = []
                self.resample_calls = 0

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                self.seen_batches.append(x.detach().clone())
                self.seen_weights.append(float(self.weight.detach().item()))
                return {
                    "latents": torch.zeros((1, int(x.shape[0]), 1), device=x.device)
                }

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del y_true
                del sparsity_coeffs
                del output
                loss = (self.weight - 1.0) ** 2
                return {
                    "losses": loss.reshape(1),
                    "reconstruction_losses": loss.reshape(1),
                    "sparsity_losses": torch.zeros(1, device=self.weight.device),
                }

            def resample_dead_latents(
                self, inactive_latents_count: torch.Tensor
            ) -> torch.Tensor:
                self.resample_calls += 1
                return torch.zeros_like(inactive_latents_count, dtype=torch.bool)

        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=4,
            num_val=1,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
            train_anomaly_fraction=0.0,
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
            learning_rate=0.1,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
            warmup_steps=3,
        )
        experiment_config = self._build_experiment_config(
            data_config, anomaly_config, model_config
        )
        normal_data = {
            "x_train": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        }
        torch.manual_seed(7)
        model = RecordingWarmupModel()

        # when
        with patch(
            "sgnsae.experiments.directional_deviation.training._instantiate_model",
            return_value=model,
        ):
            output = run_training_for_variant(
                experiment_config, normal_data, model_config, self.device
            )

        # then
        self.assertEqual(output["training_metrics"]["warmup_steps"], 3)
        self.assertEqual(model.resample_calls, 2)
        self.assertEqual(len(model.seen_batches), 5)
        self.assertTrue(torch.equal(model.seen_batches[0], model.seen_batches[2]))
        self.assertTrue(torch.equal(model.seen_batches[0], model.seen_batches[3]))
        self.assertTrue(torch.equal(model.seen_batches[1], model.seen_batches[4]))
        self.assertEqual(model.seen_weights[0], model.seen_weights[1])
        self.assertEqual(model.seen_weights[0], model.seen_weights[2])
        self.assertEqual(model.seen_weights[0], model.seen_weights[3])

    def test_run_training_for_variant_marks_terminated_early_on_warmup_nan(
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
            num_channels=4,
            train_anomaly_fraction=0.0,
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
            learning_rate=0.1,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
            warmup_steps=1,
        )
        experiment_config = self._build_experiment_config(
            data_config, anomaly_config, model_config
        )
        normal_data = {"x_train": torch.zeros((2, 4), dtype=torch.float32)}

        class _NaNWarmupModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {
                    "latents": torch.zeros((1, int(x.shape[0]), 1), device=x.device)
                }

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del y_true
                del sparsity_coeffs
                del output
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
        with patch(
            "sgnsae.experiments.directional_deviation.training._instantiate_model",
            return_value=_NaNWarmupModel(),
        ):
            output = run_training_for_variant(
                experiment_config, normal_data, model_config, self.device
            )

        # then
        self.assertTrue(output["training_metrics"]["terminated_early"])
        self.assertEqual(output["training_metrics"]["warmup_steps"], 1)
        self.assertEqual(output["training_metrics"]["epoch"], [])

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

    def test_run_training_for_variant_raises_when_real_training_has_no_steps(
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
            num_channels=4,
            train_anomaly_fraction=0.0,
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
            learning_rate=0.1,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = self._build_experiment_config(
            data_config, anomaly_config, model_config
        )
        normal_data = {"x_train": torch.zeros((2, 4), dtype=torch.float32)}

        class _FiniteModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {
                    "latents": torch.zeros((1, int(x.shape[0]), 1), device=x.device)
                }

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del y_true
                del sparsity_coeffs
                del output
                loss = (self.weight - 1.0) ** 2
                return {
                    "losses": loss.reshape(1),
                    "reconstruction_losses": loss.reshape(1),
                    "sparsity_losses": torch.zeros(1, device=self.weight.device),
                }

            def resample_dead_latents(
                self, inactive_latents_count: torch.Tensor
            ) -> torch.Tensor:
                return torch.zeros_like(inactive_latents_count, dtype=torch.bool)

        warmup_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros((2, 4), dtype=torch.float32)),
            batch_size=2,
            shuffle=False,
        )
        empty_training_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros((0, 4), dtype=torch.float32)),
            batch_size=1,
            shuffle=False,
        )

        # when / then
        with patch.object(
            training_module, "_instantiate_model", return_value=_FiniteModel()
        ), patch.object(
            training_module,
            "_build_training_dataloader",
            side_effect=[warmup_dataloader, empty_training_dataloader],
        ):
            with self.assertRaises(ValueError):
                _ = run_training_for_variant(
                    experiment_config, normal_data, model_config, self.device
                )

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
