import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Dict
from unittest.mock import patch

import torch

from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DDataConfig,
    Toy2DExperimentConfig,
    Toy2DModelVariantConfig,
)
from sgnsae.experiments.toy_model_geometry_validation import training as training_module


class TestToyModelGeometryValidationTraining(unittest.TestCase):
    def test_effective_sparsity_coeff_returns_base_without_warmup(self) -> None:
        # given
        base_coeff = 0.5

        # when
        result = training_module._effective_sparsity_coeff(
            base_coeff=base_coeff,
            epoch_index=0,
            warmup_epochs=0,
        )

        # then
        self.assertEqual(result, base_coeff)

    def test_effective_sparsity_coeff_linearly_warms_up_then_caps(self) -> None:
        # given
        base_coeff = 0.8
        warmup_epochs = 4

        # when
        early = training_module._effective_sparsity_coeff(
            base_coeff=base_coeff,
            epoch_index=0,
            warmup_epochs=warmup_epochs,
        )
        mid = training_module._effective_sparsity_coeff(
            base_coeff=base_coeff,
            epoch_index=1,
            warmup_epochs=warmup_epochs,
        )
        full = training_module._effective_sparsity_coeff(
            base_coeff=base_coeff,
            epoch_index=3,
            warmup_epochs=warmup_epochs,
        )
        capped = training_module._effective_sparsity_coeff(
            base_coeff=base_coeff,
            epoch_index=10,
            warmup_epochs=warmup_epochs,
        )

        # then
        self.assertAlmostEqual(early, 0.2)
        self.assertAlmostEqual(mid, 0.4)
        self.assertAlmostEqual(full, base_coeff)
        self.assertAlmostEqual(capped, base_coeff)

    def test_run_training_for_variant_rejects_empty_training_data(self) -> None:
        # given
        model_config = Toy2DModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.1,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = Toy2DExperimentConfig(
            output_dir="output",
            device="cpu",
            data=Toy2DDataConfig(
                seed=0,
                num_train=1,
                num_val=1,
                num_test=1,
                num_pairs=1,
                noise_std=0.0,
            ),
            models=[model_config],
        )
        data = {"x_train": torch.zeros((0, 2), dtype=torch.float32)}

        # when / then
        with self.assertRaises(ValueError):
            _ = training_module.run_training_for_variant(
                experiment_config=experiment_config,
                data=data,  # type: ignore[arg-type]
                model_config=model_config,
                device="cpu",
            )

    def test_instantiate_model_raises_on_unknown_sae_class(self) -> None:
        # given
        model_config = SimpleNamespace(
            sae_class="UnknownSAE",
            d_hidden=1,
            resample_latents_epochs=1.0,
            sign_aware=False,
            reset_activation_offsets_on_resample=False,
            single_activation_offset=False,
            symmetric_magnitude=False,
        )

        # when / then
        with self.assertRaises(ValueError):
            _ = training_module._instantiate_model(  # type: ignore[arg-type]
                model_config=model_config,
                d_in=2,
                steps_per_epoch=1,
                device="cpu",
            )

    def test_instantiate_model_builds_expected_model_classes(self) -> None:
        # given
        sae_config = Toy2DModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.1,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        gated_config = Toy2DModelVariantConfig(
            name="gated_sae",
            sae_class="GatedSAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            symmetric_magnitude=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.1,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )

        # when
        sae_model = training_module._instantiate_model(
            model_config=sae_config,
            d_in=2,
            steps_per_epoch=1,
            device="cpu",
        )
        gated_model = training_module._instantiate_model(
            model_config=gated_config,
            d_in=2,
            steps_per_epoch=1,
            device="cpu",
        )

        # then
        self.assertEqual(sae_model.__class__.__name__, "SAE")
        self.assertEqual(gated_model.__class__.__name__, "GatedSAE")

    def test_log_step_progress_writes_expected_message(self) -> None:
        # given
        losses_dict = {
            "losses": torch.tensor([1.0]),
            "reconstruction_losses": torch.tensor([0.5]),
            "sparsity_losses": torch.tensor([0.25]),
        }
        stdout = io.StringIO()

        # when
        with patch.object(training_module.sys, "stdout", stdout):
            training_module._log_step_progress(
                epoch=0,
                num_epochs=1,
                step=0,
                total_steps=1,
                losses_dict=losses_dict,
                mean_active_latents=1.5,
                model_name="model",
            )

        # then
        self.assertIn("[model] Epoch 1/1 Step 1/1", stdout.getvalue())

    def test_log_epoch_summary_includes_aux_loss_when_present(self) -> None:
        # given
        epoch_mean_losses = {
            "losses": 1.0,
            "reconstruction_losses": 0.5,
            "sparsity_losses": 0.25,
            "aux_loss": 0.1,
        }

        # when / then
        training_module._log_epoch_summary(
            epoch=0,
            epoch_mean_losses=epoch_mean_losses,
            model_name="model",
        )

    def test_run_training_for_variant_marks_terminated_early_on_non_finite_loss(
        self,
    ) -> None:
        # given
        model_config = Toy2DModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.1,
            num_epochs=2,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = Toy2DExperimentConfig(
            output_dir="output",
            device="cpu",
            data=Toy2DDataConfig(
                seed=0,
                num_train=1,
                num_val=1,
                num_test=1,
                num_pairs=1,
                noise_std=0.0,
            ),
            models=[model_config],
            verbose=False,
            log_every_n_steps=1,
            log_every_n_epochs=1,
        )
        data = {"x_train": torch.zeros((1, 2), dtype=torch.float32)}

        class _NonFiniteLossModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.W = torch.nn.Parameter(torch.zeros(()))
                self.resample_calls = 0
                self.loss_value = float("nan")

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"latents": torch.zeros((1, int(x.shape[0]), 1))}

            def calculate_losses(
                self,
                x: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: dict[str, torch.Tensor],
            ) -> dict[str, torch.Tensor]:
                del x
                del sparsity_coeffs
                del output
                return {
                    "losses": torch.tensor([self.loss_value]),
                    "reconstruction_losses": torch.tensor([0.0]),
                    "sparsity_losses": torch.tensor([0.0]),
                }

            def resample_dead_latents(
                self, inactive_latents_count: torch.Tensor
            ) -> torch.Tensor:
                self.resample_calls += 1
                return torch.zeros_like(inactive_latents_count, dtype=torch.bool)

        # when / then
        for loss_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(loss_value=loss_value):
                model = _NonFiniteLossModel()
                model.loss_value = loss_value
                with patch.object(
                    training_module, "_instantiate_model", return_value=model
                ):
                    output = training_module.run_training_for_variant(
                        experiment_config=experiment_config,
                        data=data,  # type: ignore[arg-type]
                        model_config=model_config,
                        device="cpu",
                    )

                self.assertTrue(output["training_metrics"]["terminated_early"])
                self.assertEqual(output["training_metrics"]["epoch"], [])
                self.assertEqual(model.resample_calls, 0)

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
                x: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del x
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
            x_train=torch.ones((2, 2), dtype=torch.float32),
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
                x: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del x
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
            torch.utils.data.TensorDataset(torch.zeros((0, 2), dtype=torch.float32)),
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

    def test_run_training_for_variant_restarts_training_after_warmup(self) -> None:
        # given
        model_config = Toy2DModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.1,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
            warmup_steps=3,
        )
        experiment_config = Toy2DExperimentConfig(
            output_dir="output",
            device="cpu",
            data=Toy2DDataConfig(
                seed=0,
                num_train=4,
                num_val=1,
                num_test=1,
                num_pairs=1,
                noise_std=0.0,
            ),
            models=[model_config],
            verbose=False,
            log_every_n_steps=1,
            log_every_n_epochs=1,
        )
        data = {
            "x_train": torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                ],
                dtype=torch.float32,
            )
        }

        class _RecordingWarmupModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.seen_batches = []
                self.seen_weights = []
                self.resample_calls = 0

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                self.seen_batches.append(x.detach().clone())
                self.seen_weights.append(float(self.weight.detach().item()))
                return {
                    "latents": torch.zeros((1, int(x.shape[0]), 1), device=x.device)
                }

            def calculate_losses(
                self,
                x: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: dict[str, torch.Tensor],
            ) -> dict[str, torch.Tensor]:
                del x
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

        model = _RecordingWarmupModel()
        dataloader_generator = torch.Generator(device="cpu").manual_seed(7)

        # when
        with patch.object(training_module, "_instantiate_model", return_value=model):
            output = training_module.run_training_for_variant(
                experiment_config=experiment_config,
                data=data,  # type: ignore[arg-type]
                model_config=model_config,
                device="cpu",
                dataloader_generator=dataloader_generator,
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

    def test_run_training_for_variant_uses_epoch_zero_sparsity_coeff_for_warmup(
        self,
    ) -> None:
        # given
        model_config = Toy2DModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=1.0,
            sparsity_coeff_warmup_epochs=10,
            learning_rate=0.1,
            num_epochs=10,
            batch_size=1,
            adam_betas=(0.9, 0.999),
            warmup_steps=1,
        )
        experiment_config = Toy2DExperimentConfig(
            output_dir="output",
            device="cpu",
            data=Toy2DDataConfig(
                seed=0,
                num_train=1,
                num_val=1,
                num_test=1,
                num_pairs=1,
                noise_std=0.0,
            ),
            models=[model_config],
            verbose=False,
            log_every_n_steps=1,
            log_every_n_epochs=1,
        )
        data = {"x_train": torch.ones((1, 2), dtype=torch.float32)}

        class _RecordingSparsityCoeffModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.sparsity_coeffs = []

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {
                    "latents": torch.zeros((1, int(x.shape[0]), 1), device=x.device)
                }

            def calculate_losses(
                self,
                x: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del x
                del output
                self.sparsity_coeffs.append(float(sparsity_coeffs[0].item()))
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

        model = _RecordingSparsityCoeffModel()

        # when
        with patch.object(training_module, "_instantiate_model", return_value=model):
            output = training_module.run_training_for_variant(
                experiment_config=experiment_config,
                data=data,  # type: ignore[arg-type]
                model_config=model_config,
                device="cpu",
            )

        # then
        self.assertAlmostEqual(model.sparsity_coeffs[0], 0.1)
        self.assertAlmostEqual(model.sparsity_coeffs[1], 0.1)
        self.assertAlmostEqual(
            output["training_metrics"]["warmup_sparsity_coeff_effective"],
            0.1,
        )

    def test_run_training_for_variant_marks_terminated_early_on_warmup_nan(
        self,
    ) -> None:
        # given
        model_config = Toy2DModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.1,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
            warmup_steps=1,
        )
        experiment_config = Toy2DExperimentConfig(
            output_dir="output",
            device="cpu",
            data=Toy2DDataConfig(
                seed=0,
                num_train=1,
                num_val=1,
                num_test=1,
                num_pairs=1,
                noise_std=0.0,
            ),
            models=[model_config],
            verbose=False,
            log_every_n_steps=1,
            log_every_n_epochs=1,
        )
        data = {"x_train": torch.zeros((1, 2), dtype=torch.float32)}

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
                x: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del x
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
        with patch.object(
            training_module, "_instantiate_model", return_value=_NaNWarmupModel()
        ):
            output = training_module.run_training_for_variant(
                experiment_config=experiment_config,
                data=data,  # type: ignore[arg-type]
                model_config=model_config,
                device="cpu",
            )

        # then
        self.assertTrue(output["training_metrics"]["terminated_early"])
        self.assertEqual(output["training_metrics"]["warmup_steps"], 1)
        self.assertEqual(output["training_metrics"]["epoch"], [])

    def test_run_training_for_variant_logs_verbose_and_records_aux_loss(self) -> None:
        # given
        model_config = Toy2DModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.1,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        experiment_config = Toy2DExperimentConfig(
            output_dir="output",
            device="cpu",
            data=Toy2DDataConfig(
                seed=0,
                num_train=2,
                num_val=1,
                num_test=1,
                num_pairs=1,
                noise_std=0.0,
            ),
            models=[model_config],
            verbose=True,
            log_every_n_steps=1,
            log_every_n_epochs=1,
        )
        data = {"x_train": torch.ones((2, 2), dtype=torch.float32)}

        class _AuxLossModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {
                    "latents": torch.ones((1, int(x.shape[0]), 1), device=x.device)
                    + 0.0 * self.weight
                }

            def calculate_losses(
                self,
                x: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del x
                del sparsity_coeffs
                del output
                loss = (self.weight - 1.0) ** 2
                return {
                    "losses": loss.reshape(1),
                    "reconstruction_losses": loss.reshape(1),
                    "sparsity_losses": torch.zeros(1, device=self.weight.device),
                    "aux_loss": torch.ones(1, device=self.weight.device) * 0.25,
                }

            def resample_dead_latents(
                self, inactive_latents_count: torch.Tensor
            ) -> torch.Tensor:
                return torch.zeros_like(inactive_latents_count, dtype=torch.bool)

        # when
        with patch.object(
            training_module, "_instantiate_model", return_value=_AuxLossModel()
        ), patch.object(
            training_module, "_log_step_progress"
        ) as log_step_mock, patch.object(
            training_module, "_log_epoch_summary"
        ) as log_epoch_mock:
            output = training_module.run_training_for_variant(
                experiment_config=experiment_config,
                data=data,  # type: ignore[arg-type]
                model_config=model_config,
                device="cpu",
            )

        # then
        self.assertTrue(log_step_mock.called)
        self.assertTrue(log_epoch_mock.called)
        self.assertEqual(output["training_metrics"]["loss_aux"], [0.25])

    def test_persist_model_artifacts_records_optional_model_dimensions(self) -> None:
        # given
        model = torch.nn.Linear(2, 2)
        model.d_in = 2  # type: ignore[attr-defined]
        model.d_hidden = 3  # type: ignore[attr-defined]
        training_metrics = {
            "epoch": [1],
            "loss_total": [1.0],
            "loss_reconstruction": [0.5],
            "loss_sparsity": [0.25],
        }
        model_config = Toy2DModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=1,
            sparsity_coeff=0.0,
            learning_rate=0.1,
            num_epochs=1,
            batch_size=1,
            adam_betas=(0.9, 0.999),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")

            # when
            with patch.object(training_module, "plot_training_loss") as plot_mock:
                training_module.persist_model_artifacts(
                    model=model,
                    training_metrics=training_metrics,
                    model_config=model_config,
                    model_dir=model_dir,
                )

            # then
            with open(os.path.join(model_dir, "model_config.json"), "r") as file:
                persisted_config = json.load(file)
            self.assertEqual(persisted_config["d_in"], 2)
            self.assertEqual(persisted_config["d_hidden_effective"], 3)
            self.assertTrue(plot_mock.called)

    def test_plot_training_loss_raises_when_matplotlib_missing(self) -> None:
        # given
        training_metrics = {
            "epoch": [1],
            "loss_total": [1.0],
            "loss_reconstruction": [0.5],
        }

        original_import = __import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "matplotlib":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        # when / then
        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaises(RuntimeError):
                training_module.plot_training_loss(
                    training_metrics=training_metrics,
                    model_name="model",
                    model_dir="output",
                )

    def test_plot_training_loss_ignores_matplotlib_use_failure(self) -> None:
        # given
        training_metrics = {
            "epoch": [1],
            "loss_total": [1.0],
            "loss_reconstruction": [0.5],
        }

        # when
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("matplotlib.use", side_effect=RuntimeError("boom")):
                output = training_module.plot_training_loss(
                    training_metrics=training_metrics,
                    model_name="model",
                    model_dir=tmpdir,
                )

            # then
            output_path = os.path.join(tmpdir, "plots", "training_loss_model.png")
            self.assertEqual(output, None)
            self.assertTrue(os.path.isfile(output_path))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
