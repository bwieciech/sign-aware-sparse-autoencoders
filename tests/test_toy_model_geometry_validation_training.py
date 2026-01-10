import os
import tempfile
import unittest
from types import SimpleNamespace
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

    def test_run_training_for_variant_marks_terminated_early_on_nan(self) -> None:
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

        class _NaNLossModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.W = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"latents": torch.zeros((1, int(x.shape[0]), 1))}

            def calculate_losses(
                self,
                x: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: dict[str, torch.Tensor],
            ) -> dict[str, torch.Tensor]:
                return {
                    "losses": torch.tensor([float("nan")]),
                    "reconstruction_losses": torch.tensor([0.0]),
                    "sparsity_losses": torch.tensor([0.0]),
                }

        # when
        with patch.object(training_module, "_instantiate_model", return_value=_NaNLossModel()):
            output = training_module.run_training_for_variant(
                experiment_config=experiment_config,
                data=data,  # type: ignore[arg-type]
                model_config=model_config,
                device="cpu",
            )

        # then
        self.assertTrue(output["training_metrics"]["terminated_early"])

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
