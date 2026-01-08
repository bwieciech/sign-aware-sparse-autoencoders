import builtins
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

import torch

from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DDataConfig,
    Toy2DExperimentConfig,
    Toy2DModelVariantConfig,
)
from sgnsae.experiments.toy_model_geometry_validation.data import generate_toy2d_dataset
from sgnsae.experiments.toy_model_geometry_validation import training as training_module


class TestToyModelGeometryValidationTraining(unittest.TestCase):
    def _build_data_config(
        self, overrides: Dict[str, Any] | None = None
    ) -> Toy2DDataConfig:
        base: Dict[str, Any] = {
            "seed": 0,
            "num_train": 16,
            "num_val": 4,
            "num_test": 4,
            "num_pairs": 2,
            "direction_mode": "uniform",
            "pair_axis_active_prob": 1.0,
            "pair_feature_prob": 0.5,
            "pair_correlation": -1.0,
            "magnitude_dist": "constant",
            "constant_magnitude": 1.0,
            "noise_std": 0.0,
        }
        if overrides:
            base.update(overrides)
        return Toy2DDataConfig(**base)

    def _build_model_config(
        self, overrides: Dict[str, Any] | None = None
    ) -> Toy2DModelVariantConfig:
        base: Dict[str, Any] = {
            "name": "relu_sae",
            "sae_class": "SAE",
            "sign_aware": False,
            "single_activation_offset": False,
            "reset_activation_offsets_on_resample": False,
            "d_hidden": 4,
            "sparsity_coeff": 0.0,
            "learning_rate": 1e-2,
            "num_epochs": 1,
            "batch_size": 4,
            "adam_betas": (0.9, 0.999),
        }
        if overrides:
            base.update(overrides)
        return Toy2DModelVariantConfig(**base)

    def _build_experiment_config(
        self,
        data_config: Toy2DDataConfig,
        model_config: Toy2DModelVariantConfig,
        overrides: Dict[str, Any] | None = None,
    ) -> Toy2DExperimentConfig:
        base: Dict[str, Any] = {
            "output_dir": "unused",
            "device": "cpu",
            "data": data_config,
            "models": [model_config],
            "verbose": False,
            "log_every_n_steps": 1,
            "log_every_n_epochs": 1,
            "sample_preview_count": 4,
        }
        if overrides:
            base.update(overrides)
        return Toy2DExperimentConfig(**base)

    def test_instantiate_model_rejects_unknown_sae_class(self) -> None:
        # given
        model_config = self._build_model_config()
        model_config.sae_class = "BadSAE"

        # when & then
        with self.assertRaises(ValueError):
            _ = training_module._instantiate_model(
                model_config=model_config, d_in=2, steps_per_epoch=1, device="cpu"
            )

    def test_run_training_for_variant_trains_sae_and_persists_artifacts(self) -> None:
        # given
        model_config = self._build_model_config({"num_epochs": 1, "sae_class": "SAE"})
        experiment_config = self._build_experiment_config(
            self._build_data_config(), model_config, overrides={"verbose": True}
        )
        data = generate_toy2d_dataset(experiment_config.data, device="cpu")
        dataloader_generator = torch.Generator(device="cpu").manual_seed(123)

        # when
        training_output = training_module.run_training_for_variant(
            experiment_config=experiment_config,
            data=data,
            model_config=model_config,
            device="cpu",
            dataloader_generator=dataloader_generator,
        )

        # then
        self.assertEqual(training_output["model"].__class__.__name__, "SAE")
        self.assertEqual(training_output["training_metrics"]["epoch"], [1])
        self.assertEqual(len(training_output["training_metrics"]["loss_total"]), 1)
        self.assertNotIn("loss_aux", training_output["training_metrics"])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "models", model_config.name)
            training_module.persist_model_artifacts(
                model=training_output["model"],
                training_metrics=training_output["training_metrics"],
                model_config=model_config,
                model_dir=model_dir,
            )
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "model.pt")))
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "model_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "training_metrics.json"))
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        model_dir, "plots", f"training_loss_{model_config.name}.png"
                    )
                )
            )

    def test_run_training_for_variant_records_aux_loss_for_gated_sae(self) -> None:
        # given
        model_config = self._build_model_config(
            {"name": "gated_sae", "sae_class": "GatedSAE", "num_epochs": 1}
        )
        experiment_config = self._build_experiment_config(
            self._build_data_config(),
            model_config,
            overrides={"verbose": True, "log_every_n_steps": 100},
        )
        data = generate_toy2d_dataset(experiment_config.data, device="cpu")

        # when
        training_output = training_module.run_training_for_variant(
            experiment_config=experiment_config,
            data=data,
            model_config=model_config,
            device="cpu",
        )

        # then
        self.assertEqual(training_output["model"].__class__.__name__, "GatedSAE")
        self.assertEqual(training_output["training_metrics"]["epoch"], [1])
        self.assertEqual(len(training_output["training_metrics"]["loss_aux"]), 1)

    def test_plot_training_loss_ignores_backend_selection_errors(self) -> None:
        # given
        import matplotlib

        matplotlib.use("Agg")
        training_metrics = {
            "epoch": [1],
            "loss_total": [1.0],
            "loss_reconstruction": [1.0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            with patch.object(matplotlib, "use", side_effect=RuntimeError("locked")):
                training_module.plot_training_loss(
                    training_metrics=training_metrics,
                    model_name="relu_sae",
                    model_dir=tmpdir,
                )

            # then
            self.assertTrue(
                os.path.isfile(
                    os.path.join(tmpdir, "plots", "training_loss_relu_sae.png")
                )
            )

    def test_run_training_for_variant_marks_terminated_early_on_nan(self) -> None:
        # given
        class DummyNaNModel(torch.nn.Module):
            def __init__(self, d_hidden: int):
                super().__init__()
                self.d_hidden = d_hidden
                self.param = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                batch = int(x.shape[0])
                return {
                    "latents": torch.zeros((1, batch, self.d_hidden), device=x.device)
                }

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                nan = torch.tensor([float("nan")], device=y_true.device)
                zero = torch.zeros_like(nan)
                return {
                    "losses": nan,
                    "reconstruction_losses": zero,
                    "sparsity_losses": zero,
                    "aux_loss": zero,
                }

            def resample_dead_latents(
                self, inactive_latents_count: torch.Tensor
            ) -> torch.Tensor:
                return torch.zeros_like(inactive_latents_count, dtype=torch.bool)

        model_config = self._build_model_config({"num_epochs": 2})
        experiment_config = self._build_experiment_config(
            self._build_data_config(), model_config
        )
        data = generate_toy2d_dataset(experiment_config.data, device="cpu")

        # when
        with patch.object(
            training_module,
            "_instantiate_model",
            return_value=DummyNaNModel(model_config.d_hidden),
        ):
            training_output = training_module.run_training_for_variant(
                experiment_config=experiment_config,
                data=data,
                model_config=model_config,
                device="cpu",
            )

        # then
        self.assertTrue(training_output["training_metrics"]["terminated_early"])
        self.assertEqual(training_output["training_metrics"]["epoch"], [])

    def test_run_training_for_variant_rejects_empty_training_tensor(self) -> None:
        # given
        model_config = self._build_model_config({"batch_size": 4})
        experiment_config = self._build_experiment_config(
            self._build_data_config(), model_config
        )
        data = {
            "x_train": torch.zeros((0, 2), dtype=torch.float32),
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = training_module.run_training_for_variant(
                experiment_config=experiment_config,
                data=data,
                model_config=model_config,
                device="cpu",
            )

    def test_plot_training_loss_raises_when_matplotlib_import_fails(self) -> None:
        # given
        training_metrics = {
            "epoch": [1],
            "loss_total": [1.0],
            "loss_reconstruction": [1.0],
        }

        real_import = builtins.__import__

        def fake_import(
            name: str,
            globals: Dict[str, Any] | None = None,
            locals: Dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib is unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmpdir:
            # when & then
            with patch.object(builtins, "__import__", side_effect=fake_import):
                with self.assertRaises(RuntimeError):
                    training_module.plot_training_loss(
                        training_metrics=training_metrics,
                        model_name="relu_sae",
                        model_dir=tmpdir,
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
