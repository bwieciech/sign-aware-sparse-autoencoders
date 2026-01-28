import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

from sgnsae.experiments.polarity_dial.runner import run_polarity_dial_experiment


class TestPolarityDialMultiSparsitySmoke(unittest.TestCase):
    def test_end_to_end_run_creates_expected_multi_sparsity_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config: Dict[str, Any] = {
                "output_dir": tmpdir,
                "device": "cpu",
                "data": {
                    "seed": 17,
                    "num_train": 12,
                    "num_val": 4,
                    "num_test": 4,
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
                        "name": "scalar_sae",
                        "sae_class": "SAE",
                        "sign_aware": False,
                        "single_activation_offset": False,
                        "reset_activation_offsets_on_resample": False,
                        "d_hidden": 4,
                        "sparsity_coeff": 0.0005,
                        "learning_rate": 0.001,
                        "num_epochs": 1,
                        "batch_size": 4,
                        "adam_betas": [0.9, 0.999],
                    },
                    {
                        "name": "list_sae_short",
                        "sae_class": "SAE",
                        "sign_aware": False,
                        "single_activation_offset": False,
                        "reset_activation_offsets_on_resample": False,
                        "d_hidden": 4,
                        "sparsity_coeff": [0.0003, 0.0007],
                        "learning_rate": 0.001,
                        "num_epochs": 1,
                        "batch_size": 4,
                        "adam_betas": [0.9, 0.999],
                    },
                    {
                        "name": "list_gated_long",
                        "sae_class": "GatedSAE",
                        "sign_aware": False,
                        "single_activation_offset": False,
                        "reset_activation_offsets_on_resample": False,
                        "d_hidden": 4,
                        "sparsity_coeff": [0.0002, 0.0005, 0.001],
                        "learning_rate": 0.001,
                        "num_epochs": 1,
                        "batch_size": 4,
                        "adam_betas": [0.9, 0.999],
                    },
                ],
                "log_every_n_steps": 10,
                "log_every_n_epochs": 1,
                "verbose": False,
                "sample_preview_count": 4,
                "max_scatter_points_per_regime": 200,
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            run_dir = run_polarity_dial_experiment(config_path)

            # then
            self.assertTrue(os.path.isdir(run_dir))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "config.json")))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "environment.json")))

            data_dir = os.path.join(run_dir, "data")
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "generator_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "coefficients_stats.json"))
            )
            self.assertTrue(os.path.isfile(os.path.join(data_dir, "sampled_data.pt")))

            self._assert_multi_sparsity_plots_exist(
                seed_run_dir=run_dir,
                model_name="list_sae_short",
                expected_prefixes=[
                    "mean_active_latents",
                    "reconstruction_loss",
                    "sparsity_loss",
                ],
            )
            self._assert_multi_sparsity_plots_exist(
                seed_run_dir=run_dir,
                model_name="list_gated_long",
                expected_prefixes=[
                    "mean_active_latents",
                    "reconstruction_loss",
                    "sparsity_loss",
                    "aux_loss",
                ],
            )

            expected_subruns = ["0", "1", "2"]
            for subrun_index in expected_subruns:
                subrun_dir = os.path.join(run_dir, subrun_index)
                self._assert_subrun_artifacts_exist(subrun_dir)

            self._assert_subrun_model_inclusion(
                run_dir=run_dir,
                subrun_index="0",
                expected_models=[
                    "scalar_sae",
                    "list_sae_short",
                    "list_gated_long",
                ],
            )
            self._assert_subrun_model_inclusion(
                run_dir=run_dir,
                subrun_index="1",
                expected_models=[
                    "scalar_sae",
                    "list_sae_short",
                    "list_gated_long",
                ],
            )
            self._assert_subrun_model_inclusion(
                run_dir=run_dir,
                subrun_index="2",
                expected_models=[
                    "scalar_sae",
                    "list_gated_long",
                ],
            )

    def _assert_multi_sparsity_plots_exist(
        self,
        seed_run_dir: str,
        model_name: str,
        expected_prefixes: List[str],
    ) -> None:
        plots_dir = os.path.join(seed_run_dir, "plots", "multi_sparsity", model_name)
        for prefix in expected_prefixes:
            self.assertTrue(
                os.path.isfile(
                    os.path.join(plots_dir, f"{prefix}_vs_sparsity_coeff.png")
                )
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(plots_dir, f"{prefix}_vs_log_sparsity_coeff.png")
                )
            )

    def _assert_subrun_artifacts_exist(self, subrun_dir: str) -> None:
        self.assertTrue(os.path.isdir(subrun_dir))
        self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "config.json")))
        self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "environment.json")))
        self.assertTrue(
            os.path.isfile(os.path.join(subrun_dir, "comparison_summary.json"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(subrun_dir, "gamma_diff_bar_comparison.png"))
        )

        data_dir = os.path.join(subrun_dir, "data")
        self.assertTrue(os.path.isfile(os.path.join(data_dir, "generator_config.json")))
        self.assertTrue(
            os.path.isfile(os.path.join(data_dir, "coefficients_stats.json"))
        )
        self.assertTrue(os.path.isfile(os.path.join(data_dir, "sampled_data.pt")))

    def _assert_subrun_model_inclusion(
        self,
        run_dir: str,
        subrun_index: str,
        expected_models: List[str],
    ) -> None:
        subrun_dir = os.path.join(run_dir, subrun_index)
        with open(os.path.join(subrun_dir, "config.json"), "r") as f:
            subrun_config = json.load(f)
        models_in_config = {
            model["name"]: model for model in (subrun_config.get("models") or [])
        }
        self.assertEqual(sorted(models_in_config.keys()), sorted(expected_models))
        for model_name in expected_models:
            self._assert_model_artifacts_exist(subrun_dir, model_name)

        unexpected = {"scalar_sae", "list_sae_short", "list_gated_long"} - set(
            expected_models
        )
        for model_name in unexpected:
            self.assertFalse(
                os.path.isdir(os.path.join(subrun_dir, "models", model_name))
            )

        for model_name, model in models_in_config.items():
            self.assertIsInstance(model.get("sparsity_coeff"), float)
            if model_name == "scalar_sae":
                self.assertAlmostEqual(float(model["sparsity_coeff"]), 0.0005, places=7)
            if model_name == "list_sae_short":
                expected = [0.0003, 0.0007][int(subrun_index)]
                self.assertAlmostEqual(
                    float(model["sparsity_coeff"]), expected, places=7
                )
            if model_name == "list_gated_long":
                expected = [0.0002, 0.0005, 0.001][int(subrun_index)]
                self.assertAlmostEqual(
                    float(model["sparsity_coeff"]), expected, places=7
                )

    def _assert_model_artifacts_exist(self, subrun_dir: str, model_name: str) -> None:
        model_dir = os.path.join(subrun_dir, "models", model_name)
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "model_config.json")))
        self.assertTrue(
            os.path.isfile(os.path.join(model_dir, "training_metrics.json"))
        )
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "alignment.json")))
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "axis_alignment.json")))
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "calibration.json")))
        plots_dir = os.path.join(model_dir, "plots")
        self.assertTrue(
            os.path.isfile(
                os.path.join(plots_dir, f"calibration_scatter_{model_name}.png")
            )
        )
        self.assertTrue(
            os.path.isfile(os.path.join(plots_dir, f"gamma_hist_{model_name}.png"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(plots_dir, f"gamma_plus_minus_strip_{model_name}.png")
            )
        )
        self.assertTrue(
            os.path.isfile(os.path.join(plots_dir, f"training_loss_{model_name}.png"))
        )
        with open(os.path.join(model_dir, "training_metrics.json"), "r") as f:
            training_metrics = json.load(f)
        self.assertIn("loss_total", training_metrics)
        self.assertIn("loss_reconstruction", training_metrics)
        self.assertIn("loss_sparsity", training_metrics)
        self.assertIn("mean_active_latents", training_metrics)
        self.assertNotIn("loss_total_by_instance", training_metrics)
        self.assertNotIn("loss_reconstruction_by_instance", training_metrics)
        self.assertNotIn("loss_sparsity_by_instance", training_metrics)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
