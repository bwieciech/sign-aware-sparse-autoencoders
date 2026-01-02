import json
import os
import tempfile
import unittest

from sgnsae.experiments.polarity_dial.runner import run_polarity_dial_experiment


class TestPolarityDialSmoke(unittest.TestCase):
    def test_end_to_end_run_creates_expected_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config = {
                "output_dir": tmpdir,
                "device": "cpu",
                "data": {
                    "seed": 11,
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
                        "name": "relu_sae_smoke",
                        "sae_class": "SAE",
                        "sign_aware": False,
                        "single_activation_offset": False,
                        "reset_activation_offsets_on_resample": False,
                        "d_hidden": 4,
                        "sparsity_coeff": 0.0005,
                        "learning_rate": 0.001,
                        "num_epochs": 2,
                        "batch_size": 4,
                        "adam_betas": [0.9, 0.999],
                    }
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
            data_dir = os.path.join(run_dir, "data")
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "generator_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "coefficients_stats.json"))
            )
            self.assertTrue(os.path.isfile(os.path.join(data_dir, "sampled_data.pt")))
            model_dir = os.path.join(run_dir, "models", "relu_sae_smoke")
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "model_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "training_metrics.json"))
            )
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "alignment.json")))
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "axis_alignment.json"))
            )
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "calibration.json")))
            with open(os.path.join(model_dir, "model_config.json"), "r") as f:
                model_config = json.load(f)
            self.assertIn("parameter_count", model_config)
            self.assertIn("d_hidden_effective", model_config)
            plots_dir = os.path.join(model_dir, "plots")
            self.assertTrue(
                os.path.isfile(
                    os.path.join(plots_dir, "calibration_scatter_relu_sae_smoke.png")
                )
            )
            self.assertTrue(
                os.path.isfile(os.path.join(plots_dir, "gamma_hist_relu_sae_smoke.png"))
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(plots_dir, "gamma_plus_minus_strip_relu_sae_smoke.png")
                )
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(plots_dir, "training_loss_relu_sae_smoke.png")
                )
            )
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "comparison_summary.json"))
            )
            with open(os.path.join(run_dir, "comparison_summary.json"), "r") as f:
                comparison_summary = json.load(f)
            self.assertIn("model_metadata", comparison_summary)
            self.assertIn("relu_sae_smoke", comparison_summary["model_metadata"])
            self.assertIn(
                "parameter_count",
                comparison_summary["model_metadata"]["relu_sae_smoke"],
            )
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "gamma_diff_bar_comparison.png"))
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
