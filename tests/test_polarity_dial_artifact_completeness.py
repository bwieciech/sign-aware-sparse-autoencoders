import json
import os
import tempfile
import unittest

from sgnsae.experiments.polarity_dial.runner import run_polarity_dial_experiment


class TestPolarityDialArtifactCompleteness(unittest.TestCase):
    def test_end_to_end_run_produces_all_documented_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            model_name = "relu_sae_artifacts"
            config = {
                "output_dir": tmpdir,
                "device": "cpu",
                "data": {
                    "seed": 123,
                    "num_train": 8,
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
                        "name": model_name,
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
                    }
                ],
                "verbose": False,
                "sample_preview_count": 4,
                "max_scatter_points_per_regime": 200,
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            run_dir = run_polarity_dial_experiment(config_path)

            # then
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "config.json")))
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "comparison_summary.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "gamma_diff_bar_comparison.png"))
            )

            data_dir = os.path.join(run_dir, "data")
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "generator_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "feature_directions.pt"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "coefficients_stats.json"))
            )
            self.assertTrue(os.path.isfile(os.path.join(data_dir, "sampled_data.pt")))

            model_dir = os.path.join(run_dir, "models", model_name)
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "model.pt")))
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
                os.path.isfile(
                    os.path.join(plots_dir, f"training_loss_{model_name}.png")
                )
            )

    def test_end_to_end_run_with_sweeps_produces_sweep_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            model_name = "relu_sae_sweep_artifacts"
            config = {
                "output_dir": tmpdir,
                "device": "cpu",
                "data": {
                    "seed": 321,
                    "num_train": 8,
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
                        "name": model_name,
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
                    }
                ],
                "verbose": False,
                "sample_preview_count": 4,
                "max_scatter_points_per_regime": 100,
                "sparsity_coeff_grid": [0.0005, 0.005],
                "p_plus_grid": [0.4, 0.6],
                "sweeps": {"exponential_lambda_grid": [1.5]},
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            run_dir = run_polarity_dial_experiment(config_path)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
