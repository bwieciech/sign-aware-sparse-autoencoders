import json
import os
import tempfile
import unittest
from typing import Any, Dict

from sgnsae.experiments.polarity_dial.runner import run_polarity_dial_experiment


class TestPolarityDialRunnerOrderIndependence(unittest.TestCase):
    def test_model_calibration_summary_is_order_independent(self) -> None:
        # given
        base_config = {
            "device": "cpu",
            "data": {
                "seed": 7,
                "num_train": 32,
                "num_val": 8,
                "num_test": 8,
                "input_dim": 8,
                "num_features": 3,
                "activation_prob": 0.5,
                "positive_sign_prob": 0.6,
                "lognormal_mu": 0.0,
                "lognormal_sigma": 0.4,
                "exponential_lambda": 1.2,
                "noise_std": 0.05,
            },
            "evaluation": {
                "min_regime_samples": 1,
                "max_points_per_regime_scatter": 16,
            },
            "verbose": False,
            "sample_preview_count": 8,
            "log_every_n_steps": 100,
            "log_every_n_epochs": 100,
        }
        model_1 = {
            "name": "relu_sae_order_independent",
            "sae_class": "SAE",
            "sign_aware": False,
            "single_activation_offset": False,
            "reset_activation_offsets_on_resample": False,
            "d_hidden": 3,
            "sparsity_coeff": 0.001,
            "learning_rate": 0.01,
            "num_epochs": 2,
            "batch_size": 8,
            "adam_betas": [0.9, 0.999],
        }
        model_2 = {
            "name": "gated_sae_order_independent",
            "sae_class": "GatedSAE",
            "sign_aware": False,
            "single_activation_offset": False,
            "reset_activation_offsets_on_resample": False,
            "d_hidden": 3,
            "sparsity_coeff": 0.001,
            "learning_rate": 0.01,
            "num_epochs": 2,
            "batch_size": 8,
            "adam_betas": [0.9, 0.999],
        }
        model_3 = {
            "name": "sign_aware_gated_sae_order_independent",
            "sae_class": "GatedSAE",
            "sign_aware": True,
            "single_activation_offset": False,
            "reset_activation_offsets_on_resample": True,
            "d_hidden": 3,
            "sparsity_coeff": 0.001,
            "learning_rate": 0.01,
            "num_epochs": 2,
            "batch_size": 8,
            "adam_betas": [0.9, 0.999],
        }
        model_names = [model_1["name"], model_2["name"], model_3["name"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_a_path = os.path.join(tmpdir, "config_a.json")
            config_b_path = os.path.join(tmpdir, "config_b.json")
            config_a = {
                **base_config,
                "output_dir": tmpdir,
                "run_id": "order_independence_a",
                "models": [model_1, model_2, model_3],
            }
            config_b = {
                **base_config,
                "output_dir": tmpdir,
                "run_id": "order_independence_b",
                "models": [model_3, model_1, model_2],
            }
            with open(config_a_path, "w") as f:
                json.dump(config_a, f)
            with open(config_b_path, "w") as f:
                json.dump(config_b, f)

            # when
            run_dir_a = run_polarity_dial_experiment(config_a_path)
            run_dir_b = run_polarity_dial_experiment(config_b_path)

            # then
            for model_name in model_names:
                summary_a = self._load_model_calibration_summary(run_dir_a, model_name)
                summary_b = self._load_model_calibration_summary(run_dir_b, model_name)
                self._assert_summaries_almost_equal(summary_a, summary_b)

                metrics_a = self._load_model_training_metrics(run_dir_a, model_name)
                metrics_b = self._load_model_training_metrics(run_dir_b, model_name)
                self.assertEqual(metrics_a["model_seed"], metrics_b["model_seed"])

    def _load_model_calibration_summary(
        self, run_dir: str, model_name: str
    ) -> Dict[str, Any]:
        calibration_path = os.path.join(
            run_dir, "models", model_name, "calibration.json"
        )
        with open(calibration_path, "r") as f:
            calibration = json.load(f)
        return calibration["summary"]

    def _load_model_training_metrics(
        self, run_dir: str, model_name: str
    ) -> Dict[str, Any]:
        metrics_path = os.path.join(
            run_dir, "models", model_name, "training_metrics.json"
        )
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics

    def _assert_summaries_almost_equal(
        self, summary_a: Dict[str, Any], summary_b: Dict[str, Any]
    ) -> None:
        self.assertEqual(set(summary_a.keys()), set(summary_b.keys()))
        for key in summary_a:
            value_a = summary_a[key]
            value_b = summary_b[key]
            if value_a is None or value_b is None:
                self.assertEqual(value_a, value_b)
            elif isinstance(value_a, int) and isinstance(value_b, int):
                self.assertEqual(value_a, value_b)
            else:
                self.assertAlmostEqual(
                    float(value_a),
                    float(value_b),
                    places=7,
                    msg=f"Mismatch for summary['{key}']",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
