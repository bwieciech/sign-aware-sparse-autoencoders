import io
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from sgnsae.scripts import run_toy_model_geometry_validation_experiment as script_module


def _smoke_config_template() -> Dict[str, Any]:
    shared_model_settings: Dict[str, Any] = {
        "sae_class": "GatedSAE",
        "single_activation_offset": False,
        "reset_activation_offsets_on_resample": False,
        "sparsity_coeff": 0.01,
        "learning_rate": 0.01,
        "num_epochs": 3,
        "batch_size": 256,
        "adam_betas": [0.9, 0.999],
    }
    return {
        "output_dir": "assets/toy_model_geometry_validation_runs",
        "device": "cpu",
        "run_id": "toy_model_geometry_validation_smoke",
        "verbose": False,
        "log_every_n_steps": 1000,
        "log_every_n_epochs": 1,
        "sample_preview_count": 1024,
        "data": {
            "seed": 0,
            "num_train": 2000,
            "num_val": 512,
            "num_test": 512,
            "num_pairs": 8,
            "direction_mode": "uniform",
            "pair_axis_active_prob": 0.2,
            "pair_feature_prob": 0.5,
            "pair_correlation": -1.0,
            "magnitude_dist": "lognormal",
            "lognormal_mu": 0.0,
            "lognormal_sigma": 0.25,
            "noise_std": 0.01,
        },
        "evaluation": {
            "min_axis_abs_cos": 0.9,
            "min_activation_corr": 0.8,
            "min_regime_samples": 100,
            "hist_num_bins": 80,
        },
        "models": [
            {
                "name": "gated_baseline_2k",
                "sign_aware": False,
                "d_hidden": 16,
                **shared_model_settings,
            },
            {
                "name": "gated_sign_aware_2k",
                "sign_aware": True,
                "d_hidden": 16,
                **shared_model_settings,
            },
            {
                "name": "gated_sign_aware_k",
                "sign_aware": True,
                "d_hidden": 8,
                **shared_model_settings,
            },
        ],
    }


class TestToyModelGeometryValidationSmoke(unittest.TestCase):
    def test_smoke_config_run_produces_expected_artifacts(self) -> None:
        # given
        raw_config: Dict[str, Any] = _smoke_config_template()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "outputs")
            raw_config["output_dir"] = output_dir
            raw_config["run_id"] = "unittest_smoke"
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            stdout = io.StringIO()
            argv = ["prog", "--config-path", config_path]
            with (
                patch.object(script_module.sys, "argv", argv),
                patch.object(script_module.sys, "stdout", stdout),
            ):
                script_module.main()

            # then
            run_dir = None
            marker = "Run complete. Artifacts written to:"
            for line in stdout.getvalue().splitlines():
                if line.startswith(marker):
                    run_dir = line[len(marker) :].strip()
                    break
            self.assertIsNotNone(run_dir)

            expected_run_dir = os.path.join(
                output_dir, "toy_model_geometry_validation", raw_config["run_id"]
            )
            self.assertEqual(run_dir, expected_run_dir)
            self.assertTrue(os.path.isdir(run_dir))

            for filename in (
                "config.json",
                "environment.json",
                "comparison_summary.json",
            ):
                self.assertTrue(os.path.isfile(os.path.join(run_dir, filename)))

            data_dir = os.path.join(run_dir, "data")
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "generator_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "feature_directions.pt"))
            )
            self.assertTrue(os.path.isfile(os.path.join(data_dir, "sampled_data.pt")))
            for split in ("train", "val", "test"):
                self.assertTrue(
                    os.path.isfile(os.path.join(data_dir, f"coeffs_{split}.pt"))
                )
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(data_dir, f"pair_active_mask_{split}.pt")
                    )
                )

            model_names: List[str] = [model["name"] for model in raw_config["models"]]
            for model_entry in raw_config["models"]:
                model_name = model_entry["name"]
                model_dir = os.path.join(run_dir, "models", model_name)
                self.assertTrue(os.path.isfile(os.path.join(model_dir, "model.pt")))
                self.assertTrue(
                    os.path.isfile(os.path.join(model_dir, "model_config.json"))
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(model_dir, "training_metrics.json"))
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(model_dir, "evaluation.json"))
                )

                plots_dir = os.path.join(model_dir, "plots")
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(plots_dir, f"training_loss_{model_name}.png")
                    )
                )
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(plots_dir, f"unit_circle_{model_name}.png")
                    )
                )

                if bool(model_entry.get("sign_aware", False)):
                    with open(os.path.join(model_dir, "evaluation.json"), "r") as f:
                        evaluation = json.load(f)
                    self.assertIsNotNone(evaluation["histogram_latent"])
                    latent_index = int(evaluation["histogram_latent"]["latent_index"])
                    self.assertTrue(
                        os.path.isfile(
                            os.path.join(
                                plots_dir,
                                f"bi_jump_hist_{model_name}_latent{latent_index}.png",
                            )
                        )
                    )

            with open(os.path.join(run_dir, "comparison_summary.json"), "r") as f:
                comparison_summary = json.load(f)
            self.assertEqual(set(comparison_summary["models"]), set(model_names))
            self.assertEqual(
                set(comparison_summary["per_model"].keys()), set(model_names)
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
