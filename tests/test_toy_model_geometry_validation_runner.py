import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

from sgnsae.experiments.toy_model_geometry_validation import runner as runner_module


class TestToyModelGeometryValidationRunner(unittest.TestCase):
    def _build_raw_config(self, output_dir: str, run_id: str | None) -> Dict[str, Any]:
        return {
            "output_dir": output_dir,
            "device": "auto",
            "run_id": run_id,
            "data": {
                "seed": 0,
                "num_train": 2,
                "num_val": 2,
                "num_test": 2,
                "num_pairs": 2,
                "direction_mode": "uniform",
                "pair_axis_active_prob": 0.2,
                "pair_feature_prob": 0.5,
                "pair_correlation": -1.0,
                "magnitude_dist": "constant",
                "constant_magnitude": 1.0,
                "noise_std": 0.0,
            },
            "models": [
                {
                    "name": "relu_sae",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "single_activation_offset": False,
                    "reset_activation_offsets_on_resample": False,
                    "d_hidden": 4,
                    "sparsity_coeff": 0.01,
                    "learning_rate": 0.1,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "adam_betas": [0.9, 0.999],
                }
            ],
        }

    def test_run_experiment_supports_pair_correlation_grid_and_persists_leaf_runs(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_sweep")
            raw_config["data"]["pair_correlation"] = None
            raw_config["data"]["pair_correlation_grid"] = [-1.0, 0.0]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            run_dir = runner_module.run_toy_model_geometry_validation_experiment(
                config_path
            )

            # then
            expected_run_dir = os.path.join(
                tmpdir, "toy_model_geometry_validation", "unittest_sweep"
            )
            self.assertEqual(run_dir, expected_run_dir)

            for correlation_value in (-1.0, 0.0):
                leaf_dir = os.path.join(
                    run_dir,
                    f"corr_{runner_module._format_pair_correlation(correlation_value)}",
                )
                self.assertTrue(os.path.isdir(leaf_dir))
                with open(os.path.join(leaf_dir, "config.json"), "r") as f:
                    leaf_config = json.load(f)
                self.assertEqual(
                    leaf_config["data"]["pair_correlation"], correlation_value
                )
                self.assertIsNone(leaf_config["data"]["pair_correlation_grid"])

                generator_config_path = os.path.join(
                    leaf_dir, "data", "generator_config.json"
                )
                self.assertTrue(os.path.isfile(generator_config_path))
                with open(generator_config_path, "r") as f:
                    generator_config = json.load(f)
                self.assertEqual(
                    generator_config["pair_correlation"], correlation_value
                )
                self.assertIsNone(generator_config["pair_correlation_grid"])

    def test_run_sweep_data_generation_rejects_missing_grid(self) -> None:
        # given
        from sgnsae.experiments.toy_model_geometry_validation import (
            config as config_module,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)
            config = config_module.load_experiment_config(config_path)

            # when & then
            with self.assertRaises(ValueError):
                _ = runner_module._run_sweep_experiment(config, tmpdir)

    def test_extract_final_metric_returns_none_when_last_value_is_none(self) -> None:
        # given / when / then
        self.assertIsNone(runner_module._extract_final_metric([None]))

    def test_run_experiment_creates_run_dir_and_writes_config_and_environment(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            run_dir = runner_module.run_toy_model_geometry_validation_experiment(
                config_path
            )

            # then
            expected_run_dir = os.path.join(
                tmpdir, "toy_model_geometry_validation", "unittest_run"
            )
            self.assertEqual(run_dir, expected_run_dir)
            self.assertTrue(os.path.isdir(run_dir))

            config_artifact_path = os.path.join(run_dir, "config.json")
            environment_path = os.path.join(run_dir, "environment.json")
            comparison_summary_path = os.path.join(run_dir, "comparison_summary.json")
            self.assertTrue(os.path.isfile(config_artifact_path))
            self.assertTrue(os.path.isfile(environment_path))
            self.assertTrue(os.path.isfile(comparison_summary_path))

            model_dir = os.path.join(run_dir, "models", "relu_sae")
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "model.pt")))
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "model_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(model_dir, "training_metrics.json"))
            )
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "evaluation.json")))
            self.assertTrue(
                os.path.isfile(
                    os.path.join(model_dir, "plots", "training_loss_relu_sae.png")
                )
            )

            with open(config_artifact_path, "r") as f:
                persisted_config = json.load(f)
            self.assertEqual(persisted_config["run_id"], "unittest_run")
            self.assertIn(persisted_config["device"], {"cpu", "cuda"})
            self.assertNotEqual(persisted_config["device"], "auto")

            with open(environment_path, "r") as f:
                environment_metadata = json.load(f)
            for key in ("python", "os", "torch", "dependencies"):
                self.assertIn(key, environment_metadata)
            for dependency in ("numpy", "scipy", "matplotlib", "einops"):
                self.assertIn(dependency, environment_metadata["dependencies"])

            with open(os.path.join(model_dir, "evaluation.json"), "r") as f:
                evaluation = json.load(f)
            self.assertEqual(evaluation["status"], "ok")
            self.assertEqual(evaluation["num_pairs_total"], 2)
            self.assertEqual(evaluation["num_pairs_evaluable"], 0)
            self.assertEqual(evaluation["num_pairs_consolidated"], 0)
            self.assertIsNone(
                evaluation["pair_consolidation_rate_over_evaluable_pairs"]
            )
            self.assertIsNone(evaluation["histogram_latent"])

    def test_run_experiment_respects_output_dir_when_already_parent(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "toy_model_geometry_validation")
            raw_config = self._build_raw_config(output_dir, run_id="unittest_run")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            run_dir = runner_module.run_toy_model_geometry_validation_experiment(
                config_path
            )

            # then
            self.assertEqual(run_dir, os.path.join(output_dir, "unittest_run"))

    def test_run_experiment_raises_when_requested_run_dir_already_exists(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            existing_dir = os.path.join(
                tmpdir, "toy_model_geometry_validation", "unittest_run"
            )
            os.makedirs(existing_dir, exist_ok=True)

            # when & then
            with self.assertRaises(ValueError):
                _ = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

    def test_run_experiment_suffixes_generated_run_id_on_collision(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id=None)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module, "_generate_run_id", return_value="fixed_id"
            ):
                run_parent = os.path.join(tmpdir, "toy_model_geometry_validation")
                os.makedirs(os.path.join(run_parent, "fixed_id"), exist_ok=True)
                os.makedirs(os.path.join(run_parent, "fixed_id_1"), exist_ok=True)
                run_dir = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            self.assertTrue(run_dir.endswith(os.path.join("fixed_id_2")))
            self.assertTrue(os.path.isdir(run_dir))

    def test_run_experiment_falls_back_to_cpu_when_cuda_unavailable(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            raw_config["device"] = "cuda"
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module.torch.cuda, "is_available", return_value=False
            ):
                run_dir = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            with open(os.path.join(run_dir, "config.json"), "r") as f:
                persisted_config = json.load(f)
            self.assertEqual(persisted_config["device"], "cpu")

    def test_run_experiment_generates_run_id_when_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id=None)
            raw_config["device"] = "cpu"
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            run_dir = runner_module.run_toy_model_geometry_validation_experiment(
                config_path
            )

            # then
            self.assertTrue(
                run_dir.startswith(
                    os.path.join(tmpdir, "toy_model_geometry_validation")
                )
            )
            self.assertTrue(os.path.isdir(run_dir))
            with open(os.path.join(run_dir, "config.json"), "r") as f:
                persisted_config = json.load(f)
            self.assertEqual(persisted_config["run_id"], os.path.basename(run_dir))
            self.assertEqual(persisted_config["device"], "cpu")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
