import json
import os
import tempfile
import unittest
import warnings
from typing import Any, Dict
from unittest.mock import patch

from sgnsae.experiments.toy_model_geometry_validation import runner as runner_module


class TestToyModelGeometryValidationRunner(unittest.TestCase):
    def _stub_leaf_experiment(self, _config: Any, run_dir: str) -> Dict[str, Any]:
        summary_path = os.path.join(run_dir, "comparison_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"status": "stub"}, f)
        return {"status": "stub"}

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

            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "pair_consolidation_sweep.json"))
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(run_dir, "plots", "pair_consolidation_sweep.png")
                )
            )
            with open(os.path.join(run_dir, "pair_consolidation_sweep.json"), "r") as f:
                sweep_payload = json.load(f)
            self.assertEqual(
                sweep_payload["sweep"]["pair_correlation_grid"], [-1.0, 0.0]
            )
            self.assertEqual(len(sweep_payload["points"]), 2)
            self.assertIn("relu_sae", sweep_payload["points"][0]["per_model"])
            first_metrics = sweep_payload["points"][0]["per_model"]["relu_sae"]
            self.assertIn("pair_consolidation_rate", first_metrics)
            self.assertIn("rate_over_evaluable_pairs", first_metrics)

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

    def test_run_experiment_generates_bijump_histogram_for_sign_aware_models(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_sign_aware")
            raw_config["models"][0]["name"] = "sign_sae"
            raw_config["models"][0]["sign_aware"] = True

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            run_dir = runner_module.run_toy_model_geometry_validation_experiment(
                config_path
            )

            # then
            model_dir = os.path.join(run_dir, "models", "sign_sae")
            with open(os.path.join(model_dir, "evaluation.json"), "r") as f:
                evaluation = json.load(f)
            self.assertIsNotNone(evaluation["histogram_latent"])
            latent_index = int(evaluation["histogram_latent"]["latent_index"])
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        model_dir,
                        "plots",
                        f"bi_jump_hist_sign_sae_latent{latent_index}.png",
                    )
                )
            )

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
            self.assertTrue(
                os.path.isfile(
                    os.path.join(model_dir, "plots", "unit_circle_relu_sae.png")
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
            self.assertEqual(evaluation["pair_consolidation_rate"], 0.0)
            self.assertIsNone(
                evaluation["pair_consolidation_rate_over_evaluable_pairs"]
            )
            self.assertIsNone(evaluation["rate_over_evaluable_pairs"])
            self.assertIsNone(evaluation["histogram_latent"])

    def test_run_experiment_supports_seed_list_and_appends_run_id(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            raw_config["data"]["seed"] = [0, 3]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            run_dirs = runner_module.run_toy_model_geometry_validation_experiment(
                config_path
            )

            # then
            self.assertIsInstance(run_dirs, list)
            expected_dirs = [
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_seed0"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_seed3"
                ),
            ]
            self.assertEqual(run_dirs, expected_dirs)

            for run_dir, expected_seed, expected_run_id in zip(
                run_dirs,
                [0, 3],
                ["unittest_run_seed0", "unittest_run_seed3"],
            ):
                self.assertTrue(os.path.isdir(run_dir))
                with open(os.path.join(run_dir, "config.json"), "r") as f:
                    persisted_config = json.load(f)
                self.assertEqual(persisted_config["run_id"], expected_run_id)
                self.assertEqual(persisted_config["data"]["seed"], expected_seed)

    def test_run_experiment_supports_active_pairs_per_sample_list(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            raw_config["data"]["seed"] = 1
            raw_config["data"]["num_pairs"] = 4
            raw_config["data"]["active_pairs_per_sample"] = [1, 2, 3]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module,
                "_run_leaf_experiment",
                side_effect=self._stub_leaf_experiment,
            ), patch.object(
                runner_module, "persist_environment_metadata", return_value=None
            ):
                run_dirs = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            self.assertIsInstance(run_dirs, list)
            expected_dirs = [
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_ap1"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_ap2"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_ap3"
                ),
            ]
            self.assertEqual(run_dirs, expected_dirs)
            for run_dir, expected_ap, expected_run_id in zip(
                run_dirs,
                [1, 2, 3],
                ["unittest_run_ap1", "unittest_run_ap2", "unittest_run_ap3"],
            ):
                with open(os.path.join(run_dir, "config.json"), "r") as f:
                    persisted_config = json.load(f)
                self.assertEqual(persisted_config["run_id"], expected_run_id)
                self.assertEqual(persisted_config["data"]["seed"], 1)
                self.assertEqual(
                    persisted_config["data"]["active_pairs_per_sample"], expected_ap
                )

    def test_run_experiment_supports_seed_and_active_pairs_per_sample_lists(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            raw_config["data"]["seed"] = [0, 1]
            raw_config["data"]["num_pairs"] = 4
            raw_config["data"]["active_pairs_per_sample"] = [1, 2, 3]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module,
                "_run_leaf_experiment",
                side_effect=self._stub_leaf_experiment,
            ), patch.object(
                runner_module, "persist_environment_metadata", return_value=None
            ):
                run_dirs = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            self.assertIsInstance(run_dirs, list)
            expected_dirs = [
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_seed0_ap1"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_seed0_ap2"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_seed0_ap3"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_seed1_ap1"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_seed1_ap2"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "unittest_run_seed1_ap3"
                ),
            ]
            self.assertEqual(run_dirs, expected_dirs)
            expected_pairs = [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
            ]
            for run_dir, (expected_seed, expected_ap) in zip(run_dirs, expected_pairs):
                with open(os.path.join(run_dir, "config.json"), "r") as f:
                    persisted_config = json.load(f)
                self.assertEqual(persisted_config["data"]["seed"], expected_seed)
                self.assertEqual(
                    persisted_config["data"]["active_pairs_per_sample"], expected_ap
                )

    def test_run_experiment_seed_list_with_scalar_active_pairs_per_sample_keeps_seed_suffix(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            raw_config["data"]["seed"] = [0, 1]
            raw_config["data"]["active_pairs_per_sample"] = 1
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module,
                "_run_leaf_experiment",
                side_effect=self._stub_leaf_experiment,
            ), patch.object(
                runner_module, "persist_environment_metadata", return_value=None
            ):
                run_dirs = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            self.assertEqual(
                run_dirs,
                [
                    os.path.join(
                        tmpdir, "toy_model_geometry_validation", "unittest_run_seed0"
                    ),
                    os.path.join(
                        tmpdir, "toy_model_geometry_validation", "unittest_run_seed1"
                    ),
                ],
            )
            with open(os.path.join(run_dirs[1], "config.json"), "r") as f:
                persisted_config = json.load(f)
            self.assertEqual(persisted_config["run_id"], "unittest_run_seed1")
            self.assertEqual(persisted_config["data"]["seed"], 1)
            self.assertEqual(persisted_config["data"]["active_pairs_per_sample"], 1)

    def test_run_experiment_active_pairs_per_sample_list_raises_when_run_dir_exists(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            raw_config["data"]["active_pairs_per_sample"] = [1, 2]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            existing_dir = os.path.join(
                tmpdir, "toy_model_geometry_validation", "unittest_run_ap2"
            )
            os.makedirs(existing_dir, exist_ok=True)

            # when & then
            with self.assertRaises(ValueError):
                _ = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

    def test_run_experiment_active_pairs_per_sample_list_suffixes_generated_run_id_on_collision(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id=None)
            raw_config["data"]["seed"] = 1
            raw_config["data"]["active_pairs_per_sample"] = [1]
            raw_config["device"] = "cpu"
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module,
                "_generate_run_id",
                return_value="fixed_id",
            ), patch.object(
                runner_module,
                "_run_leaf_experiment",
                side_effect=self._stub_leaf_experiment,
            ), patch.object(
                runner_module, "persist_environment_metadata", return_value=None
            ):
                run_parent = os.path.join(tmpdir, "toy_model_geometry_validation")
                os.makedirs(os.path.join(run_parent, "fixed_id_ap1"), exist_ok=True)
                os.makedirs(os.path.join(run_parent, "fixed_id_1_ap1"), exist_ok=True)
                run_dirs = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            self.assertEqual(
                run_dirs,
                [
                    os.path.join(
                        tmpdir, "toy_model_geometry_validation", "fixed_id_2_ap1"
                    )
                ],
            )

    def test_run_experiment_active_pairs_per_sample_list_uses_generated_run_id_when_no_collision(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id=None)
            raw_config["data"]["seed"] = 1
            raw_config["data"]["active_pairs_per_sample"] = [1]
            raw_config["device"] = "cpu"
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module,
                "_generate_run_id",
                return_value="fixed_id",
            ), patch.object(
                runner_module,
                "_run_leaf_experiment",
                side_effect=self._stub_leaf_experiment,
            ), patch.object(
                runner_module, "persist_environment_metadata", return_value=None
            ):
                run_dirs = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            self.assertEqual(
                run_dirs,
                [os.path.join(tmpdir, "toy_model_geometry_validation", "fixed_id_ap1")],
            )

    def test_run_experiment_seed_list_raises_when_any_requested_run_dir_exists(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            raw_config["data"]["seed"] = [0, 3]
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            existing_dir = os.path.join(
                tmpdir, "toy_model_geometry_validation", "unittest_run_seed3"
            )
            os.makedirs(existing_dir, exist_ok=True)

            # when & then
            with self.assertRaises(ValueError):
                _ = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

    def test_run_experiment_seed_list_suffixes_generated_run_id_on_collision(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id=None)
            raw_config["data"]["seed"] = [0, 3]
            raw_config["device"] = "cpu"
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module, "_generate_run_id", return_value="fixed_id"
            ):
                run_parent = os.path.join(tmpdir, "toy_model_geometry_validation")
                os.makedirs(os.path.join(run_parent, "fixed_id_seed0"), exist_ok=True)
                os.makedirs(os.path.join(run_parent, "fixed_id_1_seed0"), exist_ok=True)
                run_dirs = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            self.assertIsInstance(run_dirs, list)
            expected_dirs = [
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "fixed_id_2_seed0"
                ),
                os.path.join(
                    tmpdir, "toy_model_geometry_validation", "fixed_id_2_seed3"
                ),
            ]
            self.assertEqual(run_dirs, expected_dirs)
            self.assertTrue(os.path.isdir(expected_dirs[0]))
            self.assertTrue(os.path.isdir(expected_dirs[1]))

    def test_run_experiment_seed_list_uses_generated_run_id_when_no_collision(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id=None)
            raw_config["data"]["seed"] = [0, 3]
            raw_config["device"] = "cpu"
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            with patch.object(
                runner_module, "_generate_run_id", return_value="fixed_id"
            ):
                run_dirs = runner_module.run_toy_model_geometry_validation_experiment(
                    config_path
                )

            # then
            self.assertIsInstance(run_dirs, list)
            expected_dirs = [
                os.path.join(tmpdir, "toy_model_geometry_validation", "fixed_id_seed0"),
                os.path.join(tmpdir, "toy_model_geometry_validation", "fixed_id_seed3"),
            ]
            self.assertEqual(run_dirs, expected_dirs)
            self.assertTrue(os.path.isdir(expected_dirs[0]))
            self.assertTrue(os.path.isdir(expected_dirs[1]))

    def test_warn_if_width_name_mismatch_emits_warning(self) -> None:
        # given
        warnings.simplefilter("always")

        # when / then
        with self.assertWarns(UserWarning):
            runner_module._warn_if_width_name_mismatch(
                model_name="baseline_k",
                d_hidden=5,
                num_pairs=4,
            )

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

    def test_run_experiment_raises_when_cuda_unavailable(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir, run_id="unittest_run")
            raw_config["device"] = "cuda"
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when & then
            with patch(
                "sgnsae.experiments.device.torch.cuda.is_available",
                return_value=False,
            ):
                with self.assertRaises(ValueError):
                    _ = runner_module.run_toy_model_geometry_validation_experiment(
                        config_path
                    )

    def test_resolve_device_accepts_cuda_index_when_available(self) -> None:
        # given
        with patch(
            "sgnsae.experiments.device.torch.cuda.is_available", return_value=True
        ), patch("sgnsae.experiments.device.torch.cuda.device_count", return_value=2):
            # when
            device = runner_module._resolve_device("cuda:1")

            # then
            self.assertEqual(device, "cuda:1")

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
