import io
import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

from sgnsae.scripts import run_directional_deviation_experiment as script_module


class TestDirectionalDeviationCli(unittest.TestCase):
    def _build_raw_config(self, output_dir: str) -> Dict[str, Any]:
        return {
            "output_dir": output_dir,
            "device": "auto",
            "run_id": "unittest_run",
            "data": {
                "seed": 0,
                "num_train": 2,
                "num_val": 2,
                "num_test_normal": 1,
                "num_test_high_per_channel": 1,
                "num_test_low_per_channel": 1,
                "num_channels": 4,
                "noise_std": 0.0,
                "normal_mode": "standard_normal",
                "anomaly_eval_mode": "stream",
            },
            "anomaly": {
                "delta": 2.0,
                "shift_mode": "single_channel",
            },
            "thresholds": {
                "latent_quantile": 0.99,
                "score_fpr": 0.01,
                "min_validation_samples": 2,
                "deterministic_algorithms": False,
            },
            "models": [
                {
                    "name": "relu_sae",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "single_activation_offset": False,
                    "reset_activation_offsets_on_resample": False,
                    "d_hidden": 2,
                    "sparsity_coeff": 0.01,
                    "learning_rate": 0.1,
                    "num_epochs": 1,
                    "batch_size": 2,
                    "adam_betas": [0.9, 0.999],
                }
            ],
            "wandb_project": "project",
            "wandb_api_key": "SHOULD_BE_REDACTED_IN_ARTIFACTS",
        }

    def test_cli_main_runs_and_writes_redacted_config(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._build_raw_config(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            stdout = io.StringIO()
            argv = ["prog", "--config-path", config_path]

            # when
            with patch.object(script_module.sys, "argv", argv), patch.object(
                script_module.sys, "stdout", stdout
            ):
                script_module.main()

            # then
            output = stdout.getvalue()
            self.assertIn("Loaded Directional Deviation config:", output)
            self.assertIn("Run complete. Artifacts written to:", output)

            expected_run_dir = os.path.join(
                tmpdir, "directional_deviation", raw_config["run_id"]
            )
            self.assertTrue(os.path.isdir(expected_run_dir))

            config_artifact_path = os.path.join(expected_run_dir, "config.json")
            self.assertTrue(os.path.isfile(config_artifact_path))
            with open(config_artifact_path, "r") as f:
                persisted = json.load(f)
            self.assertIsNone(persisted["wandb_api_key"])
            self.assertIn(persisted["device"], {"cpu", "cuda"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

