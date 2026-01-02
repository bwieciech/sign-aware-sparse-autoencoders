import json
import os
import tempfile
import unittest
from typing import Any, Dict

from sgnsae.experiments.directional_deviation.runner import (
    run_directional_deviation_experiment,
)


class TestDirectionalDeviationSmoke(unittest.TestCase):
    def _load_tiny_config_template(self) -> Dict[str, Any]:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        template_path = os.path.join(
            repo_root, "configs", "directional_deviation_tiny.json"
        )
        with open(template_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                "Expected configs/directional_deviation_tiny.json to be a JSON object."
            )
        return data

    def test_run_directional_deviation_experiment_stream_mode_writes_expected_artifacts_and_plots(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = self._load_tiny_config_template()
            raw_config["output_dir"] = tmpdir
            raw_config["run_id"] = "unittest_smoke"
            raw_config["data"]["anomaly_eval_mode"] = "stream"

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(raw_config, f)

            # when
            run_dir = run_directional_deviation_experiment(config_path)

            # then
            expected_run_dir = os.path.join(
                tmpdir, "directional_deviation", raw_config["run_id"]
            )
            self.assertEqual(run_dir, expected_run_dir)
            self.assertTrue(os.path.isdir(run_dir))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "config.json")))
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "comparison_summary.json"))
            )

            data_dir = os.path.join(run_dir, "data")
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "generator_config.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(data_dir, "anomaly_manifest.json"))
            )
            self.assertTrue(os.path.isfile(os.path.join(data_dir, "sampled_data.pt")))

            model_dir = os.path.join(run_dir, "models", "relu_sae")
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "model.pt")))
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "evaluation.json")))
            self.assertTrue(
                os.path.isfile(
                    os.path.join(model_dir, "thresholds", "score_thresholds.json")
                )
            )

            with open(os.path.join(model_dir, "evaluation.json"), "r") as f:
                evaluation = json.load(f)
            self.assertIsNone(evaluation.get("error"))
            self.assertIsNotNone(evaluation.get("metrics"))

            plots_dir = os.path.join(run_dir, "plots")
            self.assertTrue(os.path.isfile(os.path.join(plots_dir, "leaf_summary.png")))
            self.assertTrue(
                os.path.isfile(os.path.join(plots_dir, "score_histograms.png"))
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(plots_dir, "relu_sae", "score_high_hist.png")
                )
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(plots_dir, "relu_sae", "score_low_hist.png")
                )
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
