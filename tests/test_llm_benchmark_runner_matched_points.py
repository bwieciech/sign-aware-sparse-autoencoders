import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig
from sgnsae.experiments.llm_benchmark.runner import _run_summarize_stage


def _config_payload(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": 0,
        "llm": {"model_id": "sshleifer/tiny-gpt2"},
        "data": {
            "backend": "local_jsonl",
            "data_files": ["dummy.jsonl"],
            "text_field": "text",
        },
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "variants": [
            {
                "name": "variant_a",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": 1e-3,
            }
        ],
    }


def _comparison_summary_payload() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": {"llm_id": "dummy-llm", "hook_name": "mlp_out", "layer": 0},
        "aggregates": [
            {
                "variant_slug": "variant_a",
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "num_seeds": 1,
                "mse_mean": 0.3,
                "loss_recovered_mean": 0.4,
                "mean_l0_mean": 10.0,
                "dead_fraction_mean": 0.2,
            },
            {
                "variant_slug": "variant_a",
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "num_seeds": 1,
                "mse_mean": 0.1,
                "loss_recovered_mean": 0.8,
                "mean_l0_mean": 80.0,
                "dead_fraction_mean": 0.05,
            },
        ],
        "subruns": [
            {
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "relative_dir": "models/variant_a/seed_0/lambda_0p01",
                "mean_l0": 10.0,
                "mse": 0.3,
                "loss_recovered": 0.4,
                "dead_fraction": 0.2,
            },
            {
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "relative_dir": "models/variant_a/seed_0/lambda_0p001",
                "mean_l0": 80.0,
                "mse": 0.1,
                "loss_recovered": 0.8,
                "dead_fraction": 0.05,
            },
        ],
    }


class TestLLMBenchmarkRunnerMatchedPoints(unittest.TestCase):
    def test_run_summarize_persists_matched_points_artifact(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _config_payload(output_dir=tmpdir)
            )
            summary_payload = _comparison_summary_payload()

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value=summary_payload,
            ):
                summarize_result = _run_summarize_stage(
                    config=config,
                    resolved_config={},
                    output_dir=tmpdir,
                )

            # then
            results_summary_path = str(summarize_result["results_summary_path"])
            self.assertTrue(os.path.isfile(results_summary_path))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "matched_points.json")))
            with open(results_summary_path, "r") as file:
                results_summary = json.load(file)
            self.assertIn("matched_points", results_summary)
            self.assertIn("matched_points_path", results_summary)
            self.assertEqual(
                os.path.basename(str(results_summary["matched_points_path"])),
                "matched_points.json",
            )

    def test_run_summarize_rebuilds_when_comparison_summary_is_not_dict(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _config_payload(output_dir=tmpdir)
            )
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump([], file)
            rebuilt_summary = _comparison_summary_payload()

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value=rebuilt_summary,
            ) as build_mock:
                summarize_result = _run_summarize_stage(
                    config=config,
                    resolved_config={},
                    output_dir=tmpdir,
                )

            # then
            self.assertTrue(
                os.path.isfile(str(summarize_result["results_summary_path"]))
            )
            build_mock.assert_called_once()

    def test_run_summarize_rebuilds_when_comparison_summary_is_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _config_payload(output_dir=tmpdir)
            )
            rebuilt_summary = _comparison_summary_payload()
            rebuilt_summary["aggregates"] = "bad"

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value=rebuilt_summary,
            ) as build_mock:
                summarize_result = _run_summarize_stage(
                    config=config,
                    resolved_config={},
                    output_dir=tmpdir,
                )

            # then
            self.assertTrue(
                os.path.isfile(str(summarize_result["results_summary_path"]))
            )
            build_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
