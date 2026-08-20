import csv
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

from sgnsae.experiments.llm_benchmark.plotting import generate_plots_for_run


def _build_summary_payload() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": {"llm_id": "dummy-llm", "hook_name": "mlp_out", "layer": 0},
        "aggregates": [
            {
                "variant_slug": "sign_aware_gated_sae",
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "num_seeds": 1,
                "mse_mean": 0.25,
                "mse_se": 0.0,
                "mean_l0_mean": 10.0,
                "mean_l0_se": 0.0,
                "dead_fraction_mean": 0.3,
                "dead_fraction_se": 0.0,
                "loss_recovered_mean": 0.4,
                "loss_recovered_se": 0.0,
            },
            {
                "variant_slug": "sign_aware_gated_sae",
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "num_seeds": 1,
                "mse_mean": 0.1,
                "mse_se": 0.0,
                "mean_l0_mean": 80.0,
                "mean_l0_se": 0.0,
                "dead_fraction_mean": 0.05,
                "dead_fraction_se": 0.0,
                "loss_recovered_mean": 0.9,
                "loss_recovered_se": 0.0,
            },
        ],
        "subruns": [
            {
                "variant_slug": "sign_aware_gated_sae",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "relative_dir": "models/sign_aware_gated_sae/seed_0/lambda_0p01",
                "mean_l0": 10.0,
                "mse": 0.25,
                "loss_recovered": 0.4,
                "dead_fraction": 0.3,
            },
            {
                "variant_slug": "sign_aware_gated_sae",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "relative_dir": "models/sign_aware_gated_sae/seed_0/lambda_0p001",
                "mean_l0": 80.0,
                "mse": 0.1,
                "loss_recovered": 0.9,
                "dead_fraction": 0.05,
            },
        ],
    }


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


class TestLLMBenchmarkMatchedTablesContract(unittest.TestCase):
    def test_generate_plots_writes_matched_tables_with_expected_contract(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "comparison_summary.json")
            with open(summary_path, "w") as file:
                json.dump(_build_summary_payload(), file, indent=2)

            # when
            result = generate_plots_for_run(output_dir=tmpdir)

            # then
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "matched_points.json")))
            table_paths = result.get("table_paths", {})
            self.assertIn("matched_l0_points", table_paths)
            self.assertIn("matched_lr_points", table_paths)
            matched_l0_path = str(table_paths["matched_l0_points"])
            matched_lr_path = str(table_paths["matched_lr_points"])
            self.assertTrue(os.path.isfile(matched_l0_path))
            self.assertTrue(os.path.isfile(matched_lr_path))

            l0_rows = _read_rows(matched_l0_path)
            lr_rows = _read_rows(matched_lr_path)
            self.assertEqual(len(l0_rows), 2)
            self.assertEqual(len(lr_rows), 3)

            expected_columns = {
                "variant_slug",
                "seed",
                "anchor_type",
                "anchor_value",
                "method",
                "distance_to_target",
                "mean_l0",
                "mse",
                "loss_recovered",
                "dead_fraction",
                "supporting_relative_dir_1",
                "supporting_relative_dir_2",
            }
            self.assertEqual(set(l0_rows[0].keys()), expected_columns)
            self.assertEqual(set(lr_rows[0].keys()), expected_columns)
            self.assertTrue(all(row["anchor_type"] == "l0" for row in l0_rows))
            self.assertTrue(all(row["anchor_type"] == "lr" for row in lr_rows))


if __name__ == "__main__":
    unittest.main()
