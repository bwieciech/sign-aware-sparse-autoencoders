import csv
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from sgnsae.experiments.llm_benchmark import plotting as plotting_module


def _history_entry(
    step: int,
    positive_fraction: float,
    negative_fraction: float,
    bidirectional_fraction: float,
    flip_rate: float,
    flip_indicator: float,
) -> Dict[str, Any]:
    return {
        "step": int(step),
        "polarity_positive_fraction_nonzero": float(positive_fraction),
        "polarity_negative_fraction_nonzero": float(negative_fraction),
        "polarity_bidirectional_usage_fraction": float(bidirectional_fraction),
        "polarity_dominance_flip_rate": float(flip_rate),
        "polarity_dominance_flip_indicator": float(flip_indicator),
    }


def _write_training_metrics(subrun_dir: str, history: List[Dict[str, Any]]) -> None:
    os.makedirs(subrun_dir, exist_ok=True)
    with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
        json.dump({"history": history}, file, indent=2)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


class TestPolarityUsageStabilityPlotting(unittest.TestCase):
    def test_polarity_usage_helpers_cover_additional_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            relative_dir = os.path.join("models", "variant_a", "seed_0", "lambda_1e-03")
            subrun_dir = os.path.join(tmpdir, relative_dir)
            os.makedirs(subrun_dir, exist_ok=True)
            metrics_path = os.path.join(subrun_dir, "training_metrics.json")
            with open(metrics_path, "w") as file:
                json.dump({"history": {}}, file, indent=2)

            # when
            non_list_history = plotting_module._load_last_training_history_entry(
                output_dir=tmpdir,
                relative_dir=relative_dir,
            )

            # then
            self.assertIsNone(non_list_history)

            # given
            with open(metrics_path, "w") as file:
                json.dump({"history": [1, 2]}, file, indent=2)

            # when
            non_dict_last_entry = plotting_module._load_last_training_history_entry(
                output_dir=tmpdir,
                relative_dir=relative_dir,
            )
            rows_for_invalid_subruns = (
                plotting_module._collect_polarity_usage_stability_rows(
                    output_dir=tmpdir,
                    summary={"subruns": {"bad": "payload"}},
                )
            )
            skipped_invalid_row = (
                plotting_module._collect_polarity_usage_stability_rows(
                    output_dir=tmpdir,
                    summary={
                        "subruns": [
                            {
                                "variant_slug": "variant_a",
                                "seed": True,
                                "sweep_param": "lambda",
                                "sweep_value": 1e-3,
                                "relative_dir": relative_dir,
                            }
                        ]
                    },
                )
            )
            aggregate_skips_invalid_rows = (
                plotting_module._aggregate_polarity_usage_stability_rows(
                    rows=[
                        {
                            "variant_slug": "variant_a",
                            "sweep_param": "lambda",
                            "sweep_value": True,
                        }
                    ]
                )
            )

            # then
            self.assertIsNone(non_dict_last_entry)
            self.assertEqual(rows_for_invalid_subruns, [])
            self.assertEqual(skipped_invalid_row, [])
            self.assertEqual(aggregate_skips_invalid_rows, [])

    def test_collect_and_aggregate_polarity_rows_skip_invalid_entries(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_relative_dir_a = os.path.join(
                "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            valid_relative_dir_b = os.path.join(
                "models", "variant_a", "seed_1", "lambda_1e-03"
            )
            invalid_relative_dir = os.path.join(
                "models", "variant_a", "seed_2", "lambda_1e-03"
            )
            _write_training_metrics(
                subrun_dir=os.path.join(tmpdir, valid_relative_dir_a),
                history=[
                    _history_entry(
                        step=1,
                        positive_fraction=0.5,
                        negative_fraction=0.5,
                        bidirectional_fraction=0.75,
                        flip_rate=0.2,
                        flip_indicator=1.0,
                    )
                ],
            )
            _write_training_metrics(
                subrun_dir=os.path.join(tmpdir, valid_relative_dir_b),
                history=[
                    _history_entry(
                        step=1,
                        positive_fraction=0.25,
                        negative_fraction=0.75,
                        bidirectional_fraction=0.25,
                        flip_rate=0.0,
                        flip_indicator=0.0,
                    )
                ],
            )
            _write_training_metrics(
                subrun_dir=os.path.join(tmpdir, invalid_relative_dir),
                history=[{"step": 1, "polarity_positive_fraction_nonzero": 0.2}],
            )
            summary = {
                "subruns": [
                    {
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_dir": valid_relative_dir_a,
                    },
                    {
                        "variant_slug": "variant_a",
                        "seed": 1,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_dir": valid_relative_dir_b,
                    },
                    {
                        "variant_slug": "variant_a",
                        "seed": 2,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_dir": invalid_relative_dir,
                    },
                    "bad",
                ]
            }

            # when
            rows = plotting_module._collect_polarity_usage_stability_rows(
                output_dir=tmpdir,
                summary=summary,
            )
            aggregates = plotting_module._aggregate_polarity_usage_stability_rows(rows)

            # then
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(aggregates), 1)
            aggregate = aggregates[0]
            self.assertEqual(str(aggregate["variant_slug"]), "variant_a")
            self.assertEqual(str(aggregate["sweep_param"]), "lambda")
            self.assertAlmostEqual(float(aggregate["sweep_value"]), 1e-3, places=12)
            self.assertEqual(int(aggregate["num_seeds"]), 2)
            self.assertAlmostEqual(
                float(aggregate["polarity_positive_fraction_nonzero_mean"]),
                0.375,
                places=8,
            )
            self.assertAlmostEqual(
                float(aggregate["polarity_dominance_flip_rate_mean"]),
                0.1,
                places=8,
            )

    def test_plot_polarity_usage_stability_writes_plot_and_table(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            relative_dir_a = os.path.join(
                "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            relative_dir_b = os.path.join(
                "models", "variant_a", "seed_1", "lambda_1e-03"
            )
            relative_dir_c = os.path.join(
                "models", "variant_a", "seed_0", "lambda_1e-04"
            )
            _write_training_metrics(
                subrun_dir=os.path.join(tmpdir, relative_dir_a),
                history=[
                    _history_entry(
                        step=1,
                        positive_fraction=0.6,
                        negative_fraction=0.4,
                        bidirectional_fraction=0.7,
                        flip_rate=0.3,
                        flip_indicator=1.0,
                    )
                ],
            )
            _write_training_metrics(
                subrun_dir=os.path.join(tmpdir, relative_dir_b),
                history=[
                    _history_entry(
                        step=1,
                        positive_fraction=0.4,
                        negative_fraction=0.6,
                        bidirectional_fraction=0.5,
                        flip_rate=0.1,
                        flip_indicator=1.0,
                    )
                ],
            )
            _write_training_metrics(
                subrun_dir=os.path.join(tmpdir, relative_dir_c),
                history=[
                    _history_entry(
                        step=1,
                        positive_fraction=0.8,
                        negative_fraction=0.2,
                        bidirectional_fraction=0.2,
                        flip_rate=0.0,
                        flip_indicator=0.0,
                    )
                ],
            )
            summary = {
                "subruns": [
                    {
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_dir": relative_dir_a,
                    },
                    {
                        "variant_slug": "variant_a",
                        "seed": 1,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_dir": relative_dir_b,
                    },
                    {
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-4,
                        "relative_dir": relative_dir_c,
                    },
                ]
            }
            output_path = os.path.join(tmpdir, "plots", "polarity_usage_stability.png")

            # when
            table_path = plotting_module._plot_polarity_usage_stability(
                output_dir=tmpdir,
                summary=summary,
                output_path=output_path,
            )

            # then
            self.assertEqual(
                table_path,
                os.path.join(tmpdir, "tables", "polarity_usage_stability.csv"),
            )
            self.assertTrue(os.path.isfile(output_path))
            self.assertTrue(os.path.isfile(table_path))
            rows = _read_csv_rows(table_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["variant_slug"], "variant_a")
            self.assertEqual(rows[0]["sweep_param"], "lambda")
            self.assertIn("polarity_dominance_flip_rate_mean", rows[0])

    def test_plot_polarity_usage_stability_handles_no_data(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = {
                "subruns": [
                    {
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_dir": "models/variant_a/seed_0/lambda_1e-03",
                    }
                ]
            }
            output_path = os.path.join(tmpdir, "plots", "polarity_usage_stability.png")

            # when
            with patch.object(
                plotting_module,
                "_save_no_data_plot",
                wraps=plotting_module._save_no_data_plot,
            ) as save_no_data_plot_call:
                table_path = plotting_module._plot_polarity_usage_stability(
                    output_dir=tmpdir,
                    summary=summary,
                    output_path=output_path,
                )

            # then
            self.assertTrue(os.path.isfile(output_path))
            self.assertTrue(os.path.isfile(table_path))
            rows = _read_csv_rows(table_path)
            self.assertEqual(rows, [])
            self.assertEqual(save_no_data_plot_call.call_count, 1)
            self.assertEqual(
                str(save_no_data_plot_call.call_args.kwargs["message"]),
                "No polarity-usage stability data available "
                "(missing training history or required keys).",
            )

    def test_generate_plots_for_run_returns_polarity_usage_table_path(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            relative_dir = os.path.join("models", "variant_a", "seed_0", "lambda_1e-03")
            _write_training_metrics(
                subrun_dir=os.path.join(tmpdir, relative_dir),
                history=[
                    _history_entry(
                        step=1,
                        positive_fraction=0.5,
                        negative_fraction=0.5,
                        bidirectional_fraction=0.4,
                        flip_rate=0.2,
                        flip_indicator=1.0,
                    )
                ],
            )
            summary = {
                "aggregates": [
                    {
                        "variant_slug": "variant_a",
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "mse_mean": 0.5,
                        "mean_l0_mean": 10.0,
                        "dead_fraction_mean": 0.2,
                        "loss_recovered_mean": 0.6,
                    }
                ],
                "subruns": [
                    {
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "mean_l0": 10.0,
                        "mse": 0.5,
                        "dead_fraction": 0.2,
                        "loss_recovered": 0.6,
                        "relative_dir": relative_dir,
                    }
                ],
            }
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(summary, file, indent=2)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.plotting.compute_and_persist_matched_points",
                return_value={"aggregate": [], "per_seed": [], "anchors": {}},
            ):
                result = plotting_module.generate_plots_for_run(output_dir=tmpdir)

            # then
            self.assertIn("polarity_usage_stability", result["plot_paths"])
            self.assertIn("polarity_usage_stability", result["table_paths"])
            self.assertTrue(
                os.path.isfile(result["plot_paths"]["polarity_usage_stability"])
            )
            self.assertTrue(
                os.path.isfile(result["table_paths"]["polarity_usage_stability"])
            )
