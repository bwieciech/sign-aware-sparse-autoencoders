import csv
import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

from sgnsae.experiments.llm_benchmark import plotting as plotting_module


def _summary_payload() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": {"llm_id": "dummy", "hook_name": "mlp_out", "layer": 0},
        "aggregates": [
            {
                "variant_slug": "variant_a",
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "num_seeds": 1,
                "mse_mean": 0.3,
                "mse_se": 0.0,
                "mean_l0_mean": 10.0,
                "mean_l0_se": 0.0,
                "dead_fraction_mean": 0.25,
                "dead_fraction_se": 0.0,
                "loss_recovered_mean": 0.4,
                "loss_recovered_se": 0.0,
            },
            {
                "variant_slug": "variant_a",
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
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "relative_dir": "models/variant_a/seed_0/lambda_0p01",
                "mean_l0": 10.0,
                "mse": 0.3,
                "loss_recovered": 0.4,
                "dead_fraction": 0.25,
            },
            {
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "relative_dir": "models/variant_a/seed_0/lambda_0p001",
                "mean_l0": 80.0,
                "mse": 0.1,
                "loss_recovered": 0.9,
                "dead_fraction": 0.05,
            },
        ],
    }


class TestLLMBenchmarkPlottingMatchedPointsHelpers(unittest.TestCase):
    def test_generate_plots_for_run_raises_when_summary_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            # when / then
            with self.assertRaises(ValueError):
                _ = plotting_module.generate_plots_for_run(output_dir=tmpdir)

    def test_load_or_compute_matched_points_recomputes_when_file_is_not_dict(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "matched_points.json"), "w") as file:
                json.dump([], file)
            expected = {
                "anchors": {"l0": [16], "lr": [0.55]},
                "per_seed": [],
                "aggregate": [],
            }

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.plotting.compute_and_persist_matched_points",
                return_value=expected,
            ) as compute_mock:
                loaded = plotting_module._load_or_compute_matched_points(
                    output_dir=tmpdir,
                    summary=_summary_payload(),
                )

            # then
            self.assertEqual(loaded, expected)
            compute_mock.assert_called_once()

    def test_load_or_compute_matched_points_reuses_valid_payload(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {
                "anchors": {"l0": [16], "lr": [0.55]},
                "per_seed": [],
                "aggregate": [],
            }
            with open(os.path.join(tmpdir, "matched_points.json"), "w") as file:
                json.dump(payload, file, indent=2)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.plotting.compute_and_persist_matched_points"
            ) as compute_mock:
                loaded = plotting_module._load_or_compute_matched_points(
                    output_dir=tmpdir,
                    summary=_summary_payload(),
                )

            # then
            self.assertEqual(loaded, payload)
            compute_mock.assert_not_called()

    def test_load_or_compute_matched_points_recomputes_invalid_payload(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "matched_points.json"), "w") as file:
                json.dump(
                    {"anchors": [], "per_seed": "bad", "aggregate": []}, file, indent=2
                )
            expected = {
                "anchors": {"l0": [16], "lr": [0.55]},
                "per_seed": [],
                "aggregate": [],
            }

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.plotting.compute_and_persist_matched_points",
                return_value=expected,
            ) as compute_mock:
                loaded = plotting_module._load_or_compute_matched_points(
                    output_dir=tmpdir,
                    summary=_summary_payload(),
                )

            # then
            self.assertEqual(loaded, expected)
            compute_mock.assert_called_once()

    def test_matched_table_rows_filters_and_normalizes_entries(self) -> None:
        # given
        invalid_rows = plotting_module._matched_table_rows(
            matched_points_payload={"per_seed": "bad"},
            anchor_type="l0",
        )
        payload = {
            "per_seed": [
                "bad",
                {"anchor_type": "lr"},
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "anchor_type": "l0",
                    "anchor_value": 16.0,
                    "method": "interpolated",
                    "distance_to_target": 0.0,
                    "metrics": "bad",
                    "supporting_points": [{"relative_dir": "a"}, "bad"],
                },
                {
                    "variant_slug": "variant_a",
                    "seed": 1,
                    "anchor_type": "l0",
                    "anchor_value": 64.0,
                    "method": "out_of_range",
                    "distance_to_target": None,
                    "metrics": {},
                    "supporting_points": "bad",
                },
            ]
        }

        # when
        rows = plotting_module._matched_table_rows(
            matched_points_payload=payload,
            anchor_type="l0",
        )

        # then
        self.assertEqual(invalid_rows, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["supporting_relative_dir_1"], "a")
        self.assertIsNone(rows[0]["supporting_relative_dir_2"])
        self.assertIsNone(rows[0]["mse"])
        self.assertIsNone(rows[1]["supporting_relative_dir_1"])

    def test_write_matched_table_writes_expected_columns(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tables", "matched_l0_points.csv")
            rows = [
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "anchor_type": "l0",
                    "anchor_value": 16.0,
                    "method": "interpolated",
                    "distance_to_target": 0.0,
                    "mean_l0": 16.0,
                    "mse": 0.2,
                    "loss_recovered": 0.6,
                    "dead_fraction": 0.1,
                    "supporting_relative_dir_1": "a",
                    "supporting_relative_dir_2": "b",
                }
            ]

            # when
            plotting_module._write_matched_table(path=path, rows=rows)

            # then
            self.assertTrue(os.path.isfile(path))
            with open(path, "r", newline="") as file:
                reader = csv.DictReader(file)
                loaded_rows = list(reader)
            self.assertEqual(len(loaded_rows), 1)
            self.assertEqual(
                set(loaded_rows[0].keys()),
                {
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
                },
            )

    def test_write_matched_tables_writes_l0_and_lr_files(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {
                "per_seed": [
                    {
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "anchor_type": "l0",
                        "anchor_value": 16.0,
                        "method": "interpolated",
                        "distance_to_target": 0.0,
                        "metrics": {
                            "mean_l0": 16.0,
                            "mse": 0.2,
                            "loss_recovered": 0.6,
                            "dead_fraction": 0.1,
                        },
                        "supporting_points": [],
                    },
                    {
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "anchor_type": "lr",
                        "anchor_value": 0.55,
                        "method": "interpolated",
                        "distance_to_target": 0.0,
                        "metrics": {
                            "mean_l0": 12.0,
                            "mse": 0.25,
                            "loss_recovered": 0.55,
                            "dead_fraction": 0.12,
                        },
                        "supporting_points": [],
                    },
                ]
            }

            # when
            paths = plotting_module.write_matched_tables(
                output_dir=tmpdir,
                matched_points_payload=payload,
            )

            # then
            self.assertTrue(os.path.isfile(paths["matched_l0_points"]))
            self.assertTrue(os.path.isfile(paths["matched_lr_points"]))

    def test_plot_metric_vs_l0_handles_overlays_and_no_data(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with_data_path = os.path.join(tmpdir, "with_data.png")
            no_data_path = os.path.join(tmpdir, "no_data.png")
            aggregates = [
                {
                    "variant_slug": "variant_a",
                    "mean_l0_mean": 16.0,
                    "mse_mean": 0.2,
                }
            ]
            matched_aggregate = [
                "bad",
                {"anchor_type": "lr"},
                {
                    "variant_slug": 7,
                    "anchor_type": "l0",
                    "anchor_value": 16.0,
                    "mse_mean": 0.2,
                },
                {
                    "variant_slug": "variant_a",
                    "anchor_type": "l0",
                    "anchor_value": 16.0,
                    "mse_mean": 0.2,
                },
                {
                    "variant_slug": "variant_a",
                    "anchor_type": "l0",
                    "anchor_value": "bad",
                    "mse_mean": 0.2,
                },
            ]

            # when
            plotting_module._plot_metric_vs_l0(
                output_path=with_data_path,
                aggregates=aggregates,
                y_key="mse_mean",
                y_label="MSE",
                title="With Data",
                matched_points_aggregate=matched_aggregate,
            )
            plotting_module._plot_metric_vs_l0(
                output_path=no_data_path,
                aggregates=[],
                y_key="mse_mean",
                y_label="MSE",
                title="No Data",
                matched_points_aggregate=None,
            )
            plotting_module._plot_metric_vs_l0(
                output_path=os.path.join(tmpdir, "with_data_no_overlay_list.png"),
                aggregates=aggregates
                + [{"variant_slug": None, "mean_l0_mean": 12.0, "mse_mean": 0.5}],
                y_key="mse_mean",
                y_label="MSE",
                title="With Data No Overlay List",
                matched_points_aggregate="bad",
            )

            # then
            self.assertTrue(os.path.isfile(with_data_path))
            self.assertTrue(os.path.isfile(no_data_path))
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "with_data_no_overlay_list.png"))
            )

    def test_generate_plots_for_run_handles_non_list_aggregate_payload(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(_summary_payload(), file, indent=2)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.plotting.compute_and_persist_matched_points",
                return_value={"aggregate": "bad", "per_seed": [], "anchors": {}},
            ):
                result = plotting_module.generate_plots_for_run(output_dir=tmpdir)

            # then
            self.assertEqual(
                int(result["plots_written"]),
                len(plotting_module.canonical_plot_paths(output_dir=tmpdir)),
            )

    def test_generate_plots_for_run_handles_non_list_aggregates(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_payload = _summary_payload()
            summary_payload["aggregates"] = "bad"
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(summary_payload, file, indent=2)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.plotting.compute_and_persist_matched_points",
                return_value={"aggregate": [], "per_seed": [], "anchors": {}},
            ):
                result = plotting_module.generate_plots_for_run(output_dir=tmpdir)

            # then
            self.assertEqual(
                int(result["plots_written"]),
                len(plotting_module.canonical_plot_paths(output_dir=tmpdir)),
            )


if __name__ == "__main__":
    unittest.main()
