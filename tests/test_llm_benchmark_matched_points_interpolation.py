import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import sgnsae.experiments.llm_benchmark.matched_points as matched_points_module
from sgnsae.experiments.llm_benchmark.matched_points import (
    compute_and_persist_matched_points,
    compute_matched_points,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    MATCHED_POINTS_SCHEMA_VERSION,
)


def _continuous_summary() -> Dict[str, Any]:
    return {
        "subruns": [
            {
                "variant_slug": "continuous_variant",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "relative_dir": "models/continuous_variant/seed_0/lambda_0p01",
                "mean_l0": 10.0,
                "mse": 0.20,
                "loss_recovered": 0.50,
                "dead_fraction": 0.40,
            },
            {
                "variant_slug": "continuous_variant",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "relative_dir": "models/continuous_variant/seed_0/lambda_0p001",
                "mean_l0": 20.0,
                "mse": 0.40,
                "loss_recovered": 0.70,
                "dead_fraction": 0.20,
            },
            {
                "variant_slug": "continuous_variant",
                "seed": 1,
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "relative_dir": "models/continuous_variant/seed_1/lambda_0p01",
                "mean_l0": 10.0,
                "mse": 0.10,
                "loss_recovered": 0.45,
                "dead_fraction": 0.35,
            },
            {
                "variant_slug": "continuous_variant",
                "seed": 1,
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "relative_dir": "models/continuous_variant/seed_1/lambda_0p001",
                "mean_l0": 20.0,
                "mse": 0.30,
                "loss_recovered": 0.65,
                "dead_fraction": 0.15,
            },
        ]
    }


def _find_entry(
    entries: List[Dict[str, Any]],
    variant_slug: str,
    seed: int,
    anchor_type: str,
    anchor_value: float,
) -> Dict[str, Any]:
    for entry in entries:
        if (
            str(entry.get("variant_slug")) == str(variant_slug)
            and int(entry.get("seed")) == int(seed)
            and str(entry.get("anchor_type")) == str(anchor_type)
            and abs(float(entry.get("anchor_value")) - float(anchor_value)) < 1e-12
        ):
            return entry
    raise ValueError("Unable to locate expected per-seed matched-point entry.")


def _find_aggregate(
    entries: List[Dict[str, Any]],
    variant_slug: str,
    anchor_type: str,
    anchor_value: float,
) -> Dict[str, Any]:
    for entry in entries:
        if (
            str(entry.get("variant_slug")) == str(variant_slug)
            and str(entry.get("anchor_type")) == str(anchor_type)
            and abs(float(entry.get("anchor_value")) - float(anchor_value)) < 1e-12
        ):
            return entry
    raise ValueError("Unable to locate expected aggregate matched-point entry.")


class TestLLMBenchmarkMatchedPointsInterpolation(unittest.TestCase):
    def test_find_helpers_raise_when_entry_is_missing(self) -> None:
        # given
        per_seed_entries = [
            {
                "variant_slug": "variant_a",
                "seed": 1,
                "anchor_type": "l0",
                "anchor_value": 16.0,
            }
        ]
        aggregate_entries = [
            {
                "variant_slug": "variant_b",
                "anchor_type": "lr",
                "anchor_value": 0.55,
            }
        ]

        # when / then
        with self.assertRaises(ValueError):
            _ = _find_entry(
                entries=per_seed_entries,
                variant_slug="variant_a",
                seed=2,
                anchor_type="l0",
                anchor_value=16.0,
            )
        with self.assertRaises(ValueError):
            _ = _find_aggregate(
                entries=aggregate_entries,
                variant_slug="variant_a",
                anchor_type="lr",
                anchor_value=0.55,
            )

    def test_private_helpers_cover_guard_paths(self) -> None:
        # given
        grouped_entries = {
            ("guard_variant", 0): [
                {
                    "variant_slug": "guard_variant",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "mean_l0": None,
                }
            ]
        }

        # when
        safe_float_from_bool = matched_points_module._safe_float(True)
        safe_float_from_text = matched_points_module._safe_float("bad")
        safe_float_from_nan = matched_points_module._safe_float(float("nan"))
        safe_int_from_bool = matched_points_module._safe_int(True)
        safe_mean_empty = matched_points_module._safe_mean([])
        safe_se_empty = matched_points_module._safe_standard_error([])
        safe_se_single = matched_points_module._safe_standard_error([1.0])
        normalized_l0, normalized_lr = matched_points_module._normalize_anchors(
            l0_anchors=[16.0, "bad", True],
            lr_anchors=[0.6, float("inf"), False],
        )
        grouped_non_list = matched_points_module._group_subruns_by_variant_and_seed(
            summary={"subruns": "bad"}
        )
        grouped_invalid_entries = (
            matched_points_module._group_subruns_by_variant_and_seed(
                summary={
                    "subruns": [
                        "bad",
                        {"variant_slug": "missing_seed", "seed": "0"},
                        {"variant_slug": "bad_seed", "seed": True},
                        {"variant_slug": "ok_variant", "seed": 1},
                    ]
                }
            )
        )
        missing_point = matched_points_module._build_point(
            entry={"mean_l0": None},
            anchor_type="l0",
        )
        tie_break_key = matched_points_module._nearest_tie_break_key(
            anchor=16.0,
            point={
                "x": 8.0,
                "mse": None,
                "sweep_value": False,
                "relative_dir": "models/x",
            },
        )
        interpolated_none = matched_points_module._interpolate_metric(
            low_value=None,
            high_value=0.1,
            interpolation_ratio=0.5,
        )
        empty_continuous_match = matched_points_module._match_continuous_anchor(
            points=[],
            anchor=16.0,
        )
        out_of_range_entries = matched_points_module._compute_entries_for_anchor_type(
            grouped=grouped_entries,
            anchor_type="l0",
            anchors=[16.0],
        )
        aggregate_from_mixed = matched_points_module._aggregate_matched_entries(
            per_seed_entries=[
                "bad",
                {"variant_slug": "bad", "anchor_type": "l0", "anchor_value": "16"},
                {
                    "variant_slug": "guard_variant",
                    "anchor_type": "l0",
                    "anchor_value": 16.0,
                    "method": "interpolated",
                    "metrics": "not_a_dict",
                },
                {
                    "variant_slug": "guard_variant",
                    "anchor_type": "l0",
                    "anchor_value": 16.0,
                    "method": "custom_method",
                    "metrics": {"mse": None},
                },
            ]
        )

        # then
        self.assertIsNone(safe_float_from_bool)
        self.assertIsNone(safe_float_from_text)
        self.assertIsNone(safe_float_from_nan)
        self.assertIsNone(safe_int_from_bool)
        self.assertIsNone(safe_mean_empty)
        self.assertIsNone(safe_se_empty)
        self.assertEqual(float(safe_se_single), 0.0)
        self.assertEqual(normalized_l0, [16.0])
        self.assertEqual(normalized_lr, [0.6])
        self.assertEqual(grouped_non_list, {})
        self.assertEqual(set(grouped_invalid_entries.keys()), {("ok_variant", 1)})
        self.assertIsNone(missing_point)
        self.assertAlmostEqual(float(tie_break_key[0]), 8.0, places=12)
        self.assertTrue(tie_break_key[1] > 1e30)
        self.assertTrue(tie_break_key[2] > 1e30)
        self.assertIsNone(interpolated_none)
        self.assertEqual(empty_continuous_match["method"], "out_of_range")
        self.assertEqual(out_of_range_entries[0]["method"], "out_of_range")
        self.assertEqual(len(aggregate_from_mixed), 1)
        self.assertEqual(
            aggregate_from_mixed[0]["method_counts"],
            {
                "interpolated": 1,
                "discrete_nearest": 0,
                "out_of_range": 0,
            },
        )

    def test_match_continuous_anchor_handles_exact_matches(self) -> None:
        # given
        points = [
            {
                "x": 16.0,
                "relative_dir": "models/continuous/a",
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "mean_l0": 16.0,
                "mse": 0.3,
                "loss_recovered": 0.5,
                "dead_fraction": 0.2,
            },
            {
                "x": 16.0,
                "relative_dir": "models/continuous/b",
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "mean_l0": 16.0,
                "mse": 0.2,
                "loss_recovered": 0.6,
                "dead_fraction": 0.1,
            },
        ]

        # when
        match = matched_points_module._match_continuous_anchor(
            points=points, anchor=16.0
        )

        # then
        self.assertEqual(match["method"], "interpolated")
        self.assertAlmostEqual(float(match["distance_to_target"]), 0.0, places=12)
        self.assertAlmostEqual(float(match["metrics"]["mse"]), 0.2, places=12)
        self.assertEqual(len(match["supporting_points"]), 2)

    def test_match_continuous_anchor_skips_non_bracketing_first_pair(self) -> None:
        # given
        points = [
            {
                "x": 10.0,
                "relative_dir": "models/continuous/a",
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "mean_l0": 10.0,
                "mse": 0.4,
                "loss_recovered": 0.4,
                "dead_fraction": 0.4,
            },
            {
                "x": 10.0,
                "relative_dir": "models/continuous/b",
                "sweep_param": "lambda",
                "sweep_value": 0.005,
                "mean_l0": 10.0,
                "mse": 0.3,
                "loss_recovered": 0.5,
                "dead_fraction": 0.3,
            },
            {
                "x": 20.0,
                "relative_dir": "models/continuous/c",
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "mean_l0": 20.0,
                "mse": 0.1,
                "loss_recovered": 0.7,
                "dead_fraction": 0.1,
            },
        ]

        # when
        match = matched_points_module._match_continuous_anchor(
            points=points, anchor=15.0
        )

        # then
        self.assertEqual(match["method"], "interpolated")
        self.assertAlmostEqual(float(match["metrics"]["mse"]), 0.2, places=12)

    def test_match_continuous_anchor_raises_when_bracket_cannot_be_found(self) -> None:
        # given
        points = [
            {
                "x": float("nan"),
                "relative_dir": "models/continuous/nan",
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "mean_l0": 10.0,
                "mse": 0.4,
                "loss_recovered": 0.4,
                "dead_fraction": 0.4,
            }
        ]

        # when / then
        with self.assertRaises(StopIteration):
            _ = matched_points_module._match_continuous_anchor(
                points=points, anchor=15.0
            )

    def test_compute_matched_points_interpolates_continuous_l0_metrics(self) -> None:
        # given
        summary = _continuous_summary()

        # when
        payload = compute_matched_points(
            summary=summary, l0_anchors=[15], lr_anchors=[]
        )
        entry = _find_entry(
            entries=payload["per_seed"],
            variant_slug="continuous_variant",
            seed=0,
            anchor_type="l0",
            anchor_value=15.0,
        )

        # then
        self.assertEqual(payload["anchors"]["l0"], [15.0])
        self.assertEqual(entry["method"], "interpolated")
        self.assertAlmostEqual(float(entry["distance_to_target"]), 0.0, places=12)
        self.assertAlmostEqual(float(entry["metrics"]["mean_l0"]), 15.0, places=12)
        self.assertAlmostEqual(float(entry["metrics"]["mse"]), 0.30, places=12)
        self.assertAlmostEqual(
            float(entry["metrics"]["loss_recovered"]), 0.60, places=12
        )
        self.assertAlmostEqual(
            float(entry["metrics"]["dead_fraction"]), 0.30, places=12
        )
        self.assertEqual(len(entry["supporting_points"]), 2)

    def test_compute_matched_points_marks_continuous_out_of_range(self) -> None:
        # given
        summary = _continuous_summary()

        # when
        payload = compute_matched_points(
            summary=summary, l0_anchors=[64], lr_anchors=[]
        )
        entry = _find_entry(
            entries=payload["per_seed"],
            variant_slug="continuous_variant",
            seed=0,
            anchor_type="l0",
            anchor_value=64.0,
        )

        # then
        self.assertEqual(entry["method"], "out_of_range")
        self.assertIsNone(entry["distance_to_target"])
        self.assertEqual(entry["supporting_points"], [])
        self.assertEqual(
            entry["metrics"],
            {
                "mean_l0": None,
                "mse": None,
                "loss_recovered": None,
                "dead_fraction": None,
            },
        )
        aggregate = _find_aggregate(
            entries=payload["aggregate"],
            variant_slug="continuous_variant",
            anchor_type="l0",
            anchor_value=64.0,
        )
        self.assertEqual(int(aggregate["num_seeds_matched"]), 0)
        self.assertEqual(int(aggregate["method_counts"]["out_of_range"]), 2)
        self.assertIsNone(aggregate["mse_mean"])

    def test_compute_matched_points_uses_discrete_nearest_for_topk_sweeps(self) -> None:
        # given
        summary = {
            "subruns": [
                {
                    "variant_slug": "abstopk_variant",
                    "seed": 0,
                    "sweep_param": "k",
                    "sweep_value": 8,
                    "relative_dir": "models/abstopk_variant/seed_0/k_8",
                    "mean_l0": 8.0,
                    "mse": 0.40,
                    "loss_recovered": 0.45,
                    "dead_fraction": 0.35,
                },
                {
                    "variant_slug": "abstopk_variant",
                    "seed": 0,
                    "sweep_param": "k",
                    "sweep_value": 16,
                    "relative_dir": "models/abstopk_variant/seed_0/k_16",
                    "mean_l0": 16.0,
                    "mse": 0.20,
                    "loss_recovered": 0.65,
                    "dead_fraction": 0.15,
                },
            ]
        }

        # when
        payload = compute_matched_points(
            summary=summary, l0_anchors=[12], lr_anchors=[]
        )
        entry = _find_entry(
            entries=payload["per_seed"],
            variant_slug="abstopk_variant",
            seed=0,
            anchor_type="l0",
            anchor_value=12.0,
        )

        # then
        self.assertEqual(entry["method"], "discrete_nearest")
        self.assertAlmostEqual(float(entry["distance_to_target"]), 4.0, places=12)
        self.assertAlmostEqual(float(entry["metrics"]["mean_l0"]), 16.0, places=12)
        self.assertAlmostEqual(float(entry["metrics"]["mse"]), 0.20, places=12)
        self.assertEqual(len(entry["supporting_points"]), 1)
        self.assertEqual(
            entry["supporting_points"][0]["relative_dir"],
            "models/abstopk_variant/seed_0/k_16",
        )

    def test_compute_matched_points_interpolates_matched_lr(self) -> None:
        # given
        summary = _continuous_summary()

        # when
        payload = compute_matched_points(
            summary=summary, l0_anchors=[], lr_anchors=[0.60]
        )
        entry = _find_entry(
            entries=payload["per_seed"],
            variant_slug="continuous_variant",
            seed=0,
            anchor_type="lr",
            anchor_value=0.60,
        )

        # then
        self.assertEqual(entry["method"], "interpolated")
        self.assertAlmostEqual(
            float(entry["metrics"]["loss_recovered"]), 0.60, places=12
        )
        self.assertAlmostEqual(float(entry["metrics"]["mean_l0"]), 15.0, places=12)
        self.assertAlmostEqual(float(entry["metrics"]["mse"]), 0.30, places=12)
        self.assertAlmostEqual(
            float(entry["metrics"]["dead_fraction"]), 0.30, places=12
        )
        aggregate = _find_aggregate(
            entries=payload["aggregate"],
            variant_slug="continuous_variant",
            anchor_type="lr",
            anchor_value=0.60,
        )
        self.assertEqual(int(aggregate["num_seeds_total"]), 2)
        self.assertEqual(int(aggregate["num_seeds_matched"]), 2)
        self.assertAlmostEqual(float(aggregate["mean_l0_mean"]), 16.25, places=12)
        self.assertAlmostEqual(float(aggregate["mse_mean"]), 0.275, places=12)
        self.assertGreaterEqual(float(aggregate["mse_se"]), 0.0)

    def test_compute_matched_points_handles_invalid_summary_payload(self) -> None:
        # given
        summary = {"subruns": "bad"}

        # when
        payload = compute_matched_points(
            summary=summary, l0_anchors=[16], lr_anchors=[0.6]
        )

        # then
        self.assertEqual(payload["per_seed"], [])
        self.assertEqual(payload["aggregate"], [])
        self.assertEqual(payload["anchors"]["l0"], [16.0])
        self.assertEqual(payload["anchors"]["lr"], [0.6])

    def test_compute_and_persist_matched_points_writes_json_artifact(self) -> None:
        # given
        summary = _continuous_summary()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "matched_points.json")

            # when
            payload = compute_and_persist_matched_points(
                summary=summary,
                output_dir=tmpdir,
                l0_anchors=[16],
                lr_anchors=[0.55],
            )

            # then
            self.assertTrue(os.path.isfile(output_path))
            with open(output_path, "r") as file:
                persisted_payload = json.load(file)
            self.assertEqual(payload, persisted_payload)
            self.assertEqual(
                int(persisted_payload["schema_version"]),
                MATCHED_POINTS_SCHEMA_VERSION,
            )


if __name__ == "__main__":
    unittest.main()
