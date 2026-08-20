import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sgnsae.experiments.llm_benchmark.schema_versions import (
    MATCHED_POINTS_SCHEMA_VERSION,
)


DEFAULT_MATCHED_L0_ANCHORS: Tuple[float, float] = (16.0, 64.0)
DEFAULT_MATCHED_LR_ANCHORS: Tuple[float, float, float] = (0.55, 0.65, 0.75)
_MATCHED_METRICS: Tuple[str, str, str, str] = (
    "mean_l0",
    "mse",
    "loss_recovered",
    "dead_fraction",
)
_INTERPOLATION_EPSILON = 1e-12


def matched_points_path(output_dir: str) -> str:
    return os.path.join(output_dir, "matched_points.json")


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    cast_value = float(value)
    if not math.isfinite(cast_value):
        return None
    return cast_value


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _safe_standard_error(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    if len(values) == 1:
        return 0.0
    mean_value = float(sum(values) / float(len(values)))
    variance = sum((value - mean_value) ** 2 for value in values) / float(
        len(values) - 1
    )
    return float(math.sqrt(variance) / math.sqrt(float(len(values))))


def _normalize_anchors(
    l0_anchors: Optional[Sequence[float]],
    lr_anchors: Optional[Sequence[float]],
) -> Tuple[List[float], List[float]]:
    resolved_l0 = (
        list(DEFAULT_MATCHED_L0_ANCHORS) if l0_anchors is None else list(l0_anchors)
    )
    resolved_lr = (
        list(DEFAULT_MATCHED_LR_ANCHORS) if lr_anchors is None else list(lr_anchors)
    )
    output_l0: List[float] = []
    output_lr: List[float] = []
    for value in resolved_l0:
        parsed = _safe_float(value)
        if parsed is not None:
            output_l0.append(float(parsed))
    for value in resolved_lr:
        parsed = _safe_float(value)
        if parsed is not None:
            output_lr.append(float(parsed))
    return output_l0, output_lr


def _group_subruns_by_variant_and_seed(
    summary: Dict[str, Any],
) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    subruns = summary.get("subruns", [])
    if not isinstance(subruns, list):
        return {}
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for entry in subruns:
        if not isinstance(entry, dict):
            continue
        variant_slug = entry.get("variant_slug")
        seed = _safe_int(entry.get("seed"))
        if not isinstance(variant_slug, str) or seed is None:
            continue
        grouped.setdefault((variant_slug, int(seed)), []).append(dict(entry))
    return grouped


def _build_point(entry: Dict[str, Any], anchor_type: str) -> Optional[Dict[str, Any]]:
    x_key = "mean_l0" if anchor_type == "l0" else "loss_recovered"
    x_value = _safe_float(entry.get(x_key))
    if x_value is None:
        return None
    point = {
        "x": float(x_value),
        "relative_dir": entry.get("relative_dir"),
        "sweep_param": entry.get("sweep_param"),
        "sweep_value": entry.get("sweep_value"),
    }
    for metric in _MATCHED_METRICS:
        point[metric] = _safe_float(entry.get(metric))
    return point


def _group_is_discrete(group_entries: Sequence[Dict[str, Any]]) -> bool:
    for entry in group_entries:
        if str(entry.get("sweep_param", "")).strip().lower() == "k":
            return True
    return False


def _supporting_point_payload(point: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "relative_dir": point.get("relative_dir"),
        "sweep_param": point.get("sweep_param"),
        "sweep_value": point.get("sweep_value"),
        "x_value": point.get("x"),
    }
    for metric in _MATCHED_METRICS:
        payload[metric] = point.get(metric)
    return payload


def _nearest_tie_break_key(
    anchor: float, point: Dict[str, Any]
) -> Tuple[float, float, float, str]:
    distance = abs(float(point["x"]) - float(anchor))
    mse_value = point.get("mse")
    if mse_value is None:
        mse = float("inf")
    else:
        mse = float(mse_value)
    sweep_value = point.get("sweep_value")
    if isinstance(sweep_value, bool) or not isinstance(sweep_value, (int, float)):
        sweep_sort = float("inf")
    else:
        sweep_sort = float(sweep_value)
    relative_dir = str(point.get("relative_dir", ""))
    return distance, mse, sweep_sort, relative_dir


def _match_discrete_anchor(
    points: Sequence[Dict[str, Any]],
    anchor: float,
) -> Dict[str, Any]:
    nearest = min(points, key=lambda point: _nearest_tie_break_key(anchor, point))
    metrics: Dict[str, Optional[float]] = {}
    for metric in _MATCHED_METRICS:
        metric_value = nearest.get(metric)
        metrics[metric] = None if metric_value is None else float(metric_value)
    return {
        "method": "discrete_nearest",
        "distance_to_target": abs(float(nearest["x"]) - float(anchor)),
        "metrics": metrics,
        "supporting_points": [_supporting_point_payload(nearest)],
    }


def _interpolate_metric(
    low_value: Optional[float], high_value: Optional[float], interpolation_ratio: float
) -> Optional[float]:
    if low_value is None or high_value is None:
        return None
    return float(
        float(low_value) + interpolation_ratio * (float(high_value) - float(low_value))
    )


def _match_continuous_anchor(
    points: Sequence[Dict[str, Any]],
    anchor: float,
) -> Dict[str, Any]:
    sorted_points = sorted(
        list(points),
        key=lambda point: (float(point["x"]), str(point.get("relative_dir", ""))),
    )
    if not sorted_points:
        return {
            "method": "out_of_range",
            "distance_to_target": None,
            "metrics": {metric: None for metric in _MATCHED_METRICS},
            "supporting_points": [],
        }

    minimum_x = float(sorted_points[0]["x"])
    maximum_x = float(sorted_points[-1]["x"])
    if (
        anchor < minimum_x - _INTERPOLATION_EPSILON
        or anchor > maximum_x + _INTERPOLATION_EPSILON
    ):
        return {
            "method": "out_of_range",
            "distance_to_target": None,
            "metrics": {metric: None for metric in _MATCHED_METRICS},
            "supporting_points": [],
        }

    exact_matches = [
        point
        for point in sorted_points
        if abs(float(point["x"]) - float(anchor)) <= _INTERPOLATION_EPSILON
    ]
    if exact_matches:
        selected = min(
            exact_matches, key=lambda point: _nearest_tie_break_key(anchor, point)
        )
        metrics: Dict[str, Optional[float]] = {}
        for metric in _MATCHED_METRICS:
            metric_value = selected.get(metric)
            metrics[metric] = None if metric_value is None else float(metric_value)
        return {
            "method": "interpolated",
            "distance_to_target": 0.0,
            "metrics": metrics,
            "supporting_points": [
                _supporting_point_payload(selected),
                _supporting_point_payload(selected),
            ],
        }

    upper_index = next(
        index
        for index, point in enumerate(sorted_points)
        if float(point["x"]) > float(anchor)
    )
    lower_index = max(0, upper_index - 1)
    lower_point = sorted_points[lower_index]
    upper_point = sorted_points[upper_index]

    lower_x = float(lower_point["x"])
    upper_x = float(upper_point["x"])
    denominator = upper_x - lower_x
    interpolation_ratio = (float(anchor) - lower_x) / denominator
    metrics = {
        metric: _interpolate_metric(
            low_value=lower_point.get(metric),
            high_value=upper_point.get(metric),
            interpolation_ratio=interpolation_ratio,
        )
        for metric in _MATCHED_METRICS
    }
    return {
        "method": "interpolated",
        "distance_to_target": 0.0,
        "metrics": metrics,
        "supporting_points": [
            _supporting_point_payload(lower_point),
            _supporting_point_payload(upper_point),
        ],
    }


def _compute_entries_for_anchor_type(
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]],
    anchor_type: str,
    anchors: Sequence[float],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for variant_slug, seed in sorted(
        grouped.keys(), key=lambda item: (item[0], item[1])
    ):
        group_entries = grouped[(variant_slug, seed)]
        points: List[Dict[str, Any]] = []
        for entry in group_entries:
            point = _build_point(entry=entry, anchor_type=anchor_type)
            if point is not None:
                points.append(point)
        is_discrete = _group_is_discrete(group_entries=group_entries)
        for anchor in anchors:
            if not points:
                match = {
                    "method": "out_of_range",
                    "distance_to_target": None,
                    "metrics": {metric: None for metric in _MATCHED_METRICS},
                    "supporting_points": [],
                }
            elif is_discrete:
                match = _match_discrete_anchor(points=points, anchor=float(anchor))
            else:
                match = _match_continuous_anchor(points=points, anchor=float(anchor))
            output.append(
                {
                    "variant_slug": str(variant_slug),
                    "seed": int(seed),
                    "anchor_type": str(anchor_type),
                    "anchor_value": float(anchor),
                    "method": str(match["method"]),
                    "distance_to_target": match["distance_to_target"],
                    "metrics": dict(match["metrics"]),
                    "supporting_points": list(match["supporting_points"]),
                }
            )
    return output


def _aggregate_matched_entries(
    per_seed_entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for entry in per_seed_entries:
        if not isinstance(entry, dict):
            continue
        variant_slug = entry.get("variant_slug")
        anchor_type = entry.get("anchor_type")
        anchor_value = _safe_float(entry.get("anchor_value"))
        if (
            not isinstance(variant_slug, str)
            or not isinstance(anchor_type, str)
            or anchor_value is None
        ):
            continue
        grouped.setdefault((variant_slug, anchor_type, float(anchor_value)), []).append(
            entry
        )

    aggregates: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1], item[2])):
        variant_slug, anchor_type, anchor_value = key
        entries = grouped[key]
        matched_entries = [
            entry for entry in entries if str(entry.get("method")) != "out_of_range"
        ]
        method_counts = {
            "interpolated": 0,
            "discrete_nearest": 0,
            "out_of_range": 0,
        }
        for entry in entries:
            method = str(entry.get("method", ""))
            if method in method_counts:
                method_counts[method] += 1

        aggregate_entry: Dict[str, Any] = {
            "variant_slug": str(variant_slug),
            "anchor_type": str(anchor_type),
            "anchor_value": float(anchor_value),
            "num_seeds_total": int(len(entries)),
            "num_seeds_matched": int(len(matched_entries)),
            "method_counts": method_counts,
        }
        for metric in _MATCHED_METRICS:
            values: List[float] = []
            for entry in matched_entries:
                metrics = entry.get("metrics")
                if not isinstance(metrics, dict):
                    continue
                metric_value = _safe_float(metrics.get(metric))
                if metric_value is not None:
                    values.append(float(metric_value))
            aggregate_entry[f"{metric}_mean"] = _safe_mean(values)
            aggregate_entry[f"{metric}_se"] = _safe_standard_error(values)
            aggregate_entry[f"{metric}_num_seeds"] = int(len(values))
        aggregates.append(aggregate_entry)
    return aggregates


def compute_matched_points(
    summary: Dict[str, Any],
    l0_anchors: Optional[Sequence[float]] = None,
    lr_anchors: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    resolved_l0_anchors, resolved_lr_anchors = _normalize_anchors(
        l0_anchors=l0_anchors,
        lr_anchors=lr_anchors,
    )
    grouped = _group_subruns_by_variant_and_seed(summary=summary)
    per_seed_entries = _compute_entries_for_anchor_type(
        grouped=grouped,
        anchor_type="l0",
        anchors=resolved_l0_anchors,
    )
    per_seed_entries.extend(
        _compute_entries_for_anchor_type(
            grouped=grouped,
            anchor_type="lr",
            anchors=resolved_lr_anchors,
        )
    )
    aggregate_entries = _aggregate_matched_entries(per_seed_entries=per_seed_entries)
    return {
        "schema_version": MATCHED_POINTS_SCHEMA_VERSION,
        "anchors": {
            "l0": [float(value) for value in resolved_l0_anchors],
            "lr": [float(value) for value in resolved_lr_anchors],
        },
        "per_seed": per_seed_entries,
        "aggregate": aggregate_entries,
    }


def compute_and_persist_matched_points(
    summary: Dict[str, Any],
    output_dir: str,
    l0_anchors: Optional[Sequence[float]] = None,
    lr_anchors: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    payload = compute_matched_points(
        summary=summary,
        l0_anchors=l0_anchors,
        lr_anchors=lr_anchors,
    )
    output_path = matched_points_path(output_dir=output_dir)
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)
    return payload
