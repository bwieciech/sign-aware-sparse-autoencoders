import csv
import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from sgnsae.experiments.llm_benchmark.matched_points import (
    compute_and_persist_matched_points,
    matched_points_path,
)


DEFAULT_MATCHED_L0_ANCHORS: Tuple[int, int] = (16, 64)
_DUPLICATION_THRESHOLDS: Tuple[float, float] = (-0.90, -0.95)
_DUPLICATION_SAMPLE_SIZE = 4096
_DUPLICATION_CHUNK_SIZE = 512
_POLARITY_STABILITY_METRIC_KEYS: Tuple[str, ...] = (
    "polarity_positive_fraction_nonzero",
    "polarity_negative_fraction_nonzero",
    "polarity_bidirectional_usage_fraction",
    "polarity_dominance_flip_rate",
    "polarity_dominance_flip_indicator",
)


def canonical_plot_paths(output_dir: str) -> Dict[str, str]:
    plots_dir = os.path.join(output_dir, "plots")
    output_paths: Dict[str, str] = {
        "recon_vs_l0": os.path.join(plots_dir, "recon_vs_l0.png"),
        "loss_recovered_vs_l0": os.path.join(plots_dir, "loss_recovered_vs_l0.png"),
        "dead_fraction_vs_l0": os.path.join(plots_dir, "dead_fraction_vs_l0.png"),
        "polarity_imbalance_hist": os.path.join(
            plots_dir, "polarity_imbalance_hist.png"
        ),
        "duplication_nearest_negative_cosine_hist": os.path.join(
            plots_dir, "duplication_nearest_negative_cosine_hist.png"
        ),
        "threshold_stability": os.path.join(plots_dir, "threshold_stability.png"),
        "polarity_usage_stability": os.path.join(
            plots_dir, "polarity_usage_stability.png"
        ),
    }
    for anchor in DEFAULT_MATCHED_L0_ANCHORS:
        output_paths[f"gamma_hist_l0_{anchor}"] = os.path.join(
            plots_dir, f"gamma_hist_l0_{anchor}.png"
        )
    return output_paths


def plot_stage_artifact_paths(output_dir: str) -> Dict[str, str]:
    output_paths = dict(canonical_plot_paths(output_dir=output_dir))
    output_paths["matched_points"] = matched_points_path(output_dir=output_dir)
    matched_tables = matched_table_paths(output_dir=output_dir)
    output_paths["matched_l0_points_table"] = matched_tables["matched_l0_points"]
    output_paths["matched_lr_points_table"] = matched_tables["matched_lr_points"]
    output_paths["polarity_usage_stability_table"] = (
        _polarity_usage_stability_table_path(output_dir=output_dir)
    )
    return output_paths


def _save_no_data_plot(path: str, title: str, message: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        transform=axis.transAxes,
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        return None
    return payload


def _load_or_compute_matched_points(
    output_dir: str, summary: Dict[str, Any]
) -> Dict[str, Any]:
    path = matched_points_path(output_dir=output_dir)
    loaded_payload = _load_json_if_exists(path)
    if loaded_payload is not None:
        anchors = loaded_payload.get("anchors")
        per_seed = loaded_payload.get("per_seed")
        aggregate = loaded_payload.get("aggregate")
        if (
            isinstance(anchors, dict)
            and isinstance(per_seed, list)
            and isinstance(aggregate, list)
        ):
            return loaded_payload
    return compute_and_persist_matched_points(summary=summary, output_dir=output_dir)


def matched_table_paths(output_dir: str) -> Dict[str, str]:
    tables_dir = os.path.join(output_dir, "tables")
    return {
        "matched_l0_points": os.path.join(tables_dir, "matched_l0_points.csv"),
        "matched_lr_points": os.path.join(tables_dir, "matched_lr_points.csv"),
    }


def _matched_table_rows(
    matched_points_payload: Dict[str, Any], anchor_type: str
) -> List[Dict[str, Any]]:
    per_seed = matched_points_payload.get("per_seed", [])
    if not isinstance(per_seed, list):
        return []
    rows: List[Dict[str, Any]] = []
    for entry in per_seed:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("anchor_type")) != str(anchor_type):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        supporting_points = entry.get("supporting_points")
        first_relative_dir: Optional[str] = None
        second_relative_dir: Optional[str] = None
        if isinstance(supporting_points, list):
            if supporting_points and isinstance(supporting_points[0], dict):
                first_relative_dir = supporting_points[0].get("relative_dir")
            if len(supporting_points) > 1 and isinstance(supporting_points[1], dict):
                second_relative_dir = supporting_points[1].get("relative_dir")
        rows.append(
            {
                "variant_slug": entry.get("variant_slug"),
                "seed": entry.get("seed"),
                "anchor_type": entry.get("anchor_type"),
                "anchor_value": entry.get("anchor_value"),
                "method": entry.get("method"),
                "distance_to_target": entry.get("distance_to_target"),
                "mean_l0": metrics.get("mean_l0"),
                "mse": metrics.get("mse"),
                "loss_recovered": metrics.get("loss_recovered"),
                "dead_fraction": metrics.get("dead_fraction"),
                "supporting_relative_dir_1": first_relative_dir,
                "supporting_relative_dir_2": second_relative_dir,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("variant_slug", "")),
            int(row.get("seed", 0)) if isinstance(row.get("seed"), int) else -1,
            (
                float(row.get("anchor_value", 0.0))
                if isinstance(row.get("anchor_value"), (int, float))
                and not isinstance(row.get("anchor_value"), bool)
                else float("inf")
            ),
        )
    )
    return rows


def _write_matched_table(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
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
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload: Dict[str, Any] = {}
            for key in fieldnames:
                payload[key] = row.get(key)
            writer.writerow(payload)


def write_matched_tables(
    output_dir: str, matched_points_payload: Dict[str, Any]
) -> Dict[str, str]:
    table_paths = matched_table_paths(output_dir=output_dir)
    l0_rows = _matched_table_rows(
        matched_points_payload=matched_points_payload,
        anchor_type="l0",
    )
    lr_rows = _matched_table_rows(
        matched_points_payload=matched_points_payload,
        anchor_type="lr",
    )
    _write_matched_table(path=table_paths["matched_l0_points"], rows=l0_rows)
    _write_matched_table(path=table_paths["matched_lr_points"], rows=lr_rows)
    return table_paths


def _plot_metric_vs_l0(
    output_path: str,
    aggregates: List[Dict[str, Any]],
    y_key: str,
    y_label: str,
    title: str,
    matched_points_aggregate: Optional[List[Dict[str, Any]]] = None,
) -> None:
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for aggregate in aggregates:
        x_value = aggregate.get("mean_l0_mean")
        y_value = aggregate.get(y_key)
        variant_slug = aggregate.get("variant_slug")
        if not isinstance(variant_slug, str) or x_value is None or y_value is None:
            continue
        grouped.setdefault(variant_slug, []).append((float(x_value), float(y_value)))

    if not grouped:
        _save_no_data_plot(
            output_path,
            title=title,
            message=f"No data available for '{y_key}'.",
        )
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    variant_colors: Dict[str, Any] = {}
    for variant_slug in sorted(grouped.keys()):
        points = sorted(grouped[variant_slug], key=lambda value: value[0])
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        line = axis.plot(x_values, y_values, marker="o", label=variant_slug)[0]
        variant_colors[variant_slug] = line.get_color()

    if isinstance(matched_points_aggregate, list):
        metric_name = (
            y_key[:-5] if y_key.endswith("_mean") else y_key.replace("_mean", "")
        )
        for entry in matched_points_aggregate:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("anchor_type")) != "l0":
                continue
            variant_slug = entry.get("variant_slug")
            if not isinstance(variant_slug, str):
                continue
            anchor_value = entry.get("anchor_value")
            metric_value = entry.get(f"{metric_name}_mean")
            if (
                isinstance(anchor_value, bool)
                or isinstance(metric_value, bool)
                or not isinstance(anchor_value, (int, float))
                or not isinstance(metric_value, (int, float))
            ):
                continue
            color = variant_colors.get(variant_slug)
            axis.scatter(
                [float(anchor_value)],
                [float(metric_value)],
                marker="X",
                s=60,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
    axis.set_xlabel("mean L0")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _select_matched_l0_entries(
    summary: Dict[str, Any],
    targets: Sequence[int],
) -> List[Dict[str, Any]]:
    subruns = summary.get("subruns", [])
    if not isinstance(subruns, list):
        return []
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for entry in subruns:
        if not isinstance(entry, dict):
            continue
        variant_slug = entry.get("variant_slug")
        seed = entry.get("seed")
        mean_l0 = entry.get("mean_l0")
        mse = entry.get("mse")
        if (
            not isinstance(variant_slug, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or mean_l0 is None
            or mse is None
        ):
            continue
        grouped.setdefault((variant_slug, int(seed)), []).append(entry)

    selected: List[Dict[str, Any]] = []
    for variant_slug, seed in sorted(
        grouped.keys(), key=lambda value: (value[0], value[1])
    ):
        candidates = grouped[(variant_slug, seed)]
        for target in targets:

            def _tie_key(entry: Dict[str, Any]) -> Tuple[float, float, float, str]:
                distance = abs(float(entry["mean_l0"]) - float(target))
                if not math.isfinite(distance):
                    distance = float("inf")
                mse = float(entry["mse"])
                if not math.isfinite(mse):
                    mse = float("inf")
                sweep_value = entry.get("sweep_value")
                if isinstance(sweep_value, bool):
                    sweep_sort_value = float("inf")
                elif isinstance(sweep_value, (int, float)):
                    sweep_sort_value = float(sweep_value)
                else:
                    sweep_sort_value = float("inf")
                if not math.isfinite(sweep_sort_value):
                    sweep_sort_value = float("inf")
                relative_dir = str(entry.get("relative_dir", ""))
                return (distance, mse, sweep_sort_value, relative_dir)

            best = min(
                candidates,
                key=_tie_key,
            )
            selected.append(
                {
                    "variant_slug": variant_slug,
                    "seed": int(seed),
                    "target": int(target),
                    "selected_sweep_param": best.get("sweep_param"),
                    "selected_sweep_value": best.get("sweep_value"),
                    "selected_mean_l0": float(best["mean_l0"]),
                    "selected_mse": float(best["mse"]),
                    "relative_dir": best.get("relative_dir"),
                }
            )
    return selected


def compute_matched_l0_anchors(
    summary: Dict[str, Any], targets: Sequence[int]
) -> Dict[str, Any]:
    normalized_targets = [int(value) for value in targets]
    selected = _select_matched_l0_entries(
        summary=summary,
        targets=normalized_targets,
    )
    return {
        "targets": normalized_targets,
        "selected": selected,
    }


def _augment_summary_with_matched_l0(
    output_dir: str, summary: Dict[str, Any]
) -> Dict[str, Any]:
    summary["matched_l0_anchors"] = compute_matched_l0_anchors(
        summary=summary,
        targets=DEFAULT_MATCHED_L0_ANCHORS,
    )
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)
    return summary


def ensure_summary_has_matched_l0_anchors(
    output_dir: str, summary: Dict[str, Any]
) -> Dict[str, Any]:
    matched = summary.get("matched_l0_anchors")
    if isinstance(matched, dict):
        targets = matched.get("targets")
        selected = matched.get("selected")
        if isinstance(targets, list) and isinstance(selected, list):
            parsed_targets: Optional[List[int]] = []
            for value in targets:
                if isinstance(value, bool) or not isinstance(value, int):
                    parsed_targets = None
                    break
                parsed_targets.append(int(value))
            if parsed_targets == [int(value) for value in DEFAULT_MATCHED_L0_ANCHORS]:
                return summary
    return _augment_summary_with_matched_l0(output_dir=output_dir, summary=summary)


def _iter_selected_subrun_dirs_for_target(
    summary: Dict[str, Any], target: int
) -> Iterable[str]:
    matched = summary.get("matched_l0_anchors", {})
    if not isinstance(matched, dict):
        return []
    selected = matched.get("selected", [])
    if not isinstance(selected, list):
        return []
    paths: List[str] = []
    for entry in selected:
        if not isinstance(entry, dict):
            continue
        if int(entry.get("target", -1)) != int(target):
            continue
        relative_dir = entry.get("relative_dir")
        if isinstance(relative_dir, str) and relative_dir:
            paths.append(relative_dir)
    return paths


def _plot_gamma_histogram_for_anchor(
    output_dir: str,
    summary: Dict[str, Any],
    target: int,
    output_path: str,
) -> None:
    gamma_pos_values: List[float] = []
    gamma_neg_values: List[float] = []
    selected_dirs = _iter_selected_subrun_dirs_for_target(
        summary=summary, target=target
    )
    for relative_dir in selected_dirs:
        gamma_path = os.path.join(output_dir, relative_dir, "gamma_metrics.json")
        gamma_payload = _load_json_if_exists(gamma_path)
        if gamma_payload is None:
            continue
        gamma_pos = gamma_payload.get("gamma_pos_values", [])
        if isinstance(gamma_pos, list):
            for value in gamma_pos:
                if value is not None:
                    gamma_pos_values.append(float(value))
        gamma_neg = gamma_payload.get("gamma_neg_values", [])
        if isinstance(gamma_neg, list):
            for value in gamma_neg:
                if value is not None:
                    gamma_neg_values.append(float(value))

    if not gamma_pos_values and not gamma_neg_values:
        _save_no_data_plot(
            output_path,
            title=f"Gamma Histogram (L0 target {target})",
            message="No gamma values available for matched subruns.",
        )
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    if gamma_pos_values:
        axis.hist(
            gamma_pos_values,
            bins=40,
            alpha=0.65,
            color="#1f77b4",
            label="gamma+",
        )
    if gamma_neg_values:
        axis.hist(
            gamma_neg_values,
            bins=40,
            alpha=0.65,
            color="#d62728",
            label="gamma-",
        )
    axis.set_xlabel("gamma")
    axis.set_ylabel("count")
    axis.set_title(f"Gamma Histogram (L0 target {target})")
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_polarity_imbalance_histogram(
    output_dir: str,
    summary: Dict[str, Any],
    output_path: str,
    anchor_target: int = 64,
) -> None:
    pooled_values: List[float] = []
    selected_dirs = _iter_selected_subrun_dirs_for_target(
        summary=summary, target=anchor_target
    )
    for relative_dir in selected_dirs:
        evaluation_payload = _load_json_if_exists(
            os.path.join(output_dir, relative_dir, "evaluation.json")
        )
        if evaluation_payload is None:
            continue
        values = evaluation_payload.get("polarity_imbalance_by_latent", [])
        if not isinstance(values, list):
            continue
        for value in values:
            if value is not None:
                pooled_values.append(float(value))
    if not pooled_values:
        _save_no_data_plot(
            output_path,
            title="Polarity Imbalance Histogram",
            message="No sign-aware polarity imbalance values available.",
        )
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    axis.hist(pooled_values, bins=40, alpha=0.75, color="#2ca02c")
    axis.set_xlabel("polarity imbalance")
    axis.set_ylabel("count")
    axis.set_title("Polarity Imbalance Histogram")
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _sample_decoder_rows(
    decoder_rows: torch.Tensor,
    sample_size: int,
    seed: int,
) -> torch.Tensor:
    total_features = int(decoder_rows.shape[0])
    if total_features <= sample_size:
        return decoder_rows
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected_indices = torch.randperm(total_features, generator=generator)[:sample_size]
    return decoder_rows.index_select(0, selected_indices)


def _compute_duplication_metrics_for_decoder(
    decoder_rows: torch.Tensor,
) -> Dict[str, Any]:
    normalized = decoder_rows / (decoder_rows.norm(dim=1, keepdim=True) + 1e-8)
    count_rows = int(normalized.shape[0])
    if count_rows <= 1:
        return {
            "sample_size": count_rows,
            "min_cosine_values": [],
            "counts_below_thresholds": {
                str(value): 0 for value in _DUPLICATION_THRESHOLDS
            },
            "pair_count": 0,
        }

    min_cosine = torch.full((count_rows,), 1.0, dtype=torch.float32)
    threshold_counts = {str(value): 0 for value in _DUPLICATION_THRESHOLDS}
    pair_count = 0
    for row_start in range(0, count_rows, _DUPLICATION_CHUNK_SIZE):
        row_end = min(row_start + _DUPLICATION_CHUNK_SIZE, count_rows)
        block_row = normalized[row_start:row_end]
        for col_start in range(row_start, count_rows, _DUPLICATION_CHUNK_SIZE):
            col_end = min(col_start + _DUPLICATION_CHUNK_SIZE, count_rows)
            block_col = normalized[col_start:col_end]
            similarity = block_row @ block_col.T
            if row_start == col_start:
                diagonal_count = int(similarity.shape[0])
                similarity[
                    torch.arange(diagonal_count), torch.arange(diagonal_count)
                ] = 1.0
                upper_mask = torch.triu(
                    torch.ones_like(similarity, dtype=torch.bool), diagonal=1
                )
                comparison_values = similarity[upper_mask]
            else:
                comparison_values = similarity.reshape(-1)
            pair_count += int(comparison_values.numel())
            for threshold in _DUPLICATION_THRESHOLDS:
                threshold_counts[str(threshold)] += int(
                    (comparison_values < float(threshold)).sum().item()
                )
            min_cosine[row_start:row_end] = torch.minimum(
                min_cosine[row_start:row_end], similarity.min(dim=1).values.cpu()
            )
            if row_start != col_start:
                min_cosine[col_start:col_end] = torch.minimum(
                    min_cosine[col_start:col_end], similarity.min(dim=0).values.cpu()
                )
    return {
        "sample_size": int(count_rows),
        "min_cosine_values": [float(value) for value in min_cosine.tolist()],
        "counts_below_thresholds": threshold_counts,
        "pair_count": int(pair_count),
    }


def _resolve_alive_feature_indices_for_duplication(
    output_dir: str,
    relative_dir: str,
    d_hidden: int,
) -> Optional[torch.Tensor]:
    evaluation_payload = _load_json_if_exists(
        os.path.join(output_dir, relative_dir, "evaluation.json")
    )
    if evaluation_payload is None:
        return None
    alive_indices_raw = evaluation_payload.get("alive_indices")
    if not isinstance(alive_indices_raw, list):
        return None
    if not alive_indices_raw:
        return torch.empty(0, dtype=torch.long)

    valid_indices: List[int] = []
    for value in alive_indices_raw:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if 0 <= int(value) < int(d_hidden):
            valid_indices.append(int(value))
    if not valid_indices:
        return None
    deduped_indices = sorted(set(valid_indices))
    return torch.tensor(deduped_indices, dtype=torch.long)


def _stable_duplication_seed(relative_dir: str) -> int:
    digest = hashlib.sha256(relative_dir.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_or_compute_duplication_metrics(
    output_dir: str,
    relative_dir: str,
) -> Optional[Dict[str, Any]]:
    subrun_dir = os.path.join(output_dir, relative_dir)
    metrics_path = os.path.join(subrun_dir, "duplication_metrics.json")
    loaded_metrics = _load_json_if_exists(metrics_path)
    if loaded_metrics is not None:
        return loaded_metrics

    checkpoint_path = os.path.join(subrun_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        return None
    w_dec = state_dict.get("W_dec")
    if not torch.is_tensor(w_dec) or w_dec.ndim != 3 or int(w_dec.shape[0]) != 1:
        return None

    decoder_rows = w_dec[0].transpose(0, 1).to(dtype=torch.float32, device="cpu")
    alive_indices = _resolve_alive_feature_indices_for_duplication(
        output_dir=output_dir,
        relative_dir=relative_dir,
        d_hidden=int(decoder_rows.shape[0]),
    )
    if alive_indices is None:
        candidate_decoder_rows = decoder_rows
        alive_filter_applied = False
    elif int(alive_indices.numel()) == 0:
        candidate_decoder_rows = decoder_rows[:0]
        alive_filter_applied = True
    else:
        candidate_decoder_rows = decoder_rows.index_select(0, alive_indices)
        alive_filter_applied = True

    sample_seed = _stable_duplication_seed(relative_dir=relative_dir)
    sampled_decoder_rows = _sample_decoder_rows(
        decoder_rows=candidate_decoder_rows,
        sample_size=_DUPLICATION_SAMPLE_SIZE,
        seed=sample_seed,
    )
    metrics = _compute_duplication_metrics_for_decoder(sampled_decoder_rows)
    metrics["schema_version"] = 1
    metrics["thresholds"] = [float(value) for value in _DUPLICATION_THRESHOLDS]
    metrics["seed"] = int(sample_seed)
    metrics["relative_dir"] = relative_dir
    metrics["candidate_feature_count"] = int(candidate_decoder_rows.shape[0])
    metrics["alive_filter_applied"] = bool(alive_filter_applied)
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def _plot_duplication_histogram(
    output_dir: str,
    summary: Dict[str, Any],
    output_path: str,
    anchor_target: int = 64,
) -> None:
    pooled_min_cosine: List[float] = []
    aggregate_counts = {str(value): 0 for value in _DUPLICATION_THRESHOLDS}
    pair_count = 0
    selected_dirs = _iter_selected_subrun_dirs_for_target(
        summary=summary, target=anchor_target
    )
    for relative_dir in selected_dirs:
        metrics = _load_or_compute_duplication_metrics(
            output_dir=output_dir, relative_dir=relative_dir
        )
        if metrics is None:
            continue
        values = metrics.get("min_cosine_values", [])
        if isinstance(values, list):
            for value in values:
                pooled_min_cosine.append(float(value))
        counts = metrics.get("counts_below_thresholds", {})
        if isinstance(counts, dict):
            for threshold in _DUPLICATION_THRESHOLDS:
                key = str(threshold)
                value = counts.get(key)
                if isinstance(value, (int, float)):
                    aggregate_counts[key] += int(value)
        pair_value = metrics.get("pair_count")
        if isinstance(pair_value, int):
            pair_count += int(pair_value)
    if not pooled_min_cosine:
        _save_no_data_plot(
            output_path,
            title="Decoder Duplication Histogram",
            message="No decoder duplication metrics available.",
        )
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    axis.hist(pooled_min_cosine, bins=40, alpha=0.75, color="#9467bd")
    axis.set_xlabel("nearest negative cosine")
    axis.set_ylabel("count")
    axis.set_title("Decoder Near-Opposite Duplication")
    annotation_lines = [f"pairs: {pair_count}"]
    for threshold in _DUPLICATION_THRESHOLDS:
        annotation_lines.append(
            f"cos<{threshold:.2f}: {aggregate_counts[str(threshold)]}"
        )
    axis.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _extract_step_value_pairs(
    history: List[Any], metric_key: str
) -> Tuple[List[float], List[float]]:
    steps: List[float] = []
    values: List[float] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        value = entry.get(metric_key)
        if (
            isinstance(step, bool)
            or isinstance(value, bool)
            or not isinstance(step, (int, float))
            or not isinstance(value, (int, float))
        ):
            continue
        steps.append(float(step))
        values.append(float(value))
    return steps, values


def _polarity_usage_stability_table_path(output_dir: str) -> str:
    return os.path.join(output_dir, "tables", "polarity_usage_stability.csv")


def _load_last_training_history_entry(
    output_dir: str, relative_dir: str
) -> Optional[Dict[str, Any]]:
    training_metrics = _load_json_if_exists(
        os.path.join(output_dir, relative_dir, "training_metrics.json")
    )
    if training_metrics is None:
        return None
    history = training_metrics.get("history")
    if not isinstance(history, list) or not history:
        return None
    last_entry = history[-1]
    if not isinstance(last_entry, dict):
        return None
    return last_entry


def _collect_polarity_usage_stability_rows(
    output_dir: str, summary: Dict[str, Any]
) -> List[Dict[str, Any]]:
    subruns = summary.get("subruns", [])
    if not isinstance(subruns, list):
        return []

    output: List[Dict[str, Any]] = []
    for subrun in subruns:
        if not isinstance(subrun, dict):
            continue
        variant_slug = subrun.get("variant_slug")
        seed = subrun.get("seed")
        sweep_param = subrun.get("sweep_param")
        sweep_value = subrun.get("sweep_value")
        relative_dir = subrun.get("relative_dir")
        if (
            not isinstance(variant_slug, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(sweep_param, str)
            or isinstance(sweep_value, bool)
            or not isinstance(sweep_value, (int, float))
            or not isinstance(relative_dir, str)
            or not relative_dir
        ):
            continue

        history_entry = _load_last_training_history_entry(
            output_dir=output_dir, relative_dir=relative_dir
        )
        if history_entry is None:
            continue

        metrics: Dict[str, float] = {}
        is_valid = True
        for key in _POLARITY_STABILITY_METRIC_KEYS:
            value = history_entry.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                is_valid = False
                break
            metrics[key] = float(value)
        if not is_valid:
            continue

        output.append(
            {
                "variant_slug": variant_slug,
                "seed": int(seed),
                "sweep_param": sweep_param,
                "sweep_value": float(sweep_value),
                **metrics,
            }
        )
    output.sort(
        key=lambda row: (
            str(row["variant_slug"]),
            str(row["sweep_param"]),
            float(row["sweep_value"]),
            int(row["seed"]),
        )
    )
    return output


def _aggregate_polarity_usage_stability_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for row in rows:
        variant_slug = row.get("variant_slug")
        sweep_param = row.get("sweep_param")
        sweep_value = row.get("sweep_value")
        if (
            not isinstance(variant_slug, str)
            or not isinstance(sweep_param, str)
            or isinstance(sweep_value, bool)
            or not isinstance(sweep_value, (int, float))
        ):
            continue
        grouped.setdefault(
            (variant_slug, sweep_param, float(sweep_value)),
            [],
        ).append(dict(row))

    output: List[Dict[str, Any]] = []
    for variant_slug, sweep_param, sweep_value in sorted(
        grouped.keys(), key=lambda item: (item[0], item[1], item[2])
    ):
        entries = grouped[(variant_slug, sweep_param, sweep_value)]
        aggregated: Dict[str, Any] = {
            "variant_slug": variant_slug,
            "sweep_param": sweep_param,
            "sweep_value": float(sweep_value),
            "num_seeds": int(len(entries)),
        }
        for key in _POLARITY_STABILITY_METRIC_KEYS:
            values = np.array(
                [float(entry[key]) for entry in entries],
                dtype=np.float64,
            )
            mean = float(values.mean()) if values.size else float("nan")
            se = (
                float(values.std(ddof=0) / math.sqrt(float(values.size)))
                if values.size
                else float("nan")
            )
            aggregated[f"{key}_mean"] = mean
            aggregated[f"{key}_se"] = se
        output.append(aggregated)
    return output


def _write_polarity_usage_stability_table(
    path: str, rows: Sequence[Dict[str, Any]]
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "variant_slug",
        "sweep_param",
        "sweep_value",
        "num_seeds",
    ]
    for key in _POLARITY_STABILITY_METRIC_KEYS:
        fieldnames.append(f"{key}_mean")
        fieldnames.append(f"{key}_se")
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _plot_polarity_usage_stability(
    output_dir: str,
    summary: Dict[str, Any],
    output_path: str,
) -> str:
    table_path = _polarity_usage_stability_table_path(output_dir=output_dir)
    per_subrun_rows = _collect_polarity_usage_stability_rows(
        output_dir=output_dir, summary=summary
    )
    aggregate_rows = _aggregate_polarity_usage_stability_rows(rows=per_subrun_rows)
    _write_polarity_usage_stability_table(path=table_path, rows=aggregate_rows)
    if not aggregate_rows:
        _save_no_data_plot(
            output_path,
            title="Polarity Usage Stability",
            message=(
                "No polarity-usage stability data available "
                "(missing training history or required keys)."
            ),
        )
        return table_path

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), dpi=150, sharex=True)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in aggregate_rows:
        grouped.setdefault(
            (str(row["variant_slug"]), str(row["sweep_param"])),
            [],
        ).append(row)

    for (variant_slug, sweep_param), rows in sorted(grouped.items()):
        sorted_rows = sorted(rows, key=lambda row: float(row["sweep_value"]))
        x_values = [float(row["sweep_value"]) for row in sorted_rows]
        label_prefix = f"{variant_slug} ({sweep_param})"

        positive_values = [
            float(row["polarity_positive_fraction_nonzero_mean"]) for row in sorted_rows
        ]
        negative_values = [
            float(row["polarity_negative_fraction_nonzero_mean"]) for row in sorted_rows
        ]
        bidirectional_values = [
            float(row["polarity_bidirectional_usage_fraction_mean"])
            for row in sorted_rows
        ]
        bidirectional_se = [
            float(row["polarity_bidirectional_usage_fraction_se"])
            for row in sorted_rows
        ]
        flip_values = [
            float(row["polarity_dominance_flip_rate_mean"]) for row in sorted_rows
        ]
        flip_se = [float(row["polarity_dominance_flip_rate_se"]) for row in sorted_rows]

        axes[0].plot(
            x_values,
            positive_values,
            marker="o",
            label=f"{label_prefix}: pos",
            alpha=0.9,
        )
        axes[0].plot(
            x_values,
            negative_values,
            marker="s",
            linestyle="--",
            label=f"{label_prefix}: neg",
            alpha=0.9,
        )

        axes[1].plot(
            x_values,
            bidirectional_values,
            marker="o",
            label=label_prefix,
            alpha=0.9,
        )
        axes[1].fill_between(
            x_values,
            [
                value - error
                for value, error in zip(bidirectional_values, bidirectional_se)
            ],
            [
                value + error
                for value, error in zip(bidirectional_values, bidirectional_se)
            ],
            alpha=0.2,
        )

        axes[2].plot(
            x_values,
            flip_values,
            marker="o",
            label=label_prefix,
            alpha=0.9,
        )
        axes[2].fill_between(
            x_values,
            [value - error for value, error in zip(flip_values, flip_se)],
            [value + error for value, error in zip(flip_values, flip_se)],
            alpha=0.2,
        )

    axes[0].set_title("Polarity Usage Stability Across Seeds and Sweep Values")
    axes[0].set_ylabel("positive/negative fraction")
    axes[1].set_ylabel("bidirectional usage")
    axes[2].set_ylabel("flip rate")
    axes[2].set_xlabel("sweep value (lambda or k)")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return table_path


def _plot_threshold_stability(
    output_dir: str,
    summary: Dict[str, Any],
    output_path: str,
    anchor_target: int = 64,
) -> None:
    selected_dirs = _iter_selected_subrun_dirs_for_target(
        summary=summary, target=anchor_target
    )
    plotted_any = False
    figure, (axis_thresholds, axis_dead_zone) = plt.subplots(
        2, 1, figsize=(9.0, 7.0), dpi=150, sharex=True
    )
    for relative_dir in selected_dirs:
        metrics = _load_json_if_exists(
            os.path.join(output_dir, relative_dir, "training_metrics.json")
        )
        if metrics is None:
            continue
        history = metrics.get("history", [])
        if not isinstance(history, list) or not history:
            continue
        delta_pos_steps, delta_pos = _extract_step_value_pairs(
            history=history, metric_key="delta_pos_mean"
        )
        delta_neg_steps, delta_neg = _extract_step_value_pairs(
            history=history, metric_key="delta_neg_mean"
        )
        theta_steps, theta_values = _extract_step_value_pairs(
            history=history, metric_key="theta_mean"
        )
        jump_steps, jump_values = _extract_step_value_pairs(
            history=history, metric_key="jump_mean"
        )
        dead_zone_steps, dead_zone = _extract_step_value_pairs(
            history=history, metric_key="dead_zone_frac_mean"
        )
        label_prefix = relative_dir.replace("\\", "/")
        if delta_pos:
            axis_thresholds.plot(
                delta_pos_steps,
                delta_pos,
                label=f"{label_prefix}: delta+",
                alpha=0.85,
            )
            plotted_any = True
        if delta_neg:
            axis_thresholds.plot(
                delta_neg_steps,
                delta_neg,
                label=f"{label_prefix}: delta-",
                alpha=0.85,
                linestyle="--",
            )
            plotted_any = True
        if theta_values:
            axis_thresholds.plot(
                theta_steps,
                theta_values,
                label=f"{label_prefix}: theta",
                alpha=0.85,
                linestyle=":",
            )
            plotted_any = True
        if jump_values:
            axis_thresholds.plot(
                jump_steps,
                jump_values,
                label=f"{label_prefix}: jump",
                alpha=0.85,
                linestyle="-.",
            )
            plotted_any = True
        if dead_zone:
            axis_dead_zone.plot(
                dead_zone_steps,
                dead_zone,
                label=label_prefix,
                alpha=0.85,
            )
            plotted_any = True

    if not plotted_any:
        plt.close(figure)
        _save_no_data_plot(
            output_path,
            title="Threshold Stability",
            message="No threshold history data available.",
        )
        return

    axis_thresholds.set_ylabel("threshold statistic")
    axis_thresholds.set_title("Threshold Stability")
    axis_thresholds.grid(True, linestyle="--", alpha=0.35)
    axis_thresholds.legend(fontsize=7, ncol=1, loc="best")

    axis_dead_zone.set_xlabel("training step")
    axis_dead_zone.set_ylabel("dead-zone fraction")
    axis_dead_zone.grid(True, linestyle="--", alpha=0.35)
    axis_dead_zone.legend(fontsize=7, ncol=1, loc="best")

    figure.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def generate_plots_for_run(output_dir: str) -> Dict[str, Any]:
    comparison_path = os.path.join(output_dir, "comparison_summary.json")
    summary = _load_json_if_exists(comparison_path)
    if summary is None:
        raise ValueError(
            "comparison_summary.json is required before plot stage can run."
        )
    summary = ensure_summary_has_matched_l0_anchors(
        output_dir=output_dir, summary=summary
    )
    matched_points_payload = compute_and_persist_matched_points(
        output_dir=output_dir,
        summary=summary,
    )
    matched_points_aggregate = matched_points_payload.get("aggregate", [])
    if not isinstance(matched_points_aggregate, list):
        matched_points_aggregate = []
    plot_paths = canonical_plot_paths(output_dir=output_dir)
    aggregates = summary.get("aggregates", [])
    if not isinstance(aggregates, list):
        aggregates = []

    _plot_metric_vs_l0(
        output_path=plot_paths["recon_vs_l0"],
        aggregates=aggregates,
        y_key="mse_mean",
        y_label="MSE",
        title="Reconstruction Error vs Mean L0",
        matched_points_aggregate=matched_points_aggregate,
    )
    _plot_metric_vs_l0(
        output_path=plot_paths["dead_fraction_vs_l0"],
        aggregates=aggregates,
        y_key="dead_fraction_mean",
        y_label="dead fraction",
        title="Dead Fraction vs Mean L0",
        matched_points_aggregate=matched_points_aggregate,
    )
    _plot_metric_vs_l0(
        output_path=plot_paths["loss_recovered_vs_l0"],
        aggregates=aggregates,
        y_key="loss_recovered_mean",
        y_label="loss recovered",
        title="Loss Recovered vs Mean L0",
        matched_points_aggregate=matched_points_aggregate,
    )
    for anchor in DEFAULT_MATCHED_L0_ANCHORS:
        _plot_gamma_histogram_for_anchor(
            output_dir=output_dir,
            summary=summary,
            target=anchor,
            output_path=plot_paths[f"gamma_hist_l0_{anchor}"],
        )
    _plot_polarity_imbalance_histogram(
        output_dir=output_dir,
        summary=summary,
        output_path=plot_paths["polarity_imbalance_hist"],
    )
    _plot_duplication_histogram(
        output_dir=output_dir,
        summary=summary,
        output_path=plot_paths["duplication_nearest_negative_cosine_hist"],
    )
    _plot_threshold_stability(
        output_dir=output_dir,
        summary=summary,
        output_path=plot_paths["threshold_stability"],
    )
    polarity_usage_table_path = _plot_polarity_usage_stability(
        output_dir=output_dir,
        summary=summary,
        output_path=plot_paths["polarity_usage_stability"],
    )
    table_paths = write_matched_tables(
        output_dir=output_dir,
        matched_points_payload=matched_points_payload,
    )
    table_paths["polarity_usage_stability"] = polarity_usage_table_path
    return {
        "plots_written": len(plot_paths),
        "plot_paths": plot_paths,
        "table_paths": table_paths,
    }
