import json
import math
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from sgnsae.experiments.directional_deviation.data import (
    LABEL_TYPE_HIGH,
    LABEL_TYPE_LOW,
)
from sgnsae.experiments.directional_deviation.scoring import (
    BaselineEvidenceThresholds,
    SignAwareLatentThresholds,
    assign_baseline_latent_directions,
    calculate_baseline_directional_scores,
    calculate_sign_aware_directional_scores,
    extract_detection_signal,
    extract_one_sided_evidence,
)


def generate_plots_for_run(run_dir: str) -> None:
    comparison_summary = _load_json_if_present(
        os.path.join(run_dir, "comparison_summary.json")
    )
    if comparison_summary is None:
        return

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    sweep = comparison_summary.get("sweep")
    if isinstance(sweep, dict):
        sweep_type = sweep.get("type")
        if sweep_type == "width":
            _plot_width_sweep_summary(plots_dir=plots_dir, summary=comparison_summary)
        if sweep_type == "seed":
            _plot_seed_sweep_summary(plots_dir=plots_dir, summary=comparison_summary)
        return

    models = comparison_summary.get("models")
    if isinstance(models, dict):
        _plot_leaf_run_summary(plots_dir=plots_dir, models=models)
        _plot_score_distributions_for_models(run_dir=run_dir, plots_dir=plots_dir)


def _plot_width_sweep_summary(plots_dir: str, summary: Dict[str, Any]) -> None:
    sweep = summary.get("sweep")
    per_width = summary.get("per_width")
    if not isinstance(sweep, dict) or not isinstance(per_width, dict):
        return

    widths_raw = sweep.get("widths")
    if not isinstance(widths_raw, list) or not widths_raw:
        return

    widths: List[int] = [int(value) for value in widths_raw]
    widths_sorted = sorted(widths)

    series = _collect_width_sweep_series(per_width=per_width, widths=widths_sorted)
    if not series:
        return

    _plot_recall_vs_width(
        plots_dir=plots_dir,
        widths=widths_sorted,
        series=series,
        filename="recall_vs_width.png",
    )

    for metric, title, filename in (
        ("recall_high", "Recall (High anomalies)", "recall_high_vs_width.png"),
        ("recall_low", "Recall (Low anomalies)", "recall_low_vs_width.png"),
        (
            "covered_channels_fraction",
            "Covered Channels Fraction",
            "coverage_vs_width.png",
        ),
        ("fpr_high", "False Positive Rate (High)", "fpr_high_vs_width.png"),
        ("fpr_low", "False Positive Rate (Low)", "fpr_low_vs_width.png"),
    ):
        _plot_metric_lines(
            plots_dir=plots_dir,
            widths=widths_sorted,
            series=series,
            metric=metric,
            title=title,
            filename=filename,
        )


def _plot_recall_vs_width(
    plots_dir: str,
    widths: Sequence[int],
    series: Dict[str, Dict[int, Dict[str, Any]]],
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)

    model_names = sorted(series.keys())
    for index, model_name in enumerate(model_names):
        recalls_high: List[float] = []
        recalls_low: List[float] = []
        for width in widths:
            width_metrics = series[model_name].get(int(width))
            recall_high = (
                _extract_metric_value(width_metrics, "recall_high")
                if width_metrics
                else None
            )
            recall_low = (
                _extract_metric_value(width_metrics, "recall_low")
                if width_metrics
                else None
            )
            recalls_high.append(np.nan if recall_high is None else float(recall_high))
            recalls_low.append(np.nan if recall_low is None else float(recall_low))

        color = _color_for_index(index)
        ax.plot(
            list(widths),
            recalls_high,
            marker="o",
            linewidth=2,
            linestyle="-",
            label=f"{model_name} (high)",
            color=color,
        )
        ax.plot(
            list(widths),
            recalls_low,
            marker="s",
            linewidth=2,
            linestyle="--",
            label=f"{model_name} (low)",
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel("Width (d_hidden)")
    ax.set_ylabel("recall")
    ax.set_title("Directional Recall @ Fixed FPR vs Width")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(-0.02, 1.02)

    path = os.path.join(plots_dir, filename)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _collect_width_sweep_series(
    per_width: Dict[str, Any],
    widths: Sequence[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    series: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for width in widths:
        width_entry = per_width.get(str(width))
        models = _extract_models_dict_from_summary(width_entry)
        if not models:
            continue
        for model_name, metrics in models.items():
            if not isinstance(metrics, dict):
                continue
            series.setdefault(str(model_name), {})[int(width)] = metrics
    return series


def _plot_metric_lines(
    plots_dir: str,
    widths: Sequence[int],
    series: Dict[str, Dict[int, Dict[str, Any]]],
    metric: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)

    model_names = sorted(series.keys())
    for index, model_name in enumerate(model_names):
        values = []
        for width in widths:
            width_metrics = series[model_name].get(int(width))
            value = (
                _extract_metric_value(width_metrics, metric) if width_metrics else None
            )
            values.append(np.nan if value is None else float(value))
        ax.plot(
            list(widths),
            values,
            marker="o",
            linewidth=2,
            label=model_name,
            color=_color_for_index(index),
        )

    ax.set_xlabel("Width (d_hidden)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="best")
    if metric.startswith("recall") or metric.startswith("covered_channels"):
        ax.set_ylim(-0.02, 1.02)

    path = os.path.join(plots_dir, filename)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_seed_sweep_summary(plots_dir: str, summary: Dict[str, Any]) -> None:
    aggregate = summary.get("aggregate_over_seeds")
    aggregate_models = None
    if isinstance(aggregate, dict):
        aggregate_models = aggregate.get("models")
    if not isinstance(aggregate_models, dict):
        return

    model_names = sorted(aggregate_models.keys())
    if not model_names:
        return

    metric_specs = (
        (
            "recall_high",
            "Recall High (mean)",
            "recall_high_mean",
            "recall_high_seed_sweep.png",
        ),
        (
            "recall_low",
            "Recall Low (mean)",
            "recall_low_mean",
            "recall_low_seed_sweep.png",
        ),
        (
            "covered_channels_fraction",
            "Covered Channels Fraction (mean)",
            "covered_channels_fraction_mean",
            "coverage_seed_sweep.png",
        ),
    )
    for base_metric, title, mean_key, filename in metric_specs:
        means: List[float] = []
        stds: List[float] = []
        for model_name in model_names:
            metrics = aggregate_models.get(model_name, {})
            mean_value = _extract_metric_value(metrics, mean_key)
            std_value = _extract_metric_value(metrics, f"{base_metric}_std")
            means.append(np.nan if mean_value is None else float(mean_value))
            stds.append(0.0 if std_value is None else float(std_value))

        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
        indices = np.arange(len(model_names))
        ax.bar(
            indices,
            means,
            yerr=stds,
            capsize=4,
            color=[_color_for_index(i) for i in indices],
        )
        ax.set_xticks(indices)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylabel(base_metric)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if base_metric.startswith("recall") or base_metric.startswith(
            "covered_channels"
        ):
            ax.set_ylim(-0.02, 1.02)

        path = os.path.join(plots_dir, filename)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def _plot_leaf_run_summary(plots_dir: str, models: Dict[str, Any]) -> None:
    model_names = sorted(models.keys())
    if not model_names:
        return

    recalls_high = [
        _extract_metric_value(models.get(name, {}), "recall_high")
        for name in model_names
    ]
    recalls_low = [
        _extract_metric_value(models.get(name, {}), "recall_low")
        for name in model_names
    ]
    coverage = [
        _extract_metric_value(models.get(name, {}), "covered_channels_fraction")
        for name in model_names
    ]

    data = {
        "recall_high": recalls_high,
        "recall_low": recalls_low,
        "coverage": coverage,
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.5, 7.2), dpi=150, sharex=True)
    for axis, (metric, values) in zip(axes, data.items()):
        numeric = [np.nan if value is None else float(value) for value in values]
        axis.bar(
            np.arange(len(model_names)),
            numeric,
            color=[_color_for_index(i) for i in range(len(model_names))],
        )
        axis.set_ylabel(metric)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[-1].set_xticks(np.arange(len(model_names)))
    axes[-1].set_xticklabels(model_names, rotation=30, ha="right")
    fig.suptitle("Directional Deviation Metrics")
    path = os.path.join(plots_dir, "leaf_summary.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_score_distributions_for_models(run_dir: str, plots_dir: str) -> None:
    preview = _load_torch_if_present(os.path.join(run_dir, "data", "sampled_data.pt"))
    if preview is None:
        return

    anomaly_manifest = _load_json_if_present(
        os.path.join(run_dir, "data", "anomaly_manifest.json")
    )
    if anomaly_manifest is None:
        return

    models_root = os.path.join(run_dir, "models")
    if not os.path.isdir(models_root):
        return

    model_names = sorted(
        name
        for name in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, name))
    )
    for model_name in model_names:
        model_dir = os.path.join(models_root, model_name)
        try:
            _plot_score_distributions_for_model(
                model_dir=model_dir,
                model_name=model_name,
                anomaly_manifest=anomaly_manifest,
                preview=preview,
                plots_root=plots_dir,
            )
        except Exception:
            continue

    _plot_score_histograms_grid_for_models(run_dir=run_dir, plots_dir=plots_dir)


def _plot_score_histograms_grid_for_models(run_dir: str, plots_dir: str) -> None:
    preview = _load_torch_if_present(os.path.join(run_dir, "data", "sampled_data.pt"))
    if preview is None:
        return

    anomaly_manifest = _load_json_if_present(
        os.path.join(run_dir, "data", "anomaly_manifest.json")
    )
    if anomaly_manifest is None:
        return

    models_root = os.path.join(run_dir, "models")
    if not os.path.isdir(models_root):
        return

    x_test_normal = preview.get("x_test_normal")
    x_test_anomaly = preview.get("x_test_anomaly")
    labels_type = preview.get("labels_type")
    if x_test_normal is None or x_test_anomaly is None or labels_type is None:
        return

    high_mask = labels_type == int(LABEL_TYPE_HIGH)
    low_mask = labels_type == int(LABEL_TYPE_LOW)

    plotted_models: List[Dict[str, Any]] = []
    model_names = sorted(
        name
        for name in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, name))
    )
    for model_name in model_names:
        model_dir = os.path.join(models_root, model_name)
        evaluation = _load_json_if_present(os.path.join(model_dir, "evaluation.json"))
        if evaluation is None:
            continue
        scoring_mode = evaluation.get("scoring_mode")
        if scoring_mode not in {"sign_aware", "baseline"}:
            continue

        model_path = os.path.join(model_dir, "model.pt")
        if not os.path.isfile(model_path):
            continue
        try:
            model = _load_pickled_model(model_path)
        except Exception:
            continue

        thresholds_dir = os.path.join(model_dir, "thresholds")
        score_thresholds = _load_json_if_present(
            os.path.join(thresholds_dir, "score_thresholds.json")
        )
        if score_thresholds is None:
            continue

        theta_high_raw = score_thresholds.get("theta_high")
        theta_low_raw = score_thresholds.get("theta_low")
        if not isinstance(theta_high_raw, (int, float)) or not isinstance(
            theta_low_raw, (int, float)
        ):
            continue
        theta_high = float(theta_high_raw)
        theta_low = float(theta_low_raw)
        if not math.isfinite(theta_high) or not math.isfinite(theta_low):
            continue

        try:
            score_high_normal, score_low_normal = _compute_directional_scores(
                model=model,
                scoring_mode=scoring_mode,
                anomaly_manifest=anomaly_manifest,
                thresholds_dir=thresholds_dir,
                x=x_test_normal,
            )
            score_high_anom, score_low_anom = _compute_directional_scores(
                model=model,
                scoring_mode=scoring_mode,
                anomaly_manifest=anomaly_manifest,
                thresholds_dir=thresholds_dir,
                x=x_test_anomaly,
            )
        except Exception:
            continue

        plotted_models.append(
            {
                "model_name": model_name,
                "theta_high": theta_high,
                "theta_low": theta_low,
                "score_high_normal": score_high_normal,
                "score_low_normal": score_low_normal,
                "score_high_high": score_high_anom[high_mask],
                "score_high_low": score_high_anom[low_mask],
                "score_low_high": score_low_anom[high_mask],
                "score_low_low": score_low_anom[low_mask],
            }
        )

    if not plotted_models:
        return

    num_models = len(plotted_models)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_models,
        figsize=(6.0 * num_models, 7.5),
        dpi=150,
        squeeze=False,
    )
    for column_index, model_entry in enumerate(plotted_models):
        model_name = str(model_entry["model_name"])
        _add_score_histogram_to_axis(
            axis=axes[0][column_index],
            title=f"{model_name} Score_high",
            normal=model_entry["score_high_normal"],
            high=model_entry["score_high_high"],
            low=model_entry["score_high_low"],
            threshold=float(model_entry["theta_high"]),
        )
        _add_score_histogram_to_axis(
            axis=axes[1][column_index],
            title=f"{model_name} Score_low",
            normal=model_entry["score_low_normal"],
            high=model_entry["score_low_high"],
            low=model_entry["score_low_low"],
            threshold=float(model_entry["theta_low"]),
        )

    path = os.path.join(plots_dir, "score_histograms.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _add_score_histogram_to_axis(
    axis: plt.Axes,
    title: str,
    normal: Tensor,
    high: Tensor,
    low: Tensor,
    threshold: float,
    max_bins: int = 60,
) -> None:
    normal_flat = normal.detach().flatten().to(device="cpu")
    high_flat = high.detach().flatten().to(device="cpu")
    low_flat = low.detach().flatten().to(device="cpu")

    normal_flat = normal_flat[torch.isfinite(normal_flat)]
    high_flat = high_flat[torch.isfinite(high_flat)]
    low_flat = low_flat[torch.isfinite(low_flat)]

    all_scores = torch.cat([normal_flat, high_flat, low_flat], dim=0)
    max_value = float(all_scores.max().item()) if all_scores.numel() > 0 else 1.0
    if math.isfinite(threshold):
        max_value = max(max_value, threshold)
    max_value = max(max_value, 1e-6)
    bins = np.linspace(0.0, max_value, num=max_bins, dtype=np.float64)

    axis.hist(
        normal_flat.numpy(), bins=bins, alpha=0.6, label="normal", color="#4c72b0"
    )
    axis.hist(
        high_flat.numpy(), bins=bins, alpha=0.6, label="high anomaly", color="#dd8452"
    )
    axis.hist(
        low_flat.numpy(), bins=bins, alpha=0.6, label="low anomaly", color="#55a868"
    )
    if math.isfinite(threshold):
        axis.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label="threshold",
        )
    axis.set_title(title)
    axis.set_xlabel("score")
    axis.set_ylabel("count")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=8, loc="best")


def _plot_score_distributions_for_model(
    model_dir: str,
    model_name: str,
    anomaly_manifest: Dict[str, Any],
    preview: Dict[str, Tensor],
    plots_root: str,
) -> None:
    evaluation = _load_json_if_present(os.path.join(model_dir, "evaluation.json"))
    if evaluation is None:
        return
    scoring_mode = evaluation.get("scoring_mode")
    if scoring_mode not in {"sign_aware", "baseline"}:
        return

    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.isfile(model_path):
        return
    model = _load_pickled_model(model_path)

    thresholds_dir = os.path.join(model_dir, "thresholds")
    score_thresholds = _load_json_if_present(
        os.path.join(thresholds_dir, "score_thresholds.json")
    )
    if score_thresholds is None:
        return

    theta_high_raw = score_thresholds.get("theta_high")
    theta_low_raw = score_thresholds.get("theta_low")
    if not isinstance(theta_high_raw, (int, float)) or not isinstance(
        theta_low_raw, (int, float)
    ):
        return
    theta_high = float(theta_high_raw)
    theta_low = float(theta_low_raw)
    if not math.isfinite(theta_high) or not math.isfinite(theta_low):
        return

    x_test_normal = preview.get("x_test_normal")
    x_test_anomaly = preview.get("x_test_anomaly")
    labels_type = preview.get("labels_type")
    if x_test_normal is None or x_test_anomaly is None or labels_type is None:
        return

    score_high_normal, score_low_normal = _compute_directional_scores(
        model=model,
        scoring_mode=scoring_mode,
        anomaly_manifest=anomaly_manifest,
        thresholds_dir=thresholds_dir,
        x=x_test_normal,
    )
    score_high_anom, score_low_anom = _compute_directional_scores(
        model=model,
        scoring_mode=scoring_mode,
        anomaly_manifest=anomaly_manifest,
        thresholds_dir=thresholds_dir,
        x=x_test_anomaly,
    )

    high_mask = labels_type == int(LABEL_TYPE_HIGH)
    low_mask = labels_type == int(LABEL_TYPE_LOW)

    plots_dir = os.path.join(plots_root, model_name)
    os.makedirs(plots_dir, exist_ok=True)

    _plot_score_histogram(
        path=os.path.join(plots_dir, "score_high_hist.png"),
        title=f"{model_name} Score_high",
        normal=score_high_normal,
        high=score_high_anom[high_mask],
        low=score_high_anom[low_mask],
        threshold=theta_high,
    )
    _plot_score_histogram(
        path=os.path.join(plots_dir, "score_low_hist.png"),
        title=f"{model_name} Score_low",
        normal=score_low_normal,
        high=score_low_anom[high_mask],
        low=score_low_anom[low_mask],
        threshold=theta_low,
    )


def _compute_directional_scores(
    model: torch.nn.Module,
    scoring_mode: Literal["sign_aware", "baseline"],
    anomaly_manifest: Dict[str, Any],
    thresholds_dir: str,
    x: Tensor,
) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        if scoring_mode == "sign_aware":
            tau_plus = _load_torch_tensor_if_present(
                os.path.join(thresholds_dir, "tau_plus.pt")
            )
            tau_minus = _load_torch_tensor_if_present(
                os.path.join(thresholds_dir, "tau_minus.pt")
            )
            if tau_plus is None or tau_minus is None:
                raise ValueError("Missing tau_plus/tau_minus for sign_aware plotting.")
            latent_thresholds = SignAwareLatentThresholds(
                tau_plus=tau_plus, tau_minus=tau_minus
            )
            pi = extract_detection_signal(model, x)
            score_high, score_low = calculate_sign_aware_directional_scores(
                pi=pi, latent_thresholds=latent_thresholds
            )
            return score_high.detach(), score_low.detach()

        evidence_tau = _load_torch_tensor_if_present(
            os.path.join(thresholds_dir, "evidence_tau.pt")
        )
        if evidence_tau is None:
            raise ValueError("Missing evidence_tau for baseline plotting.")

        shift_mode = anomaly_manifest.get("shift_mode")
        if shift_mode not in {"single_channel", "subset"}:
            raise ValueError(f"Unsupported shift_mode: {shift_mode}.")
        subset_indices = anomaly_manifest.get("subset_indices")
        subset_indices_list = (
            None
            if subset_indices is None
            else (
                [int(value) for value in subset_indices]
                if isinstance(subset_indices, list)
                else None
            )
        )
        latent_directions = assign_baseline_latent_directions(
            model=model,
            shift_mode=shift_mode,
            subset_indices=subset_indices_list,
        ).to("cpu")
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=evidence_tau,
            latent_directions=latent_directions,
        )
        evidence = extract_one_sided_evidence(model, x)
        score_high, score_low = calculate_baseline_directional_scores(
            one_sided_evidence=evidence, baseline_thresholds=baseline_thresholds
        )
        return score_high.detach(), score_low.detach()


def _plot_score_histogram(
    path: str,
    title: str,
    normal: Tensor,
    high: Tensor,
    low: Tensor,
    threshold: float,
    max_bins: int = 60,
) -> None:
    normal_flat = normal.detach().flatten().to(device="cpu")
    high_flat = high.detach().flatten().to(device="cpu")
    low_flat = low.detach().flatten().to(device="cpu")

    normal_flat = normal_flat[torch.isfinite(normal_flat)]
    high_flat = high_flat[torch.isfinite(high_flat)]
    low_flat = low_flat[torch.isfinite(low_flat)]

    all_scores = torch.cat([normal_flat, high_flat, low_flat], dim=0)
    max_value = float(all_scores.max().item()) if all_scores.numel() > 0 else 1.0
    if math.isfinite(threshold):
        max_value = max(max_value, threshold)
    max_value = max(max_value, 1e-6)
    bins = np.linspace(0.0, max_value, num=max_bins, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    ax.hist(normal_flat.numpy(), bins=bins, alpha=0.6, label="normal", color="#4c72b0")
    ax.hist(
        high_flat.numpy(), bins=bins, alpha=0.6, label="high anomaly", color="#dd8452"
    )
    ax.hist(
        low_flat.numpy(), bins=bins, alpha=0.6, label="low anomaly", color="#55a868"
    )
    if math.isfinite(threshold):
        ax.axvline(
            threshold, color="black", linestyle="--", linewidth=2, label="threshold"
        )
    ax.set_title(title)
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _extract_models_dict_from_summary(summary: Any) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    models = summary.get("models")
    if isinstance(models, dict):
        return models
    aggregate = summary.get("aggregate_over_seeds")
    if isinstance(aggregate, dict) and isinstance(aggregate.get("models"), dict):
        return aggregate["models"]
    return {}


def _extract_metric_value(metrics: Any, key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if key.endswith(("_mean", "_std", "_n")):
        return None
    mean_value = metrics.get(f"{key}_mean")
    if isinstance(mean_value, (int, float)):
        return float(mean_value)
    return None


def _color_for_index(index: int) -> str:
    palette = [
        "#4c72b0",
        "#dd8452",
        "#55a868",
        "#c44e52",
        "#8172b3",
        "#937860",
        "#da8bc3",
        "#8c8c8c",
    ]
    return palette[index % len(palette)]


def _load_json_if_present(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _load_torch_if_present(path: str) -> Optional[Dict[str, Tensor]]:
    if not os.path.isfile(path):
        return None
    data = torch.load(path, map_location="cpu")
    return data if isinstance(data, dict) else None


def _load_torch_tensor_if_present(path: str) -> Optional[Tensor]:
    if not os.path.isfile(path):
        return None
    tensor = torch.load(path, map_location="cpu")
    return tensor if isinstance(tensor, Tensor) else None


def _load_pickled_model(path: str) -> torch.nn.Module:
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Expected model.pt to contain a torch.nn.Module.")
    return model
