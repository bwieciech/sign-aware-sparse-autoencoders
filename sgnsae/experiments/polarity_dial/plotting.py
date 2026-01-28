import os
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_plots_dir(model_dir: str) -> str:
    plots_dir = os.path.join(model_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_calibration_scatter(
    scatter_data: Dict[str, List[float]],
    model_name: str,
    model_dir: str,
) -> None:
    plots_dir = _ensure_plots_dir(model_dir)
    plt.figure(figsize=(6, 6))
    plt.scatter(
        scatter_data["positive_pred"],
        scatter_data["positive_true"],
        color="#1f77b4",
        alpha=0.4,
        marker="o",
        label="c_j > 0",
    )
    plt.scatter(
        scatter_data["negative_pred"],
        scatter_data["negative_true"],
        color="#d62728",
        alpha=0.4,
        marker="x",
        label="c_j < 0",
    )
    all_points = (
        scatter_data["positive_pred"]
        + scatter_data["positive_true"]
        + scatter_data["negative_pred"]
        + scatter_data["negative_true"]
    )
    max_abs = max(1.0, max(map(abs, all_points)) if all_points else 1.0)
    plt.plot(
        [-max_abs, max_abs],
        [-max_abs, max_abs],
        color="#555555",
        linestyle="-",
        linewidth=1.0,
        label="y = x",
    )
    plt.xlabel("Predicted coefficient (hat_c)")
    plt.ylabel("Ground-truth coefficient (c)")
    plt.title(f"Calibration Scatter - {model_name}")
    plt.legend()
    plt.grid(axis="y", color="#e0e0e0")
    scatter_path = os.path.join(plots_dir, f"calibration_scatter_{model_name}.png")
    plt.savefig(scatter_path, bbox_inches="tight")
    plt.close()


def plot_gamma_histograms(
    calibration: Dict[str, Any],
    model_name: str,
    model_dir: str,
) -> None:
    plots_dir = _ensure_plots_dir(model_dir)
    gamma_plus = [
        value["gamma_plus"]
        for value in calibration["per_pair"]
        if value["gamma_plus"] is not None
    ]
    gamma_minus = [
        value["gamma_minus"]
        for value in calibration["per_pair"]
        if value["gamma_minus"] is not None
    ]
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.hist(gamma_plus, color="#1f77b4", alpha=0.7, bins=20)
    plt.axvline(1.0, color="#555555", linestyle="--")
    plt.title(f"Gamma+ Distribution - {model_name}")
    plt.grid(axis="y", color="#e0e0e0")
    plt.subplot(2, 1, 2)
    plt.hist(gamma_minus, color="#d62728", alpha=0.7, bins=20)
    plt.axvline(1.0, color="#555555", linestyle="--")
    plt.title(f"Gamma- Distribution - {model_name}")
    plt.xlabel("Gamma value")
    plt.grid(axis="y", color="#e0e0e0")
    hist_path = os.path.join(plots_dir, f"gamma_hist_{model_name}.png")
    plt.tight_layout()
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close()


def plot_gamma_strip(
    calibration: Dict[str, Any],
    model_name: str,
    model_dir: str,
) -> None:
    plots_dir = _ensure_plots_dir(model_dir)
    gamma_plus = []
    gamma_minus = []
    x_positions = []
    for idx, entry in enumerate(calibration["per_pair"]):
        gp = entry["gamma_plus"]
        gm = entry["gamma_minus"]
        if gp is None and gm is None:
            continue
        x_positions.append(idx)
        gamma_plus.append(np.nan if gp is None else gp)
        gamma_minus.append(np.nan if gm is None else gm)
    plt.figure(figsize=(8, 4))
    plt.plot(
        x_positions,
        gamma_plus,
        "o",
        color="#1f77b4",
        label="gamma_plus",
    )
    plt.plot(
        x_positions,
        gamma_minus,
        "o",
        color="#d62728",
        label="gamma_minus",
    )
    for idx, (gp, gm) in enumerate(zip(gamma_plus, gamma_minus)):
        if np.isnan(gp) or np.isnan(gm):
            continue
        plt.plot(
            [x_positions[idx], x_positions[idx]],
            [gp, gm],
            color="#999999",
            alpha=0.6,
        )
    plt.axhline(1.0, color="#555555", linestyle="--")
    plt.xlabel("Matched pair index")
    plt.ylabel("Gamma value")
    plt.title(f"Gamma+/- per latent - {model_name}")
    plt.legend()
    plt.grid(axis="y", color="#e0e0e0")
    strip_path = os.path.join(plots_dir, f"gamma_plus_minus_strip_{model_name}.png")
    plt.savefig(strip_path, bbox_inches="tight")
    plt.close()


def plot_training_loss(
    training_metrics: Dict[str, Any],
    model_name: str,
    model_dir: str,
) -> None:
    plots_dir = _ensure_plots_dir(model_dir)
    epochs = training_metrics.get(
        "epoch", list(range(1, len(training_metrics.get("loss_total", [])) + 1))
    )
    plt.figure(figsize=(8, 4))
    plt.plot(
        epochs, training_metrics["loss_total"], label="Total Loss", color="#1f77b4"
    )
    plt.plot(
        epochs,
        training_metrics["loss_reconstruction"],
        label="Reconstruction Loss",
        color="#ff7f0e",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {model_name}")
    plt.legend()
    plt.grid(axis="y", color="#e0e0e0")
    training_path = os.path.join(plots_dir, f"training_loss_{model_name}.png")
    plt.savefig(training_path, bbox_inches="tight")
    plt.close()


def plot_gamma_comparison(
    calibration_summaries: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    model_names = list(calibration_summaries.keys())
    indices = np.arange(len(model_names))
    width = 0.35

    rel_error_plus = []
    rel_error_minus = []
    std_plus = []
    std_minus = []
    for name in model_names:
        summary = calibration_summaries[name]
        rel_error_plus.append(summary.get("mean_rel_error_plus") or 0.0)
        rel_error_minus.append(summary.get("mean_rel_error_minus") or 0.0)
        std_plus.append(summary.get("std_rel_error_plus") or 0.0)
        std_minus.append(summary.get("std_rel_error_minus") or 0.0)

    plt.figure(figsize=(10, 5))
    plt.bar(
        indices - width / 2,
        rel_error_plus,
        width,
        label="rel_error_plus",
        color="#1f77b4",
        yerr=std_plus,
        capsize=4,
    )
    plt.bar(
        indices + width / 2,
        rel_error_minus,
        width,
        label="rel_error_minus",
        color="#d62728",
        yerr=std_minus,
        capsize=4,
    )
    plt.xticks(indices, model_names, rotation=15)
    plt.ylabel("Mean relative error")
    plt.title("Gamma Calibration Error Comparison")
    plt.legend()
    plt.grid(axis="y", color="#e0e0e0")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_mse_vs_l0_tradeoff(tradeoff_summary: Dict[str, Any], output_path: str) -> None:
    model_points = tradeoff_summary.get("models") or {}
    plt.figure(figsize=(8, 5))
    for model_name, points in model_points.items():
        xs: List[float] = []
        ys: List[float] = []
        annotations: List[str] = []
        for point in points:
            test_l0 = point.get("test_l0")
            test_mse = point.get("test_reconstruction_mse")
            if test_l0 is None or test_mse is None:
                continue
            xs.append(float(test_l0))
            ys.append(float(test_mse))
            sparsity_coeff = point.get("sparsity_coeff")
            annotations.append(
                "" if sparsity_coeff is None else f"{float(sparsity_coeff):.3g}"
            )
        if not xs:
            continue
        ordering = np.argsort(xs)
        xs_sorted = [xs[idx] for idx in ordering]
        ys_sorted = [ys[idx] for idx in ordering]
        annotations_sorted = [annotations[idx] for idx in ordering]
        plt.plot(xs_sorted, ys_sorted, marker="o", label=model_name)
        for x_val, y_val, label in zip(xs_sorted, ys_sorted, annotations_sorted):
            if not label:
                continue
            plt.annotate(
                label,
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#444444",
            )
    plt.xlabel("Test L0 (active latents per example)")
    plt.ylabel("Test reconstruction MSE")
    plt.title("Sparsity-Fidelity Trade-off (MSE vs L0)")
    plt.legend()
    plt.grid(axis="y", color="#e0e0e0")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_asymmetry_sweep_calibration(
    asymmetry_summary: Dict[str, Any], output_path: str
) -> None:
    p_plus_grid = [
        float(value) for value in (asymmetry_summary.get("p_plus_grid") or [])
    ]
    if not p_plus_grid:
        raise ValueError("asymmetry_summary must include a non-empty p_plus_grid.")
    models = list(asymmetry_summary.get("models") or [])
    points = list(asymmetry_summary.get("points") or [])
    if not models:
        if points:
            models = list((points[0].get("models") or {}).keys())
    if not models:
        raise ValueError("asymmetry_summary must include model names.")

    tail_grid_key = None
    tail_key = None
    if "lognormal_sigma_grid" in asymmetry_summary:
        tail_grid_key = "lognormal_sigma_grid"
        tail_key = "lognormal_sigma"
    if "exponential_lambda_grid" in asymmetry_summary:
        tail_grid_key = "exponential_lambda_grid"
        tail_key = "exponential_lambda"
    if tail_grid_key is None or tail_key is None:
        raise ValueError(
            "asymmetry_summary must include either lognormal_sigma_grid or exponential_lambda_grid."
        )
    tail_values = [float(value) for value in asymmetry_summary.get(tail_grid_key) or []]
    if not tail_values:
        raise ValueError(f"{tail_grid_key} must be non-empty for asymmetry sweep.")

    point_map: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for point in points:
        p_plus = float(point.get("positive_sign_prob"))
        tail_value = float(point.get(tail_key))
        point_map[(p_plus, tail_value)] = point.get("models") or {}

    n_cols = len(tail_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(max(6, 4 * n_cols), 6), sharex=True)
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col_idx, tail_value in enumerate(tail_values):
        ax_plus = axes[0, col_idx]
        ax_minus = axes[1, col_idx]
        for model_name in models:
            plus_values: List[float] = []
            minus_values: List[float] = []
            for p_plus in p_plus_grid:
                model_metrics = point_map.get((p_plus, tail_value), {}).get(
                    model_name, {}
                )
                plus = model_metrics.get("mean_rel_error_plus")
                minus = model_metrics.get("mean_rel_error_minus")
                plus_values.append(np.nan if plus is None else float(plus))
                minus_values.append(np.nan if minus is None else float(minus))
            ax_plus.plot(p_plus_grid, plus_values, marker="o", label=model_name)
            ax_minus.plot(p_plus_grid, minus_values, marker="o", label=model_name)

        ax_plus.set_title(f"{tail_key}={tail_value:.3g}")
        ax_plus.set_ylabel("Mean |gamma_+ - 1|")
        ax_minus.set_ylabel("Mean |gamma_- - 1|")
        ax_minus.set_xlabel("p_plus")
        ax_plus.grid(axis="y", color="#e0e0e0")
        ax_minus.grid(axis="y", color="#e0e0e0")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(models), 4))
    fig.suptitle("Asymmetry Sweep Calibration")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _ensure_multi_sparsity_plots_dir(seed_run_dir: str, model_name: str) -> str:
    plots_dir = os.path.join(seed_run_dir, "plots", "multi_sparsity", model_name)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _final_epoch_values_by_instance(
    metric_by_instance: List[List[float]],
) -> List[float]:
    final_values: List[float] = []
    for instance_values in metric_by_instance:
        final_values.append(
            float(np.nan) if not instance_values else float(instance_values[-1])
        )
    return final_values


def _sort_xy(
    x_values: List[float], y_values: List[float]
) -> Tuple[List[float], List[float]]:
    ordering = np.argsort(np.array(x_values, dtype=np.float64))
    x_sorted = [float(x_values[idx]) for idx in ordering]
    y_sorted = [float(y_values[idx]) for idx in ordering]
    return x_sorted, y_sorted


def _plot_metric_vs_sparsity(
    x_values: List[float],
    y_values: List[float],
    x_label: str,
    y_label: str,
    title: str,
    output_path: str,
    color: str,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(x_values, y_values, marker="o", color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(axis="y", color="#e0e0e0")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _final_metric_values_for_sparsity_coeff_sweep(
    training_metrics: Dict[str, Any],
    scalar_metric_key: str,
    metric_by_instance_key: str,
    n_instances: int,
) -> List[float]:
    if metric_by_instance_key in training_metrics:
        metric_by_instance = training_metrics[metric_by_instance_key]
        if not isinstance(metric_by_instance, list):
            raise TypeError(
                f"{metric_by_instance_key} must be a list; received {type(metric_by_instance)}."
            )
        return _final_epoch_values_by_instance(metric_by_instance)

    scalar_values = training_metrics.get(scalar_metric_key) or []
    scalar_final = float(np.nan) if not scalar_values else float(scalar_values[-1])
    return [scalar_final for _ in range(n_instances)]


def plot_multi_sparsity_loss_summaries(
    training_metrics: Dict[str, Any],
    model_name: str,
    seed_run_dir: str,
    sparsity_coeff_list: List[float],
) -> None:
    if not sparsity_coeff_list:
        raise ValueError("sparsity_coeff_list must be non-empty.")
    plots_dir = _ensure_multi_sparsity_plots_dir(seed_run_dir, model_name)

    epsilon = 1e-12
    n_instances = len(sparsity_coeff_list)

    final_reconstruction = _final_metric_values_for_sparsity_coeff_sweep(
        training_metrics=training_metrics,
        scalar_metric_key="loss_reconstruction",
        metric_by_instance_key="loss_reconstruction_by_instance",
        n_instances=n_instances,
    )
    final_sparsity = _final_metric_values_for_sparsity_coeff_sweep(
        training_metrics=training_metrics,
        scalar_metric_key="loss_sparsity",
        metric_by_instance_key="loss_sparsity_by_instance",
        n_instances=n_instances,
    )
    final_mean_active_latents = _final_metric_values_for_sparsity_coeff_sweep(
        training_metrics=training_metrics,
        scalar_metric_key="mean_active_latents",
        metric_by_instance_key="mean_active_latents_by_instance",
        n_instances=n_instances,
    )
    final_aux_loss: List[float] | None = None
    if "loss_aux_by_instance" in training_metrics or "loss_aux" in training_metrics:
        final_aux_loss = _final_metric_values_for_sparsity_coeff_sweep(
            training_metrics=training_metrics,
            scalar_metric_key="loss_aux",
            metric_by_instance_key="loss_aux_by_instance",
            n_instances=n_instances,
        )

    series = [
        (
            "reconstruction_loss",
            "Reconstruction loss",
            final_reconstruction,
            "#ff7f0e",
        ),
        ("sparsity_loss", "Sparsity loss", final_sparsity, "#2ca02c"),
        (
            "mean_active_latents",
            "Mean active latents",
            final_mean_active_latents,
            "#1f77b4",
        ),
    ]
    if final_aux_loss is not None:
        series.append(("aux_loss", "Auxiliary loss", final_aux_loss, "#9467bd"))

    for prefix, metric_label, y_values, color in series:
        if len(y_values) != len(sparsity_coeff_list):
            raise ValueError(
                "Final metric list length must match sparsity_coeff_list length."
            )
        coeffs_sorted, y_sorted = _sort_xy(
            [float(value) for value in sparsity_coeff_list],
            [float(value) for value in y_values],
        )
        log_coeffs_sorted = [
            float(np.log10(max(float(value), epsilon))) for value in coeffs_sorted
        ]

        _plot_metric_vs_sparsity(
            coeffs_sorted,
            y_sorted,
            x_label="sparsity_coeff",
            y_label=metric_label,
            title=f"{metric_label} vs sparsity_coeff - {model_name}",
            output_path=os.path.join(plots_dir, f"{prefix}_vs_sparsity_coeff.png"),
            color=color,
        )
        _plot_metric_vs_sparsity(
            log_coeffs_sorted,
            y_sorted,
            x_label="log10(sparsity_coeff)",
            y_label=metric_label,
            title=f"{metric_label} vs log10(sparsity_coeff) - {model_name}",
            output_path=os.path.join(plots_dir, f"{prefix}_vs_log_sparsity_coeff.png"),
            color=color,
        )
