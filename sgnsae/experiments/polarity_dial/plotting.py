import os
from typing import Any, Dict, List

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
    epochs = training_metrics.get("epoch", list(range(1, len(training_metrics.get("loss_total", [])) + 1)))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, training_metrics["loss_total"], label="Total Loss", color="#1f77b4")
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
