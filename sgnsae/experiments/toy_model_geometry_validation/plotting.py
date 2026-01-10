import os
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from sgnsae.experiments.toy_model_geometry_validation.evaluation import (
    extract_decoder_directions,
)


def _import_matplotlib() -> Any:
    try:
        import matplotlib

        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to plot Toy Model Geometry Validation artifacts."
        ) from exc
    return plt


def plot_unit_circle(
    model: torch.nn.Module,
    model_name: str,
    feature_directions: Tensor,
    model_dir: str,
    *,
    x_test: Optional[Tensor] = None,
    min_active_fraction: Optional[float] = 1e-3,
) -> str:
    plt = _import_matplotlib()

    decoder_directions = extract_decoder_directions(model)
    decoder_label = "Decoder"
    if x_test is not None and min_active_fraction is not None:
        model.eval()
        with torch.no_grad():
            output = model(x_test.to(device=next(model.parameters()).device))
        if "latents" not in output:
            raise ValueError("Model output is missing required key 'latents'.")
        latents_tensor = output["latents"]
        if latents_tensor.ndim != 3 or int(latents_tensor.shape[0]) != 1:
            raise ValueError(
                "Model output 'latents' must have shape (1, batch, d_hidden); "
                f"received {tuple(int(dim) for dim in latents_tensor.shape)}."
            )
        latents = latents_tensor[0].detach().cpu().to(dtype=torch.float32)
        active_fraction = (latents.abs() > 1e-8).float().mean(dim=0)
        active_mask = active_fraction >= float(min_active_fraction)
        if bool(active_mask.any().item()):
            decoder_directions = decoder_directions[active_mask]
            decoder_label = f"Decoder (active ≥ {float(min_active_fraction):g})"
    feature_unit = feature_directions.detach().cpu().to(dtype=torch.float32)
    feature_unit = feature_unit / (feature_unit.norm(dim=1, keepdim=True) + 1e-8)

    plots_dir = os.path.join(model_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    theta = torch.linspace(0.0, 2.0 * torch.pi, 512)
    circle_x = torch.cos(theta)
    circle_y = torch.sin(theta)

    plt.figure(figsize=(6, 6))
    plt.plot(
        circle_x.numpy(),
        circle_y.numpy(),
        color="#222222",
        linewidth=1.0,
        alpha=0.6,
    )
    plt.scatter(
        feature_unit[:, 0].numpy(),
        feature_unit[:, 1].numpy(),
        c="#1f77b4",
        s=28,
        alpha=0.8,
        label="Ground truth",
    )
    plt.scatter(
        decoder_directions[:, 0].numpy(),
        decoder_directions[:, 1].numpy(),
        c="#ff7f0e",
        s=28,
        alpha=0.9,
        label=decoder_label,
    )
    plt.title(f"Unit Circle Decoder Geometry - {model_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(axis="both", color="#e0e0e0", linewidth=0.8)
    plt.legend(loc="upper right")

    output_path = os.path.join(plots_dir, f"unit_circle_{model_name}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def plot_bijump_histogram(
    model: torch.nn.Module,
    model_name: str,
    x_test: Tensor,
    latent_index: int,
    hist_num_bins: int,
    model_dir: str,
    device: str,
    axis_abs_cos: Optional[float] = None,
) -> str:
    plt = _import_matplotlib()

    model.eval()
    with torch.no_grad():
        output = model(x_test.to(device=device))
    if "latents" not in output:
        raise ValueError("Model output is missing required key 'latents'.")
    latents_tensor = output["latents"]
    if latents_tensor.ndim != 3 or int(latents_tensor.shape[0]) != 1:
        raise ValueError(
            "Model output 'latents' must have shape (1, batch, d_hidden); "
            f"received {tuple(int(dim) for dim in latents_tensor.shape)}."
        )
    latents = latents_tensor[0].detach().cpu().to(dtype=torch.float32)
    if latent_index < 0 or latent_index >= int(latents.shape[1]):
        raise ValueError(
            "latent_index must be within [0, d_hidden); "
            f"received latent_index={latent_index}."
        )
    values = latents[:, latent_index]
    max_abs = float(values.abs().max().item())
    if max_abs == 0.0:
        max_abs = 1.0
    max_abs *= 1.05

    plots_dir = os.path.join(model_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    title = f"Bi-Jump Histogram - {model_name} latent {latent_index}"
    if axis_abs_cos is not None:
        title = f"{title} (axis abs cos {axis_abs_cos:.3f})"

    plt.figure(figsize=(8, 4))
    plt.hist(
        values.numpy(),
        bins=int(hist_num_bins),
        range=(-max_abs, max_abs),
        color="#1f77b4",
        alpha=0.85,
    )
    plt.axvline(0.0, color="#222222", linewidth=1.0, alpha=0.7)
    plt.title(title)
    plt.xlabel("Signed latent activation")
    plt.ylabel("Count")
    plt.grid(axis="y", color="#e0e0e0", linewidth=0.8)

    output_path = os.path.join(
        plots_dir, f"bi_jump_hist_{model_name}_latent{latent_index}.png"
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def plot_pair_consolidation_sweep(
    pair_correlation_grid: list[float],
    series_by_model: dict[str, list[Optional[float]]],
    output_path: str,
) -> str:
    plt = _import_matplotlib()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    x_values = [float(value) for value in pair_correlation_grid]
    plt.figure(figsize=(8, 4))
    for model_name, series in series_by_model.items():
        y_values = [float("nan") if value is None else float(value) for value in series]
        plt.plot(x_values, y_values, marker="o", label=model_name)

    plt.title("Pair Consolidation Sweep")
    plt.xlabel("Within-pair correlation")
    plt.ylabel("Consolidation rate (over evaluable pairs)")
    plt.grid(axis="both", color="#e0e0e0", linewidth=0.8)
    plt.legend(loc="best")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path
