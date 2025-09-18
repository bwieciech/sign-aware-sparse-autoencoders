"""Visualization utilities for the 2-D Protocol A experiments."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:  # pragma: no cover - gracefully degrade in environments without matplotlib
    matplotlib = None
    plt = None
    Line2D = None
    HAS_MATPLOTLIB = False

from sgnsae.data.toy2d import BUNDLE_TYPE_ANTICORRELATED, BUNDLE_TYPE_CORRELATED, Toy2DData
from sgnsae.eval.metrics import canonicalize_2d_vectors
from sgnsae.eval.toy2d_eval import DecoderSignUsage


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / (tensor.norm(dim=1, keepdim=True) + 1e-8)


def _assign_bundle_indices(
    decoder_vectors: torch.Tensor,
    feature_vectors: torch.Tensor,
) -> List[int]:
    assignments: List[int] = []
    for decoder_vec in decoder_vectors:
        angles = torch.acos(
            torch.clamp(
                (decoder_vec.unsqueeze(0) * feature_vectors).sum(dim=1), -1.0, 1.0
            )
        )
        assignments.append(int(torch.argmin(angles)))
    return assignments


def _bundle_colors(data: Toy2DData) -> List[str]:
    correlated_palette = plt.cm.Blues(np.linspace(0.4, 0.9, data.config.num_correlated_bundles))
    anti_palette = plt.cm.Oranges(
        np.linspace(0.4, 0.9, data.config.num_anticorrelated_bundles)
    )
    colors: List[str] = []
    correlated_idx = 0
    anti_idx = 0
    for bundle in data.bundles:
        if bundle.bundle_type == BUNDLE_TYPE_CORRELATED:
            colors.append(matplotlib.colors.rgb2hex(correlated_palette[correlated_idx]))
            correlated_idx += 1
        else:
            colors.append(matplotlib.colors.rgb2hex(anti_palette[anti_idx]))
            anti_idx += 1
    return colors


def _format_label(bundle_type: str, bundle_name: str) -> str:
    prefix = "Corr" if bundle_type == BUNDLE_TYPE_CORRELATED else "Anti"
    return f"{prefix}-{bundle_name.split('_')[-1]}"


def plot_decoder_arrangement(
    decoder_vectors: torch.Tensor,
    data: Toy2DData,
    sign_usage: Sequence[DecoderSignUsage],
    *,
    title: str,
):
    """Plot decoder columns on the unit circle with sign overlays."""

    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is required for plotting but is not installed. Install matplotlib to generate figures."
        )

    decoder_vectors = _normalize_rows(decoder_vectors)
    stacked = torch.cat([decoder_vectors, data.features], dim=0)
    canonical = canonicalize_2d_vectors(stacked)
    decoder_canon = canonical[: decoder_vectors.shape[0]]
    feature_canon = canonical[decoder_vectors.shape[0] :]

    bundle_colors = _bundle_colors(data)
    bundle_labels = {
        bundle.name: _format_label(bundle.bundle_type, bundle.name)
        for bundle in data.bundles
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    unit_circle = plt.Circle((0, 0), 1.0, color="lightgray", fill=False, linewidth=1.5)
    ax.add_patch(unit_circle)

    # Plot ground truth features as hollow markers.
    for bundle, color in zip(data.bundles, bundle_colors):
        coords = feature_canon[list(bundle.indices)].numpy()
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=60,
            facecolors="white",
            edgecolors=color,
            label=f"GT {bundle_labels[bundle.name]}",
            marker="o" if bundle.bundle_type == BUNDLE_TYPE_CORRELATED else "s",
        )

    assignments = _assign_bundle_indices(decoder_canon, _normalize_rows(feature_canon))
    for idx, decoder_vec in enumerate(decoder_canon):
        bundle_idx = data.feature_to_bundle[assignments[idx]]
        bundle = data.bundles[bundle_idx]
        color = bundle_colors[bundle_idx]
        coords = decoder_vec.numpy()
        ax.scatter(
            coords[0],
            coords[1],
            s=100,
            color=color,
            marker="^" if bundle.bundle_type == BUNDLE_TYPE_CORRELATED else "D",
            edgecolors="black",
            linewidths=0.5,
        )

        usage = sign_usage[idx]
        direction = coords / (np.linalg.norm(coords) + 1e-8)
        outer = coords + 0.25 * usage.positive_rate * direction
        inner = coords - 0.25 * usage.negative_rate * direction
        ax.plot([coords[0], outer[0]], [coords[1], outer[1]], color=color, linewidth=1.5)
        ax.plot(
            [coords[0], inner[0]],
            [coords[1], inner[1]],
            color=color,
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_title(title)
    ax.set_xlabel("Canonical dim-1")
    ax.set_ylabel("Canonical dim-2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.grid(alpha=0.2)

    # Deduplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    sign_handles = [
        Line2D([0, 1], [0, 1], color="black", linewidth=1.5, label="Positive usage"),
        Line2D([0, 1], [0, 1], color="black", linewidth=1.5, linestyle="--", label="Negative usage"),
    ]
    for handle in sign_handles:
        unique[handle.get_label()] = handle
    ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1.0))

    return fig


__all__ = ["plot_decoder_arrangement"]
