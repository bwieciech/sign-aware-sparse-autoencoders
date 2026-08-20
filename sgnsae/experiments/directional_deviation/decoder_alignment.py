import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


_EPSILON = 1e-12


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}.")


def _ensure_single_instance(model: torch.nn.Module) -> None:
    if not hasattr(model, "n_instances"):
        raise ValueError("Model is missing required attribute n_instances.")
    if int(getattr(model, "n_instances")) != 1:
        raise ValueError(
            f"Expected model.n_instances == 1, got {getattr(model, 'n_instances')}."
        )


def _extract_decoder_matrix(model: torch.nn.Module) -> Tensor:
    if not hasattr(model, "W_dec"):
        raise ValueError("Model is missing required attribute W_dec.")
    decoder = getattr(model, "W_dec")
    if not isinstance(decoder, Tensor):
        raise ValueError("Model attribute W_dec must be a torch.Tensor.")
    if decoder.ndim != 3:
        raise ValueError(f"Expected W_dec.ndim == 3, got {decoder.ndim}.")
    _ensure_single_instance(model)
    return decoder[0].detach()


def calculate_latent_argmax_channel_and_purity(
    model: torch.nn.Module,
) -> Tuple[Tensor, Tensor]:
    decoder = _extract_decoder_matrix(model)
    abs_weights = decoder.abs()
    latent_argmax_channel = abs_weights.argmax(dim=0).to(dtype=torch.int64)
    max_abs = abs_weights.max(dim=0).values
    norms = decoder.norm(dim=0)
    purity = torch.where(norms > _EPSILON, max_abs / norms, torch.zeros_like(norms))
    return latent_argmax_channel, purity


def _calculate_purity_summary(purity: np.ndarray) -> Dict[str, float]:
    if purity.ndim != 1:
        raise ValueError(f"Expected purity.ndim == 1, got {purity.ndim}.")
    if purity.size == 0:
        raise ValueError("purity must be non-empty.")

    purity = np.clip(purity, 0.0, 1.0)
    percentiles = np.percentile(purity, [10, 50, 90])
    return {
        "mean": float(np.mean(purity)),
        "median": float(np.median(purity)),
        "p10": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p90": float(percentiles[2]),
    }


def build_decoder_alignment_diagnostics(
    model: torch.nn.Module,
    num_bins: int = 20,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _validate_positive_int(num_bins, "num_bins")

    latent_argmax_channel, latent_purity = calculate_latent_argmax_channel_and_purity(
        model
    )
    latent_purity_cpu = latent_purity.to(device="cpu", dtype=torch.float64).numpy()
    latent_purity_cpu = np.clip(latent_purity_cpu, 0.0, 1.0)

    summary = _calculate_purity_summary(latent_purity_cpu)
    unique_channels_claimed = int(torch.unique(latent_argmax_channel).numel())

    counts, bin_edges = np.histogram(latent_purity_cpu, bins=num_bins, range=(0.0, 1.0))
    histogram = {
        "num_bins": int(num_bins),
        "bin_edges": [float(value) for value in bin_edges.tolist()],
        "counts": [int(value) for value in counts.tolist()],
    }

    decoder = _extract_decoder_matrix(model)
    diagnostics = {
        "num_channels": int(decoder.shape[0]),
        "d_hidden": int(decoder.shape[1]),
        "latent_argmax_channel": [
            int(value) for value in latent_argmax_channel.to(device="cpu").tolist()
        ],
        "latent_purity": [
            float(value) for value in latent_purity.to(device="cpu").tolist()
        ],
        "unique_channels_claimed": unique_channels_claimed,
        "purity_summary": summary,
        "purity_histogram": histogram,
    }

    summary_output = {
        "unique_channels_claimed": unique_channels_claimed,
        "purity_mean": summary["mean"],
        "purity_median": summary["median"],
        "purity_p10": summary["p10"],
        "purity_p50": summary["p50"],
        "purity_p90": summary["p90"],
    }
    return diagnostics, summary_output


def persist_decoder_alignment_diagnostics(
    model: torch.nn.Module,
    model_dir: str,
    num_bins: int = 20,
) -> Optional[Dict[str, Any]]:
    os.makedirs(model_dir, exist_ok=True)
    diagnostics_path = os.path.join(model_dir, "decoder_alignment.json")
    try:
        diagnostics, summary = build_decoder_alignment_diagnostics(
            model=model, num_bins=num_bins
        )
    except Exception as exc:
        error_payload = {"error": f"{type(exc).__name__}: {exc}"}
        with open(diagnostics_path, "w") as f:
            json.dump(error_payload, f, indent=2)
        return None

    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    return summary
