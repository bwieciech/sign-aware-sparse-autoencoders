"""Evaluation utilities shared across Protocol A experiments."""
from __future__ import annotations

import math

import torch


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute the average mean-squared reconstruction error."""

    if y_true.shape != y_pred.shape:
        raise ValueError("Tensors must have matching shapes")
    return torch.mean((y_true - y_pred) ** 2).item()


def l0_from_latents(latents: torch.Tensor, threshold: float = 1e-6) -> float:
    """Return the average :math:`L_0` (active latent count) per sample."""

    if latents.ndim != 2:
        raise ValueError("Expected a (num_samples, d_hidden) tensor")
    active_counts = (latents.abs() > threshold).float().sum(dim=1)
    return active_counts.mean().item()


def angle_between(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Return the unsigned angle between two vectors in radians."""

    if vec_a.shape != vec_b.shape:
        raise ValueError("Vectors must have matching shapes")
    a_norm = torch.nn.functional.normalize(vec_a, dim=0)
    b_norm = torch.nn.functional.normalize(vec_b, dim=0)
    cosine = torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0)
    return float(torch.arccos(cosine))


def degrees(rad: float) -> float:
    return rad * 180.0 / math.pi


def canonicalize_2d_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Rotate a 2-D point cloud so that the first principal axis aligns with ``x``."""

    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError("Expected an (N, 2) tensor")

    # Perform PCA via SVD to obtain orthonormal axes.
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    u, _, _ = torch.linalg.svd(centered.t() @ centered)
    rotation = u
    rotated = vectors @ rotation
    if rotated.mean(dim=0)[0] < 0:
        rotated[:, 0] *= -1
    if rotated.mean(dim=0)[1] < 0:
        rotated[:, 1] *= -1
    return rotated


# --- Alignment metrics ---------------------------------------------------------


def best_alignment_angles_deg(
    decoder_vectors: torch.Tensor,
    feature_vectors: torch.Tensor,
) -> torch.Tensor:
    """Compute per-feature minimal decoder angles in degrees (sign-invariant).

    Args:
        decoder_vectors: ``(n_latents, 2)`` decoder directions.
        feature_vectors: ``(n_features, 2)`` ground-truth feature directions.

    Returns:
        Tensor of shape ``(n_features,)`` with the best angles in degrees.
    """

    if decoder_vectors.ndim != 2 or decoder_vectors.shape[1] != 2:
        raise ValueError("decoder_vectors must have shape (n_latents, 2)")
    if feature_vectors.ndim != 2 or feature_vectors.shape[1] != 2:
        raise ValueError("feature_vectors must have shape (n_features, 2)")

    dec = torch.nn.functional.normalize(decoder_vectors, dim=1)
    feats = torch.nn.functional.normalize(feature_vectors, dim=1)

    cos = feats @ dec.t()
    abs_cos = cos.abs().clamp(-1.0, 1.0)
    best_abs_cos, _ = abs_cos.max(dim=1)
    angles_rad = torch.arccos(best_abs_cos)
    return torch.rad2deg(angles_rad)


__all__ = [
    "angle_between",
    "canonicalize_2d_vectors",
    "degrees",
    "l0_from_latents",
    "mean_squared_error",
    "best_alignment_angles_deg",
]
