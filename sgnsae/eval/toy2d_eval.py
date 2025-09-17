"""Evaluation helpers specific to the Protocol A 2-D synthetic dataset."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from sgnsae.data.toy2d import Toy2DData, Toy2DTensorDataset
from sgnsae.eval.metrics import angle_between, l0_from_latents, mean_squared_error


@dataclass
class DecoderSignUsage:
    positive_rate: float
    negative_rate: float


@dataclass
class Toy2DEvaluation:
    model_name: str
    mse: float
    l0: float
    decoder_angles: torch.Tensor
    collapse_rate: float
    collapsed_pairs: List[Tuple[int, int]]
    sign_usage: List[DecoderSignUsage]


def _collect_latents(
    model: torch.nn.Module,
    dataset: Toy2DTensorDataset,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        observations = dataset.observations.to(device)
        outputs = model(observations)
    # Standardise keys across SAE variants.
    if "x_hat" in outputs:
        predictions = outputs["x_hat"]
    else:
        predictions = outputs["y_pred"]
    latents = outputs["latents"]
    return {
        "predictions": predictions,
        "latents": latents,
    }


def _flatten_instances(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        # (n_instances, batch, dim) -> (batch, dim) assuming identical instances.
        return tensor[0]
    if tensor.ndim == 2:
        return tensor
    raise ValueError("Unexpected tensor shape for flattening")


def _decoder_matrix(model: torch.nn.Module) -> torch.Tensor:
    if not hasattr(model, "W_dec"):
        raise AttributeError("Model does not expose a decoder matrix")
    W_dec: torch.Tensor = model.W_dec.detach().cpu()
    W_dec = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)
    if W_dec.ndim == 3:
        return W_dec[0].t()
    if W_dec.ndim == 2:
        return W_dec.t()
    raise ValueError("Unsupported decoder shape")


def _sign_usage(latents: torch.Tensor) -> List[DecoderSignUsage]:
    sign_usage: List[DecoderSignUsage] = []
    signs = latents
    for latent_idx in range(signs.shape[1]):
        latent_values = signs[:, latent_idx]
        total = max(len(latent_values), 1)
        pos = (latent_values > 0).float().sum().item() / total
        neg = (latent_values < 0).float().sum().item() / total
        sign_usage.append(DecoderSignUsage(positive_rate=pos, negative_rate=neg))
    return sign_usage


def _detect_collapsed_pairs(
    decoder_vectors: torch.Tensor,
    latents: torch.Tensor,
    data: Toy2DData,
    *,
    angle_threshold_degrees: float = 15.0,
    sign_threshold: float = 0.55,
) -> Tuple[int, List[Tuple[int, int]]]:
    angle_threshold = math.radians(angle_threshold_degrees)
    collapsed: List[Tuple[int, int]] = []

    for pair in data.anticorrelated_pairs():
        feature_a = data.features[pair[0]]
        feature_b = data.features[pair[1]]
        a_mask = data.activations[:, pair[0]] > 0
        b_mask = data.activations[:, pair[1]] > 0
        if a_mask.sum() == 0 or b_mask.sum() == 0:
            continue
        for latent_idx, decoder_vec in enumerate(decoder_vectors):
            # Orientation 1: decoder aligns with feature A; negative aligns with feature B.
            angle_a = angle_between(decoder_vec, feature_a)
            angle_b = angle_between(-decoder_vec, feature_b)
            orientation1_ok = angle_a <= angle_threshold and angle_b <= angle_threshold

            # Orientation 2: decoder aligns with feature B; negative aligns with feature A.
            angle_a_flip = angle_between(decoder_vec, feature_b)
            angle_b_flip = angle_between(-decoder_vec, feature_a)
            orientation2_ok = angle_a_flip <= angle_threshold and angle_b_flip <= angle_threshold

            if not (orientation1_ok or orientation2_ok):
                continue

            latent_values = latents[:, latent_idx]
            pos_rate_a = (latent_values[a_mask] > 0).float().mean().item()
            neg_rate_b = (latent_values[b_mask] < 0).float().mean().item()
            orientation1_sign = (
                orientation1_ok
                and pos_rate_a >= sign_threshold
                and neg_rate_b >= sign_threshold
            )

            pos_rate_b = (latent_values[b_mask] > 0).float().mean().item()
            neg_rate_a = (latent_values[a_mask] < 0).float().mean().item()
            orientation2_sign = (
                orientation2_ok
                and pos_rate_b >= sign_threshold
                and neg_rate_a >= sign_threshold
            )

            if orientation1_sign or orientation2_sign:
                collapsed.append(pair)
                break
    return len(collapsed), collapsed


def evaluate_toy2d_model(
    model: torch.nn.Module,
    model_name: str,
    dataset: Toy2DTensorDataset,
    data: Toy2DData,
    *,
    angle_threshold_degrees: float = 15.0,
    sign_threshold: float = 0.55,
) -> Toy2DEvaluation:
    device = next(model.parameters()).device
    outputs = _collect_latents(model, dataset, device)
    predictions = _flatten_instances(outputs["predictions"]).cpu()
    latents = _flatten_instances(outputs["latents"]).cpu()
    targets = dataset.observations

    mse = mean_squared_error(targets, predictions)
    l0 = l0_from_latents(latents)

    decoder_vectors = _decoder_matrix(model)
    _, collapsed_pairs = _detect_collapsed_pairs(
        decoder_vectors,
        latents,
        data,
        angle_threshold_degrees=angle_threshold_degrees,
        sign_threshold=sign_threshold,
    )
    collapse_rate = 0.0
    total_pairs = len(data.anticorrelated_pairs())
    if total_pairs > 0:
        collapse_rate = len(collapsed_pairs) / total_pairs

    angles = torch.zeros(data.num_features, decoder_vectors.shape[0])
    for feature_idx in range(data.num_features):
        feature_vec = data.features[feature_idx]
        for latent_idx, decoder_vec in enumerate(decoder_vectors):
            angles[feature_idx, latent_idx] = angle_between(decoder_vec, feature_vec)

    return Toy2DEvaluation(
        model_name=model_name,
        mse=mse,
        l0=l0,
        decoder_angles=angles,
        collapse_rate=collapse_rate,
        collapsed_pairs=collapsed_pairs,
        sign_usage=_sign_usage(latents),
    )


__all__ = [
    "DecoderSignUsage",
    "Toy2DEvaluation",
    "evaluate_toy2d_model",
]
