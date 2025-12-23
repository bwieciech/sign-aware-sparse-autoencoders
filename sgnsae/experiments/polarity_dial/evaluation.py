import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def _get_decoder_directions(model: torch.nn.Module) -> Tensor:
    if not hasattr(model, "W_dec"):
        raise ValueError("Model does not expose a decoder matrix (W_dec).")
    decoder = model.W_dec.detach()
    decoder = decoder / (decoder.norm(dim=1, keepdim=True) + 1e-8)
    decoder = decoder[0].transpose(0, 1)
    decoder = decoder / (decoder.norm(dim=1, keepdim=True) + 1e-8)
    return decoder


def align_decoder_and_features(
    model: torch.nn.Module, feature_directions: Tensor
) -> Tuple[Dict[str, Any], Tensor]:
    decoder_directions = _get_decoder_directions(model)
    cosine_matrix = decoder_directions.cpu() @ feature_directions.cpu().transpose(0, 1)
    row_indices, col_indices = linear_sum_assignment(-cosine_matrix.numpy())

    matched_pairs = []
    for latent_idx, feature_idx in zip(row_indices, col_indices):
        matched_pairs.append(
            {
                "latent_index": int(latent_idx),
                "feature_index": int(feature_idx),
                "cosine_similarity": float(cosine_matrix[latent_idx, feature_idx]),
            }
        )
    unmatched_latents = [
        idx for idx in range(decoder_directions.shape[0]) if idx not in set(row_indices)
    ]
    alignment = {
        "matched_pairs": matched_pairs,
        "unmatched_latents": unmatched_latents,
    }
    return alignment, decoder_directions


def _compute_gamma(
    true_values: Tensor,
    predicted_values: Tensor,
    eps: float,
    min_count: int,
) -> Optional[float]:
    if true_values.numel() < min_count:
        return None
    denominator = predicted_values.pow(2).sum() + eps
    if denominator.item() == 0:
        return None
    gamma_value = (true_values * predicted_values).sum() / denominator
    return float(gamma_value.item())


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _safe_std(values: List[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if len(filtered) < 2:
        return None
    mean_value = sum(filtered) / len(filtered)
    variance = sum((value - mean_value) ** 2 for value in filtered) / (len(filtered) - 1)
    return float(variance**0.5)


def _sample_points(
    true_values: Tensor,
    predicted_values: Tensor,
    remaining_budget: int,
) -> Tuple[List[float], List[float]]:
    if true_values.numel() == 0 or remaining_budget <= 0:
        return [], []
    if true_values.numel() > remaining_budget:
        indices = torch.randperm(true_values.numel(), device=true_values.device)[
            :remaining_budget
        ]
        true_values = true_values[indices]
        predicted_values = predicted_values[indices]
    return true_values.cpu().tolist(), predicted_values.cpu().tolist()


def _calculate_per_pair_metrics(
    true_coefficients: Tensor,
    predicted_coefficients: Tensor,
    projection_factor: float,
    latent_index: int,
    feature_index: int,
    eps: float,
    min_count: int,
) -> Dict[str, Any]:
    predicted_scaled = predicted_coefficients * projection_factor
    positive_mask = true_coefficients > 0
    negative_mask = true_coefficients < 0
    gamma_plus = _compute_gamma(
        true_coefficients[positive_mask],
        predicted_scaled[positive_mask],
        eps,
        min_count,
    )
    gamma_minus = _compute_gamma(
        true_coefficients[negative_mask],
        predicted_scaled[negative_mask],
        eps,
        min_count,
    )
    rel_error_plus = None if gamma_plus is None else abs(gamma_plus - 1.0)
    rel_error_minus = None if gamma_minus is None else abs(gamma_minus - 1.0)
    return {
        "latent_index": latent_index,
        "feature_index": feature_index,
        "gamma_plus": gamma_plus,
        "gamma_minus": gamma_minus,
        "rel_error_plus": rel_error_plus,
        "rel_error_minus": rel_error_minus,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "predicted_scaled": predicted_scaled,
    }


def compute_alignment_and_calibration(
    model: torch.nn.Module,
    x_test: Tensor,
    coefficients_test: Tensor,
    feature_directions: Tensor,
    alignment: Dict[str, Any],
    decoder_directions: Tensor,
    max_scatter_points_per_regime: int,
    min_gamma_count: int = 5,
    eps: float = 1e-8,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    model.eval()
    with torch.no_grad():
        output = model(x_test)
    latents = output["latents"][0]

    scatter_positive_true: List[float] = []
    scatter_positive_pred: List[float] = []
    scatter_negative_true: List[float] = []
    scatter_negative_pred: List[float] = []

    per_pair_entries: List[Dict[str, Any]] = []
    for pair in alignment["matched_pairs"]:
        latent_index = pair["latent_index"]
        feature_index = pair["feature_index"]
        projection_factor = float(
            torch.dot(
                decoder_directions[latent_index],
                feature_directions[feature_index],
            ).item()
        )
        metrics = _calculate_per_pair_metrics(
            coefficients_test[:, feature_index],
            latents[:, latent_index],
            projection_factor,
            latent_index,
            feature_index,
            eps,
            min_gamma_count,
        )
        per_pair_entries.append(
            {
                "latent_index": latent_index,
                "feature_index": feature_index,
                "gamma_plus": metrics["gamma_plus"],
                "gamma_minus": metrics["gamma_minus"],
                "rel_error_plus": metrics["rel_error_plus"],
                "rel_error_minus": metrics["rel_error_minus"],
            }
        )

        remaining_positive_budget = max_scatter_points_per_regime - len(
            scatter_positive_true
        )
        remaining_negative_budget = max_scatter_points_per_regime - len(
            scatter_negative_true
        )
        true_pos, pred_pos = _sample_points(
            coefficients_test[:, feature_index][metrics["positive_mask"]],
            metrics["predicted_scaled"][metrics["positive_mask"]],
            remaining_positive_budget,
        )
        true_neg, pred_neg = _sample_points(
            coefficients_test[:, feature_index][metrics["negative_mask"]],
            metrics["predicted_scaled"][metrics["negative_mask"]],
            remaining_negative_budget,
        )
        scatter_positive_true.extend(true_pos)
        scatter_positive_pred.extend(pred_pos)
        scatter_negative_true.extend(true_neg)
        scatter_negative_pred.extend(pred_neg)

    gamma_plus_values = [entry["gamma_plus"] for entry in per_pair_entries]
    gamma_minus_values = [entry["gamma_minus"] for entry in per_pair_entries]
    rel_error_plus_values = [entry["rel_error_plus"] for entry in per_pair_entries]
    rel_error_minus_values = [entry["rel_error_minus"] for entry in per_pair_entries]

    calibration_summary = {
        "mean_gamma_plus": _safe_mean(gamma_plus_values),
        "mean_gamma_minus": _safe_mean(gamma_minus_values),
        "mean_rel_error_plus": _safe_mean(rel_error_plus_values),
        "mean_rel_error_minus": _safe_mean(rel_error_minus_values),
        "std_rel_error_plus": _safe_std(rel_error_plus_values),
        "std_rel_error_minus": _safe_std(rel_error_minus_values),
        "num_pairs": len(per_pair_entries),
    }

    calibration = {
        "per_pair": per_pair_entries,
        "summary": calibration_summary,
    }
    scatter_data = {
        "positive_true": scatter_positive_true,
        "positive_pred": scatter_positive_pred,
        "negative_true": scatter_negative_true,
        "negative_pred": scatter_negative_pred,
    }
    return calibration, scatter_data


def persist_alignment_and_calibration(
    model_dir: str,
    alignment: Dict[str, Any],
    calibration: Dict[str, Any],
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    alignment_path = os.path.join(model_dir, "alignment.json")
    with open(alignment_path, "w") as f:
        json.dump(alignment, f, indent=2)
    calibration_path = os.path.join(model_dir, "calibration.json")
    with open(calibration_path, "w") as f:
        json.dump(calibration, f, indent=2)
