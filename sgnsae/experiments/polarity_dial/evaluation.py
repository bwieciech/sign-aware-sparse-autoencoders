import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from sgnsae.experiments.polarity_dial.config import PolarityDialEvaluationConfig


def _get_decoder_directions(model: torch.nn.Module, normalize: bool) -> Tensor:
    if not hasattr(model, "W_dec"):
        raise ValueError("Model does not expose a decoder matrix (W_dec).")
    decoder = model.W_dec.detach()
    if normalize:
        decoder = decoder / (decoder.norm(dim=1, keepdim=True) + 1e-8)
    decoder = decoder[0].transpose(0, 1).contiguous()
    return decoder


def _safe_unit_norm(tensor: Tensor, eps: float) -> Tensor:
    return tensor / (tensor.norm(dim=-1, keepdim=True) + eps)


def align_decoder_and_features(
    model: torch.nn.Module,
    feature_directions: Tensor,
    evaluation_config: PolarityDialEvaluationConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], Tensor]:
    decoder_directions = _get_decoder_directions(
        model, normalize=evaluation_config.normalize_decoder_for_eval
    )
    decoder_unit = _safe_unit_norm(decoder_directions, eps=1e-8)
    feature_unit = _safe_unit_norm(feature_directions, eps=1e-8)

    signed_cosine_matrix = decoder_unit.cpu() @ feature_unit.cpu().transpose(0, 1)
    axis_cosine_matrix = signed_cosine_matrix.abs()
    row_indices, col_indices = linear_sum_assignment((-axis_cosine_matrix).numpy())

    matched_pairs = []
    for latent_idx, feature_idx in zip(row_indices, col_indices):
        signed_cos = float(signed_cosine_matrix[latent_idx, feature_idx])
        axis_cos = float(axis_cosine_matrix[latent_idx, feature_idx])
        sign_correction = 1.0 if signed_cos >= 0 else -1.0
        matched_pairs.append(
            {
                "latent_index": int(latent_idx),
                "feature_index": int(feature_idx),
                "signed_cos": signed_cos,
                "axis_cos": axis_cos,
                "sign_correction": sign_correction,
                "cosine_similarity": signed_cos,
                "abs_cosine_similarity": axis_cos,
            }
        )
    unmatched_latents = [
        idx for idx in range(decoder_directions.shape[0]) if idx not in set(row_indices)
    ]
    alignment = {
        "alignment_mode": evaluation_config.alignment_mode,
        "eval_decoder_normalized": evaluation_config.normalize_decoder_for_eval,
        "matched_pairs": matched_pairs,
        "unmatched_latents": unmatched_latents,
    }

    feature_to_match = {p["feature_index"]: p for p in matched_pairs}
    axes: List[Dict[str, Any]] = []
    for feature_index in range(feature_unit.shape[0]):
        signed_cos_values = signed_cosine_matrix[:, feature_index]
        pos_mask = signed_cos_values >= 0
        neg_mask = signed_cos_values < 0
        pos_idx = None
        neg_idx = None
        if pos_mask.any():
            pos_candidates = torch.where(pos_mask)[0]
            best_pos = signed_cos_values[pos_candidates].argmax()
            pos_idx = int(pos_candidates[best_pos].item())
        if neg_mask.any():
            neg_candidates = torch.where(neg_mask)[0]
            best_neg = signed_cos_values[neg_candidates].argmin()
            neg_idx = int(neg_candidates[best_neg].item())

        signed_cos_pos = (
            None
            if pos_idx is None
            else float(signed_cosine_matrix[pos_idx, feature_index])
        )
        signed_cos_neg = (
            None
            if neg_idx is None
            else float(signed_cosine_matrix[neg_idx, feature_index])
        )
        axis_cos_pos = (
            None
            if pos_idx is None
            else float(axis_cosine_matrix[pos_idx, feature_index])
        )
        axis_cos_neg = (
            None
            if neg_idx is None
            else float(axis_cosine_matrix[neg_idx, feature_index])
        )

        if (
            axis_cos_pos is not None
            and axis_cos_pos < evaluation_config.min_axis_abs_cos
        ):
            pos_idx = None
            signed_cos_pos = None
            axis_cos_pos = None
        if (
            axis_cos_neg is not None
            and axis_cos_neg < evaluation_config.min_axis_abs_cos
        ):
            neg_idx = None
            signed_cos_neg = None
            axis_cos_neg = None

        match = feature_to_match.get(feature_index)
        axes.append(
            {
                "feature_index": int(feature_index),
                "one_to_one_latent": None if match is None else match["latent_index"],
                "one_to_one_signed_cos": None if match is None else match["signed_cos"],
                "one_to_one_axis_cos": None if match is None else match["axis_cos"],
                "one_to_one_sign_correction": (
                    None if match is None else match["sign_correction"]
                ),
                "latent_pos": pos_idx,
                "latent_neg": neg_idx,
                "signed_cos_pos": signed_cos_pos,
                "signed_cos_neg": signed_cos_neg,
                "axis_cos_pos": axis_cos_pos,
                "axis_cos_neg": axis_cos_neg,
            }
        )

    axes_with_pos = sum(axis["latent_pos"] is not None for axis in axes)
    axes_with_neg = sum(axis["latent_neg"] is not None for axis in axes)
    axes_with_both = sum(
        axis["latent_pos"] is not None and axis["latent_neg"] is not None
        for axis in axes
    )
    axes_with_neither = sum(
        axis["latent_pos"] is None and axis["latent_neg"] is None for axis in axes
    )
    axis_alignment: Dict[str, Any] = {
        "alignment_mode": evaluation_config.alignment_mode,
        "eval_decoder_normalized": evaluation_config.normalize_decoder_for_eval,
        "min_axis_abs_cos": float(evaluation_config.min_axis_abs_cos),
        "axes": axes,
        "coverage_summary": {
            "total_axes": len(axes),
            "axes_with_pos": int(axes_with_pos),
            "axes_with_neg": int(axes_with_neg),
            "axes_with_both": int(axes_with_both),
            "axes_with_neither": int(axes_with_neither),
        },
    }
    return alignment, axis_alignment, decoder_directions


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
    variance = sum((value - mean_value) ** 2 for value in filtered) / (
        len(filtered) - 1
    )
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


def _compute_coef_mse(true_values: Tensor, predicted_values: Tensor) -> Optional[float]:
    if true_values.numel() == 0:
        return None
    return float((predicted_values - true_values).pow(2).mean().item())


def _compute_pearson_r(
    true_values: Tensor, predicted_values: Tensor, eps: float
) -> Optional[float]:
    if true_values.numel() < 2:
        return None
    true_centered = true_values - true_values.mean()
    predicted_centered = predicted_values - predicted_values.mean()
    denominator = (
        torch.sqrt(true_centered.pow(2).sum() * predicted_centered.pow(2).sum()) + eps
    )
    if denominator.item() == 0.0:
        return None
    r_value = (true_centered * predicted_centered).sum() / denominator
    return float(r_value.clamp(min=-1.0, max=1.0).item())


def _compute_r2(
    true_values: Tensor, predicted_values: Tensor, eps: float
) -> Optional[float]:
    if true_values.numel() < 2:
        return None
    ss_res = (true_values - predicted_values).pow(2).sum()
    ss_tot = (true_values - true_values.mean()).pow(2).sum()
    if ss_tot.item() <= eps:
        return None
    r2_value = 1.0 - ss_res / ss_tot
    return float(r2_value.item())


def _calculate_per_axis_metrics(
    true_coefficients: Tensor,
    predicted_coefficients: Tensor,
    eps: float,
    min_regime_samples: int,
) -> Dict[str, Any]:
    positive_mask = true_coefficients > 0
    negative_mask = true_coefficients < 0
    n_pos = int(positive_mask.sum().item())
    n_neg = int(negative_mask.sum().item())

    gamma_plus = _compute_gamma(
        true_coefficients[positive_mask],
        predicted_coefficients[positive_mask],
        eps,
        min_regime_samples,
    )
    gamma_minus = _compute_gamma(
        true_coefficients[negative_mask],
        predicted_coefficients[negative_mask],
        eps,
        min_regime_samples,
    )
    coef_mse_pos = _compute_coef_mse(
        true_coefficients[positive_mask],
        predicted_coefficients[positive_mask],
    )
    coef_mse_neg = _compute_coef_mse(
        true_coefficients[negative_mask],
        predicted_coefficients[negative_mask],
    )
    pearson_r_pos = _compute_pearson_r(
        true_coefficients[positive_mask],
        predicted_coefficients[positive_mask],
        eps,
    )
    pearson_r_neg = _compute_pearson_r(
        true_coefficients[negative_mask],
        predicted_coefficients[negative_mask],
        eps,
    )
    r2_pos = _compute_r2(
        true_coefficients[positive_mask],
        predicted_coefficients[positive_mask],
        eps,
    )
    r2_neg = _compute_r2(
        true_coefficients[negative_mask],
        predicted_coefficients[negative_mask],
        eps,
    )

    rel_error_plus = None if gamma_plus is None else abs(gamma_plus - 1.0)
    rel_error_minus = None if gamma_minus is None else abs(gamma_minus - 1.0)
    return {
        "gamma_plus": gamma_plus,
        "gamma_minus": gamma_minus,
        "rel_error_plus": rel_error_plus,
        "rel_error_minus": rel_error_minus,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "coef_mse_pos": coef_mse_pos,
        "coef_mse_neg": coef_mse_neg,
        "pearson_r_pos": pearson_r_pos,
        "pearson_r_neg": pearson_r_neg,
        "r2_pos": r2_pos,
        "r2_neg": r2_neg,
    }


def compute_alignment_and_calibration(
    model: torch.nn.Module,
    x_test: Tensor,
    coefficients_test: Tensor,
    alignment: Dict[str, Any],
    axis_alignment: Dict[str, Any],
    evaluation_config: PolarityDialEvaluationConfig,
    eps: float = 1e-8,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    model.eval()
    with torch.no_grad():
        output = model(x_test)
    latents = output["latents"][0]

    # Protocol-level metrics (works for SAE and GatedSAE outputs).
    recon_key = (
        "x_hat" if "x_hat" in output else ("y_pred" if "y_pred" in output else None)
    )
    if recon_key is None:
        raise ValueError(
            "Model output does not contain a reconstruction key ('x_hat' or 'y_pred')."
        )
    x_hat = output[recon_key][0]
    test_reconstruction_mse = float((x_hat - x_test).pow(2).mean().item())
    test_l0 = float((latents.abs() > 1e-8).float().sum(dim=-1).mean().item())

    scatter_positive_true: List[float] = []
    scatter_positive_pred: List[float] = []
    scatter_negative_true: List[float] = []
    scatter_negative_pred: List[float] = []

    per_pair_entries: List[Dict[str, Any]] = []
    signed_latents = False
    if hasattr(model, "signed_latents"):
        signed_latents = bool(getattr(model, "signed_latents"))
    if hasattr(model, "sign_aware"):
        signed_latents = signed_latents or bool(getattr(model, "sign_aware"))
    is_non_negative_model = not signed_latents
    use_axis_aggregation = (
        evaluation_config.axis_aggregation_for_nonnegative and is_non_negative_model
    )
    if use_axis_aggregation:
        axis_entries = axis_alignment["axes"]
    else:
        axis_entries = alignment["matched_pairs"]

    for entry in axis_entries:
        if use_axis_aggregation:
            feature_index = int(entry["feature_index"])
            latent_pos = entry.get("latent_pos")
            latent_neg = entry.get("latent_neg")
            predicted = torch.zeros(latents.shape[0], device=latents.device)
            if latent_pos is not None and entry.get("signed_cos_pos") is not None:
                predicted = predicted + latents[:, latent_pos] * float(
                    entry["signed_cos_pos"]
                )
            if latent_neg is not None and entry.get("signed_cos_neg") is not None:
                predicted = predicted + latents[:, latent_neg] * float(
                    entry["signed_cos_neg"]
                )
            metrics = _calculate_per_axis_metrics(
                coefficients_test[:, feature_index],
                predicted,
                eps,
                evaluation_config.min_regime_samples,
            )
            per_pair_entries.append(
                {
                    "latent_index": None,
                    "feature_index": feature_index,
                    "latent_pos": latent_pos,
                    "latent_neg": latent_neg,
                    "gamma_plus": metrics["gamma_plus"],
                    "gamma_minus": metrics["gamma_minus"],
                    "rel_error_plus": metrics["rel_error_plus"],
                    "rel_error_minus": metrics["rel_error_minus"],
                    "n_pos": metrics["n_pos"],
                    "n_neg": metrics["n_neg"],
                    "coef_mse_pos": metrics["coef_mse_pos"],
                    "coef_mse_neg": metrics["coef_mse_neg"],
                    "pearson_r_pos": metrics["pearson_r_pos"],
                    "pearson_r_neg": metrics["pearson_r_neg"],
                    "r2_pos": metrics["r2_pos"],
                    "r2_neg": metrics["r2_neg"],
                }
            )

            remaining_positive_budget = (
                evaluation_config.max_points_per_regime_scatter
                - len(scatter_positive_true)
            )
            remaining_negative_budget = (
                evaluation_config.max_points_per_regime_scatter
                - len(scatter_negative_true)
            )
            positive_mask = coefficients_test[:, feature_index] > 0
            negative_mask = coefficients_test[:, feature_index] < 0
            true_pos, pred_pos = _sample_points(
                coefficients_test[:, feature_index][positive_mask],
                predicted[positive_mask],
                remaining_positive_budget,
            )
            true_neg, pred_neg = _sample_points(
                coefficients_test[:, feature_index][negative_mask],
                predicted[negative_mask],
                remaining_negative_budget,
            )
            scatter_positive_true.extend(true_pos)
            scatter_positive_pred.extend(pred_pos)
            scatter_negative_true.extend(true_neg)
            scatter_negative_pred.extend(pred_neg)
            continue

        latent_index = int(entry["latent_index"])
        feature_index = int(entry["feature_index"])
        predicted = latents[:, latent_index] * float(entry["signed_cos"])
        metrics = _calculate_per_axis_metrics(
            coefficients_test[:, feature_index],
            predicted,
            eps,
            evaluation_config.min_regime_samples,
        )
        per_pair_entries.append(
            {
                "latent_index": latent_index,
                "feature_index": feature_index,
                "gamma_plus": metrics["gamma_plus"],
                "gamma_minus": metrics["gamma_minus"],
                "rel_error_plus": metrics["rel_error_plus"],
                "rel_error_minus": metrics["rel_error_minus"],
                "n_pos": metrics["n_pos"],
                "n_neg": metrics["n_neg"],
                "coef_mse_pos": metrics["coef_mse_pos"],
                "coef_mse_neg": metrics["coef_mse_neg"],
                "pearson_r_pos": metrics["pearson_r_pos"],
                "pearson_r_neg": metrics["pearson_r_neg"],
                "r2_pos": metrics["r2_pos"],
                "r2_neg": metrics["r2_neg"],
            }
        )

        remaining_positive_budget = (
            evaluation_config.max_points_per_regime_scatter - len(scatter_positive_true)
        )
        remaining_negative_budget = (
            evaluation_config.max_points_per_regime_scatter - len(scatter_negative_true)
        )
        positive_mask = coefficients_test[:, feature_index] > 0
        negative_mask = coefficients_test[:, feature_index] < 0
        true_pos, pred_pos = _sample_points(
            coefficients_test[:, feature_index][positive_mask],
            predicted[positive_mask],
            remaining_positive_budget,
        )
        true_neg, pred_neg = _sample_points(
            coefficients_test[:, feature_index][negative_mask],
            predicted[negative_mask],
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
        "test_reconstruction_mse": test_reconstruction_mse,
        "test_l0": test_l0,
        "mean_gamma_plus": _safe_mean(gamma_plus_values),
        "mean_gamma_minus": _safe_mean(gamma_minus_values),
        "std_gamma_plus": _safe_std(gamma_plus_values),
        "std_gamma_minus": _safe_std(gamma_minus_values),
        "mean_rel_error_plus": _safe_mean(rel_error_plus_values),
        "mean_rel_error_minus": _safe_mean(rel_error_minus_values),
        "std_rel_error_plus": _safe_std(rel_error_plus_values),
        "std_rel_error_minus": _safe_std(rel_error_minus_values),
        "num_pairs": len(per_pair_entries),
        "mean_coef_mse_pos": _safe_mean(
            [entry["coef_mse_pos"] for entry in per_pair_entries]
        ),
        "mean_coef_mse_neg": _safe_mean(
            [entry["coef_mse_neg"] for entry in per_pair_entries]
        ),
        "std_coef_mse_pos": _safe_std(
            [entry["coef_mse_pos"] for entry in per_pair_entries]
        ),
        "std_coef_mse_neg": _safe_std(
            [entry["coef_mse_neg"] for entry in per_pair_entries]
        ),
        "mean_pearson_r_pos": _safe_mean(
            [entry["pearson_r_pos"] for entry in per_pair_entries]
        ),
        "mean_pearson_r_neg": _safe_mean(
            [entry["pearson_r_neg"] for entry in per_pair_entries]
        ),
        "std_pearson_r_pos": _safe_std(
            [entry["pearson_r_pos"] for entry in per_pair_entries]
        ),
        "std_pearson_r_neg": _safe_std(
            [entry["pearson_r_neg"] for entry in per_pair_entries]
        ),
        "mean_r2_pos": _safe_mean([entry["r2_pos"] for entry in per_pair_entries]),
        "mean_r2_neg": _safe_mean([entry["r2_neg"] for entry in per_pair_entries]),
        "std_r2_pos": _safe_std([entry["r2_pos"] for entry in per_pair_entries]),
        "std_r2_neg": _safe_std([entry["r2_neg"] for entry in per_pair_entries]),
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
    axis_alignment: Dict[str, Any],
    calibration: Dict[str, Any],
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    alignment_path = os.path.join(model_dir, "alignment.json")
    with open(alignment_path, "w") as f:
        json.dump(alignment, f, indent=2)
    axis_alignment_path = os.path.join(model_dir, "axis_alignment.json")
    with open(axis_alignment_path, "w") as f:
        json.dump(axis_alignment, f, indent=2)
    calibration_path = os.path.join(model_dir, "calibration.json")
    with open(calibration_path, "w") as f:
        json.dump(calibration, f, indent=2)
