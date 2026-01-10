import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DEvaluationConfig,
)


def extract_decoder_directions(model: torch.nn.Module) -> Tensor:
    if not hasattr(model, "W_dec"):
        raise ValueError("Model is missing required attribute 'W_dec'.")
    W_dec = getattr(model, "W_dec")
    if not isinstance(W_dec, torch.Tensor):
        raise ValueError("Model attribute 'W_dec' must be a torch.Tensor.")
    if W_dec.ndim != 3:
        raise ValueError(
            "Model attribute 'W_dec' must have shape (n_instances, d_in, d_hidden); "
            f"received tensor with {W_dec.ndim} dimensions."
        )
    if int(W_dec.shape[0]) != 1:
        raise ValueError(
            "Toy Model Geometry Validation expects n_instances=1; "
            f"received n_instances={int(W_dec.shape[0])}."
        )

    W_dec_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)
    directions = W_dec_norm[0].transpose(0, 1).detach().cpu().to(dtype=torch.float32)
    if int(directions.shape[1]) != 2:
        raise ValueError(
            "Toy Model Geometry Validation expects decoder directions in R^2; "
            f"received d_in={int(directions.shape[1])}."
        )
    return directions


def extract_signed_pair_coefficients(coeffs: Tensor) -> Tensor:
    if coeffs.ndim != 2:
        raise ValueError(
            f"coeffs must have shape (num_samples, 2k); received {coeffs}."
        )
    if int(coeffs.shape[1]) % 2 != 0:
        raise ValueError(
            f"coeffs must have an even number of columns (2k); received {int(coeffs.shape[1])}."
        )
    plus_coeffs = coeffs[:, 0::2]
    minus_coeffs = coeffs[:, 1::2]
    return plus_coeffs - minus_coeffs


def _pearsonr(x: Tensor, y: Tensor) -> Optional[float]:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("pearsonr inputs must be 1D tensors.")
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("pearsonr inputs must have the same length.")
    if int(x.shape[0]) < 2:
        return None

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = x_centered.norm() * y_centered.norm()
    if float(denominator.item()) == 0.0:
        return None
    return float((x_centered * y_centered).sum().item() / denominator.item())


def _best_aligned_latent_by_pair(dot_products: Tensor) -> List[Dict[str, Any]]:
    abs_cos = dot_products.abs()
    best_latents = abs_cos.argmax(dim=0)
    output: List[Dict[str, Any]] = []
    for pair_index in range(int(dot_products.shape[1])):
        latent_index = int(best_latents[pair_index].item())
        dot_value = float(dot_products[latent_index, pair_index].item())
        output.append(
            {
                "pair_index": pair_index,
                "latent_index": latent_index,
                "axis_abs_cos": float(abs_cos[latent_index, pair_index].item()),
                "orientation_sign": 1.0 if dot_value >= 0.0 else -1.0,
            }
        )
    return output


def _select_histogram_latent(
    dot_products: Tensor,
    pair_index: int,
) -> Dict[str, Any]:
    if pair_index < 0 or pair_index >= int(dot_products.shape[1]):
        raise ValueError(
            "pair_index must be within [0, num_pairs); "
            f"received pair_index={pair_index}."
        )
    abs_cos = dot_products[:, pair_index].abs()
    best_latent_index = int(abs_cos.argmax().item())
    dot_value = float(dot_products[best_latent_index, pair_index].item())
    return {
        "pair_index": int(pair_index),
        "latent_index": best_latent_index,
        "axis_abs_cos": float(abs_cos[best_latent_index].item()),
        "orientation_sign": 1.0 if dot_value >= 0.0 else -1.0,
    }


def _evaluate_pair_consolidation(
    latents: Tensor,
    signed_coeffs: Tensor,
    pair_active_mask: Tensor,
    dot_products: Tensor,
    evaluation_config: Toy2DEvaluationConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    d_hidden = int(latents.shape[1])
    num_pairs = int(signed_coeffs.shape[1])

    abs_cos = dot_products.abs()
    orientation_sign = torch.where(dot_products >= 0.0, 1.0, -1.0)

    evaluability: List[Dict[str, Any]] = []
    consolidated_pairs: List[Dict[str, Any]] = []
    evaluable_count = 0
    consolidated_count = 0
    eps = 1e-8

    for pair_index in range(num_pairs):
        eligible_mask = pair_active_mask[:, pair_index].to(dtype=torch.bool)
        eligible_coeffs = signed_coeffs[:, pair_index][eligible_mask]

        positive_mask = eligible_coeffs > 0.0
        negative_mask = eligible_coeffs < 0.0
        pos_count = int(positive_mask.sum().item())
        neg_count = int(negative_mask.sum().item())

        evaluable = pos_count >= int(
            evaluation_config.min_regime_samples
        ) and neg_count >= int(evaluation_config.min_regime_samples)
        evaluability.append(
            {
                "pair_index": pair_index,
                "eligible_samples": int(eligible_mask.sum().item()),
                "positive_regime_samples": pos_count,
                "negative_regime_samples": neg_count,
                "evaluable": evaluable,
            }
        )
        if not evaluable:
            continue

        evaluable_count += 1
        candidate_latents = abs_cos[:, pair_index] >= float(
            evaluation_config.min_axis_abs_cos
        )
        if not bool(candidate_latents.any().item()):
            continue

        best_candidate: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple[float, float]] = None
        for latent_index in range(d_hidden):
            if not bool(candidate_latents[latent_index].item()):
                continue

            oriented_latents = (
                latents[:, latent_index][eligible_mask]
                * orientation_sign[latent_index, pair_index]
            )
            # Spec: correlation must be computed only on samples where |c_j(x)| > 0.
            # (eligible_coeffs already conditions on pair_active_mask for this pair)
            corr_mask = eligible_coeffs != 0.0
            activation_corr = _pearsonr(
                oriented_latents[corr_mask], eligible_coeffs[corr_mask]
            )
            if activation_corr is None or activation_corr < float(
                evaluation_config.min_activation_corr
            ):
                continue

            pos_sign_agreement = float(
                (oriented_latents[positive_mask] > eps).float().mean().item()
            )
            neg_sign_agreement = float(
                (oriented_latents[negative_mask] < -eps).float().mean().item()
            )
            if (
                pos_sign_agreement < float(evaluation_config.min_sign_agreement)
                or neg_sign_agreement < float(evaluation_config.min_sign_agreement)
            ):
                continue

            mean_pos = float(oriented_latents[positive_mask].mean().item())
            mean_neg = float(oriented_latents[negative_mask].mean().item())
            if mean_pos <= 0.0 or mean_neg >= 0.0:
                continue

            score = (
                float(activation_corr),
                float(abs_cos[latent_index, pair_index].item()),
            )
            candidate = {
                "pair_index": pair_index,
                "latent_index": int(latent_index),
                "axis_abs_cos": score[1],
                "activation_corr": score[0],
                "positive_sign_agreement": pos_sign_agreement,
                "negative_sign_agreement": neg_sign_agreement,
                "positive_regime_samples": pos_count,
                "negative_regime_samples": neg_count,
                "mean_activation_positive_regime": mean_pos,
                "mean_activation_negative_regime": mean_neg,
            }
            if best_candidate is None or best_score is None or score > best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is not None:
            consolidated_count += 1
            consolidated_pairs.append(best_candidate)

    pair_consolidation_rate = (
        consolidated_count / float(num_pairs) if num_pairs else 0.0
    )
    rate_over_evaluable = (
        None if evaluable_count == 0 else consolidated_count / float(evaluable_count)
    )
    evaluable_fraction = evaluable_count / float(num_pairs) if num_pairs else 0.0
    summary = {
        "num_pairs_total": num_pairs,
        "num_pairs_evaluable": evaluable_count,
        "num_pairs_consolidated": consolidated_count,
        "pair_consolidation_rate": pair_consolidation_rate,
        "pair_consolidation_rate_over_evaluable_pairs": rate_over_evaluable,
        # Alias to satisfy requirements.md naming.
        "rate_over_evaluable_pairs": rate_over_evaluable,
        "evaluable_pair_fraction": evaluable_fraction,
    }
    return summary, evaluability, consolidated_pairs


def evaluate_toy2d_model(
    model: torch.nn.Module,
    data: Dict[str, Tensor],
    evaluation_config: Toy2DEvaluationConfig,
    model_name: str,
    sign_aware: bool,
    device: str,
) -> Dict[str, Any]:
    required_keys = (
        "x_test",
        "coeffs_test",
        "pair_active_mask_test",
        "pair_axis_directions",
    )
    for key in required_keys:
        if key not in data:
            raise ValueError(f"data is missing required key '{key}'.")

    x_test = data["x_test"].to(device=device)
    coeffs_test = data["coeffs_test"].detach().cpu()
    pair_active_mask_test = data["pair_active_mask_test"].detach().cpu()
    pair_axis_directions = data["pair_axis_directions"].detach().cpu()
    signed_coeffs = extract_signed_pair_coefficients(coeffs_test)

    model.eval()
    with torch.no_grad():
        output = model(x_test)
    if "latents" not in output:
        raise ValueError("Model output is missing required key 'latents'.")
    latents_tensor = output["latents"]
    if latents_tensor.ndim != 3 or int(latents_tensor.shape[0]) != 1:
        raise ValueError(
            "Model output 'latents' must have shape (1, batch, d_hidden); "
            f"received {tuple(int(dim) for dim in latents_tensor.shape)}."
        )
    latents = latents_tensor[0].detach().cpu().to(dtype=torch.float32)
    if int(latents.shape[0]) != int(coeffs_test.shape[0]):
        raise ValueError(
            "Latents and coeffs_test must have matching sample counts; "
            f"received latents={int(latents.shape[0])} coeffs_test={int(coeffs_test.shape[0])}."
        )
    latent_active_fraction = (
        (latents.abs() > 1e-8).float().mean(dim=0).detach().cpu().numpy()
    )

    decoder_directions = extract_decoder_directions(model)
    pair_axis_unit = pair_axis_directions.to(dtype=torch.float32)
    pair_axis_unit = pair_axis_unit / (pair_axis_unit.norm(dim=1, keepdim=True) + 1e-8)
    dot_products = decoder_directions @ pair_axis_unit.transpose(0, 1)

    alignment_summary = _best_aligned_latent_by_pair(dot_products)
    consolidation_summary, pair_evaluability, consolidated_pairs = (
        _evaluate_pair_consolidation(
            latents=latents,
            signed_coeffs=signed_coeffs,
            pair_active_mask=pair_active_mask_test,
            dot_products=dot_products,
            evaluation_config=evaluation_config,
        )
    )
    histogram_selection: Optional[Dict[str, Any]] = None
    if sign_aware:
        if consolidated_pairs:
            best = max(
                consolidated_pairs,
                key=lambda candidate: (
                    candidate.get("activation_corr", -math.inf),
                    candidate.get("axis_abs_cos", -math.inf),
                ),
            )
            latent_index = int(best["latent_index"])
            histogram_selection = {
                "pair_index": int(best["pair_index"]),
                "latent_index": latent_index,
                "axis_abs_cos": float(best["axis_abs_cos"]),
                "orientation_sign": int(
                    1 if float(dot_products[latent_index, int(best["pair_index"])]) >= 0.0 else -1
                ),
                "latent_active_fraction": float(latent_active_fraction[latent_index]),
                "selection_reason": "best_consolidated_pair",
            }
        else:
            pair_index = 0
            abs_cos = dot_products[:, pair_index].abs().detach().cpu().numpy()
            aligned_mask = abs_cos >= float(evaluation_config.min_axis_abs_cos)
            if bool(np.any(aligned_mask)):
                candidates = np.where(aligned_mask)[0]
                best_latent_index = int(
                    candidates[int(np.argmax(latent_active_fraction[candidates]))]
                )
                selection_reason = "most_active_aligned_latent"
            else:
                best_latent_index = int(abs_cos.argmax())
                selection_reason = "best_aligned_latent_no_candidates"
            histogram_selection = {
                "pair_index": int(pair_index),
                "latent_index": best_latent_index,
                "axis_abs_cos": float(abs_cos[best_latent_index]),
                "orientation_sign": int(
                    1 if float(dot_products[best_latent_index, pair_index]) >= 0.0 else -1
                ),
                "latent_active_fraction": float(latent_active_fraction[best_latent_index]),
                "selection_reason": selection_reason,
            }

    return {
        "model_name": model_name,
        "status": "ok",
        "sign_aware": bool(sign_aware),
        "alignment": alignment_summary,
        "histogram_latent": histogram_selection,
        **consolidation_summary,
        "pair_evaluability": pair_evaluability,
        "consolidated_pairs": consolidated_pairs,
        "thresholds": {
            "min_axis_abs_cos": float(evaluation_config.min_axis_abs_cos),
            "min_activation_corr": float(evaluation_config.min_activation_corr),
            "min_sign_agreement": float(evaluation_config.min_sign_agreement),
            "min_regime_samples": int(evaluation_config.min_regime_samples),
        },
    }


def persist_evaluation(evaluation: Dict[str, Any], model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    evaluation_path = os.path.join(model_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=2)
