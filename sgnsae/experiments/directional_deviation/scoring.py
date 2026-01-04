import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import einops
import torch
from torch import Tensor

from sgnsae.experiments.directional_deviation.baseline_scoring import (
    calculate_directional_scores_from_one_sided_evidence,
)
from sgnsae.experiments.directional_deviation.pooling import (
    PoolingMode,
    pool_exceedances,
)
from sgnsae.models import GatedSAE, SAE


_EPSILON = 1e-8


@dataclass(frozen=True)
class SignAwareLatentThresholds:
    tau_plus: Tensor
    tau_minus: Tensor


@dataclass(frozen=True)
class BaselineEvidenceThresholds:
    evidence_tau: Tensor
    latent_directions: Tensor


@dataclass(frozen=True)
class ScoreThresholds:
    theta_high: float
    theta_low: float
    score_fpr: float
    score_quantile: float


def _validate_tensor_ndim(tensor: Tensor, expected_ndim: int, name: str) -> None:
    if tensor.ndim != expected_ndim:
        raise ValueError(f"Expected {name}.ndim == {expected_ndim}, got {tensor.ndim}.")


def _validate_open_unit_interval(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be within (0, 1); received {value}.")


def _validate_min_validation_samples(
    num_samples: int,
    min_validation_samples: int,
    name: str,
) -> None:
    if num_samples < min_validation_samples:
        raise ValueError(
            f"{name} must contain at least min_validation_samples={min_validation_samples} samples; "
            f"received {num_samples}."
        )


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
    _validate_tensor_ndim(decoder, 3, "W_dec")
    _ensure_single_instance(model)
    return decoder[0]


def extract_detection_signal(model: torch.nn.Module, x: Tensor) -> Tensor:
    if isinstance(model, GatedSAE):
        return _extract_detection_signal_gated(model, x)
    if isinstance(model, SAE):
        return _extract_detection_signal_sae(model, x)
    raise ValueError(f"Unsupported model type: {type(model).__name__}.")


def _extract_detection_signal_gated(model: GatedSAE, x: Tensor) -> Tensor:
    output = model(x)
    if "pi_gate" not in output:
        raise ValueError("GatedSAE output is missing required key 'pi_gate'.")
    pi_gate = output["pi_gate"]
    _validate_tensor_ndim(pi_gate, 3, "pi_gate")
    _ensure_single_instance(model)
    return pi_gate[0]


def _extract_detection_signal_sae(model: SAE, x: Tensor) -> Tensor:
    _ensure_single_instance(model)

    if x.ndim < 2 or x.ndim > 3:
        raise ValueError(f"Invalid dimension count of input: {x.ndim}")

    x = x.to(model.device)
    while x.ndim < 3:
        x = x.unsqueeze(0)

    decoder = model.W_dec
    decoder_norm = decoder / (decoder.norm(dim=1, keepdim=True) + _EPSILON)

    preactivations = (
        einops.einsum(
            x - model.b_dec.unsqueeze(1),
            decoder_norm,
            "n_instances batch d_in, n_instances d_in d_hidden -> batch n_instances d_hidden",
        )
        + model.b_enc
    )
    _validate_tensor_ndim(preactivations, 3, "sae_preactivations")
    return preactivations[:, 0, :]


def extract_one_sided_evidence(model: torch.nn.Module, x: Tensor) -> Tensor:
    if hasattr(model, "sign_aware") and bool(getattr(model, "sign_aware")):
        raise ValueError("Baseline scoring requires sign_aware=False.")

    output = model(x)
    if "latents" not in output:
        raise ValueError("Model output is missing required key 'latents'.")
    latents = output["latents"]
    _validate_tensor_ndim(latents, 3, "latents")
    _ensure_single_instance(model)

    evidence = latents[0]
    if torch.any(evidence < -_EPSILON):
        raise ValueError("Expected baseline evidence to be non-negative.")
    return evidence


def assign_baseline_latent_directions(
    model: torch.nn.Module,
    shift_mode: Literal["single_channel", "subset"],
    subset_indices: Optional[List[int]] = None,
) -> Tensor:
    decoder = _extract_decoder_matrix(model).detach()
    _validate_tensor_ndim(decoder, 2, "decoder")
    if shift_mode == "single_channel":
        return _assign_directions_single_channel(decoder)
    if shift_mode == "subset":
        if subset_indices is None:
            raise ValueError(
                "subset_indices must be provided when shift_mode='subset'."
            )
        return _assign_directions_subset(decoder, subset_indices)
    raise ValueError(f"Unsupported shift_mode: {shift_mode}.")


def _assign_directions_single_channel(decoder: Tensor) -> Tensor:
    abs_weights = decoder.abs()
    channel_indices = abs_weights.argmax(dim=0)
    latent_indices = torch.arange(decoder.shape[1], device=decoder.device)
    selected_weights = decoder[channel_indices, latent_indices]
    directions = torch.sign(selected_weights)
    directions = torch.where(directions == 0, torch.ones_like(directions), directions)
    return directions.to(dtype=torch.int64)


def _assign_directions_subset(decoder: Tensor, subset_indices: List[int]) -> Tensor:
    if not subset_indices:
        raise ValueError("subset_indices must be a non-empty list.")
    subset_tensor = torch.tensor(
        subset_indices, dtype=torch.int64, device=decoder.device
    )
    dot_products = decoder.index_select(dim=0, index=subset_tensor).sum(dim=0)
    directions = torch.sign(dot_products)
    directions = torch.where(directions == 0, torch.ones_like(directions), directions)
    return directions.to(dtype=torch.int64)


def calibrate_sign_aware_latent_thresholds(
    pi_validation: Tensor,
    latent_quantile: float,
    min_validation_samples: int,
) -> SignAwareLatentThresholds:
    _validate_open_unit_interval(latent_quantile, "latent_quantile")
    _validate_tensor_ndim(pi_validation, 2, "pi_validation")
    _validate_min_validation_samples(
        pi_validation.shape[0], min_validation_samples, "pi_validation"
    )
    tau_plus = torch.quantile(pi_validation, latent_quantile, dim=0)
    tau_minus = torch.quantile(-pi_validation, latent_quantile, dim=0)
    return SignAwareLatentThresholds(tau_plus=tau_plus, tau_minus=tau_minus)


def calibrate_baseline_evidence_thresholds(
    evidence_validation: Tensor,
    latent_directions: Tensor,
    latent_quantile: float,
    min_validation_samples: int,
) -> BaselineEvidenceThresholds:
    _validate_open_unit_interval(latent_quantile, "latent_quantile")
    _validate_tensor_ndim(evidence_validation, 2, "evidence_validation")
    _validate_tensor_ndim(latent_directions, 1, "latent_directions")
    _validate_min_validation_samples(
        evidence_validation.shape[0], min_validation_samples, "evidence_validation"
    )
    if evidence_validation.shape[1] != latent_directions.shape[0]:
        raise ValueError(
            "Expected evidence_validation.shape[1] == latent_directions.shape[0], got "
            f"{evidence_validation.shape[1]} vs {latent_directions.shape[0]}."
        )
    evidence_tau = torch.quantile(evidence_validation, latent_quantile, dim=0)
    return BaselineEvidenceThresholds(
        evidence_tau=evidence_tau,
        latent_directions=latent_directions.to(dtype=torch.int64),
    )


def calculate_sign_aware_directional_scores(
    pi: Tensor,
    latent_thresholds: SignAwareLatentThresholds,
    pooling: PoolingMode = "sum",
    topk_k: Optional[int] = None,
    targets: Optional[Tensor] = None,
    latent_argmax_channel: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    _validate_tensor_ndim(pi, 2, "pi")
    _validate_tensor_ndim(latent_thresholds.tau_plus, 1, "tau_plus")
    _validate_tensor_ndim(latent_thresholds.tau_minus, 1, "tau_minus")
    if pi.shape[1] != latent_thresholds.tau_plus.shape[0]:
        raise ValueError(
            "Expected pi.shape[1] == tau_plus.shape[0], got "
            f"{pi.shape[1]} vs {latent_thresholds.tau_plus.shape[0]}."
        )
    if pi.shape[1] != latent_thresholds.tau_minus.shape[0]:
        raise ValueError(
            "Expected pi.shape[1] == tau_minus.shape[0], got "
            f"{pi.shape[1]} vs {latent_thresholds.tau_minus.shape[0]}."
        )

    exceed_high = torch.relu(pi - latent_thresholds.tau_plus.unsqueeze(0))
    exceed_low = torch.relu(-pi - latent_thresholds.tau_minus.unsqueeze(0))

    if targets is None and latent_argmax_channel is None:
        score_high = pool_exceedances(exceed_high, pooling=pooling, topk_k=topk_k)
        score_low = pool_exceedances(exceed_low, pooling=pooling, topk_k=topk_k)
        return score_high, score_low

    if targets is None or latent_argmax_channel is None:
        raise ValueError(
            "targets and latent_argmax_channel must both be provided for target-restricted scoring."
        )

    score_high = _pool_exceedances_by_target(
        exceedances=exceed_high,
        targets=targets,
        latent_argmax_channel=latent_argmax_channel,
        pooling=pooling,
        topk_k=topk_k,
        latent_filter_mask=None,
    )
    score_low = _pool_exceedances_by_target(
        exceedances=exceed_low,
        targets=targets,
        latent_argmax_channel=latent_argmax_channel,
        pooling=pooling,
        topk_k=topk_k,
        latent_filter_mask=None,
    )
    return score_high, score_low


def calculate_baseline_directional_scores(
    one_sided_evidence: Tensor,
    baseline_thresholds: BaselineEvidenceThresholds,
    pooling: PoolingMode = "sum",
    topk_k: Optional[int] = None,
    targets: Optional[Tensor] = None,
    latent_argmax_channel: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if targets is None and latent_argmax_channel is None:
        return calculate_directional_scores_from_one_sided_evidence(
            one_sided_evidence=one_sided_evidence,
            evidence_thresholds=baseline_thresholds.evidence_tau,
            latent_directions=baseline_thresholds.latent_directions,
            pooling=pooling,
            topk_k=topk_k,
        )

    if targets is None or latent_argmax_channel is None:
        raise ValueError(
            "targets and latent_argmax_channel must both be provided for target-restricted scoring."
        )

    evidence_excess = torch.relu(
        one_sided_evidence - baseline_thresholds.evidence_tau.unsqueeze(0)
    )
    directions = baseline_thresholds.latent_directions.to(device=evidence_excess.device)
    if directions.ndim != 1:
        raise ValueError(
            "Expected baseline_thresholds.latent_directions.ndim == 1, got "
            f"{directions.ndim}."
        )
    if evidence_excess.shape[1] != directions.shape[0]:
        raise ValueError(
            "Expected evidence_excess.shape[1] == latent_directions.shape[0], got "
            f"{evidence_excess.shape[1]} vs {directions.shape[0]}."
        )

    high_mask = directions > 0
    low_mask = directions < 0
    score_high = _pool_exceedances_by_target(
        exceedances=evidence_excess,
        targets=targets,
        latent_argmax_channel=latent_argmax_channel,
        pooling=pooling,
        topk_k=topk_k,
        latent_filter_mask=high_mask,
    )
    score_low = _pool_exceedances_by_target(
        exceedances=evidence_excess,
        targets=targets,
        latent_argmax_channel=latent_argmax_channel,
        pooling=pooling,
        topk_k=topk_k,
        latent_filter_mask=low_mask,
    )
    return score_high, score_low


def calculate_latent_argmax_channel(model: torch.nn.Module) -> Tensor:
    decoder = _extract_decoder_matrix(model).detach()
    abs_weights = decoder.abs()
    return abs_weights.argmax(dim=0).to(dtype=torch.int64)


def _pool_exceedances_by_target(
    exceedances: Tensor,
    targets: Tensor,
    latent_argmax_channel: Tensor,
    pooling: PoolingMode,
    topk_k: Optional[int],
    latent_filter_mask: Optional[Tensor],
) -> Tensor:
    _validate_tensor_ndim(exceedances, 2, "exceedances")
    _validate_tensor_ndim(targets, 1, "targets")
    _validate_tensor_ndim(latent_argmax_channel, 1, "latent_argmax_channel")
    if exceedances.shape[0] != targets.shape[0]:
        raise ValueError(
            "Expected exceedances.shape[0] == targets.shape[0], got "
            f"{exceedances.shape[0]} vs {targets.shape[0]}."
        )
    if exceedances.shape[1] != latent_argmax_channel.shape[0]:
        raise ValueError(
            "Expected exceedances.shape[1] == latent_argmax_channel.shape[0], got "
            f"{exceedances.shape[1]} vs {latent_argmax_channel.shape[0]}."
        )

    scores = torch.zeros(
        exceedances.shape[0],
        dtype=exceedances.dtype,
        device=exceedances.device,
    )

    targets_device = targets.to(device=exceedances.device, dtype=torch.int64)
    latent_channels = latent_argmax_channel.to(
        device=exceedances.device, dtype=torch.int64
    )
    latent_mask_base = (
        torch.ones(
            latent_channels.shape[0],
            dtype=torch.bool,
            device=exceedances.device,
        )
        if latent_filter_mask is None
        else latent_filter_mask.to(device=exceedances.device, dtype=torch.bool)
    )
    if latent_mask_base.shape[0] != latent_channels.shape[0]:
        raise ValueError(
            "Expected latent_filter_mask to have the same length as latent_argmax_channel; "
            f"received {latent_mask_base.shape[0]} vs {latent_channels.shape[0]}."
        )

    for target in torch.unique(targets_device):
        sample_mask = targets_device == target
        latent_mask = (latent_channels == target) & latent_mask_base
        pooled = pool_exceedances(
            exceedances=exceedances[sample_mask][:, latent_mask],
            pooling=pooling,
            topk_k=topk_k,
        )
        scores[sample_mask] = pooled

    return scores


def calibrate_score_thresholds(
    score_high_validation: Tensor,
    score_low_validation: Tensor,
    score_fpr: float,
    min_validation_samples: int,
) -> ScoreThresholds:
    _validate_open_unit_interval(score_fpr, "score_fpr")
    _validate_tensor_ndim(score_high_validation, 1, "score_high_validation")
    _validate_tensor_ndim(score_low_validation, 1, "score_low_validation")
    if score_high_validation.shape[0] != score_low_validation.shape[0]:
        raise ValueError(
            "Expected score_high_validation.shape[0] == score_low_validation.shape[0], got "
            f"{score_high_validation.shape[0]} vs {score_low_validation.shape[0]}."
        )
    _validate_min_validation_samples(
        score_high_validation.shape[0], min_validation_samples, "score_validation"
    )

    score_quantile = 1.0 - score_fpr
    theta_high = float(torch.quantile(score_high_validation, score_quantile).item())
    theta_low = float(torch.quantile(score_low_validation, score_quantile).item())
    return ScoreThresholds(
        theta_high=theta_high,
        theta_low=theta_low,
        score_fpr=score_fpr,
        score_quantile=score_quantile,
    )


def build_threshold_metadata(
    model: torch.nn.Module,
    scoring_mode: Literal["sign_aware", "baseline"],
    latent_quantile: float,
    score_thresholds: ScoreThresholds,
    shift_mode: Literal["single_channel", "subset"],
    subset_indices: Optional[List[int]] = None,
    pooling: PoolingMode = "sum",
    topk_k: Optional[int] = None,
    target_restricted_scoring: bool = False,
) -> Dict[str, Any]:
    _validate_open_unit_interval(latent_quantile, "latent_quantile")
    learned_thresholds = _extract_learned_model_thresholds(model)
    metadata: Dict[str, Any] = {
        "scoring_mode": scoring_mode,
        "learned_model_thresholds": learned_thresholds,
        "calibrated_eval_thresholds": {
            "latent_quantile": latent_quantile,
            "score_fpr": score_thresholds.score_fpr,
            "score_quantile": score_thresholds.score_quantile,
            "theta_high": score_thresholds.theta_high,
            "theta_low": score_thresholds.theta_low,
        },
        "scoring_config": {
            "pooling": pooling,
            "topk_k": topk_k,
            "target_restricted_scoring": bool(target_restricted_scoring),
        },
        "signal_definition": {
            "sign_aware": {
                "detection_signal": "pi_gate (gated) or SAE preactivations",
                "score_high": f"{pooling} ReLU(pi - tau_plus)",
                "score_low": f"{pooling} ReLU(-pi - tau_minus)",
            },
            "baseline": {
                "evidence_signal": "non-negative latents",
                "score_high": f"{pooling}{{dir=+1}} ReLU(e - eta)",
                "score_low": f"{pooling}{{dir=-1}} ReLU(e - eta)",
            },
        }[scoring_mode],
        "baseline_direction_assignment": _baseline_direction_assignment_metadata(
            scoring_mode=scoring_mode,
            shift_mode=shift_mode,
            subset_indices=subset_indices,
        ),
    }
    return metadata


def _baseline_direction_assignment_metadata(
    scoring_mode: Literal["sign_aware", "baseline"],
    shift_mode: Literal["single_channel", "subset"],
    subset_indices: Optional[List[int]],
) -> Optional[Dict[str, Any]]:
    if scoring_mode != "baseline":
        return None
    if shift_mode == "single_channel":
        return {"rule": "argmax_abs_decoder_component_sign", "subset_indices": None}
    if shift_mode == "subset":
        return {
            "rule": "sign_of_decoder_dot_subset_mask",
            "subset_indices": subset_indices,
        }
    raise ValueError(f"Unsupported shift_mode: {shift_mode}.")


def _extract_learned_model_thresholds(
    model: torch.nn.Module,
) -> Optional[Dict[str, Any]]:
    if not hasattr(model, "sign_aware") or not bool(getattr(model, "sign_aware")):
        return None

    if hasattr(model, "single_activation_offset") and bool(
        getattr(model, "single_activation_offset")
    ):
        if not hasattr(model, "act_offset"):
            raise ValueError("sign_aware model missing act_offset.")
        act_offset = getattr(model, "act_offset")
        if not isinstance(act_offset, Tensor):
            raise ValueError("act_offset must be a torch.Tensor.")
        _validate_tensor_ndim(act_offset, 2, "act_offset")
        _ensure_single_instance(model)
        delta = torch.relu(act_offset[0]).detach().cpu().tolist()
        return {"delta_plus": delta, "delta_minus": delta}

    if not hasattr(model, "act_offset_right") or not hasattr(model, "act_offset_left"):
        raise ValueError("sign_aware model missing act_offset_right/act_offset_left.")
    act_offset_right = getattr(model, "act_offset_right")
    act_offset_left = getattr(model, "act_offset_left")
    if not isinstance(act_offset_right, Tensor) or not isinstance(
        act_offset_left, Tensor
    ):
        raise ValueError("act_offset_right/act_offset_left must be torch.Tensors.")
    _validate_tensor_ndim(act_offset_right, 2, "act_offset_right")
    _validate_tensor_ndim(act_offset_left, 2, "act_offset_left")
    _ensure_single_instance(model)

    delta_plus = torch.relu(act_offset_right[0]).detach().cpu().tolist()
    delta_minus = torch.relu(act_offset_left[0]).detach().cpu().tolist()
    return {"delta_plus": delta_plus, "delta_minus": delta_minus}


def persist_sign_aware_threshold_artifacts(
    thresholds_dir: str,
    latent_thresholds: SignAwareLatentThresholds,
    score_thresholds: ScoreThresholds,
    metadata: Dict[str, Any],
) -> None:
    os.makedirs(thresholds_dir, exist_ok=True)
    torch.save(
        latent_thresholds.tau_plus.detach().cpu(),
        os.path.join(thresholds_dir, "tau_plus.pt"),
    )
    torch.save(
        latent_thresholds.tau_minus.detach().cpu(),
        os.path.join(thresholds_dir, "tau_minus.pt"),
    )
    _persist_score_thresholds(
        thresholds_dir=thresholds_dir,
        score_thresholds=score_thresholds,
    )
    _persist_metadata(thresholds_dir=thresholds_dir, metadata=metadata)


def persist_baseline_threshold_artifacts(
    thresholds_dir: str,
    baseline_thresholds: BaselineEvidenceThresholds,
    score_thresholds: ScoreThresholds,
    metadata: Dict[str, Any],
) -> None:
    os.makedirs(thresholds_dir, exist_ok=True)
    torch.save(
        baseline_thresholds.evidence_tau.detach().cpu(),
        os.path.join(thresholds_dir, "evidence_tau.pt"),
    )
    torch.save(
        baseline_thresholds.latent_directions.detach().cpu(),
        os.path.join(thresholds_dir, "latent_directions.pt"),
    )
    _persist_score_thresholds(
        thresholds_dir=thresholds_dir,
        score_thresholds=score_thresholds,
    )
    _persist_metadata(thresholds_dir=thresholds_dir, metadata=metadata)


def _persist_score_thresholds(
    thresholds_dir: str, score_thresholds: ScoreThresholds
) -> None:
    thresholds_path = os.path.join(thresholds_dir, "score_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(
            {
                "theta_high": score_thresholds.theta_high,
                "theta_low": score_thresholds.theta_low,
                "score_fpr": score_thresholds.score_fpr,
                "score_quantile": score_thresholds.score_quantile,
            },
            f,
            indent=2,
        )


def _persist_metadata(thresholds_dir: str, metadata: Dict[str, Any]) -> None:
    metadata_path = os.path.join(thresholds_dir, "thresholds_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
