import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import torch
from torch import Tensor

from sgnsae.experiments.directional_deviation.data import (
    LABEL_TYPE_HIGH,
    LABEL_TYPE_LOW,
    DirectionalDeviationAnomalyBatch,
)
from sgnsae.experiments.directional_deviation.scoring import (
    BaselineEvidenceThresholds,
    ScoreThresholds,
    SignAwareLatentThresholds,
    calculate_baseline_directional_scores,
    calculate_sign_aware_directional_scores,
    extract_detection_signal,
    extract_one_sided_evidence,
)


@dataclass(frozen=True)
class DirectionalDeviationEvaluationInputs:
    model: torch.nn.Module
    model_name: str
    scoring_mode: Literal["sign_aware", "baseline"]
    anomaly_data: Dict[str, Any]
    x_test_normal: Tensor
    score_thresholds: ScoreThresholds
    evaluation_batch_size: int
    coverage_recall_target: float
    latent_thresholds: Optional[SignAwareLatentThresholds] = None
    baseline_thresholds: Optional[BaselineEvidenceThresholds] = None


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]; received {value}.")


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}.")


def _iterate_tensor_batches(x: Tensor, batch_size: int) -> Iterator[Tensor]:
    _validate_positive_int(batch_size, "batch_size")
    for start in range(0, x.shape[0], batch_size):
        yield x[start : start + batch_size]


def _materialized_anomaly_batches(
    x_test_anomaly: Tensor,
    labels_type: Tensor,
    labels_channel: Tensor,
    labels_mask: Optional[Tensor],
    batch_size: int,
) -> Iterator[DirectionalDeviationAnomalyBatch]:
    _validate_positive_int(batch_size, "batch_size")
    for start in range(0, x_test_anomaly.shape[0], batch_size):
        end = start + batch_size
        yield DirectionalDeviationAnomalyBatch(
            x=x_test_anomaly[start:end],
            labels_type=labels_type[start:end],
            labels_channel=labels_channel[start:end],
            labels_mask=None if labels_mask is None else labels_mask[start:end],
        )


def _iterate_anomaly_batches(
    anomaly_data: Dict[str, Any],
    batch_size: int,
) -> Iterator[DirectionalDeviationAnomalyBatch]:
    if "x_test_anomaly" in anomaly_data:
        labels_mask = anomaly_data.get("labels_mask")
        return _materialized_anomaly_batches(
            x_test_anomaly=anomaly_data["x_test_anomaly"],
            labels_type=anomaly_data["labels_type"],
            labels_channel=anomaly_data["labels_channel"],
            labels_mask=labels_mask,
            batch_size=batch_size,
        )

    anomaly_stream = anomaly_data.get("anomaly_stream")
    if anomaly_stream is None:
        raise ValueError(
            "Expected anomaly_data to contain either x_test_anomaly or anomaly_stream."
        )
    return iter(anomaly_stream)


def _compute_directional_scores(
    model: torch.nn.Module,
    scoring_mode: Literal["sign_aware", "baseline"],
    x: Tensor,
    latent_thresholds: Optional[SignAwareLatentThresholds],
    baseline_thresholds: Optional[BaselineEvidenceThresholds],
) -> Tuple[Tensor, Tensor]:
    if scoring_mode == "sign_aware":
        if latent_thresholds is None:
            raise ValueError(
                "latent_thresholds must be provided for sign_aware scoring."
            )
        pi = extract_detection_signal(model, x)
        return calculate_sign_aware_directional_scores(
            pi=pi, latent_thresholds=latent_thresholds
        )
    if scoring_mode == "baseline":
        if baseline_thresholds is None:
            raise ValueError(
                "baseline_thresholds must be provided for baseline scoring."
            )
        evidence = extract_one_sided_evidence(model, x)
        return calculate_baseline_directional_scores(
            one_sided_evidence=evidence,
            baseline_thresholds=baseline_thresholds,
        )
    raise ValueError(f"Unsupported scoring_mode: {scoring_mode}.")


def _calculate_flag_rates_for_normal_set(
    model: torch.nn.Module,
    scoring_mode: Literal["sign_aware", "baseline"],
    x_test_normal: Tensor,
    score_thresholds: ScoreThresholds,
    latent_thresholds: Optional[SignAwareLatentThresholds],
    baseline_thresholds: Optional[BaselineEvidenceThresholds],
    batch_size: int,
) -> Tuple[Optional[float], Optional[float]]:
    num_normals = int(x_test_normal.shape[0])
    if num_normals == 0:
        return None, None

    total_flagged_high = 0
    total_flagged_low = 0
    for batch in _iterate_tensor_batches(x_test_normal, batch_size=batch_size):
        score_high, score_low = _compute_directional_scores(
            model=model,
            scoring_mode=scoring_mode,
            x=batch,
            latent_thresholds=latent_thresholds,
            baseline_thresholds=baseline_thresholds,
        )
        total_flagged_high += int(
            (score_high > score_thresholds.theta_high).sum().item()
        )
        total_flagged_low += int((score_low > score_thresholds.theta_low).sum().item())

    return total_flagged_high / num_normals, total_flagged_low / num_normals


def evaluate_directional_deviation(
    model: torch.nn.Module,
    model_name: str,
    scoring_mode: Literal["sign_aware", "baseline"],
    anomaly_data: Dict[str, Any],
    x_test_normal: Tensor,
    score_thresholds: ScoreThresholds,
    latent_thresholds: Optional[SignAwareLatentThresholds] = None,
    baseline_thresholds: Optional[BaselineEvidenceThresholds] = None,
    evaluation_batch_size: int = 4096,
    coverage_recall_target: float = 0.8,
) -> Dict[str, Any]:
    _validate_positive_int(evaluation_batch_size, "evaluation_batch_size")
    _validate_probability(coverage_recall_target, "coverage_recall_target")

    if "anomaly_manifest" not in anomaly_data:
        raise ValueError("anomaly_data must include anomaly_manifest.")
    anomaly_manifest = anomaly_data["anomaly_manifest"]
    if not isinstance(anomaly_manifest, dict):
        raise ValueError("anomaly_manifest must be a dict.")

    anomaly_batches = _iterate_anomaly_batches(
        anomaly_data, batch_size=evaluation_batch_size
    )
    total_high = 0
    detected_high = 0
    total_low = 0
    detected_low = 0
    total_high_by_channel: Optional[Tensor] = None
    detected_high_by_channel: Optional[Tensor] = None
    total_low_by_channel: Optional[Tensor] = None
    detected_low_by_channel: Optional[Tensor] = None

    shift_mode = anomaly_manifest.get("shift_mode")
    per_channel_enabled = shift_mode == "single_channel"
    per_direction_enabled = shift_mode == "feature_direction"
    num_channels = int(anomaly_manifest.get("num_channels", x_test_normal.shape[1]))
    if per_channel_enabled:
        total_high_by_channel = torch.zeros(num_channels, dtype=torch.int64)
        detected_high_by_channel = torch.zeros(num_channels, dtype=torch.int64)
        total_low_by_channel = torch.zeros(num_channels, dtype=torch.int64)
        detected_low_by_channel = torch.zeros(num_channels, dtype=torch.int64)
    total_high_by_direction: Optional[Tensor] = None
    detected_high_by_direction: Optional[Tensor] = None
    total_low_by_direction: Optional[Tensor] = None
    detected_low_by_direction: Optional[Tensor] = None
    num_directions = int(anomaly_manifest.get("num_directions", 0))
    if per_direction_enabled:
        if num_directions <= 0:
            raise ValueError(
                "anomaly_manifest must include a positive num_directions when shift_mode='feature_direction'."
            )
        total_high_by_direction = torch.zeros(num_directions, dtype=torch.int64)
        detected_high_by_direction = torch.zeros(num_directions, dtype=torch.int64)
        total_low_by_direction = torch.zeros(num_directions, dtype=torch.int64)
        detected_low_by_direction = torch.zeros(num_directions, dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for batch in anomaly_batches:
            score_high, score_low = _compute_directional_scores(
                model=model,
                scoring_mode=scoring_mode,
                x=batch.x,
                latent_thresholds=latent_thresholds,
                baseline_thresholds=baseline_thresholds,
            )
            predicted_high = score_high > score_thresholds.theta_high
            predicted_low = score_low > score_thresholds.theta_low

            high_mask = batch.labels_type == LABEL_TYPE_HIGH
            low_mask = batch.labels_type == LABEL_TYPE_LOW

            total_high += int(high_mask.sum().item())
            detected_high += int((predicted_high & high_mask).sum().item())
            total_low += int(low_mask.sum().item())
            detected_low += int((predicted_low & low_mask).sum().item())

            if per_channel_enabled:
                assert total_high_by_channel is not None
                assert detected_high_by_channel is not None
                assert total_low_by_channel is not None
                assert detected_low_by_channel is not None
                if torch.any(batch.labels_channel < 0):
                    raise ValueError(
                        "Per-channel evaluation requires non-negative labels_channel entries."
                    )

                if torch.any(high_mask):
                    channels_high = batch.labels_channel[high_mask].to("cpu")
                    channels_high_detected = batch.labels_channel[
                        predicted_high & high_mask
                    ].to("cpu")
                    total_high_by_channel += torch.bincount(
                        channels_high, minlength=num_channels
                    )
                    detected_high_by_channel += torch.bincount(
                        channels_high_detected, minlength=num_channels
                    )

                if torch.any(low_mask):
                    channels_low = batch.labels_channel[low_mask].to("cpu")
                    channels_low_detected = batch.labels_channel[
                        predicted_low & low_mask
                    ].to("cpu")
                    total_low_by_channel += torch.bincount(
                        channels_low, minlength=num_channels
                    )
                    detected_low_by_channel += torch.bincount(
                        channels_low_detected, minlength=num_channels
                    )

            if per_direction_enabled:
                assert total_high_by_direction is not None
                assert detected_high_by_direction is not None
                assert total_low_by_direction is not None
                assert detected_low_by_direction is not None
                if torch.any(batch.labels_channel < 0):
                    raise ValueError(
                        "Per-direction evaluation requires non-negative labels_channel entries."
                    )

                if torch.any(high_mask):
                    directions_high = batch.labels_channel[high_mask].to("cpu")
                    directions_high_detected = batch.labels_channel[
                        predicted_high & high_mask
                    ].to("cpu")
                    total_high_by_direction += torch.bincount(
                        directions_high, minlength=num_directions
                    )
                    detected_high_by_direction += torch.bincount(
                        directions_high_detected, minlength=num_directions
                    )

                if torch.any(low_mask):
                    directions_low = batch.labels_channel[low_mask].to("cpu")
                    directions_low_detected = batch.labels_channel[
                        predicted_low & low_mask
                    ].to("cpu")
                    total_low_by_direction += torch.bincount(
                        directions_low, minlength=num_directions
                    )
                    detected_low_by_direction += torch.bincount(
                        directions_low_detected, minlength=num_directions
                    )

    recall_high = None if total_high == 0 else float(detected_high) / total_high
    recall_low = None if total_low == 0 else float(detected_low) / total_low

    fpr_high, fpr_low = _calculate_flag_rates_for_normal_set(
        model=model,
        scoring_mode=scoring_mode,
        x_test_normal=x_test_normal,
        score_thresholds=score_thresholds,
        latent_thresholds=latent_thresholds,
        baseline_thresholds=baseline_thresholds,
        batch_size=evaluation_batch_size,
    )

    per_channel: Optional[Dict[str, Any]] = None
    if per_channel_enabled:
        channels_to_evaluate = anomaly_manifest.get("channels_to_evaluate")
        if channels_to_evaluate is None:
            raise ValueError("anomaly_manifest is missing channels_to_evaluate.")
        if not isinstance(channels_to_evaluate, list):
            raise ValueError("channels_to_evaluate must be a list.")

        assert total_high_by_channel is not None
        assert detected_high_by_channel is not None
        assert total_low_by_channel is not None
        assert detected_low_by_channel is not None

        channels_to_evaluate_ints = [int(value) for value in channels_to_evaluate]
        per_channel_recall_high: List[Optional[float]] = []
        per_channel_recall_low: List[Optional[float]] = []
        per_channel_num_high: List[int] = []
        per_channel_num_low: List[int] = []
        covered_channels = 0

        for channel in channels_to_evaluate_ints:
            num_high = int(total_high_by_channel[channel].item())
            num_low = int(total_low_by_channel[channel].item())
            per_channel_num_high.append(num_high)
            per_channel_num_low.append(num_low)

            recall_high_channel = (
                None
                if num_high == 0
                else float(detected_high_by_channel[channel].item()) / num_high
            )
            recall_low_channel = (
                None
                if num_low == 0
                else float(detected_low_by_channel[channel].item()) / num_low
            )
            per_channel_recall_high.append(recall_high_channel)
            per_channel_recall_low.append(recall_low_channel)

            if (
                recall_high_channel is not None
                and recall_low_channel is not None
                and recall_high_channel >= coverage_recall_target
                and recall_low_channel >= coverage_recall_target
            ):
                covered_channels += 1

        covered_fraction = (
            None
            if not channels_to_evaluate_ints
            else float(covered_channels) / len(channels_to_evaluate_ints)
        )
        per_channel = {
            "channels": channels_to_evaluate_ints,
            "recall_high": per_channel_recall_high,
            "recall_low": per_channel_recall_low,
            "num_high": per_channel_num_high,
            "num_low": per_channel_num_low,
            "coverage_recall_target": coverage_recall_target,
            "covered_channels_count": covered_channels,
            "covered_channels_fraction": covered_fraction,
        }

    per_direction: Optional[Dict[str, Any]] = None
    if per_direction_enabled:
        direction_indices_raw = anomaly_manifest.get("direction_indices_to_evaluate")
        direction_indices = (
            list(range(num_directions))
            if direction_indices_raw is None
            else [int(value) for value in direction_indices_raw]
        )
        if not direction_indices:
            raise ValueError("direction_indices_to_evaluate must be non-empty.")

        assert total_high_by_direction is not None
        assert detected_high_by_direction is not None
        assert total_low_by_direction is not None
        assert detected_low_by_direction is not None

        per_direction_recall_high: List[Optional[float]] = []
        per_direction_recall_low: List[Optional[float]] = []
        per_direction_num_high: List[int] = []
        per_direction_num_low: List[int] = []
        covered_directions = 0

        for direction_index in direction_indices:
            num_high = int(total_high_by_direction[direction_index].item())
            num_low = int(total_low_by_direction[direction_index].item())
            per_direction_num_high.append(num_high)
            per_direction_num_low.append(num_low)

            recall_high_direction = (
                None
                if num_high == 0
                else float(detected_high_by_direction[direction_index].item())
                / num_high
            )
            recall_low_direction = (
                None
                if num_low == 0
                else float(detected_low_by_direction[direction_index].item()) / num_low
            )
            per_direction_recall_high.append(recall_high_direction)
            per_direction_recall_low.append(recall_low_direction)

            if (
                recall_high_direction is not None
                and recall_low_direction is not None
                and recall_high_direction >= coverage_recall_target
                and recall_low_direction >= coverage_recall_target
            ):
                covered_directions += 1

        covered_fraction = float(covered_directions) / len(direction_indices)
        per_direction = {
            "directions": direction_indices,
            "recall_high": per_direction_recall_high,
            "recall_low": per_direction_recall_low,
            "num_high": per_direction_num_high,
            "num_low": per_direction_num_low,
            "coverage_recall_target": coverage_recall_target,
            "covered_directions_count": covered_directions,
            "covered_directions_fraction": covered_fraction,
        }

    target_kind = (
        "channel"
        if per_channel_enabled
        else "direction" if per_direction_enabled else None
    )
    num_targets: Optional[int] = None
    if per_channel is not None:
        num_targets = len(per_channel.get("channels", []))
    if per_direction is not None:
        num_targets = len(per_direction.get("directions", []))

    return {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "scoring_mode": scoring_mode,
        "target_kind": target_kind,
        "num_targets": num_targets,
        "metrics": {
            "recall_high": recall_high,
            "recall_low": recall_low,
            "fpr_high": fpr_high,
            "fpr_low": fpr_low,
            "num_high": total_high,
            "num_low": total_low,
            "num_test_normal": int(x_test_normal.shape[0]),
            "theta_high": score_thresholds.theta_high,
            "theta_low": score_thresholds.theta_low,
            "score_fpr": score_thresholds.score_fpr,
            "score_quantile": score_thresholds.score_quantile,
        },
        "per_channel": per_channel,
        "per_direction": per_direction,
        "error": None,
    }


def evaluate_and_persist_directional_deviation(
    model_dir: str,
    inputs: DirectionalDeviationEvaluationInputs,
) -> Dict[str, Any]:
    os.makedirs(model_dir, exist_ok=True)
    evaluation_path = os.path.join(model_dir, "evaluation.json")
    try:
        evaluation = evaluate_directional_deviation(
            model=inputs.model,
            model_name=inputs.model_name,
            scoring_mode=inputs.scoring_mode,
            anomaly_data=inputs.anomaly_data,
            x_test_normal=inputs.x_test_normal,
            score_thresholds=inputs.score_thresholds,
            latent_thresholds=inputs.latent_thresholds,
            baseline_thresholds=inputs.baseline_thresholds,
            evaluation_batch_size=inputs.evaluation_batch_size,
            coverage_recall_target=inputs.coverage_recall_target,
        )
    except Exception as exc:
        evaluation = {
            "model_name": inputs.model_name,
            "model_type": type(inputs.model).__name__,
            "scoring_mode": inputs.scoring_mode,
            "target_kind": None,
            "num_targets": None,
            "metrics": {
                "recall_high": None,
                "recall_low": None,
                "fpr_high": None,
                "fpr_low": None,
                "num_high": None,
                "num_low": None,
                "num_test_normal": int(inputs.x_test_normal.shape[0]),
                "theta_high": inputs.score_thresholds.theta_high,
                "theta_low": inputs.score_thresholds.theta_low,
                "score_fpr": inputs.score_thresholds.score_fpr,
                "score_quantile": inputs.score_thresholds.score_quantile,
            },
            "per_channel": None,
            "per_direction": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    return evaluation


def build_comparison_summary(
    evaluations_by_model_name: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    target_kind: Optional[str] = None
    num_targets: Optional[int] = None
    for model_name, evaluation in evaluations_by_model_name.items():
        metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
        per_channel = (
            evaluation.get("per_channel") if isinstance(evaluation, dict) else None
        )
        per_direction = (
            evaluation.get("per_direction") if isinstance(evaluation, dict) else None
        )
        if target_kind is None and isinstance(evaluation, dict):
            target_kind_value = evaluation.get("target_kind")
            target_kind = (
                str(target_kind_value) if target_kind_value is not None else None
            )
        if num_targets is None and isinstance(evaluation, dict):
            num_targets_value = evaluation.get("num_targets")
            num_targets = (
                int(num_targets_value) if num_targets_value is not None else None
            )

        per_channel_fraction = (
            None
            if not isinstance(per_channel, dict)
            else per_channel.get("covered_channels_fraction")
        )
        per_channel_count = (
            None
            if not isinstance(per_channel, dict)
            else per_channel.get("covered_channels_count")
        )
        per_direction_fraction = (
            None
            if not isinstance(per_direction, dict)
            else per_direction.get("covered_directions_fraction")
        )
        per_direction_count = (
            None
            if not isinstance(per_direction, dict)
            else per_direction.get("covered_directions_count")
        )
        models[model_name] = {
            "recall_high": (
                None if not isinstance(metrics, dict) else metrics.get("recall_high")
            ),
            "recall_low": (
                None if not isinstance(metrics, dict) else metrics.get("recall_low")
            ),
            "fpr_high": (
                None if not isinstance(metrics, dict) else metrics.get("fpr_high")
            ),
            "fpr_low": (
                None if not isinstance(metrics, dict) else metrics.get("fpr_low")
            ),
            "target_kind": (
                evaluation.get("target_kind") if isinstance(evaluation, dict) else None
            ),
            "num_targets": (
                evaluation.get("num_targets") if isinstance(evaluation, dict) else None
            ),
            "covered_channels_fraction": per_channel_fraction,
            "covered_channels_count": per_channel_count,
            "covered_directions_fraction": per_direction_fraction,
            "covered_directions_count": per_direction_count,
            "error": evaluation.get("error") if isinstance(evaluation, dict) else None,
        }

    targets_summary = (
        None
        if target_kind is None
        else {"kind": target_kind, "num_targets": num_targets}
    )
    return {"targets": targets_summary, "models": models}


def persist_comparison_summary(
    run_dir: str,
    evaluations_by_model_name: Dict[str, Dict[str, Any]],
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    summary = build_comparison_summary(evaluations_by_model_name)
    summary_path = os.path.join(run_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
