import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig
from sgnsae.experiments.llm_benchmark.dataset import create_activation_dataset
from sgnsae.experiments.llm_benchmark.feature_ablation import (
    compute_and_write_feature_ablation,
)
from sgnsae.experiments.llm_benchmark.loss_recovered import (
    compute_and_write_subrun_loss_recovered,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    COMPARISON_SUMMARY_SCHEMA_VERSION,
    EVALUATION_SCHEMA_VERSION,
    GAMMA_METRICS_SCHEMA_VERSION,
)
from sgnsae.experiments.llm_benchmark.subrun_state import (
    legacy_migration_error,
    subrun_training_state,
)
from sgnsae.experiments.llm_benchmark.training import (
    build_model_for_subrun,
    load_activation_manifest,
    resolve_manifest_path,
    resolve_runtime_device,
)


_EVAL_ARTIFACT_FILENAMES = {
    "evaluation": "evaluation.json",
    "gamma_metrics": "gamma_metrics.json",
}


def run_level_matched_points_path(output_dir: str) -> str:
    return os.path.join(output_dir, "matched_points.json")


def eval_artifact_paths(
    subrun_dir: str,
    config: Optional[LLMBenchmarkConfig] = None,
) -> Dict[str, str]:
    output_paths = {
        key: os.path.join(subrun_dir, filename)
        for key, filename in _EVAL_ARTIFACT_FILENAMES.items()
    }
    if config is not None and bool(config.loss_recovered.enabled):
        output_paths["loss_recovered"] = os.path.join(subrun_dir, "loss_recovered.json")
    if config is not None and bool(config.feature_ablation.enabled):
        output_paths["feature_ablation"] = os.path.join(
            subrun_dir, "feature_ablation.json"
        )
    return output_paths


def _delete_file_if_exists(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def _resolve_reconstruction_output(output: Dict[str, Tensor]) -> Tensor:
    if "x_hat" in output:
        return output["x_hat"]
    if "y_pred" in output:
        return output["y_pred"]
    raise ValueError(
        "Model forward output must include reconstruction key 'x_hat' or 'y_pred'."
    )


def _flatten_single_instance(values: Tensor) -> Tensor:
    if values.ndim == 3 and int(values.shape[0]) == 1:
        return values[0]
    return values.reshape(-1, values.shape[-1])


def _summary_statistics(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "num_valid": 0,
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "p10": float(torch.quantile(tensor, 0.1).item()),
        "p90": float(torch.quantile(tensor, 0.9).item()),
        "num_valid": int(tensor.numel()),
    }


def _compute_shared_projection_t(model: torch.nn.Module, x_batch: Tensor) -> Tensor:
    x_instances = x_batch.unsqueeze(0)
    w_dec_norm = model.W_dec / (model.W_dec.norm(dim=1, keepdim=True) + 1e-8)
    x_centered = x_instances - model.b_dec.unsqueeze(1)
    return torch.einsum("nbd,ndh->nbh", x_centered, w_dec_norm)


def _to_optional_float_list(
    values: Tensor, valid_mask: Tensor
) -> List[Optional[float]]:
    flattened_values = values.reshape(-1).to(dtype=torch.float64)
    flattened_mask = valid_mask.reshape(-1)
    output: List[Optional[float]] = []
    for index in range(flattened_values.numel()):
        if bool(flattened_mask[index]):
            output.append(float(flattened_values[index].item()))
        else:
            output.append(None)
    return output


def run_evaluation_for_subrun(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    output_dir: str,
    overwrite: bool,
) -> Dict[str, Any]:
    relative_subrun_dir = str(subrun["relative_subrun_dir"])
    subrun_dir = os.path.join(output_dir, relative_subrun_dir.replace("/", os.sep))
    os.makedirs(subrun_dir, exist_ok=True)
    artifact_paths = eval_artifact_paths(subrun_dir, config=config)
    if overwrite:
        for path in artifact_paths.values():
            _delete_file_if_exists(path)

    should_compute_core_metrics = overwrite or (
        not os.path.isfile(artifact_paths["evaluation"])
        or not os.path.isfile(artifact_paths["gamma_metrics"])
    )
    should_compute_loss_recovered = bool(config.loss_recovered.enabled) and (
        overwrite or not os.path.isfile(artifact_paths["loss_recovered"])
    )
    should_compute_feature_ablation = bool(config.feature_ablation.enabled) and (
        overwrite or not os.path.isfile(artifact_paths["feature_ablation"])
    )
    if not (
        should_compute_core_metrics
        or should_compute_loss_recovered
        or should_compute_feature_ablation
    ):
        return {
            "subrun_dir": subrun_dir,
            "artifact_paths": artifact_paths,
            "reused_all": True,
        }

    checkpoint_path = os.path.join(subrun_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(
            f"Missing trained checkpoint for evaluation: {checkpoint_path}."
        )

    manifest_path = resolve_manifest_path(output_dir=output_dir)
    manifest = load_activation_manifest(manifest_path=manifest_path)
    runtime_device = resolve_runtime_device(config.device)
    model, _, _, variant_config = build_model_for_subrun(
        config=config,
        resolved_config=resolved_config,
        subrun=subrun,
        manifest=manifest,
        device=runtime_device,
    )
    checkpoint_payload = torch.load(checkpoint_path, map_location=runtime_device)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"Invalid checkpoint payload at {checkpoint_path}.")
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a 'state_dict' dictionary."
        )
    model.load_state_dict(state_dict)
    model = model.to(runtime_device)
    model.eval()

    if should_compute_core_metrics:
        val_dataset = create_activation_dataset(
            manifest_path=manifest_path,
            split="val",
            batch_size=int(config.training.batch_size),
            shuffle=False,
            seed=int(subrun["seed"]),
            max_train_samples=config.training.max_train_samples,
            max_val_samples=config.training.max_val_samples,
            max_test_samples=config.training.max_test_samples,
        )

        d_hidden = int(variant_config["d_hidden"])
        num_instances = int(getattr(model, "n_instances", 1))
        alive_threshold = float(config.evaluation.alive_threshold)
        alive_min_fraction = float(config.evaluation.alive_min_fraction)
        min_gamma_samples = int(config.evaluation.min_gamma_samples)
        gamma_denominator_epsilon = float(config.evaluation.gamma_denominator_epsilon)
        r2_denominator_epsilon = float(config.evaluation.r2_denominator_epsilon)

        sse = 0.0
        sum_x = 0.0
        sum_x2 = 0.0
        count_elements = 0
        l0_total = 0.0
        l1_total = 0.0
        l0_sample_count = 0
        num_val_vectors = 0

        active_count = torch.zeros((num_instances, d_hidden), dtype=torch.float64)
        pos_count = torch.zeros((num_instances, d_hidden), dtype=torch.float64)
        neg_count = torch.zeros((num_instances, d_hidden), dtype=torch.float64)

        count_pos = torch.zeros((num_instances, d_hidden), dtype=torch.float64)
        sum_a2_pos = torch.zeros((num_instances, d_hidden), dtype=torch.float64)
        sum_at_pos = torch.zeros((num_instances, d_hidden), dtype=torch.float64)
        count_neg = torch.zeros((num_instances, d_hidden), dtype=torch.float64)
        sum_a2_neg = torch.zeros((num_instances, d_hidden), dtype=torch.float64)
        sum_at_neg = torch.zeros((num_instances, d_hidden), dtype=torch.float64)

        signed_latents = bool(
            bool(variant_config["sign_aware"])
            or bool(getattr(model, "signed_latents", False))
        )

        with torch.no_grad():
            for batch in val_dataset:
                x_batch = batch.to(device=runtime_device, dtype=torch.float32)
                output = model(x_batch)
                latents = output["latents"]
                reconstruction = _resolve_reconstruction_output(output)
                y_true = x_batch.unsqueeze(0)
                residual = (reconstruction - y_true).to(dtype=torch.float64)
                sse += float(residual.pow(2).sum().item())
                count_elements += int(residual.numel())
                y_true_f64 = y_true.to(dtype=torch.float64)
                sum_x += float(y_true_f64.sum().item())
                sum_x2 += float(y_true_f64.pow(2).sum().item())

                if bool(config.evaluation.l0_is_nonzero):
                    l0_per_sample = (latents != 0).sum(dim=-1).to(dtype=torch.float64)
                else:
                    l0_per_sample = (
                        (latents.abs() > 1e-8).sum(dim=-1).to(dtype=torch.float64)
                    )
                l1_per_sample = latents.abs().sum(dim=-1).to(dtype=torch.float64)
                l0_total += float(l0_per_sample.sum().item())
                l1_total += float(l1_per_sample.sum().item())
                l0_sample_count += int(l0_per_sample.numel())
                num_val_vectors += int(x_batch.shape[0])

                active_count += (
                    (latents.abs() > alive_threshold)
                    .sum(dim=1)
                    .to(dtype=torch.float64, device="cpu")
                )
                if signed_latents:
                    pos_count += (
                        (latents > 0).sum(dim=1).to(dtype=torch.float64, device="cpu")
                    )
                    neg_count += (
                        (latents < 0).sum(dim=1).to(dtype=torch.float64, device="cpu")
                    )

                t_values = _compute_shared_projection_t(model, x_batch).to(
                    dtype=torch.float64
                )
                latents_f64 = latents.to(dtype=torch.float64)
                pos_mask = latents_f64 > 0
                count_pos += pos_mask.sum(dim=1).to(dtype=torch.float64, device="cpu")
                sum_a2_pos += (
                    (latents_f64.pow(2) * pos_mask)
                    .sum(dim=1)
                    .to(dtype=torch.float64, device="cpu")
                )
                sum_at_pos += (
                    (latents_f64 * t_values * pos_mask)
                    .sum(dim=1)
                    .to(dtype=torch.float64, device="cpu")
                )
                if signed_latents:
                    neg_mask = latents_f64 < 0
                    count_neg += neg_mask.sum(dim=1).to(
                        dtype=torch.float64, device="cpu"
                    )
                    sum_a2_neg += (
                        (latents_f64.pow(2) * neg_mask)
                        .sum(dim=1)
                        .to(dtype=torch.float64, device="cpu")
                    )
                    sum_at_neg += (
                        (latents_f64 * t_values * neg_mask)
                        .sum(dim=1)
                        .to(dtype=torch.float64, device="cpu")
                    )

        if count_elements == 0 or l0_sample_count == 0 or num_val_vectors == 0:
            raise ValueError(
                "Validation split produced zero samples. Adjust split fractions or "
                "max_val_samples."
            )

        mse = float(sse / float(count_elements))
        sst = float(sum_x2 - (sum_x * sum_x / float(count_elements)))
        if sst <= r2_denominator_epsilon:
            r2: Optional[float] = None
        else:
            r2 = float(1.0 - sse / sst)
        mean_l0 = float(l0_total / float(l0_sample_count))
        mean_l1 = float(l1_total / float(l0_sample_count))

        activation_fraction = active_count / float(num_val_vectors)
        alive_mask = activation_fraction >= alive_min_fraction
        rare_mask = (active_count > 0.0) & (~alive_mask)
        dead_mask = ~alive_mask
        alive_count_value = int(alive_mask.sum().item())
        dead_count_value = int(dead_mask.sum().item())
        rare_count_value = int(rare_mask.sum().item())
        total_latents = int(alive_mask.numel())
        alive_indices: Optional[List[int]] = None
        if int(alive_mask.shape[0]) == 1:
            alive_indices = [
                int(index)
                for index in torch.nonzero(alive_mask[0], as_tuple=False)
                .reshape(-1)
                .tolist()
            ]

        gamma_pos_valid = (count_pos >= float(min_gamma_samples)) & (
            sum_a2_pos > gamma_denominator_epsilon
        )
        gamma_pos_values = torch.where(
            gamma_pos_valid,
            sum_at_pos / sum_a2_pos,
            torch.zeros_like(sum_at_pos),
        )
        gamma_pos_list = _to_optional_float_list(gamma_pos_values, gamma_pos_valid)
        gamma_pos_summary = _summary_statistics(
            [value for value in gamma_pos_list if value is not None]
        )
        pos_fraction_valid = float(
            gamma_pos_valid.sum().item() / float(gamma_pos_valid.numel())
        )

        gamma_neg_list: Optional[List[Optional[float]]] = None
        gamma_neg_summary: Optional[Dict[str, Any]] = None
        neg_fraction_valid: Optional[float] = None
        both_fraction_valid: Optional[float] = None
        if signed_latents:
            gamma_neg_valid = (count_neg >= float(min_gamma_samples)) & (
                sum_a2_neg > gamma_denominator_epsilon
            )
            gamma_neg_values = torch.where(
                gamma_neg_valid,
                sum_at_neg / sum_a2_neg,
                torch.zeros_like(sum_at_neg),
            )
            gamma_neg_list = _to_optional_float_list(gamma_neg_values, gamma_neg_valid)
            gamma_neg_summary = _summary_statistics(
                [value for value in gamma_neg_list if value is not None]
            )
            neg_fraction_valid = float(
                gamma_neg_valid.sum().item() / float(gamma_neg_valid.numel())
            )
            both_fraction_valid = float(
                (gamma_pos_valid & gamma_neg_valid).sum().item()
                / float(gamma_pos_valid.numel())
            )

        polarity_imbalance_by_latent: Optional[List[Optional[float]]] = None
        polarity_imbalance_summary: Optional[Dict[str, Any]] = None
        if signed_latents:
            active_signed_count = pos_count + neg_count
            valid_polarity_mask = active_signed_count > 0.0
            safe_denominator = torch.where(
                valid_polarity_mask,
                active_signed_count,
                torch.ones_like(active_signed_count),
            )
            pos_fraction = pos_count / safe_denominator
            imbalance = (2.0 * pos_fraction - 1.0).abs()
            polarity_imbalance_by_latent = _to_optional_float_list(
                imbalance, valid_polarity_mask
            )
            polarity_imbalance_summary = _summary_statistics(
                [value for value in polarity_imbalance_by_latent if value is not None]
            )

        gamma_summary = {
            "pos": gamma_pos_summary,
            "neg": gamma_neg_summary,
            "coverage": {
                "pos_fraction_valid": pos_fraction_valid,
                "neg_fraction_valid": neg_fraction_valid,
                "both_fraction_valid": both_fraction_valid,
            },
        }
        gamma_metrics_payload: Dict[str, Any] = {
            "schema_version": GAMMA_METRICS_SCHEMA_VERSION,
            "min_samples": int(min_gamma_samples),
            "gamma_denominator_epsilon": float(gamma_denominator_epsilon),
            "gamma_pos_values": gamma_pos_list,
            "summary": gamma_summary,
        }
        if gamma_neg_list is not None:
            gamma_metrics_payload["gamma_neg_values"] = gamma_neg_list

        evaluation_payload: Dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "variant_name": str(subrun["variant_name"]),
            "variant_slug": str(subrun["variant_slug"]),
            "seed": int(subrun["seed"]),
            "sweep_param": str(subrun["sweep_param"]),
            "sweep_value": subrun["sweep_value"],
            "num_val_samples": int(num_val_vectors),
            "mse": mse,
            "r2": r2,
            "mean_l0": mean_l0,
            "mean_l1": mean_l1,
            "alive_threshold": alive_threshold,
            "alive_min_fraction": alive_min_fraction,
            "alive_count": alive_count_value,
            "dead_count": dead_count_value,
            "alive_fraction": float(alive_count_value / float(total_latents)),
            "dead_fraction": float(dead_count_value / float(total_latents)),
            "rare_count": rare_count_value,
            "rare_fraction": float(rare_count_value / float(total_latents)),
            "gamma_summary": gamma_summary,
        }
        if alive_indices is not None:
            evaluation_payload["alive_indices"] = alive_indices
        if polarity_imbalance_summary is not None:
            evaluation_payload["polarity_imbalance_summary"] = (
                polarity_imbalance_summary
            )
        if polarity_imbalance_by_latent is not None:
            evaluation_payload["polarity_imbalance_by_latent"] = (
                polarity_imbalance_by_latent
            )

        with open(artifact_paths["gamma_metrics"], "w") as file:
            json.dump(gamma_metrics_payload, file, indent=2)
        with open(artifact_paths["evaluation"], "w") as file:
            json.dump(evaluation_payload, file, indent=2)

    if should_compute_loss_recovered:
        _ = compute_and_write_subrun_loss_recovered(
            config=config,
            subrun=subrun,
            output_dir=output_dir,
            subrun_dir=subrun_dir,
            sae_model=model,
            overwrite=overwrite,
        )

    if should_compute_feature_ablation:
        _ = compute_and_write_feature_ablation(
            config=config,
            subrun=subrun,
            output_dir=output_dir,
            subrun_dir=subrun_dir,
            model=model,
            overwrite=overwrite,
        )

    return {
        "subrun_dir": subrun_dir,
        "artifact_paths": artifact_paths,
    }


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _safe_standard_error(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    if len(values) == 1:
        return 0.0
    mean_value = float(sum(values) / float(len(values)))
    variance = sum((value - mean_value) ** 2 for value in values) / float(
        len(values) - 1
    )
    return float(math.sqrt(variance) / math.sqrt(float(len(values))))


def build_and_persist_comparison_summary(
    resolved_config: Dict[str, Any], output_dir: str
) -> Dict[str, Any]:
    manifest_path = resolve_manifest_path(output_dir=output_dir)
    manifest = load_activation_manifest(manifest_path=manifest_path)
    hook_config = resolved_config.get("hook", {})
    if not isinstance(hook_config, dict):
        hook_config = {}
    manifest_hook = manifest.get("hook", {})
    if not isinstance(manifest_hook, dict):
        manifest_hook = {}
    hook_layer = manifest_hook.get("layer")
    if hook_layer is None:
        hook_layer = hook_config.get("layer")
    hook_layer_selector = manifest_hook.get("layer_selector")
    if hook_layer_selector is None:
        hook_layer_selector = hook_config.get("layer_selector")
    layer_label = hook_layer if hook_layer is not None else hook_layer_selector
    modality = str(resolved_config.get("modality", manifest.get("modality", "text")))
    dataset_metadata = manifest.get("dataset", {})
    if not isinstance(dataset_metadata, dict):
        dataset_metadata = {}
    tokenization_metadata = manifest.get("tokenization", {})
    if not isinstance(tokenization_metadata, dict):
        tokenization_metadata = {}
    metadata: Dict[str, Any] = {
        "modality": modality,
        "hook_name": manifest_hook.get("hook_name", hook_config.get("hook_name")),
        "layer": layer_label,
        "module_path": manifest_hook.get("module_path_resolved"),
        "dataset_id": dataset_metadata.get("dataset_id"),
        "split": dataset_metadata.get("split"),
    }
    if modality == "image":
        vision = resolved_config.get("vision", {})
        if not isinstance(vision, dict):
            vision = {}
        metadata["vision_model_id"] = vision.get(
            "model_id", manifest.get("vision_model_id")
        )
        metadata["image_size"] = manifest.get("image_size")
        metadata["patch_size"] = manifest.get("patch_size")
        metadata["token_selection"] = manifest.get("token_selection")
    else:
        llm = resolved_config.get("llm", {})
        if not isinstance(llm, dict):
            llm = {}
        metadata["llm_id"] = llm.get("model_id")
        metadata["max_seq_len"] = tokenization_metadata.get("max_seq_len")

    subruns_summary: List[Dict[str, Any]] = []
    aggregate_buckets: Dict[Tuple[str, str, Any], Dict[str, List[float]]] = {}

    for subrun in resolved_config.get("subruns", []):
        relative_dir = str(subrun["relative_subrun_dir"])
        subrun_dir = os.path.join(output_dir, relative_dir.replace("/", os.sep))
        state = subrun_training_state(subrun_dir)
        if state == "legacy_unmigrated":
            raise legacy_migration_error(
                output_dir=output_dir,
                subrun_dir=subrun_dir,
            )
        if state != "completed":
            continue
        evaluation_path = os.path.join(subrun_dir, "evaluation.json")
        if not os.path.isfile(evaluation_path):
            continue
        with open(evaluation_path, "r") as file:
            evaluation = json.load(file)
        gamma_summary = evaluation.get("gamma_summary", {})
        coverage = gamma_summary.get("coverage", {})
        loss_recovered_path = os.path.join(subrun_dir, "loss_recovered.json")
        loss_recovered_payload: Optional[Dict[str, Any]] = None
        if os.path.isfile(loss_recovered_path):
            with open(loss_recovered_path, "r") as file:
                loaded_loss_recovered = json.load(file)
            if isinstance(loaded_loss_recovered, dict):
                loss_recovered_payload = loaded_loss_recovered
        feature_ablation_path = os.path.join(subrun_dir, "feature_ablation.json")
        feature_ablation_payload: Optional[Dict[str, Any]] = None
        if os.path.isfile(feature_ablation_path):
            with open(feature_ablation_path, "r") as file:
                loaded_feature_ablation = json.load(file)
            if isinstance(loaded_feature_ablation, dict):
                feature_ablation_payload = loaded_feature_ablation
        feature_ablation_summary = (
            feature_ablation_payload.get("summary", {})
            if isinstance(feature_ablation_payload, dict)
            else {}
        )
        if not isinstance(feature_ablation_summary, dict):
            feature_ablation_summary = {}
        subrun_entry = {
            "variant_name": str(subrun["variant_name"]),
            "variant_slug": str(subrun["variant_slug"]),
            "seed": int(subrun["seed"]),
            "sweep_param": str(subrun["sweep_param"]),
            "sweep_value": subrun["sweep_value"],
            "relative_dir": relative_dir,
            "mse": evaluation.get("mse"),
            "r2": evaluation.get("r2"),
            "mean_l0": evaluation.get("mean_l0"),
            "dead_fraction": evaluation.get("dead_fraction"),
            "gamma_pos_fraction_valid": coverage.get("pos_fraction_valid"),
            "loss_recovered": (
                None
                if loss_recovered_payload is None
                else loss_recovered_payload.get("loss_recovered")
            ),
            "loss_original": (
                None
                if loss_recovered_payload is None
                else loss_recovered_payload.get("loss_original")
            ),
            "loss_ablated": (
                None
                if loss_recovered_payload is None
                else loss_recovered_payload.get("loss_ablated")
            ),
            "loss_reconstructed": (
                None
                if loss_recovered_payload is None
                else loss_recovered_payload.get("loss_reconstructed")
            ),
            "feature_ablation_mean_delta": feature_ablation_summary.get("mean"),
            "feature_ablation_p90_delta": feature_ablation_summary.get("p90"),
        }
        subruns_summary.append(subrun_entry)

        aggregate_key = (
            str(subrun["variant_slug"]),
            str(subrun["sweep_param"]),
            subrun["sweep_value"],
        )
        if aggregate_key not in aggregate_buckets:
            aggregate_buckets[aggregate_key] = {
                "mse": [],
                "r2": [],
                "mean_l0": [],
                "dead_fraction": [],
                "loss_recovered": [],
                "loss_original": [],
                "loss_ablated": [],
                "loss_reconstructed": [],
                "feature_ablation_mean_delta": [],
                "feature_ablation_p90_delta": [],
            }
        for metric_name in (
            "mse",
            "r2",
            "mean_l0",
            "dead_fraction",
            "loss_recovered",
            "loss_original",
            "loss_ablated",
            "loss_reconstructed",
            "feature_ablation_mean_delta",
            "feature_ablation_p90_delta",
        ):
            metric_value = subrun_entry.get(metric_name)
            if metric_value is None:
                continue
            aggregate_buckets[aggregate_key][metric_name].append(float(metric_value))

    aggregates: List[Dict[str, Any]] = []
    for aggregate_key, metric_values in aggregate_buckets.items():
        variant_slug, sweep_param, sweep_value = aggregate_key
        aggregates.append(
            {
                "variant_slug": variant_slug,
                "sweep_param": sweep_param,
                "sweep_value": sweep_value,
                "num_seeds": len(metric_values["mse"]),
                "mse_mean": _safe_mean(metric_values["mse"]),
                "mse_se": _safe_standard_error(metric_values["mse"]),
                "r2_mean": _safe_mean(metric_values["r2"]),
                "r2_se": _safe_standard_error(metric_values["r2"]),
                "mean_l0_mean": _safe_mean(metric_values["mean_l0"]),
                "mean_l0_se": _safe_standard_error(metric_values["mean_l0"]),
                "dead_fraction_mean": _safe_mean(metric_values["dead_fraction"]),
                "dead_fraction_se": _safe_standard_error(
                    metric_values["dead_fraction"]
                ),
                "loss_recovered_mean": _safe_mean(metric_values["loss_recovered"]),
                "loss_recovered_se": _safe_standard_error(
                    metric_values["loss_recovered"]
                ),
                "num_seeds_loss_recovered": len(metric_values["loss_recovered"]),
                "loss_original_mean": _safe_mean(metric_values["loss_original"]),
                "loss_ablated_mean": _safe_mean(metric_values["loss_ablated"]),
                "loss_reconstructed_mean": _safe_mean(
                    metric_values["loss_reconstructed"]
                ),
                "feature_ablation_mean_delta": _safe_mean(
                    metric_values["feature_ablation_mean_delta"]
                ),
                "feature_ablation_mean_delta_se": _safe_standard_error(
                    metric_values["feature_ablation_mean_delta"]
                ),
                "feature_ablation_p90_delta": _safe_mean(
                    metric_values["feature_ablation_p90_delta"]
                ),
                "num_seeds_feature_ablation": len(
                    metric_values["feature_ablation_mean_delta"]
                ),
            }
        )

    comparison_summary = {
        "schema_version": COMPARISON_SUMMARY_SCHEMA_VERSION,
        "metadata": metadata,
        "subruns": subruns_summary,
        "aggregates": aggregates,
    }
    comparison_path = os.path.join(output_dir, "comparison_summary.json")
    with open(comparison_path, "w") as file:
        json.dump(comparison_summary, file, indent=2)
    return comparison_summary
