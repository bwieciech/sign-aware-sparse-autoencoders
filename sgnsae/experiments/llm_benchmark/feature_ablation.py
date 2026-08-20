import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig
from sgnsae.experiments.llm_benchmark.dataset import create_activation_dataset
from sgnsae.experiments.llm_benchmark.loss_recovered import compute_reconstructed_loss
from sgnsae.experiments.llm_benchmark.schema_versions import (
    FEATURE_ABLATION_SCHEMA_VERSION,
)
from sgnsae.experiments.llm_benchmark.training import resolve_manifest_path


def subrun_feature_ablation_path(subrun_dir: str) -> str:
    return os.path.join(subrun_dir, "feature_ablation.json")


def _resolve_reconstruction_output(output: Dict[str, Tensor]) -> Tensor:
    if "x_hat" in output:
        return output["x_hat"]
    if "y_pred" in output:
        return output["y_pred"]
    raise ValueError(
        "Model forward output must include reconstruction key 'x_hat' or 'y_pred'."
    )


def _summary_statistics(values: List[Optional[float]]) -> Dict[str, Any]:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "num_valid": 0,
        }
    tensor_values = torch.tensor(valid_values, dtype=torch.float64)
    return {
        "mean": float(tensor_values.mean().item()),
        "median": float(torch.quantile(tensor_values, 0.5).item()),
        "p10": float(torch.quantile(tensor_values, 0.1).item()),
        "p90": float(torch.quantile(tensor_values, 0.9).item()),
        "num_valid": int(tensor_values.numel()),
    }


def _iter_val_batches(
    config: LLMBenchmarkConfig,
    subrun: Dict[str, Any],
    output_dir: str,
) -> Tuple[Any, str]:
    manifest_path = resolve_manifest_path(output_dir=output_dir)
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
    return val_dataset, manifest_path


def _select_features_for_ablation(
    config: LLMBenchmarkConfig,
    model: torch.nn.Module,
    subrun: Dict[str, Any],
    output_dir: str,
) -> Tuple[List[int], int, int]:
    val_dataset, _ = _iter_val_batches(
        config=config,
        subrun=subrun,
        output_dir=output_dir,
    )
    max_eval_batches = int(config.feature_ablation.max_eval_batches)
    alive_threshold = float(config.evaluation.alive_threshold)
    alive_min_fraction = float(config.evaluation.alive_min_fraction)
    d_hidden = int(getattr(model, "d_hidden"))
    model_device = next(model.parameters()).device

    sum_abs_latents = torch.zeros(d_hidden, dtype=torch.float64, device=model_device)
    active_count = torch.zeros(d_hidden, dtype=torch.float64, device=model_device)
    num_samples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_dataset):
            if batch_index >= max_eval_batches:
                break
            x_batch = batch.to(device=model_device, dtype=torch.float32)
            output = model(x_batch)
            latents = output["latents"]
            if latents.ndim != 3 or int(latents.shape[0]) != 1:
                raise ValueError(
                    "Feature ablation currently requires latents with shape "
                    "(1, batch, d_hidden)."
                )
            latents_flat = latents[0].to(dtype=torch.float64)
            sum_abs_latents += latents_flat.abs().sum(dim=0)
            active_count += (
                (latents_flat.abs() > alive_threshold)
                .sum(dim=0)
                .to(dtype=torch.float64)
            )
            num_samples += int(latents_flat.shape[0])

    if num_samples <= 0:
        raise ValueError(
            "Feature ablation selection pass observed zero validation samples. "
            "Adjust cache split fractions or max_val_samples."
        )

    alive_fraction = active_count / float(num_samples)
    alive_indices = torch.nonzero(alive_fraction >= alive_min_fraction).reshape(-1)
    candidate_alive_count = int(alive_indices.numel())
    if candidate_alive_count == 0:
        return [], 0, num_samples

    max_features = min(int(config.feature_ablation.num_features), candidate_alive_count)
    if config.feature_ablation.selection == "top_mean_abs":
        mean_abs = sum_abs_latents / float(num_samples)
        alive_scores = mean_abs.index_select(0, alive_indices)
        topk_scores, topk_indices = torch.topk(
            alive_scores, k=max_features, largest=True, sorted=True
        )
        threshold = float(topk_scores[-1].item())
        tied_at_threshold_total = int((alive_scores == threshold).sum().item())
        tied_at_threshold_topk = int((topk_scores == threshold).sum().item())
        if tied_at_threshold_total > tied_at_threshold_topk:
            alive_scores_cpu = alive_scores.to(
                device="cpu", dtype=torch.float64
            ).tolist()
            alive_indices_cpu = alive_indices.to(
                device="cpu", dtype=torch.long
            ).tolist()
            scored_alive = list(zip(alive_scores_cpu, alive_indices_cpu))
            scored_alive.sort(key=lambda value: (-float(value[0]), int(value[1])))
            selected_features = [int(value[1]) for value in scored_alive[:max_features]]
        else:
            topk_features = alive_indices.index_select(0, topk_indices)
            scored_selected = list(
                zip(
                    topk_scores.to(device="cpu", dtype=torch.float64).tolist(),
                    topk_features.to(device="cpu", dtype=torch.long).tolist(),
                )
            )
            scored_selected.sort(key=lambda value: (-float(value[0]), int(value[1])))
            selected_features = [int(value[1]) for value in scored_selected]
    else:
        generator = torch.Generator()
        generator.manual_seed(int(config.feature_ablation.seed))
        permutation = torch.randperm(candidate_alive_count, generator=generator)
        selected = (
            alive_indices.to(device="cpu", dtype=torch.long)
            .index_select(0, permutation[:max_features])
            .tolist()
        )
        selected_features = [int(value) for value in selected]
    return selected_features, candidate_alive_count, num_samples


def _compute_delta_mse_impacts(
    config: LLMBenchmarkConfig,
    model: torch.nn.Module,
    subrun: Dict[str, Any],
    output_dir: str,
    selected_features: List[int],
) -> Tuple[List[float], float, int]:
    if not selected_features:
        return [], 0.0, 0

    val_dataset, _ = _iter_val_batches(
        config=config,
        subrun=subrun,
        output_dir=output_dir,
    )
    max_eval_batches = int(config.feature_ablation.max_eval_batches)
    model_device = next(model.parameters()).device
    selected_feature_tensor = torch.tensor(
        selected_features, dtype=torch.long, device=model_device
    )
    w_dec_norm = model.W_dec / (model.W_dec.norm(dim=1, keepdim=True) + 1e-8)
    d_sub = w_dec_norm.index_select(dim=2, index=selected_feature_tensor)
    d_sub_norm_sq = d_sub.pow(2).sum(dim=1)

    sum_latent2 = torch.zeros(len(selected_features), dtype=torch.float64)
    sum_latent_proj_error = torch.zeros(len(selected_features), dtype=torch.float64)
    total_sse = 0.0
    num_elements_total = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_dataset):
            if batch_index >= max_eval_batches:
                break
            x_batch = batch.to(device=model_device, dtype=torch.float32)
            output = model(x_batch)
            latents = output["latents"]
            reconstruction = _resolve_reconstruction_output(output)
            y_true = x_batch.unsqueeze(0)

            if latents.ndim != 3 or int(latents.shape[0]) != 1:
                raise ValueError(
                    "Feature ablation currently requires latents with shape "
                    "(1, batch, d_hidden)."
                )
            residual = reconstruction - y_true
            total_sse += float(residual.pow(2).sum().item())
            num_elements_total += int(residual.numel())

            selected_latents = latents.index_select(
                dim=2, index=selected_feature_tensor
            )
            projected_error = torch.einsum("nbd,ndk->nbk", residual, d_sub)
            latent2_term = (selected_latents.pow(2) * d_sub_norm_sq.unsqueeze(1)).sum(
                dim=(0, 1)
            )
            cross_term = (selected_latents * projected_error).sum(dim=(0, 1))
            sum_latent2 += latent2_term.to(dtype=torch.float64, device="cpu")
            sum_latent_proj_error += cross_term.to(dtype=torch.float64, device="cpu")

    if num_elements_total <= 0:
        raise ValueError(
            "Feature ablation delta_mse pass observed zero validation elements."
        )

    delta_sse = sum_latent2 - 2.0 * sum_latent_proj_error
    delta_mse = (delta_sse / float(num_elements_total)).tolist()
    baseline_mse = float(total_sse / float(num_elements_total))
    return [float(value) for value in delta_mse], baseline_mse, num_elements_total


def _compute_delta_lm_loss_impacts(
    config: LLMBenchmarkConfig,
    output_dir: str,
    model: torch.nn.Module,
    selected_features: List[int],
) -> Tuple[List[Optional[float]], int]:
    if not selected_features:
        return [], 0
    capped_feature_count = min(
        int(config.feature_ablation.lm_max_features), len(selected_features)
    )
    features_for_lm = selected_features[:capped_feature_count]
    reconstructed_loss, _ = compute_reconstructed_loss(
        config=config,
        output_dir=output_dir,
        sae_model=model,
    )
    impacts: List[Optional[float]] = []
    for feature_index in features_for_lm:
        reconstructed_minus_feature, _ = compute_reconstructed_loss(
            config=config,
            output_dir=output_dir,
            sae_model=model,
            ablated_feature_indices=[int(feature_index)],
        )
        impacts.append(float(reconstructed_minus_feature - reconstructed_loss))
    if len(impacts) < len(selected_features):
        impacts.extend([None] * (len(selected_features) - len(impacts)))
    return impacts, int(capped_feature_count)


def compute_and_write_feature_ablation(
    config: LLMBenchmarkConfig,
    subrun: Dict[str, Any],
    output_dir: str,
    subrun_dir: str,
    model: torch.nn.Module,
    overwrite: bool,
) -> Optional[Dict[str, Any]]:
    if not bool(config.feature_ablation.enabled):
        return None

    output_path = subrun_feature_ablation_path(subrun_dir=subrun_dir)
    if os.path.isfile(output_path) and not overwrite:
        with open(output_path, "r") as file:
            return json.load(file)

    selected_features, candidate_alive_count, num_samples = (
        _select_features_for_ablation(
            config=config,
            model=model,
            subrun=subrun,
            output_dir=output_dir,
        )
    )
    impacts_key = str(config.feature_ablation.metric)
    impacts: List[Optional[float]]
    baseline_mse: Optional[float]
    num_features_evaluated: int
    if config.feature_ablation.metric == "delta_mse":
        impacts, baseline_mse, _ = _compute_delta_mse_impacts(
            config=config,
            model=model,
            subrun=subrun,
            output_dir=output_dir,
            selected_features=selected_features,
        )
        num_features_evaluated = len(impacts)
    else:
        impacts, num_features_evaluated = _compute_delta_lm_loss_impacts(
            config=config,
            output_dir=output_dir,
            model=model,
            selected_features=selected_features,
        )
        baseline_mse = None
    summary = _summary_statistics(values=impacts)

    payload: Dict[str, Any] = {
        "schema_version": FEATURE_ABLATION_SCHEMA_VERSION,
        "variant_name": str(subrun["variant_name"]),
        "variant_slug": str(subrun["variant_slug"]),
        "seed": int(subrun["seed"]),
        "sweep_param": str(subrun["sweep_param"]),
        "sweep_value": subrun["sweep_value"],
        "selection": {
            "method": str(config.feature_ablation.selection),
            "seed": int(config.feature_ablation.seed),
            "num_features": int(config.feature_ablation.num_features),
            "candidate_alive_count": int(candidate_alive_count),
        },
        "eval": {
            "split": "val",
            "max_eval_batches": int(config.feature_ablation.max_eval_batches),
            "num_samples": int(num_samples),
            "baseline_mse": baseline_mse,
            "num_features_evaluated": int(num_features_evaluated),
        },
        "metric": impacts_key,
        "features": selected_features,
        "impacts": {
            impacts_key: impacts,
        },
        "summary": summary,
    }
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)
    return payload
