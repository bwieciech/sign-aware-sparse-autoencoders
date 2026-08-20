import collections
import json
import os
import re
import shutil
import warnings
from contextlib import nullcontext
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import tqdm
from torch import Tensor

from sgnsae.experiments.device import resolve_experiment_device
from sgnsae.experiments.llm_benchmark.config import (
    LLMBenchmarkConfig,
    TrainingConfig,
)
from sgnsae.experiments.llm_benchmark.dataset import create_activation_dataset
from sgnsae.experiments.llm_benchmark.model_registry import (
    LAMBDA_SWEEP_SAE_CLASS_NAMES,
    TOPK_SAE_CLASS_NAMES,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
    TRAINING_METRICS_SCHEMA_VERSION,
)
from sgnsae.experiments.llm_benchmark.subrun_state import (
    is_completed_payload,
    load_training_metrics_if_present,
)
from sgnsae.models import (
    AbsTopKSAE,
    AbsTopKGatedMagnitudeSAE,
    GatedSAE,
    JumpReLUSAE,
    SAE,
    SignedSoftThresholdSAE,
    TopKSAE,
)


_TRAIN_ARTIFACT_FILENAMES = {
    "model": "model.pt",
    "model_config": "model_config.json",
    "training_metrics": "training_metrics.json",
}

_DOWNSTREAM_SUBRUN_ARTIFACT_FILENAMES = (
    "evaluation.json",
    "gamma_metrics.json",
    "loss_recovered.json",
    "feature_ablation.json",
    "duplication_metrics.json",
    "latent_examples.json",
)

_DOWNSTREAM_SUBRUN_ARTIFACT_DIRNAMES = ("latent_examples_assets",)


def train_artifact_paths(subrun_dir: str) -> Dict[str, str]:
    return {
        key: os.path.join(subrun_dir, filename)
        for key, filename in _TRAIN_ARTIFACT_FILENAMES.items()
    }


def resolve_manifest_path(output_dir: str) -> str:
    manifest_path = os.path.join(output_dir, "activations", "activations_manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError(
            "Missing activation manifest required for training: "
            f"{manifest_path}. Run the cache stage first."
        )
    return manifest_path


def load_activation_manifest(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Activation manifest must be a JSON object.")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(
            "Activation manifest must include integer schema_version; "
            f"received {schema_version}."
        )
    if int(schema_version) != int(ACTIVATIONS_MANIFEST_SCHEMA_VERSION):
        raise ValueError(
            "Activation manifest schema_version mismatch. Expected "
            f"{ACTIVATIONS_MANIFEST_SCHEMA_VERSION}, received {schema_version}."
        )
    return payload


def resolve_runtime_device(requested_device: str) -> str:
    return resolve_experiment_device(requested_device)


def _is_cuda_runtime_device(runtime_device: str) -> bool:
    return runtime_device == "cuda" or runtime_device.startswith("cuda:")


def _should_use_mixed_precision(
    runtime_device: str, mixed_precision_enabled: bool
) -> bool:
    return _is_cuda_runtime_device(runtime_device) and bool(mixed_precision_enabled)


def _resolve_variant_config(
    resolved_config: Dict[str, Any], subrun: Dict[str, Any]
) -> Dict[str, Any]:
    variant_slug = str(subrun["variant_slug"])
    variant_name = str(subrun["variant_name"])
    for variant in resolved_config["variants"]:
        if (
            str(variant["variant_slug"]) == variant_slug
            and str(variant["name"]) == variant_name
        ):
            return dict(variant)
    raise ValueError(
        "Unable to locate variant configuration for subrun: "
        f"variant_slug={variant_slug}, variant_name={variant_name}."
    )


def _resolve_manifest_d_in(manifest: Dict[str, Any]) -> int:
    activation = manifest.get("activation")
    if not isinstance(activation, dict):
        raise ValueError("Invalid activation manifest: missing 'activation'.")
    d_in = activation.get("d_in")
    if isinstance(d_in, bool) or not isinstance(d_in, int) or d_in <= 0:
        raise ValueError(f"Invalid activation manifest activation.d_in value: {d_in}.")
    return int(d_in)


def _resolve_split_sample_count(
    manifest: Dict[str, Any],
    split: str,
    max_samples: Optional[int],
) -> int:
    split_metadata = manifest.get("split")
    if not isinstance(split_metadata, dict):
        raise ValueError("Invalid activation manifest: missing split metadata.")
    split_method = split_metadata.get("method")
    if split_method == "stored_indices":
        key = f"{split}_count"
        split_count = split_metadata.get(key)
        if isinstance(split_count, bool) or not isinstance(split_count, int):
            raise ValueError(
                f"Invalid activation manifest split count '{key}': {split_count}."
            )
        if split_count < 0:
            raise ValueError(
                f"Invalid activation manifest split count '{key}': {split_count}."
            )
    elif split_method == "shard_ranges":
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise ValueError("Invalid activation manifest: missing shards list.")
        split_key = f"{split}_shard_indices"
        selected_indices = split_metadata.get(split_key)
        if not isinstance(selected_indices, list):
            raise ValueError(
                "Invalid activation manifest: missing split shard indices for "
                f"'{split_key}'."
            )
        split_count = 0
        for index in selected_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(
                    "Split shard indices must be integers; received "
                    f"{index} in {split_key}."
                )
            if index < 0 or index >= len(shards):
                raise ValueError(
                    f"Split shard index out of range in {split_key}: {index}."
                )
            shard = shards[index]
            if not isinstance(shard, dict):
                raise ValueError(f"Invalid shard entry at index {index}.")
            if isinstance(shard.get("num_vectors"), bool) or not isinstance(
                shard.get("num_vectors"), int
            ):
                raise ValueError(
                    f"Missing or invalid num_vectors for shard index {index}."
                )
            split_count += int(shard["num_vectors"])
    else:
        raise ValueError(
            "Invalid activation manifest split method. Expected 'stored_indices' or "
            f"'shard_ranges', received {split_method}."
        )

    if max_samples is not None:
        split_count = min(int(split_count), int(max_samples))
    return int(split_count)


def _resolve_resample_latents_threshold(
    training_config: TrainingConfig,
) -> int:
    if training_config.resample_latents_steps is None:
        return int(training_config.train_steps) + 1
    return int(training_config.resample_latents_steps)


def _inverse_softplus(value: float) -> float:
    clamped_value = max(float(value), 1e-8)
    return float(torch.log(torch.expm1(torch.tensor(clamped_value))).item())


def _resolve_effective_loss_knobs(variant_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "aux_loss_coeff": float(variant_config.get("aux_loss_coeff", 1.0)),
        "dead_zone_reg_coeff": float(variant_config.get("dead_zone_reg_coeff", 0.0)),
        "dead_zone_reg_type": str(variant_config.get("dead_zone_reg_type", "l2")),
    }


def _apply_initialization_overrides(
    model: torch.nn.Module, variant_config: Dict[str, Any]
) -> None:
    with torch.no_grad():
        if bool(variant_config.get("sign_aware", False)):
            single_activation_offset = bool(
                variant_config.get("single_activation_offset", False)
            )
            delta_init = variant_config.get("delta_init")
            delta_pos_init = variant_config.get("delta_pos_init")
            delta_neg_init = variant_config.get("delta_neg_init")

            if single_activation_offset and hasattr(model, "act_offset"):
                resolved_delta = delta_init
                if resolved_delta is None:
                    if delta_pos_init is not None and delta_neg_init is not None:
                        if abs(float(delta_pos_init) - float(delta_neg_init)) > 1e-12:
                            raise ValueError(
                                "delta_pos_init and delta_neg_init must match when "
                                "single_activation_offset=true."
                            )
                        resolved_delta = float(delta_pos_init)
                    elif delta_pos_init is not None:
                        resolved_delta = float(delta_pos_init)
                    elif delta_neg_init is not None:
                        resolved_delta = float(delta_neg_init)
                if resolved_delta is not None:
                    model.act_offset.data.fill_(float(resolved_delta))
            if not single_activation_offset:
                if delta_pos_init is not None and hasattr(model, "act_offset_right"):
                    model.act_offset_right.data.fill_(float(delta_pos_init))
                if delta_neg_init is not None and hasattr(model, "act_offset_left"):
                    model.act_offset_left.data.fill_(float(delta_neg_init))

        if isinstance(model, SignedSoftThresholdSAE):
            theta_init = variant_config.get("theta_init")
            if theta_init is not None:
                model.theta.data.fill_(float(theta_init))

        if isinstance(model, JumpReLUSAE):
            theta_init = variant_config.get("theta_init")
            jump_init = variant_config.get("jump_init")
            if theta_init is not None:
                if float(theta_init) <= 0.0:
                    raise ValueError(
                        "theta_init must be > 0 for JumpReLUSAE "
                        "(softplus parameterization)."
                    )
                model.theta_param.data.fill_(_inverse_softplus(float(theta_init)))
            if jump_init is not None:
                if float(jump_init) <= 0.0:
                    raise ValueError(
                        "jump_init must be > 0 for JumpReLUSAE "
                        "(softplus parameterization)."
                    )
                model.jump_param.data.fill_(_inverse_softplus(float(jump_init)))


def _copy_instance_0_parameters_to_all_instances(model: torch.nn.Module) -> None:
    if not hasattr(model, "n_instances"):
        return
    n_instances = int(getattr(model, "n_instances"))
    if n_instances <= 1:
        return
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim < 1 or parameter.shape[0] != n_instances:
                continue
            parameter.copy_(parameter[0:1].expand_as(parameter))


def instantiate_sae_model(
    variant_config: Dict[str, Any],
    n_instances: int,
    d_in: int,
    resample_latents_threshold: int,
    device: str,
    sweep_param: str,
    sweep_value: Optional[Any],
) -> torch.nn.Module:
    sae_class = str(variant_config["sae_class"])
    if sae_class in TOPK_SAE_CLASS_NAMES and sweep_param != "k":
        raise ValueError(
            f"{sae_class} requires sweep_param='k', received {sweep_param}."
        )
    if sae_class in LAMBDA_SWEEP_SAE_CLASS_NAMES and sweep_param != "lambda":
        raise ValueError(
            f"{sae_class} requires sweep_param='lambda', received {sweep_param}."
        )

    d_hidden = int(variant_config["d_hidden"])
    if sae_class == SAE.__name__:
        return SAE(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            sign_aware=bool(variant_config["sign_aware"]),
            reset_activation_offsets_on_resample=bool(
                variant_config["reset_activation_offsets_on_resample"]
            ),
            single_activation_offset=bool(variant_config["single_activation_offset"]),
            device=device,
        )
    if sae_class == GatedSAE.__name__:
        loss_knobs = _resolve_effective_loss_knobs(variant_config=variant_config)
        return GatedSAE(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            sign_aware=bool(variant_config["sign_aware"]),
            symmetric_magnitude=bool(variant_config["symmetric_magnitude"]),
            aux_loss_coeff=float(loss_knobs["aux_loss_coeff"]),
            dead_zone_reg_coeff=float(loss_knobs["dead_zone_reg_coeff"]),
            dead_zone_reg_type=str(loss_knobs["dead_zone_reg_type"]),
            reset_activation_offsets_on_resample=bool(
                variant_config["reset_activation_offsets_on_resample"]
            ),
            single_activation_offset=bool(variant_config["single_activation_offset"]),
            device=device,
        )
    if sae_class == SignedSoftThresholdSAE.__name__:
        return SignedSoftThresholdSAE(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            device=device,
        )
    if sae_class == JumpReLUSAE.__name__:
        return JumpReLUSAE(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            device=device,
        )
    if sae_class == TopKSAE.__name__:
        if sweep_value is None:
            raise ValueError("instantiate_sae_model requires sweep_value for TopKSAE.")
        return TopKSAE(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            topk_k=sweep_value if isinstance(sweep_value, list) else int(sweep_value),
            resample_latents_threshold=resample_latents_threshold,
            device=device,
        )
    if sae_class == AbsTopKSAE.__name__:
        if sweep_value is None:
            raise ValueError(
                "instantiate_sae_model requires sweep_value for AbsTopKSAE."
            )
        return AbsTopKSAE(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            topk_k=sweep_value if isinstance(sweep_value, list) else int(sweep_value),
            resample_latents_threshold=resample_latents_threshold,
            device=device,
        )
    if sae_class == AbsTopKGatedMagnitudeSAE.__name__:
        if sweep_value is None:
            raise ValueError(
                "instantiate_sae_model requires sweep_value for "
                "AbsTopKGatedMagnitudeSAE."
            )
        return AbsTopKGatedMagnitudeSAE(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            topk_k=sweep_value if isinstance(sweep_value, list) else int(sweep_value),
            resample_latents_threshold=resample_latents_threshold,
            device=device,
        )
    raise ValueError(f"Unknown SAE class for llm benchmark: {sae_class}.")


def build_model_for_subrun(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    manifest: Dict[str, Any],
    device: str,
) -> Tuple[torch.nn.Module, float, int, Dict[str, Any]]:
    variant_config = _resolve_variant_config(resolved_config, subrun)
    variant_slug = str(subrun["variant_slug"])
    seed = int(subrun["seed"])
    rng_fingerprint = str(resolved_config.get("rng_fingerprint", ""))
    if not rng_fingerprint.strip():
        raise ValueError(
            "Resolved config is missing rng_fingerprint required for "
            "deterministic initialization."
        )
    torch.manual_seed(
        _group_initialization_seed(
            rng_fingerprint=rng_fingerprint,
            variant_slug=variant_slug,
            seed=seed,
        )
    )
    d_in = _resolve_manifest_d_in(manifest)
    d_hidden = int(variant_config["d_hidden"])
    resample_latents_threshold = _resolve_resample_latents_threshold(
        training_config=config.training
    )
    sweep_param = str(subrun["sweep_param"])
    sweep_value = subrun["sweep_value"]
    model = instantiate_sae_model(
        variant_config=variant_config,
        n_instances=1,
        d_in=d_in,
        resample_latents_threshold=resample_latents_threshold,
        device=device,
        sweep_param=sweep_param,
        sweep_value=sweep_value,
    )

    sparsity_coeff = 0.0 if sweep_param == "k" else float(sweep_value)

    _apply_initialization_overrides(model=model, variant_config=variant_config)
    return model, float(sparsity_coeff), int(resample_latents_threshold), variant_config


def _resolve_reconstruction_output(output: Dict[str, Tensor]) -> Tensor:
    if "x_hat" in output:
        return output["x_hat"]
    if "y_pred" in output:
        return output["y_pred"]
    raise ValueError(
        "Model forward output must include reconstruction key 'x_hat' or 'y_pred'."
    )


def _distribution_summary(values: Tensor, prefix: str) -> Dict[str, float]:
    flattened = values.reshape(-1).to(dtype=torch.float32)
    if flattened.numel() == 0:
        return {}
    return {
        f"{prefix}_mean": float(flattened.mean().item()),
        f"{prefix}_p10": float(torch.quantile(flattened, 0.1).item()),
        f"{prefix}_p90": float(torch.quantile(flattened, 0.9).item()),
    }


def _distribution_summary_per_instance(
    values: Tensor, prefix: str
) -> List[Dict[str, float]]:
    if values.ndim < 1:
        raise ValueError(
            "Per-instance distribution summary requires leading instance dimension."
        )
    if values.shape[0] == 0:
        return []
    flattened = values.to(dtype=torch.float32).reshape(values.shape[0], -1)
    if flattened.shape[1] == 0:
        return [{} for _ in range(flattened.shape[0])]
    means = flattened.mean(dim=1)
    p10 = torch.quantile(flattened, 0.1, dim=1)
    p90 = torch.quantile(flattened, 0.9, dim=1)
    output: List[Dict[str, float]] = []
    for instance_index in range(flattened.shape[0]):
        output.append(
            {
                f"{prefix}_mean": float(means[instance_index].item()),
                f"{prefix}_p10": float(p10[instance_index].item()),
                f"{prefix}_p90": float(p90[instance_index].item()),
            }
        )
    return output


def _collect_threshold_statistics(model: torch.nn.Module) -> Dict[str, float]:
    output: Dict[str, float] = {}
    if hasattr(model, "act_offset"):
        delta = torch.relu(model.act_offset.detach())
        output.update(_distribution_summary(delta, "delta_pos"))
        output.update(_distribution_summary(delta, "delta_neg"))
    if hasattr(model, "act_offset_right"):
        output.update(
            _distribution_summary(
                torch.relu(model.act_offset_right.detach()), "delta_pos"
            )
        )
    if hasattr(model, "act_offset_left"):
        output.update(
            _distribution_summary(
                torch.relu(model.act_offset_left.detach()), "delta_neg"
            )
        )
    if hasattr(model, "theta"):
        output.update(_distribution_summary(torch.relu(model.theta.detach()), "theta"))
    if hasattr(model, "theta_param"):
        output.update(
            _distribution_summary(
                torch.nn.functional.softplus(model.theta_param.detach()), "theta"
            )
        )
    if hasattr(model, "jump_param"):
        output.update(
            _distribution_summary(
                torch.nn.functional.softplus(model.jump_param.detach()), "jump"
            )
        )
    return output


def _collect_threshold_statistics_per_instance(
    model: torch.nn.Module,
) -> List[Dict[str, float]]:
    n_instances = int(getattr(model, "n_instances", 1))
    output: List[Dict[str, float]] = [{} for _ in range(n_instances)]

    def _merge(stats_by_instance: List[Dict[str, float]]) -> None:
        for instance_index, stats in enumerate(stats_by_instance):
            output[instance_index].update(stats)

    if hasattr(model, "act_offset"):
        delta = torch.relu(model.act_offset.detach())
        _merge(_distribution_summary_per_instance(delta, "delta_pos"))
        _merge(_distribution_summary_per_instance(delta, "delta_neg"))
    if hasattr(model, "act_offset_right"):
        _merge(
            _distribution_summary_per_instance(
                torch.relu(model.act_offset_right.detach()), "delta_pos"
            )
        )
    if hasattr(model, "act_offset_left"):
        _merge(
            _distribution_summary_per_instance(
                torch.relu(model.act_offset_left.detach()), "delta_neg"
            )
        )
    if hasattr(model, "theta"):
        _merge(
            _distribution_summary_per_instance(
                torch.relu(model.theta.detach()), "theta"
            )
        )
    if hasattr(model, "theta_param"):
        _merge(
            _distribution_summary_per_instance(
                torch.nn.functional.softplus(model.theta_param.detach()), "theta"
            )
        )
    if hasattr(model, "jump_param"):
        _merge(
            _distribution_summary_per_instance(
                torch.nn.functional.softplus(model.jump_param.detach()), "jump"
            )
        )
    return output


def _collect_dead_zone_statistics(latents: Tensor) -> Dict[str, float]:
    # Exact zero: measures the activation function's hard-zero region,
    # distinct from the 1e-8 threshold used for L0 / resampling.
    dead_zone_mask = (latents == 0).to(dtype=torch.float32)
    dead_zone_fraction_by_latent = dead_zone_mask.mean(dim=1)
    return _distribution_summary(dead_zone_fraction_by_latent, "dead_zone_frac")


def _collect_dead_zone_statistics_per_instance(
    latents: Tensor,
) -> List[Dict[str, float]]:
    # Exact zero: see _collect_dead_zone_statistics.
    dead_zone_mask = (latents == 0).to(dtype=torch.float32)
    dead_zone_fraction_by_latent = dead_zone_mask.mean(dim=1)
    return _distribution_summary_per_instance(
        dead_zone_fraction_by_latent, "dead_zone_frac"
    )


def _latents_with_instance_dimension(latents: Tensor) -> Tensor:
    if latents.ndim == 2:
        return latents.unsqueeze(0)
    if latents.ndim == 3:
        return latents
    raise ValueError(
        "Polarity usage statistics require latents with shape "
        "[batch, d_hidden] or [n_instances, batch, d_hidden]."
    )


def _collect_polarity_usage_statistics_per_instance(
    latents: Tensor,
) -> List[Dict[str, float]]:
    latents_with_instances = _latents_with_instance_dimension(latents)
    positive_mask = latents_with_instances > 0
    negative_mask = latents_with_instances < 0

    positive_count = positive_mask.sum(dim=(1, 2)).to(dtype=torch.float32)
    negative_count = negative_mask.sum(dim=(1, 2)).to(dtype=torch.float32)
    nonzero_count = positive_count + negative_count
    positive_fraction = torch.where(
        nonzero_count > 0,
        positive_count / nonzero_count,
        torch.zeros_like(nonzero_count),
    )
    negative_fraction = torch.where(
        nonzero_count > 0,
        negative_count / nonzero_count,
        torch.zeros_like(nonzero_count),
    )

    latent_has_positive = positive_mask.any(dim=1)
    latent_has_negative = negative_mask.any(dim=1)
    active_latent_count = (
        (latent_has_positive | latent_has_negative).sum(dim=1).to(dtype=torch.float32)
    )
    bidirectional_latent_count = (
        (latent_has_positive & latent_has_negative).sum(dim=1).to(dtype=torch.float32)
    )
    bidirectional_fraction = torch.where(
        active_latent_count > 0,
        bidirectional_latent_count / active_latent_count,
        torch.zeros_like(active_latent_count),
    )

    output: List[Dict[str, float]] = []
    for instance_index in range(int(latents_with_instances.shape[0])):
        output.append(
            {
                "polarity_positive_fraction_nonzero": float(
                    positive_fraction[instance_index].item()
                ),
                "polarity_negative_fraction_nonzero": float(
                    negative_fraction[instance_index].item()
                ),
                "polarity_bidirectional_usage_fraction": float(
                    bidirectional_fraction[instance_index].item()
                ),
            }
        )
    return output


def _collect_polarity_usage_statistics(latents: Tensor) -> Dict[str, float]:
    stats_by_instance = _collect_polarity_usage_statistics_per_instance(latents)
    if not stats_by_instance:
        return {
            "polarity_positive_fraction_nonzero": 0.0,
            "polarity_negative_fraction_nonzero": 0.0,
            "polarity_bidirectional_usage_fraction": 0.0,
        }
    return stats_by_instance[0]


def _dominant_polarity_per_instance(latents: Tensor) -> Tensor:
    latents_with_instances = _latents_with_instance_dimension(latents)
    positive_count = (latents_with_instances > 0).sum(dim=1)
    negative_count = (latents_with_instances < 0).sum(dim=1)
    dominant_polarity = torch.zeros_like(positive_count, dtype=torch.int8)
    dominant_polarity = torch.where(
        positive_count > negative_count,
        torch.ones_like(dominant_polarity),
        dominant_polarity,
    )
    dominant_polarity = torch.where(
        negative_count > positive_count,
        -torch.ones_like(dominant_polarity),
        dominant_polarity,
    )
    return dominant_polarity


def _collect_polarity_flip_statistics_per_instance(
    previous_dominant_polarity: Optional[Tensor],
    current_dominant_polarity: Tensor,
) -> List[Dict[str, float]]:
    n_instances = int(current_dominant_polarity.shape[0])
    if (
        previous_dominant_polarity is None
        or previous_dominant_polarity.shape != current_dominant_polarity.shape
    ):
        return [
            {
                "polarity_dominance_flip_rate": 0.0,
                "polarity_dominance_flip_indicator": 0.0,
            }
            for _ in range(n_instances)
        ]

    previous = previous_dominant_polarity.to(
        device=current_dominant_polarity.device, dtype=torch.int8
    )
    comparable_mask = (previous != 0) & (current_dominant_polarity != 0)
    flipped_mask = comparable_mask & (previous != current_dominant_polarity)
    comparable_count = comparable_mask.sum(dim=1).to(dtype=torch.float32)
    flipped_count = flipped_mask.sum(dim=1).to(dtype=torch.float32)
    flip_rate = torch.where(
        comparable_count > 0,
        flipped_count / comparable_count,
        torch.zeros_like(comparable_count),
    )
    flip_indicator = (flipped_count > 0).to(dtype=torch.float32)

    output: List[Dict[str, float]] = []
    for instance_index in range(n_instances):
        output.append(
            {
                "polarity_dominance_flip_rate": float(flip_rate[instance_index].item()),
                "polarity_dominance_flip_indicator": float(
                    flip_indicator[instance_index].item()
                ),
            }
        )
    return output


def _collect_polarity_flip_statistics(
    previous_dominant_polarity: Optional[Tensor],
    current_dominant_polarity: Tensor,
) -> Dict[str, float]:
    stats_by_instance = _collect_polarity_flip_statistics_per_instance(
        previous_dominant_polarity=previous_dominant_polarity,
        current_dominant_polarity=current_dominant_polarity,
    )
    if not stats_by_instance:
        return {
            "polarity_dominance_flip_rate": 0.0,
            "polarity_dominance_flip_indicator": 0.0,
        }
    return stats_by_instance[0]


def _collect_threshold_gradient_norms(
    model: torch.nn.Module,
) -> Dict[str, float]:
    gradient_norms: Dict[str, float] = {}
    parameter_mappings = [
        ("act_offset", "grad_norm_delta"),
        ("act_offset_right", "grad_norm_delta_pos"),
        ("act_offset_left", "grad_norm_delta_neg"),
        ("theta", "grad_norm_theta"),
        ("theta_param", "grad_norm_theta"),
        ("jump_param", "grad_norm_jump"),
    ]
    for parameter_name, output_name in parameter_mappings:
        if not hasattr(model, parameter_name):
            continue
        parameter = getattr(model, parameter_name)
        if not isinstance(parameter, torch.nn.Parameter):
            continue
        if parameter.grad is None:
            continue
        gradient_norms[output_name] = float(parameter.grad.detach().norm().item())
    return gradient_norms


def _collect_threshold_gradient_norms_per_instance(
    model: torch.nn.Module,
) -> List[Dict[str, float]]:
    n_instances = int(getattr(model, "n_instances", 1))
    output: List[Dict[str, float]] = [{} for _ in range(n_instances)]
    parameter_mappings = [
        ("act_offset", "grad_norm_delta"),
        ("act_offset_right", "grad_norm_delta_pos"),
        ("act_offset_left", "grad_norm_delta_neg"),
        ("theta", "grad_norm_theta"),
        ("theta_param", "grad_norm_theta"),
        ("jump_param", "grad_norm_jump"),
    ]
    for parameter_name, output_name in parameter_mappings:
        if not hasattr(model, parameter_name):
            continue
        parameter = getattr(model, parameter_name)
        if not isinstance(parameter, torch.nn.Parameter):
            continue
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if gradient.ndim >= 1 and gradient.shape[0] == n_instances:
            norms = gradient.reshape(n_instances, -1).norm(dim=1)
            for instance_index in range(n_instances):
                output[instance_index][output_name] = float(
                    norms[instance_index].item()
                )
        elif n_instances == 1:
            output[0][output_name] = float(gradient.norm().item())
    return output


def _threshold_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    parameter_names = [
        "act_offset",
        "act_offset_right",
        "act_offset_left",
        "theta",
        "theta_param",
        "jump_param",
    ]
    output: List[torch.nn.Parameter] = []
    for parameter_name in parameter_names:
        if not hasattr(model, parameter_name):
            continue
        parameter = getattr(model, parameter_name)
        if isinstance(parameter, torch.nn.Parameter):
            output.append(parameter)
    return output


def _set_requires_grad(
    parameters: List[torch.nn.Parameter], requires_grad: bool
) -> None:
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def _delete_file_if_exists(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def _delete_downstream_subrun_artifacts(subrun_dir: str) -> None:
    for filename in _DOWNSTREAM_SUBRUN_ARTIFACT_FILENAMES:
        _delete_file_if_exists(os.path.join(subrun_dir, filename))
    for dirname in _DOWNSTREAM_SUBRUN_ARTIFACT_DIRNAMES:
        path = os.path.join(subrun_dir, dirname)
        if os.path.isdir(path):
            shutil.rmtree(path)


_CHECKPOINT_FILENAME_PATTERN = re.compile(r"^model_\d+\.pt$")


def _delete_checkpoint_files(subrun_dir: str) -> None:
    for filename in os.listdir(subrun_dir):
        if _CHECKPOINT_FILENAME_PATTERN.match(filename):
            os.remove(os.path.join(subrun_dir, filename))


def _write_training_status_marker(
    artifact_path: str,
    *,
    subrun: Dict[str, Any],
    config: LLMBenchmarkConfig,
    status: str,
    failure: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
        "variant_name": str(subrun["variant_name"]),
        "variant_slug": str(subrun["variant_slug"]),
        "seed": int(subrun["seed"]),
        "sweep_param": str(subrun["sweep_param"]),
        "sweep_value": subrun["sweep_value"],
        "warmup_steps": int(config.training.warmup_steps),
        "status": status,
        "terminated_early": status != "completed",
    }
    if failure is not None:
        payload["failure"] = failure
    with open(artifact_path, "w") as file:
        json.dump(payload, file, indent=2)


def _instance_seed_from_subrun(
    rng_fingerprint: str,
    variant_slug: str,
    seed: int,
    sweep_index: int,
) -> int:
    digest = sha256(
        f"{rng_fingerprint}|{variant_slug}|{seed}|{sweep_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _group_initialization_seed(
    rng_fingerprint: str, variant_slug: str, seed: int
) -> int:
    payload = "::".join(
        ["init", str(rng_fingerprint), str(variant_slug), str(seed)]
    ).encode("utf-8")
    digest = sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _build_training_metrics_base_payload(
    subrun: Dict[str, Any],
    config: LLMBenchmarkConfig,
    manifest: Dict[str, Any],
    variant_config: Dict[str, Any],
    use_mixed_precision: bool,
    resample_latents_threshold: int,
    thresholds_unfrozen_at_step: Optional[int],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    loss_knobs = _resolve_effective_loss_knobs(variant_config=variant_config)
    payload: Dict[str, Any] = {
        "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
        "variant_name": str(subrun["variant_name"]),
        "variant_slug": str(subrun["variant_slug"]),
        "seed": int(subrun["seed"]),
        "sweep_param": str(subrun["sweep_param"]),
        "sweep_value": subrun["sweep_value"],
        "d_in": int(_resolve_manifest_d_in(manifest)),
        "d_hidden": int(variant_config["d_hidden"]),
        "train_steps": int(config.training.train_steps),
        "batch_size": int(config.training.batch_size),
        "learning_rate": float(config.training.learning_rate),
        "adam_betas": [
            float(config.training.adam_beta1),
            float(config.training.adam_beta2),
        ],
        "warmup_steps": int(config.training.warmup_steps),
        "mixed_precision": bool(use_mixed_precision),
        "mixed_precision_dtype": str(config.training.mixed_precision_dtype),
        "threshold_warmup_steps": int(config.training.threshold_warmup_steps),
        "resample_dead_latents": bool(config.training.resample_dead_latents),
        "resample_latents_threshold": int(resample_latents_threshold),
        "loss_knobs": loss_knobs,
        "history": history,
    }
    if thresholds_unfrozen_at_step is not None:
        payload["thresholds_unfrozen_at_step"] = int(thresholds_unfrozen_at_step)
    return payload


def _dump_training_metrics_snapshot(
    artifact_path: str,
    subrun: Dict[str, Any],
    config: LLMBenchmarkConfig,
    manifest: Dict[str, Any],
    variant_config: Dict[str, Any],
    use_mixed_precision: bool,
    resample_latents_threshold: int,
    thresholds_unfrozen_at_step: Optional[int],
    history: List[Dict[str, Any]],
    continuation_boundary_step: Optional[int] = None,
) -> None:
    payload = _build_training_metrics_base_payload(
        subrun=subrun,
        config=config,
        manifest=manifest,
        variant_config=variant_config,
        use_mixed_precision=use_mixed_precision,
        resample_latents_threshold=resample_latents_threshold,
        thresholds_unfrozen_at_step=thresholds_unfrozen_at_step,
        history=history,
    )
    payload["status"] = "running"
    if continuation_boundary_step is not None:
        payload["continuation_boundary_step"] = int(continuation_boundary_step)
    with open(artifact_path, "w") as file:
        json.dump(payload, file, indent=2)


def _build_resample_generators_for_subruns(
    runtime_device: str,
    rng_fingerprint: str,
    variant_slug: str,
    seed: int,
    subruns: List[Dict[str, Any]],
) -> List[torch.Generator]:
    generators: List[torch.Generator] = []
    for subrun in subruns:
        if "sweep_index" not in subrun:
            raise ValueError(
                "Subrun is missing required stable sweep_index for deterministic "
                "resampling generators."
            )
        sweep_index = int(subrun["sweep_index"])
        generator = torch.Generator(device=runtime_device)
        generator.manual_seed(
            _instance_seed_from_subrun(
                rng_fingerprint=rng_fingerprint,
                variant_slug=variant_slug,
                seed=seed,
                sweep_index=sweep_index,
            )
        )
        generators.append(generator)
    return generators


def _load_continuation_state(
    subrun_dir: str,
) -> Tuple[Dict[str, Tensor], List[Dict[str, Any]], Optional[int]]:
    model_path = os.path.join(subrun_dir, "model.pt")
    if not os.path.isfile(model_path):
        raise ValueError(
            f"Cannot continue training: model.pt not found in {subrun_dir}."
        )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format in {model_path}: "
            f"expected dict with 'state_dict' key."
        )
    state_dict = checkpoint["state_dict"]

    history: List[Dict[str, Any]] = []
    prev_thresholds_unfrozen_at_step: Optional[int] = None

    metrics = load_training_metrics_if_present(subrun_dir)
    if metrics is not None and is_completed_payload(metrics):
        raw_history = metrics.get("history")
        if isinstance(raw_history, list):
            history = list(raw_history)
        raw_tuas = metrics.get("thresholds_unfrozen_at_step")
        if isinstance(raw_tuas, int) and not isinstance(raw_tuas, bool):
            prev_thresholds_unfrozen_at_step = int(raw_tuas)

    return state_dict, history, prev_thresholds_unfrozen_at_step


def _reassemble_state_dict_from_instances(
    instance_state_dicts: List[Dict[str, Tensor]],
    n_instances: int,
) -> Dict[str, Tensor]:
    if not instance_state_dicts:
        raise ValueError(
            "_reassemble_state_dict_from_instances requires at least one state_dict."
        )
    if len(instance_state_dicts) != n_instances:
        raise ValueError(
            f"Expected {n_instances} state_dicts, "
            f"received {len(instance_state_dicts)}."
        )
    reference = instance_state_dicts[0]
    reference_keys = set(reference.keys())
    for idx, sd in enumerate(instance_state_dicts[1:], start=1):
        if set(sd.keys()) != reference_keys:
            missing = reference_keys - set(sd.keys())
            extra = set(sd.keys()) - reference_keys
            raise ValueError(
                f"State dict key mismatch at instance {idx}: "
                f"missing={missing}, extra={extra}."
            )
    assembled: Dict[str, Tensor] = {}
    for name in reference.keys():
        tensors = [sd[name] for sd in instance_state_dicts]
        ref_tensor = tensors[0]
        if ref_tensor.ndim >= 1 and ref_tensor.shape[0] == 1:
            assembled[name] = torch.cat(tensors, dim=0)
        else:
            if not all(torch.equal(t, ref_tensor) for t in tensors[1:]):
                raise ValueError(
                    f"Non-instance-dimensioned parameter '{name}' "
                    f"differs across instances; cannot reassemble."
                )
            assembled[name] = ref_tensor.clone()
    return assembled


def _slice_state_dict_to_single_instance(
    state_dict: Dict[str, Tensor],
    n_instances: int,
    instance_index: int,
) -> Dict[str, Tensor]:
    sliced_state_dict: Dict[str, Tensor] = {}
    for name, value in state_dict.items():
        tensor = value.detach()
        if tensor.ndim >= 1 and tensor.shape[0] == n_instances:
            tensor = tensor[instance_index : instance_index + 1]
        sliced_state_dict[name] = tensor.to("cpu")
    return sliced_state_dict


def _create_train_activation_dataset(
    manifest_path: str, training_config: TrainingConfig, seed: int
) -> Any:
    return create_activation_dataset(
        manifest_path=manifest_path,
        split="train",
        batch_size=int(training_config.batch_size),
        shuffle=True,
        seed=int(seed),
        max_train_samples=training_config.max_train_samples,
        max_val_samples=training_config.max_val_samples,
        max_test_samples=training_config.max_test_samples,
    )


def _next_activation_batch(train_dataset: Any, train_iterator: Any) -> Tuple[Any, Any]:
    try:
        batch = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_dataset)
        try:
            batch = next(train_iterator)
        except StopIteration as exc:
            raise ValueError(
                "Training split yielded no activation batches. "
                "Regenerate the activation cache or adjust split/sample limits."
            ) from exc
    return batch, train_iterator


def _run_optimizer_warmup(
    warmup_steps: int,
    train_dataset: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    grad_clip_norm: Optional[float],
    use_mixed_precision: bool,
    mixed_precision_dtype: torch.dtype,
    runtime_device: str,
    compute_total_loss: Callable[[Tensor], Tensor],
) -> Optional[Dict[str, Any]]:
    if warmup_steps <= 0:
        return None

    original_learning_rates = [
        float(param_group["lr"]) for param_group in optimizer.param_groups
    ]
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.0
    try:
        train_iterator = iter(train_dataset)
        for warmup_step in range(1, int(warmup_steps) + 1):
            batch, train_iterator = _next_activation_batch(
                train_dataset, train_iterator
            )
            x = batch.to(device=runtime_device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.autocast(device_type="cuda", dtype=mixed_precision_dtype)
                if use_mixed_precision
                else nullcontext()
            ):
                total_loss = compute_total_loss(x)
            if not bool(torch.isfinite(total_loss).item()):
                return {
                    "type": "non_finite_loss",
                    "step": 0,
                    "warmup_step": int(warmup_step),
                    "phase": "warmup",
                    "message": (
                        "Encountered non-finite loss value during optimizer warmup."
                    ),
                }
            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(grad_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(grad_clip_norm)
                    )
                optimizer.step()
        return None
    finally:
        for param_group, learning_rate in zip(
            optimizer.param_groups, original_learning_rates
        ):
            param_group["lr"] = learning_rate


def run_training_for_sweep_chunk(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    variant: Dict[str, Any],
    seed: int,
    chunk_subruns: List[Dict[str, Any]],
    output_dir: str,
    overwrite: bool,
    continuation_from_steps: Optional[int] = None,
) -> Dict[str, Any]:
    if not chunk_subruns:
        raise ValueError("run_training_for_sweep_chunk requires at least one subrun.")
    sweep_params_in_chunk = {str(subrun.get("sweep_param")) for subrun in chunk_subruns}
    if len(sweep_params_in_chunk) != 1:
        raise ValueError(
            "run_training_for_sweep_chunk requires all subruns to share the same "
            f"sweep_param; received {sweep_params_in_chunk}."
        )
    sweep_param = sweep_params_in_chunk.pop()
    if sweep_param not in {"lambda", "k"}:
        raise ValueError(
            f"run_training_for_sweep_chunk requires sweep_param in "
            f'{{"lambda", "k"}}; received \'{sweep_param}\'.'
        )
    for subrun in chunk_subruns:
        if int(subrun.get("seed")) != int(seed):
            raise ValueError(
                "run_training_for_sweep_chunk requires all subruns to share seed."
            )

    variant_slug = str(variant["variant_slug"])
    variant_name = str(variant["name"])
    for subrun in chunk_subruns:
        if (
            str(subrun.get("variant_slug")) != variant_slug
            or str(subrun.get("variant_name")) != variant_name
        ):
            raise ValueError(
                "run_training_for_sweep_chunk requires all subruns to belong to the "
                "provided variant."
            )

    rng_fingerprint = str(resolved_config.get("rng_fingerprint", ""))
    if not rng_fingerprint.strip():
        raise ValueError(
            "Resolved config is missing rng_fingerprint required for "
            "deterministic initialization."
        )
    torch.manual_seed(
        _group_initialization_seed(
            rng_fingerprint=rng_fingerprint,
            variant_slug=variant_slug,
            seed=int(seed),
        )
    )

    chunk_subrun_dirs: List[str] = []
    chunk_artifact_paths: List[Dict[str, str]] = []
    for subrun in chunk_subruns:
        relative_subrun_dir = str(subrun["relative_subrun_dir"])
        subrun_dir = os.path.join(output_dir, relative_subrun_dir.replace("/", os.sep))
        os.makedirs(subrun_dir, exist_ok=True)
        artifact_paths = train_artifact_paths(subrun_dir)
        if overwrite:
            _delete_file_if_exists(artifact_paths["model"])
            _delete_file_if_exists(artifact_paths["model_config"])
            _delete_file_if_exists(artifact_paths["training_metrics"])
            _delete_checkpoint_files(subrun_dir)
            _delete_downstream_subrun_artifacts(subrun_dir)
        chunk_subrun_dirs.append(subrun_dir)
        chunk_artifact_paths.append(artifact_paths)

    is_continuation = continuation_from_steps is not None
    manifest: Optional[Dict[str, Any]] = None

    if is_continuation and overwrite:
        raise ValueError(
            "continuation_from_steps and overwrite=True are "
            "mutually exclusive. Set overwrite=false to "
            "continue training, or remove "
            "continuation_from_steps to retrain from scratch."
        )

    _continuation_instance_state_dicts: List[Dict[str, Tensor]] = []
    _continuation_previous_histories: List[List[Dict[str, Any]]] = [
        [] for _ in range(len(chunk_subruns))
    ]
    _continuation_thresholds_unfrozen_at_step: Optional[int] = None
    if is_continuation:
        for inst_idx, subrun in enumerate(chunk_subruns):
            inst_dir = os.path.join(
                output_dir,
                str(subrun["relative_subrun_dir"]).replace("/", os.sep),
            )
            sd, hist, prev_tuas = _load_continuation_state(inst_dir)
            _continuation_instance_state_dicts.append(sd)
            _continuation_previous_histories[inst_idx] = hist
            if prev_tuas is not None:
                if (
                    _continuation_thresholds_unfrozen_at_step is not None
                    and _continuation_thresholds_unfrozen_at_step != prev_tuas
                ):
                    raise ValueError(
                        "Inconsistent "
                        "thresholds_unfrozen_at_step across "
                        f"instances: "
                        f"{_continuation_thresholds_unfrozen_at_step}"
                        f" vs {prev_tuas} in {inst_dir}."
                    )
                _continuation_thresholds_unfrozen_at_step = prev_tuas

    # WARNING: this overwrites training_metrics.json — all
    # continuation state must be extracted above this line.
    for subrun, artifact_paths in zip(chunk_subruns, chunk_artifact_paths):
        _write_training_status_marker(
            artifact_path=artifact_paths["training_metrics"],
            subrun=subrun,
            config=config,
            status="running",
        )

    # Safe defaults for the except handler — these are
    # reassigned inside the try block but must be in scope
    # if an exception fires before the reassignment.
    use_mixed_precision = False
    resample_latents_threshold = 0
    thresholds_unfrozen_at_step: Optional[int] = None
    history_by_instance: List[List[Dict[str, Any]]] = [
        [] for _ in range(len(chunk_subruns))
    ]
    previous_history_by_instance: List[List[Dict[str, Any]]] = [
        [] for _ in range(len(chunk_subruns))
    ]
    start_step = 1
    pbar: Optional[tqdm.tqdm[int]] = None

    try:
        manifest_path = resolve_manifest_path(output_dir=output_dir)
        manifest = load_activation_manifest(manifest_path=manifest_path)
        runtime_device = resolve_runtime_device(config.device)
        d_in = _resolve_manifest_d_in(manifest)
        resample_latents_threshold = _resolve_resample_latents_threshold(
            training_config=config.training,
        )
        n_instances = len(chunk_subruns)
        sweep_value_for_instantiation: Optional[List[int]]
        if sweep_param == "k":
            sweep_value_for_instantiation = [
                int(subrun["sweep_value"]) for subrun in chunk_subruns
            ]
        elif sweep_param == "lambda":
            sweep_value_for_instantiation = None
        else:
            raise AssertionError(f"Unexpected sweep_param: {sweep_param!r}")
        model = instantiate_sae_model(
            variant_config=variant,
            n_instances=n_instances,
            d_in=d_in,
            resample_latents_threshold=resample_latents_threshold,
            device=runtime_device,
            sweep_param=sweep_param,
            sweep_value=sweep_value_for_instantiation,
        )
        if not is_continuation:
            _apply_initialization_overrides(model=model, variant_config=variant)
            _copy_instance_0_parameters_to_all_instances(model)
        model = model.to(runtime_device)

        start_step = 1
        previous_history_by_instance: List[List[Dict[str, Any]]] = [
            [] for _ in range(n_instances)
        ]
        if is_continuation:
            start_step = int(continuation_from_steps) + 1
            if start_step > int(config.training.train_steps):
                raise ValueError(
                    f"continuation_from_steps "
                    f"({continuation_from_steps}) >= "
                    f"train_steps "
                    f"({config.training.train_steps})."
                )
            assembled_sd = _reassemble_state_dict_from_instances(
                _continuation_instance_state_dicts, n_instances
            )
            model.load_state_dict(assembled_sd)
            previous_history_by_instance = _continuation_previous_histories
            if config.training.warmup_steps == 0:
                warnings.warn(
                    "Continuing training with warmup_steps=0. "
                    "The optimizer starts with cold Adam "
                    "moments, which may cause a transient "
                    "learning rate spike. Consider setting "
                    "warmup_steps > 0."
                )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.learning_rate),
            betas=(
                float(config.training.adam_beta1),
                float(config.training.adam_beta2),
            ),
            weight_decay=float(config.training.weight_decay),
        )

        use_mixed_precision = _should_use_mixed_precision(
            runtime_device=runtime_device,
            mixed_precision_enabled=bool(config.training.mixed_precision),
        )
        mixed_precision_dtype = (
            torch.float16
            if config.training.mixed_precision_dtype == "float16"
            else torch.bfloat16
        )
        grad_scaler_enabled = bool(
            use_mixed_precision and mixed_precision_dtype == torch.float16
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=grad_scaler_enabled)
        else:  # pragma: no cover
            scaler = torch.cuda.amp.GradScaler(enabled=grad_scaler_enabled)

        threshold_parameters = _threshold_parameters(model)
        if not is_continuation:
            thresholds_unfrozen_at_step = None
        else:
            thresholds_unfrozen_at_step = _continuation_thresholds_unfrozen_at_step
        should_freeze_thresholds = (
            config.training.threshold_warmup_steps > 0
            and threshold_parameters
            and start_step <= int(config.training.threshold_warmup_steps)
        )
        if should_freeze_thresholds:
            _set_requires_grad(threshold_parameters, False)

        train_dataset = _create_train_activation_dataset(
            manifest_path=manifest_path,
            training_config=config.training,
            seed=int(seed),
        )

        if sweep_param == "lambda":
            sparsity_coeffs = torch.tensor(
                [float(subrun["sweep_value"]) for subrun in chunk_subruns],
                dtype=torch.float32,
                device=runtime_device,
            )
        elif sweep_param == "k":
            sparsity_coeffs = torch.zeros(
                n_instances,
                dtype=torch.float32,
                device=runtime_device,
            )
        else:
            raise AssertionError(f"Unexpected sweep_param: {sweep_param!r}")
        inactive_latents_count = torch.zeros(
            n_instances, int(variant["d_hidden"]), device=runtime_device
        )
        resample_generators: Optional[List[torch.Generator]] = None
        if bool(config.training.resample_dead_latents):
            resample_generators = _build_resample_generators_for_subruns(
                runtime_device=runtime_device,
                rng_fingerprint=rng_fingerprint,
                variant_slug=variant_slug,
                seed=int(seed),
                subruns=chunk_subruns,
            )

        history_by_instance: List[List[Dict[str, Any]]] = [
            [] for _ in range(n_instances)
        ]
        previous_dominant_polarity_by_instance: Optional[Tensor] = None
        terminated_early = False
        first_batch_seen = False
        downstream_artifacts_invalidated = False

        if is_continuation:
            for subrun_dir in chunk_subrun_dirs:
                _delete_downstream_subrun_artifacts(subrun_dir)
            downstream_artifacts_invalidated = True

        def _compute_warmup_total_loss(x: Tensor) -> Tensor:
            output = model(x)
            _resolve_reconstruction_output(output)
            y_true = x.unsqueeze(0).expand(n_instances, -1, -1)
            losses = model.calculate_losses(y_true, sparsity_coeffs, output)
            return losses["losses"].sum()

        failure_payload = _run_optimizer_warmup(
            warmup_steps=int(config.training.warmup_steps),
            train_dataset=train_dataset,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=config.training.grad_clip_norm,
            use_mixed_precision=use_mixed_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            runtime_device=runtime_device,
            compute_total_loss=_compute_warmup_total_loss,
        )
        if failure_payload is not None:
            terminated_early = True
            if not downstream_artifacts_invalidated:
                for subrun_dir in chunk_subrun_dirs:
                    _delete_downstream_subrun_artifacts(subrun_dir)
                downstream_artifacts_invalidated = True
        train_iterator = iter(train_dataset)
        l0_running_sum: List[float] = []
        l0_running_count: int = 0
        l0_rolling: collections.deque[List[float]] = collections.deque(maxlen=1024)

        pbar = tqdm.tqdm(range(start_step, int(config.training.train_steps) + 1))
        for step in pbar:
            if terminated_early:
                break
            if (
                config.training.threshold_warmup_steps > 0
                and threshold_parameters
                and step == int(config.training.threshold_warmup_steps) + 1
            ):
                _set_requires_grad(threshold_parameters, True)
                thresholds_unfrozen_at_step = int(step)

            batch, train_iterator = _next_activation_batch(
                train_dataset, train_iterator
            )

            if not downstream_artifacts_invalidated:
                for subrun_dir in chunk_subrun_dirs:
                    _delete_downstream_subrun_artifacts(subrun_dir)
                downstream_artifacts_invalidated = True
            first_batch_seen = True
            x = batch.to(device=runtime_device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)

            with (
                torch.autocast(device_type="cuda", dtype=mixed_precision_dtype)
                if use_mixed_precision
                else nullcontext()
            ):
                output = model(x)
                reconstruction = _resolve_reconstruction_output(output)
                y_true = x.unsqueeze(0).expand(n_instances, -1, -1)
                losses = model.calculate_losses(y_true, sparsity_coeffs, output)
                total_loss = losses["losses"].sum()

            if not torch.isfinite(total_loss):
                terminated_early = True
                failure_payload = {
                    "type": "non_finite_loss",
                    "step": int(step),
                    "message": "Encountered non-finite loss value during training.",
                }
                break

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config.training.grad_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config.training.grad_clip_norm)
                    )
                optimizer.step()

            latents = output["latents"].detach()
            latents_active = latents.abs() > 1e-8
            if bool(config.training.resample_dead_latents):
                inactive_latents_count = torch.where(
                    latents_active.any(dim=1),
                    torch.zeros_like(inactive_latents_count),
                    inactive_latents_count + 1,
                )
                resampled_mask = model.resample_dead_latents(
                    inactive_latents_count,
                    generators=resample_generators,
                )
                inactive_latents_count = torch.where(
                    resampled_mask,
                    torch.zeros_like(inactive_latents_count),
                    inactive_latents_count,
                )

            should_log_step = step % int(
                config.training.log_every_n_steps
            ) == 0 or step == int(config.training.train_steps)
            if should_log_step:
                gradient_norms_by_instance = (
                    _collect_threshold_gradient_norms_per_instance(model)
                )
                mean_l0_by_instance = latents_active.sum(dim=-1).float().mean(dim=-1)
                batch_l0_list = mean_l0_by_instance.tolist()
                if len(l0_rolling) == l0_rolling.maxlen:
                    evicted = l0_rolling[0]
                    for i in range(len(evicted)):
                        l0_running_sum[i] -= evicted[i]
                l0_rolling.append(batch_l0_list)
                if not l0_running_sum:
                    l0_running_sum = [0.0] * len(batch_l0_list)
                for i in range(len(batch_l0_list)):
                    l0_running_sum[i] += batch_l0_list[i]
                l0_running_count = len(l0_rolling)
                rolling_mean = [s / l0_running_count for s in l0_running_sum]
                pbar.set_postfix_str(
                    f"L0 batch: {[round(v, 1) for v in batch_l0_list]}"
                    f", L0 avg({l0_running_count}): "
                    f"{[round(v, 1) for v in rolling_mean]}"
                )
                mean_l1_by_instance = latents.abs().sum(dim=-1).mean(dim=-1)
                threshold_stats_by_instance = (
                    _collect_threshold_statistics_per_instance(model)
                )
                dead_zone_stats_by_instance = (
                    _collect_dead_zone_statistics_per_instance(latents)
                )
                polarity_usage_stats_by_instance = (
                    _collect_polarity_usage_statistics_per_instance(latents)
                )
                current_dominant_polarity_by_instance = _dominant_polarity_per_instance(
                    latents
                )
                polarity_flip_stats_by_instance = _collect_polarity_flip_statistics_per_instance(
                    previous_dominant_polarity=previous_dominant_polarity_by_instance,
                    current_dominant_polarity=current_dominant_polarity_by_instance,
                )
                previous_dominant_polarity_by_instance = (
                    current_dominant_polarity_by_instance.to(
                        dtype=torch.int8, device="cpu"
                    )
                )
                for instance_index in range(n_instances):
                    history_entry: Dict[str, Any] = {
                        "step": int(step),
                        "loss_total": float(losses["losses"][instance_index].item()),
                        "loss_recon": float(
                            losses["reconstruction_losses"][instance_index].item()
                        ),
                        "mean_l0": float(mean_l0_by_instance[instance_index].item()),
                        "mean_l1": float(mean_l1_by_instance[instance_index].item()),
                    }
                    if "sparsity_losses" in losses:
                        history_entry["loss_sparsity"] = float(
                            losses["sparsity_losses"][instance_index].item()
                        )
                    if "aux_loss" in losses:
                        history_entry["loss_aux"] = float(
                            losses["aux_loss"][instance_index].item()
                        )
                    history_entry.update(
                        polarity_usage_stats_by_instance[instance_index]
                    )
                    history_entry.update(
                        polarity_flip_stats_by_instance[instance_index]
                    )
                    history_entry.update(threshold_stats_by_instance[instance_index])
                    history_entry.update(dead_zone_stats_by_instance[instance_index])
                    history_entry.update(gradient_norms_by_instance[instance_index])
                    history_by_instance[instance_index].append(history_entry)

            del reconstruction

            if (
                config.training.checkpoint_model_steps is not None
                and step % int(config.training.checkpoint_model_steps) == 0
            ):
                chunk_state_dict = model.state_dict()
                for inst_idx, inst_subrun_dir in enumerate(chunk_subrun_dirs):
                    sliced_sd = _slice_state_dict_to_single_instance(
                        state_dict=chunk_state_dict,
                        n_instances=n_instances,
                        instance_index=inst_idx,
                    )
                    checkpoint_path = os.path.join(inst_subrun_dir, f"model_{step}.pt")
                    torch.save({"state_dict": sliced_sd}, checkpoint_path)
                    del sliced_sd
                del chunk_state_dict

            if (
                config.training.dump_training_metrics_steps is not None
                and step % int(config.training.dump_training_metrics_steps) == 0
            ):
                for inst_idx, inst_subrun in enumerate(chunk_subruns):
                    _dump_training_metrics_snapshot(
                        artifact_path=chunk_artifact_paths[inst_idx][
                            "training_metrics"
                        ],
                        subrun=inst_subrun,
                        config=config,
                        manifest=manifest,
                        variant_config=variant,
                        use_mixed_precision=use_mixed_precision,
                        resample_latents_threshold=resample_latents_threshold,
                        thresholds_unfrozen_at_step=thresholds_unfrozen_at_step,
                        history=previous_history_by_instance[inst_idx]
                        + history_by_instance[inst_idx],
                        continuation_boundary_step=(
                            start_step if is_continuation else None
                        ),
                    )

        pbar.close()

        if not first_batch_seen and not terminated_early:
            raise ValueError("Training loop did not process any batches.")

        model_state_dict = model.state_dict()
        loss_knobs = _resolve_effective_loss_knobs(variant_config=variant)
        for instance_index, subrun in enumerate(chunk_subruns):
            artifact_paths = chunk_artifact_paths[instance_index]
            sliced_state_dict = _slice_state_dict_to_single_instance(
                state_dict=model_state_dict,
                n_instances=n_instances,
                instance_index=instance_index,
            )
            torch.save({"state_dict": sliced_state_dict}, artifact_paths["model"])
            del sliced_state_dict

            persisted_subrun = {
                key: value for key, value in subrun.items() if key != "sweep_index"
            }
            model_config_payload = {
                "variant": variant,
                "subrun": persisted_subrun,
                "training": config.training.model_dump(),
                "manifest_path": manifest_path,
                "manifest_cache_fingerprint": manifest.get("cache_fingerprint"),
                "resample_latents_threshold": int(resample_latents_threshold),
                "loss_knobs": loss_knobs,
            }
            with open(artifact_paths["model_config"], "w") as file:
                json.dump(model_config_payload, file, indent=2)

            training_metrics_payload = _build_training_metrics_base_payload(
                subrun=subrun,
                config=config,
                manifest=manifest,
                variant_config=variant,
                use_mixed_precision=use_mixed_precision,
                resample_latents_threshold=resample_latents_threshold,
                thresholds_unfrozen_at_step=thresholds_unfrozen_at_step,
                history=previous_history_by_instance[instance_index]
                + history_by_instance[instance_index],
            )
            if terminated_early:
                training_metrics_payload["status"] = "failed"
                training_metrics_payload["terminated_early"] = True
                training_metrics_payload["failure"] = failure_payload
            else:
                training_metrics_payload["status"] = "completed"
            if is_continuation:
                training_metrics_payload["continuation_boundary_step"] = int(start_step)
            with open(artifact_paths["training_metrics"], "w") as file:
                json.dump(training_metrics_payload, file, indent=2)
    except Exception as exc:
        if pbar is not None:
            pbar.close()
        failure = {
            "type": "exception",
            "message": str(exc),
        }
        # variant is a required parameter so no None guard needed
        # (unlike run_training_for_subrun where variant_config
        # is loaded inside the try block).
        if is_continuation and manifest is not None:
            for inst_idx, subrun in enumerate(chunk_subruns):
                failure_payload_full = _build_training_metrics_base_payload(
                    subrun=subrun,
                    config=config,
                    manifest=manifest,
                    variant_config=variant,
                    use_mixed_precision=use_mixed_precision,
                    resample_latents_threshold=(resample_latents_threshold),
                    thresholds_unfrozen_at_step=(thresholds_unfrozen_at_step),
                    history=(
                        previous_history_by_instance[inst_idx]
                        + history_by_instance[inst_idx]
                    ),
                )
                failure_payload_full["status"] = "failed"
                failure_payload_full["terminated_early"] = True
                failure_payload_full["failure"] = failure
                failure_payload_full["continuation_boundary_step"] = int(start_step)
                with open(
                    chunk_artifact_paths[inst_idx]["training_metrics"],
                    "w",
                ) as file:
                    json.dump(failure_payload_full, file, indent=2)
        else:
            for subrun, artifact_paths in zip(chunk_subruns, chunk_artifact_paths):
                _write_training_status_marker(
                    artifact_path=artifact_paths["training_metrics"],
                    subrun=subrun,
                    config=config,
                    status="failed",
                    failure=failure,
                )
        raise

    return {
        "subrun_dirs": chunk_subrun_dirs,
        "artifact_paths_by_subrun": chunk_artifact_paths,
        "terminated_early": terminated_early,
    }


run_training_for_lambda_chunk = run_training_for_sweep_chunk


def run_training_for_subrun(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    output_dir: str,
    overwrite: bool,
    continuation_from_steps: Optional[int] = None,
) -> Dict[str, Any]:
    relative_subrun_dir = str(subrun["relative_subrun_dir"])
    subrun_dir = os.path.join(output_dir, relative_subrun_dir.replace("/", os.sep))
    os.makedirs(subrun_dir, exist_ok=True)

    artifact_paths = train_artifact_paths(subrun_dir)
    if overwrite:
        _delete_file_if_exists(artifact_paths["model"])
        _delete_file_if_exists(artifact_paths["model_config"])
        _delete_file_if_exists(artifact_paths["training_metrics"])
        _delete_checkpoint_files(subrun_dir)
        _delete_downstream_subrun_artifacts(subrun_dir)

    is_continuation = continuation_from_steps is not None
    manifest: Optional[Dict[str, Any]] = None
    variant_config: Optional[Dict[str, Any]] = None

    _continuation_state_dict: Optional[Dict[str, Tensor]] = None
    _continuation_previous_history: List[Dict[str, Any]] = []
    _continuation_thresholds_unfrozen_at_step: Optional[int] = None
    if is_continuation:
        if overwrite:
            raise ValueError(
                "continuation_from_steps and overwrite=True are "
                "mutually exclusive. Set overwrite=false to "
                "continue training, or remove "
                "continuation_from_steps to retrain from scratch."
            )
        # Extract state BEFORE the marker overwrites
        # training_metrics.json (F1 ordering hazard).
        (
            _continuation_state_dict,
            _continuation_previous_history,
            _continuation_thresholds_unfrozen_at_step,
        ) = _load_continuation_state(subrun_dir)

    # WARNING: this overwrites training_metrics.json — all
    # continuation state must be extracted above this line.
    _write_training_status_marker(
        artifact_path=artifact_paths["training_metrics"],
        subrun=subrun,
        config=config,
        status="running",
    )

    # Safe defaults for the except handler — these are
    # reassigned inside the try block but must be in scope
    # if an exception fires before the reassignment.
    use_mixed_precision = False
    resample_latents_threshold = 0
    thresholds_unfrozen_at_step: Optional[int] = None
    history: List[Dict[str, Any]] = []
    previous_history: List[Dict[str, Any]] = []
    start_step = 1
    pbar: Optional[tqdm.tqdm[int]] = None

    try:
        manifest_path = resolve_manifest_path(output_dir=output_dir)
        manifest = load_activation_manifest(manifest_path=manifest_path)
        runtime_device = resolve_runtime_device(config.device)
        model, sparsity_coeff, resample_latents_threshold, variant_config = (
            build_model_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                manifest=manifest,
                device=runtime_device,
            )
        )
        model = model.to(runtime_device)

        start_step = 1
        previous_history: List[Dict[str, Any]] = []
        if is_continuation:
            start_step = int(continuation_from_steps) + 1
            if start_step > int(config.training.train_steps):
                raise ValueError(
                    f"continuation_from_steps "
                    f"({continuation_from_steps}) >= "
                    f"train_steps "
                    f"({config.training.train_steps})."
                )
            assert _continuation_state_dict is not None
            model.load_state_dict(_continuation_state_dict)
            previous_history = _continuation_previous_history
            thresholds_unfrozen_at_step = _continuation_thresholds_unfrozen_at_step
            if config.training.warmup_steps == 0:
                warnings.warn(
                    "Continuing training with warmup_steps=0. "
                    "The optimizer starts with cold Adam "
                    "moments, which may cause a transient "
                    "learning rate spike. Consider setting "
                    "warmup_steps > 0."
                )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.learning_rate),
            betas=(
                float(config.training.adam_beta1),
                float(config.training.adam_beta2),
            ),
            weight_decay=float(config.training.weight_decay),
        )

        use_mixed_precision = _should_use_mixed_precision(
            runtime_device=runtime_device,
            mixed_precision_enabled=bool(config.training.mixed_precision),
        )
        mixed_precision_dtype = (
            torch.float16
            if config.training.mixed_precision_dtype == "float16"
            else torch.bfloat16
        )
        grad_scaler_enabled = bool(
            use_mixed_precision and mixed_precision_dtype == torch.float16
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=grad_scaler_enabled)
        else:  # pragma: no cover
            scaler = torch.cuda.amp.GradScaler(enabled=grad_scaler_enabled)
        threshold_parameters = _threshold_parameters(model)
        if not is_continuation:
            thresholds_unfrozen_at_step = None
        should_freeze_thresholds = (
            config.training.threshold_warmup_steps > 0
            and threshold_parameters
            and start_step <= int(config.training.threshold_warmup_steps)
        )
        if should_freeze_thresholds:
            _set_requires_grad(threshold_parameters, False)

        train_dataset = _create_train_activation_dataset(
            manifest_path=manifest_path,
            training_config=config.training,
            seed=int(subrun["seed"]),
        )
        sparsity_coeffs = torch.tensor(
            [float(sparsity_coeff)], dtype=torch.float32, device=runtime_device
        )
        inactive_latents_count = torch.zeros(
            1, int(variant_config["d_hidden"]), device=runtime_device
        )

        history: List[Dict[str, Any]] = []
        previous_dominant_polarity: Optional[Tensor] = None
        terminated_early = False
        first_batch_seen = False
        downstream_artifacts_invalidated = False

        if is_continuation:
            _delete_downstream_subrun_artifacts(subrun_dir)
            downstream_artifacts_invalidated = True

        def _compute_warmup_total_loss(x: Tensor) -> Tensor:
            output = model(x)
            _resolve_reconstruction_output(output)
            y_true = x.unsqueeze(0)
            losses = model.calculate_losses(y_true, sparsity_coeffs, output)
            return losses["losses"].sum()

        failure_payload = _run_optimizer_warmup(
            warmup_steps=int(config.training.warmup_steps),
            train_dataset=train_dataset,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=config.training.grad_clip_norm,
            use_mixed_precision=use_mixed_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            runtime_device=runtime_device,
            compute_total_loss=_compute_warmup_total_loss,
        )
        if failure_payload is not None:
            terminated_early = True
            if not downstream_artifacts_invalidated:
                _delete_downstream_subrun_artifacts(subrun_dir)
                downstream_artifacts_invalidated = True
        train_iterator = iter(train_dataset)
        l0_running_sum: List[float] = []
        l0_running_count: int = 0
        l0_rolling: collections.deque[List[float]] = collections.deque(maxlen=1024)

        pbar = tqdm.tqdm(range(start_step, int(config.training.train_steps) + 1))
        for step in pbar:
            if terminated_early:
                break
            if (
                config.training.threshold_warmup_steps > 0
                and threshold_parameters
                and step == int(config.training.threshold_warmup_steps) + 1
            ):
                _set_requires_grad(threshold_parameters, True)
                thresholds_unfrozen_at_step = int(step)

            batch, train_iterator = _next_activation_batch(
                train_dataset, train_iterator
            )
            if not downstream_artifacts_invalidated:
                _delete_downstream_subrun_artifacts(subrun_dir)
                downstream_artifacts_invalidated = True
            first_batch_seen = True
            x = batch.to(device=runtime_device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)

            with (
                torch.autocast(device_type="cuda", dtype=mixed_precision_dtype)
                if use_mixed_precision
                else nullcontext()
            ):
                output = model(x)
                reconstruction = _resolve_reconstruction_output(output)
                y_true = x.unsqueeze(0)
                losses = model.calculate_losses(y_true, sparsity_coeffs, output)
                total_loss = losses["losses"].sum()

            if not torch.isfinite(total_loss):
                terminated_early = True
                failure_payload = {
                    "type": "non_finite_loss",
                    "step": int(step),
                    "message": "Encountered non-finite loss value during training.",
                }
                break

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config.training.grad_clip_norm)
                    )
                gradient_norms = _collect_threshold_gradient_norms(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config.training.grad_clip_norm)
                    )
                gradient_norms = _collect_threshold_gradient_norms(model)
                optimizer.step()

            latents = output["latents"].detach()
            latents_active = latents.abs() > 1e-8
            if bool(config.training.resample_dead_latents):
                inactive_latents_count = torch.where(
                    latents_active.any(dim=1),
                    torch.zeros_like(inactive_latents_count),
                    inactive_latents_count + 1,
                )
                resampled_mask = model.resample_dead_latents(inactive_latents_count)
                inactive_latents_count = torch.where(
                    resampled_mask,
                    torch.zeros_like(inactive_latents_count),
                    inactive_latents_count,
                )

            should_log_step = step % int(
                config.training.log_every_n_steps
            ) == 0 or step == int(config.training.train_steps)
            if should_log_step:
                mean_l0 = float(latents_active.sum(dim=-1).float().mean().item())
                batch_l0_list = [mean_l0]
                if len(l0_rolling) == l0_rolling.maxlen:
                    evicted = l0_rolling[0]
                    l0_running_sum[0] -= evicted[0]
                l0_rolling.append(batch_l0_list)
                if not l0_running_sum:
                    l0_running_sum = [0.0]
                l0_running_sum[0] += mean_l0
                l0_running_count = len(l0_rolling)
                pbar.set_postfix_str(
                    f"L0 batch: {[round(mean_l0, 1)]}"
                    f", L0 avg({l0_running_count}): "
                    f"{[round(l0_running_sum[0] / l0_running_count, 1)]}"
                )
                mean_l1 = float(latents.abs().sum(dim=-1).mean().item())
                history_entry: Dict[str, Any] = {
                    "step": int(step),
                    "loss_total": float(losses["losses"].mean().item()),
                    "loss_recon": float(losses["reconstruction_losses"].mean().item()),
                    "mean_l0": mean_l0,
                    "mean_l1": mean_l1,
                }
                if "sparsity_losses" in losses:
                    history_entry["loss_sparsity"] = float(
                        losses["sparsity_losses"].mean().item()
                    )
                if "aux_loss" in losses:
                    history_entry["loss_aux"] = float(losses["aux_loss"].mean().item())
                polarity_usage_stats = _collect_polarity_usage_statistics(latents)
                current_dominant_polarity = _dominant_polarity_per_instance(latents)
                polarity_flip_stats = _collect_polarity_flip_statistics(
                    previous_dominant_polarity=previous_dominant_polarity,
                    current_dominant_polarity=current_dominant_polarity,
                )
                previous_dominant_polarity = current_dominant_polarity.to(
                    dtype=torch.int8, device="cpu"
                )
                history_entry.update(polarity_usage_stats)
                history_entry.update(polarity_flip_stats)
                history_entry.update(_collect_threshold_statistics(model))
                history_entry.update(_collect_dead_zone_statistics(latents))
                history_entry.update(gradient_norms)
                history.append(history_entry)

            del reconstruction

            if (
                config.training.checkpoint_model_steps is not None
                and step % int(config.training.checkpoint_model_steps) == 0
            ):
                checkpoint_path = os.path.join(subrun_dir, f"model_{step}.pt")
                torch.save({"state_dict": model.state_dict()}, checkpoint_path)

            if (
                config.training.dump_training_metrics_steps is not None
                and step % int(config.training.dump_training_metrics_steps) == 0
            ):
                _dump_training_metrics_snapshot(
                    artifact_path=artifact_paths["training_metrics"],
                    subrun=subrun,
                    config=config,
                    manifest=manifest,
                    variant_config=variant_config,
                    use_mixed_precision=use_mixed_precision,
                    resample_latents_threshold=resample_latents_threshold,
                    thresholds_unfrozen_at_step=thresholds_unfrozen_at_step,
                    history=previous_history + history,
                    continuation_boundary_step=(
                        start_step if is_continuation else None
                    ),
                )

        pbar.close()

        if not first_batch_seen and not terminated_early:
            raise ValueError("Training loop did not process any batches.")

        torch.save({"state_dict": model.state_dict()}, artifact_paths["model"])

        loss_knobs = _resolve_effective_loss_knobs(variant_config=variant_config)
        persisted_subrun = {
            key: value for key, value in subrun.items() if key != "sweep_index"
        }
        model_config_payload = {
            "variant": variant_config,
            "subrun": persisted_subrun,
            "training": config.training.model_dump(),
            "manifest_path": manifest_path,
            "manifest_cache_fingerprint": manifest.get("cache_fingerprint"),
            "resample_latents_threshold": int(resample_latents_threshold),
            "loss_knobs": loss_knobs,
        }
        with open(artifact_paths["model_config"], "w") as file:
            json.dump(model_config_payload, file, indent=2)

        training_metrics_payload = _build_training_metrics_base_payload(
            subrun=subrun,
            config=config,
            manifest=manifest,
            variant_config=variant_config,
            use_mixed_precision=use_mixed_precision,
            resample_latents_threshold=resample_latents_threshold,
            thresholds_unfrozen_at_step=thresholds_unfrozen_at_step,
            history=previous_history + history,
        )
        if terminated_early:
            training_metrics_payload["status"] = "failed"
            training_metrics_payload["terminated_early"] = True
            training_metrics_payload["failure"] = failure_payload
        else:
            training_metrics_payload["status"] = "completed"
        if is_continuation:
            training_metrics_payload["continuation_boundary_step"] = int(start_step)

        with open(artifact_paths["training_metrics"], "w") as file:
            json.dump(training_metrics_payload, file, indent=2)
    except Exception as exc:
        if pbar is not None:
            pbar.close()
        failure = {
            "type": "exception",
            "message": str(exc),
        }
        if is_continuation and manifest is not None and variant_config is not None:
            failure_payload_full = _build_training_metrics_base_payload(
                subrun=subrun,
                config=config,
                manifest=manifest,
                variant_config=variant_config,
                use_mixed_precision=use_mixed_precision,
                resample_latents_threshold=(resample_latents_threshold),
                thresholds_unfrozen_at_step=(thresholds_unfrozen_at_step),
                history=previous_history + history,
            )
            failure_payload_full["status"] = "failed"
            failure_payload_full["terminated_early"] = True
            failure_payload_full["failure"] = failure
            failure_payload_full["continuation_boundary_step"] = int(start_step)
            with open(artifact_paths["training_metrics"], "w") as file:
                json.dump(failure_payload_full, file, indent=2)
        else:
            _write_training_status_marker(
                artifact_path=artifact_paths["training_metrics"],
                subrun=subrun,
                config=config,
                status="failed",
                failure=failure,
            )
        raise

    return {
        "subrun_dir": subrun_dir,
        "artifact_paths": artifact_paths,
        "terminated_early": terminated_early,
    }
