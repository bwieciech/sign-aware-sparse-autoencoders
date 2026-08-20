import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from sgnsae.models import (
    AbsTopKGatedMagnitudeSAE,
    AbsTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    SAE,
    SignedSoftThresholdSAE,
    TopKSAE,
)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def estimate_parameter_memory_bytes(
    param_count: int, dtype: Literal["float16", "float32", "bfloat16"]
) -> int:
    if dtype == "float32":
        bytes_per_parameter = 4
    elif dtype in {"float16", "bfloat16"}:
        bytes_per_parameter = 2
    else:
        raise ValueError(f"Unsupported dtype for memory estimate: {dtype}.")
    return int(param_count) * int(bytes_per_parameter)


def _resolve_topk_k_value(variant_config: Dict[str, Any], d_hidden: int) -> int:
    topk_value = variant_config.get("topk_k")
    if isinstance(topk_value, list):
        if not topk_value:
            raise ValueError("topk_k list cannot be empty.")
        topk_value = topk_value[0]
    if isinstance(topk_value, bool) or not isinstance(topk_value, int):
        return int(min(32, d_hidden))
    return int(min(max(1, topk_value), d_hidden))


def build_variant_model(
    variant_config: Dict[str, Any],
    d_in: int,
    device: str = "cpu",
    resample_latents_threshold: int = 1,
) -> torch.nn.Module:
    sae_class = str(variant_config["sae_class"])
    d_hidden = int(variant_config["d_hidden"])
    sign_aware = bool(variant_config.get("sign_aware", False))
    if sae_class == SAE.__name__:
        return SAE(
            n_instances=1,
            d_in=int(d_in),
            d_hidden=d_hidden,
            resample_latents_threshold=int(resample_latents_threshold),
            sign_aware=sign_aware,
            reset_activation_offsets_on_resample=bool(
                variant_config.get("reset_activation_offsets_on_resample", True)
            ),
            single_activation_offset=bool(
                variant_config.get("single_activation_offset", False)
            ),
            device=device,
        )
    if sae_class == GatedSAE.__name__:
        return GatedSAE(
            n_instances=1,
            d_in=int(d_in),
            d_hidden=d_hidden,
            resample_latents_threshold=int(resample_latents_threshold),
            sign_aware=sign_aware,
            symmetric_magnitude=bool(variant_config.get("symmetric_magnitude", False)),
            reset_activation_offsets_on_resample=bool(
                variant_config.get("reset_activation_offsets_on_resample", True)
            ),
            single_activation_offset=bool(
                variant_config.get("single_activation_offset", False)
            ),
            device=device,
        )
    if sae_class == SignedSoftThresholdSAE.__name__:
        return SignedSoftThresholdSAE(
            n_instances=1,
            d_in=int(d_in),
            d_hidden=d_hidden,
            resample_latents_threshold=int(resample_latents_threshold),
            device=device,
        )
    if sae_class == JumpReLUSAE.__name__:
        return JumpReLUSAE(
            n_instances=1,
            d_in=int(d_in),
            d_hidden=d_hidden,
            resample_latents_threshold=int(resample_latents_threshold),
            device=device,
        )
    if sae_class == TopKSAE.__name__:
        return TopKSAE(
            n_instances=1,
            d_in=int(d_in),
            d_hidden=d_hidden,
            topk_k=_resolve_topk_k_value(
                variant_config=variant_config, d_hidden=d_hidden
            ),
            resample_latents_threshold=int(resample_latents_threshold),
            device=device,
        )
    if sae_class == AbsTopKSAE.__name__:
        return AbsTopKSAE(
            n_instances=1,
            d_in=int(d_in),
            d_hidden=d_hidden,
            topk_k=_resolve_topk_k_value(
                variant_config=variant_config, d_hidden=d_hidden
            ),
            resample_latents_threshold=int(resample_latents_threshold),
            device=device,
        )
    if sae_class == AbsTopKGatedMagnitudeSAE.__name__:
        return AbsTopKGatedMagnitudeSAE(
            n_instances=1,
            d_in=int(d_in),
            d_hidden=d_hidden,
            topk_k=_resolve_topk_k_value(
                variant_config=variant_config, d_hidden=d_hidden
            ),
            resample_latents_threshold=int(resample_latents_threshold),
            device=device,
        )
    raise ValueError(f"Unknown SAE class for overhead utilities: {sae_class}.")


def summarize_variant_parameters(
    variant_config: Dict[str, Any], d_in: int
) -> Dict[str, Any]:
    model = build_variant_model(
        variant_config=variant_config,
        d_in=int(d_in),
        device="cpu",
    )
    param_count = count_parameters(model=model)
    d_hidden = int(variant_config["d_hidden"])
    return {
        "name": variant_config.get("name"),
        "variant_slug": variant_config.get("variant_slug"),
        "sae_class": variant_config.get("sae_class"),
        "d_in": int(d_in),
        "d_hidden": d_hidden,
        "param_count": int(param_count),
        "params_per_latent": float(param_count / float(d_hidden)),
        "estimated_memory_bytes": {
            "float16": estimate_parameter_memory_bytes(param_count, "float16"),
            "bfloat16": estimate_parameter_memory_bytes(param_count, "bfloat16"),
            "float32": estimate_parameter_memory_bytes(param_count, "float32"),
        },
    }


def _load_resolved_run_config(output_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"Missing resolved config at: {config_path}.")
    with open(config_path, "r") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Resolved config payload must be a JSON object.")
    return payload


def _load_manifest_d_in(output_dir: str) -> int:
    manifest_path = os.path.join(output_dir, "activations", "activations_manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError(f"Missing activation manifest at: {manifest_path}.")
    with open(manifest_path, "r") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Activation manifest must be a JSON object.")
    activation = payload.get("activation", {})
    if not isinstance(activation, dict):
        raise ValueError("Activation manifest missing 'activation' dictionary.")
    d_in = activation.get("d_in")
    if isinstance(d_in, bool) or not isinstance(d_in, int) or d_in <= 0:
        raise ValueError(f"Invalid activation.d_in value in manifest: {d_in}.")
    return int(d_in)


def summarize_run_parameters(output_dir: str) -> Dict[str, Any]:
    resolved_config = _load_resolved_run_config(output_dir=output_dir)
    d_in = _load_manifest_d_in(output_dir=output_dir)
    variants = resolved_config.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Resolved config variants must be a list.")

    seen_keys = set()
    variant_summaries: List[Dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("Resolved config variant entries must be objects.")
        dedupe_key = (
            str(variant.get("variant_slug")),
            str(variant.get("sae_class")),
            int(variant.get("d_hidden")),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        variant_summaries.append(
            summarize_variant_parameters(
                variant_config=variant,
                d_in=int(d_in),
            )
        )

    return {
        "schema_version": 1,
        "output_dir": output_dir,
        "d_in": int(d_in),
        "variants": variant_summaries,
    }
