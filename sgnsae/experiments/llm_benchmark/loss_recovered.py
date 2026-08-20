from __future__ import annotations

import difflib
import json
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as functional
from torch import Tensor

from sgnsae.experiments.llm_benchmark.cache import _iter_text_documents
from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig
from sgnsae.experiments.llm_benchmark.schema_versions import (
    LOSS_RECOVERED_SCHEMA_VERSION,
)
from sgnsae.experiments.llm_benchmark.training import (
    load_activation_manifest,
    resolve_manifest_path,
    resolve_runtime_device,
)

PreTrainedModel = Any
PreTrainedTokenizerBase = Any


_EVAL_TOKENS_FILENAME = "eval_tokens.pt"
_EVAL_TOKENS_META_FILENAME = "eval_tokens_meta.json"
_BASELINES_FILENAME = "baselines.json"

_MODEL_CACHE: Dict[str, PreTrainedModel] = {}
_TOKENS_CACHE: Dict[str, Dict[str, Tensor]] = {}
_KNOWN_LAYER_FAMILY_PREFIXES: Tuple[str, ...] = (
    "transformer.h",
    "h",
    "gpt_neox.layers",
    "model.layers",
    "layers",
)


def run_level_loss_recovered_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "loss_recovered")


def run_level_loss_recovered_paths(output_dir: str) -> Dict[str, str]:
    base_dir = run_level_loss_recovered_dir(output_dir)
    return {
        "directory": base_dir,
        "eval_tokens": os.path.join(base_dir, _EVAL_TOKENS_FILENAME),
        "eval_tokens_meta": os.path.join(base_dir, _EVAL_TOKENS_META_FILENAME),
        "baselines": os.path.join(base_dir, _BASELINES_FILENAME),
    }


def subrun_loss_recovered_path(subrun_dir: str) -> str:
    return os.path.join(subrun_dir, "loss_recovered.json")


def _delete_file_if_exists(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def _resolve_model_dtype(
    config: LLMBenchmarkConfig, runtime_device: str
) -> torch.dtype:
    dtype_name = str(config.llm.torch_dtype)
    if dtype_name == "auto":
        if runtime_device == "cuda" or runtime_device.startswith("cuda:"):
            return torch.float16
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported llm.torch_dtype value: {dtype_name}.")


def _load_llm_model(config: LLMBenchmarkConfig, runtime_device: str) -> PreTrainedModel:
    try:
        from transformers import AutoModelForCausalLM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loss Recovered requires transformers. "
            "Install with `pip install transformers`."
        ) from exc

    resolved_dtype = _resolve_model_dtype(config=config, runtime_device=runtime_device)
    cache_key = "::".join(
        [
            str(config.llm.model_id),
            str(config.llm.revision),
            runtime_device,
            str(resolved_dtype),
        ]
    )
    if cache_key in _MODEL_CACHE:
        model = _MODEL_CACHE[cache_key]
        model.eval()
        return model

    model = AutoModelForCausalLM.from_pretrained(
        str(config.llm.model_id),
        revision=config.llm.revision,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(runtime_device)
    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    _MODEL_CACHE[cache_key] = model
    return model


def _load_tokenizer(config: LLMBenchmarkConfig) -> PreTrainedTokenizerBase:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loss Recovered requires transformers. "
            "Install with `pip install transformers`."
        ) from exc

    tokenizer_id = config.llm.tokenizer_id or config.llm.model_id
    tokenizer_revision = config.llm.tokenizer_revision or config.llm.revision
    return AutoTokenizer.from_pretrained(
        str(tokenizer_id),
        revision=tokenizer_revision,
    )


def _resolve_hook_module_path(manifest: Dict[str, Any]) -> str:
    hook_payload = manifest.get("hook")
    if not isinstance(hook_payload, dict):
        raise ValueError("Activation manifest is missing hook metadata.")
    module_path = hook_payload.get("module_path_resolved")
    if not isinstance(module_path, str) or not module_path:
        raise ValueError(
            "Activation manifest is missing hook.module_path_resolved required for "
            "Loss Recovered."
        )
    return module_path


def _resolve_hook_layer(manifest: Dict[str, Any]) -> Optional[int]:
    hook_payload = manifest.get("hook")
    if not isinstance(hook_payload, dict):
        return None
    layer = hook_payload.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        return None
    return int(layer)


def _resolve_layer_family_prefix(model: torch.nn.Module) -> Optional[str]:
    for candidate_path in _KNOWN_LAYER_FAMILY_PREFIXES:
        try:
            module = model.get_submodule(candidate_path)
        except AttributeError:
            continue
        if isinstance(module, (torch.nn.ModuleList, list, tuple)):
            return candidate_path
    return None


def _remap_module_path_to_layer_family(
    module_path: str,
    resolved_family_prefix: Optional[str],
) -> Optional[str]:
    if resolved_family_prefix is None:
        return None
    for family_prefix in _KNOWN_LAYER_FAMILY_PREFIXES:
        if module_path == family_prefix:
            return resolved_family_prefix
        prefix_with_dot = f"{family_prefix}."
        if module_path.startswith(prefix_with_dot):
            suffix = module_path[len(family_prefix) :]
            return f"{resolved_family_prefix}{suffix}"
    return None


def _suggest_module_paths(model: torch.nn.Module, attempted_path: str) -> List[str]:
    module_names = [name for name, _ in model.named_modules()]
    return difflib.get_close_matches(attempted_path, module_names, n=5, cutoff=0.0)


def _resolve_hook_module(
    config: LLMBenchmarkConfig,
    manifest: Dict[str, Any],
    llm_model: torch.nn.Module,
) -> Tuple[torch.nn.Module, str]:
    module_path = _resolve_hook_module_path(manifest=manifest)
    original_error: Optional[AttributeError] = None
    try:
        return llm_model.get_submodule(module_path), module_path
    except AttributeError as error:
        original_error = error

    resolved_family_prefix = _resolve_layer_family_prefix(model=llm_model)
    candidate_paths: List[str] = []
    remapped_path = _remap_module_path_to_layer_family(
        module_path=module_path,
        resolved_family_prefix=resolved_family_prefix,
    )
    if isinstance(remapped_path, str) and remapped_path != module_path:
        candidate_paths.append(remapped_path)

    hook_payload = manifest.get("hook")
    hook_name_raw: Optional[Any] = None
    if isinstance(hook_payload, dict):
        hook_name_raw = hook_payload.get("hook_name")
    hook_name = (
        str(hook_name_raw)
        if isinstance(hook_name_raw, str) and hook_name_raw
        else str(config.hook.hook_name)
    )
    layer_index = _resolve_hook_layer(manifest=manifest)
    if (
        hook_name == "mlp_out"
        and isinstance(layer_index, int)
        and resolved_family_prefix is not None
    ):
        candidate_paths.append(f"{resolved_family_prefix}.{layer_index}.mlp")

    attempted_candidates: List[str] = []
    for candidate_path in candidate_paths:
        if candidate_path == module_path or candidate_path in attempted_candidates:
            continue
        attempted_candidates.append(candidate_path)
        try:
            return llm_model.get_submodule(candidate_path), candidate_path
        except AttributeError:
            continue

    close_matches = _suggest_module_paths(model=llm_model, attempted_path=module_path)
    raise ValueError(
        "Unable to resolve hook module path from activation manifest. "
        f"module_path_resolved='{module_path}', "
        f"fallback_candidates={attempted_candidates}, "
        f"close_matches={close_matches}"
    ) from original_error


def _dataset_config_for_loss_recovered(config: LLMBenchmarkConfig) -> Tuple[Any, str]:
    dataset_split = config.loss_recovered.dataset_split
    if dataset_split is None:
        if config.data.backend == "hf":
            if config.data.split is None:
                raise ValueError(
                    "loss_recovered requires data.split for backend='hf' when "
                    "loss_recovered.dataset_split is omitted."
                )
            dataset_split = str(config.data.split)
        else:
            dataset_split = "train"
    if config.data.backend == "hf":
        data_config = config.data.model_copy(update={"split": dataset_split})
        return data_config, dataset_split
    return config.data, dataset_split


def _build_eval_tokens(
    config: LLMBenchmarkConfig,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    tokenizer = _load_tokenizer(config=config)
    data_config, dataset_split = _dataset_config_for_loss_recovered(config=config)
    text_documents, dataset_metadata = _iter_text_documents(
        data=data_config,
        cache=config.cache,
    )

    eos_token_id = tokenizer.eos_token_id
    doc_separator_setting = str(config.cache.doc_separator)
    doc_separator_id: Optional[int] = None
    doc_separator_effective = doc_separator_setting
    if doc_separator_setting == "eos":
        if eos_token_id is not None:
            doc_separator_id = int(eos_token_id)
        else:
            doc_separator_effective = "none"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad_token_id and no eos_token; cannot apply "
                "pad-token fallback for Loss Recovered."
            )
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_fallback_applied = True
    else:
        pad_token_fallback_applied = False

    max_eval_sequences = int(config.loss_recovered.max_eval_sequences)
    max_seq_len = int(config.cache.max_seq_len)
    token_buffer: List[int] = []
    packed_blocks: List[List[int]] = []
    first_document = True

    for document_text in text_documents:
        encoded_tokens = tokenizer.encode(
            str(document_text),
            add_special_tokens=False,
        )
        if not isinstance(encoded_tokens, list):
            raise ValueError("Tokenizer.encode must return a list of token ids.")

        if (
            not first_document
            and doc_separator_effective == "eos"
            and doc_separator_id is not None
        ):
            token_buffer.append(int(doc_separator_id))
        token_buffer.extend(int(token_id) for token_id in encoded_tokens)
        first_document = False

        while (
            len(token_buffer) >= max_seq_len and len(packed_blocks) < max_eval_sequences
        ):
            packed_blocks.append(token_buffer[:max_seq_len])
            del token_buffer[:max_seq_len]

        if len(packed_blocks) >= max_eval_sequences:
            break

    if not packed_blocks:
        raise ValueError(
            "Loss Recovered token cache produced zero fixed-length blocks. "
            "Increase the amount of source data or lower cache.max_seq_len."
        )

    input_ids = torch.tensor(packed_blocks, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    token_payload = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    token_metadata: Dict[str, Any] = {
        "schema_version": LOSS_RECOVERED_SCHEMA_VERSION,
        "dataset": {
            **dataset_metadata,
            "split": dataset_split,
        },
        "tokenizer": {
            "id": config.llm.tokenizer_id or config.llm.model_id,
            "revision": config.llm.tokenizer_revision or config.llm.revision,
        },
        "tokenization": {
            "mode": "packed_fixed_length",
            "max_seq_len": max_seq_len,
            "doc_separator": doc_separator_effective,
            "doc_separator_id": doc_separator_id,
            "pad_token_fallback_applied": bool(pad_token_fallback_applied),
            "pad_token_id_used": tokenizer.pad_token_id,
        },
        "generation": {
            "seed": int(config.loss_recovered.seed),
            "num_sequences": int(input_ids.shape[0]),
            "llm_batch_size": int(config.loss_recovered.llm_batch_size),
        },
    }
    return token_payload, token_metadata


def _load_tokens_payload(
    config: LLMBenchmarkConfig, output_dir: str
) -> Dict[str, Tensor]:
    run_paths = run_level_loss_recovered_paths(output_dir=output_dir)
    if os.path.isfile(run_paths["eval_tokens"]):
        payload = torch.load(run_paths["eval_tokens"], map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid eval token payload at {run_paths['eval_tokens']}."
            )
        if "input_ids" not in payload or "attention_mask" not in payload:
            raise ValueError(
                f"Eval token payload must contain 'input_ids' and 'attention_mask': "
                f"{run_paths['eval_tokens']}."
            )
        input_ids = payload["input_ids"]
        attention_mask = payload["attention_mask"]
        if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
            raise ValueError(
                "Eval token payload contains non-tensor values for input_ids or "
                "attention_mask."
            )
        return {
            "input_ids": input_ids.to(torch.long),
            "attention_mask": attention_mask.to(torch.long),
        }

    if output_dir in _TOKENS_CACHE:
        cached = _TOKENS_CACHE[output_dir]
        return {
            "input_ids": cached["input_ids"].clone(),
            "attention_mask": cached["attention_mask"].clone(),
        }

    token_payload, token_metadata = _build_eval_tokens(config=config)
    _TOKENS_CACHE[output_dir] = {
        "input_ids": token_payload["input_ids"].clone(),
        "attention_mask": token_payload["attention_mask"].clone(),
    }
    run_paths = run_level_loss_recovered_paths(output_dir=output_dir)
    os.makedirs(run_paths["directory"], exist_ok=True)
    with open(run_paths["eval_tokens_meta"], "w") as file:
        json.dump(token_metadata, file, indent=2)
    if bool(config.loss_recovered.persist_tokens):
        torch.save(token_payload, run_paths["eval_tokens"])
    return token_payload


def _extract_logits(model_output: Any) -> Tensor:
    if hasattr(model_output, "logits") and torch.is_tensor(model_output.logits):
        return model_output.logits
    if isinstance(model_output, (tuple, list)) and model_output:
        first = model_output[0]
        if torch.is_tensor(first):
            return first
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        if torch.is_tensor(logits):
            return logits
    raise ValueError("Unable to extract logits from LLM model output.")


def _replace_single_tensor_in_output(output: Any, replacement: Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        replaced = False
        values = list(output)
        for index, value in enumerate(values):
            if torch.is_tensor(value):
                values[index] = replacement
                replaced = True
                break
        if not replaced:
            raise ValueError(
                "Hook output tuple did not contain a tensor to replace for Loss "
                "Recovered patching."
            )
        return tuple(values)
    if isinstance(output, list):
        replaced = False
        values = list(output)
        for index, value in enumerate(values):
            if torch.is_tensor(value):
                values[index] = replacement
                replaced = True
                break
        if not replaced:
            raise ValueError(
                "Hook output list did not contain a tensor to replace for Loss "
                "Recovered patching."
            )
        return values
    raise ValueError(
        "Loss Recovered patching requires tensor, tuple, or list hook outputs. "
        f"Received type {type(output)}."
    )


def _extract_tensor_from_hook_output(output: Any) -> Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value
    raise ValueError(
        "Hook output must be a tensor or tuple/list containing a tensor. "
        f"Received {type(output)}."
    )


def _resolve_reconstruction_from_sae_output(output: Dict[str, Tensor]) -> Tensor:
    if "x_hat" in output:
        return output["x_hat"]
    if "y_pred" in output:
        return output["y_pred"]
    raise ValueError(
        "SAE forward output must contain 'x_hat' or 'y_pred' for Loss Recovered."
    )


def _reshape_reconstruction_to_hook_shape(
    reconstruction: Tensor, hook_tensor: Tensor
) -> Tensor:
    if reconstruction.ndim == 3 and int(reconstruction.shape[0]) == 1:
        reconstruction = reconstruction[0]
    if hook_tensor.ndim == 3:
        batch_size = int(hook_tensor.shape[0])
        seq_len = int(hook_tensor.shape[1])
        d_in = int(hook_tensor.shape[2])
        reconstruction_flat = reconstruction.reshape(-1, d_in)
        return reconstruction_flat.reshape(batch_size, seq_len, d_in)
    if hook_tensor.ndim == 2:
        d_in = int(hook_tensor.shape[1])
        return reconstruction.reshape(-1, d_in)
    raise ValueError(
        "Loss Recovered currently supports rank-2 or rank-3 hook tensors. "
        f"Received shape {tuple(hook_tensor.shape)}."
    )


def _build_reconstructed_hook_tensor(
    hook_tensor: Tensor,
    sae_model: torch.nn.Module,
    ablated_feature_indices: Optional[List[int]] = None,
) -> Tensor:
    if hook_tensor.ndim == 3:
        flattened = hook_tensor.reshape(-1, int(hook_tensor.shape[-1]))
    elif hook_tensor.ndim == 2:
        flattened = hook_tensor
    else:
        raise ValueError(
            "Loss Recovered currently supports rank-2 or rank-3 hook tensors. "
            f"Received shape {tuple(hook_tensor.shape)}."
        )
    if int(flattened.shape[-1]) != int(getattr(sae_model, "d_in")):
        raise ValueError(
            "SAE input dimension does not match hook activation dimension: "
            f"sae.d_in={int(getattr(sae_model, 'd_in'))}, "
            f"hook_d_in={int(flattened.shape[-1])}."
        )
    sae_device = next(sae_model.parameters()).device
    sae_input = flattened.to(device=sae_device, dtype=torch.float32)
    with torch.no_grad():
        sae_output = sae_model(sae_input)
    reconstruction = _resolve_reconstruction_from_sae_output(sae_output)
    if ablated_feature_indices:
        latents = sae_output.get("latents")
        if not torch.is_tensor(latents):
            raise ValueError(
                "SAE forward output must include 'latents' for latent-space "
                "ablation in Loss Recovered."
            )
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        if latents.ndim != 3:
            raise ValueError(
                "SAE latents for Loss Recovered ablation must be rank 2 or 3; "
                f"received shape {tuple(latents.shape)}."
            )
        if reconstruction.ndim == 2:
            reconstruction = reconstruction.unsqueeze(0)
        if reconstruction.ndim != 3:
            raise ValueError(
                "SAE reconstruction for Loss Recovered ablation must be rank 2 or 3; "
                f"received shape {tuple(reconstruction.shape)}."
            )
        w_dec_norm = sae_model.W_dec / (
            sae_model.W_dec.norm(dim=1, keepdim=True) + 1e-8
        )
        for feature_index in ablated_feature_indices:
            if (
                isinstance(feature_index, bool)
                or not isinstance(feature_index, int)
                or feature_index < 0
                or feature_index >= int(latents.shape[-1])
            ):
                raise ValueError(
                    "Invalid ablated feature index for Loss Recovered: "
                    f"{feature_index}."
                )
            feature_latents = latents[:, :, feature_index].unsqueeze(-1)
            feature_direction = w_dec_norm[:, :, feature_index].unsqueeze(1)
            reconstruction = reconstruction - feature_latents * feature_direction
    reconstruction = _reshape_reconstruction_to_hook_shape(
        reconstruction=reconstruction,
        hook_tensor=hook_tensor,
    )
    return reconstruction.to(device=hook_tensor.device, dtype=hook_tensor.dtype)


def _loss_for_mode(
    model: torch.nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    llm_batch_size: int,
    hook_module: Optional[torch.nn.Module] = None,
    mode: str = "none",
    sae_model: Optional[torch.nn.Module] = None,
    ablated_feature_indices: Optional[List[int]] = None,
) -> Tuple[float, int]:
    model_device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    if mode not in {"none", "ablate", "reconstruct"}:
        raise ValueError(f"Unsupported Loss Recovered mode: {mode}.")

    hook_handle = None
    if mode != "none":
        if hook_module is None:
            raise ValueError("hook_module is required for ablate/reconstruct mode.")
        if mode == "reconstruct" and sae_model is None:
            raise ValueError("sae_model is required for reconstruct mode.")

        def _hook_fn(_module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
            hook_tensor = _extract_tensor_from_hook_output(output)
            if mode == "ablate":
                patched = torch.zeros_like(hook_tensor)
            else:
                patched = _build_reconstructed_hook_tensor(
                    hook_tensor=hook_tensor,
                    sae_model=sae_model,  # type: ignore[arg-type]
                    ablated_feature_indices=ablated_feature_indices,
                )
            return _replace_single_tensor_in_output(output, patched)

        hook_handle = hook_module.register_forward_hook(_hook_fn)

    try:
        batch_start = 0
        while batch_start < int(input_ids.shape[0]):
            batch_end = min(batch_start + int(llm_batch_size), int(input_ids.shape[0]))
            input_ids_batch = input_ids[batch_start:batch_end].to(model_device)
            attention_mask_batch = attention_mask[batch_start:batch_end].to(
                model_device
            )
            labels = input_ids_batch.clone()
            labels[attention_mask_batch == 0] = -100

            autocast_context = nullcontext()
            if model_device.type == "cuda":
                model_dtype = next(model.parameters()).dtype
                if model_dtype in (torch.float16, torch.bfloat16):
                    autocast_context = torch.autocast(
                        device_type="cuda", dtype=model_dtype
                    )

            with torch.inference_mode():
                with autocast_context:
                    model_output = model(
                        input_ids=input_ids_batch,
                        attention_mask=attention_mask_batch,
                        use_cache=False,
                        return_dict=True,
                    )
            logits = _extract_logits(model_output)
            shift_logits = logits[:, :-1, :].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            batch_nll = functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            tokens_in_batch = int((shift_labels != -100).sum().item())
            total_nll += float(batch_nll.item())
            total_tokens += tokens_in_batch
            batch_start = batch_end
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if total_tokens <= 0:
        raise ValueError(
            "Loss Recovered evaluation found zero valid next-token labels."
        )
    return float(total_nll / float(total_tokens)), int(total_tokens)


def ensure_run_level_loss_recovered_artifacts(
    config: LLMBenchmarkConfig,
    output_dir: str,
    overwrite: bool,
) -> Dict[str, Any]:
    if not bool(config.loss_recovered.enabled):
        return {"enabled": False}

    run_paths = run_level_loss_recovered_paths(output_dir=output_dir)
    os.makedirs(run_paths["directory"], exist_ok=True)

    if overwrite:
        _delete_file_if_exists(run_paths["eval_tokens"])
        _delete_file_if_exists(run_paths["eval_tokens_meta"])
        _delete_file_if_exists(run_paths["baselines"])
        if output_dir in _TOKENS_CACHE:
            del _TOKENS_CACHE[output_dir]

    if os.path.isfile(run_paths["eval_tokens_meta"]) and os.path.isfile(
        run_paths["baselines"]
    ):
        if bool(config.loss_recovered.persist_tokens) and not os.path.isfile(
            run_paths["eval_tokens"]
        ):
            tokens_payload, _ = _build_eval_tokens(config=config)
            torch.save(tokens_payload, run_paths["eval_tokens"])
            _TOKENS_CACHE[output_dir] = {
                "input_ids": tokens_payload["input_ids"].clone(),
                "attention_mask": tokens_payload["attention_mask"].clone(),
            }
        return {
            "enabled": True,
            "reused": True,
            "paths": run_paths,
        }

    token_payload, token_meta = _build_eval_tokens(config=config)
    _TOKENS_CACHE[output_dir] = {
        "input_ids": token_payload["input_ids"].clone(),
        "attention_mask": token_payload["attention_mask"].clone(),
    }
    with open(run_paths["eval_tokens_meta"], "w") as file:
        json.dump(token_meta, file, indent=2)
    if bool(config.loss_recovered.persist_tokens):
        torch.save(token_payload, run_paths["eval_tokens"])

    runtime_device = resolve_runtime_device(config.device)
    llm_model = _load_llm_model(config=config, runtime_device=runtime_device)
    manifest_path = resolve_manifest_path(output_dir=output_dir)
    manifest = load_activation_manifest(manifest_path=manifest_path)
    hook_module, module_path = _resolve_hook_module(
        config=config,
        manifest=manifest,
        llm_model=llm_model,
    )
    loss_original, num_tokens = _loss_for_mode(
        model=llm_model,
        input_ids=token_payload["input_ids"],
        attention_mask=token_payload["attention_mask"],
        llm_batch_size=int(config.loss_recovered.llm_batch_size),
        mode="none",
    )
    loss_ablated, _ = _loss_for_mode(
        model=llm_model,
        input_ids=token_payload["input_ids"],
        attention_mask=token_payload["attention_mask"],
        llm_batch_size=int(config.loss_recovered.llm_batch_size),
        hook_module=hook_module,
        mode="ablate",
    )
    baselines_payload = {
        "schema_version": LOSS_RECOVERED_SCHEMA_VERSION,
        "llm": {
            "model_id": config.llm.model_id,
            "revision": config.llm.revision,
        },
        "hook": {
            "module_path_resolved": module_path,
            "layer": _resolve_hook_layer(manifest=manifest),
        },
        "eval": {
            "num_sequences": int(token_payload["input_ids"].shape[0]),
            "num_tokens": int(num_tokens),
            "llm_batch_size": int(config.loss_recovered.llm_batch_size),
        },
        "loss_original": float(loss_original),
        "loss_ablated": float(loss_ablated),
    }
    with open(run_paths["baselines"], "w") as file:
        json.dump(baselines_payload, file, indent=2)
    return {
        "enabled": True,
        "reused": False,
        "paths": run_paths,
    }


def compute_reconstructed_loss(
    config: LLMBenchmarkConfig,
    output_dir: str,
    sae_model: torch.nn.Module,
    ablated_feature_indices: Optional[List[int]] = None,
) -> Tuple[float, int]:
    if not bool(config.loss_recovered.enabled):
        raise ValueError(
            "compute_reconstructed_loss requires loss_recovered.enabled=true."
        )
    token_payload = _load_tokens_payload(config=config, output_dir=output_dir)
    runtime_device = resolve_runtime_device(config.device)
    llm_model = _load_llm_model(config=config, runtime_device=runtime_device)
    llm_device = next(llm_model.parameters()).device
    sae_device = next(sae_model.parameters()).device
    if llm_device != sae_device:
        raise ValueError(
            "Loss Recovered requires SAE and LLM on the same device. "
            f"Received sae_device={sae_device}, llm_device={llm_device}."
        )
    manifest_path = resolve_manifest_path(output_dir=output_dir)
    manifest = load_activation_manifest(manifest_path=manifest_path)
    hook_module, _ = _resolve_hook_module(
        config=config,
        manifest=manifest,
        llm_model=llm_model,
    )
    return _loss_for_mode(
        model=llm_model,
        input_ids=token_payload["input_ids"],
        attention_mask=token_payload["attention_mask"],
        llm_batch_size=int(config.loss_recovered.llm_batch_size),
        hook_module=hook_module,
        mode="reconstruct",
        sae_model=sae_model,
        ablated_feature_indices=ablated_feature_indices,
    )


def compute_and_write_subrun_loss_recovered(
    config: LLMBenchmarkConfig,
    subrun: Dict[str, Any],
    output_dir: str,
    subrun_dir: str,
    sae_model: torch.nn.Module,
    overwrite: bool,
) -> Optional[Dict[str, Any]]:
    if not bool(config.loss_recovered.enabled):
        return None

    output_path = subrun_loss_recovered_path(subrun_dir=subrun_dir)
    if os.path.isfile(output_path) and not overwrite:
        with open(output_path, "r") as file:
            return json.load(file)

    run_paths = run_level_loss_recovered_paths(output_dir=output_dir)
    if not os.path.isfile(run_paths["baselines"]):
        _ = ensure_run_level_loss_recovered_artifacts(
            config=config,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    if not os.path.isfile(run_paths["baselines"]):
        raise ValueError("Loss Recovered baselines.json is missing after setup stage.")
    with open(run_paths["baselines"], "r") as file:
        baselines = json.load(file)

    token_payload = _load_tokens_payload(config=config, output_dir=output_dir)
    loss_reconstructed, num_tokens = compute_reconstructed_loss(
        config=config,
        output_dir=output_dir,
        sae_model=sae_model,
    )

    loss_original = float(baselines["loss_original"])
    loss_ablated = float(baselines["loss_ablated"])
    denominator = float(loss_ablated - loss_original)
    warnings: List[str] = []
    loss_recovered: Optional[float]
    if abs(denominator) < float(config.loss_recovered.denominator_epsilon):
        warnings.append(
            "loss_ablated - loss_original is too small for stable division; "
            "loss_recovered set to null."
        )
        loss_recovered = None
    else:
        loss_recovered = float((loss_ablated - loss_reconstructed) / denominator)

    payload = {
        "schema_version": LOSS_RECOVERED_SCHEMA_VERSION,
        "variant_name": str(subrun["variant_name"]),
        "variant_slug": str(subrun["variant_slug"]),
        "seed": int(subrun["seed"]),
        "sweep_param": str(subrun["sweep_param"]),
        "sweep_value": subrun["sweep_value"],
        "eval": {
            "num_sequences": int(token_payload["input_ids"].shape[0]),
            "num_tokens": int(num_tokens),
            "llm_batch_size": int(config.loss_recovered.llm_batch_size),
        },
        "loss_original": loss_original,
        "loss_ablated": loss_ablated,
        "loss_reconstructed": float(loss_reconstructed),
        "denominator": denominator,
        "loss_recovered": loss_recovered,
        "warnings": warnings,
    }
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)
    return payload
