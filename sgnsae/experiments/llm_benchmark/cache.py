import difflib
import glob
import hashlib
import io
import json
import os
import shutil
import warnings
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageOps

from sgnsae.experiments.device import resolve_experiment_device
from sgnsae.experiments.llm_benchmark.config import (
    ActivationCacheConfig,
    HookpointConfig,
    ImageActivationCacheConfig,
    ImageDataConfig,
    LLMBenchmarkConfig,
    TextDataConfig,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _compute_cache_fingerprint(cache_identity: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(cache_identity)).hexdigest()


def _storage_dtype_was_explicitly_set(config: LLMBenchmarkConfig) -> bool:
    return "storage_dtype" in config.cache.model_fields_set


def _resolve_runtime_device(requested_device: str) -> str:
    return resolve_experiment_device(requested_device)


def _is_cuda_runtime_device(runtime_device: str) -> bool:
    return runtime_device == "cuda" or runtime_device.startswith("cuda:")


def _resolve_storage_dtype(
    config: LLMBenchmarkConfig,
    runtime_device: str,
) -> str:
    configured_storage_dtype = str(config.cache.storage_dtype)
    if _storage_dtype_was_explicitly_set(config):
        return configured_storage_dtype
    if runtime_device == "cpu":
        return "float32"
    return "float16"


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype name: {dtype_name}")


def _resolve_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported torch dtype: {dtype}.")


def _resolve_runtime_device_kind(runtime_device: str) -> str:
    if _is_cuda_runtime_device(runtime_device):
        return "cuda"
    return "cpu"


def _resolve_model_torch_dtype(
    config: LLMBenchmarkConfig,
    runtime_device: str,
) -> torch.dtype:
    configured_torch_dtype = str(config.llm.torch_dtype)
    if configured_torch_dtype == "auto":
        if _is_cuda_runtime_device(runtime_device):
            return torch.float16
        return torch.float32
    return _resolve_dtype(configured_torch_dtype)


def _resolve_vision_model_torch_dtype(
    config: LLMBenchmarkConfig,
    runtime_device: str,
) -> torch.dtype:
    configured_torch_dtype = str(config.vision.torch_dtype)
    if configured_torch_dtype == "auto":
        if _is_cuda_runtime_device(runtime_device):
            return torch.float16
        return torch.float32
    return _resolve_dtype(configured_torch_dtype)


def _resolve_layer_count_and_family_prefix(model: torch.nn.Module) -> Tuple[int, str]:
    candidates: List[Tuple[str, str]] = [
        ("transformer.h", "transformer.h"),
        ("h", "h"),
        ("gpt_neox.layers", "gpt_neox.layers"),
        ("model.layers", "model.layers"),
        ("layers", "layers"),
    ]
    for candidate_path, family_prefix in candidates:
        try:
            module = model.get_submodule(candidate_path)
        except AttributeError:
            continue
        if isinstance(module, torch.nn.ModuleList):
            return len(module), family_prefix
        if isinstance(module, (list, tuple)):
            return len(module), family_prefix
    raise ValueError(
        "Unable to infer transformer layer container from model. "
        "Expected one of: transformer.h, h, gpt_neox.layers, model.layers, layers."
    )


def _resolve_layer_index(model: torch.nn.Module, hook: HookpointConfig) -> int:
    if hook.layer is not None:
        return int(hook.layer)

    num_layers, _ = _resolve_layer_count_and_family_prefix(model)
    if num_layers <= 0:
        raise ValueError("Model appears to have zero layers.")

    selector = hook.layer_selector
    if selector == "mid":
        return int(num_layers // 2)
    if selector == "late":
        return int(0.75 * (num_layers - 1))
    if isinstance(selector, float):
        return int(round(selector * (num_layers - 1)))
    raise ValueError(f"Unsupported hook.layer_selector: {selector}")


def _resolve_default_module_path(
    model: torch.nn.Module,
    hook: HookpointConfig,
    layer_index: int,
) -> str:
    hook_name = str(hook.hook_name)
    if hook_name != "mlp_out":
        raise ValueError(
            "Automatic hook resolution currently supports hook_name='mlp_out' only. "
            f"Received: {hook_name}"
        )

    num_layers, family_prefix = _resolve_layer_count_and_family_prefix(model)
    if layer_index < 0 or layer_index >= num_layers:
        raise ValueError(
            f"Layer index {layer_index} is out of range for model with {num_layers} layers."
        )
    return f"{family_prefix}.{layer_index}.mlp"


def _suggest_module_paths(model: torch.nn.Module, attempted_path: str) -> List[str]:
    module_names = [name for name, _ in model.named_modules()]
    return difflib.get_close_matches(attempted_path, module_names, n=5, cutoff=0.0)


def resolve_hook_module(
    model: torch.nn.Module,
    hook: HookpointConfig,
) -> Tuple[torch.nn.Module, str, int]:
    layer_index = _resolve_layer_index(model, hook)
    if hook.module_path is not None:
        module_path = str(hook.module_path)
    else:
        module_path = _resolve_default_module_path(
            model=model,
            hook=hook,
            layer_index=layer_index,
        )

    try:
        module = model.get_submodule(module_path)
    except AttributeError as exc:
        close_matches = _suggest_module_paths(model, module_path)
        raise ValueError(
            f"Unable to resolve hook module path '{module_path}'. "
            f"Close matches: {close_matches}"
        ) from exc

    return module, module_path, layer_index


def _resolve_local_data_files(data: TextDataConfig) -> List[str]:
    if data.data_files is not None:
        return [str(path) for path in data.data_files]
    if data.data_files_glob is None:
        raise ValueError(
            "Local JSONL backend requires data_files or data_files_glob, but neither was provided."
        )
    matched_paths = sorted(glob.glob(str(data.data_files_glob)))
    if not matched_paths:
        raise ValueError(
            "Local JSONL backend matched no files for data_files_glob='"
            f"{data.data_files_glob}'."
        )
    return matched_paths


def _iter_jsonl_records(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError as exc:
            raise ValueError(
                "Reading .jsonl.zst files requires the 'zstandard' package. "
                "Install dependencies with '.venv/bin/pip install -r requirements.txt'."
            ) from exc

        with open(path, "rb") as compressed_file:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(compressed_file) as stream:
                text_stream = io.TextIOWrapper(stream, encoding="utf-8")
                for line_number, line in enumerate(text_stream, start=1):
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    try:
                        parsed = json.loads(stripped_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Failed parsing JSON in '{path}' at line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"Expected JSON object in '{path}' at line {line_number}, "
                            f"received {type(parsed)}."
                        )
                    yield parsed
        return

    with open(path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                parsed = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed parsing JSON in '{path}' at line {line_number}: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected JSON object in '{path}' at line {line_number}, "
                    f"received {type(parsed)}."
                )
            yield parsed


def _iter_text_documents(
    data: TextDataConfig,
    cache: ActivationCacheConfig,
) -> Tuple[Iterable[str], Dict[str, Any]]:
    text_field = str(data.text_field)
    if data.backend == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ValueError(
                "The 'datasets' package is required for backend='hf'. "
                "Install dependencies with '.venv/bin/pip install -r requirements.txt'."
            ) from exc

        dataset_revision = cache.hf_revision or data.revision
        dataset_stream = load_dataset(
            path=str(data.dataset_id),
            split=str(data.split),
            revision=dataset_revision,
            streaming=bool(data.streaming),
        )

        def _generator() -> Iterable[str]:
            for entry in dataset_stream:
                if not isinstance(entry, dict):
                    raise ValueError(
                        "HF dataset entry must be a dictionary, received "
                        f"{type(entry)}."
                    )
                if text_field not in entry:
                    available_keys = sorted(entry.keys())
                    raise ValueError(
                        f"Missing text field '{text_field}' in HF dataset entry. "
                        f"Available keys: {available_keys}"
                    )
                text = entry[text_field]
                if not isinstance(text, str):
                    raise ValueError(
                        f"HF dataset field '{text_field}' must be a string, "
                        f"received {type(text)}."
                    )
                yield text

        dataset_metadata: Dict[str, Any] = {
            "backend": data.backend,
            "dataset_id": data.dataset_id,
            "split": data.split,
            "revision": dataset_revision,
            "streaming": bool(data.streaming),
            "text_field": text_field,
            "dataset_fingerprint": getattr(dataset_stream, "_fingerprint", None),
        }
        return _generator(), dataset_metadata

    local_files = _resolve_local_data_files(data)

    def _generator() -> Iterable[str]:
        for path in local_files:
            for entry in _iter_jsonl_records(path):
                if text_field not in entry:
                    available_keys = sorted(entry.keys())
                    raise ValueError(
                        f"Missing text field '{text_field}' in file '{path}'. "
                        f"Available keys: {available_keys}"
                    )
                text = entry[text_field]
                if not isinstance(text, str):
                    raise ValueError(
                        f"Field '{text_field}' in file '{path}' must be a string, "
                        f"received {type(text)}."
                    )
                yield text

    dataset_metadata = {
        "backend": data.backend,
        "dataset_id": None,
        "split": None,
        "revision": None,
        "streaming": False,
        "text_field": text_field,
        "data_files": local_files,
        "data_files_glob": data.data_files_glob,
        "dataset_fingerprint": None,
    }
    return _generator(), dataset_metadata


def _build_cache_identity(
    config: LLMBenchmarkConfig,
    module_path_resolved: str,
    layer_index: int,
    storage_dtype: str,
    runtime_device_kind: str,
    model_torch_dtype_resolved: str,
) -> Dict[str, Any]:
    dataset_revision = config.cache.hf_revision or config.data.revision
    if config.data.backend == "hf":
        dataset_identity: Dict[str, Any] = {
            "backend": config.data.backend,
            "dataset_id": config.data.dataset_id,
            "split": config.data.split,
            "revision": dataset_revision,
            "streaming": bool(config.data.streaming),
            "text_field": config.data.text_field,
        }
    else:
        dataset_identity = {
            "backend": config.data.backend,
            "data_files": config.data.data_files,
            "data_files_glob": config.data.data_files_glob,
            "text_field": config.data.text_field,
        }

    split_identity: Dict[str, Any] = {
        "split_method": config.cache.split_method,
        "train_fraction": float(config.cache.train_fraction),
        "val_fraction": float(config.cache.val_fraction),
        "test_fraction": float(config.cache.test_fraction),
    }

    cache_identity = {
        "model": {
            "model_id": config.llm.model_id,
            "model_revision": config.llm.revision,
            "tokenizer_id": config.llm.tokenizer_id,
            "tokenizer_revision": config.llm.tokenizer_revision,
        },
        "hook": {
            "hook_name": config.hook.hook_name,
            "layer": layer_index,
            "module_path_resolved": module_path_resolved,
        },
        "dataset": dataset_identity,
        "cache": {
            "max_seq_len": int(config.cache.max_seq_len),
            "doc_separator": config.cache.doc_separator,
            "max_tokens": config.cache.max_tokens,
            "max_sequences": config.cache.max_sequences,
            "vectors_per_shard": int(config.cache.vectors_per_shard),
            "compute_dtype": config.cache.compute_dtype,
            "storage_dtype": storage_dtype,
            "cache_seed": config.cache.cache_seed,
            "store_tokens": bool(config.cache.store_tokens),
            "store_positions": bool(config.cache.store_positions),
            "store_sequence_ids": bool(config.cache.store_sequence_ids),
        },
        "split": split_identity,
        "execution": {
            "runtime_device_kind": runtime_device_kind,
            "model_torch_dtype_resolved": model_torch_dtype_resolved,
        },
    }
    return cache_identity


def _load_manifest(manifest_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, "r") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        return None
    return payload


def _manifest_schema_version_matches_expected(manifest: Dict[str, Any]) -> bool:
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        return False
    return int(schema_version) == int(ACTIVATIONS_MANIFEST_SCHEMA_VERSION)


def _cache_has_all_shards(manifest: Dict[str, Any], activations_dir: str) -> bool:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        return False
    for shard_entry in shards:
        if not isinstance(shard_entry, dict):
            return False
        filename = shard_entry.get("filename")
        if not isinstance(filename, str) or not filename:
            return False
        shard_path = os.path.join(activations_dir, filename)
        if not os.path.isfile(shard_path):
            return False

    split = manifest.get("split")
    if isinstance(split, dict) and split.get("method") == "stored_indices":
        indices_filename = split.get("indices_filename")
        if not isinstance(indices_filename, str) or not indices_filename:
            return False
        indices_path = os.path.join(activations_dir, indices_filename)
        if not os.path.isfile(indices_path):
            return False

    return True


def _extract_hook_tensor(output: Any, module_path: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value
    raise TypeError(
        "Hook output must be a tensor or tuple/list containing a tensor at module "
        f"'{module_path}', received type {type(output)}."
    )


def _select_split_from_hash(
    global_index: int,
    cache_seed: int,
    train_fraction: float,
    val_fraction: float,
) -> str:
    digest = hashlib.sha256(f"{cache_seed}:{global_index}".encode("utf-8")).digest()
    hashed_value = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    if hashed_value < train_fraction:
        return "train"
    if hashed_value < train_fraction + val_fraction:
        return "val"
    return "test"


def _assign_split_shard_ranges(
    num_shards: int,
    train_fraction: float,
    val_fraction: float,
) -> Dict[str, List[int]]:
    if num_shards <= 0:
        return {"train": [], "val": [], "test": []}

    train_count = int(num_shards * train_fraction)
    val_count = int(num_shards * val_fraction)
    test_count = num_shards - train_count - val_count

    if train_count == 0:
        train_count = 1
        if test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1

    if train_count + val_count + test_count != num_shards:
        test_count = num_shards - train_count - val_count  # pragma: no cover

    train_indices = list(range(0, train_count))
    val_indices = list(range(train_count, train_count + val_count))
    test_indices = list(range(train_count + val_count, num_shards))
    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def _build_stored_indices(
    num_vectors_total: int,
    cache: Union[ActivationCacheConfig, ImageActivationCacheConfig],
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    max_test_samples: Optional[int],
) -> Dict[str, torch.Tensor]:
    if (
        num_vectors_total > 50_000_000
        and max_train_samples is None
        and max_val_samples is None
        and max_test_samples is None
    ):
        raise ValueError(
            "stored_indices split_method is disabled for caches larger than 50,000,000 "
            "vectors unless max_*_samples limits are set in training config."
        )

    cache_seed = int(cache.cache_seed or 0)
    split_to_limit: Dict[str, Optional[int]] = {
        "train": max_train_samples,
        "val": max_val_samples,
        "test": max_test_samples,
    }
    split_to_indices: Dict[str, List[int]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    limitless_splits = [
        split_name
        for split_name, split_limit in split_to_limit.items()
        if split_limit is None
    ]

    for global_index in range(num_vectors_total):
        split_name = _select_split_from_hash(
            global_index=global_index,
            cache_seed=cache_seed,
            train_fraction=float(cache.train_fraction),
            val_fraction=float(cache.val_fraction),
        )

        split_limit = split_to_limit[split_name]
        if split_limit is not None and len(split_to_indices[split_name]) >= split_limit:
            pass
        else:
            split_to_indices[split_name].append(global_index)

        if not limitless_splits:
            train_done = len(split_to_indices["train"]) >= int(max_train_samples or 0)
            val_done = len(split_to_indices["val"]) >= int(max_val_samples or 0)
            test_done = len(split_to_indices["test"]) >= int(max_test_samples or 0)
            if train_done and val_done and test_done:
                break

    return {
        "train": torch.tensor(split_to_indices["train"], dtype=torch.int64),
        "val": torch.tensor(split_to_indices["val"], dtype=torch.int64),
        "test": torch.tensor(split_to_indices["test"], dtype=torch.int64),
    }


def _is_probable_hf_access_error(exception: Exception) -> bool:
    normalized_message = str(exception).strip().lower()
    if not normalized_message:
        return False
    keywords = [
        "401",
        "403",
        "gated",
        "forbidden",
        "not authorized",
        "unauthorized",
        "permission",
        "requires authentication",
        "access to model",
        "access to dataset",
        "accept",
        "license",
        "terms",
        "token",
    ]
    return any(keyword in normalized_message for keyword in keywords)


def _raise_hf_access_error(
    exception: Exception,
    resource_kind: str,
    resource_id: str,
) -> None:
    raise ValueError(
        f"Unable to access {resource_kind} '{resource_id}'. "
        "If this resource is gated, accept the terms on Hugging Face and run "
        "`huggingface-cli login` (or provide `HF_TOKEN`) before rerunning."
    ) from exception


def _coerce_image_value_to_pil(
    image_value: Any,
    image_field: str,
) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, dict):
        bytes_value = image_value.get("bytes")
        path_value = image_value.get("path")
        if isinstance(bytes_value, bytes):
            with Image.open(io.BytesIO(bytes_value)) as decoded_image:
                return decoded_image.convert("RGB")
        if isinstance(path_value, str) and path_value:
            with Image.open(path_value) as decoded_image:
                return decoded_image.convert("RGB")

    raise ValueError(
        f"Expected field '{image_field}' to be a PIL image or decoded image record, "
        f"received {type(image_value)}."
    )


def _iter_image_records(
    data: ImageDataConfig,
    cache: ImageActivationCacheConfig,
) -> Tuple[Iterable[Dict[str, Any]], Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exception:
        raise ValueError(
            "The 'datasets' package is required for image caching. "
            "Install dependencies with '.venv/bin/pip install -r requirements.txt'."
        ) from exception

    dataset_revision = cache.hf_revision or data.revision

    if data.backend == "hf":
        try:
            dataset_stream = load_dataset(
                path=str(data.dataset_id),
                split=str(data.split),
                revision=dataset_revision,
                streaming=bool(data.streaming),
            )
        except Exception as exception:
            if _is_probable_hf_access_error(exception):
                _raise_hf_access_error(
                    exception=exception,
                    resource_kind="dataset",
                    resource_id=str(data.dataset_id),
                )
            raise
    else:
        dataset_stream = load_dataset(
            path="imagefolder",
            data_dir=str(data.data_dir),
            split=str(data.split),
            streaming=bool(data.streaming),
        )

    image_field = str(data.image_field)

    def _generator() -> Iterable[Dict[str, Any]]:
        for entry in dataset_stream:
            if not isinstance(entry, dict):
                raise ValueError(
                    "Image dataset entry must be a dictionary, received "
                    f"{type(entry)}."
                )
            if image_field not in entry:
                available_keys = sorted(entry.keys())
                raise ValueError(
                    f"Missing image field '{image_field}' in dataset entry. "
                    f"Available keys: {available_keys}"
                )
            yield entry

    dataset_metadata: Dict[str, Any] = {
        "backend": data.backend,
        "dataset_id": data.dataset_id,
        "split": data.split,
        "revision": dataset_revision,
        "streaming": bool(data.streaming),
        "image_field": data.image_field,
        "label_field": data.label_field,
        "data_dir": data.data_dir,
        "dataset_fingerprint": getattr(dataset_stream, "_fingerprint", None),
    }
    return _generator(), dataset_metadata


def _resolve_image_hidden_state_layer_index(
    model: torch.nn.Module,
    hook: HookpointConfig,
) -> int:
    hook_name = str(hook.hook_name).strip().lower()
    if hook_name != "hidden_state":
        raise ValueError(
            "Image caching requires hook.hook_name='hidden_state'. "
            f"Received {hook.hook_name}."
        )

    num_hidden_state_layers = _resolve_image_hidden_state_count(model)
    if num_hidden_state_layers is None:
        if hook.layer is None:
            raise ValueError(
                "Vision model does not expose enough metadata to resolve "
                "hook.layer_selector automatically. Set hook.layer explicitly "
                "for image caching."
            )
        return int(hook.layer)

    if hook.layer is not None:
        layer_index = int(hook.layer)
    else:
        selector = hook.layer_selector
        if selector == "mid":
            layer_index = int(num_hidden_state_layers // 2)
        elif selector == "late":
            layer_index = int(0.75 * (num_hidden_state_layers - 1))
        elif isinstance(selector, float):
            layer_index = int(round(selector * (num_hidden_state_layers - 1)))
        else:
            raise ValueError(f"Unsupported hook.layer_selector: {selector}")

    if layer_index < 0 or layer_index >= num_hidden_state_layers:
        raise ValueError(
            f"Resolved hidden-state layer index {layer_index} is out of range for "
            f"{num_hidden_state_layers} hidden states."
        )
    return int(layer_index)


def _resolve_image_hidden_state_count(model: torch.nn.Module) -> Optional[int]:
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    if not isinstance(num_hidden_layers, bool) and isinstance(num_hidden_layers, int):
        return int(num_hidden_layers) + 1

    candidate_models = [getattr(model, "timm_model", None), model]
    for candidate_model in candidate_models:
        if candidate_model is None:
            continue
        blocks = getattr(candidate_model, "blocks", None)
        if isinstance(blocks, torch.nn.ModuleList):
            if len(blocks) <= 0:
                continue
            return int(len(blocks))
        if isinstance(blocks, (list, tuple)):
            if len(blocks) <= 0:
                continue
            return int(len(blocks))
    return None


def _validate_patch_size_value(patch_size: Any) -> int:
    if isinstance(patch_size, bool):
        raise ValueError(f"Unsupported patch_size value on model config: {patch_size}.")
    if isinstance(patch_size, int):
        if patch_size <= 0:
            raise ValueError(
                f"Vision model patch_size must be positive, received {patch_size}."
            )
        return int(patch_size)
    if isinstance(patch_size, (list, tuple)):
        if len(patch_size) == 0:
            raise ValueError("Vision model patch_size sequence cannot be empty.")
        first_value = patch_size[0]
        if isinstance(first_value, bool) or not isinstance(first_value, int):
            raise ValueError(
                f"Vision model patch_size values must be integers, received {patch_size}."
            )
        if len(patch_size) > 1 and any(value != first_value for value in patch_size):
            raise ValueError(
                "Only square patch sizes are supported for image caching. "
                f"Received patch_size={patch_size}."
            )
        if first_value <= 0:
            raise ValueError(
                f"Vision model patch_size must be positive, received {patch_size}."
            )
        return int(first_value)
    raise ValueError(
        "Unable to infer vision patch_size from model config. "
        "Expected `patch_size` to be an int or tuple/list of ints."
    )


def _resolve_patch_size(model: torch.nn.Module) -> int:
    patch_size = getattr(model.config, "patch_size", None)
    if patch_size is not None:
        return _validate_patch_size_value(patch_size)

    candidate_models = [getattr(model, "timm_model", None), model]
    for candidate_model in candidate_models:
        if candidate_model is None:
            continue
        patch_embed = getattr(candidate_model, "patch_embed", None)
        candidate_patch_size = getattr(patch_embed, "patch_size", None)
        if candidate_patch_size is not None:
            return _validate_patch_size_value(candidate_patch_size)

    raise ValueError(
        "Unable to infer vision patch_size from model config. "
        "Expected `model.config.patch_size` or `model.patch_embed.patch_size` "
        "to be an int or tuple/list of ints."
    )


def _reshape_image_hidden_state_to_token_sequence(
    hidden_state: torch.Tensor,
    expected_patch_tokens: int,
) -> torch.Tensor:
    if not torch.is_tensor(hidden_state):
        raise ValueError("hidden_state must be a tensor.")
    if hidden_state.ndim == 3:
        num_tokens_total = int(hidden_state.shape[1])
        if num_tokens_total < int(expected_patch_tokens):
            raise ValueError(
                "Vision model produced fewer tokens than expected patch tokens. "
                f"Got num_tokens_total={num_tokens_total}, "
                f"expected_patch_tokens={expected_patch_tokens}."
            )
        return hidden_state
    if hidden_state.ndim == 4:
        batch_size = int(hidden_state.shape[0])
        channel_count = int(hidden_state.shape[1])
        patch_rows = int(hidden_state.shape[2])
        patch_columns = int(hidden_state.shape[3])
        num_tokens_total = int(patch_rows * patch_columns)
        if num_tokens_total != int(expected_patch_tokens):
            raise ValueError(
                "Vision feature map shape does not match the expected number of "
                "patch tokens. "
                f"Got feature_map_shape={tuple(hidden_state.shape)}, "
                f"expected_patch_tokens={expected_patch_tokens}."
            )
        return hidden_state.permute(0, 2, 3, 1).reshape(
            batch_size, num_tokens_total, channel_count
        )
    raise ValueError(
        "Image hidden-state activations must have shape (batch, tokens, d_in) "
        "or (batch, channels, height, width); "
        f"received {type(hidden_state)} with shape "
        f"{getattr(hidden_state, 'shape', None)}."
    )


def _build_image_cache_identity(
    config: LLMBenchmarkConfig,
    storage_dtype: str,
    image_processor_id: str,
    image_processor_revision: Optional[str],
    layer_index: int,
    runtime_device_kind: str,
    model_torch_dtype_resolved: str,
) -> Dict[str, Any]:
    dataset_revision = config.cache.hf_revision or config.data.revision
    if config.data.backend == "hf":
        dataset_identity: Dict[str, Any] = {
            "backend": config.data.backend,
            "dataset_id": config.data.dataset_id,
            "split": config.data.split,
            "revision": dataset_revision,
            "streaming": bool(config.data.streaming),
            "image_field": config.data.image_field,
        }
    else:
        dataset_identity = {
            "backend": config.data.backend,
            "dataset_id": config.data.dataset_id,
            "split": config.data.split,
            "data_dir": config.data.data_dir,
            "streaming": bool(config.data.streaming),
            "image_field": config.data.image_field,
        }

    split_identity: Dict[str, Any] = {
        "split_method": config.cache.split_method,
        "train_fraction": float(config.cache.train_fraction),
        "val_fraction": float(config.cache.val_fraction),
        "test_fraction": float(config.cache.test_fraction),
    }

    cache_identity = {
        "modality": "image",
        "model": {
            "model_id": config.vision.model_id,
            "model_revision": config.vision.revision,
            "image_processor_id": image_processor_id,
            "image_processor_revision": image_processor_revision,
            "hidden_state_layer": int(layer_index),
        },
        "dataset": dataset_identity,
        "cache": {
            "image_batch_size": int(config.cache.image_batch_size),
            "image_size": int(config.cache.image_size),
            "max_images": config.cache.max_images,
            "vectors_per_shard": int(config.cache.vectors_per_shard),
            "compute_dtype": config.cache.compute_dtype,
            "storage_dtype": storage_dtype,
            "cache_seed": config.cache.cache_seed,
            "token_selection": config.cache.token_selection,
            "representation": config.cache.representation,
        },
        "split": split_identity,
        "execution": {
            "runtime_device_kind": runtime_device_kind,
            "model_torch_dtype_resolved": model_torch_dtype_resolved,
        },
    }
    return cache_identity


def _run_image_cache_stage(
    config: LLMBenchmarkConfig,
    output_dir: str,
    runner_overwrite: bool,
) -> Dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModel

    if config.vision is None:
        raise ValueError("vision must be provided when modality='image'.")
    if not isinstance(config.data, ImageDataConfig):
        raise ValueError("data must use ImageDataConfig when modality='image'.")
    if not isinstance(config.cache, ImageActivationCacheConfig):
        raise ValueError(
            "cache must use ImageActivationCacheConfig when modality='image'."
        )

    runtime_device = _resolve_runtime_device(config.device)
    storage_dtype_name = _resolve_storage_dtype(
        config=config, runtime_device=runtime_device
    )
    model_torch_dtype = _resolve_vision_model_torch_dtype(
        config=config,
        runtime_device=runtime_device,
    )
    runtime_device_kind = _resolve_runtime_device_kind(runtime_device)
    model_torch_dtype_resolved = _resolve_dtype_name(model_torch_dtype)
    storage_dtype = _resolve_dtype(storage_dtype_name)
    compute_dtype = _resolve_dtype(str(config.cache.compute_dtype))

    image_processor_id = config.vision.image_processor_id or config.vision.model_id
    image_processor_revision = (
        config.vision.image_processor_revision or config.vision.revision
    )
    use_auth_token = bool(config.vision.use_auth_token)

    try:
        processor = AutoImageProcessor.from_pretrained(
            str(image_processor_id),
            revision=image_processor_revision,
            use_auth_token=use_auth_token,
        )
    except Exception as exception:
        if _is_probable_hf_access_error(exception):
            _raise_hf_access_error(
                exception=exception,
                resource_kind="image processor",
                resource_id=str(image_processor_id),
            )
        raise

    try:
        model = AutoModel.from_pretrained(
            str(config.vision.model_id),
            revision=config.vision.revision,
            torch_dtype=model_torch_dtype,
            low_cpu_mem_usage=True,
            use_auth_token=use_auth_token,
        )
    except Exception as exception:
        if _is_probable_hf_access_error(exception):
            _raise_hf_access_error(
                exception=exception,
                resource_kind="model",
                resource_id=str(config.vision.model_id),
            )
        raise

    model = model.to(runtime_device)
    model.eval()

    layer_index = _resolve_image_hidden_state_layer_index(
        model=model,
        hook=config.hook,
    )
    patch_size = _resolve_patch_size(model)
    image_size = int(config.cache.image_size)
    if image_size % int(patch_size) != 0:
        raise ValueError(
            "cache.image_size must be divisible by the model patch_size. "
            f"Received image_size={config.cache.image_size}, patch_size={patch_size}."
        )
    n_patches_h = image_size // int(patch_size)
    n_patches_w = image_size // int(patch_size)
    expected_patch_tokens = int(n_patches_h * n_patches_w)

    cache_identity = _build_image_cache_identity(
        config=config,
        storage_dtype=storage_dtype_name,
        image_processor_id=str(image_processor_id),
        image_processor_revision=image_processor_revision,
        layer_index=layer_index,
        runtime_device_kind=runtime_device_kind,
        model_torch_dtype_resolved=model_torch_dtype_resolved,
    )
    cache_fingerprint = _compute_cache_fingerprint(cache_identity)

    activations_dir = os.path.join(output_dir, "activations")
    manifest_path = os.path.join(activations_dir, "activations_manifest.json")

    allow_recaching = bool(runner_overwrite or config.cache.cache_overwrite)
    if os.path.isdir(activations_dir) and not os.path.isfile(manifest_path):
        if os.listdir(activations_dir):
            if not allow_recaching:
                raise ValueError(
                    "Activation cache directory exists but manifest is missing: "
                    f"{activations_dir}. Regenerate by setting "
                    "cache.cache_overwrite=true, passing --overwrite, or changing "
                    "output_dir."
                )
            shutil.rmtree(activations_dir)

    existing_manifest = _load_manifest(manifest_path)
    if existing_manifest is not None:
        existing_fingerprint = existing_manifest.get("cache_fingerprint")
        fingerprints_match = existing_fingerprint == cache_fingerprint
        cache_complete = _cache_has_all_shards(existing_manifest, activations_dir)
        schema_version_matches = _manifest_schema_version_matches_expected(
            existing_manifest
        )

        if fingerprints_match and cache_complete and schema_version_matches:
            return {
                "reused": True,
                "manifest_path": manifest_path,
                "cache_fingerprint": cache_fingerprint,
                "message": "Reused existing activation cache.",
            }

        if not allow_recaching:
            if not schema_version_matches:
                schema_version = existing_manifest.get("schema_version")
                raise ValueError(
                    "Activation cache manifest schema_version mismatch with "
                    "cache_overwrite=false and overwrite=false. "
                    f"Expected {ACTIVATIONS_MANIFEST_SCHEMA_VERSION}, "
                    f"received {schema_version}. Regenerate by setting "
                    "cache.cache_overwrite=true or passing --overwrite."
                )
            if not fingerprints_match:
                raise ValueError(
                    "Activation cache fingerprint mismatch with cache_overwrite=false and "
                    "overwrite=false. Regenerate by setting cache.cache_overwrite=true, "
                    "passing --overwrite, or changing output_dir."
                )
            raise ValueError(
                "Activation cache is incomplete or corrupted and recaching is not allowed. "
                "Regenerate by setting cache.cache_overwrite=true or passing --overwrite."
            )

        shutil.rmtree(activations_dir)

    os.makedirs(activations_dir, exist_ok=True)

    image_records, dataset_metadata = _iter_image_records(
        data=config.data, cache=config.cache
    )

    shard_entries: List[Dict[str, Any]] = []
    shard_buffers: Dict[str, List[torch.Tensor]] = {"acts": []}
    shard_index = 0
    shard_row_count = 0
    global_vector_index = 0
    activation_dim: Optional[int] = None
    observed_num_tokens_total: Optional[int] = None
    num_images_processed = 0
    pending_images: List[Image.Image] = []

    def _flush_current_shard() -> None:
        nonlocal shard_index
        nonlocal shard_row_count
        nonlocal global_vector_index

        if shard_row_count <= 0:
            return

        acts_tensor = torch.cat(shard_buffers["acts"], dim=0).to(storage_dtype)
        shard_filename = f"shard_{shard_index:05d}.pt"
        shard_path = os.path.join(activations_dir, shard_filename)
        torch.save({"acts": acts_tensor}, shard_path)

        start_vector_index = global_vector_index
        end_vector_index = global_vector_index + int(acts_tensor.shape[0])
        shard_entries.append(
            {
                "shard_index": int(shard_index),
                "filename": shard_filename,
                "num_vectors": int(acts_tensor.shape[0]),
                "start_vector_index": int(start_vector_index),
                "end_vector_index": int(end_vector_index),
            }
        )

        shard_index += 1
        global_vector_index = end_vector_index
        shard_row_count = 0
        shard_buffers["acts"].clear()

    def _append_batch(acts_flat: torch.Tensor) -> None:
        nonlocal shard_row_count

        start_offset = 0
        rows = int(acts_flat.shape[0])
        while start_offset < rows:
            remaining_capacity = int(config.cache.vectors_per_shard) - shard_row_count
            take_count = min(remaining_capacity, rows - start_offset)
            end_offset = start_offset + take_count

            shard_buffers["acts"].append(acts_flat[start_offset:end_offset].cpu())
            shard_row_count += take_count
            start_offset = end_offset

            if shard_row_count >= int(config.cache.vectors_per_shard):
                _flush_current_shard()

    def _run_pending_images() -> None:
        nonlocal activation_dim
        nonlocal observed_num_tokens_total
        nonlocal num_images_processed

        if not pending_images:
            return

        resized_images = [
            ImageOps.fit(
                image.convert("RGB"),
                (image_size, image_size),
                method=Image.BICUBIC,
            )
            for image in pending_images
        ]

        processor_inputs = processor(images=resized_images, return_tensors="pt")
        model_inputs: Dict[str, Any] = {}
        for key, value in processor_inputs.items():
            if torch.is_tensor(value):
                value = value.to(runtime_device)
                if value.is_floating_point() and value.dtype != model_torch_dtype:
                    value = value.to(dtype=model_torch_dtype)
                model_inputs[key] = value
            else:
                model_inputs[key] = value

        pixel_values = model_inputs.get("pixel_values")
        if not torch.is_tensor(pixel_values) or pixel_values.ndim != 4:
            raise ValueError("Image processor must produce a 4D tensor `pixel_values`.")
        height = int(pixel_values.shape[-2])
        width = int(pixel_values.shape[-1])
        if height != image_size or width != image_size:
            raise ValueError(
                "Image processor output size mismatch: "
                f"pixel_values is {height}x{width}, but config.cache.image_size={image_size}. "
                "This would corrupt patch-token indexing. Fix by setting cache.image_size "
                "to the processor/model expected size (often 224), or ensure the "
                "processor is configured not to resize/crop away from cache.image_size."
            )

        autocast_context = nullcontext()
        if _is_cuda_runtime_device(runtime_device) and compute_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            autocast_context = torch.autocast(
                device_type="cuda",
                dtype=compute_dtype,
            )

        with torch.no_grad():
            with autocast_context:
                outputs = model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, (list, tuple)):
            raise ValueError(
                "Vision model output must include hidden_states when "
                "output_hidden_states=True."
            )
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise ValueError(
                f"Resolved hidden-state layer index {layer_index} is out of range for "
                f"{len(hidden_states)} returned hidden states."
            )

        captured = hidden_states[layer_index]
        captured_tokens = _reshape_image_hidden_state_to_token_sequence(
            hidden_state=captured,
            expected_patch_tokens=expected_patch_tokens,
        )

        current_dim = int(captured_tokens.shape[-1])
        if activation_dim is None:
            activation_dim = current_dim
        elif activation_dim != current_dim:
            raise ValueError(
                "Inconsistent activation dimensionality across batches. Expected "
                f"{activation_dim}, got {current_dim}."
            )

        num_tokens_total = int(captured_tokens.shape[1])
        if observed_num_tokens_total is None:
            observed_num_tokens_total = num_tokens_total
        elif observed_num_tokens_total != num_tokens_total:
            raise ValueError(
                "Inconsistent token count across image batches. Expected "
                f"{observed_num_tokens_total}, got {num_tokens_total}. "
                "This can happen when image_size or processor behavior changes "
                "across batches."
            )
        num_special_tokens = num_tokens_total - expected_patch_tokens
        token_selection = str(config.cache.token_selection)
        if token_selection == "patch_only":
            selected_tokens = (
                captured_tokens
                if num_special_tokens == 0
                else captured_tokens[:, -expected_patch_tokens:, :]
            )
        elif token_selection == "cls_only":
            if num_special_tokens < 1:
                raise ValueError(
                    "token_selection='cls_only' requires at least one special token "
                    f"(CLS). Got num_tokens_total={num_tokens_total}, "
                    f"expected_patch_tokens={expected_patch_tokens}."
                )
            selected_tokens = captured_tokens[:, :1, :]
        elif token_selection == "all":
            selected_tokens = captured_tokens
        else:
            raise ValueError(f"Unsupported token_selection={token_selection}.")

        acts_flat = selected_tokens.reshape(-1, int(captured_tokens.shape[-1]))
        _append_batch(acts_flat=acts_flat)

        num_images_processed += len(pending_images)
        pending_images.clear()

    max_images = config.cache.max_images
    for entry in image_records:
        if max_images is not None and (
            num_images_processed + len(pending_images)
        ) >= int(max_images):
            break
        pil_image = _coerce_image_value_to_pil(
            image_value=entry[str(config.data.image_field)],
            image_field=str(config.data.image_field),
        )
        pending_images.append(pil_image)
        if len(pending_images) >= int(config.cache.image_batch_size):
            _run_pending_images()

    _run_pending_images()
    _flush_current_shard()

    num_vectors_total = int(global_vector_index)
    if num_images_processed <= 0 or num_vectors_total <= 0 or activation_dim is None:
        raise ValueError(
            "Image activation cache produced zero vectors. Check data.image_field, "
            "cache.max_images, and cache.image_batch_size settings."
        )

    split_metadata: Dict[str, Any] = {
        "method": config.cache.split_method,
        "train_fraction": float(config.cache.train_fraction),
        "val_fraction": float(config.cache.val_fraction),
        "test_fraction": float(config.cache.test_fraction),
    }

    if config.cache.split_method == "shard_ranges":
        split_indices = _assign_split_shard_ranges(
            num_shards=len(shard_entries),
            train_fraction=float(config.cache.train_fraction),
            val_fraction=float(config.cache.val_fraction),
        )
        split_metadata["train_shard_indices"] = split_indices["train"]
        split_metadata["val_shard_indices"] = split_indices["val"]
        split_metadata["test_shard_indices"] = split_indices["test"]
    else:
        split_indices_filename = "split_indices.pt"
        split_indices_path = os.path.join(activations_dir, split_indices_filename)
        stored_indices = _build_stored_indices(
            num_vectors_total=num_vectors_total,
            cache=config.cache,
            max_train_samples=config.training.max_train_samples,
            max_val_samples=config.training.max_val_samples,
            max_test_samples=config.training.max_test_samples,
        )
        torch.save(stored_indices, split_indices_path)
        split_metadata["indices_filename"] = split_indices_filename
        split_metadata["train_count"] = int(stored_indices["train"].numel())
        split_metadata["val_count"] = int(stored_indices["val"].numel())
        split_metadata["test_count"] = int(stored_indices["test"].numel())

    manifest: Dict[str, Any] = {
        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
        "modality": "image",
        "cache_fingerprint": cache_fingerprint,
        "model": {
            "id": config.vision.model_id,
            "revision": config.vision.revision,
        },
        "vision_model_id": config.vision.model_id,
        "vision_model_revision": config.vision.revision,
        "image_processor_id": str(image_processor_id),
        "image_processor_revision": image_processor_revision,
        "image_processor": {
            "id": str(image_processor_id),
            "revision": image_processor_revision,
        },
        "representation": {
            "kind": "hidden_state",
            "layer_index": int(layer_index),
        },
        "activation": {
            "d_in": int(activation_dim),
            "activation_kind": "hidden_state",
            "hidden_state_layer": int(layer_index),
        },
        "dataset": dataset_metadata,
        "limits": {
            "max_images": config.cache.max_images,
            "vectors_per_shard": int(config.cache.vectors_per_shard),
        },
        "num_images_processed": int(num_images_processed),
        "image_size": int(config.cache.image_size),
        "patch_size": int(patch_size),
        "n_patches_h": int(n_patches_h),
        "n_patches_w": int(n_patches_w),
        "token_selection": config.cache.token_selection,
        "counts": {
            "num_images_processed": int(num_images_processed),
            "num_vectors_total": int(num_vectors_total),
        },
        "split": split_metadata,
        "seeds": {
            "cache_seed": config.cache.cache_seed,
        },
        "storage": {
            "compute_dtype": config.cache.compute_dtype,
            "storage_dtype": storage_dtype_name,
        },
        "shards": shard_entries,
    }

    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)

    return {
        "reused": False,
        "manifest_path": manifest_path,
        "cache_fingerprint": cache_fingerprint,
        "message": "Image activation cache generated.",
    }


def _default_stage_status(message: str) -> Dict[str, Any]:
    return {
        "status": "done",
        "message": message,
    }


def run_cache_stage(
    config: LLMBenchmarkConfig,
    output_dir: str,
    runner_overwrite: bool,
) -> Dict[str, Any]:
    if config.modality == "image":
        return _run_image_cache_stage(
            config=config,
            output_dir=output_dir,
            runner_overwrite=runner_overwrite,
        )

    from transformers import AutoModel, AutoTokenizer

    runtime_device = _resolve_runtime_device(config.device)
    storage_dtype_name = _resolve_storage_dtype(
        config=config, runtime_device=runtime_device
    )
    model_torch_dtype = _resolve_model_torch_dtype(
        config=config,
        runtime_device=runtime_device,
    )
    runtime_device_kind = _resolve_runtime_device_kind(runtime_device)
    model_torch_dtype_resolved = _resolve_dtype_name(model_torch_dtype)
    storage_dtype = _resolve_dtype(storage_dtype_name)
    compute_dtype = _resolve_dtype(str(config.cache.compute_dtype))

    model = AutoModel.from_pretrained(
        str(config.llm.model_id),
        revision=config.llm.revision,
        torch_dtype=model_torch_dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer_id = config.llm.tokenizer_id or config.llm.model_id
    tokenizer_revision = config.llm.tokenizer_revision or config.llm.revision
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_id),
        revision=tokenizer_revision,
    )

    model = model.to(runtime_device)
    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    hook_module, module_path_resolved, layer_index = resolve_hook_module(
        model=model,
        hook=config.hook,
    )

    cache_identity = _build_cache_identity(
        config=config,
        module_path_resolved=module_path_resolved,
        layer_index=layer_index,
        storage_dtype=storage_dtype_name,
        runtime_device_kind=runtime_device_kind,
        model_torch_dtype_resolved=model_torch_dtype_resolved,
    )
    cache_fingerprint = _compute_cache_fingerprint(cache_identity)

    activations_dir = os.path.join(output_dir, "activations")
    manifest_path = os.path.join(activations_dir, "activations_manifest.json")

    allow_recaching = bool(runner_overwrite or config.cache.cache_overwrite)
    if os.path.isdir(activations_dir) and not os.path.isfile(manifest_path):
        if os.listdir(activations_dir):
            if not allow_recaching:
                raise ValueError(
                    "Activation cache directory exists but manifest is missing: "
                    f"{activations_dir}. Regenerate by setting "
                    "cache.cache_overwrite=true, passing --overwrite, or changing "
                    "output_dir."
                )
            shutil.rmtree(activations_dir)

    existing_manifest = _load_manifest(manifest_path)

    if existing_manifest is not None:
        existing_fingerprint = existing_manifest.get("cache_fingerprint")
        fingerprints_match = existing_fingerprint == cache_fingerprint
        cache_complete = _cache_has_all_shards(existing_manifest, activations_dir)
        schema_version_matches = _manifest_schema_version_matches_expected(
            existing_manifest
        )

        if fingerprints_match and cache_complete and schema_version_matches:
            return {
                "reused": True,
                "manifest_path": manifest_path,
                "cache_fingerprint": cache_fingerprint,
                "message": "Reused existing activation cache.",
            }

        if not allow_recaching:
            if not schema_version_matches:
                schema_version = existing_manifest.get("schema_version")
                raise ValueError(
                    "Activation cache manifest schema_version mismatch with "
                    "cache_overwrite=false and overwrite=false. "
                    f"Expected {ACTIVATIONS_MANIFEST_SCHEMA_VERSION}, "
                    f"received {schema_version}. Regenerate by setting "
                    "cache.cache_overwrite=true or passing --overwrite."
                )
            if not fingerprints_match:
                raise ValueError(
                    "Activation cache fingerprint mismatch with cache_overwrite=false and "
                    "overwrite=false. Regenerate by setting cache.cache_overwrite=true, "
                    "passing --overwrite, or changing output_dir."
                )
            raise ValueError(
                "Activation cache is incomplete or corrupted and recaching is not allowed. "
                "Regenerate by setting cache.cache_overwrite=true or passing --overwrite."
            )

        shutil.rmtree(activations_dir)

    os.makedirs(activations_dir, exist_ok=True)

    text_documents, dataset_metadata = _iter_text_documents(
        data=config.data, cache=config.cache
    )

    eos_token_id = tokenizer.eos_token_id
    doc_separator_setting = str(config.cache.doc_separator)
    doc_separator_id: Optional[int] = None
    doc_separator_effective = doc_separator_setting
    if doc_separator_setting == "eos":
        if eos_token_id is None:
            doc_separator_effective = "none"
            warnings.warn(
                "doc_separator='eos' requested but tokenizer has no eos_token_id; "
                "falling back to doc_separator='none'.",
                RuntimeWarning,
            )
        else:
            doc_separator_id = int(eos_token_id)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad_token_id and no eos_token; cannot apply "
                "pad-token fallback."
            )
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_fallback_applied = True
    else:
        pad_token_fallback_applied = False

    pad_token_id_used = tokenizer.pad_token_id

    captured_outputs: List[torch.Tensor] = []

    def _capture_hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        captured_outputs.append(
            _extract_hook_tensor(output, module_path_resolved).detach()
        )

    hook_handle = hook_module.register_forward_hook(_capture_hook)

    try:
        shard_entries: List[Dict[str, Any]] = []
        shard_index = 0
        shard_row_count = 0
        global_vector_index = 0
        activation_dim: Optional[int] = None

        token_buffer: List[int] = []
        first_document = True
        stop_emitting = False

        tokens_emitted = 0
        sequences_emitted = 0

        pending_blocks: List[List[int]] = []
        next_sequence_index = 0

        shard_buffers: Dict[str, List[torch.Tensor]] = {
            "acts": [],
            "tokens": [],
            "positions": [],
            "sequence_id": [],
        }

        def _flush_current_shard() -> None:
            nonlocal shard_index
            nonlocal shard_row_count
            nonlocal global_vector_index

            if shard_row_count <= 0:
                return

            acts_tensor = torch.cat(shard_buffers["acts"], dim=0).to(storage_dtype)
            shard_filename = f"shard_{shard_index:05d}.pt"
            shard_path = os.path.join(activations_dir, shard_filename)

            shard_payload: Dict[str, torch.Tensor] = {"acts": acts_tensor}
            if config.cache.store_tokens:
                shard_payload["tokens"] = torch.cat(shard_buffers["tokens"], dim=0)
            if config.cache.store_positions:
                shard_payload["positions"] = torch.cat(
                    shard_buffers["positions"], dim=0
                )
            if config.cache.store_sequence_ids:
                shard_payload["sequence_id"] = torch.cat(
                    shard_buffers["sequence_id"], dim=0
                )

            start_vector_index = global_vector_index
            end_vector_index = global_vector_index + int(acts_tensor.shape[0])
            torch.save(shard_payload, shard_path)

            shard_entries.append(
                {
                    "shard_index": int(shard_index),
                    "filename": shard_filename,
                    "num_vectors": int(acts_tensor.shape[0]),
                    "start_vector_index": int(start_vector_index),
                    "end_vector_index": int(end_vector_index),
                }
            )

            shard_index += 1
            global_vector_index = end_vector_index
            shard_row_count = 0
            for key in shard_buffers:
                shard_buffers[key].clear()

        def _append_batch(
            acts_flat: torch.Tensor,
            tokens_flat: torch.Tensor,
            positions_flat: torch.Tensor,
            sequence_ids_flat: torch.Tensor,
        ) -> None:
            nonlocal shard_row_count

            start_offset = 0
            rows = int(acts_flat.shape[0])
            while start_offset < rows:
                remaining_capacity = (
                    int(config.cache.vectors_per_shard) - shard_row_count
                )
                take_count = min(remaining_capacity, rows - start_offset)
                end_offset = start_offset + take_count

                shard_buffers["acts"].append(acts_flat[start_offset:end_offset].cpu())
                if config.cache.store_tokens:
                    shard_buffers["tokens"].append(
                        tokens_flat[start_offset:end_offset].cpu()
                    )
                if config.cache.store_positions:
                    shard_buffers["positions"].append(
                        positions_flat[start_offset:end_offset].cpu()
                    )
                if config.cache.store_sequence_ids:
                    shard_buffers["sequence_id"].append(
                        sequence_ids_flat[start_offset:end_offset].cpu()
                    )

                shard_row_count += take_count
                start_offset = end_offset

                if shard_row_count >= int(config.cache.vectors_per_shard):
                    _flush_current_shard()

        def _run_pending_blocks() -> None:
            nonlocal activation_dim
            nonlocal next_sequence_index

            if not pending_blocks:
                return

            batch_size = len(pending_blocks)
            input_ids = torch.tensor(
                pending_blocks, dtype=torch.long, device=runtime_device
            )
            attention_mask = torch.ones_like(input_ids)

            del captured_outputs[:]
            autocast_context = nullcontext()
            if _is_cuda_runtime_device(runtime_device) and compute_dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                autocast_context = torch.autocast(
                    device_type="cuda",
                    dtype=compute_dtype,
                )

            with torch.no_grad():
                with autocast_context:
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=False,
                    )

            if len(captured_outputs) != 1:
                raise ValueError(
                    "Expected exactly one hook output per forward pass, received "
                    f"{len(captured_outputs)} outputs."
                )

            captured = captured_outputs.pop(0)
            if captured.ndim == 3:
                _, seq_len, d_in = captured.shape
                acts_flat = captured.reshape(-1, int(d_in))
                tokens_flat = input_ids.reshape(-1)
                positions_flat = (
                    torch.arange(seq_len, device=input_ids.device)
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                    .reshape(-1)
                )
                sequence_ids_flat = (
                    torch.arange(
                        next_sequence_index,
                        next_sequence_index + batch_size,
                        device=input_ids.device,
                    )
                    .unsqueeze(1)
                    .repeat(1, seq_len)
                    .reshape(-1)
                )
            elif captured.ndim == 2:
                _, d_in = captured.shape
                acts_flat = captured
                tokens_flat = input_ids[:, 0]
                positions_flat = torch.zeros(
                    batch_size, dtype=torch.long, device=input_ids.device
                )
                sequence_ids_flat = torch.arange(
                    next_sequence_index,
                    next_sequence_index + batch_size,
                    device=input_ids.device,
                )
            else:
                raise ValueError(
                    "Unsupported hook output shape at module "
                    f"{module_path_resolved}': expected rank 2 or 3, got {tuple(captured.shape)}"
                )

            current_dim = int(acts_flat.shape[-1])
            if activation_dim is None:
                activation_dim = current_dim
            elif activation_dim != current_dim:
                raise ValueError(
                    "Inconsistent activation dimensionality across batches. Expected "
                    f"{activation_dim}, got {current_dim}."
                )

            _append_batch(
                acts_flat=acts_flat,
                tokens_flat=tokens_flat,
                positions_flat=positions_flat,
                sequence_ids_flat=sequence_ids_flat,
            )

            next_sequence_index += batch_size
            pending_blocks.clear()

        for document_text in text_documents:
            encoded_tokens = tokenizer.encode(
                str(document_text), add_special_tokens=False
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

            while len(token_buffer) >= int(config.cache.max_seq_len):
                if config.cache.max_sequences is not None and sequences_emitted >= int(
                    config.cache.max_sequences
                ):
                    stop_emitting = True
                    break
                if config.cache.max_tokens is not None:
                    projected_tokens = tokens_emitted + int(config.cache.max_seq_len)
                    if projected_tokens > int(config.cache.max_tokens):
                        stop_emitting = True
                        break

                block_tokens = token_buffer[: int(config.cache.max_seq_len)]
                del token_buffer[: int(config.cache.max_seq_len)]

                pending_blocks.append(block_tokens)
                tokens_emitted += int(config.cache.max_seq_len)
                sequences_emitted += 1

                if len(pending_blocks) >= int(config.cache.llm_batch_size):
                    _run_pending_blocks()

            if stop_emitting:
                break

        _run_pending_blocks()

        tokens_dropped_tail = len(token_buffer)
        _flush_current_shard()

        num_vectors_total = int(global_vector_index)
        if sequences_emitted == 0 or num_vectors_total == 0 or activation_dim is None:
            raise ValueError(
                "Activation cache produced zero packed blocks. This usually means "
                "the dataset is too small for max_seq_len, or max_tokens/max_sequences "
                "are too small. "
                f"Got max_seq_len={config.cache.max_seq_len}, "
                f"max_tokens={config.cache.max_tokens}, "
                f"max_sequences={config.cache.max_sequences}."
            )

        split_metadata: Dict[str, Any] = {
            "method": config.cache.split_method,
            "train_fraction": float(config.cache.train_fraction),
            "val_fraction": float(config.cache.val_fraction),
            "test_fraction": float(config.cache.test_fraction),
        }

        if config.cache.split_method == "shard_ranges":
            split_indices = _assign_split_shard_ranges(
                num_shards=len(shard_entries),
                train_fraction=float(config.cache.train_fraction),
                val_fraction=float(config.cache.val_fraction),
            )
            split_metadata["train_shard_indices"] = split_indices["train"]
            split_metadata["val_shard_indices"] = split_indices["val"]
            split_metadata["test_shard_indices"] = split_indices["test"]
        else:
            split_indices_filename = "split_indices.pt"
            split_indices_path = os.path.join(activations_dir, split_indices_filename)
            stored_indices = _build_stored_indices(
                num_vectors_total=num_vectors_total,
                cache=config.cache,
                max_train_samples=config.training.max_train_samples,
                max_val_samples=config.training.max_val_samples,
                max_test_samples=config.training.max_test_samples,
            )
            torch.save(stored_indices, split_indices_path)
            split_metadata["indices_filename"] = split_indices_filename
            split_metadata["train_count"] = int(stored_indices["train"].numel())
            split_metadata["val_count"] = int(stored_indices["val"].numel())
            split_metadata["test_count"] = int(stored_indices["test"].numel())

        manifest: Dict[str, Any] = {
            "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
            "cache_fingerprint": cache_fingerprint,
            "model": {
                "id": config.llm.model_id,
                "revision": config.llm.revision,
            },
            "tokenizer": {
                "id": tokenizer_id,
                "revision": tokenizer_revision,
            },
            "hook": {
                "hook_name": config.hook.hook_name,
                "activation_kind": "module_output",
                "layer": int(layer_index),
                "module_path_resolved": module_path_resolved,
            },
            "activation": {
                "d_in": int(activation_dim),
                "activation_kind": "module_output",
            },
            "dataset": dataset_metadata,
            "limits": {
                "max_tokens": config.cache.max_tokens,
                "max_sequences": config.cache.max_sequences,
                "vectors_per_shard": int(config.cache.vectors_per_shard),
            },
            "counts": {
                "tokens_emitted": int(tokens_emitted),
                "sequences_emitted": int(sequences_emitted),
                "tokens_dropped_tail": int(tokens_dropped_tail),
                "num_vectors_total": int(num_vectors_total),
            },
            "tokenization": {
                "mode": "packed_fixed_length",
                "max_seq_len": int(config.cache.max_seq_len),
                "doc_separator": doc_separator_effective,
                "doc_separator_id": doc_separator_id,
                "pad_token_fallback_applied": bool(pad_token_fallback_applied),
                "pad_token_id_used": pad_token_id_used,
            },
            "split": split_metadata,
            "seeds": {
                "cache_seed": config.cache.cache_seed,
            },
            "storage": {
                "compute_dtype": config.cache.compute_dtype,
                "storage_dtype": storage_dtype_name,
            },
            "store_tokens": bool(config.cache.store_tokens),
            "store_positions": bool(config.cache.store_positions),
            "store_sequence_ids": bool(config.cache.store_sequence_ids),
            "shards": shard_entries,
        }

        with open(manifest_path, "w") as file:
            json.dump(manifest, file, indent=2)

        return {
            "reused": False,
            "manifest_path": manifest_path,
            "cache_fingerprint": cache_fingerprint,
            "message": "Activation cache generated.",
        }
    finally:
        hook_handle.remove()
