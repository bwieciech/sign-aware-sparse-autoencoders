import base64
import heapq
import io
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageOps

from sgnsae.experiments.llm_benchmark.cache import (
    _coerce_image_value_to_pil,
    _extract_hook_tensor,
    _iter_image_records,
    _iter_text_documents,
    _reshape_image_hidden_state_to_token_sequence,
    _resolve_image_hidden_state_layer_index,
    _resolve_patch_size,
    resolve_hook_module,
)
from sgnsae.experiments.llm_benchmark.config import (
    ImageActivationCacheConfig,
    ImageDataConfig,
    LLMBenchmarkConfig,
    TextDataConfig,
)
from sgnsae.experiments.llm_benchmark.metrics import (
    build_and_persist_comparison_summary,
)
from sgnsae.experiments.llm_benchmark.plotting import (
    ensure_summary_has_matched_l0_anchors,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    LATENT_EXAMPLES_INDEX_SCHEMA_VERSION,
    LATENT_EXAMPLES_SCHEMA_VERSION,
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


def subrun_latent_examples_path(subrun_dir: str) -> str:
    return os.path.join(subrun_dir, "latent_examples.json")


def run_level_latent_examples_index_path(output_dir: str) -> str:
    return os.path.join(output_dir, "latent_examples_index.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str) -> Any:
    with open(path, "r") as file:
        return json.load(file)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        converted = float(value)
        if math.isfinite(converted):
            return converted
    return None


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


def _output_subdir_path(output_dir: str, relative_dir: str) -> str:
    normalized = relative_dir.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        raise ValueError("Resolved config contains an empty relative_subrun_dir.")
    base_directory = os.path.abspath(output_dir)
    candidate_path = os.path.abspath(os.path.join(base_directory, *parts))
    try:
        common_path = os.path.commonpath([base_directory, candidate_path])
    except ValueError as exception:
        raise ValueError(
            f"Resolved relative_subrun_dir escapes output_dir: {relative_dir}"
        ) from exception
    if common_path != base_directory:
        raise ValueError(
            f"Resolved relative_subrun_dir escapes output_dir: {relative_dir}"
        )
    return candidate_path


def _resolve_seed_filter(
    config: LLMBenchmarkConfig, resolved_config: Dict[str, Any]
) -> List[int]:
    if config.latent_examples.seeds is not None:
        return sorted([int(seed) for seed in config.latent_examples.seeds])
    seeds = resolved_config.get("seeds")
    if isinstance(seeds, list) and seeds:
        first_seed = _safe_int(seeds[0])
        if first_seed is not None:
            return [int(first_seed)]
    return []


def _load_or_build_summary(
    resolved_config: Dict[str, Any], output_dir: str
) -> Dict[str, Any]:
    summary = build_and_persist_comparison_summary(
        resolved_config=resolved_config,
        output_dir=output_dir,
    )
    return ensure_summary_has_matched_l0_anchors(
        output_dir=output_dir,
        summary=summary,
    )


def _summary_entries_with_metrics(
    summary: Dict[str, Any],
    seed_filter: Sequence[int],
) -> List[Dict[str, Any]]:
    filter_set = set(int(seed) for seed in seed_filter)
    subruns = summary.get("subruns")
    if not isinstance(subruns, list):
        return []
    entries: List[Dict[str, Any]] = []
    for entry in subruns:
        if not isinstance(entry, dict):
            continue
        relative_dir = entry.get("relative_dir")
        variant_slug = entry.get("variant_slug")
        seed = _safe_int(entry.get("seed"))
        mse = _safe_float(entry.get("mse"))
        mean_l0 = _safe_float(entry.get("mean_l0"))
        if (
            not isinstance(relative_dir, str)
            or not relative_dir
            or not isinstance(variant_slug, str)
            or seed is None
            or mse is None
            or mean_l0 is None
        ):
            continue
        if filter_set and int(seed) not in filter_set:
            continue
        entries.append(
            {
                "relative_dir": relative_dir,
                "variant_slug": variant_slug,
                "seed": int(seed),
                "mse": float(mse),
                "mean_l0": float(mean_l0),
                "sweep_param": entry.get("sweep_param"),
                "sweep_value": entry.get("sweep_value"),
            }
        )
    return entries


def _matched_anchor_entries_from_summary(
    summary: Dict[str, Any], anchor_l0: float, seed_filter: Sequence[int]
) -> List[Dict[str, Any]]:
    matched = summary.get("matched_l0_anchors")
    if not isinstance(matched, dict):
        return []
    targets = matched.get("targets")
    selected = matched.get("selected")
    if not isinstance(targets, list) or not isinstance(selected, list):
        return []
    rounded_anchor = int(round(float(anchor_l0)))
    if abs(float(anchor_l0) - float(rounded_anchor)) > 1e-9:
        return []
    parsed_targets: List[int] = []
    for target in targets:
        target_int = _safe_int(target)
        if target_int is None:
            return []
        parsed_targets.append(int(target_int))
    if rounded_anchor not in parsed_targets:
        return []
    filter_set = set(int(seed) for seed in seed_filter)
    selected_entries: List[Dict[str, Any]] = []
    for entry in selected:
        if not isinstance(entry, dict):
            continue
        entry_target = _safe_int(entry.get("target"))
        if entry_target != rounded_anchor:
            continue
        relative_dir = entry.get("relative_dir")
        variant_slug = entry.get("variant_slug")
        seed = _safe_int(entry.get("seed"))
        mse = _safe_float(entry.get("selected_mse"))
        mean_l0 = _safe_float(entry.get("selected_mean_l0"))
        if (
            not isinstance(relative_dir, str)
            or not relative_dir
            or not isinstance(variant_slug, str)
            or seed is None
            or mse is None
            or mean_l0 is None
        ):
            continue
        if filter_set and int(seed) not in filter_set:
            continue
        selected_entries.append(
            {
                "relative_dir": relative_dir,
                "variant_slug": variant_slug,
                "seed": int(seed),
                "mse": float(mse),
                "mean_l0": float(mean_l0),
                "sweep_param": entry.get("selected_sweep_param"),
                "sweep_value": entry.get("selected_sweep_value"),
            }
        )
    return selected_entries


def _select_matched_anchor_entries(
    entries: Sequence[Dict[str, Any]], anchor_l0: float
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault((str(entry["variant_slug"]), int(entry["seed"])), []).append(
            dict(entry)
        )

    selected: List[Dict[str, Any]] = []
    for variant_slug, seed in sorted(
        grouped.keys(), key=lambda value: (value[0], value[1])
    ):
        candidates = grouped[(variant_slug, seed)]

        def _tie_key(entry: Dict[str, Any]) -> Tuple[float, float, float, str]:
            distance = abs(float(entry["mean_l0"]) - float(anchor_l0))
            mse = float(entry["mse"])
            sweep_value = entry.get("sweep_value")
            sweep_sort_value = float("inf")
            if isinstance(sweep_value, (int, float)) and not isinstance(
                sweep_value, bool
            ):
                sweep_sort_value = float(sweep_value)
            relative_dir = str(entry.get("relative_dir", ""))
            return (distance, mse, sweep_sort_value, relative_dir)

        selected.append(min(candidates, key=_tie_key))
    return selected


def _select_best_mse_entries(
    entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault((str(entry["variant_slug"]), int(entry["seed"])), []).append(
            dict(entry)
        )

    selected: List[Dict[str, Any]] = []
    for variant_slug, seed in sorted(
        grouped.keys(), key=lambda value: (value[0], value[1])
    ):
        candidates = grouped[(variant_slug, seed)]

        def _tie_key(entry: Dict[str, Any]) -> Tuple[float, float, str]:
            mse = float(entry["mse"])
            sweep_value = entry.get("sweep_value")
            sweep_sort_value = float("inf")
            if isinstance(sweep_value, (int, float)) and not isinstance(
                sweep_value, bool
            ):
                sweep_sort_value = float(sweep_value)
            relative_dir = str(entry.get("relative_dir", ""))
            return (mse, sweep_sort_value, relative_dir)

        selected.append(min(candidates, key=_tie_key))
    return selected


def select_target_subruns_for_examples(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    output_dir: str,
) -> List[Dict[str, Any]]:
    subruns = resolved_config.get("subruns")
    if not isinstance(subruns, list):
        return []
    all_subruns: List[Dict[str, Any]] = [dict(subrun) for subrun in subruns]
    seed_filter = _resolve_seed_filter(config=config, resolved_config=resolved_config)
    seed_filter_set = set(seed_filter)
    filtered_subruns = [
        subrun
        for subrun in all_subruns
        if not seed_filter_set or int(subrun["seed"]) in seed_filter_set
    ]

    target_mode = str(config.latent_examples.target_subruns)
    if target_mode == "explicit":
        filtered_subruns.sort(
            key=lambda subrun: (
                str(subrun.get("variant_slug", "")),
                int(subrun.get("seed", 0)),
                str(subrun.get("sweep_param", "")),
                str(subrun.get("relative_subrun_dir", "")),
            )
        )
        return filtered_subruns

    summary = _load_or_build_summary(
        resolved_config=resolved_config, output_dir=output_dir
    )
    selected_entries: List[Dict[str, Any]]
    if target_mode == "matched_l0_anchor":
        selected_entries = _matched_anchor_entries_from_summary(
            summary=summary,
            anchor_l0=float(config.latent_examples.anchor_l0),
            seed_filter=seed_filter,
        )
        if not selected_entries:
            metric_entries = _summary_entries_with_metrics(
                summary=summary, seed_filter=seed_filter
            )
            if not metric_entries:
                raise ValueError(
                    "Examples stage requires evaluation metrics in "
                    "comparison_summary.json. Run the eval stage first or use "
                    "latent_examples.target_subruns='explicit'."
                )
            selected_entries = _select_matched_anchor_entries(
                entries=metric_entries,
                anchor_l0=float(config.latent_examples.anchor_l0),
            )
    elif target_mode == "best_mse":
        metric_entries = _summary_entries_with_metrics(
            summary=summary, seed_filter=seed_filter
        )
        if not metric_entries:
            raise ValueError(
                "Examples stage requires evaluation metrics in "
                "comparison_summary.json. Run the eval stage first or use "
                "latent_examples.target_subruns='explicit'."
            )
        selected_entries = _select_best_mse_entries(entries=metric_entries)
    else:
        raise ValueError(
            f"Unsupported latent_examples.target_subruns value: {target_mode}."
        )

    selected_dirs = [str(entry["relative_dir"]) for entry in selected_entries]
    selected_set = set(selected_dirs)
    selected_subruns = [
        subrun
        for subrun in filtered_subruns
        if str(subrun["relative_subrun_dir"]) in selected_set
    ]
    selected_subruns.sort(
        key=lambda subrun: (
            str(subrun.get("variant_slug", "")),
            int(subrun.get("seed", 0)),
            str(subrun.get("sweep_param", "")),
            str(subrun.get("relative_subrun_dir", "")),
        )
    )
    return selected_subruns


def _resolve_text_backbone_dtype(
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


def _resolve_vision_backbone_dtype(
    config: LLMBenchmarkConfig, runtime_device: str
) -> torch.dtype:
    dtype_name = str(config.vision.torch_dtype)
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
    raise ValueError(f"Unsupported vision.torch_dtype value: {dtype_name}.")


def _flatten_single_instance_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 3 and int(latents.shape[0]) == 1:
        return latents[0]
    if latents.ndim == 2:
        return latents
    return latents.reshape(-1, latents.shape[-1])


def _decode_token_pieces(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    pieces: List[str] = []
    for token_id in token_ids:
        pieces.append(
            str(
                tokenizer.decode(
                    [int(token_id)],
                    clean_up_tokenization_spaces=False,
                )
            )
        )
    return pieces


def render_highlighted_text(
    token_pieces: Sequence[str],
    active_token_indices: Sequence[int],
    left_marker: str,
    right_marker: str,
) -> str:
    active_index_set = {
        int(index)
        for index in active_token_indices
        if isinstance(index, int) and 0 <= int(index) < len(token_pieces)
    }
    output_parts: List[str] = []
    in_highlight = False
    for index, piece in enumerate(token_pieces):
        is_active = index in active_index_set
        if is_active and not in_highlight:
            output_parts.append(str(left_marker))
            in_highlight = True
        if not is_active and in_highlight:
            output_parts.append(str(right_marker))
            in_highlight = False
        output_parts.append(str(piece))
    if in_highlight:
        output_parts.append(str(right_marker))
    return "".join(output_parts)


def _select_latent_indices_from_statistics(
    max_abs_by_latent: torch.Tensor,
    activation_count_by_latent: torch.Tensor,
    config: LLMBenchmarkConfig,
    d_hidden: int,
) -> List[int]:
    max_latents = min(int(config.latent_examples.max_latents), int(d_hidden))
    if max_latents <= 0:
        return []

    if config.latent_examples.latent_selection == "explicit":
        explicit_latents = config.latent_examples.explicit_latents or []
        selected: List[int] = []
        seen: set[int] = set()
        for latent_index in explicit_latents:
            normalized = int(latent_index)
            if normalized < 0 or normalized >= int(d_hidden):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(normalized)
            if len(selected) >= max_latents:
                break
        return selected

    scores: List[Tuple[float, int]] = []
    if config.latent_examples.latent_selection == "top_by_freq":
        source = activation_count_by_latent.reshape(-1).to(
            dtype=torch.float64, device="cpu"
        )
        for latent_index in range(int(d_hidden)):
            score = float(source[latent_index].item())
            if score <= 0.0:
                continue
            scores.append((score, latent_index))
    else:
        source = max_abs_by_latent.reshape(-1).to(dtype=torch.float64, device="cpu")
        activation_threshold = float(config.latent_examples.activation_threshold)
        for latent_index in range(int(d_hidden)):
            score = float(source[latent_index].item())
            if not math.isfinite(score):
                continue
            if score <= activation_threshold:
                continue
            scores.append((score, latent_index))

    scores.sort(key=lambda entry: (-entry[0], entry[1]))
    return [int(entry[1]) for entry in scores[:max_latents]]


def _push_bounded_example_heap(
    heaps_by_latent: Dict[int, List[Tuple[float, int, Any]]],
    latent_index: int,
    score: float,
    payload: Any,
    max_examples_per_latent: int,
    sequence_order: int,
) -> None:
    entry = (float(score), -int(sequence_order), payload)
    heap = heaps_by_latent.setdefault(int(latent_index), [])
    if len(heap) < int(max_examples_per_latent):
        heapq.heappush(heap, entry)
        return
    if entry[0] > heap[0][0] or (entry[0] == heap[0][0] and entry[1] > heap[0][1]):
        heapq.heapreplace(heap, entry)


def _extract_selected_token_activations(
    latents: torch.Tensor,
    latent_index: int,
    threshold: float,
) -> torch.Tensor:
    latent_values = latents[:, int(latent_index)]
    active_positions = torch.nonzero(latent_values.abs() > float(threshold)).reshape(-1)
    return active_positions


def _finalize_text_examples(
    heaps_by_latent: Dict[int, List[Tuple[float, int, Any]]],
) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for latent_index in sorted(heaps_by_latent.keys()):
        ranked_entries = sorted(
            heaps_by_latent[latent_index],
            key=lambda entry: (-entry[0], -entry[1]),
        )
        output[str(latent_index)] = [str(entry[2]) for entry in ranked_entries]
    return output


def _image_normalization_stats(
    processor: Any, channels: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_raw = getattr(processor, "image_mean", None)
    std_raw = getattr(processor, "image_std", None)

    def _normalize_values(raw: Any, default_value: float) -> List[float]:
        if raw is None:
            return [default_value for _ in range(channels)]
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return [float(raw) for _ in range(channels)]
        if isinstance(raw, (list, tuple)):
            if len(raw) == 1:
                value = float(raw[0])
                return [value for _ in range(channels)]
            if len(raw) == channels:
                return [float(value) for value in raw]
        raise ValueError(
            "Image processor normalization stats must be scalar or length-C list/tuple."
        )

    mean_values = _normalize_values(mean_raw, 0.0)
    std_values = _normalize_values(std_raw, 1.0)
    mean = torch.tensor(mean_values, dtype=torch.float32).reshape(channels, 1, 1)
    std = torch.tensor(std_values, dtype=torch.float32).reshape(channels, 1, 1)
    return mean, std


def _pixel_values_to_pil_rgb_image(
    pixel_values: torch.Tensor, processor: Any
) -> Image.Image:
    if pixel_values.ndim != 3:
        raise ValueError(
            f"Expected pixel_values rank=3 (C,H,W), received shape {tuple(pixel_values.shape)}."
        )
    channels = int(pixel_values.shape[0])
    mean, std = _image_normalization_stats(processor=processor, channels=channels)
    image_tensor = (
        pixel_values.detach().to(dtype=torch.float32, device="cpu") * std + mean
    ).clamp(0.0, 1.0)
    if channels == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    if int(image_tensor.shape[0]) != 3:
        raise ValueError(
            f"Expected image tensor with 3 channels after normalization, got {tuple(image_tensor.shape)}."
        )
    array = (
        image_tensor.permute(1, 2, 0).mul(255.0).round().to(dtype=torch.uint8).numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _validate_image_processor_pixel_values(
    model_inputs: Dict[str, Any], image_size: int
) -> None:
    pixel_values = model_inputs.get("pixel_values")
    if not torch.is_tensor(pixel_values) or pixel_values.ndim != 4:
        raise ValueError("Image processor must produce a 4D tensor `pixel_values`.")
    height = int(pixel_values.shape[-2])
    width = int(pixel_values.shape[-1])
    if height != image_size or width != image_size:
        raise ValueError(
            "Image processor output size mismatch while generating latent examples: "
            f"pixel_values is {height}x{width}, but manifest image_size={image_size}. "
            "This would corrupt patch-index alignment. Ensure the processor is configured "
            "to output the manifest image_size."
        )


def compute_patch_crop_bounds(
    patch_index: int,
    n_patches_w: int,
    patch_size: int,
    image_size: int,
) -> Tuple[int, int, int, int]:
    if (
        isinstance(patch_index, bool)
        or isinstance(n_patches_w, bool)
        or isinstance(patch_size, bool)
        or isinstance(image_size, bool)
    ):
        raise ValueError("Patch geometry values must be integers.")
    if patch_index < 0:
        raise ValueError(f"patch_index must be >= 0, received {patch_index}.")
    if n_patches_w <= 0 or patch_size <= 0 or image_size <= 0:
        raise ValueError(
            "n_patches_w, patch_size, and image_size must be positive integers."
        )

    patch_row = int(patch_index) // int(n_patches_w)
    patch_col = int(patch_index) % int(n_patches_w)
    left = int(patch_col * patch_size)
    top = int(patch_row * patch_size)
    right = int(left + patch_size)
    bottom = int(top + patch_size)
    if right > int(image_size) or bottom > int(image_size):
        raise ValueError(
            "Patch index resolves outside image bounds: "
            f"patch_index={patch_index}, n_patches_w={n_patches_w}, "
            f"patch_size={patch_size}, image_size={image_size}."
        )
    return left, top, right, bottom


def crop_patch_image(
    image: Image.Image,
    patch_index: int,
    n_patches_w: int,
    patch_size: int,
    image_size: int,
) -> Image.Image:
    bounds = compute_patch_crop_bounds(
        patch_index=patch_index,
        n_patches_w=n_patches_w,
        patch_size=patch_size,
        image_size=image_size,
    )
    return image.crop(bounds)


def encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _finalize_image_examples(
    heaps_by_latent: Dict[int, List[Tuple[float, int, Any]]],
) -> Dict[str, List[List[str]]]:
    output: Dict[str, List[List[str]]] = {}
    for latent_index in sorted(heaps_by_latent.keys()):
        ranked_entries = sorted(
            heaps_by_latent[latent_index],
            key=lambda entry: (-entry[0], -entry[1]),
        )
        output[str(latent_index)] = [list(entry[2]) for entry in ranked_entries]
    return output


def _load_sae_model_for_subrun(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    output_dir: str,
) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    runtime_device = resolve_runtime_device(config.device)
    manifest_path = resolve_manifest_path(output_dir=output_dir)
    manifest = load_activation_manifest(manifest_path=manifest_path)
    sae_model, _, _, _ = build_model_for_subrun(
        config=config,
        resolved_config=resolved_config,
        subrun=subrun,
        manifest=manifest,
        device=runtime_device,
    )

    subrun_dir = _output_subdir_path(
        output_dir=output_dir,
        relative_dir=str(subrun["relative_subrun_dir"]),
    )
    checkpoint_path = os.path.join(subrun_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(
            f"Missing trained checkpoint required for examples stage: {checkpoint_path}."
        )

    checkpoint_payload = torch.load(checkpoint_path, map_location=runtime_device)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"Invalid checkpoint payload at {checkpoint_path}.")
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a 'state_dict' dictionary."
        )
    sae_model.load_state_dict(state_dict)
    sae_model = sae_model.to(runtime_device)
    sae_model.eval()
    return sae_model, manifest, runtime_device


def _generate_text_examples(
    config: LLMBenchmarkConfig,
    sae_model: torch.nn.Module,
    runtime_device: str,
    d_hidden: int,
) -> Tuple[Dict[str, List[str]], List[int], int]:
    from transformers import AutoModel, AutoTokenizer

    if config.llm is None:
        raise ValueError("llm must be provided for text latent examples.")
    if not isinstance(config.data, TextDataConfig):
        raise ValueError("Text latent examples require TextDataConfig.")

    tokenizer_id = config.llm.tokenizer_id or config.llm.model_id
    tokenizer_revision = config.llm.tokenizer_revision or config.llm.revision
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_id),
        revision=tokenizer_revision,
    )
    text_model = AutoModel.from_pretrained(
        str(config.llm.model_id),
        revision=config.llm.revision,
        torch_dtype=_resolve_text_backbone_dtype(
            config=config,
            runtime_device=runtime_device,
        ),
        low_cpu_mem_usage=True,
    )
    text_model = text_model.to(runtime_device)
    text_model.eval()
    if hasattr(text_model, "config") and hasattr(text_model.config, "use_cache"):
        text_model.config.use_cache = False

    hook_module, module_path_resolved, _ = resolve_hook_module(
        model=text_model,
        hook=config.hook,
    )

    max_seq_len = int(config.cache.max_seq_len)
    sample_size = int(config.latent_examples.sample_size)
    activation_threshold = float(config.latent_examples.activation_threshold)
    max_examples_per_latent = int(config.latent_examples.max_examples_per_latent)
    marker_left, marker_right = config.latent_examples.highlight_markers

    def _iter_token_samples() -> List[List[int]]:
        text_documents, _ = _iter_text_documents(data=config.data, cache=config.cache)
        collected: List[List[int]] = []
        for document_text in text_documents:
            if len(collected) >= sample_size:
                break
            encoded = tokenizer.encode(str(document_text), add_special_tokens=False)
            if not isinstance(encoded, list) or not encoded:
                continue
            truncated = [int(token_id) for token_id in encoded[:max_seq_len]]
            if not truncated:
                continue
            collected.append(truncated)
        return collected

    token_samples = _iter_token_samples()
    if not token_samples:
        return {}, [], 0

    captured_outputs: List[torch.Tensor] = []

    def _capture_hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        captured_outputs.append(
            _extract_hook_tensor(output, module_path_resolved).detach()
        )

    hook_handle = hook_module.register_forward_hook(_capture_hook)
    try:
        max_abs_by_latent = torch.full(
            (int(d_hidden),), float("-inf"), dtype=torch.float64
        )
        activation_count_by_latent = torch.zeros((int(d_hidden),), dtype=torch.float64)
        processed_samples = 0

        with torch.no_grad():
            for token_ids in token_samples:
                input_ids = torch.tensor(
                    [token_ids],
                    dtype=torch.long,
                    device=runtime_device,
                )
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                captured_outputs.clear()
                text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=False,
                )
                if len(captured_outputs) != 1:
                    raise ValueError(
                        "Expected exactly one text hook output per forward pass, "
                        f"received {len(captured_outputs)} outputs."
                    )
                hook_output = captured_outputs.pop(0)
                if hook_output.ndim == 3:
                    activations = hook_output[0]
                elif hook_output.ndim == 2:
                    activations = hook_output
                else:
                    raise ValueError(
                        "Unsupported text hook output shape: "
                        f"{tuple(hook_output.shape)}."
                    )
                output = sae_model(
                    activations.to(device=runtime_device, dtype=torch.float32)
                )
                latents = _flatten_single_instance_latents(output["latents"])
                if int(latents.shape[1]) != int(d_hidden):
                    raise ValueError(
                        "Latent width mismatch while generating text examples. "
                        f"Expected {d_hidden}, got {int(latents.shape[1])}."
                    )
                latents_abs = torch.nan_to_num(
                    latents.abs(), nan=0.0, posinf=0.0, neginf=0.0
                )
                max_abs_by_latent = torch.maximum(
                    max_abs_by_latent,
                    latents_abs.max(dim=0).values.to(dtype=torch.float64, device="cpu"),
                )
                activation_count_by_latent += (
                    (latents_abs > activation_threshold)
                    .sum(dim=0)
                    .to(dtype=torch.float64, device="cpu")
                )
                processed_samples += 1

        selected_latents = _select_latent_indices_from_statistics(
            max_abs_by_latent=max_abs_by_latent,
            activation_count_by_latent=activation_count_by_latent,
            config=config,
            d_hidden=d_hidden,
        )
        if not selected_latents:
            return {}, [], processed_samples

        heaps_by_latent: Dict[int, List[Tuple[float, int, Any]]] = {
            int(latent_index): [] for latent_index in selected_latents
        }
        selected_latent_tensor = torch.tensor(
            selected_latents, dtype=torch.long, device=runtime_device
        )
        sequence_order = 0
        with torch.no_grad():
            for token_ids in token_samples:
                input_ids = torch.tensor(
                    [token_ids],
                    dtype=torch.long,
                    device=runtime_device,
                )
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                captured_outputs.clear()
                text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=False,
                )
                if len(captured_outputs) != 1:
                    raise ValueError(
                        "Expected exactly one text hook output per forward pass, "
                        f"received {len(captured_outputs)} outputs."
                    )
                hook_output = captured_outputs.pop(0)
                if hook_output.ndim == 3:
                    activations = hook_output[0]
                elif hook_output.ndim == 2:
                    activations = hook_output
                else:
                    raise ValueError(
                        "Unsupported text hook output shape: "
                        f"{tuple(hook_output.shape)}."
                    )
                output = sae_model(
                    activations.to(device=runtime_device, dtype=torch.float32)
                )
                latents = _flatten_single_instance_latents(output["latents"])
                latents_selected = latents.index_select(
                    dim=1, index=selected_latent_tensor
                )
                latents_selected_cpu = latents_selected.detach().to(
                    dtype=torch.float32, device="cpu"
                )
                token_pieces = _decode_token_pieces(
                    tokenizer=tokenizer, token_ids=token_ids
                )

                for local_latent_index, latent_index in enumerate(selected_latents):
                    active_positions = _extract_selected_token_activations(
                        latents=latents_selected_cpu,
                        latent_index=int(local_latent_index),
                        threshold=activation_threshold,
                    )
                    if int(active_positions.numel()) == 0:
                        continue
                    highlighted = render_highlighted_text(
                        token_pieces=token_pieces,
                        active_token_indices=[
                            int(index) for index in active_positions.tolist()
                        ],
                        left_marker=str(marker_left),
                        right_marker=str(marker_right),
                    )
                    latent_values = latents_selected_cpu[:, int(local_latent_index)]
                    score = float(
                        latent_values.index_select(0, active_positions)
                        .abs()
                        .max()
                        .item()
                    )
                    _push_bounded_example_heap(
                        heaps_by_latent=heaps_by_latent,
                        latent_index=int(latent_index),
                        score=score,
                        payload=highlighted,
                        max_examples_per_latent=max_examples_per_latent,
                        sequence_order=sequence_order,
                    )
                sequence_order += 1
    finally:
        hook_handle.remove()

    return _finalize_text_examples(heaps_by_latent), selected_latents, processed_samples


def _extract_patch_hidden_states(
    hidden_state: torch.Tensor,
    expected_patch_tokens: int,
) -> torch.Tensor:
    patch_hidden_states = _reshape_image_hidden_state_to_token_sequence(
        hidden_state=hidden_state,
        expected_patch_tokens=expected_patch_tokens,
    )
    if int(patch_hidden_states.shape[0]) != 1:
        raise ValueError(
            "latent examples expects batch size 1; received "
            f"batch={patch_hidden_states.shape[0]}."
        )
    return patch_hidden_states[0, -int(expected_patch_tokens) :, :]


def _generate_image_examples(
    config: LLMBenchmarkConfig,
    manifest: Dict[str, Any],
    sae_model: torch.nn.Module,
    runtime_device: str,
    d_hidden: int,
) -> Tuple[Dict[str, List[List[str]]], List[int], int]:
    from transformers import AutoImageProcessor, AutoModel

    if config.vision is None:
        raise ValueError("vision must be provided for image latent examples.")
    if not isinstance(config.data, ImageDataConfig):
        raise ValueError("Image latent examples require ImageDataConfig.")
    if not isinstance(config.cache, ImageActivationCacheConfig):
        raise ValueError("Image latent examples require ImageActivationCacheConfig.")

    image_processor_id = config.vision.image_processor_id or config.vision.model_id
    image_processor_revision = (
        config.vision.image_processor_revision or config.vision.revision
    )
    processor = AutoImageProcessor.from_pretrained(
        str(image_processor_id),
        revision=image_processor_revision,
        use_auth_token=bool(config.vision.use_auth_token),
    )
    vision_model = AutoModel.from_pretrained(
        str(config.vision.model_id),
        revision=config.vision.revision,
        torch_dtype=_resolve_vision_backbone_dtype(
            config=config,
            runtime_device=runtime_device,
        ),
        low_cpu_mem_usage=True,
        use_auth_token=bool(config.vision.use_auth_token),
    )
    vision_model = vision_model.to(runtime_device)
    vision_model.eval()
    model_torch_dtype_any = getattr(vision_model, "dtype", None)
    if isinstance(model_torch_dtype_any, torch.dtype):
        model_torch_dtype = model_torch_dtype_any
    else:
        first_parameter = next(vision_model.parameters(), None)
        if first_parameter is not None:
            model_torch_dtype = first_parameter.dtype
        else:
            model_torch_dtype = _resolve_vision_backbone_dtype(
                config=config, runtime_device=runtime_device
            )

    layer_index = _resolve_image_hidden_state_layer_index(
        model=vision_model,
        hook=config.hook,
    )

    manifest_patch_size = _safe_int(manifest.get("patch_size"))
    patch_size = (
        int(manifest_patch_size)
        if manifest_patch_size is not None
        else int(_resolve_patch_size(vision_model))
    )
    manifest_image_size = _safe_int(manifest.get("image_size"))
    image_size = (
        int(manifest_image_size)
        if manifest_image_size is not None
        else int(config.cache.image_size)
    )
    if image_size % patch_size != 0:
        raise ValueError(
            "Image latent examples require image_size divisible by patch_size. "
            f"Received image_size={image_size}, patch_size={patch_size}."
        )
    n_patches_h = int(image_size // patch_size)
    n_patches_w = int(image_size // patch_size)
    expected_patch_tokens = int(n_patches_h * n_patches_w)

    sample_size = int(config.latent_examples.sample_size)
    activation_threshold = float(config.latent_examples.activation_threshold)
    max_examples_per_latent = int(config.latent_examples.max_examples_per_latent)
    image_records, _ = _iter_image_records(data=config.data, cache=config.cache)

    images: List[Image.Image] = []
    for entry in image_records:
        if len(images) >= sample_size:
            break
        image_value = entry.get(str(config.data.image_field))
        images.append(
            _coerce_image_value_to_pil(
                image_value=image_value,
                image_field=str(config.data.image_field),
            )
        )
    if not images:
        return {}, [], 0

    max_abs_by_latent = torch.full((int(d_hidden),), float("-inf"), dtype=torch.float64)
    activation_count_by_latent = torch.zeros((int(d_hidden),), dtype=torch.float64)
    processed_samples = 0

    with torch.no_grad():
        for image in images:
            image_for_grid = ImageOps.fit(
                image.convert("RGB"),
                (image_size, image_size),
                method=Image.BICUBIC,
            )
            processor_inputs = processor(images=[image_for_grid], return_tensors="pt")
            model_inputs: Dict[str, Any] = {}
            for key, value in processor_inputs.items():
                if torch.is_tensor(value):
                    value = value.to(runtime_device)
                    if value.is_floating_point() and value.dtype != model_torch_dtype:
                        value = value.to(dtype=model_torch_dtype)
                    model_inputs[key] = value
                else:
                    model_inputs[key] = value

            _validate_image_processor_pixel_values(
                model_inputs=model_inputs,
                image_size=image_size,
            )
            outputs = vision_model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if not isinstance(hidden_states, (list, tuple)):
                raise ValueError(
                    "Vision model output must include hidden_states for latent examples."
                )
            if layer_index < 0 or layer_index >= len(hidden_states):
                raise ValueError(
                    f"Resolved hidden-state layer index {layer_index} is out of range for "
                    f"{len(hidden_states)} returned hidden states."
                )
            patch_hidden_states = _extract_patch_hidden_states(
                hidden_state=hidden_states[layer_index],
                expected_patch_tokens=expected_patch_tokens,
            )
            output = sae_model(
                patch_hidden_states.to(device=runtime_device, dtype=torch.float32)
            )
            latents = _flatten_single_instance_latents(output["latents"])
            if int(latents.shape[1]) != int(d_hidden):
                raise ValueError(
                    "Latent width mismatch while generating image examples. "
                    f"Expected {d_hidden}, got {int(latents.shape[1])}."
                )
            latents_abs = torch.nan_to_num(
                latents.abs(), nan=0.0, posinf=0.0, neginf=0.0
            )
            max_abs_by_latent = torch.maximum(
                max_abs_by_latent,
                latents_abs.max(dim=0).values.to(dtype=torch.float64, device="cpu"),
            )
            activation_count_by_latent += (
                (latents_abs > activation_threshold)
                .sum(dim=0)
                .to(dtype=torch.float64, device="cpu")
            )
            processed_samples += 1

    selected_latents = _select_latent_indices_from_statistics(
        max_abs_by_latent=max_abs_by_latent,
        activation_count_by_latent=activation_count_by_latent,
        config=config,
        d_hidden=d_hidden,
    )
    if not selected_latents:
        return {}, [], processed_samples

    heaps_by_latent: Dict[int, List[Tuple[float, int, Any]]] = {
        int(latent_index): [] for latent_index in selected_latents
    }
    selected_latent_tensor = torch.tensor(
        selected_latents, dtype=torch.long, device=runtime_device
    )
    sequence_order = 0
    with torch.no_grad():
        for image in images:
            image_for_grid = ImageOps.fit(
                image.convert("RGB"),
                (image_size, image_size),
                method=Image.BICUBIC,
            )
            processor_inputs = processor(images=[image_for_grid], return_tensors="pt")
            model_inputs: Dict[str, Any] = {}
            for key, value in processor_inputs.items():
                if torch.is_tensor(value):
                    value = value.to(runtime_device)
                    if value.is_floating_point() and value.dtype != model_torch_dtype:
                        value = value.to(dtype=model_torch_dtype)
                    model_inputs[key] = value
                else:
                    model_inputs[key] = value

            _validate_image_processor_pixel_values(
                model_inputs=model_inputs,
                image_size=image_size,
            )
            full_image_base64 = encode_png_base64(image_for_grid)

            outputs = vision_model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if not isinstance(hidden_states, (list, tuple)):
                raise ValueError(
                    "Vision model output must include hidden_states for latent examples."
                )
            if layer_index < 0 or layer_index >= len(hidden_states):
                raise ValueError(
                    f"Resolved hidden-state layer index {layer_index} is out of range for "
                    f"{len(hidden_states)} returned hidden states."
                )
            patch_hidden_states = _extract_patch_hidden_states(
                hidden_state=hidden_states[layer_index],
                expected_patch_tokens=expected_patch_tokens,
            )
            output = sae_model(
                patch_hidden_states.to(device=runtime_device, dtype=torch.float32)
            )
            latents = _flatten_single_instance_latents(output["latents"])
            latents_selected = latents.index_select(dim=1, index=selected_latent_tensor)
            latents_selected_cpu = latents_selected.detach().to(
                dtype=torch.float32, device="cpu"
            )

            for local_latent_index, latent_index in enumerate(selected_latents):
                latent_values = latents_selected_cpu[:, int(local_latent_index)]
                abs_values = latent_values.abs()
                best_score_value, best_patch_index_tensor = torch.max(abs_values, dim=0)
                best_score = float(best_score_value.item())
                if best_score <= activation_threshold:
                    continue
                best_patch_index = int(best_patch_index_tensor.item())
                patch_image = crop_patch_image(
                    image=image_for_grid,
                    patch_index=best_patch_index,
                    n_patches_w=n_patches_w,
                    patch_size=patch_size,
                    image_size=image_size,
                )
                payload = [
                    full_image_base64,
                    encode_png_base64(patch_image),
                ]
                _push_bounded_example_heap(
                    heaps_by_latent=heaps_by_latent,
                    latent_index=int(latent_index),
                    score=best_score,
                    payload=payload,
                    max_examples_per_latent=max_examples_per_latent,
                    sequence_order=sequence_order,
                )
            sequence_order += 1

    return (
        _finalize_image_examples(heaps_by_latent),
        selected_latents,
        processed_samples,
    )


def generate_latent_examples_for_subrun(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    output_dir: str,
    overwrite: bool,
) -> Dict[str, Any]:
    relative_subrun_dir = str(subrun["relative_subrun_dir"])
    subrun_dir = _output_subdir_path(
        output_dir=output_dir, relative_dir=relative_subrun_dir
    )
    os.makedirs(subrun_dir, exist_ok=True)
    training_state = subrun_training_state(subrun_dir)
    if training_state == "legacy_unmigrated":
        raise legacy_migration_error(
            output_dir=output_dir,
            subrun_dir=subrun_dir,
        )
    if training_state != "completed":
        raise ValueError(
            "Examples stage requires training status='completed' for "
            f"subrun_dir={subrun_dir}."
        )
    output_path = subrun_latent_examples_path(subrun_dir=subrun_dir)

    if os.path.isfile(output_path) and not bool(overwrite):
        return {
            "reused": True,
            "latent_examples_path": output_path,
            "subrun_dir": subrun_dir,
            "num_latents": None,
            "num_samples_scanned": None,
        }

    sae_model, manifest, runtime_device = _load_sae_model_for_subrun(
        config=config,
        resolved_config=resolved_config,
        subrun=subrun,
        output_dir=output_dir,
    )
    d_hidden = int(getattr(sae_model, "d_hidden"))

    if config.modality == "image":
        token_selection = str(manifest.get("token_selection", "")).strip().lower()
        if token_selection != "patch_only":
            raise ValueError(
                "Image latent examples require token_selection='patch_only' because "
                "the output format expects a single activating patch per example."
            )
        examples, selected_latents, num_samples_scanned = _generate_image_examples(
            config=config,
            manifest=manifest,
            sae_model=sae_model,
            runtime_device=runtime_device,
            d_hidden=d_hidden,
        )
    else:
        examples, selected_latents, num_samples_scanned = _generate_text_examples(
            config=config,
            sae_model=sae_model,
            runtime_device=runtime_device,
            d_hidden=d_hidden,
        )

    selection_mode = str(config.latent_examples.latent_selection)
    if selection_mode == "explicit":
        selection_mode = "explicit_list"
    payload = {
        "schema_version": LATENT_EXAMPLES_SCHEMA_VERSION,
        "modality": str(config.modality),
        "variant_name": str(subrun["variant_name"]),
        "variant_slug": str(subrun["variant_slug"]),
        "seed": int(subrun["seed"]),
        "sweep_param": str(subrun["sweep_param"]),
        "sweep_value": subrun["sweep_value"],
        "examples": examples,
        "selection": {
            "latent_selection": selection_mode,
            "max_latents": int(config.latent_examples.max_latents),
            "max_examples_per_latent": int(
                config.latent_examples.max_examples_per_latent
            ),
            "sample_size": int(config.latent_examples.sample_size),
            "activation_threshold": float(config.latent_examples.activation_threshold),
            "selected_latents": [int(index) for index in selected_latents],
            "target_subruns": str(config.latent_examples.target_subruns),
            "anchor_l0": float(config.latent_examples.anchor_l0),
        },
        "highlight_markers": [
            str(config.latent_examples.highlight_markers[0]),
            str(config.latent_examples.highlight_markers[1]),
        ],
        "num_samples_scanned": int(num_samples_scanned),
    }
    _write_json(output_path, payload)
    return {
        "reused": False,
        "latent_examples_path": output_path,
        "subrun_dir": subrun_dir,
        "num_latents": int(len(examples)),
        "num_samples_scanned": int(num_samples_scanned),
    }


def build_and_persist_latent_examples_index(
    output_dir: str,
    subrun_entries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    index_entries: List[Dict[str, Any]] = []
    for entry in subrun_entries:
        variant_slug = entry.get("variant_slug")
        seed = _safe_int(entry.get("seed"))
        sweep_param = entry.get("sweep_param")
        latent_examples_path = entry.get("latent_examples_path")
        if (
            not isinstance(variant_slug, str)
            or seed is None
            or not isinstance(sweep_param, str)
            or not isinstance(latent_examples_path, str)
            or not latent_examples_path
        ):
            continue
        index_entries.append(
            {
                "variant_slug": str(variant_slug),
                "seed": int(seed),
                "sweep_param": str(sweep_param),
                "sweep_value": entry.get("sweep_value"),
                "latent_examples_path": str(latent_examples_path),
            }
        )
    index_entries.sort(
        key=lambda entry: (
            str(entry["variant_slug"]),
            int(entry["seed"]),
            str(entry["sweep_param"]),
            str(entry["latent_examples_path"]),
        )
    )
    payload = {
        "schema_version": LATENT_EXAMPLES_INDEX_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "subruns": index_entries,
    }
    output_path = run_level_latent_examples_index_path(output_dir=output_dir)
    _write_json(output_path, payload)
    return {
        "index_path": output_path,
        "num_subruns": len(index_entries),
    }
