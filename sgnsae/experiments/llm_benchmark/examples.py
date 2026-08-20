import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig, TextDataConfig
from sgnsae.experiments.llm_benchmark.latent_examples import (
    build_and_persist_latent_examples_index,
    generate_latent_examples_for_subrun as generate_legacy_latent_examples_for_subrun,
    run_level_latent_examples_index_path,
    select_target_subruns_for_examples,
    subrun_latent_examples_path,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    LATENT_EXAMPLES_SCHEMA_VERSION,
    PAIRED_DUPLICATE_CASE_STUDIES_SCHEMA_VERSION,
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


_CONTEXT_RADIUS = 16
_BATCH_SIZE = 256
_DUPLICATE_THRESHOLD = -0.95
_MAX_DUPLICATE_CANDIDATES = 2048
_HISTOGRAM_BINS = 41
_HISTOGRAM_PERCENTILE = 0.995
_LATENT_EXAMPLES_ASSETS_DIR = "latent_examples_assets"
_PAIRED_DUPLICATE_CASE_STUDIES_FILENAME = "paired_duplicate_case_studies.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str) -> Any:
    with open(path, "r") as file:
        return json.load(file)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def _output_subdir_path(output_dir: str, relative_dir: str) -> str:
    normalized = relative_dir.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        raise ValueError("Resolved config contains an empty relative_subrun_dir.")
    root = os.path.abspath(output_dir)
    candidate = os.path.abspath(os.path.join(root, *parts))
    try:
        common = os.path.commonpath([root, candidate])
    except ValueError as exc:
        raise ValueError(
            f"Subrun directory {relative_dir} escapes output_dir={output_dir}."
        ) from exc
    if common != root:
        raise ValueError(
            f"Resolved relative_subrun_dir escapes output_dir: {relative_dir}"
        )
    return candidate


def _resolve_val_locations(
    manifest: Dict[str, Any], output_dir: str, max_samples: int
) -> List[Tuple[int, int]]:
    if max_samples <= 0:
        return []

    method = str(manifest["split"]["method"])
    shards = manifest["shards"]
    output: List[Tuple[int, int]] = []

    if method == "shard_ranges":
        for shard_index in manifest["split"]["val_shard_indices"]:
            shard = shards[int(shard_index)]
            num_vectors = int(
                shard.get(
                    "num_vectors",
                    int(shard["end_vector_index"]) - int(shard["start_vector_index"]),
                )
            )
            for local_index in range(num_vectors):
                output.append((int(shard_index), int(local_index)))
                if len(output) >= int(max_samples):
                    return output
        return output

    if method == "stored_indices":
        indices_path = os.path.join(
            output_dir,
            "activations",
            str(manifest["split"]["indices_filename"]),
        )
        indices_payload = torch.load(indices_path, map_location="cpu")
        if not isinstance(indices_payload, dict) or "val" not in indices_payload:
            raise ValueError(
                f"Indices file {indices_path} missing required 'val' entry."
            )
        selected = indices_payload["val"]
        selected = torch.sort(selected.to(dtype=torch.int64, device="cpu")).values
        if int(selected.numel()) > int(max_samples):
            selected = selected[: int(max_samples)]

        for shard_index, shard in enumerate(shards):
            start_index = int(shard["start_vector_index"])
            end_index = int(shard["end_vector_index"])
            left = int(torch.searchsorted(selected, start_index).item())
            right = int(torch.searchsorted(selected, end_index).item())
            for local_index in (selected[left:right] - int(start_index)).tolist():
                output.append((int(shard_index), int(local_index)))
                if len(output) >= int(max_samples):
                    return output
        return output

    raise ValueError(
        "Invalid split method in activation manifest. Expected 'shard_ranges' or "
        f"'stored_indices', got '{method}'."
    )


def _load_text_records(
    manifest: Dict[str, Any], output_dir: str, max_samples: int
) -> Tuple[torch.Tensor, List[int], List[Optional[int]], List[Optional[int]]]:
    locations = _resolve_val_locations(
        manifest=manifest,
        output_dir=output_dir,
        max_samples=max_samples,
    )
    d_in = int(manifest["activation"]["d_in"])
    if not locations:
        return torch.zeros((0, d_in), dtype=torch.float32), [], [], []

    grouped: Dict[int, List[int]] = {}
    for shard_index, local_index in locations:
        grouped.setdefault(int(shard_index), []).append(int(local_index))

    activations_out: List[torch.Tensor] = []
    token_ids_out: List[int] = []
    positions_out: List[Optional[int]] = []
    sequence_ids_out: List[Optional[int]] = []
    for shard_index in sorted(grouped.keys()):
        shard = manifest["shards"][int(shard_index)]
        shard_path = os.path.join(output_dir, "activations", str(shard["filename"]))
        shard_payload = torch.load(shard_path, map_location="cpu")
        acts = shard_payload["acts"]
        tokens = shard_payload.get("tokens")
        if not torch.is_tensor(tokens) or tokens.ndim != 1:
            raise ValueError(
                "Examples stage requires per-token metadata in cache shards. "
                "Regenerate cache with cache.store_tokens=true."
            )
        positions = shard_payload.get("positions")
        sequence_ids = shard_payload.get("sequence_id")
        for local_index in sorted(grouped[shard_index]):
            activations_out.append(acts[int(local_index)].to(dtype=torch.float32))
            token_ids_out.append(int(tokens[int(local_index)].item()))
            if torch.is_tensor(positions):
                positions_out.append(int(positions[int(local_index)].item()))
            else:
                positions_out.append(None)
            if torch.is_tensor(sequence_ids):
                sequence_ids_out.append(int(sequence_ids[int(local_index)].item()))
            else:
                sequence_ids_out.append(None)

    return (
        torch.stack(activations_out, dim=0),
        token_ids_out,
        positions_out,
        sequence_ids_out,
    )


def _flatten_single_instance_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 3 and int(latents.shape[0]) == 1:
        return latents[0]
    if latents.ndim == 2:
        return latents
    return latents.reshape(-1, latents.shape[-1])


def _compute_latents(
    model: torch.nn.Module,
    activations: torch.Tensor,
    runtime_device: str,
    d_hidden: int,
) -> torch.Tensor:
    if int(activations.shape[0]) == 0:
        return torch.zeros((0, int(d_hidden)), dtype=torch.float32)

    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start_index in range(0, int(activations.shape[0]), _BATCH_SIZE):
            end_index = min(start_index + int(_BATCH_SIZE), int(activations.shape[0]))
            batch = activations[start_index:end_index].to(
                device=runtime_device,
                dtype=torch.float32,
            )
            output = model(batch)
            latents = _flatten_single_instance_latents(output["latents"]).detach()
            latents = latents.to(device="cpu", dtype=torch.float32)
            if int(latents.shape[1]) != int(d_hidden):
                raise ValueError(
                    "Latent width mismatch while generating examples. "
                    f"Expected {d_hidden}, got {int(latents.shape[1])}."
                )
            outputs.append(latents)
    return torch.cat(outputs, dim=0)


def _load_alive_indices(subrun_dir: str, d_hidden: int) -> Optional[List[int]]:
    evaluation_path = os.path.join(subrun_dir, "evaluation.json")
    if not os.path.isfile(evaluation_path):
        return None
    payload = _read_json(evaluation_path)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("alive_indices"), list
    ):
        return None
    output = []
    for value in payload["alive_indices"]:
        if isinstance(value, int) and not isinstance(value, bool):
            if 0 <= int(value) < int(d_hidden):
                output.append(int(value))
    return sorted(set(output))


def _select_latents(
    config: LLMBenchmarkConfig,
    latents: torch.Tensor,
    alive_indices: Optional[Sequence[int]],
    d_hidden: int,
) -> List[int]:
    max_latents = min(int(config.latent_examples.max_latents), int(d_hidden))
    if max_latents <= 0:
        return []

    allowed = torch.ones((int(d_hidden),), dtype=torch.bool)
    if alive_indices is not None:
        allowed = torch.zeros((int(d_hidden),), dtype=torch.bool)
        for value in alive_indices:
            if 0 <= int(value) < int(d_hidden):
                allowed[int(value)] = True

    if str(config.latent_examples.latent_selection) == "explicit":
        explicit_latents = config.latent_examples.explicit_latents or []
        selected = [
            int(value)
            for value in explicit_latents
            if 0 <= int(value) < int(d_hidden) and bool(allowed[int(value)].item())
        ]
        return sorted(set(selected))[:max_latents]

    if int(latents.shape[0]) == 0:
        return []

    threshold = float(config.latent_examples.activation_threshold)
    mode = str(config.latent_examples.latent_selection)
    latents_abs = latents.abs()
    if mode == "top_by_freq":
        scores = (latents_abs > threshold).sum(dim=0).to(dtype=torch.float32)
    elif mode == "top_by_mean_abs":
        scores = latents_abs.mean(dim=0).to(dtype=torch.float32)
    else:
        scores = latents_abs.max(dim=0).values.to(dtype=torch.float32)

    pairs = [
        (float(scores[index].item()), index)
        for index in range(int(d_hidden))
        if bool(allowed[index].item()) and float(scores[index].item()) > threshold
    ]
    pairs.sort(key=lambda item: (-item[0], item[1]))
    return [int(item[1]) for item in pairs[:max_latents]]


def _decode_token(tokenizer: Any, token_id: int, cache: Dict[int, str]) -> str:
    normalized = int(token_id)
    if normalized in cache:
        return cache[normalized]
    try:
        decoded = tokenizer.decode([normalized], clean_up_tokenization_spaces=False)
    except TypeError:
        try:
            decoded = tokenizer.decode([normalized])
        except Exception:
            decoded = ""
    except Exception:
        decoded = ""
    normalized_decoded = str(decoded)
    if not normalized_decoded:
        normalized_decoded = f"<tok:{normalized}>"
    cache[normalized] = normalized_decoded
    return cache[normalized]


def _context_text(
    tokenizer: Any,
    token_ids: Sequence[int],
    sequence_ids: Sequence[Optional[int]],
    center_index: int,
    markers: Tuple[str, str],
    decode_cache: Dict[int, str],
) -> str:
    total = len(token_ids)
    left = max(0, int(center_index) - int(_CONTEXT_RADIUS))
    right = min(total, int(center_index) + int(_CONTEXT_RADIUS) + 1)

    sequence_id = sequence_ids[int(center_index)]
    if sequence_id is not None:
        left = int(center_index)
        left_budget = int(_CONTEXT_RADIUS)
        while left_budget > 0 and left > 0 and sequence_ids[left - 1] == sequence_id:
            left -= 1
            left_budget -= 1

        right = int(center_index) + 1
        right_budget = int(_CONTEXT_RADIUS)
        while right_budget > 0 and right < total and sequence_ids[right] == sequence_id:
            right += 1
            right_budget -= 1

    pieces = [
        _decode_token(tokenizer, int(token_ids[index]), decode_cache)
        for index in range(left, right)
    ]
    center = int(center_index) - int(left)
    left_marker, right_marker = markers
    pieces[center] = f"{left_marker}{pieces[center]}{right_marker}"
    return "".join(pieces)


def _build_example(
    tokenizer: Any,
    token_ids: Sequence[int],
    positions: Sequence[Optional[int]],
    sequence_ids: Sequence[Optional[int]],
    sample_index: int,
    activation: float,
    markers: Tuple[str, str],
    decode_cache: Dict[int, str],
) -> Dict[str, Any]:
    return {
        "activation": float(activation),
        "token_id": int(token_ids[int(sample_index)]),
        "position": positions[int(sample_index)],
        "sequence_id": sequence_ids[int(sample_index)],
        "context": _context_text(
            tokenizer=tokenizer,
            token_ids=token_ids,
            sequence_ids=sequence_ids,
            center_index=int(sample_index),
            markers=markers,
            decode_cache=decode_cache,
        ),
    }


def _collect_examples_for_latents(
    tokenizer: Any,
    token_ids: Sequence[int],
    positions: Sequence[Optional[int]],
    sequence_ids: Sequence[Optional[int]],
    latents: torch.Tensor,
    latent_indices: Sequence[int],
    max_examples_per_latent: int,
    activation_threshold: float,
    include_negative: bool,
    markers: Tuple[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    decode_cache: Dict[int, str] = {}
    entries: List[Dict[str, Any]] = []
    positive_lookup: Dict[int, List[Dict[str, Any]]] = {}

    for latent_index in latent_indices:
        values = latents[:, int(latent_index)]
        positive = [
            (int(index), float(values[int(index)].item()))
            for index in torch.nonzero(values > activation_threshold)
            .reshape(-1)
            .tolist()
        ]
        positive.sort(key=lambda item: (-item[1], item[0]))
        positive = positive[: int(max_examples_per_latent)]
        positive_examples = [
            _build_example(
                tokenizer=tokenizer,
                token_ids=token_ids,
                positions=positions,
                sequence_ids=sequence_ids,
                sample_index=int(sample_index),
                activation=float(activation),
                markers=markers,
                decode_cache=decode_cache,
            )
            for sample_index, activation in positive
        ]
        positive_lookup[int(latent_index)] = positive_examples

        negative_examples: List[Dict[str, Any]] = []
        if include_negative:
            negative = [
                (int(index), float(values[int(index)].item()))
                for index in torch.nonzero(values < -activation_threshold)
                .reshape(-1)
                .tolist()
            ]
            negative.sort(key=lambda item: (item[1], item[0]))
            negative = negative[: int(max_examples_per_latent)]
            negative_examples = [
                _build_example(
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    positions=positions,
                    sequence_ids=sequence_ids,
                    sample_index=int(sample_index),
                    activation=float(activation),
                    markers=markers,
                    decode_cache=decode_cache,
                )
                for sample_index, activation in negative
            ]

        entries.append(
            {
                "latent_index": int(latent_index),
                "positive_examples": positive_examples,
                "negative_examples": negative_examples,
            }
        )

    return entries, positive_lookup


def _find_duplicate_pairs(
    model: torch.nn.Module,
    alive_indices: Optional[Sequence[int]],
    max_pairs: int,
) -> List[Tuple[int, int, float]]:
    w_dec = getattr(model, "W_dec", None)
    if not torch.is_tensor(w_dec) or w_dec.ndim != 3 or int(w_dec.shape[0]) != 1:
        return []

    rows = w_dec[0].transpose(0, 1).to(dtype=torch.float32, device="cpu")
    d_hidden = int(rows.shape[0])
    if alive_indices is None:
        candidate_indices = list(range(d_hidden))
    else:
        candidate_indices = [
            int(value) for value in alive_indices if 0 <= int(value) < int(d_hidden)
        ]
    candidate_indices = sorted(set(candidate_indices))
    if len(candidate_indices) < 2:
        return []
    if len(candidate_indices) > int(_MAX_DUPLICATE_CANDIDATES):
        candidate_indices = candidate_indices[: int(_MAX_DUPLICATE_CANDIDATES)]

    candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long)
    normalized = rows.index_select(0, candidate_tensor)
    normalized = normalized / (normalized.norm(dim=1, keepdim=True) + 1e-8)
    similarity = normalized @ normalized.T

    pairs: Dict[Tuple[int, int], float] = {}
    for row_index in range(len(candidate_indices)):
        similarity[row_index, row_index] = 1.0
        cosine, min_index = torch.min(similarity[row_index], dim=0)
        cosine_value = float(cosine.item())
        if cosine_value >= float(_DUPLICATE_THRESHOLD):
            continue
        first = int(candidate_indices[row_index])
        second = int(candidate_indices[int(min_index.item())])
        pair = (min(first, second), max(first, second))
        previous = pairs.get(pair)
        if previous is None or cosine_value < previous:
            pairs[pair] = cosine_value

    output = [(int(a), int(b), float(v)) for (a, b), v in pairs.items()]
    output.sort(key=lambda item: (item[2], item[0], item[1]))
    return output[: int(max_pairs)]


def run_level_paired_duplicate_case_studies_path(output_dir: str) -> str:
    return os.path.join(output_dir, _PAIRED_DUPLICATE_CASE_STUDIES_FILENAME)


def _latent_examples_assets_dir(subrun_dir: str) -> str:
    return os.path.join(subrun_dir, _LATENT_EXAMPLES_ASSETS_DIR)


def _finite_values(tensor: torch.Tensor) -> torch.Tensor:
    if int(tensor.numel()) == 0:
        return torch.zeros((0,), dtype=torch.float32)
    flattened = tensor.reshape(-1).to(dtype=torch.float32, device="cpu")
    return flattened[torch.isfinite(flattened)]


def _resolve_histogram_range(
    values: torch.Tensor, sign_aware: bool
) -> Tuple[float, float]:
    finite = _finite_values(values)
    if int(finite.numel()) == 0:
        if sign_aware:
            return -1.0, 1.0
        return 0.0, 1.0

    if sign_aware:
        max_abs = float(
            torch.quantile(finite.abs(), float(_HISTOGRAM_PERCENTILE)).item()
        )
        if max_abs <= 0.0:
            max_abs = 1.0
        return -max_abs, max_abs

    low = float(torch.quantile(finite, 0.01).item())
    high = float(torch.quantile(finite, 0.99).item())
    if abs(high - low) <= 1e-12:
        center = float(finite.mean().item())
        return center - 1e-3, center + 1e-3
    return low, high


def _activation_summary(values: torch.Tensor) -> Dict[str, Any]:
    finite = _finite_values(values)
    if int(finite.numel()) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "negative_fraction": 0.0,
            "zero_fraction": 0.0,
            "positive_fraction": 0.0,
        }

    negatives = float((finite < 0).sum().item())
    zeros = float((finite == 0).sum().item())
    positives = float((finite > 0).sum().item())
    total = float(finite.numel())
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "p50": float(torch.quantile(finite, 0.50).item()),
        "p95": float(torch.quantile(finite, 0.95).item()),
        "negative_fraction": negatives / total,
        "zero_fraction": zeros / total,
        "positive_fraction": positives / total,
    }


def _generate_activation_histograms(
    latents: torch.Tensor,
    latent_indices: Sequence[int],
    subrun_dir: str,
    sign_aware: bool,
) -> List[Dict[str, Any]]:
    if not latent_indices:
        return []

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    assets_dir = _latent_examples_assets_dir(subrun_dir=subrun_dir)
    os.makedirs(assets_dir, exist_ok=True)
    histogram_entries: List[Dict[str, Any]] = []
    for latent_index in sorted({int(value) for value in latent_indices}):
        values = (
            latents[:, int(latent_index)]
            if int(latents.shape[0]) > 0
            else torch.zeros((0,))
        )
        values = values.to(dtype=torch.float32, device="cpu")
        hist_low, hist_high = _resolve_histogram_range(
            values=values, sign_aware=sign_aware
        )

        figure, axis = plt.subplots(figsize=(5.0, 3.0))
        axis.hist(
            values.numpy(),
            bins=int(_HISTOGRAM_BINS),
            range=(float(hist_low), float(hist_high)),
            color="#1f77b4",
            edgecolor="#ffffff",
            linewidth=0.5,
        )
        axis.set_title(f"Latent {int(latent_index)} activations")
        axis.set_xlabel("activation")
        axis.set_ylabel("count")
        if sign_aware:
            axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.0)
        figure.tight_layout()

        relative_path = (
            os.path.join(
                _LATENT_EXAMPLES_ASSETS_DIR,
                f"hist_latent_{int(latent_index)}.png",
            )
            .replace("\\", "/")
            .strip("/")
        )
        absolute_path = os.path.join(subrun_dir, relative_path)
        figure.savefig(absolute_path, dpi=120)
        plt.close(figure)

        histogram_entries.append(
            {
                "latent_index": int(latent_index),
                "path": relative_path,
                "stats": _activation_summary(values=values),
            }
        )
    return histogram_entries


def _load_text_tokenizer(config: LLMBenchmarkConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Examples stage requires transformers. Install with "
            "`pip install transformers`."
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(config.llm.tokenizer_id or config.llm.model_id),
        revision=config.llm.tokenizer_revision or config.llm.revision,
    )


def _decoder_columns(model: torch.nn.Module) -> torch.Tensor:
    w_dec = getattr(model, "W_dec", None)
    if not torch.is_tensor(w_dec) or w_dec.ndim != 3 or int(w_dec.shape[0]) != 1:
        raise ValueError("Expected model.W_dec with shape [1, d_in, d_hidden].")
    columns = w_dec[0].transpose(0, 1).to(dtype=torch.float32, device="cpu")
    return columns / (columns.norm(dim=1, keepdim=True) + 1e-8)


def _normalized_decoder_column(
    model: torch.nn.Module, latent_index: int
) -> torch.Tensor:
    w_dec = getattr(model, "W_dec", None)
    if not torch.is_tensor(w_dec) or int(w_dec.ndim) != 3 or int(w_dec.shape[0]) != 1:
        raise ValueError("Expected model.W_dec with shape [1, d_in, d_hidden].")
    d_hidden = int(w_dec.shape[2])
    if int(latent_index) < 0 or int(latent_index) >= int(d_hidden):
        raise ValueError(
            f"latent_index {int(latent_index)} out of range for d_hidden={int(d_hidden)}."
        )
    column = w_dec[0, :, int(latent_index)].detach().to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(column).clamp_min(1e-12)
    return column / norm


def _normalized_decoder_columns_subset(
    model: torch.nn.Module, latent_indices: Sequence[int]
) -> Tuple[torch.Tensor, List[int]]:
    w_dec = getattr(model, "W_dec", None)
    if not torch.is_tensor(w_dec) or int(w_dec.ndim) != 3 or int(w_dec.shape[0]) != 1:
        raise ValueError("Expected model.W_dec with shape [1, d_in, d_hidden].")

    d_hidden = int(w_dec.shape[2])
    cleaned: List[int] = []
    for latent_index in latent_indices:
        if isinstance(latent_index, bool) or not isinstance(latent_index, int):
            continue
        if 0 <= int(latent_index) < int(d_hidden):
            cleaned.append(int(latent_index))

    if not cleaned:
        return (
            torch.zeros(
                (0, int(w_dec.shape[1])),
                dtype=torch.float32,
                device=w_dec.device,
            ),
            [],
        )

    unique_sorted = sorted(set(cleaned))
    index_tensor = torch.tensor(unique_sorted, dtype=torch.long, device=w_dec.device)
    columns = (
        w_dec[0]
        .index_select(1, index_tensor)
        .transpose(0, 1)
        .detach()
        .to(dtype=torch.float32)
    )
    norms = torch.linalg.vector_norm(columns, dim=1, keepdim=True).clamp_min(1e-12)
    return columns / norms, unique_sorted


def _load_checkpointed_subrun_model(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    output_dir: str,
    manifest: Dict[str, Any],
    runtime_device: str,
) -> torch.nn.Module:
    subrun_dir = _output_subdir_path(output_dir, str(subrun["relative_subrun_dir"]))
    model, _, _, _ = build_model_for_subrun(
        config=config,
        resolved_config=resolved_config,
        subrun=subrun,
        manifest=manifest,
        device=runtime_device,
    )
    checkpoint_path = os.path.join(subrun_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(
            "Missing trained checkpoint required for paired duplicate case studies: "
            f"{checkpoint_path}."
        )
    checkpoint_payload = torch.load(checkpoint_path, map_location=runtime_device)
    state_dict = (
        checkpoint_payload.get("state_dict")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a 'state_dict' dictionary."
        )
    model.load_state_dict(state_dict)
    model = model.to(runtime_device)
    model.eval()
    return model


def _load_subrun_examples_payload(
    output_dir: str, subrun: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    subrun_dir = _output_subdir_path(output_dir, str(subrun["relative_subrun_dir"]))
    path = subrun_latent_examples_path(subrun_dir=subrun_dir)
    if not os.path.isfile(path):
        return None
    payload = _read_json(path)
    if isinstance(payload, dict):
        return payload
    return None


def _is_sign_aware_subrun(
    subrun: Dict[str, Any], variants_by_slug: Dict[str, Dict[str, Any]]
) -> bool:
    variant_slug = str(subrun.get("variant_slug"))
    variant = variants_by_slug.get(variant_slug, {})
    return bool(variant.get("sign_aware", False))


def _subrun_sort_key(subrun: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(subrun.get("variant_slug", "")),
        int(subrun.get("seed", 0)),
        str(subrun.get("relative_subrun_dir", "")),
    )


def _eligible_signed_latent_entries(
    sign_aware_payload: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    examples_payload = sign_aware_payload.get("examples", {})
    if not isinstance(examples_payload, dict):
        return {}
    latent_entries = examples_payload.get("latents", [])
    if not isinstance(latent_entries, list):
        return {}
    output: Dict[int, Dict[str, Any]] = {}
    for entry in latent_entries:
        if not isinstance(entry, dict):
            continue
        latent_index = entry.get("latent_index")
        if isinstance(latent_index, bool) or not isinstance(latent_index, int):
            continue
        positive_examples = entry.get("positive_examples", [])
        negative_examples = entry.get("negative_examples", [])
        if not isinstance(positive_examples, list) or not isinstance(
            negative_examples, list
        ):
            continue
        if not positive_examples or not negative_examples:
            continue
        output[int(latent_index)] = entry
    return output


def _find_best_sign_latent_for_direction(
    direction: torch.Tensor,
    sign_columns: torch.Tensor,
    eligible_latent_indices: Sequence[int],
) -> Optional[Tuple[int, float]]:
    if not eligible_latent_indices:
        return None
    candidate_tensor = torch.tensor(
        [int(index) for index in eligible_latent_indices], dtype=torch.long
    )
    valid_mask = (candidate_tensor >= 0) & (
        candidate_tensor < int(sign_columns.shape[0])
    )
    candidate_tensor = candidate_tensor[valid_mask]
    if int(candidate_tensor.numel()) == 0:
        return None
    cosine_values = sign_columns.index_select(0, candidate_tensor) @ direction
    best_position = int(torch.argmax(cosine_values.abs()).item())
    latent_index = int(candidate_tensor[best_position].item())
    cosine_value = float(cosine_values[best_position].item())
    return latent_index, cosine_value


def _subrun_reference(subrun: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "variant_name": str(subrun.get("variant_name", "")),
        "variant_slug": str(subrun.get("variant_slug", "")),
        "seed": int(subrun.get("seed", 0)),
        "sweep_param": str(subrun.get("sweep_param", "")),
        "sweep_value": subrun.get("sweep_value"),
        "relative_subrun_dir": str(subrun.get("relative_subrun_dir", "")),
    }


def _list_or_empty(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _paired_duplicate_case_studies_payload(
    baseline_subrun: Optional[Dict[str, Any]],
    sign_subrun: Optional[Dict[str, Any]],
    case_studies: Sequence[Dict[str, Any]],
    notes: Sequence[str],
) -> Dict[str, Any]:
    return {
        "schema_version": PAIRED_DUPLICATE_CASE_STUDIES_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "baseline_subrun": (
            _subrun_reference(baseline_subrun)
            if isinstance(baseline_subrun, dict)
            else None
        ),
        "sign_aware_subrun": (
            _subrun_reference(sign_subrun) if isinstance(sign_subrun, dict) else None
        ),
        "duplicate_cosine_threshold": float(_DUPLICATE_THRESHOLD),
        "case_studies": list(case_studies),
        "notes": [str(note) for note in notes],
    }


def build_and_persist_paired_duplicate_case_studies(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    selected_subruns: Sequence[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    output_path = run_level_paired_duplicate_case_studies_path(output_dir=output_dir)
    notes: List[str] = []
    if str(config.modality) != "text":
        payload = _paired_duplicate_case_studies_payload(
            baseline_subrun=None,
            sign_subrun=None,
            case_studies=[],
            notes=["Case studies are only generated for text modality."],
        )
        _write_json(output_path, payload)
        return {"case_studies_path": output_path, "num_case_studies": 0}

    variants = resolved_config.get("variants", [])
    variants_by_slug = {
        str(variant.get("variant_slug")): dict(variant)
        for variant in variants
        if isinstance(variant, dict)
    }
    sign_subruns = sorted(
        [
            dict(subrun)
            for subrun in selected_subruns
            if _is_sign_aware_subrun(subrun=subrun, variants_by_slug=variants_by_slug)
        ],
        key=_subrun_sort_key,
    )
    non_negative_subruns = sorted(
        [
            dict(subrun)
            for subrun in selected_subruns
            if not _is_sign_aware_subrun(
                subrun=subrun, variants_by_slug=variants_by_slug
            )
        ],
        key=_subrun_sort_key,
    )

    baseline_subrun: Optional[Dict[str, Any]] = None
    baseline_payload: Optional[Dict[str, Any]] = None
    for subrun in non_negative_subruns:
        payload = _load_subrun_examples_payload(output_dir=output_dir, subrun=subrun)
        if not isinstance(payload, dict):
            continue
        examples_payload = payload.get("examples", {})
        duplicate_pairs = (
            examples_payload.get("duplicate_pairs", [])
            if isinstance(examples_payload, dict)
            else []
        )
        if isinstance(duplicate_pairs, list) and duplicate_pairs:
            baseline_subrun = subrun
            baseline_payload = payload
            break
    if baseline_subrun is None or baseline_payload is None:
        notes.append(
            "No non-negative examples artifact with duplicate pairs was available."
        )

    sign_subrun: Optional[Dict[str, Any]] = None
    sign_payload: Optional[Dict[str, Any]] = None
    if baseline_subrun is not None:
        same_seed = [
            subrun
            for subrun in sign_subruns
            if int(subrun.get("seed", -1)) == int(baseline_subrun.get("seed", -2))
        ]
        candidate_sign_subruns = same_seed if same_seed else sign_subruns
    else:
        candidate_sign_subruns = sign_subruns
    for subrun in candidate_sign_subruns:
        payload = _load_subrun_examples_payload(output_dir=output_dir, subrun=subrun)
        if isinstance(payload, dict):
            sign_subrun = subrun
            sign_payload = payload
            break
    if sign_subrun is None or sign_payload is None:
        notes.append("No sign-aware examples artifact was available.")

    if (
        baseline_subrun is None
        or baseline_payload is None
        or sign_subrun is None
        or sign_payload is None
    ):
        payload = _paired_duplicate_case_studies_payload(
            baseline_subrun=baseline_subrun,
            sign_subrun=sign_subrun,
            case_studies=[],
            notes=notes,
        )
        _write_json(output_path, payload)
        return {"case_studies_path": output_path, "num_case_studies": 0}

    eligible_sign_latents = _eligible_signed_latent_entries(
        sign_aware_payload=sign_payload
    )
    duplicate_pairs = _list_or_empty(
        baseline_payload.get("examples", {}).get("duplicate_pairs", [])
        if isinstance(baseline_payload.get("examples"), dict)
        else []
    )
    if not eligible_sign_latents:
        notes.append(
            "No sign-aware latents with both positive and negative examples were found."
        )
        payload = _paired_duplicate_case_studies_payload(
            baseline_subrun=baseline_subrun,
            sign_subrun=sign_subrun,
            case_studies=[],
            notes=notes,
        )
        _write_json(output_path, payload)
        return {"case_studies_path": output_path, "num_case_studies": 0}
    if not duplicate_pairs:
        notes.append("No duplicate pairs were available to match against.")
        payload = _paired_duplicate_case_studies_payload(
            baseline_subrun=baseline_subrun,
            sign_subrun=sign_subrun,
            case_studies=[],
            notes=notes,
        )
        _write_json(output_path, payload)
        return {"case_studies_path": output_path, "num_case_studies": 0}

    manifest = load_activation_manifest(manifest_path=resolve_manifest_path(output_dir))
    runtime_device = "cpu"
    baseline_model = _load_checkpointed_subrun_model(
        config=config,
        resolved_config=resolved_config,
        subrun=baseline_subrun,
        output_dir=output_dir,
        manifest=manifest,
        runtime_device=runtime_device,
    )
    sign_model = _load_checkpointed_subrun_model(
        config=config,
        resolved_config=resolved_config,
        subrun=sign_subrun,
        output_dir=output_dir,
        manifest=manifest,
        runtime_device=runtime_device,
    )
    sign_columns, sign_index_map = _normalized_decoder_columns_subset(
        model=sign_model,
        latent_indices=sorted(eligible_sign_latents.keys()),
    )
    if int(sign_columns.shape[0]) == 0:
        notes.append(
            "Eligible sign-aware latent indices were empty after filtering to model bounds."
        )
        payload = _paired_duplicate_case_studies_payload(
            baseline_subrun=baseline_subrun,
            sign_subrun=sign_subrun,
            case_studies=[],
            notes=notes,
        )
        _write_json(output_path, payload)
        return {"case_studies_path": output_path, "num_case_studies": 0}

    baseline_w_dec = getattr(baseline_model, "W_dec", None)
    if (
        not torch.is_tensor(baseline_w_dec)
        or int(baseline_w_dec.ndim) != 3
        or int(baseline_w_dec.shape[0]) != 1
    ):
        raise ValueError("Expected model.W_dec with shape [1, d_in, d_hidden].")
    baseline_hidden = int(baseline_w_dec.shape[2])
    baseline_d_in = int(baseline_w_dec.shape[1])
    sign_d_in = int(sign_columns.shape[1])
    if sign_d_in != baseline_d_in:
        raise ValueError(
            "Decoder dimension mismatch: "
            f"baseline d_in={baseline_d_in} (variant={baseline_subrun.get('variant_slug')}, "
            f"subrun={baseline_subrun.get('relative_subrun_dir')}) vs "
            f"sign-aware d_in={sign_d_in} (variant={sign_subrun.get('variant_slug')}, "
            f"subrun={sign_subrun.get('relative_subrun_dir')})."
        )

    direction_cache: Dict[int, torch.Tensor] = {}
    case_studies: List[Dict[str, Any]] = []
    for duplicate_entry in duplicate_pairs:
        if not isinstance(duplicate_entry, dict):
            continue
        latent_a = duplicate_entry.get("latent_a")
        latent_b = duplicate_entry.get("latent_b")
        # Reject bool explicitly: bool is a subclass of int in Python.
        if isinstance(latent_a, bool) or isinstance(latent_b, bool):
            continue
        if not isinstance(latent_a, int) or not isinstance(latent_b, int):
            continue
        if latent_a == latent_b:
            continue
        if latent_a < 0 or latent_a >= int(baseline_hidden):
            continue
        if latent_b < 0 or latent_b >= int(baseline_hidden):
            continue

        if int(latent_a) not in direction_cache:
            try:
                direction_cache[int(latent_a)] = _normalized_decoder_column(
                    model=baseline_model, latent_index=int(latent_a)
                )
            except ValueError:
                continue
        direction = direction_cache[int(latent_a)]
        cosine_values = sign_columns @ direction
        best_position = int(torch.argmax(cosine_values.abs()).item())
        matched_latent_index = int(sign_index_map[best_position])
        cosine_value = float(cosine_values[best_position].item())
        sign_entry = eligible_sign_latents.get(int(matched_latent_index), {})
        latent_a_examples = _list_or_empty(duplicate_entry.get("latent_a_examples", []))
        latent_b_examples = _list_or_empty(duplicate_entry.get("latent_b_examples", []))
        sign_positive_examples = _list_or_empty(sign_entry.get("positive_examples", []))
        sign_negative_examples = _list_or_empty(sign_entry.get("negative_examples", []))
        case_studies.append(
            {
                "baseline_pair": {
                    "latent_a": int(latent_a),
                    "latent_b": int(latent_b),
                    "decoder_cosine": duplicate_entry.get("decoder_cosine"),
                    "latent_a_examples": latent_a_examples,
                    "latent_b_examples": latent_b_examples,
                },
                "matched_sign_aware_latent": {
                    "latent_index": int(matched_latent_index),
                    "cosine_to_baseline_direction": float(cosine_value),
                    "abs_cosine_to_baseline_direction": abs(float(cosine_value)),
                    "positive_examples": sign_positive_examples,
                    "negative_examples": sign_negative_examples,
                },
            }
        )

    if not case_studies:
        notes.append("No valid paired duplicate case studies could be constructed.")
    payload = _paired_duplicate_case_studies_payload(
        baseline_subrun=baseline_subrun,
        sign_subrun=sign_subrun,
        case_studies=case_studies,
        notes=notes,
    )
    _write_json(output_path, payload)
    return {
        "case_studies_path": output_path,
        "num_case_studies": int(len(case_studies)),
    }


def validate_examples_cache_metadata(
    config: LLMBenchmarkConfig, output_dir: str
) -> None:
    if str(config.modality) != "text":
        return
    manifest = load_activation_manifest(manifest_path=resolve_manifest_path(output_dir))
    if not bool(manifest.get("store_tokens", False)):
        raise ValueError(
            "Examples stage requires cached token metadata. "
            "Regenerate cache with cache.store_tokens=true."
        )
    _ = _load_text_records(manifest=manifest, output_dir=output_dir, max_samples=1)


def generate_latent_examples(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    output_dir: str,
    overwrite: bool,
) -> Dict[str, Any]:
    if str(config.modality) != "text":
        return generate_legacy_latent_examples_for_subrun(
            config=config,
            resolved_config=resolved_config,
            subrun=subrun,
            output_dir=output_dir,
            overwrite=overwrite,
        )
    if not isinstance(config.data, TextDataConfig):
        raise ValueError("Text examples stage requires TextDataConfig.")

    subrun_dir = _output_subdir_path(output_dir, str(subrun["relative_subrun_dir"]))
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

    validate_examples_cache_metadata(config=config, output_dir=output_dir)
    manifest = load_activation_manifest(manifest_path=resolve_manifest_path(output_dir))
    activations, token_ids, positions, sequence_ids = _load_text_records(
        manifest=manifest,
        output_dir=output_dir,
        max_samples=int(config.latent_examples.sample_size),
    )

    runtime_device = resolve_runtime_device(config.device)
    model, _, _, variant_config = build_model_for_subrun(
        config=config,
        resolved_config=resolved_config,
        subrun=subrun,
        manifest=manifest,
        device=runtime_device,
    )
    checkpoint_path = os.path.join(subrun_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(
            f"Missing trained checkpoint required for examples stage: {checkpoint_path}."
        )
    checkpoint_payload = torch.load(checkpoint_path, map_location=runtime_device)
    state_dict = (
        checkpoint_payload.get("state_dict")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a 'state_dict' dictionary."
        )
    model.load_state_dict(state_dict)
    model = model.to(runtime_device)
    model.eval()

    d_hidden = int(getattr(model, "d_hidden"))
    latents = _compute_latents(
        model=model,
        activations=activations,
        runtime_device=runtime_device,
        d_hidden=d_hidden,
    )
    alive_indices = _load_alive_indices(subrun_dir=subrun_dir, d_hidden=d_hidden)
    selected_latents = _select_latents(
        config=config,
        latents=latents,
        alive_indices=alive_indices,
        d_hidden=d_hidden,
    )

    tokenizer = _load_text_tokenizer(config=config)
    markers = (
        str(config.latent_examples.highlight_markers[0]),
        str(config.latent_examples.highlight_markers[1]),
    )

    sign_aware = bool(variant_config.get("sign_aware", False)) or bool(
        getattr(model, "signed_latents", False)
    )
    selected_for_payload = [int(value) for value in selected_latents]

    if sign_aware:
        latent_entries, _ = _collect_examples_for_latents(
            tokenizer=tokenizer,
            token_ids=token_ids,
            positions=positions,
            sequence_ids=sequence_ids,
            latents=latents,
            latent_indices=selected_latents,
            max_examples_per_latent=int(config.latent_examples.max_examples_per_latent),
            activation_threshold=float(config.latent_examples.activation_threshold),
            include_negative=True,
            markers=markers,
        )
        examples_payload = {
            "kind": "sign_aware",
            "latents": latent_entries,
            "duplicate_pairs": [],
            "messages": [],
        }
        num_groups = int(len(latent_entries))
    else:
        max_latents = int(config.latent_examples.max_latents)
        if max_latents < 2:
            raise ValueError(
                "latent_examples.max_latents must be >= 2 for non-negative "
                "duplicate-pair mode."
            )
        max_pairs = int(max_latents // 2)
        duplicate_pairs = _find_duplicate_pairs(
            model=model,
            alive_indices=alive_indices,
            max_pairs=max_pairs,
        )
        paired_latents = sorted(
            set(
                [int(first) for first, _, _ in duplicate_pairs]
                + [int(second) for _, second, _ in duplicate_pairs]
            )
        )
        selected_for_payload = paired_latents
        _, positive_lookup = _collect_examples_for_latents(
            tokenizer=tokenizer,
            token_ids=token_ids,
            positions=positions,
            sequence_ids=sequence_ids,
            latents=latents,
            latent_indices=paired_latents,
            max_examples_per_latent=int(config.latent_examples.max_examples_per_latent),
            activation_threshold=float(config.latent_examples.activation_threshold),
            include_negative=False,
            markers=markers,
        )
        pair_entries = [
            {
                "latent_a": int(first),
                "latent_b": int(second),
                "decoder_cosine": float(cosine),
                "latent_a_examples": positive_lookup.get(int(first), []),
                "latent_b_examples": positive_lookup.get(int(second), []),
            }
            for first, second, cosine in duplicate_pairs
        ]
        messages: List[str] = []
        if not pair_entries:
            messages.append(
                "No duplicate latent pairs with decoder cosine below -0.95 were found."
            )
        examples_payload = {
            "kind": "non_negative",
            "latents": [],
            "duplicate_pairs": pair_entries,
            "messages": messages,
        }
        num_groups = int(len(pair_entries))

    selection_mode = str(config.latent_examples.latent_selection)
    if selection_mode == "explicit":
        selection_mode = "explicit_list"
    histogram_entries = _generate_activation_histograms(
        latents=latents,
        latent_indices=selected_for_payload,
        subrun_dir=subrun_dir,
        sign_aware=sign_aware,
    )

    payload = {
        "schema_version": LATENT_EXAMPLES_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "modality": str(config.modality),
        "variant_name": str(subrun["variant_name"]),
        "variant_slug": str(subrun["variant_slug"]),
        "seed": int(subrun["seed"]),
        "sweep_param": str(subrun["sweep_param"]),
        "sweep_value": subrun["sweep_value"],
        "examples": examples_payload,
        "selection": {
            "latent_selection": selection_mode,
            "max_latents": int(config.latent_examples.max_latents),
            "max_examples_per_latent": int(
                config.latent_examples.max_examples_per_latent
            ),
            "sample_size": int(config.latent_examples.sample_size),
            "activation_threshold": float(config.latent_examples.activation_threshold),
            "selected_latents": [int(index) for index in selected_for_payload],
            "target_subruns": str(config.latent_examples.target_subruns),
            "anchor_l0": float(config.latent_examples.anchor_l0),
        },
        "highlight_markers": [markers[0], markers[1]],
        "num_samples_scanned": int(latents.shape[0]),
        "assets": {
            "activation_histograms": histogram_entries,
        },
    }
    _write_json(path=output_path, payload=payload)
    return {
        "reused": False,
        "latent_examples_path": output_path,
        "subrun_dir": subrun_dir,
        "num_latents": int(num_groups),
        "num_samples_scanned": int(latents.shape[0]),
    }


def generate_latent_examples_for_subrun(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    output_dir: str,
    overwrite: bool,
) -> Dict[str, Any]:
    return generate_latent_examples(
        config=config,
        resolved_config=resolved_config,
        subrun=subrun,
        output_dir=output_dir,
        overwrite=overwrite,
    )
