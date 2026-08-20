import hashlib
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional

import torch


_REGENERATE_GUIDANCE = (
    "Regenerate the cache by rerunning the cache stage "
    "(set cache_overwrite or use --overwrite)."
)


def _load_manifest(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Invalid manifest format: expected object, got {type(manifest)}"
        )
    return manifest


def _resolve_d_in(manifest: Dict[str, Any]) -> int:
    activation = manifest.get("activation")
    if not isinstance(activation, dict):
        raise ValueError("Invalid manifest: missing 'activation' object.")
    d_in = activation.get("d_in")
    if isinstance(d_in, bool) or not isinstance(d_in, int) or d_in <= 0:
        raise ValueError(f"Invalid manifest activation.d_in: {d_in}")
    return int(d_in)


def _resolve_shards(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("Invalid manifest: missing 'shards' list.")
    validated: List[Dict[str, Any]] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(
                f"Invalid manifest shard at index {index}: expected object."
            )
        filename = shard.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"Invalid manifest shard at index {index}: missing non-empty filename."
            )
        start_vector_index = shard.get("start_vector_index")
        end_vector_index = shard.get("end_vector_index")
        if (
            isinstance(start_vector_index, bool)
            or isinstance(end_vector_index, bool)
            or not isinstance(start_vector_index, int)
            or not isinstance(end_vector_index, int)
            or start_vector_index < 0
            or end_vector_index < start_vector_index
        ):
            raise ValueError(
                "Invalid manifest shard indices at index "
                f"{index}: start={start_vector_index}, end={end_vector_index}."
            )
        validated.append(shard)
    return validated


def _derive_seed(seed: int, shard_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{shard_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def _load_shard_tensor(shard_path: str, d_in: int) -> torch.Tensor:
    if not os.path.isfile(shard_path):
        raise ValueError(f"Missing shard file: {shard_path}. {_REGENERATE_GUIDANCE}")
    try:
        shard_payload = torch.load(shard_path)
    except Exception as exc:
        raise ValueError(
            f"Corrupt shard file: {shard_path}. Failed loading with torch.load: {exc}. "
            f"{_REGENERATE_GUIDANCE}"
        ) from exc

    if not isinstance(shard_payload, dict):
        raise ValueError(
            f"Corrupt shard file: {shard_path}. Expected dictionary payload. "
            f"{_REGENERATE_GUIDANCE}"
        )
    if "acts" not in shard_payload:
        raise ValueError(
            f"Corrupt shard file: {shard_path}. Missing required key 'acts'. "
            f"{_REGENERATE_GUIDANCE}"
        )

    acts = shard_payload["acts"]
    if not torch.is_tensor(acts):
        raise ValueError(
            f"Corrupt shard file: {shard_path}. 'acts' is not a tensor. {_REGENERATE_GUIDANCE}"
        )
    if acts.ndim != 2:
        raise ValueError(
            f"Corrupt shard file: {shard_path}. 'acts' rank must be 2, got {acts.ndim}. "
            f"{_REGENERATE_GUIDANCE}"
        )
    if int(acts.shape[1]) != int(d_in):
        raise ValueError(
            "Corrupt shard file: "
            f"{shard_path}. 'acts' second dimension {int(acts.shape[1])} != manifest d_in {d_in}. "
            f"{_REGENERATE_GUIDANCE}"
        )
    return acts


def _resolve_split_max_samples(
    split: str,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    max_test_samples: Optional[int],
) -> Optional[int]:
    if split == "train":
        return max_train_samples
    if split == "val":
        return max_val_samples
    if split == "test":
        return max_test_samples
    raise ValueError(f"Unsupported split: {split}")


class ActivationShardDataset:
    def __init__(
        self,
        manifest_path: str,
        split: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
        max_samples: Optional[int] = None,
    ) -> None:
        normalized_split = str(split).strip().lower()
        if normalized_split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be one of {{'train','val','test'}}, got {split}"
            )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if max_samples is not None and (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples <= 0
        ):
            raise ValueError(
                f"max_samples must be a positive integer when provided, got {max_samples}"
            )

        self.manifest_path = manifest_path
        self.manifest = _load_manifest(manifest_path)
        self.split = normalized_split
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_samples = max_samples

        self.d_in = _resolve_d_in(self.manifest)
        self.shards = _resolve_shards(self.manifest)
        self.activations_dir = os.path.dirname(os.path.abspath(manifest_path))
        self.split_metadata = self.manifest.get("split")
        if not isinstance(self.split_metadata, dict):
            raise ValueError("Invalid manifest: missing split metadata.")

        self.split_method = self.split_metadata.get("method")
        if self.split_method not in {"shard_ranges", "stored_indices"}:
            raise ValueError(
                "Invalid manifest split.method: expected 'shard_ranges' or 'stored_indices', "
                f"got {self.split_method}"
            )

        self._selected_global_indices = torch.empty(0, dtype=torch.int64)
        self.selected_shard_indices = self._resolve_selected_shards()

    def _resolve_selected_shards(self) -> List[int]:
        if self.split_method == "shard_ranges":
            key = f"{self.split}_shard_indices"
            split_indices = self.split_metadata.get(key)
            if not isinstance(split_indices, list):
                raise ValueError(
                    f"Invalid manifest split metadata: missing list '{key}' for shard_ranges mode."
                )
            return [int(index) for index in split_indices]

        indices_filename = self.split_metadata.get("indices_filename")
        if not isinstance(indices_filename, str) or not indices_filename:
            raise ValueError(
                "Invalid manifest split metadata: missing indices_filename for stored_indices mode."
            )

        indices_path = os.path.join(self.activations_dir, indices_filename)
        if not os.path.isfile(indices_path):
            raise ValueError(
                f"Missing split indices file: {indices_path}. {_REGENERATE_GUIDANCE}"
            )
        try:
            split_indices_payload = torch.load(indices_path)
        except Exception as exc:
            raise ValueError(
                f"Corrupt split indices file: {indices_path}. {exc}. {_REGENERATE_GUIDANCE}"
            ) from exc

        if not isinstance(split_indices_payload, dict):
            raise ValueError(
                "Corrupt split indices file: "
                f"{indices_path}. Expected dictionary payload. {_REGENERATE_GUIDANCE}"
            )

        selected_indices = split_indices_payload.get(self.split)
        if not torch.is_tensor(selected_indices):
            raise ValueError(
                "Corrupt split indices file: "
                f"{indices_path}. Missing tensor for split '{self.split}'. {_REGENERATE_GUIDANCE}"
            )
        if selected_indices.dtype != torch.int64:
            selected_indices = selected_indices.to(torch.int64)
        selected_indices = torch.sort(selected_indices).values

        if self.max_samples is not None and selected_indices.numel() > self.max_samples:
            selected_indices = selected_indices[: int(self.max_samples)]

        self._selected_global_indices = selected_indices

        shard_indices: List[int] = []
        for shard_index, shard in enumerate(self.shards):
            shard_start = int(shard["start_vector_index"])
            shard_end = int(shard["end_vector_index"])
            left = int(
                torch.searchsorted(selected_indices, shard_start, right=False).item()
            )
            right = int(
                torch.searchsorted(selected_indices, shard_end, right=False).item()
            )
            if right > left:
                shard_indices.append(shard_index)
        return shard_indices

    def _iter_ordered_shards(self) -> Iterable[int]:
        ordered = list(self.selected_shard_indices)
        if self.shuffle:
            random_generator = random.Random(int(self.seed))
            random_generator.shuffle(ordered)
        return ordered

    def __iter__(self) -> Iterable[torch.Tensor]:
        emitted_samples = 0
        for shard_index in self._iter_ordered_shards():
            if shard_index < 0 or shard_index >= len(self.shards):
                raise ValueError(
                    f"Invalid shard index {shard_index} in split metadata for split '{self.split}'."
                )
            shard = self.shards[shard_index]
            shard_path = os.path.join(self.activations_dir, str(shard["filename"]))
            acts = _load_shard_tensor(shard_path=shard_path, d_in=self.d_in)

            if self.split_method == "stored_indices":
                shard_start = int(shard["start_vector_index"])
                shard_end = int(shard["end_vector_index"])
                selected_indices = self._selected_global_indices
                left = int(
                    torch.searchsorted(
                        selected_indices, shard_start, right=False
                    ).item()
                )
                right = int(
                    torch.searchsorted(selected_indices, shard_end, right=False).item()
                )
                local_indices_tensor = selected_indices[left:right] - shard_start
                if local_indices_tensor.numel() == 0:
                    continue
                if self.shuffle:
                    random_generator = torch.Generator()
                    random_generator.manual_seed(_derive_seed(self.seed, shard_index))
                    permutation = torch.randperm(
                        int(local_indices_tensor.numel()),
                        generator=random_generator,
                    )
                    local_indices_tensor = local_indices_tensor.index_select(
                        0, permutation
                    )
            else:
                local_indices = list(range(int(acts.shape[0])))
                if self.shuffle:
                    shard_random = random.Random(_derive_seed(self.seed, shard_index))
                    shard_random.shuffle(local_indices)

            cursor = 0
            total_local_vectors = (
                int(local_indices_tensor.numel())
                if self.split_method == "stored_indices"
                else len(local_indices)
            )
            while cursor < total_local_vectors:
                if self.max_samples is not None and emitted_samples >= self.max_samples:
                    return

                if self.split_method == "stored_indices":
                    indices_slice_tensor = local_indices_tensor[
                        cursor : cursor + self.batch_size
                    ]
                    if self.max_samples is not None:
                        remaining = int(self.max_samples - emitted_samples)
                        if remaining <= 0:
                            return
                        if int(indices_slice_tensor.numel()) > remaining:
                            indices_slice_tensor = indices_slice_tensor[:remaining]
                    batch_tensor = acts.index_select(
                        0, indices_slice_tensor.to(torch.long)
                    ).to(torch.float32)
                else:
                    indices_slice = local_indices[cursor : cursor + self.batch_size]
                    if self.max_samples is not None:
                        remaining = int(self.max_samples - emitted_samples)
                        if remaining <= 0:
                            return
                        if len(indices_slice) > remaining:
                            indices_slice = indices_slice[:remaining]
                    batch_tensor = acts[indices_slice].to(torch.float32)

                emitted_samples += int(batch_tensor.shape[0])
                cursor += self.batch_size
                yield batch_tensor


def create_activation_dataset(
    manifest_path: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    max_test_samples: Optional[int],
) -> ActivationShardDataset:
    max_samples = _resolve_split_max_samples(
        split=split,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )
    return ActivationShardDataset(
        manifest_path=manifest_path,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        max_samples=max_samples,
    )
