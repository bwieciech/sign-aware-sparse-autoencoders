import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import torch

from sgnsae.experiments.llm_benchmark.dataset import (
    ActivationShardDataset,
    _derive_seed,
    _load_manifest,
    _load_shard_tensor,
    _resolve_d_in,
    _resolve_shards,
    _resolve_split_max_samples,
    create_activation_dataset,
)


def _write_manifest(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def _build_shards_and_manifest(
    root_dir: str,
    split_method: str,
) -> str:
    activations_dir = os.path.join(root_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    shard_specs: List[torch.Tensor] = [
        torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]),
        torch.tensor([[10.0, 10.1], [11.0, 11.1], [12.0, 12.1], [13.0, 13.1]]),
        torch.tensor([[20.0, 20.1], [21.0, 21.1], [22.0, 22.1], [23.0, 23.1]]),
    ]

    shards = []
    cursor = 0
    for shard_index, shard_acts in enumerate(shard_specs):
        shard_filename = f"shard_{shard_index:05d}.pt"
        torch.save({"acts": shard_acts}, os.path.join(activations_dir, shard_filename))
        next_cursor = cursor + int(shard_acts.shape[0])
        shards.append(
            {
                "shard_index": shard_index,
                "filename": shard_filename,
                "num_vectors": int(shard_acts.shape[0]),
                "start_vector_index": cursor,
                "end_vector_index": next_cursor,
            }
        )
        cursor = next_cursor

    split_payload: Dict[str, object]
    if split_method == "shard_ranges":
        split_payload = {
            "method": "shard_ranges",
            "train_shard_indices": [0, 1],
            "val_shard_indices": [2],
            "test_shard_indices": [],
        }
    else:
        split_indices = {
            "train": torch.tensor([0, 1, 4, 5, 8, 9], dtype=torch.int64),
            "val": torch.tensor([2, 6, 10], dtype=torch.int64),
            "test": torch.tensor([3, 7, 11], dtype=torch.int64),
        }
        torch.save(split_indices, os.path.join(activations_dir, "split_indices.pt"))
        split_payload = {
            "method": "stored_indices",
            "indices_filename": "split_indices.pt",
            "train_count": 6,
            "val_count": 3,
            "test_count": 3,
        }

    manifest_path = os.path.join(activations_dir, "activations_manifest.json")
    _write_manifest(
        manifest_path,
        {
            "activation": {"d_in": 2},
            "shards": shards,
            "split": split_payload,
        },
    )
    return manifest_path


def _collect_first_column(dataset: ActivationShardDataset) -> List[float]:
    first_column_values: List[float] = []
    for batch in dataset:
        first_column_values.extend(batch[:, 0].tolist())
    return first_column_values


class TestActivationShardDataset(unittest.TestCase):
    def test_dataset_helper_functions_cover_validation_paths(self) -> None:
        # given / when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, "w") as file:
                json.dump([], file)
            with self.assertRaises(ValueError):
                _ = _load_manifest(manifest_path)

            with self.assertRaises(ValueError):
                _ = _resolve_d_in({})
            with self.assertRaises(ValueError):
                _ = _resolve_d_in({"activation": {"d_in": 0}})

            with self.assertRaises(ValueError):
                _ = _resolve_shards({})
            with self.assertRaises(ValueError):
                _ = _resolve_shards({"shards": [None]})
            with self.assertRaises(ValueError):
                _ = _resolve_shards(
                    {
                        "shards": [
                            {
                                "filename": "",
                                "start_vector_index": 0,
                                "end_vector_index": 1,
                            }
                        ]
                    }
                )
            with self.assertRaises(ValueError):
                _ = _resolve_shards(
                    {
                        "shards": [
                            {
                                "filename": "a.pt",
                                "start_vector_index": 2,
                                "end_vector_index": 1,
                            }
                        ]
                    }
                )

            self.assertIsNone(_resolve_split_max_samples("train", None, 2, 3))
            self.assertEqual(_resolve_split_max_samples("val", 1, 2, 3), 2)
            self.assertEqual(_resolve_split_max_samples("test", 1, 2, 3), 3)
            with self.assertRaises(ValueError):
                _ = _resolve_split_max_samples("bad", 1, 2, 3)

            self.assertEqual(_derive_seed(7, 9), _derive_seed(7, 9))

    def test_dataset_load_shard_tensor_error_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing.pt")

            # when / then
            with self.assertRaises(ValueError):
                _ = _load_shard_tensor(missing_path, d_in=2)

            unreadable_path = os.path.join(tmpdir, "bad.pt")
            with open(unreadable_path, "w") as file:
                file.write("not a torch file")
            with self.assertRaises(ValueError):
                _ = _load_shard_tensor(unreadable_path, d_in=2)

            non_dict_path = os.path.join(tmpdir, "non_dict.pt")
            torch.save(torch.tensor([1.0]), non_dict_path)
            with self.assertRaises(ValueError):
                _ = _load_shard_tensor(non_dict_path, d_in=2)

            non_tensor_path = os.path.join(tmpdir, "non_tensor.pt")
            torch.save({"acts": [1, 2]}, non_tensor_path)
            with self.assertRaises(ValueError):
                _ = _load_shard_tensor(non_tensor_path, d_in=2)

            rank_path = os.path.join(tmpdir, "rank.pt")
            torch.save({"acts": torch.tensor([1.0, 2.0])}, rank_path)
            with self.assertRaises(ValueError):
                _ = _load_shard_tensor(rank_path, d_in=2)

            mismatch_path = os.path.join(tmpdir, "mismatch.pt")
            torch.save({"acts": torch.zeros((2, 3))}, mismatch_path)
            with self.assertRaises(ValueError):
                _ = _load_shard_tensor(mismatch_path, d_in=2)

    def test_non_shuffle_order_is_deterministic_for_shard_ranges(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )
            dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=3,
                shuffle=False,
                seed=0,
            )

            # when
            values = _collect_first_column(dataset)

            # then
            self.assertEqual(values, [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])

    def test_shuffle_is_reproducible_for_same_seed_and_differs_for_other_seed(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )
            dataset_a = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=True,
                seed=7,
            )
            dataset_b = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=True,
                seed=7,
            )
            dataset_c = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=True,
                seed=8,
            )

            # when
            values_a = _collect_first_column(dataset_a)
            values_b = _collect_first_column(dataset_b)
            values_c = _collect_first_column(dataset_c)

            # then
            self.assertEqual(values_a, values_b)
            self.assertNotEqual(values_a, values_c)

    def test_split_selection_works_for_stored_indices(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="stored_indices"
            )
            dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="val",
                batch_size=2,
                shuffle=False,
                seed=0,
            )

            # when
            values = _collect_first_column(dataset)

            # then
            self.assertEqual(values, [2.0, 12.0, 22.0])

    def test_stored_indices_shuffle_is_reproducible_for_seed(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="stored_indices"
            )
            dataset_a = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=True,
                seed=11,
            )
            dataset_b = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=True,
                seed=11,
            )
            dataset_c = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=True,
                seed=12,
            )

            # when
            values_a = _collect_first_column(dataset_a)
            values_b = _collect_first_column(dataset_b)
            values_c = _collect_first_column(dataset_c)

            # then
            self.assertEqual(values_a, values_b)
            self.assertNotEqual(values_a, values_c)

    def test_max_samples_limit_is_respected(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )
            dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=4,
                shuffle=False,
                seed=0,
                max_samples=5,
            )

            # when
            values = _collect_first_column(dataset)

            # then
            self.assertEqual(len(values), 5)
            self.assertEqual(values, [0.0, 1.0, 2.0, 3.0, 10.0])

    def test_dataset_stored_indices_empty_and_out_of_range_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            shard_path = os.path.join(activations_dir, "shard_00000.pt")
            torch.save(
                {"acts": torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])}, shard_path
            )
            split_indices_path = os.path.join(activations_dir, "split_indices.pt")
            manifest_path = os.path.join(activations_dir, "activations_manifest.json")
            _write_manifest(
                manifest_path,
                {
                    "activation": {"d_in": 2},
                    "shards": [
                        {
                            "filename": "shard_00000.pt",
                            "start_vector_index": 0,
                            "end_vector_index": 3,
                        }
                    ],
                    "split": {
                        "method": "stored_indices",
                        "indices_filename": "split_indices.pt",
                    },
                },
            )

            # when
            torch.save(
                {
                    "train": torch.tensor([], dtype=torch.int64),
                    "val": torch.tensor([], dtype=torch.int64),
                    "test": torch.tensor([], dtype=torch.int64),
                },
                split_indices_path,
            )
            empty_dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=False,
                seed=0,
            )

            # then
            self.assertEqual(empty_dataset.selected_shard_indices, [])
            self.assertEqual(list(empty_dataset), [])

            # when
            torch.save(
                {
                    "train": torch.tensor([0, 99], dtype=torch.int64),
                    "val": torch.tensor([], dtype=torch.int64),
                    "test": torch.tensor([], dtype=torch.int64),
                },
                split_indices_path,
            )
            out_of_range_dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=False,
                seed=0,
            )
            values = _collect_first_column(out_of_range_dataset)

            # then
            self.assertEqual(values, [1.0])

    def test_dataset_iter_returns_early_when_remaining_limit_rounds_to_zero(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )
            dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=False,
                seed=0,
                max_samples=1,
            )
            dataset.max_samples = 0.5

            # when
            batches = list(dataset)

            # then
            self.assertEqual(batches, [])

    def test_stored_indices_iter_handles_empty_local_indices_and_remaining_limit(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="stored_indices"
            )
            empty_local_dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=False,
                seed=0,
            )
            empty_local_dataset._selected_global_indices = torch.tensor(
                [], dtype=torch.int64
            )
            empty_local_dataset.selected_shard_indices = [0]

            # when
            empty_batches = list(empty_local_dataset)

            # then
            self.assertEqual(empty_batches, [])

            # given
            limited_dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=False,
                seed=0,
            )
            limited_dataset.max_samples = 1.5

            # when
            limited_batches = list(limited_dataset)

            # then
            self.assertEqual(sum(int(batch.shape[0]) for batch in limited_batches), 1)

    def test_missing_shard_raises_clear_error(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )
            missing_shard_path = os.path.join(
                os.path.dirname(manifest_path),
                "shard_00001.pt",
            )
            os.remove(missing_shard_path)

            dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=False,
                seed=0,
            )

            # when / then
            with self.assertRaises(ValueError) as context:
                _ = list(dataset)
            self.assertIn("Missing shard file", str(context.exception))
            self.assertIn("Regenerate the cache", str(context.exception))

    def test_corrupt_shard_raises_clear_error(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )
            shard_path = os.path.join(os.path.dirname(manifest_path), "shard_00000.pt")
            torch.save({"not_acts": torch.tensor([1.0])}, shard_path)

            dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=False,
                seed=0,
            )

            # when / then
            with self.assertRaises(ValueError) as context:
                _ = list(dataset)
            self.assertIn("Missing required key 'acts'", str(context.exception))
            self.assertIn("Regenerate the cache", str(context.exception))

    def test_dataset_constructor_validation_errors(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="bad",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=0,
                    shuffle=False,
                    seed=0,
                )
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                    max_samples=0,
                )

    def test_dataset_invalid_manifest_split_metadata_errors(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "activations_manifest.json")
            _write_manifest(
                manifest_path,
                {
                    "activation": {"d_in": 2},
                    "shards": [],
                    "split": "bad",
                },
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )

            _write_manifest(
                manifest_path,
                {
                    "activation": {"d_in": 2},
                    "shards": [],
                    "split": {"method": "bad"},
                },
            )
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )

            _write_manifest(
                manifest_path,
                {
                    "activation": {"d_in": 2},
                    "shards": [],
                    "split": {"method": "shard_ranges"},
                },
            )
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )

    def test_dataset_invalid_stored_indices_metadata_and_files(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            manifest_path = os.path.join(activations_dir, "activations_manifest.json")
            _write_manifest(
                manifest_path,
                {
                    "activation": {"d_in": 2},
                    "shards": [
                        {
                            "filename": "shard_00000.pt",
                            "start_vector_index": 0,
                            "end_vector_index": 1,
                        }
                    ],
                    "split": {"method": "stored_indices"},
                },
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )

            _write_manifest(
                manifest_path,
                {
                    "activation": {"d_in": 2},
                    "shards": [
                        {
                            "filename": "shard_00000.pt",
                            "start_vector_index": 0,
                            "end_vector_index": 1,
                        }
                    ],
                    "split": {
                        "method": "stored_indices",
                        "indices_filename": "split_indices.pt",
                    },
                },
            )
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )

            with open(os.path.join(activations_dir, "split_indices.pt"), "w") as file:
                file.write("bad")
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )

            torch.save([], os.path.join(activations_dir, "split_indices.pt"))
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )

            torch.save(
                {"val": torch.tensor([0])},
                os.path.join(activations_dir, "split_indices.pt"),
            )
            with self.assertRaises(ValueError):
                _ = ActivationShardDataset(
                    manifest_path=manifest_path,
                    split="train",
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                )

    def test_dataset_stored_indices_dtype_conversion_and_max_samples(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            shard_path = os.path.join(activations_dir, "shard_00000.pt")
            torch.save(
                {"acts": torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])}, shard_path
            )
            torch.save(
                {
                    "train": torch.tensor([2, 1, 0], dtype=torch.int32),
                    "val": torch.tensor([], dtype=torch.int64),
                    "test": torch.tensor([], dtype=torch.int64),
                },
                os.path.join(activations_dir, "split_indices.pt"),
            )
            manifest_path = os.path.join(activations_dir, "activations_manifest.json")
            _write_manifest(
                manifest_path,
                {
                    "activation": {"d_in": 2},
                    "shards": [
                        {
                            "filename": "shard_00000.pt",
                            "start_vector_index": 0,
                            "end_vector_index": 3,
                        }
                    ],
                    "split": {
                        "method": "stored_indices",
                        "indices_filename": "split_indices.pt",
                    },
                },
            )

            # when
            dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=10,
                shuffle=False,
                seed=0,
                max_samples=2,
            )
            values = _collect_first_column(dataset)

            # then
            self.assertEqual(values, [1.0, 2.0])

    def test_dataset_iter_raises_for_invalid_shard_indices(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )
            dataset = ActivationShardDataset(
                manifest_path=manifest_path,
                split="train",
                batch_size=2,
                shuffle=False,
                seed=0,
            )
            dataset.selected_shard_indices = [-1]

            # when / then
            with self.assertRaises(ValueError):
                _ = list(dataset)

    def test_create_activation_dataset_wrapper(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _build_shards_and_manifest(
                tmpdir, split_method="shard_ranges"
            )

            # when
            dataset = create_activation_dataset(
                manifest_path=manifest_path,
                split="val",
                batch_size=2,
                shuffle=False,
                seed=0,
                max_train_samples=1,
                max_val_samples=2,
                max_test_samples=3,
            )
            values = _collect_first_column(dataset)

            # then
            self.assertEqual(values, [20.0, 21.0])
