import json
import os
import tempfile
import unittest
from typing import Any, Dict, Tuple

import yaml
from pydantic import ValidationError
from unittest.mock import patch

from sgnsae.experiments.llm_benchmark.config import (
    ActivationCacheConfig,
    ImageActivationCacheConfig,
    ImageDataConfig,
    LLMModelConfig,
    LLMBenchmarkConfig,
    TextDataConfig,
    TrainingConfig,
    VisionModelConfig,
    _RNG_FINGERPRINT_EXCLUDED_TRAINING_KEYS,
    _compute_config_fingerprint,
    _compute_rng_fingerprint,
    _load_raw_config,
    _normalize_local_data_paths,
    _resolve_local_path,
    load_benchmark_config,
    resolve_benchmark_config,
)


def _base_config(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": [2, 1],
        "llm": {"model_id": "sshleifer/tiny-gpt2"},
        "data": {"backend": "hf", "dataset_id": "example/dataset", "split": "train"},
        "hook": {"hook_name": "mlp_out", "layer_selector": "mid"},
        "variants": [
            {
                "name": "variant_a",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": [1e-3, 1e-4],
            },
            {
                "name": "variant_b",
                "sae_class": "TopKSAE",
                "sign_aware": False,
                "d_hidden": 8,
                "topk_k": [8, 4],
            },
        ],
    }


def _base_image_config(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": [2, 1],
        "modality": "image",
        "vision": {
            "model_id": "facebook/dinov2-giant",
            "torch_dtype": "float16",
            "image_size": 224,
        },
        "data": {
            "backend": "hf",
            "dataset_id": "ILSVRC/imagenet-1k",
            "split": "train",
            "streaming": True,
            "image_field": "image",
        },
        "hook": {"hook_name": "hidden_state", "layer_selector": "mid"},
        "cache": {
            "image_batch_size": 2,
            "image_size": 224,
            "max_images": 16,
            "vectors_per_shard": 64,
            "split_method": "stored_indices",
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "cache_seed": 0,
            "compute_dtype": "float32",
            "storage_dtype": "float32",
            "token_selection": "patch_only",
            "representation": "hidden_state",
        },
        "variants": [
            {
                "name": "variant_a",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": [1e-3, 1e-4],
            },
            {
                "name": "variant_b",
                "sae_class": "TopKSAE",
                "sign_aware": False,
                "d_hidden": 8,
                "topk_k": [8, 4],
            },
        ],
        "loss_recovered": {"enabled": False},
    }


class TestLLMBenchmarkConfig(unittest.TestCase):
    def test_load_benchmark_config_supports_yaml(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.safe_dump(_base_config(output_dir=output_dir), f)

            # when
            config = load_benchmark_config(config_path)

            # then
            self.assertEqual(config.output_dir, output_dir)

    def test_load_benchmark_config_resolves_relative_imagefolder_data_dir(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            image_dir = os.path.join(tmpdir, "data", "images")
            os.makedirs(image_dir, exist_ok=True)
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(config_dir, "image_config.json")
            raw_config = _base_image_config(output_dir=output_dir)
            raw_config["data"]["backend"] = "imagefolder"
            raw_config["data"]["data_dir"] = os.path.join("..", "data", "images")
            with open(config_path, "w") as file:
                json.dump(raw_config, file, indent=2)

            original_cwd = os.getcwd()
            different_cwd = os.path.join(tmpdir, "different_cwd")
            os.makedirs(different_cwd, exist_ok=True)
            os.chdir(different_cwd)
            try:
                # when
                config = load_benchmark_config(config_path)
            finally:
                os.chdir(original_cwd)

            # then
            self.assertEqual(config.data.data_dir, os.path.abspath(image_dir))
            self.assertTrue(os.path.isabs(str(config.data.data_dir)))

    def test_load_benchmark_config_resolves_local_jsonl_paths_relative_to_config(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            data_file_path = os.path.join(data_dir, "sample.jsonl")
            with open(data_file_path, "w") as file:
                file.write('{"text": "hello"}\n')

            output_dir = os.path.join(tmpdir, "run_root")
            files_config_path = os.path.join(config_dir, "local_files.json")
            files_raw_config = _base_config(output_dir=output_dir)
            files_raw_config["data"] = {
                "backend": "local_jsonl",
                "data_files": [os.path.join("..", "data", "sample.jsonl")],
            }
            with open(files_config_path, "w") as file:
                json.dump(files_raw_config, file, indent=2)

            glob_config_path = os.path.join(config_dir, "local_glob.json")
            glob_raw_config = _base_config(output_dir=output_dir)
            glob_raw_config["data"] = {
                "backend": "local_jsonl",
                "data_files_glob": os.path.join("..", "data", "*.jsonl"),
            }
            with open(glob_config_path, "w") as file:
                json.dump(glob_raw_config, file, indent=2)

            original_cwd = os.getcwd()
            different_cwd = os.path.join(tmpdir, "different_cwd")
            os.makedirs(different_cwd, exist_ok=True)
            os.chdir(different_cwd)
            try:
                # when
                files_config = load_benchmark_config(files_config_path)
                glob_config = load_benchmark_config(glob_config_path)
            finally:
                os.chdir(original_cwd)

            # then
            self.assertEqual(
                files_config.data.data_files,
                [os.path.abspath(data_file_path)],
            )
            self.assertEqual(
                glob_config.data.data_files_glob,
                os.path.abspath(os.path.join(data_dir, "*.jsonl")),
            )

    def test_normalize_local_data_paths_covers_non_dict_and_non_string_entries(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            raw_non_dict_data = {"data": [1, 2, 3]}
            raw_local_jsonl = {
                "data": {
                    "backend": "local_jsonl",
                    "data_files": [123, "data/sample.jsonl"],
                }
            }

            # when
            normalized_non_dict_data = _normalize_local_data_paths(
                raw_config=raw_non_dict_data,
                config_path=config_path,
            )
            normalized_local_jsonl = _normalize_local_data_paths(
                raw_config=raw_local_jsonl,
                config_path=config_path,
            )

            # then
            self.assertEqual(normalized_non_dict_data, raw_non_dict_data)
            self.assertEqual(normalized_local_jsonl["data"]["data_files"][0], 123)
            self.assertEqual(
                normalized_local_jsonl["data"]["data_files"][1],
                os.path.abspath(os.path.join(tmpdir, "data", "sample.jsonl")),
            )

            # given
            raw_imagefolder_with_blank_dir = {
                "data": {
                    "backend": "imagefolder",
                    "data_dir": "   ",
                }
            }

            # when
            normalized_imagefolder_blank_dir = _normalize_local_data_paths(
                raw_config=raw_imagefolder_with_blank_dir,
                config_path=config_path,
            )

            # then
            self.assertEqual(
                normalized_imagefolder_blank_dir["data"]["data_dir"],
                "   ",
            )

    def test_modality_defaults_to_text_when_omitted(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw.pop("modality", None)

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.modality, "text")

    def test_modality_image_requires_vision_data_and_image_cache(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_image_config(output_dir=output_dir)
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.modality, "image")

            # when / then
            raw = _base_image_config(output_dir=output_dir)
            raw.pop("vision")
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_image_config(output_dir=output_dir)
            raw["data"] = {
                "backend": "local_jsonl",
                "data_files": ["dummy.jsonl"],
                "text_field": "text",
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_image_config(output_dir=output_dir)
            raw["cache"] = {
                "llm_batch_size": 2,
                "max_seq_len": 8,
                "vectors_per_shard": 64,
                "train_fraction": 0.8,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_modality_image_rejects_loss_recovered_enabled(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["loss_recovered"] = {"enabled": True}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_modality_image_requires_hidden_state_hook_name(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["hook"]["hook_name"] = "mlp_out"

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_modality_image_normalizes_whitespace_in_hidden_state_hook_name(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["hook"]["hook_name"] = " hidden_state "

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.hook.hook_name, "hidden_state")

    def test_modality_image_rejects_hook_module_path(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["hook"]["module_path"] = "encoder.layers.1.mlp"

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_image_size_mismatch_between_vision_and_cache_is_rejected(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["vision"]["image_size"] = 128
            raw["cache"]["image_size"] = 224

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_image_size_from_vision_is_copied_to_cache_when_cache_not_explicit(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["vision"]["image_size"] = 128
            raw["cache"].pop("image_size")

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.vision.image_size, 128)
            self.assertEqual(config.cache.image_size, 128)

    def test_image_size_from_cache_is_copied_to_vision_when_vision_not_explicit(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["vision"].pop("image_size")
            raw["cache"]["image_size"] = 192

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.cache.image_size, 192)
            self.assertEqual(config.vision.image_size, 192)

    def test_image_size_consistency_handles_manual_non_explicit_mismatch_branch(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            )
            config.vision = VisionModelConfig.model_construct(
                _fields_set=set(),
                model_id=str(config.vision.model_id),
                image_size=128,
            )
            config.cache = ImageActivationCacheConfig.model_construct(
                _fields_set=set(),
                image_size=224,
            )

            # when
            updated = config._validate_image_size_consistency()

            # then
            self.assertEqual(updated.vision.image_size, 128)
            self.assertEqual(updated.cache.image_size, 128)

    def test_image_size_consistency_handles_manual_non_explicit_equal_branch(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            )
            config.vision = VisionModelConfig.model_construct(
                _fields_set=set(),
                model_id=str(config.vision.model_id),
                image_size=224,
            )
            config.cache = ImageActivationCacheConfig.model_construct(
                _fields_set=set(),
                image_size=224,
            )

            # when
            updated = config._validate_image_size_consistency()

            # then
            self.assertEqual(updated.vision.image_size, 224)
            self.assertEqual(updated.cache.image_size, 224)

    def test_image_size_consistency_type_guard_branch_raises(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            )
            config.vision = None

            # when / then
            with self.assertRaises(ValueError):
                _ = config._validate_image_size_consistency()

    def test_stages_accept_examples_stage(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["stages"] = ["cache", "train", "eval", "examples", "plot", "summarize"]
            config = LLMBenchmarkConfig.model_validate(raw)

            # when
            resolved = resolve_benchmark_config(config=config)

            # then
            self.assertIn("examples", config.stages)
            self.assertIn("examples", resolved["stages"])

    def test_vision_config_validation_covers_optional_and_dtype_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_image_config(output_dir=output_dir)
            raw["vision"]["image_processor_id"] = None
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.vision.image_processor_id)

            # when / then
            raw = _base_image_config(output_dir=output_dir)
            raw["vision"]["torch_dtype"] = 123
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_image_config(output_dir=output_dir)
            raw["vision"]["torch_dtype"] = "bad_dtype"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_image_data_config_validation_covers_backend_and_label_branches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            image_dir = os.path.join(tmpdir, "images")
            os.makedirs(image_dir, exist_ok=True)

            # when
            raw = _base_image_config(output_dir=output_dir)
            raw["data"]["label_field"] = None
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.data.label_field)

            # when
            raw = _base_image_config(output_dir=output_dir)
            raw["data"]["backend"] = "hf"
            raw["data"]["data_dir"] = image_dir
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            # when
            raw = _base_image_config(output_dir=output_dir)
            raw["data"]["backend"] = "imagefolder"
            raw["data"]["data_dir"] = None
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            # when
            raw = _base_image_config(output_dir=output_dir)
            raw["data"]["backend"] = "imagefolder"
            raw["data"]["data_dir"] = os.path.join(tmpdir, "does_not_exist")
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            # when
            raw = _base_image_config(output_dir=output_dir)
            raw["data"]["backend"] = "imagefolder"
            raw["data"]["data_dir"] = image_dir
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.data.data_dir, image_dir)

    def test_image_cache_validation_covers_new_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_image_config(output_dir=output_dir)
            raw["cache"]["max_images"] = None
            raw["cache"]["cache_seed"] = None
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.cache.max_images)
            self.assertIsNone(config.cache.cache_seed)

            # when / then
            raw = _base_image_config(output_dir=output_dir)
            raw["cache"]["max_images"] = True
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_image_config(output_dir=output_dir)
            raw["cache"]["max_images"] = 0
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_image_config(output_dir=output_dir)
            raw["cache"]["cache_seed"] = True
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_image_config(output_dir=output_dir)
            raw["cache"]["train_fraction"] = "bad"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_image_config(output_dir=output_dir)
            raw["cache"]["train_fraction"] = 1.5
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_image_config(output_dir=output_dir)
            raw["cache"]["train_fraction"] = 0.6
            raw["cache"]["val_fraction"] = 0.3
            raw["cache"]["test_fraction"] = 0.2
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_latent_examples_validation_covers_new_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {
                "sample_size": 16,
                "max_latents": 4,
                "max_examples_per_latent": 2,
                "activation_threshold": 0.1,
                "anchor_l0": 8.0,
                "latent_selection": "explicit",
                "explicit_latents": [0, 2],
                "seeds": [0, 1],
                "highlight_markers": ["<<", ">>"],
            }
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.latent_examples.explicit_latents, [0, 2])
            self.assertEqual(config.latent_examples.highlight_markers, ("<<", ">>"))

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"activation_threshold": -1.0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"sample_size": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"anchor_l0": 0.0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"latent_selection": "explicit"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {
                "latent_selection": "top_by_freq",
                "explicit_latents": [0],
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {
                "latent_selection": "explicit",
                "explicit_latents": "bad",
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {
                "latent_selection": "explicit",
                "explicit_latents": [],
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {
                "latent_selection": "explicit",
                "explicit_latents": [True],
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {
                "latent_selection": "explicit",
                "explicit_latents": [-1],
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"seeds": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"seeds": []}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"seeds": [1, "x"]}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"seeds": [1, 1]}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"highlight_markers": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"highlight_markers": [1, "<|"]}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["latent_examples"] = {"highlight_markers": ["|>", "|>"]}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_modality_validation_rejects_non_string_and_unknown_value(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["modality"] = 123
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["modality"] = "audio"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_data_and_cache_validators_accept_model_instances(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            image_data = ImageDataConfig.model_validate(
                {
                    "backend": "hf",
                    "dataset_id": "ILSVRC/imagenet-1k",
                    "split": "train",
                    "streaming": True,
                    "image_field": "image",
                }
            )
            image_cache = ImageActivationCacheConfig.model_validate(
                {
                    "image_batch_size": 2,
                    "image_size": 224,
                    "max_images": 16,
                    "vectors_per_shard": 64,
                    "split_method": "stored_indices",
                    "train_fraction": 0.8,
                    "val_fraction": 0.1,
                    "test_fraction": 0.1,
                    "cache_seed": 0,
                    "compute_dtype": "float32",
                    "storage_dtype": "float32",
                    "token_selection": "patch_only",
                    "representation": "hidden_state",
                }
            )
            text_data = TextDataConfig.model_validate(
                {
                    "backend": "hf",
                    "dataset_id": "example/dataset",
                    "split": "train",
                    "text_field": "text",
                }
            )
            text_cache = ActivationCacheConfig.model_validate({"text_field": "text"})

            # when
            raw = _base_image_config(output_dir=output_dir)
            raw["data"] = image_data
            raw["cache"] = image_cache
            image_config = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["data"] = text_data
            raw["cache"] = text_cache
            text_config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIs(image_config.data, image_data)
            self.assertIs(image_config.cache, image_cache)
            self.assertIs(text_config.data, text_data)
            self.assertIs(text_config.cache, text_cache)

    def test_modality_requirement_and_type_guard_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=output_dir)
            )
            config.llm = None
            with self.assertRaises(ValueError):
                _ = config._validate_modality_requirements()

            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=output_dir)
            )
            config.vision = VisionModelConfig.model_validate(
                {"model_id": "facebook/dinov2-giant"}
            )
            with self.assertRaises(ValueError):
                _ = config._validate_modality_requirements()

            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=output_dir)
            )
            config.data = ImageDataConfig.model_validate(
                {
                    "backend": "hf",
                    "dataset_id": "ILSVRC/imagenet-1k",
                    "split": "train",
                    "image_field": "image",
                }
            )
            with self.assertRaises(ValueError):
                _ = config._validate_modality_requirements()

            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=output_dir)
            )
            config.cache = ImageActivationCacheConfig.model_validate(
                {
                    "image_batch_size": 2,
                    "image_size": 224,
                    "max_images": 16,
                    "vectors_per_shard": 64,
                    "split_method": "stored_indices",
                    "train_fraction": 0.8,
                    "val_fraction": 0.1,
                    "test_fraction": 0.1,
                    "cache_seed": 0,
                    "compute_dtype": "float32",
                    "storage_dtype": "float32",
                    "token_selection": "patch_only",
                    "representation": "hidden_state",
                }
            )
            with self.assertRaises(ValueError):
                _ = config._validate_modality_requirements()

            image_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir)
            )
            image_config.llm = LLMModelConfig.model_validate(
                {"model_id": "sshleifer/tiny-gpt2"}
            )
            with self.assertRaises(ValueError):
                _ = image_config._validate_modality_requirements()

            image_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir)
            )
            image_config.data = TextDataConfig.model_validate(
                {
                    "backend": "hf",
                    "dataset_id": "example/dataset",
                    "split": "train",
                }
            )
            with self.assertRaises(ValueError):
                _ = image_config._validate_modality_requirements()

            image_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir)
            )
            image_config.cache = ActivationCacheConfig.model_validate({})
            with self.assertRaises(ValueError):
                _ = image_config._validate_modality_requirements()

            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=output_dir)
            )
            config.data = ImageDataConfig.model_validate(
                {
                    "backend": "hf",
                    "dataset_id": "ILSVRC/imagenet-1k",
                    "split": "train",
                    "image_field": "image",
                }
            )
            with self.assertRaises(ValueError):
                _ = config._validate_text_field_consistency()

    def test_rng_fingerprint_handles_non_dict_training(self) -> None:
        # given
        resolved_config = {
            "schema_version": 1,
            "config_fingerprint": "unused",
            "rng_fingerprint": "unused",
            "output_dir": "out",
            "device": "cpu",
            "overwrite": False,
            "resume": True,
            "stages": ["cache"],
            "training": ["not", "a", "dict"],
        }

        # when
        fingerprint = _compute_rng_fingerprint(resolved_config)

        # then
        self.assertIsInstance(fingerprint, str)
        self.assertGreater(len(fingerprint), 0)

    def test_text_data_local_jsonl_accepts_glob_and_files_success_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["data"] = {"backend": "local_jsonl", "data_files_glob": "data/*.jsonl"}
            with_glob = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["data"] = {"backend": "local_jsonl", "data_files": ["a.jsonl"]}
            with_files = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(with_glob.data.data_files_glob, "data/*.jsonl")
            self.assertEqual(with_files.data.data_files, ["a.jsonl"])

    def test_vision_image_processor_id_non_empty_is_accepted(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["vision"]["image_processor_id"] = "facebook/dinov2-giant"

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.vision.image_processor_id, "facebook/dinov2-giant")

    def test_loss_recovered_validation_covers_remaining_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {
                "enabled": False,
                "dataset_split": None,
                "denominator_epsilon": 1e-8,
            }
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.loss_recovered.dataset_split)

            # when
            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {
                "enabled": False,
                "dataset_split": "validation",
                "denominator_epsilon": 1e-8,
            }
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.loss_recovered.dataset_split, "validation")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {"seed": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {"denominator_epsilon": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_feature_ablation_seed_rejects_bool(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["feature_ablation"] = {"seed": True}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_latent_examples_accepts_explicit_none_fields(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["latent_examples"] = {
                "explicit_latents": None,
                "seeds": None,
                "latent_selection": "top_by_max_abs",
            }

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.latent_examples.explicit_latents)
            self.assertIsNone(config.latent_examples.seeds)

    def test_latent_examples_default_selection_is_top_by_mean_abs(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.latent_examples.latent_selection, "top_by_mean_abs")

    def test_variant_loss_knob_validators_cover_error_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["variants"][0]["aux_loss_coeff"] = float("inf")
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["variants"][0]["dead_zone_reg_type"] = 1
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["variants"][0]["dead_zone_reg_type"] = "l1"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["variants"][0]["dead_zone_reg_type"] = "l2"
            config = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(config.variants[0].dead_zone_reg_type, "l2")

            raw = _base_config(output_dir=output_dir)
            raw["variants"][0]["aux_loss_coeff"] = 0.5
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["variants"][0]["sae_class"] = "GatedSAE"
            raw["variants"][0]["dead_zone_reg_coeff"] = 0.1
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            absolute_path = os.path.abspath(os.path.join(tmpdir, "absolute.jsonl"))
            resolved_path = _resolve_local_path(absolute_path, tmpdir)
            self.assertEqual(resolved_path, absolute_path)

    def test_resolve_benchmark_config_orders_subruns_deterministically(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            )

            # when
            resolved = resolve_benchmark_config(config)

            # then
            self.assertEqual(resolved["seeds"], [1, 2])
            subruns = resolved["subruns"]
            self.assertIsInstance(subruns, list)
            self.assertGreaterEqual(len(subruns), 1)
            first = subruns[0]
            self.assertEqual(first["variant_name"], "variant_a")
            self.assertEqual(first["seed"], 1)
            self.assertEqual(first["sweep_param"], "lambda")
            self.assertEqual(first["sweep_value"], 1e-3)

    def test_resolve_benchmark_config_rejects_unknown_stage_override(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = resolve_benchmark_config(config, stages_override=["unknown"])

    def test_resolve_benchmark_config_rejects_validate_and_other_stage(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = resolve_benchmark_config(
                    config,
                    stages_override=["validate", "cache"],
                )

    def test_resolve_benchmark_config_rejects_variant_slug_collision(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            raw = _base_config(output_dir=output_dir)
            raw["variants"] = [
                {
                    "name": "foo!",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                },
                {
                    "name": "foo?",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-4,
                },
            ]
            config = LLMBenchmarkConfig.model_validate(raw)

            # when / then
            with self.assertRaises(ValueError):
                _ = resolve_benchmark_config(config)

    def test_seed_list_rejects_duplicates(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["seed"] = [1, 1]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_hook_requires_exactly_one_of_layer_or_layer_selector(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["hook"] = {"hook_name": "mlp_out"}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_text_data_local_jsonl_rejects_both_glob_and_files(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["data"] = {
                "backend": "local_jsonl",
                "data_files_glob": "data/*.jsonl",
                "data_files": ["a.jsonl"],
            }

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_text_data_hf_requires_split_when_dataset_id_provided(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["data"] = {"backend": "hf", "dataset_id": "example/dataset"}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_text_data_hf_requires_dataset_id_when_split_is_provided(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["data"] = {"backend": "hf", "split": "train"}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_text_data_hf_rejects_data_files_fields(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["data"] = {
                "backend": "hf",
                "dataset_id": "example/dataset",
                "split": "train",
                "data_files": ["a.jsonl"],
            }

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_text_data_local_jsonl_rejects_dataset_id_split_revision(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["data"] = {
                "backend": "local_jsonl",
                "dataset_id": "example/dataset",
                "data_files": ["a.jsonl"],
            }

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_text_field_consistency_propagates_and_rejects_conflicts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["data"]["text_field"] = "content"
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.data.text_field, "content")
            self.assertEqual(config.cache.text_field, "content")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"text_field": "content"}
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.data.text_field, "content")
            self.assertEqual(config.cache.text_field, "content")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["data"]["text_field"] = "content"
            raw["cache"] = {"text_field": "text"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            # when
            raw = _base_config(output_dir=output_dir)
            config = LLMBenchmarkConfig.model_validate(raw)
            config.data.text_field = "body"
            config.cache.text_field = "text"
            config.data.model_fields_set.discard("text_field")
            config.cache.model_fields_set.discard("text_field")
            config = config._validate_text_field_consistency()

            # then
            self.assertEqual(config.data.text_field, "body")
            self.assertEqual(config.cache.text_field, "body")

    def test_llm_torch_dtype_validation_accepts_supported_values_only(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["llm"]["torch_dtype"] = "BFLOAT16"
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.llm.torch_dtype, "bfloat16")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["llm"]["torch_dtype"] = "bad_dtype"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["llm"]["torch_dtype"] = 123
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_device_validation_accepts_cuda_index_and_rejects_noncanonical(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["device"] = "cuda:0"
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.device, "cuda:0")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["device"] = "CUDA:0"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["device"] = " cuda:0 "
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["device"] = "cuda:-1"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["device"] = 123
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_text_data_local_jsonl_rejects_empty_glob_and_empty_files_and_blank_entries(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["data"] = {"backend": "local_jsonl", "data_files_glob": ""}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["data"] = {"backend": "local_jsonl", "data_files": []}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["data"] = {"backend": "local_jsonl", "data_files": [""]}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_hook_layer_validation_rejects_bool_and_negative(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["hook"] = {"hook_name": "mlp_out", "layer": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["hook"] = {"hook_name": "mlp_out", "layer": -1}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_loss_recovered_config_validates_batch_size_and_epsilon(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {
                "enabled": True,
                "max_eval_sequences": 8,
                "llm_batch_size": 16,
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {
                "enabled": True,
                "max_eval_sequences": 8,
                "llm_batch_size": 4,
                "denominator_epsilon": 0.0,
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {
                "enabled": True,
                "max_eval_sequences": 8,
                "llm_batch_size": 4,
                "denominator_epsilon": 1e-8,
            }
            config = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(config.loss_recovered.llm_batch_size, 4)

    def test_feature_ablation_delta_lm_loss_requires_loss_recovered_enabled(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["feature_ablation"] = {"enabled": True, "metric": "delta_lm_loss"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {"enabled": True}
            raw["feature_ablation"] = {
                "enabled": True,
                "metric": "delta_lm_loss",
                "num_features": 16,
                "lm_max_features": 17,
            }
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {"enabled": True}
            raw["feature_ablation"] = {
                "enabled": True,
                "metric": "delta_lm_loss",
                "num_features": 16,
                "lm_max_features": 8,
            }
            config = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(config.feature_ablation.metric, "delta_lm_loss")

    def test_hook_layer_selector_validation_rejects_invalid_string_bool_and_range(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["hook"] = {"hook_name": "mlp_out", "layer_selector": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["hook"] = {"hook_name": "mlp_out", "layer_selector": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["hook"] = {"hook_name": "mlp_out", "layer_selector": 1.5}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_hook_validation_accepts_none_fields_and_float_selector(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["hook"] = {
                "hook_name": "mlp_out",
                "layer": None,
                "layer_selector": 0.5,
                "module_path": None,
            }

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.hook.layer)
            self.assertEqual(config.hook.layer_selector, 0.5)
            self.assertIsNone(config.hook.module_path)

    def test_hook_validation_accepts_layer_with_layer_selector_none(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["hook"] = {
                "hook_name": "mlp_out",
                "layer": 0,
                "layer_selector": None,
            }

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.hook.layer, 0)
            self.assertIsNone(config.hook.layer_selector)

    def test_hook_module_path_must_be_non_empty_when_provided(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["hook"] = {"hook_name": "mlp_out", "layer": 0, "module_path": ""}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_cache_validation_covers_none_bool_and_positive_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"max_tokens": None, "max_sequences": None}
            config = LLMBenchmarkConfig.model_validate(raw)
            self.assertIsNone(config.cache.max_tokens)
            self.assertIsNone(config.cache.max_sequences)

            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"max_tokens": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"max_tokens": 10, "max_sequences": 10}
            config = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(config.cache.max_tokens, 10)
            self.assertEqual(config.cache.max_sequences, 10)

            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"max_sequences": 0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_cache_seed_split_fraction_and_training_sample_limit_validation(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {
                "cache_seed": None,
                "train_fraction": 0.7,
                "val_fraction": 0.2,
                "test_fraction": 0.1,
            }
            raw["training"] = {"max_train_samples": None}
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.cache.cache_seed)
            self.assertIsNone(config.training.max_train_samples)

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"cache_seed": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"train_fraction": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"val_fraction": 1.2}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_cache_and_training_validation_rejects_invalid_integers(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"max_tokens": 0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"max_sequences": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["cache"] = {"vectors_per_shard": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"train_steps": 0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"batch_size": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_evaluation_validation_rejects_invalid_values(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["evaluation"] = {"alive_threshold": -1.0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["evaluation"] = {"alive_threshold": True}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["evaluation"] = {"alive_min_fraction": 2.0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["evaluation"] = {"alive_min_fraction": 0.5}
            config = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(config.evaluation.alive_min_fraction, 0.5)

    def test_loss_recovered_and_feature_ablation_validation_rejects_invalid_values(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["loss_recovered"] = {"max_eval_sequences": 0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["feature_ablation"] = {"num_features": 0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_requires_exactly_one_sweep_controller(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_both_sparsity_coeff_and_topk_k(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "topk_k": 16,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_unknown_sae_class(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken",
                    "sae_class": "UnknownSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_topk_k_for_non_topk_classes(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": 16,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_sparsity_coeff_for_topk_classes(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_topk_k_greater_than_d_hidden(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken_k_width",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": 16,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_sign_aware_for_unsupported_class(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken",
                    "sae_class": "TopKSAE",
                    "sign_aware": True,
                    "d_hidden": 8,
                    "topk_k": 8,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_delta_initialization_when_sign_aware_is_false(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "delta_pos_init": 0.01,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_non_positive_jumprelu_initialization(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken_theta",
                    "sae_class": "JumpReLUSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "theta_init": 0.0,
                    "jump_init": 0.01,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw["variants"] = [
                {
                    "name": "broken_jump",
                    "sae_class": "JumpReLUSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "theta_init": 0.01,
                    "jump_init": 0.0,
                }
            ]
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_theta_init_for_unsupported_class(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken_theta_for_sae",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "theta_init": 0.1,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_jump_init_for_non_jumprelu_class(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken_jump_for_soft_threshold",
                    "sae_class": "SignedSoftThresholdSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "jump_init": 0.1,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_controller_empty_lists_rejected(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["variants"] = [
                {
                    "name": "broken_lambda",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": [],
                }
            ]
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["variants"] = [
                {
                    "name": "broken_k",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": [],
                }
            ]
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_validates_optional_none_fields_and_variant_slug_success(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            raw = _base_config(output_dir=output_dir)
            raw["variants"] = [
                {
                    "name": "has_slug_none",
                    "variant_slug": None,
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "topk_k": None,
                },
                {
                    "name": "k_variant",
                    "variant_slug": "k_variant",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": None,
                    "topk_k": 8,
                },
            ]

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.variants[0].variant_slug)
            self.assertIsNone(config.variants[1].sparsity_coeff)
            self.assertEqual(config.variants[1].variant_slug, "k_variant")

    def test_seed_validation_rejects_bool_empty_list_non_int_and_non_int_scalar(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["seed"] = True
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["seed"] = []
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["seed"] = [0, "x"]
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["seed"] = "x"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_seed_scalar_int_is_accepted_and_resolve_uses_single_seed(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["seed"] = 5
            config = LLMBenchmarkConfig.model_validate(raw)

            # when
            resolved = resolve_benchmark_config(config)

            # then
            self.assertEqual(config.seed, 5)
            self.assertEqual(resolved["seeds"], [5])

    def test_stages_normalization_rejects_non_strings_and_non_list(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["stages"] = [1]
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["stages"] = "cache"
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_config_stages_reject_validate_with_other_stages(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["stages"] = ["validate", "cache"]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variants_list_validation_accepts_empty_and_rejects_duplicate_names(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["variants"] = []
            config = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(config.variants, [])

            raw = _base_config(output_dir=output_dir)
            raw["variants"] = [
                {
                    "name": "dup",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                },
                {
                    "name": "dup",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-4,
                },
            ]
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_slug_validation_requires_filesystem_safe_slug(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "has_slug",
                    "variant_slug": "Bad Slug",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_load_benchmark_config_rejects_non_object_json(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump([1, 2, 3], f)

            # when / then
            with self.assertRaises(ValueError):
                _ = load_benchmark_config(config_path)

    def test_yaml_loading_requires_pyyaml_when_suffix_is_yaml(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write("output_dir: out\n")

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.config.yaml",
                None,
            ):
                with self.assertRaises(ValueError):
                    _ = _load_raw_config(config_path)

    def test_resolve_benchmark_config_validate_only_branch(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            )

            # when
            resolved = resolve_benchmark_config(config, validate_only=True)

            # then
            self.assertEqual(resolved["stages"], ["validate"])

    def test_resolve_benchmark_config_rejects_validate_only_with_override(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = resolve_benchmark_config(
                    config,
                    stages_override=["cache"],
                    validate_only=True,
                )

    def test_resolve_benchmark_config_covers_topk_k_scalar_branch(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "topk_variant",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": 8,
                }
            ]
            config = LLMBenchmarkConfig.model_validate(raw)

            # when
            resolved = resolve_benchmark_config(config)

            # then
            self.assertEqual(resolved["subruns"][0]["sweep_param"], "k")
            self.assertEqual(resolved["subruns"][0]["sweep_value"], 8)

    def test_cache_alias_fields_are_accepted(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["cache"] = {
                "batch_size": 3,
                "sequence_length": 64,
            }

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.cache.llm_batch_size, 3)
            self.assertEqual(config.cache.max_seq_len, 64)

    def test_cache_split_fractions_must_sum_to_one(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["cache"] = {
                "train_fraction": 0.8,
                "val_fraction": 0.1,
                "test_fraction": 0.2,
            }

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_cache_seed_accepts_integer_value(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["cache"] = {"cache_seed": 7}

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.cache.cache_seed, 7)

    def test_training_max_sample_limits_validation(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["training"] = {"max_train_samples": 0}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_training_field_validation_rejects_invalid_numeric_values(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"learning_rate": 0.0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"adam_beta1": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"adam_beta2": 1.0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"grad_clip_norm": 0.0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"resample_latents_steps": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"resample_latents_steps": 0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"resample_latents_steps": -1}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"checkpoint_model_steps": 0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"checkpoint_model_steps": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"dump_training_metrics_steps": 0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"dump_training_metrics_steps": -1}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"checkpoint_model_steps": 5}
            cfg = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(cfg.training.checkpoint_model_steps, 5)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"dump_training_metrics_steps": 10}
            cfg = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(cfg.training.dump_training_metrics_steps, 10)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {
                "resample_latents_steps": None,
                "checkpoint_model_steps": None,
                "dump_training_metrics_steps": None,
            }
            cfg = LLMBenchmarkConfig.model_validate(raw)
            self.assertIsNone(cfg.training.resample_latents_steps)
            self.assertIsNone(cfg.training.checkpoint_model_steps)
            self.assertIsNone(cfg.training.dump_training_metrics_steps)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"threshold_warmup_steps": 1.5}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"threshold_warmup_steps": -1}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"warmup_steps": 1.5}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["training"] = {"warmup_steps": -1}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_training_accepts_explicit_none_grad_clip_norm(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["training"] = {"grad_clip_norm": None}

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.training.grad_clip_norm)

    def test_training_normalizes_warmup_steps_none_and_missing_to_zero(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_raw = _base_config(output_dir=os.path.join(tmpdir, "missing_run"))
            explicit_none_raw = _base_config(
                output_dir=os.path.join(tmpdir, "explicit_none_run")
            )
            explicit_none_raw["training"] = {"warmup_steps": None}

            # when
            missing_config = LLMBenchmarkConfig.model_validate(missing_raw)
            explicit_none_config = LLMBenchmarkConfig.model_validate(explicit_none_raw)

            # then
            self.assertEqual(missing_config.training.warmup_steps, 0)
            self.assertEqual(explicit_none_config.training.warmup_steps, 0)

    def test_training_multi_sparsity_defaults_are_backward_compatible(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertTrue(config.training.multi_sparsity_enabled)
            self.assertIsNone(config.training.multi_sparsity_max_instances)

            # when
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["training"] = {"multi_sparsity_max_instances": None}
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.training.multi_sparsity_max_instances)

    def test_training_rejects_non_positive_multi_sparsity_max_instances(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            for invalid_value in [0, -1]:
                with self.subTest(invalid_value=invalid_value):
                    raw = _base_config(output_dir=output_dir)
                    raw["training"] = {"multi_sparsity_max_instances": invalid_value}
                    with self.assertRaises(ValidationError) as context:
                        _ = LLMBenchmarkConfig.model_validate(raw)
                    self.assertIn(
                        "training.multi_sparsity_max_instances must be positive",
                        str(context.exception),
                    )

    def test_variant_rejects_duplicate_sparsity_coeff_after_formatting(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            for sparsity_coeffs in [[1e-3, 1e-3], [1e-3, 0.0010000004]]:
                with self.subTest(sparsity_coeffs=sparsity_coeffs):
                    raw = _base_config(output_dir=output_dir)
                    raw["variants"] = [
                        {
                            "name": "collision_variant",
                            "sae_class": "SAE",
                            "sign_aware": False,
                            "d_hidden": 8,
                            "sparsity_coeff": sparsity_coeffs,
                        }
                    ]
                    with self.assertRaises(ValidationError) as context:
                        _ = LLMBenchmarkConfig.model_validate(raw)
                    self.assertIn(
                        "variants[].sparsity_coeff contains duplicate values after "
                        "directory formatting",
                        str(context.exception),
                    )

    def test_variant_rejects_duplicate_topk_k_values(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "duplicate_topk_variant",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": [8, 8],
                }
            ]

            # when / then
            with self.assertRaises(ValidationError) as context:
                _ = LLMBenchmarkConfig.model_validate(raw)
            self.assertIn(
                "variants[].topk_k contains duplicate values",
                str(context.exception),
            )

    def test_variant_accepts_abstopk_gated_magnitude_sae_with_topk_k(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "hybrid_ablation",
                    "sae_class": "AbsTopKGatedMagnitudeSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": [4, 2],
                }
            ]

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.variants[0].sae_class, "AbsTopKGatedMagnitudeSAE")
            self.assertEqual(config.variants[0].topk_k, [4, 2])
            self.assertIsNone(config.variants[0].sparsity_coeff)

    def test_variant_accepts_abstopk_gated_magnitude_sae_with_sign_aware_true(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "hybrid_ablation_sign_aware",
                    "sae_class": "AbsTopKGatedMagnitudeSAE",
                    "sign_aware": True,
                    "d_hidden": 8,
                    "topk_k": [4, 2],
                }
            ]

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.variants[0].sae_class, "AbsTopKGatedMagnitudeSAE")
            self.assertTrue(bool(config.variants[0].sign_aware))
            self.assertEqual(config.variants[0].topk_k, [4, 2])

    def test_variant_rejects_abstopk_gated_magnitude_sae_with_sparsity_coeff(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "hybrid_ablation",
                    "sae_class": "AbsTopKGatedMagnitudeSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_resolved_config_reflects_multi_sparsity_training_fields(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["training"] = {
                "multi_sparsity_enabled": False,
                "multi_sparsity_max_instances": 2,
            }
            config = LLMBenchmarkConfig.model_validate(raw)

            # when
            resolved = resolve_benchmark_config(config)

            # then
            self.assertFalse(resolved["training"]["multi_sparsity_enabled"])
            self.assertEqual(resolved["training"]["multi_sparsity_max_instances"], 2)

    def test_multi_sparsity_fields_change_config_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            raw_default = _base_config(output_dir=output_dir)
            raw_overridden = _base_config(output_dir=output_dir)
            raw_overridden["training"] = {"multi_sparsity_enabled": False}

            # when
            resolved_default = resolve_benchmark_config(
                LLMBenchmarkConfig.model_validate(raw_default)
            )
            resolved_overridden = resolve_benchmark_config(
                LLMBenchmarkConfig.model_validate(raw_overridden)
            )

            # then
            self.assertNotEqual(
                resolved_default["config_fingerprint"],
                resolved_overridden["config_fingerprint"],
            )

    def test_multi_sparsity_max_instances_changes_config_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            raw_default = _base_config(output_dir=output_dir)
            raw_overridden = _base_config(output_dir=output_dir)
            raw_overridden["training"] = {"multi_sparsity_max_instances": 4}

            # when
            resolved_default = resolve_benchmark_config(
                LLMBenchmarkConfig.model_validate(raw_default)
            )
            resolved_overridden = resolve_benchmark_config(
                LLMBenchmarkConfig.model_validate(raw_overridden)
            )

            # then
            self.assertNotEqual(
                resolved_default["config_fingerprint"],
                resolved_overridden["config_fingerprint"],
            )

    def test_multi_sparsity_max_instances_does_not_change_rng_fingerprint(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            raw_default = _base_config(output_dir=output_dir)
            raw_overridden = _base_config(output_dir=output_dir)
            raw_overridden["training"] = {"multi_sparsity_max_instances": 4}

            # when
            resolved_default = resolve_benchmark_config(
                LLMBenchmarkConfig.model_validate(raw_default)
            )
            resolved_overridden = resolve_benchmark_config(
                LLMBenchmarkConfig.model_validate(raw_overridden)
            )

            # then
            self.assertEqual(
                resolved_default["rng_fingerprint"],
                resolved_overridden["rng_fingerprint"],
            )

    def test_multi_sparsity_enabled_does_not_change_rng_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            raw_default = _base_config(output_dir=output_dir)
            raw_overridden = _base_config(output_dir=output_dir)
            raw_overridden["training"] = {"multi_sparsity_enabled": False}

            # when
            resolved_default = resolve_benchmark_config(
                LLMBenchmarkConfig.model_validate(raw_default)
            )
            resolved_overridden = resolve_benchmark_config(
                LLMBenchmarkConfig.model_validate(raw_overridden)
            )

            # then
            self.assertEqual(
                resolved_default["rng_fingerprint"],
                resolved_overridden["rng_fingerprint"],
            )

    def test_evaluation_epsilon_validation_rejects_invalid_values(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["evaluation"] = {"gamma_denominator_epsilon": "bad"}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

            raw = _base_config(output_dir=output_dir)
            raw["evaluation"] = {"r2_denominator_epsilon": 0.0}
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_delta_init_requires_single_activation_offset(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "sign_variant",
                    "sae_class": "SAE",
                    "sign_aware": True,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "single_activation_offset": False,
                    "delta_init": 0.01,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_delta_init_combined_with_delta_pos_or_delta_neg(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken_combo",
                    "sae_class": "SAE",
                    "sign_aware": True,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "single_activation_offset": True,
                    "delta_init": 0.01,
                    "delta_pos_init": 0.01,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_rejects_invalid_symmetric_magnitude_usage(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["variants"] = [
                {
                    "name": "broken",
                    "sae_class": "GatedSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "symmetric_magnitude": True,
                }
            ]

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_variant_threshold_optional_fields_validate_none_and_reject_inf(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            raw = _base_config(output_dir=output_dir)
            raw["variants"] = [
                {
                    "name": "none_ok",
                    "sae_class": "SignedSoftThresholdSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "theta_init": None,
                }
            ]
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertIsNone(config.variants[0].theta_init)

            # when / then
            raw = _base_config(output_dir=output_dir)
            raw["variants"] = [
                {
                    "name": "inf_bad",
                    "sae_class": "SignedSoftThresholdSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "sparsity_coeff": 1e-3,
                    "theta_init": float("inf"),
                }
            ]
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(raw)

    def test_text_field_consistency_accepts_explicit_equal_values(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["data"]["text_field"] = "content"
            raw["cache"] = {"text_field": "content"}

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.data.text_field, "content")
            self.assertEqual(config.cache.text_field, "content")

    def test_extended_training_and_evaluation_fields_validate(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _base_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw["training"] = {
                "train_steps": 10,
                "batch_size": 4,
                "learning_rate": 0.001,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "weight_decay": 0.0,
                "grad_clip_norm": 1.0,
                "log_every_n_steps": 2,
                "resample_latents_steps": 5,
                "mixed_precision": False,
                "mixed_precision_dtype": "bfloat16",
                "warmup_steps": 4,
                "threshold_warmup_steps": 3,
            }
            raw["evaluation"] = {
                "alive_threshold": 1e-6,
                "alive_min_fraction": 1e-4,
                "min_gamma_samples": 16,
                "gamma_denominator_epsilon": 1e-10,
                "r2_denominator_epsilon": 1e-10,
                "l0_is_nonzero": True,
            }

            # when
            config = LLMBenchmarkConfig.model_validate(raw)

            # then
            self.assertEqual(config.training.log_every_n_steps, 2)
            self.assertEqual(config.training.warmup_steps, 4)
            self.assertEqual(config.training.threshold_warmup_steps, 3)
            self.assertEqual(config.evaluation.min_gamma_samples, 16)
            self.assertTrue(config.evaluation.l0_is_nonzero)

    def _resolve_pair_with_training_override(
        self,
        output_dir: str,
        override_key: str,
        value_a: Any,
        value_b: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raw_a = _base_config(output_dir=output_dir)
        raw_a["training"] = {override_key: value_a}
        raw_b = _base_config(output_dir=output_dir)
        raw_b["training"] = {override_key: value_b}
        resolved_a = resolve_benchmark_config(LLMBenchmarkConfig.model_validate(raw_a))
        resolved_b = resolve_benchmark_config(LLMBenchmarkConfig.model_validate(raw_b))
        return resolved_a, resolved_b

    def test_train_steps_does_not_change_rng_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            resolved_a, resolved_b = self._resolve_pair_with_training_override(
                output_dir, "train_steps", 1000, 2000
            )

            # then
            self.assertEqual(
                resolved_a["rng_fingerprint"],
                resolved_b["rng_fingerprint"],
            )

    def test_train_steps_changes_config_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            resolved_a, resolved_b = self._resolve_pair_with_training_override(
                output_dir, "train_steps", 1000, 2000
            )

            # then
            self.assertNotEqual(
                resolved_a["config_fingerprint"],
                resolved_b["config_fingerprint"],
            )

    def test_log_every_n_steps_does_not_change_rng_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            resolved_a, resolved_b = self._resolve_pair_with_training_override(
                output_dir, "log_every_n_steps", 50, 100
            )

            # then
            self.assertEqual(
                resolved_a["rng_fingerprint"],
                resolved_b["rng_fingerprint"],
            )

    def test_checkpoint_model_steps_does_not_change_rng_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            resolved_a, resolved_b = self._resolve_pair_with_training_override(
                output_dir, "checkpoint_model_steps", 100, 200
            )

            # then
            self.assertEqual(
                resolved_a["rng_fingerprint"],
                resolved_b["rng_fingerprint"],
            )

    def test_dump_training_metrics_steps_does_not_change_rng_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            resolved_a, resolved_b = self._resolve_pair_with_training_override(
                output_dir, "dump_training_metrics_steps", 100, 200
            )

            # then
            self.assertEqual(
                resolved_a["rng_fingerprint"],
                resolved_b["rng_fingerprint"],
            )

    def test_learning_rate_changes_rng_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            resolved_a, resolved_b = self._resolve_pair_with_training_override(
                output_dir, "learning_rate", 1e-3, 1e-4
            )

            # then
            self.assertNotEqual(
                resolved_a["rng_fingerprint"],
                resolved_b["rng_fingerprint"],
            )

    def test_multi_sparsity_enabled_still_excluded_from_rng_fingerprint(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")

            # when
            resolved_a, resolved_b = self._resolve_pair_with_training_override(
                output_dir, "multi_sparsity_enabled", True, False
            )

            # then
            self.assertEqual(
                resolved_a["rng_fingerprint"],
                resolved_b["rng_fingerprint"],
            )

    def test_all_training_config_fields_classified_for_rng_fingerprint(
        self,
    ) -> None:
        # given
        _KNOWN_RNG_AFFECTING_TRAINING_KEYS = frozenset(
            {
                "batch_size",
                "max_train_samples",
                "max_val_samples",
                "max_test_samples",
                "learning_rate",
                "adam_beta1",
                "adam_beta2",
                "weight_decay",
                "grad_clip_norm",
                "resample_dead_latents",
                "resample_latents_steps",
                "mixed_precision",
                "mixed_precision_dtype",
                "warmup_steps",
                "threshold_warmup_steps",
            }
        )
        excluded = frozenset(_RNG_FINGERPRINT_EXCLUDED_TRAINING_KEYS)
        all_fields = frozenset(TrainingConfig.model_fields.keys())

        # when
        unclassified = all_fields - _KNOWN_RNG_AFFECTING_TRAINING_KEYS - excluded

        # then
        self.assertEqual(
            unclassified,
            frozenset(),
            f"New TrainingConfig fields must be added to either "
            f"_KNOWN_RNG_AFFECTING_TRAINING_KEYS (in this test) "
            f"or _RNG_FINGERPRINT_EXCLUDED_TRAINING_KEYS (in "
            f"config.py): {unclassified}",
        )
