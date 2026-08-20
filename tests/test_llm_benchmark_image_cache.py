import json
import os
import tempfile
import types
import unittest
from contextlib import contextmanager, nullcontext
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sgnsae.experiments.llm_benchmark.cache import (
    _build_cache_identity,
    _build_image_cache_identity,
    _compute_cache_fingerprint,
    _coerce_image_value_to_pil,
    _is_probable_hf_access_error,
    _iter_image_records,
    _reshape_image_hidden_state_to_token_sequence,
    _resolve_image_hidden_state_layer_index,
    _resolve_patch_size,
    _resolve_vision_model_torch_dtype,
    _run_image_cache_stage,
    run_cache_stage,
)
from sgnsae.experiments.llm_benchmark.config import (
    ActivationCacheConfig,
    HookpointConfig,
    ImageActivationCacheConfig,
    ImageDataConfig,
    LLMBenchmarkConfig,
    TextDataConfig,
)
from sgnsae.experiments.llm_benchmark.plotting import canonical_plot_paths
from sgnsae.experiments.llm_benchmark.runner import run_llm_sae_benchmark
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)


def _fixture_image_dir() -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "fixtures",
            "images_tiny",
        )
    )


def _base_image_config(
    output_dir: str,
    token_selection: str = "patch_only",
    data_backend: str = "imagefolder",
) -> Dict[str, Any]:
    data_section: Dict[str, Any]
    if data_backend == "imagefolder":
        data_section = {
            "backend": "imagefolder",
            "dataset_id": "local/images_tiny",
            "split": "train",
            "data_dir": _fixture_image_dir(),
            "streaming": False,
            "image_field": "image",
            "label_field": "label",
        }
    else:
        data_section = {
            "backend": "hf",
            "dataset_id": "ILSVRC/imagenet-1k",
            "split": "train",
            "streaming": True,
            "image_field": "image",
            "label_field": "label",
        }

    return {
        "output_dir": output_dir,
        "modality": "image",
        "device": "cpu",
        "seed": [0],
        "stages": ["cache"],
        "vision": {
            "model_id": "hf-internal-testing/tiny-random-vit",
            "image_size": 64,
            "torch_dtype": "float32",
            "use_auth_token": False,
        },
        "data": data_section,
        "hook": {"hook_name": "hidden_state", "layer": 1},
        "cache": {
            "image_batch_size": 2,
            "image_size": 64,
            "max_images": 4,
            "vectors_per_shard": 8,
            "cache_overwrite": False,
            "split_method": "stored_indices",
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "cache_seed": 0,
            "compute_dtype": "float32",
            "storage_dtype": "float32",
            "token_selection": token_selection,
            "representation": "hidden_state",
        },
        "variants": [
            {
                "name": "baseline_relu",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": [1e-3],
            }
        ],
        "loss_recovered": {"enabled": False},
    }


class DummyImageProcessor:
    def __call__(
        self, images: List[Image.Image], return_tensors: str
    ) -> Dict[str, torch.Tensor]:
        del return_tensors
        image_tensors: List[torch.Tensor] = []
        for image in images:
            np_image = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(np_image).permute(2, 0, 1)
            image_tensors.append(image_tensor)
        return {"pixel_values": torch.stack(image_tensors, dim=0)}


class DummyImageProcessorWithMetadata(DummyImageProcessor):
    def __call__(
        self, images: List[Image.Image], return_tensors: str
    ) -> Dict[str, Any]:
        payload = super().__call__(images=images, return_tensors=return_tensors)
        payload["metadata"] = "non_tensor_metadata"
        return payload


class DummyImageProcessorWrongSize(DummyImageProcessor):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = int(output_size)

    def __call__(
        self, images: List[Image.Image], return_tensors: str
    ) -> Dict[str, torch.Tensor]:
        del return_tensors
        image_tensors: List[torch.Tensor] = []
        for image in images:
            resized = image.convert("RGB").resize((self.output_size, self.output_size))
            np_image = np.asarray(resized, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(np_image).permute(2, 0, 1)
            image_tensors.append(image_tensor)
        return {"pixel_values": torch.stack(image_tensors, dim=0)}


class DummyVisionModel(torch.nn.Module):
    def __init__(
        self,
        d_in: int = 8,
        patch_size: int = 32,
        extra_tokens: int = 0,
        expected_pixel_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.extra_tokens = extra_tokens
        self.expected_pixel_dtype = expected_pixel_dtype
        self.config = types.SimpleNamespace(
            num_hidden_layers=1,
            patch_size=patch_size,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> Any:
        del output_hidden_states
        del return_dict

        if (
            self.expected_pixel_dtype is not None
            and pixel_values.dtype != self.expected_pixel_dtype
        ):
            raise ValueError(
                "Unexpected pixel_values dtype. Expected "
                f"{self.expected_pixel_dtype}, got {pixel_values.dtype}."
            )

        batch_size = int(pixel_values.shape[0])
        cls_tokens = torch.zeros(batch_size, 1, self.d_in, dtype=pixel_values.dtype)

        extra_prefix_tokens: List[torch.Tensor] = []
        for token_index in range(self.extra_tokens):
            token_value = float(token_index + 1)
            extra_prefix_tokens.append(
                torch.full(
                    (batch_size, self.d_in),
                    fill_value=token_value,
                    dtype=pixel_values.dtype,
                )
            )

        patch_tokens: List[torch.Tensor] = []
        for patch_index in range(4):
            patch_value = float(self.extra_tokens + patch_index + 1)
            patch_tokens.append(
                torch.full(
                    (batch_size, self.d_in),
                    fill_value=patch_value,
                    dtype=pixel_values.dtype,
                )
            )
        patch_tensor = torch.stack(patch_tokens, dim=1)

        if extra_prefix_tokens:
            extra_tensor = torch.stack(extra_prefix_tokens, dim=1)
            hidden_state = torch.cat((cls_tokens, extra_tensor, patch_tensor), dim=1)
        else:
            hidden_state = torch.cat((cls_tokens, patch_tensor), dim=1)
        return types.SimpleNamespace(
            hidden_states=(hidden_state * 0.5, hidden_state),
        )

    def to(self, device: Any) -> "DummyVisionModel":
        del device
        return self


class DummyVisionModelNoCls(DummyVisionModel):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        del output_hidden_states
        del return_dict
        del kwargs

        image_means = pixel_values.mean(dim=(1, 2, 3))
        patch_tokens: List[torch.Tensor] = []
        for patch_index in range(4):
            patch_value = image_means + float(patch_index + 1)
            patch_tokens.append(patch_value.unsqueeze(1).repeat(1, self.d_in))
        patch_tensor = torch.stack(patch_tokens, dim=1)
        return types.SimpleNamespace(hidden_states=(patch_tensor, patch_tensor))


class DummyVisionModelUnexpectedPatchCount(DummyVisionModel):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        del output_hidden_states
        del return_dict
        del kwargs

        image_means = pixel_values.mean(dim=(1, 2, 3))
        patch_tokens: List[torch.Tensor] = []
        for patch_index in range(3):
            patch_value = image_means + float(patch_index + 1)
            patch_tokens.append(patch_value.unsqueeze(1).repeat(1, self.d_in))
        patch_tensor = torch.stack(patch_tokens, dim=1)
        return types.SimpleNamespace(hidden_states=(patch_tensor, patch_tensor))


class DummyVisionModelMissingHiddenStates(DummyVisionModel):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        del pixel_values
        del output_hidden_states
        del return_dict
        del kwargs
        return types.SimpleNamespace()


class DummyVisionModelShortHiddenStates(DummyVisionModel):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        del output_hidden_states
        del return_dict
        del kwargs
        batch_size = int(pixel_values.shape[0])
        hidden_state = torch.ones(batch_size, 5, self.d_in)
        return types.SimpleNamespace(hidden_states=(hidden_state,))


class DummyVisionModelBadHiddenShape(DummyVisionModel):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        del output_hidden_states
        del return_dict
        del kwargs
        batch_size = int(pixel_values.shape[0])
        hidden_state = torch.ones(batch_size, self.d_in)
        return types.SimpleNamespace(hidden_states=(hidden_state, hidden_state))


class DummyVisionModelVaryingHiddenDim(DummyVisionModel):
    def __init__(self, patch_size: int = 32) -> None:
        super().__init__(d_in=8, patch_size=patch_size)
        self._call_count = 0

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        del output_hidden_states
        del return_dict
        del kwargs

        self._call_count += 1
        d_in = 8 if self._call_count == 1 else 7
        image_means = pixel_values.mean(dim=(1, 2, 3))
        cls_tokens = image_means.unsqueeze(1).unsqueeze(2).repeat(1, 1, d_in)
        patch_tokens: List[torch.Tensor] = []
        for patch_index in range(4):
            patch_value = image_means + float(patch_index + 1)
            patch_tokens.append(patch_value.unsqueeze(1).repeat(1, d_in))
        patch_tensor = torch.stack(patch_tokens, dim=1)
        hidden_state = torch.cat((cls_tokens, patch_tensor), dim=1)
        return types.SimpleNamespace(hidden_states=(hidden_state, hidden_state))


class DummyVisionModelVaryingTokenCount(DummyVisionModel):
    def __init__(self, patch_size: int = 32) -> None:
        super().__init__(d_in=8, patch_size=patch_size)
        self._call_count = 0

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        **kwargs: Any,
    ) -> Any:
        del output_hidden_states
        del return_dict
        del kwargs

        self._call_count += 1
        batch_size = int(pixel_values.shape[0])
        tokens = 5 if self._call_count == 1 else 6
        hidden_state = torch.zeros(
            batch_size, tokens, self.d_in, dtype=pixel_values.dtype
        )
        for token_index in range(tokens):
            hidden_state[:, token_index, :] = float(token_index)
        return types.SimpleNamespace(hidden_states=(hidden_state, hidden_state))


class DummyImageDatasetStream:
    def __init__(self, entries: List[Dict[str, Any]], fingerprint: str) -> None:
        self._entries = entries
        self._fingerprint = fingerprint

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._entries)


def _load_fixture_dataset_stream(data_dir: str) -> DummyImageDatasetStream:
    entries: List[Dict[str, Any]] = []
    image_paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(".png"):
                image_paths.append(os.path.join(root, filename))
    for image_index, image_path in enumerate(sorted(image_paths)):
        with Image.open(image_path) as decoded_image:
            entries.append(
                {
                    "image": decoded_image.convert("RGB"),
                    "label": image_index,
                }
            )
    return DummyImageDatasetStream(
        entries=entries,
        fingerprint="images-tiny-fixture",
    )


@contextmanager
def _patched_image_stack(
    model: torch.nn.Module,
    dataset_stream: Any,
    processor: Optional[Any] = None,
) -> Iterator[None]:
    if processor is None:
        processor = DummyImageProcessor()
    with patch(
        "transformers.AutoImageProcessor.from_pretrained",
        return_value=processor,
    ), patch(
        "transformers.AutoModel.from_pretrained",
        return_value=model,
    ), patch(
        "datasets.load_dataset",
        return_value=dataset_stream,
    ):
        yield


class TestLLMBenchmarkImageCache(unittest.TestCase):
    def test_image_cache_stage_writes_manifest_and_shards(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            raw_config = _base_image_config(output_dir=output_dir)
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyImageProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "datasets.load_dataset",
                return_value=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            self.assertFalse(result["reused"])
            with open(result["manifest_path"], "r") as file:
                manifest = json.load(file)

            self.assertEqual(manifest["modality"], "image")
            self.assertEqual(manifest["image_size"], 64)
            self.assertEqual(manifest["patch_size"], 32)
            self.assertEqual(manifest["token_selection"], "patch_only")
            self.assertEqual(manifest["num_images_processed"], 4)
            self.assertEqual(manifest["counts"]["num_vectors_total"], 16)
            self.assertEqual(manifest["activation"]["d_in"], 8)

            first_shard_filename = manifest["shards"][0]["filename"]
            first_shard_path = os.path.join(
                output_dir, "activations", first_shard_filename
            )
            first_shard = torch.load(first_shard_path, map_location="cpu")
            self.assertEqual(first_shard["acts"].shape[1], 8)
            self.assertGreater(int(first_shard["acts"].shape[0]), 0)
            self.assertGreater(int(torch.count_nonzero(first_shard["acts"]).item()), 0)

    def test_image_cache_stage_supports_all_token_selection_modes(self) -> None:
        # given
        expected_vectors = {
            "patch_only": 16,
            "cls_only": 4,
            "all": 20,
        }

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            for token_selection, expected_num_vectors in expected_vectors.items():
                with self.subTest(token_selection=token_selection):
                    output_dir = os.path.join(tmpdir, token_selection)
                    raw_config = _base_image_config(
                        output_dir=output_dir,
                        token_selection=token_selection,
                    )
                    config = LLMBenchmarkConfig.model_validate(raw_config)
                    with patch(
                        "transformers.AutoImageProcessor.from_pretrained",
                        return_value=DummyImageProcessor(),
                    ), patch(
                        "transformers.AutoModel.from_pretrained",
                        return_value=DummyVisionModel(),
                    ), patch(
                        "datasets.load_dataset",
                        return_value=_load_fixture_dataset_stream(_fixture_image_dir()),
                    ):
                        result = run_cache_stage(
                            config=config,
                            output_dir=output_dir,
                            runner_overwrite=False,
                        )
                    with open(result["manifest_path"], "r") as file:
                        manifest = json.load(file)
                    self.assertEqual(
                        manifest["counts"]["num_vectors_total"],
                        expected_num_vectors,
                    )

    def test_patch_only_uses_last_patch_tokens_when_extra_prefix_tokens_exist(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "patch_only_extra_tokens")
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir, token_selection="patch_only")
            )

            # when
            with _patched_image_stack(
                model=DummyVisionModel(extra_tokens=2),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            first_shard_filename = "shard_00000.pt"
            shard_path = os.path.join(output_dir, "activations", first_shard_filename)
            shard_payload = torch.load(shard_path, map_location="cpu")
            acts = shard_payload["acts"]
            unique_token_values = set(torch.unique(acts[:, 0]).tolist())
            self.assertEqual(unique_token_values, {3.0, 4.0, 5.0, 6.0})
            self.assertNotIn(1.0, unique_token_values)
            self.assertNotIn(2.0, unique_token_values)
            self.assertFalse(result["reused"])

    def test_cls_only_selects_cls_token_when_extra_prefix_tokens_exist(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "cls_only_extra_tokens")
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir, token_selection="cls_only")
            )

            # when
            with _patched_image_stack(
                model=DummyVisionModel(extra_tokens=2),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            first_shard_path = os.path.join(output_dir, "activations", "shard_00000.pt")
            shard_payload = torch.load(first_shard_path, map_location="cpu")
            acts = shard_payload["acts"]
            self.assertTrue(bool(torch.all(acts == 0.0)))
            self.assertEqual(int(acts.shape[0]), 4)
            self.assertFalse(result["reused"])

    def test_all_token_selection_includes_extra_prefix_tokens(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "all_extra_tokens")
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir, token_selection="all")
            )

            # when
            with _patched_image_stack(
                model=DummyVisionModel(extra_tokens=2),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )
            with open(result["manifest_path"], "r") as file:
                manifest = json.load(file)

            # then
            self.assertEqual(manifest["counts"]["num_vectors_total"], 28)

    def test_image_cache_casts_floating_inputs_to_model_dtype(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "dtype_cast")
            raw_config = _base_image_config(output_dir=output_dir)
            raw_config["vision"]["torch_dtype"] = "float16"
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with _patched_image_stack(
                model=DummyVisionModel(expected_pixel_dtype=torch.float16),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            self.assertFalse(result["reused"])

    def test_image_cache_stage_rejects_processor_output_size_mismatch(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "size_mismatch")
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir)
            )

            # when / then
            with _patched_image_stack(
                model=DummyVisionModelNoCls(),
                processor=DummyImageProcessorWrongSize(output_size=224),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError) as raised_error:
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

            self.assertIn(
                "Image processor output size mismatch",
                str(raised_error.exception),
            )
            self.assertIn(
                "config.cache.image_size=64",
                str(raised_error.exception),
            )

    def test_image_cache_stage_requires_rank_4_pixel_values(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "rank_4_required")
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir)
            )

            class DummyImageProcessorRank3:
                def __call__(
                    self, images: List[Image.Image], return_tensors: str
                ) -> Dict[str, torch.Tensor]:
                    del images
                    del return_tensors
                    return {"pixel_values": torch.zeros(3, 64, 64, dtype=torch.float32)}

            # when / then
            with _patched_image_stack(
                model=DummyVisionModelNoCls(),
                processor=DummyImageProcessorRank3(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError) as raised_error:
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

            self.assertIn(
                "Image processor must produce a 4D tensor `pixel_values`.",
                str(raised_error.exception),
            )

    def test_image_cache_stage_reports_auth_help_for_gated_model_and_dataset(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            model_output_dir = os.path.join(tmpdir, "model_auth_error")
            model_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=model_output_dir)
            )

            # when / then
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyImageProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                side_effect=OSError("403 Forbidden"),
            ):
                with self.assertRaises(ValueError) as raised_model_error:
                    _ = run_cache_stage(
                        config=model_config,
                        output_dir=model_output_dir,
                        runner_overwrite=False,
                    )

            model_error_message = str(raised_model_error.exception)
            self.assertIn("accept the terms", model_error_message.lower())
            self.assertIn("huggingface-cli login", model_error_message)

            dataset_output_dir = os.path.join(tmpdir, "dataset_auth_error")
            dataset_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(
                    output_dir=dataset_output_dir,
                    data_backend="hf",
                )
            )

            # when / then
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyImageProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "datasets.load_dataset",
                side_effect=RuntimeError("403 Forbidden access to dataset"),
            ):
                with self.assertRaises(ValueError) as raised_dataset_error:
                    _ = run_cache_stage(
                        config=dataset_config,
                        output_dir=dataset_output_dir,
                        runner_overwrite=False,
                    )

            dataset_error_message = str(raised_dataset_error.exception)
            self.assertIn("accept the terms", dataset_error_message.lower())
            self.assertIn("huggingface-cli login", dataset_error_message)

    def test_image_cache_helper_access_error_and_image_coercion_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "image.png")
            image = Image.new("RGB", (8, 8), color=(12, 34, 56))
            image.save(image_path, format="PNG")
            image_bytes_buffer = BytesIO()
            image.save(image_bytes_buffer, format="PNG")

            # when
            from_pil = _coerce_image_value_to_pil(image, "image")
            from_bytes = _coerce_image_value_to_pil(
                {"bytes": image_bytes_buffer.getvalue()},
                "image",
            )
            from_path = _coerce_image_value_to_pil({"path": image_path}, "image")

            # then
            self.assertEqual(from_pil.mode, "RGB")
            self.assertEqual(from_bytes.mode, "RGB")
            self.assertEqual(from_path.mode, "RGB")
            self.assertTrue(_is_probable_hf_access_error(RuntimeError("403 forbidden")))
            self.assertFalse(
                _is_probable_hf_access_error(RuntimeError("plain failure"))
            )
            self.assertFalse(_is_probable_hf_access_error(RuntimeError("   ")))

            # when / then
            with self.assertRaises(ValueError):
                _ = _coerce_image_value_to_pil({"bytes": "not-bytes"}, "image")
            with self.assertRaises(ValueError):
                _ = _coerce_image_value_to_pil(42, "image")

    def test_resolve_vision_model_torch_dtype_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = _base_image_config(output_dir=os.path.join(tmpdir, "run_root"))
            raw_config["vision"]["torch_dtype"] = "auto"
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when / then
            self.assertEqual(
                _resolve_vision_model_torch_dtype(config=config, runtime_device="cpu"),
                torch.float32,
            )
            with patch(
                "sgnsae.experiments.llm_benchmark.cache._is_cuda_runtime_device",
                return_value=True,
            ):
                self.assertEqual(
                    _resolve_vision_model_torch_dtype(
                        config=config, runtime_device="cpu"
                    ),
                    torch.float16,
                )

            explicit_dtype_config = config.model_copy(
                update={
                    "vision": config.vision.model_copy(
                        update={"torch_dtype": "bfloat16"}
                    )
                }
            )
            self.assertEqual(
                _resolve_vision_model_torch_dtype(
                    config=explicit_dtype_config,
                    runtime_device="cpu",
                ),
                torch.bfloat16,
            )

    def test_iter_image_records_covers_import_and_record_validation_branches(
        self,
    ) -> None:
        # given
        image_data_hf = ImageDataConfig.model_validate(
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
                "image_size": 64,
                "max_images": 4,
                "vectors_per_shard": 8,
                "split_method": "stored_indices",
                "train_fraction": 0.8,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
                "cache_seed": 0,
                "compute_dtype": "float32",
                "storage_dtype": "float32",
            }
        )
        original_import = __import__

        def _import_without_datasets(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "datasets":
                raise ImportError("missing datasets")
            return original_import(name, *args, **kwargs)

        # when / then
        with patch("builtins.__import__", side_effect=_import_without_datasets):
            with self.assertRaises(ValueError):
                _ = _iter_image_records(image_data_hf, image_cache)

        with patch("datasets.load_dataset", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                _ = _iter_image_records(image_data_hf, image_cache)

        with patch(
            "datasets.load_dataset",
            return_value=DummyImageDatasetStream(entries=[123], fingerprint="fp"),
        ):
            records, _ = _iter_image_records(image_data_hf, image_cache)
            with self.assertRaises(ValueError):
                _ = list(records)

        with patch(
            "datasets.load_dataset",
            return_value=DummyImageDatasetStream(
                entries=[{"other": 1}], fingerprint="fp"
            ),
        ):
            records, _ = _iter_image_records(image_data_hf, image_cache)
            with self.assertRaises(ValueError):
                _ = list(records)

    def test_hidden_state_layer_and_patch_size_resolution_cover_all_branches(
        self,
    ) -> None:
        # given
        model = DummyVisionModel()
        timm_wrapper_model = types.SimpleNamespace(
            config=types.SimpleNamespace(num_hidden_layers=None, patch_size=None),
            timm_model=types.SimpleNamespace(
                blocks=[object()] * 12,
                patch_embed=types.SimpleNamespace(patch_size=(14, 14)),
            ),
        )
        timm_wrapper_module_list_model = types.SimpleNamespace(
            config=types.SimpleNamespace(num_hidden_layers=None, patch_size=None),
            timm_model=types.SimpleNamespace(
                blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)]),
                patch_embed=types.SimpleNamespace(patch_size=(14, 14)),
            ),
        )
        empty_timm_blocks_model = types.SimpleNamespace(
            config=types.SimpleNamespace(num_hidden_layers=None, patch_size=None),
            timm_model=types.SimpleNamespace(
                blocks=[],
                patch_embed=types.SimpleNamespace(patch_size=(14, 14)),
            ),
        )
        empty_timm_module_list_model = types.SimpleNamespace(
            config=types.SimpleNamespace(num_hidden_layers=None, patch_size=None),
            timm_model=types.SimpleNamespace(
                blocks=torch.nn.ModuleList([]),
                patch_embed=types.SimpleNamespace(patch_size=(14, 14)),
            ),
        )
        mid_hook = HookpointConfig.model_validate(
            {"hook_name": "hidden_state", "layer_selector": "mid"}
        )
        late_hook = HookpointConfig.model_validate(
            {"hook_name": "hidden_state", "layer_selector": "late"}
        )
        float_hook = HookpointConfig.model_validate(
            {"hook_name": "hidden_state", "layer_selector": 0.0}
        )

        # when / then
        self.assertEqual(_resolve_image_hidden_state_layer_index(model, mid_hook), 1)
        self.assertEqual(_resolve_image_hidden_state_layer_index(model, late_hook), 0)
        self.assertEqual(_resolve_image_hidden_state_layer_index(model, float_hook), 0)
        self.assertEqual(
            _resolve_image_hidden_state_layer_index(timm_wrapper_model, mid_hook),
            6,
        )
        self.assertEqual(
            _resolve_image_hidden_state_layer_index(
                timm_wrapper_module_list_model, mid_hook
            ),
            6,
        )
        self.assertEqual(_resolve_patch_size(timm_wrapper_model), 14)
        self.assertEqual(
            _resolve_image_hidden_state_layer_index(
                empty_timm_blocks_model,
                HookpointConfig.model_validate(
                    {"hook_name": "hidden_state", "layer": 0}
                ),
            ),
            0,
        )
        self.assertEqual(
            _resolve_image_hidden_state_layer_index(
                empty_timm_module_list_model,
                HookpointConfig.model_validate(
                    {"hook_name": "hidden_state", "layer": 0}
                ),
            ),
            0,
        )

        invalid_hook_name = HookpointConfig.model_validate(
            {"hook_name": "mlp_out", "layer": 0}
        )
        with self.assertRaises(ValueError):
            _ = _resolve_image_hidden_state_layer_index(model, invalid_hook_name)

        bad_layer_model = types.SimpleNamespace(
            config=types.SimpleNamespace(num_hidden_layers=True)
        )
        with self.assertRaises(ValueError):
            _ = _resolve_image_hidden_state_layer_index(
                bad_layer_model,
                HookpointConfig.model_validate(
                    {"hook_name": "hidden_state", "layer_selector": "mid"}
                ),
            )
        self.assertEqual(
            _resolve_image_hidden_state_layer_index(
                bad_layer_model,
                HookpointConfig.model_validate(
                    {"hook_name": "hidden_state", "layer": 0}
                ),
            ),
            0,
        )

        unsupported_selector = HookpointConfig.model_construct(
            hook_name="hidden_state",
            layer=None,
            layer_selector=None,
            module_path=None,
        )
        with self.assertRaises(ValueError):
            _ = _resolve_image_hidden_state_layer_index(model, unsupported_selector)

        with self.assertRaises(ValueError):
            _ = _resolve_image_hidden_state_layer_index(
                model,
                HookpointConfig.model_validate(
                    {"hook_name": "hidden_state", "layer": 8}
                ),
            )

        self.assertEqual(
            _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=16))
            ),
            16,
        )
        self.assertEqual(
            _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=(8, 8)))
            ),
            8,
        )
        with self.assertRaises(ValueError):
            _ = _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=True))
            )
        with self.assertRaises(ValueError):
            _ = _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=0))
            )
        with self.assertRaises(ValueError):
            _ = _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=()))
            )
        with self.assertRaises(ValueError):
            _ = _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=("x", 8)))
            )
        with self.assertRaises(ValueError):
            _ = _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=(8, 4)))
            )
        with self.assertRaises(ValueError):
            _ = _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=(0, 0)))
            )
        with self.assertRaises(ValueError):
            _ = _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size=None))
            )
        with self.assertRaises(ValueError):
            _ = _resolve_patch_size(
                types.SimpleNamespace(config=types.SimpleNamespace(patch_size="x"))
            )

    def test_reshape_image_hidden_state_to_token_sequence_supports_tokens_and_maps(
        self,
    ) -> None:
        # given
        token_hidden_state = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
        feature_map_hidden_state = torch.arange(12, dtype=torch.float32).reshape(
            1, 3, 2, 2
        )
        expected_feature_map_tokens = torch.tensor(
            [
                [
                    [0.0, 4.0, 8.0],
                    [1.0, 5.0, 9.0],
                    [2.0, 6.0, 10.0],
                    [3.0, 7.0, 11.0],
                ]
            ],
            dtype=torch.float32,
        )

        # when
        reshaped_tokens = _reshape_image_hidden_state_to_token_sequence(
            hidden_state=token_hidden_state,
            expected_patch_tokens=4,
        )
        reshaped_feature_map = _reshape_image_hidden_state_to_token_sequence(
            hidden_state=feature_map_hidden_state,
            expected_patch_tokens=4,
        )

        # then
        self.assertTrue(bool(torch.equal(reshaped_tokens, token_hidden_state)))
        self.assertEqual(tuple(reshaped_feature_map.shape), (1, 4, 3))
        self.assertTrue(
            bool(torch.equal(reshaped_feature_map, expected_feature_map_tokens))
        )

        with self.assertRaises(ValueError):
            _ = _reshape_image_hidden_state_to_token_sequence(
                hidden_state=feature_map_hidden_state,
                expected_patch_tokens=5,
            )

    def test_build_image_cache_identity_supports_hf_and_imagefolder_branches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            imagefolder_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "imagefolder"))
            )
            hf_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(
                    output_dir=os.path.join(tmpdir, "hf"),
                    data_backend="hf",
                )
            )

            # when
            imagefolder_identity = _build_image_cache_identity(
                config=imagefolder_config,
                storage_dtype="float32",
                image_processor_id="processor-id",
                image_processor_revision="main",
                layer_index=1,
                runtime_device_kind="cpu",
                model_torch_dtype_resolved="float32",
            )
            hf_identity = _build_image_cache_identity(
                config=hf_config,
                storage_dtype="float32",
                image_processor_id="processor-id",
                image_processor_revision="main",
                layer_index=1,
                runtime_device_kind="cuda",
                model_torch_dtype_resolved="float16",
            )

            # then
            self.assertIn("data_dir", imagefolder_identity["dataset"])
            self.assertNotIn("data_dir", hf_identity["dataset"])
            self.assertEqual(
                imagefolder_identity["execution"]["runtime_device_kind"], "cpu"
            )
            self.assertEqual(
                hf_identity["execution"]["model_torch_dtype_resolved"], "float16"
            )

    def test_cache_fingerprint_changes_when_execution_identity_changes(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            image_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "image"))
            )
            text_config = LLMBenchmarkConfig.model_validate(
                {
                    "output_dir": os.path.join(tmpdir, "text"),
                    "device": "cpu",
                    "seed": [0],
                    "llm": {"model_id": "sshleifer/tiny-gpt2"},
                    "data": {
                        "backend": "hf",
                        "dataset_id": "example/dataset",
                        "split": "train",
                    },
                    "hook": {"hook_name": "mlp_out", "layer": 0},
                    "cache": {
                        "llm_batch_size": 1,
                        "max_seq_len": 8,
                        "vectors_per_shard": 16,
                        "train_fraction": 0.8,
                        "val_fraction": 0.1,
                        "test_fraction": 0.1,
                    },
                    "variants": [
                        {
                            "name": "baseline",
                            "sae_class": "SAE",
                            "sign_aware": False,
                            "d_hidden": 8,
                            "sparsity_coeff": 1e-3,
                        }
                    ],
                }
            )

            # when
            image_identity_cpu_float32 = _build_image_cache_identity(
                config=image_config,
                storage_dtype="float32",
                image_processor_id="processor-id",
                image_processor_revision="main",
                layer_index=1,
                runtime_device_kind="cpu",
                model_torch_dtype_resolved="float32",
            )
            image_identity_cuda_float16 = _build_image_cache_identity(
                config=image_config,
                storage_dtype="float32",
                image_processor_id="processor-id",
                image_processor_revision="main",
                layer_index=1,
                runtime_device_kind="cuda",
                model_torch_dtype_resolved="float16",
            )
            text_identity_cpu_float32 = _build_cache_identity(
                config=text_config,
                module_path_resolved="transformer.h.0.mlp",
                layer_index=0,
                storage_dtype="float32",
                runtime_device_kind="cpu",
                model_torch_dtype_resolved="float32",
            )
            text_identity_cuda_float16 = _build_cache_identity(
                config=text_config,
                module_path_resolved="transformer.h.0.mlp",
                layer_index=0,
                storage_dtype="float32",
                runtime_device_kind="cuda",
                model_torch_dtype_resolved="float16",
            )

            image_fingerprint_cpu_float32 = _compute_cache_fingerprint(
                image_identity_cpu_float32
            )
            image_fingerprint_cuda_float16 = _compute_cache_fingerprint(
                image_identity_cuda_float16
            )
            text_fingerprint_cpu_float32 = _compute_cache_fingerprint(
                text_identity_cpu_float32
            )
            text_fingerprint_cuda_float16 = _compute_cache_fingerprint(
                text_identity_cuda_float16
            )

            # then
            self.assertNotEqual(
                image_fingerprint_cpu_float32,
                image_fingerprint_cuda_float16,
            )
            self.assertNotEqual(
                text_fingerprint_cpu_float32,
                text_fingerprint_cuda_float16,
            )

    def test_run_image_cache_stage_validation_and_recache_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir)
            )

            # when / then
            invalid_vision_config = config.model_copy(update={"vision": None})
            with self.assertRaises(ValueError):
                _ = _run_image_cache_stage(invalid_vision_config, output_dir, False)

            invalid_data_config = config.model_copy(
                update={
                    "data": TextDataConfig.model_validate(
                        {
                            "backend": "local_jsonl",
                            "data_files": ["dummy.jsonl"],
                        }
                    )
                }
            )
            with self.assertRaises(ValueError):
                _ = _run_image_cache_stage(invalid_data_config, output_dir, False)

            invalid_cache_config = config.model_copy(
                update={"cache": ActivationCacheConfig.model_validate({})}
            )
            with self.assertRaises(ValueError):
                _ = _run_image_cache_stage(invalid_cache_config, output_dir, False)

            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                first = run_cache_stage(
                    config=config, output_dir=output_dir, runner_overwrite=False
                )
                reused = run_cache_stage(
                    config=config, output_dir=output_dir, runner_overwrite=False
                )

            self.assertFalse(first["reused"])
            self.assertTrue(reused["reused"])

            activations_dir = os.path.join(output_dir, "activations")
            manifest_path = os.path.join(activations_dir, "activations_manifest.json")
            with open(manifest_path, "r") as file:
                manifest = json.load(file)

            manifest["schema_version"] = int(ACTIVATIONS_MANIFEST_SCHEMA_VERSION) + 1
            with open(manifest_path, "w") as file:
                json.dump(manifest, file, indent=2)

            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

            manifest["schema_version"] = int(ACTIVATIONS_MANIFEST_SCHEMA_VERSION)
            manifest["cache_fingerprint"] = "mismatch"
            with open(manifest_path, "w") as file:
                json.dump(manifest, file, indent=2)

            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

            manifest["cache_fingerprint"] = first["cache_fingerprint"]
            manifest["shards"][0]["filename"] = "missing.pt"
            with open(manifest_path, "w") as file:
                json.dump(manifest, file, indent=2)

            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

            overwrite_config = config.model_copy(
                update={
                    "cache": config.cache.model_copy(update={"cache_overwrite": True}),
                }
            )
            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                regenerated = run_cache_stage(
                    config=overwrite_config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )
            self.assertFalse(regenerated["reused"])

            orphan_output_dir = os.path.join(tmpdir, "orphan_cache")
            orphan_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=orphan_output_dir)
            )
            orphan_activations_dir = os.path.join(orphan_output_dir, "activations")
            os.makedirs(orphan_activations_dir, exist_ok=True)
            with open(os.path.join(orphan_activations_dir, "orphan.txt"), "w") as file:
                file.write("orphan")
            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=orphan_config,
                        output_dir=orphan_output_dir,
                        runner_overwrite=False,
                    )
                rebuilt = run_cache_stage(
                    config=orphan_config,
                    output_dir=orphan_output_dir,
                    runner_overwrite=True,
                )
            self.assertFalse(rebuilt["reused"])

            empty_dir_output = os.path.join(tmpdir, "empty_activations_dir")
            empty_dir_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=empty_dir_output)
            )
            os.makedirs(
                os.path.join(empty_dir_output, "activations"),
                exist_ok=True,
            )
            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                empty_dir_result = run_cache_stage(
                    config=empty_dir_config,
                    output_dir=empty_dir_output,
                    runner_overwrite=False,
                )
            self.assertFalse(empty_dir_result["reused"])

    def test_run_image_cache_stage_runtime_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            processor_error_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "processor_error"))
            )

            # when / then
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                side_effect=OSError("401 Unauthorized"),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=processor_error_config,
                        output_dir=str(processor_error_config.output_dir),
                        runner_overwrite=False,
                    )

            processor_non_auth_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(
                    output_dir=os.path.join(tmpdir, "processor_non_auth")
                )
            )
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                side_effect=RuntimeError("processor boom"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_cache_stage(
                        config=processor_non_auth_config,
                        output_dir=str(processor_non_auth_config.output_dir),
                        runner_overwrite=False,
                    )

            model_error_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "model_error"))
            )
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyImageProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                side_effect=RuntimeError("model boom"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_cache_stage(
                        config=model_error_config,
                        output_dir=str(model_error_config.output_dir),
                        runner_overwrite=False,
                    )

            non_divisible_raw = _base_image_config(
                output_dir=os.path.join(tmpdir, "non_divisible")
            )
            non_divisible_raw["vision"]["image_size"] = 65
            non_divisible_raw["cache"]["image_size"] = 65
            non_divisible_config = LLMBenchmarkConfig.model_validate(non_divisible_raw)
            with _patched_image_stack(
                model=DummyVisionModel(patch_size=32),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=non_divisible_config,
                        output_dir=str(non_divisible_config.output_dir),
                        runner_overwrite=False,
                    )

            missing_hidden_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "missing_hidden"))
            )
            with _patched_image_stack(
                model=DummyVisionModelMissingHiddenStates(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=missing_hidden_config,
                        output_dir=str(missing_hidden_config.output_dir),
                        runner_overwrite=False,
                    )

            short_hidden_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "short_hidden"))
            )
            with _patched_image_stack(
                model=DummyVisionModelShortHiddenStates(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=short_hidden_config,
                        output_dir=str(short_hidden_config.output_dir),
                        runner_overwrite=False,
                    )

            bad_shape_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "bad_shape"))
            )
            with _patched_image_stack(
                model=DummyVisionModelBadHiddenShape(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=bad_shape_config,
                        output_dir=str(bad_shape_config.output_dir),
                        runner_overwrite=False,
                    )

            varying_dim_raw = _base_image_config(
                output_dir=os.path.join(tmpdir, "varying_dim")
            )
            varying_dim_raw["cache"]["image_batch_size"] = 1
            varying_dim_config = LLMBenchmarkConfig.model_validate(varying_dim_raw)
            with _patched_image_stack(
                model=DummyVisionModelVaryingHiddenDim(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=varying_dim_config,
                        output_dir=str(varying_dim_config.output_dir),
                        runner_overwrite=False,
                    )

            patch_only_error_raw = _base_image_config(
                output_dir=os.path.join(tmpdir, "patch_only_error")
            )
            patch_only_error_config = LLMBenchmarkConfig.model_validate(
                patch_only_error_raw
            )
            with _patched_image_stack(
                model=DummyVisionModelUnexpectedPatchCount(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=patch_only_error_config,
                        output_dir=str(patch_only_error_config.output_dir),
                        runner_overwrite=False,
                    )

            cls_error_raw = _base_image_config(
                output_dir=os.path.join(tmpdir, "cls_error"),
                token_selection="cls_only",
            )
            cls_error_config = LLMBenchmarkConfig.model_validate(cls_error_raw)
            with _patched_image_stack(
                model=DummyVisionModelNoCls(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=cls_error_config,
                        output_dir=str(cls_error_config.output_dir),
                        runner_overwrite=False,
                    )

            invalid_token_config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=os.path.join(tmpdir, "invalid_token"))
            )
            invalid_token_config = invalid_token_config.model_copy(
                update={
                    "cache": invalid_token_config.cache.model_copy(
                        update={"token_selection": "unsupported"}
                    )
                }
            )
            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=invalid_token_config,
                        output_dir=str(invalid_token_config.output_dir),
                        runner_overwrite=False,
                    )

            metadata_raw = _base_image_config(
                output_dir=os.path.join(tmpdir, "metadata_inputs")
            )
            metadata_config = LLMBenchmarkConfig.model_validate(metadata_raw)
            with _patched_image_stack(
                model=DummyVisionModelNoCls(),
                processor=DummyImageProcessorWithMetadata(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                metadata_result = run_cache_stage(
                    config=metadata_config,
                    output_dir=str(metadata_config.output_dir),
                    runner_overwrite=False,
                )
            self.assertFalse(metadata_result["reused"])

            max_images_raw = _base_image_config(
                output_dir=os.path.join(tmpdir, "max_images_break")
            )
            max_images_raw["cache"]["max_images"] = 1
            max_images_config = LLMBenchmarkConfig.model_validate(max_images_raw)
            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                max_images_result = run_cache_stage(
                    config=max_images_config,
                    output_dir=str(max_images_config.output_dir),
                    runner_overwrite=False,
                )
            with open(max_images_result["manifest_path"], "r") as file:
                max_images_manifest = json.load(file)
            self.assertEqual(max_images_manifest["num_images_processed"], 1)

            empty_raw = _base_image_config(output_dir=os.path.join(tmpdir, "empty"))
            empty_config = LLMBenchmarkConfig.model_validate(empty_raw)
            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=DummyImageDatasetStream(entries=[], fingerprint="empty"),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=empty_config,
                        output_dir=str(empty_config.output_dir),
                        runner_overwrite=False,
                    )

            shard_ranges_raw = _base_image_config(
                output_dir=os.path.join(tmpdir, "shard_ranges")
            )
            shard_ranges_raw["cache"]["split_method"] = "shard_ranges"
            shard_ranges_config = LLMBenchmarkConfig.model_validate(shard_ranges_raw)
            with _patched_image_stack(
                model=DummyVisionModelNoCls(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                shard_ranges_result = run_cache_stage(
                    config=shard_ranges_config,
                    output_dir=str(shard_ranges_config.output_dir),
                    runner_overwrite=False,
                )
            with open(shard_ranges_result["manifest_path"], "r") as file:
                shard_ranges_manifest = json.load(file)
            self.assertEqual(shard_ranges_manifest["split"]["method"], "shard_ranges")

            autocast_raw = _base_image_config(
                output_dir=os.path.join(tmpdir, "autocast_branch")
            )
            autocast_raw["cache"]["compute_dtype"] = "float16"
            autocast_config = LLMBenchmarkConfig.model_validate(autocast_raw)
            with patch(
                "sgnsae.experiments.llm_benchmark.cache._is_cuda_runtime_device",
                return_value=True,
            ), patch(
                "sgnsae.experiments.llm_benchmark.cache.torch.autocast",
                return_value=nullcontext(),
            ), _patched_image_stack(
                model=DummyVisionModelNoCls(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                autocast_result = run_cache_stage(
                    config=autocast_config,
                    output_dir=str(autocast_config.output_dir),
                    runner_overwrite=False,
                )
            self.assertFalse(autocast_result["reused"])

    def test_image_cache_stage_rejects_variable_token_counts_across_batches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_config = _base_image_config(
                output_dir=os.path.join(tmpdir, "varying_token_count")
            )
            raw_config["cache"]["image_batch_size"] = 1
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when / then
            with _patched_image_stack(
                model=DummyVisionModelVaryingTokenCount(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                with self.assertRaises(ValueError) as raised_error:
                    _ = run_cache_stage(
                        config=config,
                        output_dir=str(config.output_dir),
                        runner_overwrite=False,
                    )
            self.assertIn(
                "Inconsistent token count across image batches",
                str(raised_error.exception),
            )

    def test_image_cache_stage_empty_dataset_does_not_write_manifest(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "empty_dataset")
            config = LLMBenchmarkConfig.model_validate(
                _base_image_config(output_dir=output_dir)
            )
            manifest_path = os.path.join(
                output_dir, "activations", "activations_manifest.json"
            )

            # when / then
            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=DummyImageDatasetStream(entries=[], fingerprint="empty"),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )
            self.assertFalse(os.path.isfile(manifest_path))

    def test_image_cache_stored_indices_split_is_assigned_per_vector(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "stored_indices_split")
            raw_config = _base_image_config(output_dir=output_dir)
            raw_config["cache"]["max_images"] = 2
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with _patched_image_stack(
                model=DummyVisionModelNoCls(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )
            with open(result["manifest_path"], "r") as file:
                manifest = json.load(file)
            split_indices_path = os.path.join(
                output_dir,
                "activations",
                str(manifest["split"]["indices_filename"]),
            )
            split_indices = torch.load(split_indices_path, map_location="cpu")

            # then
            vectors_from_indices = (
                int(split_indices["train"].numel())
                + int(split_indices["val"].numel())
                + int(split_indices["test"].numel())
            )
            self.assertEqual(
                vectors_from_indices,
                int(manifest["counts"]["num_vectors_total"]),
            )
            self.assertNotEqual(
                vectors_from_indices,
                int(manifest["num_images_processed"]),
            )

    def test_image_smoke_pipeline_runs_cache_train_eval_and_plot_stages(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "image_smoke")
            raw_config = _base_image_config(output_dir=output_dir)
            raw_config["stages"] = ["cache", "train", "eval", "plot"]
            raw_config["training"] = {
                "train_steps": 1,
                "batch_size": 8,
                "log_every_n_steps": 1,
                "mixed_precision": False,
                "resample_dead_latents": False,
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as file:
                json.dump(raw_config, file, indent=2)

            # when
            with _patched_image_stack(
                model=DummyVisionModel(),
                dataset_stream=_load_fixture_dataset_stream(_fixture_image_dir()),
            ):
                written_dir = run_llm_sae_benchmark(config_path=config_path)

            # then
            self.assertEqual(written_dir, output_dir)
            subrun_dir = os.path.join(
                output_dir, "models", "baseline_relu", "seed_0", "lambda_1e-03"
            )
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model.pt")))
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "training_metrics.json"))
            )
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "evaluation.json")))
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "gamma_metrics.json"))
            )

            with open(os.path.join(subrun_dir, "evaluation.json"), "r") as file:
                evaluation = json.load(file)
            self.assertIn("mse", evaluation)
            self.assertIn("mean_l0", evaluation)
            self.assertIn("gamma_summary", evaluation)

            plot_paths = canonical_plot_paths(output_dir=output_dir)
            for plot_path in plot_paths.values():
                self.assertTrue(os.path.isfile(plot_path))
