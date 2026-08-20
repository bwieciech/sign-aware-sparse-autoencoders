import base64
import io
import json
import os
import tempfile
import types
import unittest
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from unittest.mock import patch

import torch
from PIL import Image

import sgnsae.experiments.llm_benchmark.examples as examples_module
import sgnsae.experiments.llm_benchmark.latent_examples as latent_examples_module
from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)
from tests.llm_benchmark_test_helpers import legacy_success_training_metrics_payload


def _text_config(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": [0, 1],
        "llm": {"model_id": "sshleifer/tiny-gpt2"},
        "data": {
            "backend": "local_jsonl",
            "data_files": ["dummy.jsonl"],
            "text_field": "text",
        },
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "variants": [
            {
                "name": "variant_a",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 4,
                "sparsity_coeff": 1e-3,
            }
        ],
    }


def _image_config(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "modality": "image",
        "seed": [0],
        "vision": {
            "model_id": "hf-internal-testing/tiny-random-vit",
            "image_size": 64,
            "torch_dtype": "float32",
            "use_auth_token": False,
        },
        "data": {
            "backend": "hf",
            "dataset_id": "ILSVRC/imagenet-1k",
            "split": "train",
            "image_field": "image",
            "streaming": True,
        },
        "hook": {"hook_name": "hidden_state", "layer": 1},
        "cache": {
            "image_batch_size": 2,
            "image_size": 64,
            "max_images": 8,
            "vectors_per_shard": 32,
            "split_method": "stored_indices",
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "token_selection": "patch_only",
        },
        "variants": [
            {
                "name": "variant_a",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 4,
                "sparsity_coeff": 1e-3,
            }
        ],
        "loss_recovered": {"enabled": False},
    }


def _write_training_metrics(
    subrun_dir: str, payload: Optional[Union[Dict[str, Any], str]] = None
) -> None:
    if payload is None:
        payload = {"status": "completed"}
    with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
        if isinstance(payload, str):
            file.write(payload)
        else:
            json.dump(payload, file, indent=2)


class DummyHookHandle:
    def __init__(self, remove_callback: Any) -> None:
        self._remove_callback = remove_callback

    def remove(self) -> None:
        self._remove_callback()


class DummyHookModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._hook: Any = None

    def register_forward_hook(self, hook: Any) -> DummyHookHandle:
        self._hook = hook
        return DummyHookHandle(remove_callback=lambda: setattr(self, "_hook", None))

    def emit(self, output: torch.Tensor) -> None:
        if self._hook is not None:
            self._hook(self, None, output)


class DummyTextModel(torch.nn.Module):
    def __init__(self, hook_module: DummyHookModule) -> None:
        super().__init__()
        self.hook_module = hook_module
        self.config = types.SimpleNamespace(use_cache=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> Tuple[torch.Tensor]:
        del attention_mask
        del use_cache
        del return_dict
        activations = input_ids.unsqueeze(-1).repeat(1, 1, 2).to(dtype=torch.float32)
        self.hook_module.emit(activations)
        return (activations,)

    def to(self, device: Any) -> "DummyTextModel":
        del device
        return self


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> List[int]:
        del add_special_tokens
        length = max(1, len(text.split()))
        return [index + 1 for index in range(length)]

    def decode(
        self, token_ids: Sequence[int], clean_up_tokenization_spaces: bool
    ) -> str:
        del clean_up_tokenization_spaces
        return f" t{int(token_ids[0])}"


class DummyVisionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=1, patch_size=32)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> Any:
        del output_hidden_states
        del return_dict
        batch = int(pixel_values.shape[0])
        hidden = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ).repeat(batch, 1, 1)
        return types.SimpleNamespace(hidden_states=(hidden, hidden))

    def to(self, device: Any) -> "DummyVisionModel":
        del device
        return self


class DummyVisionModelExpectPixelDtype(torch.nn.Module):
    def __init__(self, expected_pixel_dtype: torch.dtype) -> None:
        super().__init__()
        self.expected_pixel_dtype = expected_pixel_dtype
        self.dtype = expected_pixel_dtype
        self.config = types.SimpleNamespace(num_hidden_layers=1, patch_size=32)
        self.register_parameter(
            "dummy_weight",
            torch.nn.Parameter(torch.zeros(1, dtype=expected_pixel_dtype)),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> Any:
        del output_hidden_states
        del return_dict
        if pixel_values.dtype != self.expected_pixel_dtype:
            raise ValueError(
                "Unexpected pixel dtype for latent examples. "
                f"Expected {self.expected_pixel_dtype}, got {pixel_values.dtype}."
            )
        hidden = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                ]
            ],
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        return types.SimpleNamespace(hidden_states=(hidden, hidden))


class DummyProcessor:
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]

    def __call__(
        self, images: Sequence[Image.Image], return_tensors: str
    ) -> Dict[str, Any]:
        del return_tensors
        tensors: List[torch.Tensor] = []
        for image in images:
            resized = image.resize((64, 64))
            array = torch.from_numpy(
                torch.ByteTensor(torch.ByteStorage.from_buffer(resized.tobytes()))
                .reshape(64, 64, 3)
                .numpy()
            ).to(dtype=torch.float32)
            tensor = array.permute(2, 0, 1) / 255.0
            tensors.append(tensor)
        return {"pixel_values": torch.stack(tensors, dim=0)}


class DummyProcessorWrongSize:
    def __init__(self, output_size: int) -> None:
        self.output_size = int(output_size)

    def __call__(
        self, images: Sequence[Image.Image], return_tensors: str
    ) -> Dict[str, Any]:
        del return_tensors
        tensors: List[torch.Tensor] = []
        for image in images:
            resized = image.resize((self.output_size, self.output_size))
            array = torch.from_numpy(
                torch.ByteTensor(torch.ByteStorage.from_buffer(resized.tobytes()))
                .reshape(self.output_size, self.output_size, 3)
                .numpy()
            ).to(dtype=torch.float32)
            tensor = array.permute(2, 0, 1) / 255.0
            tensors.append(tensor)
        return {"pixel_values": torch.stack(tensors, dim=0)}


class DummySAE(torch.nn.Module):
    def __init__(self, d_hidden: int = 4) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.d_hidden = int(d_hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_2d = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
        length = int(x_2d.shape[0])
        latents = torch.zeros((1, length, self.d_hidden), dtype=torch.float32)
        latents[0, :, 0] = x_2d[:, 0]
        latents[0, :, 1] = torch.arange(length, dtype=torch.float32)
        latents[0, :, 2] = -x_2d[:, 0]
        latents[0, :, 3] = x_2d[:, 0] * 0.5
        return {"latents": latents}


class DummySAEWithNaNLatent(torch.nn.Module):
    def __init__(self, d_hidden: int = 4) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.d_hidden = int(d_hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_2d = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
        length = int(x_2d.shape[0])
        latents = torch.zeros((1, length, self.d_hidden), dtype=torch.float32)
        latents[0, :, 0] = x_2d[:, 0]
        latents[0, :, 1] = float("nan")
        latents[0, :, 2] = torch.arange(length, dtype=torch.float32) + 1.0
        latents[0, :, 3] = x_2d[:, 0] * 0.25
        return {"latents": latents}


class DummySAEWrongHidden(torch.nn.Module):
    def __init__(self, d_hidden: int = 4, wrong_hidden: int = 3) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.d_hidden = int(d_hidden)
        self._wrong_hidden = int(wrong_hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_2d = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
        length = int(x_2d.shape[0])
        latents = torch.zeros((1, length, self._wrong_hidden), dtype=torch.float32)
        return {"latents": latents}


class DummyTokenizerScripted:
    def __init__(self, scripted_outputs: Sequence[Any]) -> None:
        self._scripted_outputs = list(scripted_outputs)
        self._index = 0

    def encode(self, text: str, add_special_tokens: bool) -> Any:
        del text
        del add_special_tokens
        if self._index >= len(self._scripted_outputs):
            return [1]
        output = self._scripted_outputs[self._index]
        self._index += 1
        return output

    def decode(
        self, token_ids: Sequence[int], clean_up_tokenization_spaces: bool
    ) -> str:
        del clean_up_tokenization_spaces
        return f" t{int(token_ids[0])}"


class DummyTextModelScripted(torch.nn.Module):
    def __init__(
        self, hook_module: DummyHookModule, scripted_outputs: Sequence[Any]
    ) -> None:
        super().__init__()
        self.hook_module = hook_module
        self.config = types.SimpleNamespace(use_cache=True)
        self._scripted_outputs = list(scripted_outputs)
        self._index = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> Tuple[torch.Tensor]:
        del attention_mask
        del use_cache
        del return_dict
        default_output = input_ids.unsqueeze(-1).repeat(1, 1, 2).to(dtype=torch.float32)
        output_to_emit: Any
        if self._index < len(self._scripted_outputs):
            output_to_emit = self._scripted_outputs[self._index]
            self._index += 1
        else:
            output_to_emit = default_output
        if output_to_emit is not None:
            if torch.is_tensor(output_to_emit):
                self.hook_module.emit(output_to_emit)
            elif isinstance(output_to_emit, list):
                for tensor in output_to_emit:
                    self.hook_module.emit(tensor)
        return (default_output,)

    def to(self, device: Any) -> "DummyTextModelScripted":
        del device
        return self


class DummyVisionModelScripted(torch.nn.Module):
    def __init__(
        self, scripted_hidden_states: Sequence[Any], patch_size: int = 32
    ) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=1, patch_size=patch_size)
        self._scripted_hidden_states = list(scripted_hidden_states)
        self._index = 0

    def forward(
        self,
        pixel_values: Any = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        del pixel_values
        del output_hidden_states
        del return_dict
        del kwargs
        if self._index >= len(self._scripted_hidden_states):
            hidden_states = self._scripted_hidden_states[-1]
        else:
            hidden_states = self._scripted_hidden_states[self._index]
            self._index += 1
        if hidden_states is None:
            return types.SimpleNamespace()
        return types.SimpleNamespace(hidden_states=hidden_states)

    def to(self, device: Any) -> "DummyVisionModelScripted":
        del device
        return self


class DummyProcessorScripted:
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]

    def __init__(
        self, include_pixel_values: bool = True, include_metadata: bool = False
    ) -> None:
        self.include_pixel_values = bool(include_pixel_values)
        self.include_metadata = bool(include_metadata)

    def __call__(
        self, images: Sequence[Image.Image], return_tensors: str
    ) -> Dict[str, Any]:
        del return_tensors
        payload: Dict[str, Any] = {}
        if self.include_pixel_values:
            tensors: List[torch.Tensor] = []
            for image in images:
                resized = image.resize((64, 64))
                array = torch.tensor(
                    list(resized.getdata()), dtype=torch.float32
                ).reshape(64, 64, 3)
                tensors.append(array.permute(2, 0, 1) / 255.0)
            payload["pixel_values"] = torch.stack(tensors, dim=0)
        if self.include_metadata:
            payload["metadata"] = "note"
        return payload


class TestLLMBenchmarkLatentExamples(unittest.TestCase):
    def test_select_target_subruns_explicit_filters_to_first_seed(self) -> None:
        # given
        base_config = LLMBenchmarkConfig.model_validate(
            _text_config(output_dir="/tmp/out")
        )
        config = base_config.model_copy(
            update={
                "latent_examples": base_config.latent_examples.model_copy(
                    update={"enabled": True, "target_subruns": "explicit"}
                )
            }
        )
        resolved_config = {
            "seeds": [0, 1],
            "subruns": [
                {
                    "variant_name": "variant_a",
                    "variant_slug": "variant_a",
                    "seed": 1,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-4,
                    "relative_subrun_dir": "models/variant_a/seed_1/lambda_1e-04",
                },
                {
                    "variant_name": "variant_a",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
                },
            ],
        }

        # when
        selected = latent_examples_module.select_target_subruns_for_examples(
            config=config,
            resolved_config=resolved_config,
            output_dir="/tmp/out",
        )

        # then
        self.assertEqual(len(selected), 1)
        self.assertEqual(int(selected[0]["seed"]), 0)

    def test_select_target_subruns_uses_matched_anchor_entries(self) -> None:
        # given
        base_config = LLMBenchmarkConfig.model_validate(
            _text_config(output_dir="/tmp/out")
        )
        config = base_config.model_copy(
            update={
                "latent_examples": base_config.latent_examples.model_copy(
                    update={
                        "enabled": True,
                        "target_subruns": "matched_l0_anchor",
                        "anchor_l0": 64.0,
                        "seeds": [0],
                    }
                )
            }
        )
        resolved_config = {
            "seeds": [0],
            "subruns": [
                {
                    "variant_name": "variant_a",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-4,
                    "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-04",
                },
                {
                    "variant_name": "variant_a",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
                },
            ],
        }
        summary = {
            "matched_l0_anchors": {
                "targets": [16, 64],
                "selected": [
                    {
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "target": 64,
                        "selected_sweep_param": "lambda",
                        "selected_sweep_value": 1e-3,
                        "selected_mean_l0": 63.5,
                        "selected_mse": 0.3,
                        "relative_dir": "models/variant_a/seed_0/lambda_1e-03",
                    }
                ],
            },
            "subruns": [],
        }

        # when
        with patch(
            "sgnsae.experiments.llm_benchmark.latent_examples._load_or_build_summary",
            return_value=summary,
        ):
            selected = latent_examples_module.select_target_subruns_for_examples(
                config=config,
                resolved_config=resolved_config,
                output_dir="/tmp/out",
            )

        # then
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["relative_subrun_dir"],
            "models/variant_a/seed_0/lambda_1e-03",
        )

    def test_select_target_subruns_best_mse_and_missing_metrics_error(self) -> None:
        # given
        base_config = LLMBenchmarkConfig.model_validate(
            _text_config(output_dir="/tmp/out")
        )
        best_mse_config = base_config.model_copy(
            update={
                "latent_examples": base_config.latent_examples.model_copy(
                    update={"enabled": True, "target_subruns": "best_mse", "seeds": [0]}
                )
            }
        )
        resolved_config = {
            "seeds": [0],
            "subruns": [
                {
                    "variant_name": "variant_a",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-4,
                    "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-04",
                },
                {
                    "variant_name": "variant_a",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
                },
            ],
        }
        summary = {
            "subruns": [
                {
                    "relative_dir": "models/variant_a/seed_0/lambda_1e-04",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "mse": 0.5,
                    "mean_l0": 10.0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-4,
                },
                {
                    "relative_dir": "models/variant_a/seed_0/lambda_1e-03",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "mse": 0.1,
                    "mean_l0": 12.0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                },
            ]
        }

        # when
        with patch(
            "sgnsae.experiments.llm_benchmark.latent_examples._load_or_build_summary",
            return_value=summary,
        ):
            selected = latent_examples_module.select_target_subruns_for_examples(
                config=best_mse_config,
                resolved_config=resolved_config,
                output_dir="/tmp/out",
            )

        # then
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["relative_subrun_dir"],
            "models/variant_a/seed_0/lambda_1e-03",
        )

        # when / then
        with patch(
            "sgnsae.experiments.llm_benchmark.latent_examples._load_or_build_summary",
            return_value={"subruns": []},
        ):
            with self.assertRaises(ValueError):
                _ = latent_examples_module.select_target_subruns_for_examples(
                    config=best_mse_config.model_copy(
                        update={
                            "latent_examples": best_mse_config.latent_examples.model_copy(
                                update={"target_subruns": "matched_l0_anchor"}
                            )
                        }
                    ),
                    resolved_config=resolved_config,
                    output_dir="/tmp/out",
                )

    def test_build_and_persist_latent_examples_index_filters_invalid_entries(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                {
                    "variant_slug": "variant_b",
                    "seed": 1,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "latent_examples_path": "models/variant_b/seed_1/lambda_1e-03/latent_examples.json",
                },
                {"variant_slug": "bad", "seed": "x"},
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-4,
                    "latent_examples_path": "models/variant_a/seed_0/lambda_1e-04/latent_examples.json",
                },
            ]

            # when
            result = latent_examples_module.build_and_persist_latent_examples_index(
                output_dir=tmpdir,
                subrun_entries=entries,
            )

            # then
            self.assertTrue(os.path.isfile(result["index_path"]))
            with open(result["index_path"], "r") as file:
                payload = json.load(file)
            self.assertEqual(len(payload["subruns"]), 2)
            self.assertEqual(payload["subruns"][0]["variant_slug"], "variant_a")

    def test_generate_latent_examples_for_subrun_reuses_existing_file(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=output_dir)
            )
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            _write_training_metrics(subrun_dir)
            with open(os.path.join(subrun_dir, "latent_examples.json"), "w") as file:
                json.dump({"existing": True}, file)

            # when
            result = latent_examples_module.generate_latent_examples_for_subrun(
                config=config,
                resolved_config={"subruns": [subrun]},
                subrun=subrun,
                output_dir=output_dir,
                overwrite=False,
            )

            # then
            self.assertTrue(bool(result["reused"]))

    def test_generate_latent_examples_rejects_incomplete_training_before_reuse(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=output_dir)
            )
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "latent_examples.json"), "w") as file:
                json.dump({"existing": True}, file)

            for training_metrics_payload in (
                {},
                {"status": "failed"},
                {"status": "running"},
                {"status": "completed", "terminated_early": True},
                {"status": "completed", "failure": {"type": "exception"}},
                "[",
            ):
                _write_training_metrics(subrun_dir, payload=training_metrics_payload)

                # when / then
                with self.subTest(training_metrics_payload=training_metrics_payload):
                    with patch(
                        "sgnsae.experiments.llm_benchmark.examples.validate_examples_cache_metadata"
                    ) as validate_examples:
                        with self.assertRaises(ValueError):
                            _ = examples_module.generate_latent_examples(
                                config=config,
                                resolved_config={"subruns": [subrun]},
                                subrun=subrun,
                                output_dir=output_dir,
                                overwrite=False,
                            )
                    self.assertEqual(validate_examples.call_count, 0)

    def test_generate_latent_examples_for_subrun_rejects_incomplete_training(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=output_dir)
            )
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "latent_examples.json"), "w") as file:
                json.dump({"existing": True}, file)

            for training_metrics_payload in (
                {},
                {"status": "failed"},
                {"status": "running"},
                {"status": "completed", "terminated_early": True},
                {"status": "completed", "failure": {"type": "exception"}},
                "[",
            ):
                _write_training_metrics(subrun_dir, payload=training_metrics_payload)

                # when / then
                with self.subTest(training_metrics_payload=training_metrics_payload):
                    with patch(
                        "sgnsae.experiments.llm_benchmark.latent_examples._load_sae_model_for_subrun"
                    ) as load_model:
                        with self.assertRaises(ValueError):
                            _ = latent_examples_module.generate_latent_examples_for_subrun(
                                config=config,
                                resolved_config={"subruns": [subrun]},
                                subrun=subrun,
                                output_dir=output_dir,
                                overwrite=False,
                            )
                    self.assertEqual(load_model.call_count, 0)

    def test_generate_latent_examples_rejects_legacy_success_metrics_without_status(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=output_dir)
            )
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            _write_training_metrics(
                subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )
            with open(os.path.join(subrun_dir, "latent_examples.json"), "w") as file:
                json.dump({"existing": True}, file)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.examples.validate_examples_cache_metadata"
            ) as validate_examples:
                with self.assertRaises(ValueError) as error:
                    _ = examples_module.generate_latent_examples(
                        config=config,
                        resolved_config={"subruns": [subrun]},
                        subrun=subrun,
                        output_dir=output_dir,
                        overwrite=False,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            self.assertEqual(validate_examples.call_count, 0)

    def test_generate_latent_examples_for_subrun_rejects_legacy_success_metrics_without_status(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=output_dir)
            )
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            _write_training_metrics(
                subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )
            with open(os.path.join(subrun_dir, "latent_examples.json"), "w") as file:
                json.dump({"existing": True}, file)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._load_sae_model_for_subrun"
            ) as load_model:
                with self.assertRaises(ValueError) as error:
                    _ = latent_examples_module.generate_latent_examples_for_subrun(
                        config=config,
                        resolved_config={"subruns": [subrun]},
                        subrun=subrun,
                        output_dir=output_dir,
                        overwrite=False,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            self.assertEqual(load_model.call_count, 0)

    def test_generate_latent_examples_for_subrun_writes_text_and_image_payloads(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            text_output = os.path.join(tmpdir, "text_run")
            text_config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=text_output)
            )
            text_subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            image_output = os.path.join(tmpdir, "image_run")
            image_config = LLMBenchmarkConfig.model_validate(
                _image_config(output_dir=image_output)
            )
            image_subrun = dict(text_subrun)
            text_subrun_dir = os.path.join(
                text_output, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(text_subrun_dir, exist_ok=True)
            _write_training_metrics(text_subrun_dir)
            image_subrun_dir = os.path.join(
                image_output, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(image_subrun_dir, exist_ok=True)
            _write_training_metrics(image_subrun_dir)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._load_sae_model_for_subrun",
                return_value=(
                    DummySAE(d_hidden=4),
                    {"patch_size": 32, "image_size": 64},
                    "cpu",
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._generate_text_examples",
                return_value=({"0": ["|> token <|"]}, [0], 2),
            ):
                text_result = (
                    latent_examples_module.generate_latent_examples_for_subrun(
                        config=text_config,
                        resolved_config={"subruns": [text_subrun]},
                        subrun=text_subrun,
                        output_dir=text_output,
                        overwrite=True,
                    )
                )

            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._load_sae_model_for_subrun",
                return_value=(
                    DummySAE(d_hidden=4),
                    {
                        "patch_size": 32,
                        "image_size": 64,
                        "token_selection": "patch_only",
                    },
                    "cpu",
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._generate_image_examples",
                return_value=({"1": [["a", "b"]]}, [1], 3),
            ):
                image_result = (
                    latent_examples_module.generate_latent_examples_for_subrun(
                        config=image_config,
                        resolved_config={"subruns": [image_subrun]},
                        subrun=image_subrun,
                        output_dir=image_output,
                        overwrite=True,
                    )
                )

            # then
            self.assertFalse(bool(text_result["reused"]))
            self.assertFalse(bool(image_result["reused"]))
            with open(text_result["latent_examples_path"], "r") as file:
                text_payload = json.load(file)
            self.assertEqual(text_payload["modality"], "text")
            self.assertIn("selection", text_payload)
            with open(image_result["latent_examples_path"], "r") as file:
                image_payload = json.load(file)
            self.assertEqual(image_payload["modality"], "image")
            self.assertIn("examples", image_payload)

    def test_generate_latent_examples_for_subrun_rejects_non_patch_token_selection(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "image_run")
            image_config = LLMBenchmarkConfig.model_validate(
                _image_config(output_dir=output_dir)
            )
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            _write_training_metrics(subrun_dir)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._load_sae_model_for_subrun",
                return_value=(
                    DummySAE(d_hidden=4),
                    {"patch_size": 32, "image_size": 64, "token_selection": "cls_only"},
                    "cpu",
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._generate_image_examples",
            ) as generate_image_examples:
                with self.assertRaises(ValueError) as raised_error:
                    _ = latent_examples_module.generate_latent_examples_for_subrun(
                        config=image_config,
                        resolved_config={"subruns": [subrun]},
                        subrun=subrun,
                        output_dir=output_dir,
                        overwrite=True,
                    )

            self.assertEqual(generate_image_examples.call_count, 0)
            self.assertIn("token_selection='patch_only'", str(raised_error.exception))

    def test_generate_text_examples_and_empty_sample_path(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={
                            "enabled": True,
                            "sample_size": 2,
                            "max_latents": 2,
                            "max_examples_per_latent": 2,
                            "activation_threshold": 0.0,
                        }
                    )
                }
            )
            hook_module = DummyHookModule()
            text_model = DummyTextModel(hook_module=hook_module)
            tokenizer = DummyTokenizer()
            sae_model = DummySAE(d_hidden=4)

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=text_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["one two", "three four", "five six"], {}),
            ):
                examples, selected_latents, scanned = (
                    latent_examples_module._generate_text_examples(
                        config=config,
                        sae_model=sae_model,
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=text_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=([], {}),
            ):
                empty_examples, empty_latents, empty_scanned = (
                    latent_examples_module._generate_text_examples(
                        config=config,
                        sae_model=sae_model,
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            # then
            self.assertGreaterEqual(scanned, 1)
            self.assertTrue(bool(selected_latents))
            self.assertTrue(bool(examples))
            self.assertEqual(empty_examples, {})
            self.assertEqual(empty_latents, [])
            self.assertEqual(empty_scanned, 0)

    def test_generate_text_examples_includes_negative_latent_activations(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={
                            "enabled": True,
                            "sample_size": 1,
                            "max_latents": 1,
                            "max_examples_per_latent": 2,
                            "activation_threshold": 1.5,
                            "latent_selection": "explicit",
                            "explicit_latents": [2],
                        }
                    )
                }
            )
            hook_module = DummyHookModule()
            text_model = DummyTextModel(hook_module=hook_module)
            sae_model = DummySAE(d_hidden=4)

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=text_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["one two three"], {}),
            ):
                examples, selected_latents, scanned = (
                    latent_examples_module._generate_text_examples(
                        config=config,
                        sae_model=sae_model,
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            # then
            self.assertEqual(scanned, 1)
            self.assertEqual(selected_latents, [2])
            self.assertIn("2", examples)
            self.assertTrue(bool(examples["2"]))
            highlighted = examples["2"][0]
            self.assertIn("|>", highlighted)
            self.assertIn("<|", highlighted)
            inside_highlight = highlighted.split("|>", 1)[1].split("<|", 1)[0]
            self.assertIn("t2", inside_highlight)
            self.assertIn("t3", inside_highlight)

    def test_generate_text_examples_skips_nan_latents_in_top_by_max_abs(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={
                            "enabled": True,
                            "sample_size": 1,
                            "max_latents": 2,
                            "max_examples_per_latent": 2,
                            "activation_threshold": 0.0,
                            "latent_selection": "top_by_max_abs",
                        }
                    )
                }
            )
            hook_module = DummyHookModule()
            text_model = DummyTextModel(hook_module=hook_module)
            sae_model = DummySAEWithNaNLatent(d_hidden=4)

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=text_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["one two three"], {}),
            ):
                examples, selected_latents, scanned = (
                    latent_examples_module._generate_text_examples(
                        config=config,
                        sae_model=sae_model,
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            # then
            self.assertEqual(scanned, 1)
            self.assertNotIn(1, selected_latents)
            self.assertTrue(bool(selected_latents))
            self.assertTrue(all(str(index) in examples for index in selected_latents))

    def test_generate_image_examples_and_empty_sample_path(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _image_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={
                            "enabled": True,
                            "sample_size": 2,
                            "max_latents": 2,
                            "max_examples_per_latent": 2,
                            "activation_threshold": 0.0,
                        }
                    )
                }
            )
            manifest = {"patch_size": 32, "image_size": 64}
            sae_model = DummySAE(d_hidden=4)
            image_entries = [
                {"image": Image.new("RGB", (64, 64), color=(20, 30, 40))},
                {"image": Image.new("RGB", (64, 64), color=(60, 70, 80))},
            ]

            # when
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                examples, selected_latents, scanned = (
                    latent_examples_module._generate_image_examples(
                        config=config,
                        manifest=manifest,
                        sae_model=sae_model,
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter([]), {}),
            ):
                empty_examples, empty_latents, empty_scanned = (
                    latent_examples_module._generate_image_examples(
                        config=config,
                        manifest=manifest,
                        sae_model=sae_model,
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            # then
            self.assertGreaterEqual(scanned, 1)
            self.assertTrue(bool(selected_latents))
            self.assertTrue(bool(examples))
            first_payload = next(iter(examples.values()))[0]
            decoded_full = base64.b64decode(first_payload[0].encode("ascii"))
            with Image.open(io.BytesIO(decoded_full)) as decoded_image:
                self.assertEqual(decoded_image.size, (64, 64))
            self.assertEqual(empty_examples, {})
            self.assertEqual(empty_latents, [])
            self.assertEqual(empty_scanned, 0)

    def test_generate_image_examples_rejects_processor_output_size_mismatch(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _image_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"enabled": True, "sample_size": 1}
                    )
                }
            )
            image_entries = [
                {"image": Image.new("RGB", (64, 64), color=(20, 30, 40))},
            ]

            # when / then
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorWrongSize(output_size=224),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                with self.assertRaises(ValueError) as raised_error:
                    _ = latent_examples_module._generate_image_examples(
                        config=config,
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(d_hidden=4),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            self.assertIn(
                "Image processor output size mismatch while generating latent examples",
                str(raised_error.exception),
            )
            self.assertIn("pixel_values is 224x224", str(raised_error.exception))
            self.assertIn("manifest image_size=64", str(raised_error.exception))

    def test_generate_image_examples_payload_sizes_match_patch_grid(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _image_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={
                            "enabled": True,
                            "sample_size": 1,
                            "max_latents": 1,
                            "max_examples_per_latent": 1,
                            "activation_threshold": 0.0,
                        }
                    )
                }
            )
            image_entries = [
                {"image": Image.new("RGB", (80, 72), color=(20, 30, 40))},
            ]

            # when
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                examples, selected_latents, scanned = (
                    latent_examples_module._generate_image_examples(
                        config=config,
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(d_hidden=4),
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            # then
            self.assertEqual(scanned, 1)
            self.assertTrue(bool(selected_latents))
            self.assertTrue(bool(examples))
            first_payload = next(iter(examples.values()))[0]
            full_image_bytes = base64.b64decode(first_payload[0].encode("ascii"))
            patch_image_bytes = base64.b64decode(first_payload[1].encode("ascii"))
            with Image.open(io.BytesIO(full_image_bytes)) as full_image:
                self.assertEqual(full_image.size, (64, 64))
            with Image.open(io.BytesIO(patch_image_bytes)) as patch_image:
                self.assertEqual(patch_image.size, (32, 32))

    def test_generate_image_examples_casts_dtype_and_applies_imageops_fit(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _image_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "vision": base_config.vision.model_copy(
                        update={"torch_dtype": "float16"}
                    ),
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"enabled": True, "sample_size": 1}
                    ),
                }
            )
            manifest = {"patch_size": 32, "image_size": 64}
            sae_model = DummySAE(d_hidden=4)
            image_entries = [
                {"image": Image.new("RGB", (80, 72), color=(20, 30, 40))},
            ]
            fit_calls: List[Tuple[Tuple[int, int], Any]] = []
            original_fit = latent_examples_module.ImageOps.fit

            def _record_fit(
                image: Image.Image, size: Tuple[int, int], method: Any
            ) -> Any:
                fit_calls.append((tuple(size), method))
                return original_fit(image, size, method=method)

            # when
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModelExpectPixelDtype(torch.float16),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.ImageOps.fit",
                side_effect=_record_fit,
            ):
                examples, selected_latents, scanned = (
                    latent_examples_module._generate_image_examples(
                        config=config,
                        manifest=manifest,
                        sae_model=sae_model,
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            # then
            self.assertEqual(scanned, 1)
            self.assertTrue(bool(examples))
            self.assertTrue(bool(selected_latents))
            self.assertEqual(len(fit_calls), 2)
            for size, method in fit_calls:
                self.assertEqual(size, (64, 64))
                self.assertEqual(method, Image.BICUBIC)

    def test_generate_image_examples_infers_dtype_from_first_parameter(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _image_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "vision": base_config.vision.model_copy(
                        update={"torch_dtype": "float16"}
                    ),
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"enabled": True, "sample_size": 1}
                    ),
                }
            )
            manifest = {"patch_size": 32, "image_size": 64}
            sae_model = DummySAE(d_hidden=4)
            image_entries = [
                {"image": Image.new("RGB", (64, 64), color=(10, 20, 30))},
            ]
            model = DummyVisionModelExpectPixelDtype(torch.float16)
            delattr(model, "dtype")

            # when
            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessor(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                examples, selected_latents, scanned = (
                    latent_examples_module._generate_image_examples(
                        config=config,
                        manifest=manifest,
                        sae_model=sae_model,
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )

            # then
            self.assertEqual(scanned, 1)
            self.assertTrue(bool(examples))
            self.assertTrue(bool(selected_latents))

    def test_load_sae_model_for_subrun_validates_checkpoint_presence(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            _write_training_metrics(subrun_dir)
            config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=output_dir)
            )
            os.makedirs(
                os.path.join(
                    output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
                ),
                exist_ok=True,
            )

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_manifest_path",
                return_value=os.path.join(
                    output_dir, "activations", "activations_manifest.json"
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.load_activation_manifest",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.build_model_for_subrun",
                return_value=(DummySAE(d_hidden=4), 0.0, 1, {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._load_sae_model_for_subrun(
                        config=config,
                        resolved_config={"subruns": [subrun], "rng_fingerprint": "rng"},
                        subrun=subrun,
                        output_dir=output_dir,
                    )

            model_path = os.path.join(
                output_dir,
                "models",
                "variant_a",
                "seed_0",
                "lambda_1e-03",
                "model.pt",
            )
            dummy_model = DummySAE(d_hidden=4)
            torch.save({"state_dict": dummy_model.state_dict()}, model_path)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_manifest_path",
                return_value=os.path.join(
                    output_dir, "activations", "activations_manifest.json"
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.load_activation_manifest",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.build_model_for_subrun",
                return_value=(DummySAE(d_hidden=4), 0.0, 1, {}),
            ):
                loaded_model, _, _ = latent_examples_module._load_sae_model_for_subrun(
                    config=config,
                    resolved_config={"subruns": [subrun], "rng_fingerprint": "rng"},
                    subrun=subrun,
                    output_dir=output_dir,
                )

            # then
            self.assertIsInstance(loaded_model, DummySAE)

    def test_pixel_and_patch_helpers_cover_error_paths(self) -> None:
        # given
        pixel_values = torch.ones(3, 64, 64, dtype=torch.float32)
        processor = types.SimpleNamespace(
            image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0]
        )

        # when
        image = latent_examples_module._pixel_values_to_pil_rgb_image(
            pixel_values=pixel_values,
            processor=processor,
        )
        bounds = latent_examples_module.compute_patch_crop_bounds(
            patch_index=3,
            n_patches_w=4,
            patch_size=16,
            image_size=64,
        )
        encoded = latent_examples_module.encode_png_base64(image)
        decoded = base64.b64decode(encoded.encode("ascii"))

        # then
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(bounds, (48, 0, 64, 16))
        self.assertTrue(len(decoded) > 0)

        # when / then
        with self.assertRaises(ValueError):
            _ = latent_examples_module.compute_patch_crop_bounds(
                patch_index=99,
                n_patches_w=4,
                patch_size=16,
                image_size=64,
            )
        with self.assertRaises(ValueError):
            _ = latent_examples_module._pixel_values_to_pil_rgb_image(
                pixel_values=torch.ones(2, 64),
                processor=processor,
            )


class TestLLMBenchmarkLatentExamplesCoverage(unittest.TestCase):
    def test_helper_io_and_path_validation_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = os.path.join(tmpdir, "payload.json")
            payload = {"key": "value"}
            latent_examples_module._write_json(payload_path, payload)

            # when
            loaded = latent_examples_module._read_json(payload_path)

            # then
            self.assertEqual(loaded, payload)
            self.assertIsNone(latent_examples_module._safe_float(True))
            self.assertIsNone(latent_examples_module._safe_float(float("nan")))
            self.assertEqual(latent_examples_module._safe_float(3), 3.0)
            self.assertIsNone(latent_examples_module._safe_int(True))
            self.assertEqual(latent_examples_module._safe_int(7), 7)

            # when / then
            with self.assertRaises(ValueError):
                _ = latent_examples_module._output_subdir_path(
                    output_dir=tmpdir, relative_dir="/"
                )

            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.os.path.commonpath",
                side_effect=ValueError("bad commonpath"),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._output_subdir_path(
                        output_dir=tmpdir, relative_dir="models/a"
                    )

            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.os.path.commonpath",
                return_value=os.path.join(tmpdir, "other"),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._output_subdir_path(
                        output_dir=tmpdir, relative_dir="models/a"
                    )

    def test_seed_filter_and_summary_builder_branches(self) -> None:
        # given
        config = LLMBenchmarkConfig.model_validate(_text_config(output_dir="/tmp/out"))

        # when
        seed_filter_from_list = latent_examples_module._resolve_seed_filter(
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"seeds": [3, 1]}
                    )
                }
            ),
            resolved_config={"seeds": [9]},
        )
        seed_filter_from_empty = latent_examples_module._resolve_seed_filter(
            config=config,
            resolved_config={"seeds": ["x"]},
        )

        # then
        self.assertEqual(seed_filter_from_list, [1, 3])
        self.assertEqual(seed_filter_from_empty, [])

        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "comparison_summary.json")
            with open(summary_path, "w") as file:
                json.dump([], file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.build_and_persist_comparison_summary",
                return_value={"built": True},
            ) as build_summary, patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.ensure_summary_has_matched_l0_anchors",
                side_effect=lambda output_dir, summary: summary,
            ):
                summary = latent_examples_module._load_or_build_summary(
                    resolved_config={},
                    output_dir=tmpdir,
                )

            # then
            self.assertTrue(bool(summary["built"]))
            self.assertEqual(build_summary.call_count, 1)

            with open(summary_path, "w") as file:
                json.dump({"subruns": [{"seed": 0}]}, file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.build_and_persist_comparison_summary",
                return_value={"built": True},
            ) as build_summary_again, patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.ensure_summary_has_matched_l0_anchors",
                side_effect=lambda output_dir, summary: summary,
            ):
                existing_summary = latent_examples_module._load_or_build_summary(
                    resolved_config={},
                    output_dir=tmpdir,
                )

            # then
            self.assertTrue(bool(existing_summary["built"]))
            self.assertEqual(build_summary_again.call_count, 1)

    def test_summary_parsing_and_selection_helpers_cover_edge_cases(self) -> None:
        # given / when / then
        self.assertEqual(
            latent_examples_module._summary_entries_with_metrics(
                summary={"subruns": "bad"},
                seed_filter=[0],
            ),
            [],
        )
        parsed = latent_examples_module._summary_entries_with_metrics(
            summary={
                "subruns": [
                    [],
                    {
                        "relative_dir": "a",
                        "variant_slug": "v",
                        "seed": 1,
                        "mse": 0.5,
                        "mean_l0": 1.0,
                    },
                    {
                        "relative_dir": "",
                        "variant_slug": "v",
                        "seed": 0,
                        "mse": 0.1,
                        "mean_l0": 1.0,
                    },
                ]
            },
            seed_filter=[1],
        )
        self.assertEqual(len(parsed), 1)
        filtered_out = latent_examples_module._summary_entries_with_metrics(
            summary={
                "subruns": [
                    {
                        "relative_dir": "a",
                        "variant_slug": "v",
                        "seed": 1,
                        "mse": 0.5,
                        "mean_l0": 1.0,
                    }
                ]
            },
            seed_filter=[0],
        )
        self.assertEqual(filtered_out, [])

        self.assertEqual(
            latent_examples_module._matched_anchor_entries_from_summary(
                summary={"matched_l0_anchors": {"targets": "bad", "selected": []}},
                anchor_l0=64.0,
                seed_filter=[],
            ),
            [],
        )
        self.assertEqual(
            latent_examples_module._matched_anchor_entries_from_summary(
                summary={"matched_l0_anchors": {"targets": [16], "selected": []}},
                anchor_l0=64.1,
                seed_filter=[],
            ),
            [],
        )
        self.assertEqual(
            latent_examples_module._matched_anchor_entries_from_summary(
                summary={
                    "matched_l0_anchors": {
                        "targets": ["x"],
                        "selected": [],
                    }
                },
                anchor_l0=64.0,
                seed_filter=[],
            ),
            [],
        )
        self.assertEqual(
            latent_examples_module._matched_anchor_entries_from_summary(
                summary={
                    "matched_l0_anchors": {
                        "targets": [16],
                        "selected": [],
                    }
                },
                anchor_l0=64.0,
                seed_filter=[],
            ),
            [],
        )
        entries = latent_examples_module._matched_anchor_entries_from_summary(
            summary={
                "matched_l0_anchors": {
                    "targets": [64],
                    "selected": [
                        [],
                        {
                            "target": 16,
                            "relative_dir": "a",
                            "variant_slug": "v",
                            "seed": 0,
                            "selected_mse": 1.0,
                            "selected_mean_l0": 2.0,
                        },
                        {
                            "target": 64,
                            "relative_dir": "",
                            "variant_slug": "v",
                            "seed": 0,
                            "selected_mse": 1.0,
                            "selected_mean_l0": 2.0,
                        },
                    ],
                }
            },
            anchor_l0=64.0,
            seed_filter=[],
        )
        self.assertEqual(entries, [])
        filtered = latent_examples_module._matched_anchor_entries_from_summary(
            summary={
                "matched_l0_anchors": {
                    "targets": [64],
                    "selected": [
                        {
                            "target": 64,
                            "relative_dir": "a",
                            "variant_slug": "v",
                            "seed": 1,
                            "selected_mse": 1.0,
                            "selected_mean_l0": 2.0,
                        }
                    ],
                }
            },
            anchor_l0=64.0,
            seed_filter=[0],
        )
        self.assertEqual(filtered, [])

        matched_selected = latent_examples_module._select_matched_anchor_entries(
            entries=[
                {
                    "variant_slug": "v",
                    "seed": 0,
                    "mean_l0": 63.0,
                    "mse": 0.2,
                    "sweep_value": 1e-3,
                    "relative_dir": "b",
                },
                {
                    "variant_slug": "v",
                    "seed": 0,
                    "mean_l0": 65.0,
                    "mse": 0.1,
                    "sweep_value": "bad",
                    "relative_dir": "a",
                },
            ],
            anchor_l0=64.0,
        )
        self.assertEqual(len(matched_selected), 1)
        best_selected = latent_examples_module._select_best_mse_entries(
            entries=[
                {
                    "variant_slug": "v",
                    "seed": 0,
                    "mse": 0.5,
                    "sweep_value": 2e-3,
                    "relative_dir": "b",
                },
                {
                    "variant_slug": "v",
                    "seed": 0,
                    "mse": 0.5,
                    "sweep_value": "bad",
                    "relative_dir": "a",
                },
            ]
        )
        self.assertEqual(len(best_selected), 1)

    def test_select_target_subruns_for_examples_excludes_incomplete_subruns(
        self,
    ) -> None:
        # given
        for target_mode, stale_status in (
            ("matched_l0_anchor", "failed"),
            ("best_mse", "running"),
        ):
            with self.subTest(target_mode=target_mode, stale_status=stale_status):
                with tempfile.TemporaryDirectory() as tmpdir:
                    activations_dir = os.path.join(tmpdir, "activations")
                    os.makedirs(activations_dir, exist_ok=True)
                    with open(
                        os.path.join(activations_dir, "activations_manifest.json"),
                        "w",
                    ) as file:
                        json.dump(
                            {
                                "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                                "hook": {"hook_name": "mlp_out", "layer": 0},
                                "dataset": {
                                    "dataset_id": "fixture",
                                    "split": "train",
                                },
                                "tokenization": {"max_seq_len": 8},
                            },
                            file,
                            indent=2,
                        )
                    base_config = LLMBenchmarkConfig.model_validate(
                        _text_config(output_dir=tmpdir)
                    )
                    config = base_config.model_copy(
                        update={
                            "latent_examples": base_config.latent_examples.model_copy(
                                update={
                                    "target_subruns": target_mode,
                                    "anchor_l0": 16.0,
                                }
                            )
                        }
                    )
                    completed_subrun = {
                        "variant_name": "variant_a",
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
                    }
                    stale_subrun = {
                        "variant_name": "variant_a",
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-4,
                        "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-04",
                    }
                    completed_dir = os.path.join(
                        tmpdir, completed_subrun["relative_subrun_dir"]
                    )
                    stale_dir = os.path.join(
                        tmpdir, stale_subrun["relative_subrun_dir"]
                    )
                    os.makedirs(completed_dir, exist_ok=True)
                    os.makedirs(stale_dir, exist_ok=True)
                    _write_training_metrics(completed_dir, {"status": "completed"})
                    _write_training_metrics(stale_dir, {"status": stale_status})
                    with open(
                        os.path.join(completed_dir, "evaluation.json"), "w"
                    ) as file:
                        json.dump(
                            {
                                "mse": 1.0,
                                "r2": 0.5,
                                "mean_l0": 64.0,
                                "dead_fraction": 0.1,
                            },
                            file,
                            indent=2,
                        )
                    with open(os.path.join(stale_dir, "evaluation.json"), "w") as file:
                        json.dump(
                            {
                                "mse": 0.1,
                                "r2": 0.9,
                                "mean_l0": 16.0,
                                "dead_fraction": 0.01,
                            },
                            file,
                            indent=2,
                        )

                    # when
                    selected = (
                        latent_examples_module.select_target_subruns_for_examples(
                            config=config,
                            resolved_config={
                                "llm": {"model_id": "dummy-model"},
                                "hook": {"hook_name": "mlp_out", "layer": 0},
                                "subruns": [completed_subrun, stale_subrun],
                            },
                            output_dir=tmpdir,
                        )
                    )

                    # then
                    self.assertEqual(selected, [completed_subrun])

    def test_select_target_subruns_additional_branches(self) -> None:
        # given
        config = LLMBenchmarkConfig.model_validate(_text_config(output_dir="/tmp/out"))

        # when / then
        selected_empty = latent_examples_module.select_target_subruns_for_examples(
            config=config,
            resolved_config={"subruns": "bad"},
            output_dir="/tmp/out",
        )
        self.assertEqual(selected_empty, [])

        # given
        matched_config = config.model_copy(
            update={
                "latent_examples": config.latent_examples.model_copy(
                    update={
                        "enabled": True,
                        "target_subruns": "matched_l0_anchor",
                        "anchor_l0": 64.0,
                        "seeds": [0],
                    }
                )
            }
        )
        resolved = {
            "seeds": [0],
            "subruns": [
                {
                    "variant_name": "variant_a",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
                }
            ],
        }
        summary = {
            "matched_l0_anchors": {"targets": [16], "selected": []},
            "subruns": [
                {
                    "relative_dir": "models/variant_a/seed_0/lambda_1e-03",
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "mse": 0.2,
                    "mean_l0": 10.0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                }
            ],
        }
        with patch(
            "sgnsae.experiments.llm_benchmark.latent_examples._load_or_build_summary",
            return_value=summary,
        ):
            selected = latent_examples_module.select_target_subruns_for_examples(
                config=matched_config,
                resolved_config=resolved,
                output_dir="/tmp/out",
            )
        self.assertEqual(len(selected), 1)

        with patch(
            "sgnsae.experiments.llm_benchmark.latent_examples._load_or_build_summary",
            return_value={"subruns": []},
        ):
            with self.assertRaises(ValueError):
                _ = latent_examples_module.select_target_subruns_for_examples(
                    config=config.model_copy(
                        update={
                            "latent_examples": config.latent_examples.model_copy(
                                update={"enabled": True, "target_subruns": "best_mse"}
                            )
                        }
                    ),
                    resolved_config=resolved,
                    output_dir="/tmp/out",
                )

        explicit_selected = latent_examples_module.select_target_subruns_for_examples(
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={
                            "enabled": True,
                            "target_subruns": "explicit",
                        }
                    )
                }
            ),
            resolved_config=resolved,
            output_dir="/tmp/out",
        )
        self.assertEqual(len(explicit_selected), 1)
        with patch(
            "sgnsae.experiments.llm_benchmark.latent_examples._load_or_build_summary",
            return_value={"subruns": []},
        ):
            with self.assertRaises(ValueError):
                _ = latent_examples_module.select_target_subruns_for_examples(
                    config=types.SimpleNamespace(
                        latent_examples=types.SimpleNamespace(
                            target_subruns="weird",
                            seeds=None,
                        )
                    ),
                    resolved_config={"subruns": []},
                    output_dir="/tmp/out",
                )

    def test_dtype_latent_and_patch_helpers_cover_remaining_paths(self) -> None:
        # given
        config = LLMBenchmarkConfig.model_validate(_text_config(output_dir="/tmp/out"))
        image_config = LLMBenchmarkConfig.model_validate(
            _image_config(output_dir="/tmp/out")
        )

        # when / then
        self.assertEqual(
            latent_examples_module._resolve_text_backbone_dtype(
                config=config.model_copy(
                    update={
                        "llm": config.llm.model_copy(update={"torch_dtype": "auto"})
                    }
                ),
                runtime_device="cuda",
            ),
            torch.float16,
        )
        self.assertEqual(
            latent_examples_module._resolve_text_backbone_dtype(
                config=config.model_copy(
                    update={
                        "llm": config.llm.model_copy(update={"torch_dtype": "float16"})
                    }
                ),
                runtime_device="cpu",
            ),
            torch.float16,
        )
        self.assertEqual(
            latent_examples_module._resolve_text_backbone_dtype(
                config=config.model_copy(
                    update={
                        "llm": config.llm.model_copy(update={"torch_dtype": "float32"})
                    }
                ),
                runtime_device="cpu",
            ),
            torch.float32,
        )
        self.assertEqual(
            latent_examples_module._resolve_text_backbone_dtype(
                config=config.model_copy(
                    update={
                        "llm": config.llm.model_copy(update={"torch_dtype": "bfloat16"})
                    }
                ),
                runtime_device="cpu",
            ),
            torch.bfloat16,
        )
        with self.assertRaises(ValueError):
            _ = latent_examples_module._resolve_text_backbone_dtype(
                config=config.model_copy(
                    update={"llm": config.llm.model_copy(update={"torch_dtype": "bad"})}
                ),
                runtime_device="cpu",
            )

        self.assertEqual(
            latent_examples_module._resolve_vision_backbone_dtype(
                config=image_config.model_copy(
                    update={
                        "vision": image_config.vision.model_copy(
                            update={"torch_dtype": "auto"}
                        )
                    }
                ),
                runtime_device="cuda",
            ),
            torch.float16,
        )
        self.assertEqual(
            latent_examples_module._resolve_vision_backbone_dtype(
                config=image_config.model_copy(
                    update={
                        "vision": image_config.vision.model_copy(
                            update={"torch_dtype": "float16"}
                        )
                    }
                ),
                runtime_device="cpu",
            ),
            torch.float16,
        )
        self.assertEqual(
            latent_examples_module._resolve_vision_backbone_dtype(
                config=image_config.model_copy(
                    update={
                        "vision": image_config.vision.model_copy(
                            update={"torch_dtype": "auto"}
                        )
                    }
                ),
                runtime_device="cpu",
            ),
            torch.float32,
        )
        self.assertEqual(
            latent_examples_module._resolve_vision_backbone_dtype(
                config=image_config.model_copy(
                    update={
                        "vision": image_config.vision.model_copy(
                            update={"torch_dtype": "bfloat16"}
                        )
                    }
                ),
                runtime_device="cpu",
            ),
            torch.bfloat16,
        )
        with self.assertRaises(ValueError):
            _ = latent_examples_module._resolve_vision_backbone_dtype(
                config=image_config.model_copy(
                    update={
                        "vision": image_config.vision.model_copy(
                            update={"torch_dtype": "bad"}
                        )
                    }
                ),
                runtime_device="cpu",
            )

        tensor_2d = torch.zeros(2, 3)
        tensor_4d = torch.zeros(1, 1, 2, 3)
        self.assertEqual(
            tuple(
                latent_examples_module._flatten_single_instance_latents(tensor_2d).shape
            ),
            (2, 3),
        )
        self.assertEqual(
            tuple(
                latent_examples_module._flatten_single_instance_latents(tensor_4d).shape
            ),
            (2, 3),
        )

        explicit_none = latent_examples_module._select_latent_indices_from_statistics(
            max_abs_by_latent=torch.zeros(2),
            activation_count_by_latent=torch.zeros(2),
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={
                            "latent_selection": "explicit",
                            "explicit_latents": [9, 9],
                            "max_latents": 2,
                        }
                    )
                }
            ),
            d_hidden=2,
        )
        self.assertEqual(explicit_none, [])
        explicit_break = latent_examples_module._select_latent_indices_from_statistics(
            max_abs_by_latent=torch.zeros(5),
            activation_count_by_latent=torch.zeros(5),
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={
                            "latent_selection": "explicit",
                            "explicit_latents": [0, 1, 2, 3],
                            "max_latents": 2,
                        }
                    )
                }
            ),
            d_hidden=5,
        )
        self.assertEqual(explicit_break, [0, 1])
        explicit_duplicate = (
            latent_examples_module._select_latent_indices_from_statistics(
                max_abs_by_latent=torch.zeros(2),
                activation_count_by_latent=torch.zeros(2),
                config=config.model_copy(
                    update={
                        "latent_examples": config.latent_examples.model_copy(
                            update={
                                "latent_selection": "explicit",
                                "explicit_latents": [0, 0, 1],
                                "max_latents": 2,
                            }
                        )
                    }
                ),
                d_hidden=2,
            )
        )
        self.assertEqual(explicit_duplicate, [0, 1])
        top_by_abs = latent_examples_module._select_latent_indices_from_statistics(
            max_abs_by_latent=torch.tensor(
                [1.0, float("nan"), 2.0], dtype=torch.float64
            ),
            activation_count_by_latent=torch.zeros(3),
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"latent_selection": "top_by_max_abs", "max_latents": 2}
                    )
                }
            ),
            d_hidden=3,
        )
        self.assertEqual(top_by_abs[0], 2)
        no_frequency_evidence = (
            latent_examples_module._select_latent_indices_from_statistics(
                max_abs_by_latent=torch.zeros(3, dtype=torch.float64),
                activation_count_by_latent=torch.zeros(3, dtype=torch.float64),
                config=config.model_copy(
                    update={
                        "latent_examples": config.latent_examples.model_copy(
                            update={
                                "latent_selection": "top_by_freq",
                                "max_latents": 2,
                            }
                        )
                    }
                ),
                d_hidden=3,
            )
        )
        self.assertEqual(no_frequency_evidence, [])
        with_frequency_evidence = (
            latent_examples_module._select_latent_indices_from_statistics(
                max_abs_by_latent=torch.zeros(3, dtype=torch.float64),
                activation_count_by_latent=torch.tensor(
                    [0.0, 3.0, 1.0], dtype=torch.float64
                ),
                config=config.model_copy(
                    update={
                        "latent_examples": config.latent_examples.model_copy(
                            update={
                                "latent_selection": "top_by_freq",
                                "max_latents": 2,
                            }
                        )
                    }
                ),
                d_hidden=3,
            )
        )
        self.assertEqual(with_frequency_evidence, [1, 2])
        zero_latents = latent_examples_module._select_latent_indices_from_statistics(
            max_abs_by_latent=torch.ones(3),
            activation_count_by_latent=torch.ones(3),
            config=config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"max_latents": 1}
                    )
                }
            ),
            d_hidden=0,
        )
        self.assertEqual(zero_latents, [])

        heaps: Dict[int, List[Tuple[float, int, Any]]] = {}
        latent_examples_module._push_bounded_example_heap(
            heaps_by_latent=heaps,
            latent_index=0,
            score=0.1,
            payload="a",
            max_examples_per_latent=1,
            sequence_order=0,
        )
        latent_examples_module._push_bounded_example_heap(
            heaps_by_latent=heaps,
            latent_index=0,
            score=0.2,
            payload="b",
            max_examples_per_latent=1,
            sequence_order=1,
        )
        self.assertEqual(len(heaps[0]), 1)
        self.assertEqual(heaps[0][0][2], "b")

        stats_none = latent_examples_module._image_normalization_stats(
            processor=types.SimpleNamespace(image_mean=None, image_std=None),
            channels=3,
        )
        self.assertEqual(tuple(stats_none[0].shape), (3, 1, 1))
        stats_scalar = latent_examples_module._image_normalization_stats(
            processor=types.SimpleNamespace(image_mean=0.5, image_std=0.25),
            channels=3,
        )
        self.assertEqual(tuple(stats_scalar[0].shape), (3, 1, 1))
        stats_len1 = latent_examples_module._image_normalization_stats(
            processor=types.SimpleNamespace(image_mean=[0.5], image_std=[0.2]),
            channels=3,
        )
        self.assertEqual(tuple(stats_len1[0].shape), (3, 1, 1))
        with self.assertRaises(ValueError):
            _ = latent_examples_module._image_normalization_stats(
                processor=types.SimpleNamespace(image_mean=[0.1, 0.2], image_std=[0.1]),
                channels=3,
            )

        gray_image = latent_examples_module._pixel_values_to_pil_rgb_image(
            pixel_values=torch.ones(1, 16, 16),
            processor=types.SimpleNamespace(image_mean=0.0, image_std=1.0),
        )
        self.assertEqual(gray_image.mode, "RGB")
        with self.assertRaises(ValueError):
            _ = latent_examples_module._pixel_values_to_pil_rgb_image(
                pixel_values=torch.ones(2, 16, 16),
                processor=types.SimpleNamespace(image_mean=0.0, image_std=1.0),
            )

        with self.assertRaises(ValueError):
            _ = latent_examples_module.compute_patch_crop_bounds(
                patch_index=True,
                n_patches_w=4,
                patch_size=16,
                image_size=64,
            )
        with self.assertRaises(ValueError):
            _ = latent_examples_module.compute_patch_crop_bounds(
                patch_index=-1,
                n_patches_w=4,
                patch_size=16,
                image_size=64,
            )
        with self.assertRaises(ValueError):
            _ = latent_examples_module.compute_patch_crop_bounds(
                patch_index=0,
                n_patches_w=0,
                patch_size=16,
                image_size=64,
            )

    def test_extract_patch_and_model_loading_error_paths(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = latent_examples_module._extract_patch_hidden_states(
                hidden_state="not_a_tensor",
                expected_patch_tokens=4,
            )
        with self.assertRaises(ValueError):
            _ = latent_examples_module._extract_patch_hidden_states(
                hidden_state=torch.ones(3, 4),
                expected_patch_tokens=4,
            )
        with self.assertRaises(ValueError):
            _ = latent_examples_module._extract_patch_hidden_states(
                hidden_state=torch.ones(2, 4, 3),
                expected_patch_tokens=4,
            )
        no_cls = latent_examples_module._extract_patch_hidden_states(
            hidden_state=torch.ones(1, 4, 3),
            expected_patch_tokens=4,
        )
        self.assertEqual(tuple(no_cls.shape), (4, 3))
        with_prefix = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
        extracted_with_prefix = latent_examples_module._extract_patch_hidden_states(
            hidden_state=with_prefix,
            expected_patch_tokens=4,
        )
        self.assertTrue(
            bool(torch.equal(extracted_with_prefix, with_prefix[0, -4:, :]))
        )
        feature_map_hidden_state = torch.arange(12, dtype=torch.float32).reshape(
            1, 3, 2, 2
        )
        extracted_feature_map = latent_examples_module._extract_patch_hidden_states(
            hidden_state=feature_map_hidden_state,
            expected_patch_tokens=4,
        )
        expected_feature_map = torch.tensor(
            [
                [0.0, 4.0, 8.0],
                [1.0, 5.0, 9.0],
                [2.0, 6.0, 10.0],
                [3.0, 7.0, 11.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(bool(torch.equal(extracted_feature_map, expected_feature_map)))
        with self.assertRaises(ValueError):
            _ = latent_examples_module._extract_patch_hidden_states(
                hidden_state=torch.ones(1, 3, 3),
                expected_patch_tokens=4,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=output_dir)
            )
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            model_path = os.path.join(subrun_dir, "model.pt")

            torch.save([], model_path)
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_manifest_path",
                return_value=os.path.join(output_dir, "activations", "manifest.json"),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.load_activation_manifest",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.build_model_for_subrun",
                return_value=(DummySAE(d_hidden=4), 0.0, 1, {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._load_sae_model_for_subrun(
                        config=config,
                        resolved_config={"subruns": [subrun]},
                        subrun=subrun,
                        output_dir=output_dir,
                    )

            torch.save({"state_dict": []}, model_path)
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_manifest_path",
                return_value=os.path.join(output_dir, "activations", "manifest.json"),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.load_activation_manifest",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.build_model_for_subrun",
                return_value=(DummySAE(d_hidden=4), 0.0, 1, {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._load_sae_model_for_subrun(
                        config=config,
                        resolved_config={"subruns": [subrun]},
                        subrun=subrun,
                        output_dir=output_dir,
                    )

    def test_generate_text_examples_error_branches(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = latent_examples_module._generate_text_examples(
                config=types.SimpleNamespace(llm=None),
                sae_model=DummySAE(),
                runtime_device="cpu",
                d_hidden=4,
            )
        with self.assertRaises(ValueError):
            _ = latent_examples_module._generate_text_examples(
                config=types.SimpleNamespace(
                    llm=types.SimpleNamespace(
                        model_id="m",
                        revision=None,
                        tokenizer_id=None,
                        tokenizer_revision=None,
                        torch_dtype="float32",
                    ),
                    data={},
                ),
                sae_model=DummySAE(),
                runtime_device="cpu",
                d_hidden=4,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "cache": base_config.cache.model_copy(update={"max_seq_len": 0}),
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"sample_size": 1, "max_latents": 2}
                    ),
                }
            )
            hook_module = DummyHookModule()
            tokenizer = DummyTokenizerScripted(scripted_outputs=["bad", [], [1]])
            text_model = DummyTextModelScripted(
                hook_module=hook_module, scripted_outputs=[torch.ones(1, 2)]
            )

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=text_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["doc a", "doc b", "doc c"], {}),
            ):
                empty_examples, _, scanned = (
                    latent_examples_module._generate_text_examples(
                        config=config,
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )
            self.assertEqual(empty_examples, {})
            self.assertEqual(scanned, 0)

            config_valid = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"sample_size": 2, "max_latents": 2}
                    )
                }
            )

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyTextModelScripted(
                    hook_module, scripted_outputs=[None]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["a b"], {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_text_examples(
                        config=config_valid,
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyTextModelScripted(
                    hook_module, scripted_outputs=[torch.ones(1, 2, 2, 2)]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["a b"], {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_text_examples(
                        config=config_valid,
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyTextModelScripted(
                    hook_module, scripted_outputs=[torch.ones(1, 2)]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["a b"], {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_text_examples(
                        config=config_valid,
                        sae_model=DummySAEWrongHidden(d_hidden=4, wrong_hidden=3),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            explicit_config = config_valid.model_copy(
                update={
                    "latent_examples": config_valid.latent_examples.model_copy(
                        update={
                            "latent_selection": "explicit",
                            "explicit_latents": [99],
                        }
                    )
                }
            )
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyTextModelScripted(
                    hook_module, scripted_outputs=[torch.ones(1, 2, 2)]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["a b"], {}),
            ):
                examples, selected_latents, _ = (
                    latent_examples_module._generate_text_examples(
                        config=explicit_config,
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )
            self.assertEqual(examples, {})
            self.assertEqual(selected_latents, [])

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyTextModelScripted(
                    hook_module,
                    scripted_outputs=[
                        torch.ones(1, 2, 2),
                        torch.ones(1, 2, 2),
                        None,
                    ],
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["a b", "c d"], {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_text_examples(
                        config=config_valid,
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyTextModelScripted(
                    hook_module,
                    scripted_outputs=[
                        torch.ones(1, 2, 2),
                        torch.ones(1, 2, 2),
                        torch.ones(1, 2),
                        torch.ones(1, 2),
                    ],
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["a b", "c d"], {}),
            ):
                examples, _, _ = latent_examples_module._generate_text_examples(
                    config=config_valid,
                    sae_model=DummySAE(),
                    runtime_device="cpu",
                    d_hidden=4,
                )
            self.assertTrue(bool(examples))

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyTextModelScripted(
                    hook_module,
                    scripted_outputs=[
                        torch.ones(1, 2, 2),
                        torch.ones(1, 2, 2),
                        torch.ones(1, 2, 2),
                        torch.ones(1, 2, 2, 2),
                    ],
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["a b", "c d"], {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_text_examples(
                        config=config_valid,
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            text_model_without_use_cache = DummyTextModelScripted(
                hook_module,
                scripted_outputs=[torch.ones(1, 2, 2), torch.ones(1, 2, 2)],
            )
            text_model_without_use_cache.config = types.SimpleNamespace()
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=text_model_without_use_cache,
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples.resolve_hook_module",
                return_value=(hook_module, "dummy.path", 0),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_text_documents",
                return_value=(["a b"], {}),
            ):
                examples_without_use_cache, _, _ = (
                    latent_examples_module._generate_text_examples(
                        config=config_valid,
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )
            self.assertTrue(bool(examples_without_use_cache))

    def test_generate_image_examples_error_branches(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = latent_examples_module._generate_image_examples(
                config=types.SimpleNamespace(vision=None),
                manifest={},
                sae_model=DummySAE(),
                runtime_device="cpu",
                d_hidden=4,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _image_config(output_dir=tmpdir)
            )
            with self.assertRaises(ValueError):
                _ = latent_examples_module._generate_image_examples(
                    config=base_config.model_copy(update={"data": {}}),
                    manifest={},
                    sae_model=DummySAE(),
                    runtime_device="cpu",
                    d_hidden=4,
                )
            with self.assertRaises(ValueError):
                _ = latent_examples_module._generate_image_examples(
                    config=base_config.model_copy(update={"cache": {}}),
                    manifest={},
                    sae_model=DummySAE(),
                    runtime_device="cpu",
                    d_hidden=4,
                )

            image_entries = [
                {"image": Image.new("RGB", (64, 64), color=(20, 30, 40))},
                {"image": Image.new("RGB", (64, 64), color=(60, 70, 80))},
            ]

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_image_examples(
                        config=base_config,
                        manifest={"patch_size": 30, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(include_metadata=True),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModelScripted(
                    scripted_hidden_states=[
                        None,
                    ]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_image_examples(
                        config=base_config.model_copy(
                            update={
                                "latent_examples": base_config.latent_examples.model_copy(
                                    update={"sample_size": 1}
                                )
                            }
                        ),
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModelScripted(
                    scripted_hidden_states=[(torch.ones(1, 5, 2),)]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_image_examples(
                        config=base_config,
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_image_examples(
                        config=base_config,
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAEWrongHidden(d_hidden=4, wrong_hidden=3),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(include_pixel_values=False),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModelScripted(
                    scripted_hidden_states=[
                        (
                            torch.ones(1, 5, 2),
                            torch.ones(1, 5, 2),
                        )
                    ]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_image_examples(
                        config=base_config,
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(include_metadata=True),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModelScripted(
                    scripted_hidden_states=[
                        (
                            torch.ones(1, 5, 2),
                            torch.ones(1, 5, 2),
                        ),
                        (
                            torch.ones(1, 5, 2),
                            torch.ones(1, 5, 2),
                        ),
                    ]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                examples_with_metadata, selected_with_metadata, _ = (
                    latent_examples_module._generate_image_examples(
                        config=base_config.model_copy(
                            update={
                                "latent_examples": base_config.latent_examples.model_copy(
                                    update={"sample_size": 1}
                                )
                            }
                        ),
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )
            self.assertTrue(bool(examples_with_metadata))
            self.assertTrue(bool(selected_with_metadata))

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(include_metadata=True),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModelScripted(
                    scripted_hidden_states=[
                        (
                            torch.tensor(
                                [
                                    [
                                        [0.0, 0.0],
                                        [1.0, 0.0],
                                        [2.0, 0.0],
                                        [3.0, 0.0],
                                        [4.0, 0.0],
                                    ]
                                ],
                                dtype=torch.float32,
                            ),
                            torch.tensor(
                                [
                                    [
                                        [0.0, 0.0],
                                        [1.0, 0.0],
                                        [2.0, 0.0],
                                        [3.0, 0.0],
                                        [4.0, 0.0],
                                    ]
                                ],
                                dtype=torch.float32,
                            ),
                        ),
                        None,
                    ]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_image_examples(
                        config=base_config.model_copy(
                            update={
                                "latent_examples": base_config.latent_examples.model_copy(
                                    update={"sample_size": 1}
                                )
                            }
                        ),
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModelScripted(
                    scripted_hidden_states=[
                        (
                            torch.tensor(
                                [
                                    [
                                        [0.0, 0.0],
                                        [1.0, 0.0],
                                        [2.0, 0.0],
                                        [3.0, 0.0],
                                        [4.0, 0.0],
                                    ]
                                ],
                                dtype=torch.float32,
                            ),
                            torch.tensor(
                                [
                                    [
                                        [0.0, 0.0],
                                        [1.0, 0.0],
                                        [2.0, 0.0],
                                        [3.0, 0.0],
                                        [4.0, 0.0],
                                    ]
                                ],
                                dtype=torch.float32,
                            ),
                        ),
                        (torch.ones(1, 5, 2),),
                    ]
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                with self.assertRaises(ValueError):
                    _ = latent_examples_module._generate_image_examples(
                        config=base_config.model_copy(
                            update={
                                "latent_examples": base_config.latent_examples.model_copy(
                                    update={"sample_size": 1}
                                )
                            }
                        ),
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                examples, selected_latents, _ = (
                    latent_examples_module._generate_image_examples(
                        config=base_config.model_copy(
                            update={
                                "latent_examples": base_config.latent_examples.model_copy(
                                    update={"activation_threshold": 999.0}
                                )
                            }
                        ),
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )
            self.assertEqual(examples, {})
            self.assertEqual(selected_latents, [])

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                (
                    explicit_high_threshold_examples,
                    explicit_high_threshold_selected,
                    _,
                ) = latent_examples_module._generate_image_examples(
                    config=base_config.model_copy(
                        update={
                            "latent_examples": base_config.latent_examples.model_copy(
                                update={
                                    "activation_threshold": 999.0,
                                    "latent_selection": "explicit",
                                    "explicit_latents": [0],
                                }
                            )
                        }
                    ),
                    manifest={"patch_size": 32, "image_size": 64},
                    sae_model=DummySAE(),
                    runtime_device="cpu",
                    d_hidden=4,
                )
            self.assertEqual(explicit_high_threshold_selected, [0])
            self.assertEqual(explicit_high_threshold_examples, {"0": []})

            with patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=DummyProcessorScripted(),
            ), patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyVisionModel(),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._iter_image_records",
                return_value=(iter(image_entries), {}),
            ):
                explicit_examples, explicit_selected_latents, explicit_scanned = (
                    latent_examples_module._generate_image_examples(
                        config=base_config.model_copy(
                            update={
                                "latent_examples": base_config.latent_examples.model_copy(
                                    update={
                                        "sample_size": 1,
                                        "latent_selection": "explicit",
                                        "explicit_latents": [99],
                                    }
                                )
                            }
                        ),
                        manifest={"patch_size": 32, "image_size": 64},
                        sae_model=DummySAE(),
                        runtime_device="cpu",
                        d_hidden=4,
                    )
                )
            self.assertEqual(explicit_examples, {})
            self.assertEqual(explicit_selected_latents, [])
            self.assertEqual(explicit_scanned, 1)

    def test_generate_for_subrun_explicit_selection_label_branch(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _text_config(output_dir=output_dir)
            )
            config = config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={
                            "enabled": True,
                            "latent_selection": "explicit",
                            "explicit_latents": [0],
                        }
                    )
                }
            )
            subrun = {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(
                output_dir, "models", "variant_a", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            _write_training_metrics(subrun_dir)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._load_sae_model_for_subrun",
                return_value=(
                    DummySAE(d_hidden=4),
                    {"patch_size": 32, "image_size": 64},
                    "cpu",
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.latent_examples._generate_text_examples",
                return_value=({"0": ["example"]}, [0], 1),
            ):
                result = latent_examples_module.generate_latent_examples_for_subrun(
                    config=config,
                    resolved_config={"subruns": [subrun]},
                    subrun=subrun,
                    output_dir=output_dir,
                    overwrite=True,
                )

            # then
            with open(result["latent_examples_path"], "r") as file:
                payload = json.load(file)
            self.assertEqual(payload["selection"]["latent_selection"], "explicit_list")
