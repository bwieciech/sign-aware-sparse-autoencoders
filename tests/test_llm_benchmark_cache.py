import contextlib
import io
import json
import os
import tempfile
import unittest
import types
import warnings
from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import patch

import torch

from sgnsae.experiments.llm_benchmark.cache import (
    _assign_split_shard_ranges,
    _build_stored_indices,
    _cache_has_all_shards,
    _default_stage_status,
    _extract_hook_tensor,
    _iter_jsonl_records,
    _iter_text_documents,
    _load_manifest,
    _manifest_schema_version_matches_expected,
    _resolve_default_module_path,
    _resolve_dtype,
    _resolve_dtype_name,
    _resolve_layer_count_and_family_prefix,
    _resolve_layer_index,
    _resolve_local_data_files,
    _resolve_model_torch_dtype,
    _resolve_runtime_device,
    _resolve_storage_dtype,
    _select_split_from_hash,
    resolve_hook_module,
    run_cache_stage,
)
from sgnsae.experiments.llm_benchmark.config import (
    ActivationCacheConfig,
    HookpointConfig,
    LLMBenchmarkConfig,
    TextDataConfig,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)


class DummyTokenizer:
    def __init__(self, eos_token_id: int = 0) -> None:
        self.eos_token_id = eos_token_id
        self.eos_token = "<eos>"
        self.pad_token_id = None
        self._pad_token = None

    @property
    def pad_token(self) -> Any:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: Any) -> None:
        self._pad_token = value
        if value == self.eos_token:
            self.pad_token_id = self.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        tokens = [max(1, len(piece) % 11) for piece in text.split() if piece]
        if not tokens:
            return [1]
        return tokens


class DummyMLP(torch.nn.Module):
    def __init__(self, tuple_output: bool = False) -> None:
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, x: torch.Tensor) -> Any:
        output = x + 0.5
        if self.tuple_output:
            return output, output.mean(dim=-1)
        return output


class DummyLayer(torch.nn.Module):
    def __init__(self, tuple_output: bool = False) -> None:
        super().__init__()
        self.mlp = DummyMLP(tuple_output=tuple_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.mlp(x)
        if isinstance(output, tuple):
            return output[0]
        return output


class DummyModel(torch.nn.Module):
    def __init__(
        self, d_in: int = 16, n_layers: int = 2, tuple_output: bool = False
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.config = types.SimpleNamespace(use_cache=True)
        self.forward_kwargs: List[Dict[str, Any]] = []
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList(
            [
                DummyLayer(tuple_output=(tuple_output and index == 0))
                for index in range(n_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        del attention_mask
        self.forward_kwargs.append(
            {
                "use_cache": use_cache,
                "return_dict": return_dict,
            }
        )
        activations = torch.nn.functional.one_hot(
            torch.remainder(input_ids, self.d_in),
            num_classes=self.d_in,
        ).to(torch.float32)

        for layer in self.transformer.h:
            activations = layer(activations)

        return {"last_hidden_state": activations}

    def to(self, device: Any) -> "DummyModel":
        del device
        return self


class DummyModelNoConfig(DummyModel):
    def __init__(self, d_in: int = 16, n_layers: int = 2) -> None:
        super().__init__(d_in=d_in, n_layers=n_layers, tuple_output=False)
        del self.config


class DummyModuleLookupModel(torch.nn.Module):
    def __init__(self, lookup: Dict[str, Any]) -> None:
        super().__init__()
        self.lookup = lookup

    def get_submodule(self, target: str) -> Any:
        if target not in self.lookup:
            raise AttributeError(target)
        return self.lookup[target]


class DummyGPT2BaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList([DummyLayer()])


class DummyLlamaBaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyLayer()])


class DummyTokenizerNoEos:
    def __init__(self) -> None:
        self.eos_token_id = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [max(1, len(text))]


class DummyTokenizerNoEosMultiToken(DummyTokenizerNoEos):
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        tokens = [max(1, len(piece) % 11) for piece in text.split() if piece]
        if not tokens:
            return [1]
        return tokens


class DummyTokenizerNoPadOrEos:
    def __init__(self) -> None:
        self.eos_token_id = None
        self.eos_token = None
        self.pad_token_id = None
        self.pad_token = None

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [max(1, len(text))]


class DummyTokenizerBadEncode:
    def __init__(self) -> None:
        self.eos_token_id = 0
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> int:
        del text
        del add_special_tokens
        return 1


class DummyLayerTwoCalls(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = DummyMLP(tuple_output=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.mlp(x)
        return self.mlp(first)


class DummyModelTwoHookCalls(DummyModel):
    def __init__(self, d_in: int = 16) -> None:
        super().__init__(d_in=d_in, n_layers=1, tuple_output=False)
        self.transformer.h[0] = DummyLayerTwoCalls()


class DummyMLP2D(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class DummyLayer2D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = DummyMLP2D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DummyModel2D(DummyModel):
    def __init__(self, d_in: int = 16) -> None:
        super().__init__(d_in=d_in, n_layers=1, tuple_output=False)
        self.transformer.h[0] = DummyLayer2D()


class DummyVaryingDimMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        if self.call_count % 2 == 0:
            return x[..., :-1]
        return x


class DummyVaryingDimLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = DummyVaryingDimMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DummyModelVaryingDim(DummyModel):
    def __init__(self, d_in: int = 16) -> None:
        super().__init__(d_in=d_in, n_layers=1, tuple_output=False)
        self.transformer.h[0] = DummyVaryingDimLayer()


class DummyHFStream:
    def __init__(self, entries: List[Any], fingerprint: Optional[str] = None) -> None:
        self._entries = entries
        self._fingerprint = fingerprint

    def __iter__(self) -> Iterable[Any]:
        return iter(self._entries)


def _write_local_jsonl(path: str, texts: List[str]) -> None:
    with open(path, "w") as file:
        for text in texts:
            file.write(json.dumps({"text": text}))
            file.write("\n")


def _base_config(output_dir: str, data_path: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": [0],
        "llm": {"model_id": "dummy-model"},
        "data": {
            "backend": "local_jsonl",
            "data_files": [data_path],
            "text_field": "text",
        },
        "hook": {
            "hook_name": "mlp_out",
            "layer": 0,
        },
        "cache": {
            "llm_batch_size": 2,
            "max_seq_len": 4,
            "max_sequences": 4,
            "vectors_per_shard": 8,
            "cache_overwrite": False,
            "split_method": "shard_ranges",
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "cache_seed": 7,
            "store_tokens": False,
            "store_positions": False,
            "store_sequence_ids": False,
            "compute_dtype": "float32",
            "storage_dtype": "float32",
            "doc_separator": "eos",
            "text_field": "text",
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
    }


class TestLLMBenchmarkCache(unittest.TestCase):
    def test_cache_helper_runtime_device_dtype_and_status_helpers(self) -> None:
        # given / when / then
        with patch(
            "sgnsae.experiments.device.torch.cuda.is_available",
            return_value=False,
        ):
            self.assertEqual(_resolve_runtime_device("auto"), "cpu")
            with self.assertRaises(ValueError):
                _ = _resolve_runtime_device("cuda")
        with patch(
            "sgnsae.experiments.device.torch.cuda.is_available",
            return_value=True,
        ), patch("sgnsae.experiments.device.torch.cuda.device_count", return_value=2):
            self.assertEqual(_resolve_runtime_device("auto"), "cuda")
            self.assertEqual(_resolve_runtime_device("cuda"), "cuda")
            self.assertEqual(_resolve_runtime_device("cuda:1"), "cuda:1")

        with patch(
            "sgnsae.experiments.device.torch.cuda.is_available",
            return_value=True,
        ), patch("sgnsae.experiments.device.torch.cuda.device_count", return_value=1):
            with self.assertRaises(ValueError):
                _ = _resolve_runtime_device("cuda:1")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["a b"])
            raw = _base_config(
                output_dir=os.path.join(tmpdir, "run"), data_path=data_path
            )
            config = LLMBenchmarkConfig.model_validate(raw)
            self.assertEqual(
                _resolve_storage_dtype(config=config, runtime_device="cpu"), "float32"
            )

            raw_without_storage_dtype = _base_config(
                output_dir=os.path.join(tmpdir, "run_no_storage"),
                data_path=data_path,
            )
            del raw_without_storage_dtype["cache"]["storage_dtype"]
            config_without_storage_dtype = LLMBenchmarkConfig.model_validate(
                raw_without_storage_dtype
            )
            self.assertEqual(
                _resolve_storage_dtype(
                    config=config_without_storage_dtype,
                    runtime_device="cpu",
                ),
                "float32",
            )
            self.assertEqual(
                _resolve_storage_dtype(
                    config=config_without_storage_dtype,
                    runtime_device="cuda",
                ),
                "float16",
            )

            explicit = config.model_copy(
                update={
                    "cache": config.cache.model_copy(
                        update={"storage_dtype": "float16"}
                    ),
                }
            )
            explicit.cache.model_fields_set.add("storage_dtype")
            self.assertEqual(
                _resolve_storage_dtype(config=explicit, runtime_device="cpu"), "float16"
            )

        self.assertEqual(_resolve_dtype("float16"), torch.float16)
        self.assertEqual(_resolve_dtype("float32"), torch.float32)
        self.assertEqual(_resolve_dtype("bfloat16"), torch.bfloat16)
        with self.assertRaises(ValueError):
            _ = _resolve_dtype("bad")
        self.assertEqual(_resolve_dtype_name(torch.float16), "float16")
        self.assertEqual(_resolve_dtype_name(torch.float32), "float32")
        self.assertEqual(_resolve_dtype_name(torch.bfloat16), "bfloat16")
        with self.assertRaises(ValueError):
            _ = _resolve_dtype_name(torch.int64)
        self.assertEqual(
            _resolve_model_torch_dtype(config=config, runtime_device="cpu"),
            torch.float32,
        )
        self.assertEqual(
            _resolve_model_torch_dtype(config=config, runtime_device="cuda"),
            torch.float16,
        )
        self.assertEqual(
            _resolve_model_torch_dtype(config=config, runtime_device="cuda:1"),
            torch.float16,
        )
        explicit_model_dtype = config.model_copy(
            update={
                "llm": config.llm.model_copy(update={"torch_dtype": "bfloat16"}),
            }
        )
        self.assertEqual(
            _resolve_model_torch_dtype(
                config=explicit_model_dtype,
                runtime_device="cuda",
            ),
            torch.bfloat16,
        )

        status = _default_stage_status("done")
        self.assertEqual(status["status"], "done")
        self.assertEqual(status["message"], "done")

    def test_cache_helper_layer_resolution_branches(self) -> None:
        # given
        model = DummyModel()

        # when / then
        self.assertEqual(
            _resolve_layer_count_and_family_prefix(model), (2, "transformer.h")
        )
        layer_index = _resolve_layer_index(
            model=model,
            hook=HookpointConfig.model_validate(
                {"hook_name": "mlp_out", "layer_selector": "late"}
            ),
        )
        self.assertEqual(layer_index, 0)
        float_selector_index = _resolve_layer_index(
            model=model,
            hook=HookpointConfig.model_validate(
                {"hook_name": "mlp_out", "layer_selector": 0.5}
            ),
        )
        self.assertEqual(float_selector_index, 0)
        mid_selector_index = _resolve_layer_index(
            model=model,
            hook=HookpointConfig.model_validate(
                {"hook_name": "mlp_out", "layer_selector": "mid"}
            ),
        )
        self.assertEqual(mid_selector_index, 1)

        with patch(
            "sgnsae.experiments.llm_benchmark.cache._resolve_layer_count_and_family_prefix",
            return_value=(0, "transformer.h"),
        ):
            with self.assertRaises(ValueError):
                _ = _resolve_layer_index(
                    model=model,
                    hook=HookpointConfig.model_validate(
                        {"hook_name": "mlp_out", "layer_selector": "mid"}
                    ),
                )

        with self.assertRaises(ValueError):
            _ = _resolve_default_module_path(
                model=model,
                hook=HookpointConfig.model_validate(
                    {"hook_name": "bad_hook", "layer_selector": "mid"}
                ),
                layer_index=0,
            )
        with self.assertRaises(ValueError):
            _ = _resolve_default_module_path(
                model=model,
                hook=HookpointConfig.model_validate(
                    {"hook_name": "mlp_out", "layer_selector": "mid"}
                ),
                layer_index=99,
            )

        module_lookup_model = DummyModuleLookupModel(
            {
                "transformer.h": [DummyLayer()],
                "gpt_neox.layers": (DummyLayer(),),
                "model.layers": torch.nn.ModuleList([DummyLayer()]),
            }
        )
        self.assertEqual(
            _resolve_layer_count_and_family_prefix(module_lookup_model),
            (1, "transformer.h"),
        )

        gpt_neox_lookup_model = DummyModuleLookupModel(
            {"gpt_neox.layers": torch.nn.ModuleList([DummyLayer()])}
        )
        self.assertEqual(
            _resolve_layer_count_and_family_prefix(gpt_neox_lookup_model),
            (1, "gpt_neox.layers"),
        )

        gpt2_base_model = DummyGPT2BaseModel()
        gpt2_base_hook_module, gpt2_base_module_path, gpt2_base_layer_index = (
            resolve_hook_module(
                gpt2_base_model,
                HookpointConfig.model_validate({"hook_name": "mlp_out", "layer": 0}),
            )
        )
        self.assertIsNotNone(gpt2_base_hook_module)
        self.assertEqual(gpt2_base_module_path, "h.0.mlp")
        self.assertEqual(gpt2_base_layer_index, 0)

        llama_base_model = DummyLlamaBaseModel()
        llama_base_hook_module, llama_base_module_path, llama_base_layer_index = (
            resolve_hook_module(
                llama_base_model,
                HookpointConfig.model_validate({"hook_name": "mlp_out", "layer": 0}),
            )
        )
        self.assertIsNotNone(llama_base_hook_module)
        self.assertEqual(llama_base_module_path, "layers.0.mlp")
        self.assertEqual(llama_base_layer_index, 0)

        with self.assertRaises(ValueError):
            _ = _resolve_layer_count_and_family_prefix(DummyModuleLookupModel({}))

        unsupported_selector_hook = HookpointConfig.model_construct(
            hook_name="mlp_out",
            layer=None,
            layer_selector=None,
            module_path=None,
        )
        with self.assertRaises(ValueError):
            _ = _resolve_layer_index(model=model, hook=unsupported_selector_hook)

    def test_cache_helper_local_files_and_jsonl_record_errors(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "a.jsonl")
            _write_local_jsonl(jsonl_path, ["hello"])

            data_files_config = TextDataConfig.model_validate(
                {
                    "backend": "local_jsonl",
                    "data_files": [jsonl_path],
                }
            )
            self.assertEqual(_resolve_local_data_files(data_files_config), [jsonl_path])
            jsonl_path_2 = os.path.join(tmpdir, "b.jsonl")
            _write_local_jsonl(jsonl_path_2, ["hello2"])
            glob_config = TextDataConfig.model_validate(
                {
                    "backend": "local_jsonl",
                    "data_files_glob": os.path.join(tmpdir, "*.jsonl"),
                }
            )
            self.assertEqual(
                _resolve_local_data_files(glob_config),
                sorted([jsonl_path, jsonl_path_2]),
            )

            missing_glob_config = TextDataConfig.model_validate(
                {
                    "backend": "local_jsonl",
                    "data_files_glob": os.path.join(tmpdir, "*.missing"),
                }
            )
            with self.assertRaises(ValueError):
                _ = _resolve_local_data_files(missing_glob_config)

            line_with_non_dict = os.path.join(tmpdir, "non_dict.jsonl")
            with open(line_with_non_dict, "w") as file:
                file.write("[1,2,3]\n")
            with self.assertRaises(ValueError):
                _ = list(_iter_jsonl_records(line_with_non_dict))

            bad_json_line = os.path.join(tmpdir, "bad.jsonl")
            with open(bad_json_line, "w") as file:
                file.write("{bad\n")
            with self.assertRaises(ValueError):
                _ = list(_iter_jsonl_records(bad_json_line))

            original_import = __import__

            def _import_without_zstandard(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "zstandard":
                    raise ImportError("missing")
                return original_import(name, *args, **kwargs)

            with patch(
                "builtins.__import__",
                side_effect=_import_without_zstandard,
            ):
                with self.assertRaises(ValueError):
                    _ = list(
                        _iter_jsonl_records(os.path.join(tmpdir, "fake.jsonl.zst"))
                    )

            blank_lines_path = os.path.join(tmpdir, "blank_lines.jsonl")
            with open(blank_lines_path, "w") as file:
                file.write("\n")
                file.write(json.dumps({"text": "x"}))
                file.write("\n")
            parsed_blank_lines = list(_iter_jsonl_records(blank_lines_path))
            self.assertEqual(parsed_blank_lines, [{"text": "x"}])

            missing_source = TextDataConfig.model_construct(
                backend="local_jsonl",
                dataset_id=None,
                split=None,
                revision=None,
                streaming=True,
                text_field="text",
                data_files_glob=None,
                data_files=None,
            )
            with self.assertRaises(ValueError):
                _ = _resolve_local_data_files(missing_source)

    def test_cache_helper_jsonl_zst_success_and_error_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            zst_path = os.path.join(tmpdir, "data.jsonl.zst")
            with open(zst_path, "wb") as file:
                file.write(b"placeholder")

            class _FakeReader:
                def __init__(self, payload: bytes) -> None:
                    self.payload = payload
                    self.stream = io.BytesIO(payload)

                def __enter__(self) -> io.BytesIO:
                    return self.stream

                def __exit__(
                    self,
                    exc_type: Any,
                    exc: Any,
                    traceback: Any,
                ) -> bool:
                    del exc_type
                    del exc
                    del traceback
                    return False

            class _FakeDecompressor:
                def __init__(self, payload: bytes) -> None:
                    self.payload = payload

                def stream_reader(self, compressed_file: Any) -> _FakeReader:
                    del compressed_file
                    return _FakeReader(self.payload)

            def _fake_zstandard_module(payload: bytes) -> Any:
                return types.SimpleNamespace(
                    ZstdDecompressor=lambda: _FakeDecompressor(payload)
                )

            # when
            with patch.dict(
                "sys.modules",
                {"zstandard": _fake_zstandard_module(b'\n{"text":"ok"}\n')},
            ):
                parsed = list(_iter_jsonl_records(zst_path))

            # then
            self.assertEqual(parsed, [{"text": "ok"}])

            # when / then
            with patch.dict(
                "sys.modules",
                {"zstandard": _fake_zstandard_module(b"{bad\n")},
            ):
                with self.assertRaises(ValueError):
                    _ = list(_iter_jsonl_records(zst_path))

            with patch.dict(
                "sys.modules",
                {"zstandard": _fake_zstandard_module(b"[1,2,3]\n")},
            ):
                with self.assertRaises(ValueError):
                    _ = list(_iter_jsonl_records(zst_path))

    def test_cache_helper_iter_text_documents_for_hf_and_local(self) -> None:
        # given
        cache_config = ActivationCacheConfig.model_validate({"text_field": "text"})

        hf_data = TextDataConfig.model_validate(
            {"backend": "hf", "dataset_id": "dataset/id", "split": "train"}
        )

        def _fake_datasets_module(stream: DummyHFStream) -> Any:
            return types.SimpleNamespace(load_dataset=lambda **_: stream)

        # when
        fake_stream = DummyHFStream(
            entries=[{"text": "first"}, {"text": "second"}],
            fingerprint="fp-123",
        )
        with patch.dict(
            "sys.modules",
            {"datasets": _fake_datasets_module(fake_stream)},
        ):
            docs_iter, metadata = _iter_text_documents(hf_data, cache_config)
            docs = list(docs_iter)

        # then
        self.assertEqual(docs, ["first", "second"])
        self.assertEqual(metadata["dataset_fingerprint"], "fp-123")

        # when / then
        bad_stream = DummyHFStream(entries=[{"not_text": "value"}], fingerprint=None)
        with patch.dict(
            "sys.modules",
            {"datasets": _fake_datasets_module(bad_stream)},
        ):
            docs_iter, _ = _iter_text_documents(hf_data, cache_config)
            with self.assertRaises(ValueError):
                _ = list(docs_iter)

        non_dict_stream = DummyHFStream(entries=["bad"], fingerprint=None)
        with patch.dict(
            "sys.modules",
            {"datasets": _fake_datasets_module(non_dict_stream)},
        ):
            docs_iter, _ = _iter_text_documents(hf_data, cache_config)
            with self.assertRaises(ValueError):
                _ = list(docs_iter)

        non_string_stream = DummyHFStream(entries=[{"text": 123}], fingerprint=None)
        with patch.dict(
            "sys.modules",
            {"datasets": _fake_datasets_module(non_string_stream)},
        ):
            docs_iter, _ = _iter_text_documents(hf_data, cache_config)
            with self.assertRaises(ValueError):
                _ = list(docs_iter)

        original_import = __import__

        def _patched_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "datasets":
                raise ImportError("missing datasets")
            return original_import(name, *args, **kwargs)

        with patch(
            "builtins.__import__",
            side_effect=_patched_import,
        ):
            with self.assertRaises(ValueError):
                _ = _iter_text_documents(hf_data, cache_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "local.jsonl")
            with open(local_path, "w") as file:
                file.write(json.dumps({"text": 123}))
                file.write("\n")
            local_data = TextDataConfig.model_validate(
                {"backend": "local_jsonl", "data_files": [local_path]}
            )
            docs_iter, _ = _iter_text_documents(local_data, cache_config)
            with self.assertRaises(ValueError):
                _ = list(docs_iter)

            missing_text_path = os.path.join(tmpdir, "missing.jsonl")
            with open(missing_text_path, "w") as file:
                file.write(json.dumps({"not_text": "value"}))
                file.write("\n")
            missing_text_data = TextDataConfig.model_validate(
                {"backend": "local_jsonl", "data_files": [missing_text_path]}
            )
            docs_iter, _ = _iter_text_documents(missing_text_data, cache_config)
            with self.assertRaises(ValueError):
                _ = list(docs_iter)

    def test_cache_helper_manifest_and_shard_integrity_checks(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "acts")
            os.makedirs(activations_dir, exist_ok=True)

            manifest_path = os.path.join(activations_dir, "manifest.json")
            with open(manifest_path, "w") as file:
                json.dump([], file)
            self.assertIsNone(_load_manifest(manifest_path))
            self.assertFalse(
                _manifest_schema_version_matches_expected({"schema_version": True})
            )
            self.assertFalse(
                _manifest_schema_version_matches_expected({"schema_version": "1"})
            )
            self.assertTrue(
                _manifest_schema_version_matches_expected(
                    {"schema_version": int(ACTIVATIONS_MANIFEST_SCHEMA_VERSION)}
                )
            )

            self.assertFalse(_cache_has_all_shards({"shards": "bad"}, activations_dir))
            self.assertFalse(_cache_has_all_shards({"shards": [None]}, activations_dir))
            self.assertFalse(
                _cache_has_all_shards({"shards": [{"filename": ""}]}, activations_dir)
            )
            self.assertFalse(
                _cache_has_all_shards(
                    {"shards": [{"filename": "missing.pt"}]}, activations_dir
                )
            )

            shard_path = os.path.join(activations_dir, "present.pt")
            torch.save({"acts": torch.zeros((1, 2))}, shard_path)
            self.assertFalse(
                _cache_has_all_shards(
                    {
                        "shards": [{"filename": "present.pt"}],
                        "split": {"method": "stored_indices", "indices_filename": ""},
                    },
                    activations_dir,
                )
            )
            self.assertFalse(
                _cache_has_all_shards(
                    {
                        "shards": [{"filename": "present.pt"}],
                        "split": {
                            "method": "stored_indices",
                            "indices_filename": "split_indices.pt",
                        },
                    },
                    activations_dir,
                )
            )
            torch.save(
                {
                    "train": torch.tensor([0], dtype=torch.int64),
                    "val": torch.tensor([], dtype=torch.int64),
                    "test": torch.tensor([], dtype=torch.int64),
                },
                os.path.join(activations_dir, "split_indices.pt"),
            )
            self.assertTrue(
                _cache_has_all_shards(
                    {
                        "shards": [{"filename": "present.pt"}],
                        "split": {
                            "method": "stored_indices",
                            "indices_filename": "split_indices.pt",
                        },
                    },
                    activations_dir,
                )
            )

    def test_cache_helper_extract_and_split_helpers(self) -> None:
        # given
        tensor = torch.ones((2, 2))

        # when / then
        self.assertTrue(torch.equal(_extract_hook_tensor(tensor, "m"), tensor))
        self.assertTrue(torch.equal(_extract_hook_tensor((None, tensor), "m"), tensor))
        with self.assertRaises(TypeError):
            _ = _extract_hook_tensor({"not": "tensor"}, "m")

        self.assertEqual(
            _assign_split_shard_ranges(
                num_shards=0, train_fraction=0.8, val_fraction=0.1
            ),
            {"train": [], "val": [], "test": []},
        )
        split_with_val_decrement = _assign_split_shard_ranges(
            num_shards=1, train_fraction=0.0, val_fraction=1.0
        )
        self.assertEqual(split_with_val_decrement["train"], [0])
        self.assertEqual(split_with_val_decrement["val"], [])

        split_a = _select_split_from_hash(
            1, cache_seed=42, train_fraction=0.3, val_fraction=0.3
        )
        split_b = _select_split_from_hash(
            1, cache_seed=42, train_fraction=0.3, val_fraction=0.3
        )
        self.assertEqual(split_a, split_b)

    def test_resolve_hook_module_success_and_failure(self) -> None:
        # given
        model = DummyModel()
        config = LLMBenchmarkConfig.model_validate(
            {
                "output_dir": "out",
                "seed": [0],
                "llm": {"model_id": "dummy-model"},
                "data": {
                    "backend": "local_jsonl",
                    "data_files": ["dummy.jsonl"],
                },
                "hook": {"hook_name": "mlp_out", "layer": 0},
                "variants": [
                    {
                        "name": "v",
                        "sae_class": "SAE",
                        "sign_aware": False,
                        "d_hidden": 4,
                        "sparsity_coeff": 1e-3,
                    }
                ],
            }
        )

        # when
        hook_module, module_path, layer_index = resolve_hook_module(model, config.hook)

        # then
        self.assertIsNotNone(hook_module)
        self.assertEqual(module_path, "transformer.h.0.mlp")
        self.assertEqual(layer_index, 0)

        # when / then
        invalid_hook_config = config.hook.model_copy(
            update={"module_path": "transformer.h.0.bad"}
        )
        with self.assertRaises(ValueError):
            _ = resolve_hook_module(model, invalid_hook_config)

    def test_cache_stage_supports_tuple_hook_output_and_pad_fallback(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta gamma", "delta epsilon", "zeta"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            config = LLMBenchmarkConfig.model_validate(raw_config)

            dummy_model = DummyModel(tuple_output=True)
            dummy_tokenizer = DummyTokenizer(eos_token_id=0)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=dummy_model,
            ) as model_loader, patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=dummy_tokenizer,
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            self.assertFalse(result["reused"])
            manifest_path = os.path.join(
                output_dir, "activations", "activations_manifest.json"
            )
            with open(manifest_path, "r") as file:
                manifest = json.load(file)
            self.assertTrue(manifest["tokenization"]["pad_token_fallback_applied"])
            self.assertEqual(manifest["tokenization"]["pad_token_id_used"], 0)
            self.assertEqual(manifest["activation"]["d_in"], 16)
            self.assertFalse(dummy_model.config.use_cache)
            self.assertGreaterEqual(len(dummy_model.forward_kwargs), 1)
            self.assertTrue(
                all(
                    kwargs["use_cache"] is False and kwargs["return_dict"] is False
                    for kwargs in dummy_model.forward_kwargs
                )
            )
            model_loader_call = model_loader.call_args
            self.assertIsNotNone(model_loader_call)
            self.assertEqual(model_loader_call.kwargs["torch_dtype"], torch.float32)
            self.assertTrue(bool(model_loader_call.kwargs["low_cpu_mem_usage"]))

    def test_cache_stage_handles_model_without_use_cache_config(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta gamma", "delta epsilon", "zeta"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModelNoConfig(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(eos_token_id=0),
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            self.assertFalse(result["reused"])

    def test_cache_stage_accepts_empty_existing_activations_directory(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["aa bb", "cc dd"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            config = LLMBenchmarkConfig.model_validate(raw_config)
            os.makedirs(os.path.join(output_dir, "activations"), exist_ok=True)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(eos_token_id=0),
            ):
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            self.assertFalse(result["reused"])

    def test_cache_stage_respects_non_eos_doc_separator(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["aa bb", "cc dd"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            raw_config["cache"]["doc_separator"] = "none"
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(eos_token_id=0),
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
            self.assertEqual(manifest["tokenization"]["doc_separator"], "none")
            self.assertIsNone(manifest["tokenization"]["doc_separator_id"])

    def test_cache_stage_records_eos_separator_and_tail_drop(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["aa bb", "cc dd"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            raw_config["cache"]["store_tokens"] = True
            raw_config["cache"]["max_sequences"] = None
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(eos_token_id=0),
            ):
                _ = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            manifest_path = os.path.join(
                output_dir, "activations", "activations_manifest.json"
            )
            with open(manifest_path, "r") as file:
                manifest = json.load(file)

            self.assertEqual(manifest["tokenization"]["doc_separator"], "eos")
            self.assertGreaterEqual(manifest["counts"]["tokens_dropped_tail"], 1)

            first_shard = manifest["shards"][0]["filename"]
            shard_payload = torch.load(
                os.path.join(output_dir, "activations", first_shard)
            )
            self.assertIn(0, shard_payload["tokens"].tolist())

    def test_cache_stage_respects_max_tokens_and_max_sequences(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(
                data_path,
                [
                    "one two three four five six",
                    "seven eight nine ten eleven twelve",
                ],
            )
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            raw_config["cache"]["max_tokens"] = 8
            raw_config["cache"]["max_sequences"] = 99
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(eos_token_id=0),
            ):
                _ = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            manifest_path = os.path.join(
                output_dir, "activations", "activations_manifest.json"
            )
            with open(manifest_path, "r") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["counts"]["tokens_emitted"], 8)
            self.assertEqual(manifest["counts"]["sequences_emitted"], 2)

    def test_cache_stage_reuse_and_overwrite_behavior(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta", "gamma delta", "epsilon zeta"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                first_result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )
                second_result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            self.assertFalse(first_result["reused"])
            self.assertTrue(second_result["reused"])

            # when / then
            modified_raw = _base_config(output_dir=output_dir, data_path=data_path)
            modified_raw["cache"]["max_seq_len"] = 8
            modified_config = LLMBenchmarkConfig.model_validate(modified_raw)
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=modified_config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

                overwrite_result = run_cache_stage(
                    config=modified_config,
                    output_dir=output_dir,
                    runner_overwrite=True,
                )

            self.assertFalse(overwrite_result["reused"])

    def test_cache_stage_rejects_schema_mismatch_without_recache_and_recaches_with_overwrite(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta", "gamma delta"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            config = LLMBenchmarkConfig.model_validate(raw_config)

            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                _ = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            manifest_path = os.path.join(
                output_dir, "activations", "activations_manifest.json"
            )
            with open(manifest_path, "r") as file:
                manifest = json.load(file)
            manifest["schema_version"] = int(ACTIVATIONS_MANIFEST_SCHEMA_VERSION) + 1
            with open(manifest_path, "w") as file:
                json.dump(manifest, file, indent=2)

            # when / then
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

                overwrite_result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=True,
                )

            self.assertFalse(overwrite_result["reused"])

    def test_cache_stage_handles_non_empty_activations_dir_without_manifest(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta gamma delta"])
            output_dir = os.path.join(tmpdir, "run")
            activations_dir = os.path.join(output_dir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            with open(os.path.join(activations_dir, "stale.pt"), "w") as file:
                file.write("stale")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=output_dir, data_path=data_path)
            )

            # when / then
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )
                result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=True,
                )
            self.assertFalse(result["reused"])

    def test_cache_stage_reuses_cache_when_only_llm_batch_size_changes(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(
                data_path,
                [
                    "alpha beta gamma delta",
                    "epsilon zeta eta theta",
                    "iota kappa lambda mu",
                ],
            )
            output_dir = os.path.join(tmpdir, "run")
            first_raw = _base_config(output_dir=output_dir, data_path=data_path)
            second_raw = _base_config(output_dir=output_dir, data_path=data_path)
            second_raw["cache"]["llm_batch_size"] = 3
            first_config = LLMBenchmarkConfig.model_validate(first_raw)
            second_config = LLMBenchmarkConfig.model_validate(second_raw)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                first_result = run_cache_stage(
                    config=first_config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )
                second_result = run_cache_stage(
                    config=second_config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            self.assertFalse(first_result["reused"])
            self.assertTrue(second_result["reused"])

    def test_cache_stage_passes_explicit_model_torch_dtype_to_hf_loader(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta gamma delta"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            raw_config["llm"]["torch_dtype"] = "bfloat16"
            config = LLMBenchmarkConfig.model_validate(raw_config)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ) as model_loader, patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                _ = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            model_loader_call = model_loader.call_args
            self.assertIsNotNone(model_loader_call)
            self.assertEqual(model_loader_call.kwargs["torch_dtype"], torch.bfloat16)

    def test_stored_indices_split_is_deterministic_across_cache_batch_sizes(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(
                data_path,
                [
                    "alpha beta gamma delta",
                    "epsilon zeta eta theta",
                    "iota kappa lambda mu",
                    "nu xi omicron pi",
                ],
            )

            output_dir_a = os.path.join(tmpdir, "run_a")
            output_dir_b = os.path.join(tmpdir, "run_b")

            raw_config_a = _base_config(output_dir=output_dir_a, data_path=data_path)
            raw_config_a["cache"]["split_method"] = "stored_indices"
            raw_config_a["cache"]["llm_batch_size"] = 1
            raw_config_a["training"] = {
                "train_steps": 10,
                "batch_size": 4,
                "max_train_samples": 128,
                "max_val_samples": 128,
                "max_test_samples": 128,
            }

            raw_config_b = _base_config(output_dir=output_dir_b, data_path=data_path)
            raw_config_b["cache"]["split_method"] = "stored_indices"
            raw_config_b["cache"]["llm_batch_size"] = 3
            raw_config_b["training"] = {
                "train_steps": 10,
                "batch_size": 4,
                "max_train_samples": 128,
                "max_val_samples": 128,
                "max_test_samples": 128,
            }

            config_a = LLMBenchmarkConfig.model_validate(raw_config_a)
            config_b = LLMBenchmarkConfig.model_validate(raw_config_b)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                _ = run_cache_stage(
                    config=config_a, output_dir=output_dir_a, runner_overwrite=False
                )
                _ = run_cache_stage(
                    config=config_b, output_dir=output_dir_b, runner_overwrite=False
                )

            # then
            split_a = torch.load(
                os.path.join(output_dir_a, "activations", "split_indices.pt")
            )
            split_b = torch.load(
                os.path.join(output_dir_b, "activations", "split_indices.pt")
            )
            self.assertTrue(torch.equal(split_a["train"], split_b["train"]))
            self.assertTrue(torch.equal(split_a["val"], split_b["val"]))
            self.assertTrue(torch.equal(split_a["test"], split_b["test"]))

    def test_stored_indices_guardrail_rejects_large_cache_without_limits(self) -> None:
        # given
        cache_config = ActivationCacheConfig.model_validate(
            {
                "split_method": "stored_indices",
                "train_fraction": 0.8,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            }
        )

        # when / then
        with self.assertRaises(ValueError):
            _ = _build_stored_indices(
                num_vectors_total=50_000_001,
                cache=cache_config,
                max_train_samples=None,
                max_val_samples=None,
                max_test_samples=None,
            )

    def test_stored_indices_respects_limits_and_early_break(self) -> None:
        # given
        cache_config = ActivationCacheConfig.model_validate(
            {
                "split_method": "stored_indices",
                "train_fraction": 0.34,
                "val_fraction": 0.33,
                "test_fraction": 0.33,
                "cache_seed": 9,
            }
        )

        # when
        split_indices = _build_stored_indices(
            num_vectors_total=1000,
            cache=cache_config,
            max_train_samples=3,
            max_val_samples=2,
            max_test_samples=2,
        )

        # then
        self.assertLessEqual(int(split_indices["train"].numel()), 3)
        self.assertLessEqual(int(split_indices["val"].numel()), 2)
        self.assertLessEqual(int(split_indices["test"].numel()), 2)
        total = (
            int(split_indices["train"].numel())
            + int(split_indices["val"].numel())
            + int(split_indices["test"].numel())
        )
        self.assertLessEqual(total, 7)

    def test_cache_stage_hf_backend_mocked_streaming_integration(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run")
            raw = {
                "output_dir": output_dir,
                "device": "cpu",
                "seed": [0],
                "llm": {"model_id": "dummy-model"},
                "data": {
                    "backend": "hf",
                    "dataset_id": "Geralt-Targaryen/openwebtext2",
                    "split": "train",
                    "streaming": True,
                },
                "hook": {"hook_name": "mlp_out", "layer": 0},
                "cache": {
                    "llm_batch_size": 2,
                    "max_seq_len": 4,
                    "max_sequences": 2,
                    "vectors_per_shard": 8,
                    "text_field": "text",
                },
                "variants": [
                    {
                        "name": "v",
                        "sae_class": "SAE",
                        "sign_aware": False,
                        "d_hidden": 4,
                        "sparsity_coeff": 1e-3,
                    }
                ],
            }
            config = LLMBenchmarkConfig.model_validate(raw)

            fake_stream = DummyHFStream(
                entries=[{"text": "first doc"}, {"text": "second doc"}],
                fingerprint="hf-fingerprint",
            )

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch.dict(
                "sys.modules",
                {
                    "datasets": types.SimpleNamespace(
                        load_dataset=lambda **_: fake_stream
                    )
                },
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
            self.assertEqual(manifest["dataset"]["backend"], "hf")
            self.assertEqual(
                manifest["dataset"]["dataset_fingerprint"], "hf-fingerprint"
            )

    def test_cache_stage_error_paths_for_recache_rules_and_tokenizer_constraints(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta gamma delta"])
            output_dir = os.path.join(tmpdir, "run")

            base_config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=output_dir, data_path=data_path)
            )
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                _ = run_cache_stage(
                    config=base_config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            manifest_path = os.path.join(
                output_dir, "activations", "activations_manifest.json"
            )
            with open(manifest_path, "r") as file:
                manifest = json.load(file)
            first_shard = manifest["shards"][0]["filename"]
            os.remove(os.path.join(output_dir, "activations", first_shard))

            # when / then
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=base_config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

            bad_tokenizer_config_raw = _base_config(
                output_dir=os.path.join(tmpdir, "run_tokenizer"), data_path=data_path
            )
            bad_tokenizer_config = LLMBenchmarkConfig.model_validate(
                bad_tokenizer_config_raw
            )
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizerNoPadOrEos(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=bad_tokenizer_config,
                        output_dir=os.path.join(tmpdir, "run_tokenizer"),
                        runner_overwrite=False,
                    )

    def test_cache_stage_warning_autocast_and_edge_runtime_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta gamma delta epsilon zeta"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            raw_config["cache"]["store_positions"] = True
            raw_config["cache"]["store_sequence_ids"] = True
            raw_config["cache"]["max_seq_len"] = 2
            raw_config["cache"]["max_sequences"] = 1
            raw_config["device"] = "auto"
            config = LLMBenchmarkConfig.model_validate(raw_config)
            original_tensor = torch.tensor

            def _cpu_tensor(*args: Any, **kwargs: Any) -> torch.Tensor:
                kwargs.pop("device", None)
                return original_tensor(*args, **kwargs)

            # when
            with warnings.catch_warnings(record=True) as caught_warnings:
                with patch(
                    "transformers.AutoModel.from_pretrained",
                    return_value=DummyModel(),
                ), patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=DummyTokenizerNoEosMultiToken(),
                ), patch(
                    "sgnsae.experiments.llm_benchmark.cache.torch.cuda.is_available",
                    return_value=True,
                ), patch(
                    "sgnsae.experiments.llm_benchmark.cache.torch.tensor",
                    side_effect=_cpu_tensor,
                ), patch(
                    "sgnsae.experiments.llm_benchmark.cache.torch.autocast",
                    side_effect=lambda **_: contextlib.nullcontext(),
                ) as autocast_mock:
                    _ = run_cache_stage(
                        config=config,
                        output_dir=output_dir,
                        runner_overwrite=False,
                    )

            # then
            self.assertGreaterEqual(len(caught_warnings), 1)
            warning_messages = [str(w.message) for w in caught_warnings]
            self.assertTrue(
                any(
                    "falling back to doc_separator='none'" in message
                    for message in warning_messages
                )
            )
            self.assertEqual(autocast_mock.call_count, 0)

    def test_cache_stage_enables_autocast_on_cuda_for_float16_compute(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta gamma delta epsilon zeta"])
            output_dir = os.path.join(tmpdir, "run")
            raw_config = _base_config(output_dir=output_dir, data_path=data_path)
            raw_config["cache"]["compute_dtype"] = "float16"
            raw_config["cache"]["max_seq_len"] = 2
            raw_config["cache"]["max_sequences"] = 1
            raw_config["device"] = "cuda:1"
            config = LLMBenchmarkConfig.model_validate(raw_config)
            original_tensor = torch.tensor

            def _cpu_tensor(*args: Any, **kwargs: Any) -> torch.Tensor:
                kwargs.pop("device", None)
                return original_tensor(*args, **kwargs)

            # when
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ), patch(
                "sgnsae.experiments.device.torch.cuda.is_available",
                return_value=True,
            ), patch(
                "sgnsae.experiments.device.torch.cuda.device_count",
                return_value=2,
            ), patch(
                "sgnsae.experiments.llm_benchmark.cache.torch.tensor",
                side_effect=_cpu_tensor,
            ), patch(
                "sgnsae.experiments.llm_benchmark.cache.torch.autocast",
                side_effect=lambda **_: contextlib.nullcontext(),
            ) as autocast_mock:
                _ = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=False,
                )

            # then
            self.assertGreaterEqual(autocast_mock.call_count, 1)
            autocast_call = autocast_mock.call_args
            self.assertIsNotNone(autocast_call)
            self.assertEqual(autocast_call.kwargs["device_type"], "cuda")
            self.assertEqual(autocast_call.kwargs["dtype"], torch.float16)

    def test_cache_stage_errors_for_hook_output_and_activation_invariants(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tiny.jsonl")
            _write_local_jsonl(data_path, ["alpha beta gamma delta epsilon zeta"])

            # when / then
            two_calls_config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=os.path.join(tmpdir, "two_calls"), data_path=data_path
                )
            )
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModelTwoHookCalls(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=two_calls_config,
                        output_dir=os.path.join(tmpdir, "two_calls"),
                        runner_overwrite=False,
                    )

            non_list_encode_config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=os.path.join(tmpdir, "bad_encode"), data_path=data_path
                )
            )
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizerBadEncode(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=non_list_encode_config,
                        output_dir=os.path.join(tmpdir, "bad_encode"),
                        runner_overwrite=False,
                    )

            zero_vectors_config_raw = _base_config(
                output_dir=os.path.join(tmpdir, "zero_vectors"), data_path=data_path
            )
            zero_vectors_config_raw["cache"]["max_seq_len"] = 1024
            zero_vectors_config_raw["cache"]["max_sequences"] = 1
            zero_vectors_config = LLMBenchmarkConfig.model_validate(
                zero_vectors_config_raw
            )
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=zero_vectors_config,
                        output_dir=os.path.join(tmpdir, "zero_vectors"),
                        runner_overwrite=False,
                    )
            self.assertFalse(
                os.path.isfile(
                    os.path.join(
                        tmpdir,
                        "zero_vectors",
                        "activations",
                        "activations_manifest.json",
                    )
                )
            )

            two_dimensional_config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=os.path.join(tmpdir, "two_dim"), data_path=data_path
                )
            )
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel2D(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                two_dim_result = run_cache_stage(
                    config=two_dimensional_config,
                    output_dir=os.path.join(tmpdir, "two_dim"),
                    runner_overwrite=False,
                )
            with open(two_dim_result["manifest_path"], "r") as file:
                two_dim_manifest = json.load(file)
            self.assertEqual(two_dim_manifest["activation"]["d_in"], 16)

            class DummyMLP1D(torch.nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x.mean(dim=(1, 2))

            class DummyLayer1D(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.mlp = DummyMLP1D()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.mlp(x)

            class DummyModel1D(DummyModel):
                def __init__(self, d_in: int = 16) -> None:
                    super().__init__(d_in=d_in, n_layers=1, tuple_output=False)
                    self.transformer.h[0] = DummyLayer1D()

            one_dim_config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=os.path.join(tmpdir, "one_dim"), data_path=data_path
                )
            )
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModel1D(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=one_dim_config,
                        output_dir=os.path.join(tmpdir, "one_dim"),
                        runner_overwrite=False,
                    )

            inconsistent_dim_raw = _base_config(
                output_dir=os.path.join(tmpdir, "inconsistent_dim"), data_path=data_path
            )
            inconsistent_dim_raw["cache"]["llm_batch_size"] = 1
            inconsistent_dim_raw["cache"]["max_seq_len"] = 2
            inconsistent_dim_config = LLMBenchmarkConfig.model_validate(
                inconsistent_dim_raw
            )
            with patch(
                "transformers.AutoModel.from_pretrained",
                return_value=DummyModelVaryingDim(),
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = run_cache_stage(
                        config=inconsistent_dim_config,
                        output_dir=os.path.join(tmpdir, "inconsistent_dim"),
                        runner_overwrite=False,
                    )
