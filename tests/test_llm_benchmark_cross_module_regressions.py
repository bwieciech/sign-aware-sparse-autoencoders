import json
import os
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import torch
from pydantic import ValidationError

from sgnsae.experiments.llm_benchmark import feature_ablation as feature_ablation_module
from sgnsae.experiments.llm_benchmark import loss_recovered as loss_recovered_module
from sgnsae.experiments.llm_benchmark import overhead as overhead_module
from sgnsae.experiments.llm_benchmark import plotting as plotting_module
from sgnsae.experiments.llm_benchmark import runner as runner_module
from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig
from sgnsae.experiments.llm_benchmark.metrics import (
    build_and_persist_comparison_summary,
    run_evaluation_for_subrun,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)
from sgnsae.models import SAE


class _TokenizerOK:
    def __init__(
        self,
        eos_token_id: Optional[int] = 7,
        eos_token: Optional[str] = "<eos>",
        pad_token_id: Optional[int] = 0,
        pad_token: Optional[str] = "<pad>",
    ) -> None:
        self.eos_token_id = eos_token_id
        self.eos_token = eos_token
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [int((ord(character) % 17) + 1) for character in str(text)]


class _TokenizerBadEncode(_TokenizerOK):
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del text
        del add_special_tokens
        return 1  # type: ignore[return-value]


class _TinyHookedLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 19, d_in: int = 6) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_in)
        self.hook_module = torch.nn.Linear(d_in, d_in, bias=False)
        self.lm_head = torch.nn.Linear(d_in, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> Dict[str, torch.Tensor]:
        del attention_mask
        del use_cache
        del return_dict
        hidden = self.embedding(input_ids)
        hidden = self.hook_module(hidden)
        return {"logits": self.lm_head(hidden)}


class _TinyLMTupleOutput(torch.nn.Module):
    def __init__(self, vocab_size: int = 11, d_in: int = 4) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_in)
        self.proj = torch.nn.Linear(d_in, vocab_size, bias=False)

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
        return (self.proj(self.embedding(input_ids)),)


class _ToySAE(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        include_latents: bool = True,
        latents_rank: int = 3,
        reconstruction_rank: int = 3,
        use_y_pred: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_hidden = int(d_hidden)
        self.W_dec = torch.nn.Parameter(
            torch.randn(1, int(d_in), int(d_hidden), dtype=torch.float32)
        )
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.include_latents = bool(include_latents)
        self.latents_rank = int(latents_rank)
        self.reconstruction_rank = int(reconstruction_rank)
        self.use_y_pred = bool(use_y_pred)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction: torch.Tensor
        if self.reconstruction_rank == 3:
            reconstruction = x.unsqueeze(0)
        elif self.reconstruction_rank == 2:
            reconstruction = x
        else:
            reconstruction = x[:, 0]
        payload: Dict[str, torch.Tensor] = {}
        if self.use_y_pred:
            payload["y_pred"] = reconstruction
        else:
            payload["x_hat"] = reconstruction
        if self.include_latents:
            if self.latents_rank == 3:
                payload["latents"] = torch.ones(
                    1, x.shape[0], self.d_hidden, dtype=torch.float32, device=x.device
                )
            elif self.latents_rank == 2:
                payload["latents"] = torch.ones(
                    x.shape[0], self.d_hidden, dtype=torch.float32, device=x.device
                )
            else:
                payload["latents"] = torch.ones(
                    self.d_hidden, dtype=torch.float32, device=x.device
                )
        return payload


class _FeatureModel(torch.nn.Module):
    def __init__(
        self, d_in: int, d_hidden: int, invalid_latent_rank: bool = False
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_hidden = int(d_hidden)
        self.invalid_latent_rank = bool(invalid_latent_rank)
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.W_dec = torch.nn.Parameter(
            torch.randn(1, int(d_in), int(d_hidden), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = int(x.shape[0])
        latents = torch.ones(1, batch_size, self.d_hidden, device=x.device)
        if self.invalid_latent_rank:
            latents = latents[0]
        return {"x_hat": x.unsqueeze(0), "latents": latents}


class _FakeCudaDevice(str):
    @property
    def type(self) -> str:
        return "cuda"


def _base_config(
    output_dir: str,
    data_files: Optional[List[str]] = None,
    backend: str = "local_jsonl",
) -> Dict[str, Any]:
    data_payload: Dict[str, Any]
    if backend == "hf":
        data_payload = {
            "backend": "hf",
            "dataset_id": "tiny-dataset",
            "split": "train",
            "text_field": "text",
        }
    else:
        data_payload = {
            "backend": "local_jsonl",
            "data_files": data_files or [],
            "text_field": "text",
        }
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": 0,
        "llm": {"model_id": "dummy-model"},
        "data": data_payload,
        "cache": {"max_seq_len": 8},
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "training": {"batch_size": 2, "mixed_precision": False},
        "evaluation": {
            "alive_threshold": 0.0,
            "alive_min_fraction": 0.0,
            "min_gamma_samples": 1,
            "gamma_denominator_epsilon": 1e-12,
            "r2_denominator_epsilon": 1e-12,
        },
        "variants": [
            {
                "name": "variant",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 4,
                "sparsity_coeff": 1e-3,
            }
        ],
    }


def _runner_config(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": 0,
        "llm": {"model_id": "sshleifer/tiny-gpt2"},
        "data": {
            "backend": "local_jsonl",
            "data_files": ["dummy.jsonl"],
            "text_field": "text",
        },
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "variants": [
            {
                "name": "baseline_relu",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": [1e-3, 1e-4],
            }
        ],
    }


def _write_local_data(data_path: str, rows: List[str]) -> None:
    with open(data_path, "w") as file:
        for row in rows:
            file.write(json.dumps({"text": row}) + "\n")


def _write_completed_training_metrics(subrun_dir: str) -> None:
    with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
        json.dump({"status": "completed"}, file, indent=2)


class TestLossRecoveredAdditionalCoverage(unittest.TestCase):
    def setUp(self) -> None:
        loss_recovered_module._MODEL_CACHE.clear()
        loss_recovered_module._TOKENS_CACHE.clear()

    def tearDown(self) -> None:
        loss_recovered_module._MODEL_CACHE.clear()
        loss_recovered_module._TOKENS_CACHE.clear()

    def test_path_and_dtype_helpers_cover_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            _write_local_data(data_path, ["alpha beta"])
            config_auto = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=tmpdir, data_files=[data_path])
            )
            config_f16_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_f16_payload["llm"]["torch_dtype"] = "float16"
            config_f16 = LLMBenchmarkConfig.model_validate(config_f16_payload)
            config_f32_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_f32_payload["llm"]["torch_dtype"] = "float32"
            config_f32 = LLMBenchmarkConfig.model_validate(config_f32_payload)
            config_bf16_payload = _base_config(
                output_dir=tmpdir, data_files=[data_path]
            )
            config_bf16_payload["llm"]["torch_dtype"] = "bfloat16"
            config_bf16 = LLMBenchmarkConfig.model_validate(config_bf16_payload)

            file_path = os.path.join(tmpdir, "temporary.txt")
            with open(file_path, "w") as file:
                file.write("temporary")

            # when
            run_dir = loss_recovered_module.run_level_loss_recovered_dir(tmpdir)
            run_paths = loss_recovered_module.run_level_loss_recovered_paths(tmpdir)
            subrun_path = loss_recovered_module.subrun_loss_recovered_path(
                os.path.join(tmpdir, "subrun")
            )
            loss_recovered_module._delete_file_if_exists(file_path)
            loss_recovered_module._delete_file_if_exists(file_path)

            # then
            self.assertTrue(run_dir.endswith("loss_recovered"))
            self.assertIn("eval_tokens", run_paths)
            self.assertTrue(subrun_path.endswith("loss_recovered.json"))
            self.assertFalse(os.path.isfile(file_path))
            self.assertEqual(
                loss_recovered_module._resolve_model_dtype(config_auto, "cpu"),
                torch.float32,
            )
            self.assertEqual(
                loss_recovered_module._resolve_model_dtype(config_auto, "cuda"),
                torch.float16,
            )
            self.assertEqual(
                loss_recovered_module._resolve_model_dtype(config_auto, "cuda:1"),
                torch.float16,
            )
            self.assertEqual(
                loss_recovered_module._resolve_model_dtype(config_f16, "cpu"),
                torch.float16,
            )
            self.assertEqual(
                loss_recovered_module._resolve_model_dtype(config_f32, "cpu"),
                torch.float32,
            )
            self.assertEqual(
                loss_recovered_module._resolve_model_dtype(config_bf16, "cpu"),
                torch.bfloat16,
            )

            with self.assertRaises(ValueError):
                _ = loss_recovered_module._resolve_model_dtype(
                    types.SimpleNamespace(llm=types.SimpleNamespace(torch_dtype="unknown")),  # type: ignore[arg-type]
                    "cpu",
                )

    def test_load_llm_model_and_tokenizer_use_cache_and_requested_ids(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            _write_local_data(data_path, ["alpha beta"])
            config_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_payload["llm"]["tokenizer_id"] = "tokenizer-id"
            config_payload["llm"]["tokenizer_revision"] = "tokenizer-revision"
            config = LLMBenchmarkConfig.model_validate(config_payload)

            class _FakeModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.anchor = torch.nn.Parameter(
                        torch.zeros(1, dtype=torch.float32)
                    )
                    self.config = types.SimpleNamespace(use_cache=True)
                    self.eval_calls = 0

                def eval(self) -> "_FakeModel":
                    self.eval_calls += 1
                    return self

                def to(self, device: str) -> "_FakeModel":
                    del device
                    return self

            fake_model = _FakeModel()

            # when
            with patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ) as from_pretrained, patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=object(),
            ) as tokenizer_loader:
                first_model = loss_recovered_module._load_llm_model(config, "cpu")
                second_model = loss_recovered_module._load_llm_model(config, "cpu")
                _ = loss_recovered_module._load_tokenizer(config)

            # then
            self.assertIs(first_model, second_model)
            self.assertEqual(from_pretrained.call_count, 1)
            self.assertFalse(bool(fake_model.config.use_cache))
            self.assertGreaterEqual(fake_model.eval_calls, 2)
            self.assertEqual(tokenizer_loader.call_count, 1)
            tokenizer_loader.assert_called_once_with(
                "tokenizer-id", revision="tokenizer-revision"
            )

    def test_load_llm_model_and_tokenizer_raise_helpful_error_without_transformers(
        self,
    ) -> None:
        # given
        config = LLMBenchmarkConfig.model_validate(
            _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])
        )
        original_import = __import__

        def _import_with_missing_transformers(
            name: str, *args: Any, **kwargs: Any
        ) -> Any:
            if name == "transformers":
                raise ModuleNotFoundError("No module named 'transformers'")
            return original_import(name, *args, **kwargs)

        imported_json_module = _import_with_missing_transformers("json")
        self.assertTrue(hasattr(imported_json_module, "loads"))

        # when
        with patch(
            "builtins.__import__",
            side_effect=_import_with_missing_transformers,
        ):
            # then
            with self.assertRaises(RuntimeError) as model_context:
                _ = loss_recovered_module._load_llm_model(config, "cpu")
            with self.assertRaises(RuntimeError) as tokenizer_context:
                _ = loss_recovered_module._load_tokenizer(config)

        self.assertIn("pip install transformers", str(model_context.exception))
        self.assertIn("pip install transformers", str(tokenizer_context.exception))

    def test_tiny_tuple_model_and_y_pred_sae_paths_are_exercised(self) -> None:
        # given
        tuple_model = _TinyLMTupleOutput(vocab_size=11, d_in=4)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        y_pred_sae = _ToySAE(d_in=4, d_hidden=3, use_y_pred=True)

        # when
        tuple_output = tuple_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        sae_output = y_pred_sae(torch.randn(2, 4, dtype=torch.float32))

        # then
        self.assertEqual(len(tuple_output), 1)
        self.assertIn("y_pred", sae_output)

    def test_loss_recovered_helpers_cover_additional_unhit_branches(self) -> None:
        # given
        class _NoFamily(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)

        # when / then
        self.assertIsNone(
            loss_recovered_module._resolve_layer_family_prefix(model=_NoFamily())
        )
        self.assertIsNone(
            loss_recovered_module._remap_module_path_to_layer_family(
                module_path="layers.0.mlp",
                resolved_family_prefix=None,
            )
        )
        self.assertEqual(
            loss_recovered_module._remap_module_path_to_layer_family(
                module_path="layers",
                resolved_family_prefix="model.layers",
            ),
            "model.layers",
        )
        self.assertIsNone(
            loss_recovered_module._remap_module_path_to_layer_family(
                module_path="unknown.path",
                resolved_family_prefix="model.layers",
            )
        )
        suggestions = loss_recovered_module._suggest_module_paths(
            model=_NoFamily(),
            attempted_path="linear",
        )
        self.assertTrue(isinstance(suggestions, list))

        class _CandidateFailureModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([torch.nn.Module()])

        with self.assertRaises(ValueError):
            _ = loss_recovered_module._resolve_hook_module(
                config=LLMBenchmarkConfig.model_validate(
                    _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])
                ),
                manifest={
                    "hook": {
                        "hook_name": "mlp_out",
                        "layer": 0,
                        "module_path_resolved": "layers.0.mlp",
                    }
                },
                llm_model=_CandidateFailureModel(),
            )

    def test_build_eval_tokens_adds_doc_separator_and_loss_for_mode_ablate_removes_hook(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            _write_local_data(data_path, ["alpha beta", "gamma delta"])
            config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=tmpdir, data_files=[data_path])
            )

            class _TokenizerWithEos:
                def __init__(self) -> None:
                    self.eos_token_id = 7
                    self.eos_token = "<eos>"
                    self.pad_token_id = 0
                    self.pad_token = "<pad>"

                def encode(
                    self, text: str, add_special_tokens: bool = False
                ) -> List[int]:
                    del add_special_tokens
                    del text
                    return [1, 2, 3, 4]

            with patch.object(
                loss_recovered_module,
                "_load_tokenizer",
                return_value=_TokenizerWithEos(),
            ), patch.object(
                loss_recovered_module,
                "_iter_text_documents",
                return_value=(iter(["first", "second"]), {}),
            ):
                token_payload, _ = loss_recovered_module._build_eval_tokens(
                    config=config
                )

            self.assertIn(7, token_payload["input_ids"].reshape(-1).tolist())

            class _TinyLM(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.embedding = torch.nn.Embedding(32, 8)
                    self.proj = torch.nn.Linear(8, 32)

                def forward(
                    self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    use_cache: bool = False,
                    return_dict: bool = True,
                ) -> Dict[str, torch.Tensor]:
                    del attention_mask
                    del use_cache
                    del return_dict
                    hidden = self.embedding(input_ids)
                    return {"logits": self.proj(hidden)}

            model = _TinyLM()
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            # when
            nll, token_count = loss_recovered_module._loss_for_mode(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_batch_size=1,
                hook_module=model.proj,
                mode="ablate",
            )

            # then
            self.assertGreaterEqual(float(nll), 0.0)
            self.assertEqual(int(token_count), 3)

    def test_hook_and_dataset_helpers_cover_edge_cases(self) -> None:
        # given
        local_payload = _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])
        config_local = LLMBenchmarkConfig.model_validate(local_payload)
        hf_payload = _base_config(output_dir="/tmp", backend="hf")
        config_hf = LLMBenchmarkConfig.model_validate(hf_payload)
        config_hf_missing_split = config_hf.model_copy(
            update={"data": config_hf.data.model_copy(update={"split": None})}
        )

        # when / then
        self.assertEqual(
            loss_recovered_module._resolve_hook_module_path(
                {"hook": {"module_path_resolved": "transformer.h.0.mlp"}}
            ),
            "transformer.h.0.mlp",
        )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._resolve_hook_module_path({"hook": {}})
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._resolve_hook_module_path({})

        class _LegacyPathModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gpt_neox = torch.nn.Module()
                layer = torch.nn.Module()
                layer.mlp = torch.nn.Linear(2, 2, bias=False)
                self.gpt_neox.layers = torch.nn.ModuleList([layer])

        legacy_model = _LegacyPathModel()
        resolved_module, resolved_path = loss_recovered_module._resolve_hook_module(
            config=config_local,
            manifest={
                "hook": {
                    "hook_name": "mlp_out",
                    "layer": 0,
                    "module_path_resolved": "layers.0.mlp",
                }
            },
            llm_model=legacy_model,
        )
        self.assertIs(resolved_module, legacy_model.gpt_neox.layers[0].mlp)
        self.assertEqual(resolved_path, "gpt_neox.layers.0.mlp")

        self.assertEqual(
            loss_recovered_module._resolve_hook_layer({"hook": {"layer": 3}}), 3
        )
        self.assertIsNone(
            loss_recovered_module._resolve_hook_layer({"hook": {"layer": True}})
        )
        self.assertIsNone(
            loss_recovered_module._resolve_hook_layer({"hook": {"layer": "3"}})
        )
        self.assertIsNone(loss_recovered_module._resolve_hook_layer({"hook": "bad"}))

        local_data, local_split = (
            loss_recovered_module._dataset_config_for_loss_recovered(config_local)
        )
        self.assertEqual(local_split, "train")
        self.assertEqual(local_data.backend, "local_jsonl")

        hf_data, hf_split = loss_recovered_module._dataset_config_for_loss_recovered(
            config_hf
        )
        self.assertEqual(hf_split, "train")
        self.assertEqual(hf_data.backend, "hf")
        self.assertEqual(hf_data.split, "train")

        with self.assertRaises(ValueError):
            _ = loss_recovered_module._dataset_config_for_loss_recovered(
                config_hf_missing_split
            )

    def test_build_eval_tokens_covers_success_and_errors(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            _write_local_data(data_path, ["alpha", "beta", "gamma"])
            config_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_payload["loss_recovered"] = {
                "enabled": True,
                "max_eval_sequences": 2,
                "llm_batch_size": 1,
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)

            documents = ["abcdefgh", "ijklmnop", "qrstuvwx"]
            metadata = {"dataset_id": "fixture", "text_field": "text"}

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._iter_text_documents",
                return_value=(documents, metadata),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokenizer",
                return_value=_TokenizerOK(eos_token_id=None, pad_token_id=None),
            ):
                tokens_payload, tokens_meta = loss_recovered_module._build_eval_tokens(
                    config
                )

            # then
            self.assertEqual(tuple(tokens_payload["input_ids"].shape), (2, 8))
            self.assertEqual(tuple(tokens_payload["attention_mask"].shape), (2, 8))
            self.assertEqual(tokens_meta["tokenization"]["doc_separator"], "none")
            self.assertTrue(tokens_meta["tokenization"]["pad_token_fallback_applied"])
            self.assertEqual(int(tokens_meta["generation"]["num_sequences"]), 2)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._iter_text_documents",
                return_value=(documents, metadata),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokenizer",
                return_value=_TokenizerBadEncode(),
            ):
                with self.assertRaises(ValueError):
                    _ = loss_recovered_module._build_eval_tokens(config)

            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._iter_text_documents",
                return_value=(["tiny"], metadata),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokenizer",
                return_value=_TokenizerOK(
                    eos_token_id=None, eos_token=None, pad_token_id=None, pad_token=None
                ),
            ):
                with self.assertRaises(ValueError):
                    _ = loss_recovered_module._build_eval_tokens(config)

            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._iter_text_documents",
                return_value=(["tiny"], metadata),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokenizer",
                return_value=_TokenizerOK(),
            ):
                with self.assertRaises(ValueError):
                    _ = loss_recovered_module._build_eval_tokens(config)

    def test_load_tokens_payload_covers_validation_cache_and_persistence(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            _write_local_data(data_path, ["alpha beta gamma"])
            config_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_payload["loss_recovered"] = {
                "enabled": True,
                "persist_tokens": True,
                "max_eval_sequences": 2,
                "llm_batch_size": 1,
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)
            run_paths = loss_recovered_module.run_level_loss_recovered_paths(tmpdir)
            os.makedirs(run_paths["directory"], exist_ok=True)

            # when / then
            torch.save([1, 2], run_paths["eval_tokens"])
            with self.assertRaises(ValueError):
                _ = loss_recovered_module._load_tokens_payload(config, tmpdir)

            torch.save({"input_ids": torch.ones(1, 2)}, run_paths["eval_tokens"])
            with self.assertRaises(ValueError):
                _ = loss_recovered_module._load_tokens_payload(config, tmpdir)

            torch.save(
                {"input_ids": 1, "attention_mask": torch.ones(1, 2)},
                run_paths["eval_tokens"],
            )
            with self.assertRaises(ValueError):
                _ = loss_recovered_module._load_tokens_payload(config, tmpdir)

            torch.save(
                {
                    "input_ids": torch.tensor([[1.1, 2.9]], dtype=torch.float32),
                    "attention_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                },
                run_paths["eval_tokens"],
            )
            loaded = loss_recovered_module._load_tokens_payload(config, tmpdir)
            self.assertEqual(loaded["input_ids"].dtype, torch.long)
            self.assertEqual(loaded["attention_mask"].dtype, torch.long)

            os.remove(run_paths["eval_tokens"])
            loss_recovered_module._TOKENS_CACHE[tmpdir] = {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            cached = loss_recovered_module._load_tokens_payload(config, tmpdir)
            self.assertTrue(
                torch.equal(
                    cached["input_ids"],
                    loss_recovered_module._TOKENS_CACHE[tmpdir]["input_ids"],
                )
            )
            self.assertIsNot(
                cached["input_ids"],
                loss_recovered_module._TOKENS_CACHE[tmpdir]["input_ids"],
            )

            del loss_recovered_module._TOKENS_CACHE[tmpdir]
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._build_eval_tokens",
                return_value=(
                    {
                        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                    },
                    {"meta": True},
                ),
            ):
                built = loss_recovered_module._load_tokens_payload(config, tmpdir)
            self.assertEqual(tuple(built["input_ids"].shape), (1, 2))
            self.assertTrue(os.path.isfile(run_paths["eval_tokens"]))
            self.assertTrue(os.path.isfile(run_paths["eval_tokens_meta"]))

            config_no_persist_payload = _base_config(
                output_dir=tmpdir, data_files=[data_path]
            )
            config_no_persist_payload["loss_recovered"] = {
                "enabled": True,
                "persist_tokens": False,
                "max_eval_sequences": 2,
                "llm_batch_size": 1,
            }
            config_no_persist = LLMBenchmarkConfig.model_validate(
                config_no_persist_payload
            )
            Path(run_paths["eval_tokens"]).unlink(missing_ok=True)
            loss_recovered_module._TOKENS_CACHE.clear()
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._build_eval_tokens",
                return_value=(
                    {
                        "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
                        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                    },
                    {"meta": True},
                ),
            ):
                _ = loss_recovered_module._load_tokens_payload(
                    config_no_persist, tmpdir
                )
            self.assertFalse(os.path.isfile(run_paths["eval_tokens"]))

    def test_tensor_container_helpers_cover_all_branches(self) -> None:
        # given
        logits = torch.randn(2, 3, 5, dtype=torch.float32)
        replacement = torch.ones(2, 3, 4, dtype=torch.float32)

        class _OutputWithLogits:
            def __init__(self, value: torch.Tensor) -> None:
                self.logits = value

        # when / then
        self.assertTrue(
            torch.equal(
                loss_recovered_module._extract_logits(_OutputWithLogits(logits)),
                logits,
            )
        )
        self.assertTrue(
            torch.equal(loss_recovered_module._extract_logits((logits,)), logits)
        )
        self.assertTrue(
            torch.equal(
                loss_recovered_module._extract_logits({"logits": logits}), logits
            )
        )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._extract_logits({"bad": logits})

        replaced_tensor = loss_recovered_module._replace_single_tensor_in_output(
            logits, replacement
        )
        self.assertTrue(torch.equal(replaced_tensor, replacement))
        replaced_tuple = loss_recovered_module._replace_single_tensor_in_output(
            (logits, "meta"), replacement
        )
        self.assertTrue(torch.equal(replaced_tuple[0], replacement))
        replaced_list = loss_recovered_module._replace_single_tensor_in_output(
            ["meta", logits], replacement
        )
        self.assertTrue(torch.equal(replaced_list[1], replacement))
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._replace_single_tensor_in_output(
                ("meta", "none"), replacement
            )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._replace_single_tensor_in_output(
                ["meta", "none"], replacement
            )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._replace_single_tensor_in_output(
                {"bad": "output"}, replacement
            )

        self.assertTrue(
            torch.equal(
                loss_recovered_module._extract_tensor_from_hook_output(logits), logits
            )
        )
        self.assertTrue(
            torch.equal(
                loss_recovered_module._extract_tensor_from_hook_output(
                    ("meta", logits)
                ),
                logits,
            )
        )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._extract_tensor_from_hook_output(("meta", "none"))

        self.assertTrue(
            torch.equal(
                loss_recovered_module._resolve_reconstruction_from_sae_output(
                    {"x_hat": logits}
                ),
                logits,
            )
        )
        self.assertTrue(
            torch.equal(
                loss_recovered_module._resolve_reconstruction_from_sae_output(
                    {"y_pred": logits}
                ),
                logits,
            )
        )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._resolve_reconstruction_from_sae_output({})

        hook_3d = torch.zeros(2, 3, 4, dtype=torch.float32)
        reconstructed_3d = torch.zeros(1, 6, 4, dtype=torch.float32)
        reshaped_3d = loss_recovered_module._reshape_reconstruction_to_hook_shape(
            reconstructed_3d, hook_3d
        )
        self.assertEqual(tuple(reshaped_3d.shape), (2, 3, 4))
        hook_2d = torch.zeros(6, 4, dtype=torch.float32)
        reshaped_2d = loss_recovered_module._reshape_reconstruction_to_hook_shape(
            torch.zeros(6, 4), hook_2d
        )
        self.assertEqual(tuple(reshaped_2d.shape), (6, 4))
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._reshape_reconstruction_to_hook_shape(
                torch.zeros(2, 2), torch.zeros(2, 2, 2, 2)
            )

    def test_build_reconstructed_hook_tensor_covers_validation_paths(self) -> None:
        # given
        hook_tensor = torch.randn(2, 3, 4, dtype=torch.float32)
        sae = _ToySAE(d_in=4, d_hidden=5)

        # when
        reconstructed = loss_recovered_module._build_reconstructed_hook_tensor(
            hook_tensor=hook_tensor,
            sae_model=sae,
            ablated_feature_indices=[0],
        )

        # then
        self.assertEqual(tuple(reconstructed.shape), tuple(hook_tensor.shape))

        # when / then
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._build_reconstructed_hook_tensor(
                torch.zeros(1, 2, 3, 4),
                sae_model=sae,
            )

        with self.assertRaises(ValueError):
            _ = loss_recovered_module._build_reconstructed_hook_tensor(
                hook_tensor=hook_tensor,
                sae_model=_ToySAE(d_in=3, d_hidden=4),
            )

        with self.assertRaises(ValueError):
            _ = loss_recovered_module._build_reconstructed_hook_tensor(
                hook_tensor=hook_tensor,
                sae_model=_ToySAE(d_in=4, d_hidden=4, include_latents=False),
                ablated_feature_indices=[0],
            )

        with self.assertRaises(ValueError):
            _ = loss_recovered_module._build_reconstructed_hook_tensor(
                hook_tensor=hook_tensor,
                sae_model=_ToySAE(d_in=4, d_hidden=4, latents_rank=1),
                ablated_feature_indices=[0],
            )

        with self.assertRaises(ValueError):
            _ = loss_recovered_module._build_reconstructed_hook_tensor(
                hook_tensor=hook_tensor,
                sae_model=_ToySAE(d_in=4, d_hidden=4, reconstruction_rank=1),
                ablated_feature_indices=[0],
            )

        with self.assertRaises(ValueError):
            _ = loss_recovered_module._build_reconstructed_hook_tensor(
                hook_tensor=hook_tensor,
                sae_model=sae,
                ablated_feature_indices=[True],  # type: ignore[list-item]
            )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._build_reconstructed_hook_tensor(
                hook_tensor=hook_tensor,
                sae_model=sae,
                ablated_feature_indices=[-1],
            )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._build_reconstructed_hook_tensor(
                hook_tensor=hook_tensor,
                sae_model=sae,
                ablated_feature_indices=[100],
            )

        hook_tensor_2d = torch.randn(6, 4, dtype=torch.float32)
        reconstructed_2d = loss_recovered_module._build_reconstructed_hook_tensor(
            hook_tensor=hook_tensor_2d,
            sae_model=_ToySAE(
                d_in=4,
                d_hidden=5,
                latents_rank=2,
                reconstruction_rank=2,
            ),
            ablated_feature_indices=[0],
        )
        self.assertEqual(tuple(reconstructed_2d.shape), (6, 4))

    def test_loss_for_mode_covers_validation_autocast_and_zero_tokens(self) -> None:
        # given
        model = _TinyHookedLM(vocab_size=13, d_in=4)
        input_ids = torch.randint(0, 13, (2, 4), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # when / then
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._loss_for_mode(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_batch_size=2,
                mode="invalid",
            )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._loss_for_mode(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_batch_size=2,
                mode="ablate",
            )
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._loss_for_mode(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_batch_size=2,
                mode="reconstruct",
                hook_module=model.hook_module,
            )

        zero_mask = torch.zeros_like(input_ids)
        with self.assertRaises(ValueError):
            _ = loss_recovered_module._loss_for_mode(
                model=model,
                input_ids=input_ids,
                attention_mask=zero_mask,
                llm_batch_size=2,
                mode="none",
            )

        reconstruct_sae = _ToySAE(
            d_in=4,
            d_hidden=6,
            latents_rank=3,
            reconstruction_rank=3,
        )
        reconstructed_nll, reconstructed_token_count = (
            loss_recovered_module._loss_for_mode(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_batch_size=2,
                hook_module=model.hook_module,
                mode="reconstruct",
                sae_model=reconstruct_sae,
            )
        )
        self.assertGreaterEqual(float(reconstructed_nll), 0.0)
        self.assertEqual(int(reconstructed_token_count), 6)

        fake_device = _FakeCudaDevice("cpu")
        fake_device_param = types.SimpleNamespace(device=fake_device)
        fake_dtype_param = types.SimpleNamespace(dtype=torch.float16)
        with patch(
            "sgnsae.experiments.llm_benchmark.loss_recovered.next",
            side_effect=[fake_device_param, fake_dtype_param],
        ), patch(
            "sgnsae.experiments.llm_benchmark.loss_recovered.torch.autocast",
            return_value=nullcontext(),
        ) as autocast_context:
            _ = loss_recovered_module._loss_for_mode(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_batch_size=2,
                mode="none",
            )
        self.assertEqual(autocast_context.call_count, 1)

    def test_run_level_artifacts_and_subrun_payload_cover_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            _write_local_data(data_path, ["alpha beta gamma delta"])
            config_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_payload["loss_recovered"] = {
                "enabled": True,
                "max_eval_sequences": 2,
                "llm_batch_size": 1,
                "persist_tokens": True,
                "denominator_epsilon": 1e-12,
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)
            run_paths = loss_recovered_module.run_level_loss_recovered_paths(tmpdir)
            os.makedirs(run_paths["directory"], exist_ok=True)

            # when
            disabled_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            disabled_config = LLMBenchmarkConfig.model_validate(disabled_payload)
            disabled = loss_recovered_module.ensure_run_level_loss_recovered_artifacts(
                config=disabled_config,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            self.assertEqual(disabled, {"enabled": False})

            # given
            with open(run_paths["eval_tokens_meta"], "w") as file:
                json.dump({"meta": True}, file)
            with open(run_paths["baselines"], "w") as file:
                json.dump({"loss_original": 1.0, "loss_ablated": 2.0}, file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._build_eval_tokens",
                return_value=(
                    {
                        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                    },
                    {"meta": True},
                ),
            ):
                reused = (
                    loss_recovered_module.ensure_run_level_loss_recovered_artifacts(
                        config=config,
                        output_dir=tmpdir,
                        overwrite=False,
                    )
                )

            # then
            self.assertTrue(bool(reused["reused"]))
            self.assertTrue(os.path.isfile(run_paths["eval_tokens"]))

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._build_eval_tokens",
                return_value=(
                    {
                        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                    },
                    {"meta": True},
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.resolve_runtime_device",
                return_value="cpu",
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_llm_model",
                return_value=_TinyHookedLM(vocab_size=13, d_in=4),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.resolve_manifest_path",
                return_value=os.path.join(
                    tmpdir, "activations", "activations_manifest.json"
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.load_activation_manifest",
                return_value={
                    "hook": {"module_path_resolved": "hook_module", "layer": 0}
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._loss_for_mode",
                side_effect=[(1.0, 20), (2.0, 20)],
            ):
                rebuilt = (
                    loss_recovered_module.ensure_run_level_loss_recovered_artifacts(
                        config=config,
                        output_dir=tmpdir,
                        overwrite=True,
                    )
                )

            # then
            self.assertFalse(bool(rebuilt["reused"]))
            with open(run_paths["baselines"], "r") as file:
                baselines = json.load(file)
            self.assertEqual(float(baselines["loss_original"]), 1.0)
            self.assertEqual(float(baselines["loss_ablated"]), 2.0)

            # when / then
            with self.assertRaises(ValueError):
                _ = loss_recovered_module.compute_reconstructed_loss(
                    config=disabled_config,
                    output_dir=tmpdir,
                    sae_model=_ToySAE(d_in=4, d_hidden=4),
                )

            llm_meta = _TinyHookedLM(vocab_size=11, d_in=4).to(torch.device("meta"))
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokens_payload",
                return_value={
                    "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.resolve_runtime_device",
                return_value="cpu",
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_llm_model",
                return_value=llm_meta,
            ):
                with self.assertRaises(ValueError):
                    _ = loss_recovered_module.compute_reconstructed_loss(
                        config=config,
                        output_dir=tmpdir,
                        sae_model=_ToySAE(d_in=4, d_hidden=4),
                    )

            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokens_payload",
                return_value={
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.resolve_runtime_device",
                return_value="cpu",
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_llm_model",
                return_value=_TinyHookedLM(vocab_size=11, d_in=4),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.resolve_manifest_path",
                return_value=os.path.join(
                    tmpdir, "activations", "activations_manifest.json"
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.load_activation_manifest",
                return_value={
                    "hook": {"module_path_resolved": "hook_module", "layer": 0}
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._loss_for_mode",
                return_value=(1.5, 10),
            ):
                reconstructed_loss, token_count = (
                    loss_recovered_module.compute_reconstructed_loss(
                        config=config,
                        output_dir=tmpdir,
                        sae_model=_ToySAE(d_in=4, d_hidden=4),
                    )
                )
            self.assertEqual(float(reconstructed_loss), 1.5)
            self.assertEqual(int(token_count), 10)

            subrun = {
                "variant_name": "variant",
                "variant_slug": "variant",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
            }
            subrun_dir = os.path.join(
                tmpdir, "models", "variant", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            output_path = loss_recovered_module.subrun_loss_recovered_path(subrun_dir)
            with open(output_path, "w") as file:
                json.dump({"cached": True}, file)
            cached_payload = (
                loss_recovered_module.compute_and_write_subrun_loss_recovered(
                    config=config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    subrun_dir=subrun_dir,
                    sae_model=_ToySAE(d_in=4, d_hidden=4),
                    overwrite=False,
                )
            )
            self.assertEqual(cached_payload, {"cached": True})

            os.remove(output_path)
            Path(run_paths["baselines"]).unlink(missing_ok=True)
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.ensure_run_level_loss_recovered_artifacts",
                return_value={"enabled": True},
            ):
                with self.assertRaises(ValueError):
                    _ = loss_recovered_module.compute_and_write_subrun_loss_recovered(
                        config=config,
                        subrun=subrun,
                        output_dir=tmpdir,
                        subrun_dir=subrun_dir,
                        sae_model=_ToySAE(d_in=4, d_hidden=4),
                        overwrite=True,
                    )

            with open(run_paths["baselines"], "w") as file:
                json.dump({"loss_original": 1.0, "loss_ablated": 1.0 + 1e-14}, file)
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokens_payload",
                return_value={
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.compute_reconstructed_loss",
                return_value=(1.0, 3),
            ):
                warning_payload = (
                    loss_recovered_module.compute_and_write_subrun_loss_recovered(
                        config=config,
                        subrun=subrun,
                        output_dir=tmpdir,
                        subrun_dir=subrun_dir,
                        sae_model=_ToySAE(d_in=4, d_hidden=4),
                        overwrite=True,
                    )
                )
            self.assertIsNone(warning_payload["loss_recovered"])
            self.assertEqual(len(warning_payload["warnings"]), 1)

            with open(run_paths["baselines"], "w") as file:
                json.dump({"loss_original": 1.0, "loss_ablated": 2.0}, file)
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokens_payload",
                return_value={
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.compute_reconstructed_loss",
                return_value=(1.5, 3),
            ):
                payload = loss_recovered_module.compute_and_write_subrun_loss_recovered(
                    config=config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    subrun_dir=subrun_dir,
                    sae_model=_ToySAE(d_in=4, d_hidden=4),
                    overwrite=True,
                )
            self.assertAlmostEqual(float(payload["loss_recovered"]), 0.5, places=8)

            self.assertIsNone(
                loss_recovered_module.compute_and_write_subrun_loss_recovered(
                    config=disabled_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    subrun_dir=subrun_dir,
                    sae_model=_ToySAE(d_in=4, d_hidden=4),
                    overwrite=True,
                )
            )

    def test_loss_recovered_remaining_branches_are_exercised(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            _write_local_data(data_path, ["alpha beta"])
            config_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_payload["loss_recovered"] = {
                "enabled": True,
                "max_eval_sequences": 1,
                "llm_batch_size": 1,
                "persist_tokens": True,
                "denominator_epsilon": 1e-12,
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)

            class _ModelWithoutConfig(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.anchor = torch.nn.Parameter(
                        torch.zeros(1, dtype=torch.float32)
                    )

                def eval(self) -> "_ModelWithoutConfig":
                    return self

                def to(self, device: str) -> "_ModelWithoutConfig":
                    del device
                    return self

            model_without_config = _ModelWithoutConfig()
            loss_recovered_module._MODEL_CACHE.clear()
            with patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=model_without_config,
            ):
                loaded = loss_recovered_module._load_llm_model(
                    config=config, runtime_device="cpu"
                )
            self.assertIs(loaded, model_without_config)

            class _WrongLayerFamilyModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.transformer = torch.nn.Module()
                    self.transformer.h = torch.nn.Linear(2, 2, bias=False)

            wrong_layer_family_model = _WrongLayerFamilyModel()
            self.assertIsNone(
                loss_recovered_module._resolve_layer_family_prefix(
                    model=wrong_layer_family_model
                )
            )

            with patch.object(
                loss_recovered_module,
                "_resolve_hook_module_path",
                return_value="model.layers.0.mlp",
            ):
                with self.assertRaises(ValueError):
                    _ = loss_recovered_module._resolve_hook_module(
                        config=config,
                        manifest={"hook": "bad"},
                        llm_model=wrong_layer_family_model,
                    )

            config_with_dataset_split = config.model_copy(
                update={
                    "loss_recovered": config.loss_recovered.model_copy(
                        update={"dataset_split": "validation"}
                    )
                }
            )
            _, resolved_split = (
                loss_recovered_module._dataset_config_for_loss_recovered(
                    config=config_with_dataset_split
                )
            )
            self.assertEqual(str(resolved_split), "validation")

            no_separator_config = config.model_copy(
                update={
                    "cache": config.cache.model_copy(
                        update={"doc_separator": "none", "max_seq_len": 4}
                    )
                }
            )
            with patch.object(
                loss_recovered_module,
                "_load_tokenizer",
                return_value=_TokenizerOK(),
            ), patch.object(
                loss_recovered_module,
                "_iter_text_documents",
                return_value=(iter(["alpha"]), {}),
            ):
                _, token_meta = loss_recovered_module._build_eval_tokens(
                    config=no_separator_config
                )
            self.assertEqual(str(token_meta["tokenization"]["doc_separator"]), "none")

            with self.assertRaises(ValueError):
                _ = loss_recovered_module._extract_logits(("bad",))
            with self.assertRaises(ValueError):
                _ = loss_recovered_module._extract_logits(object())
            with self.assertRaises(ValueError):
                _ = loss_recovered_module._extract_tensor_from_hook_output(
                    {"bad": "value"}
                )

            model = _TinyHookedLM(vocab_size=13, d_in=4)
            input_ids = torch.randint(0, 13, (1, 4), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            fake_device = _FakeCudaDevice("cpu")
            fake_device_param = types.SimpleNamespace(device=fake_device)
            fake_dtype_param = types.SimpleNamespace(dtype=torch.float32)
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.next",
                side_effect=[fake_device_param, fake_dtype_param],
            ), patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered.torch.autocast",
                return_value=nullcontext(),
            ) as autocast_context:
                _ = loss_recovered_module._loss_for_mode(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    llm_batch_size=1,
                    mode="none",
                )
            self.assertEqual(int(autocast_context.call_count), 0)

            run_paths = loss_recovered_module.run_level_loss_recovered_paths(tmpdir)
            os.makedirs(run_paths["directory"], exist_ok=True)
            with open(run_paths["eval_tokens_meta"], "w") as file:
                json.dump({"meta": True}, file)
            with open(run_paths["baselines"], "w") as file:
                json.dump({"loss_original": 1.0, "loss_ablated": 2.0}, file)
            Path(run_paths["eval_tokens"]).unlink(missing_ok=True)
            loss_recovered_module._TOKENS_CACHE.pop(tmpdir, None)
            token_payload = {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            with patch.object(
                loss_recovered_module,
                "_build_eval_tokens",
                return_value=(token_payload, {"meta": True}),
            ):
                reused = (
                    loss_recovered_module.ensure_run_level_loss_recovered_artifacts(
                        config=config,
                        output_dir=tmpdir,
                        overwrite=False,
                    )
                )
            self.assertTrue(bool(reused["reused"]))
            self.assertTrue(os.path.isfile(run_paths["eval_tokens"]))

            Path(run_paths["eval_tokens"]).unlink(missing_ok=True)
            config_reused_without_persist = config.model_copy(
                update={
                    "loss_recovered": config.loss_recovered.model_copy(
                        update={"persist_tokens": False}
                    )
                }
            )
            with patch.object(
                loss_recovered_module,
                "_build_eval_tokens",
            ) as build_eval_tokens_call:
                reused_without_persist = (
                    loss_recovered_module.ensure_run_level_loss_recovered_artifacts(
                        config=config_reused_without_persist,
                        output_dir=tmpdir,
                        overwrite=False,
                    )
                )
            self.assertTrue(bool(reused_without_persist["reused"]))
            self.assertEqual(int(build_eval_tokens_call.call_count), 0)

            config_without_persist = config.model_copy(
                update={
                    "loss_recovered": config.loss_recovered.model_copy(
                        update={"persist_tokens": False}
                    )
                }
            )
            loss_recovered_module._TOKENS_CACHE.pop(tmpdir, None)
            with patch.object(
                loss_recovered_module,
                "_build_eval_tokens",
                return_value=(token_payload, {"meta": True}),
            ), patch.object(
                loss_recovered_module,
                "resolve_runtime_device",
                return_value="cpu",
            ), patch.object(
                loss_recovered_module,
                "_load_llm_model",
                return_value=_TinyHookedLM(vocab_size=13, d_in=4),
            ), patch.object(
                loss_recovered_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "activations_manifest.json"),
            ), patch.object(
                loss_recovered_module,
                "load_activation_manifest",
                return_value={
                    "hook": {"module_path_resolved": "hook_module", "layer": 0}
                },
            ), patch.object(
                loss_recovered_module,
                "_loss_for_mode",
                side_effect=[(1.0, 3), (2.0, 3)],
            ):
                fresh = loss_recovered_module.ensure_run_level_loss_recovered_artifacts(
                    config=config_without_persist,
                    output_dir=tmpdir,
                    overwrite=True,
                )

            # then
            self.assertFalse(bool(fresh["reused"]))
            self.assertFalse(os.path.isfile(run_paths["eval_tokens"]))


class TestFeatureAblationAdditionalCoverage(unittest.TestCase):
    def test_feature_ablation_helpers_cover_selection_and_statistics_paths(
        self,
    ) -> None:
        # given
        config_payload = _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])
        config_payload["feature_ablation"] = {
            "enabled": True,
            "num_features": 2,
            "selection": "random_alive",
            "max_eval_batches": 1,
            "seed": 7,
        }
        config = LLMBenchmarkConfig.model_validate(config_payload)
        model = _FeatureModel(d_in=4, d_hidden=3)
        subrun = {"seed": 0}

        # when / then
        self.assertEqual(
            feature_ablation_module._summary_statistics([]),
            {
                "mean": None,
                "median": None,
                "p10": None,
                "p90": None,
                "num_valid": 0,
            },
        )
        self.assertEqual(
            feature_ablation_module._summary_statistics([0.5, None, 1.5])["num_valid"],
            2,
        )
        resolved = feature_ablation_module._resolve_reconstruction_output(
            {"y_pred": torch.ones(1, 2, 4)}
        )
        self.assertEqual(tuple(resolved.shape), (1, 2, 4))
        with self.assertRaises(ValueError):
            _ = feature_ablation_module._resolve_reconstruction_output({})

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([], "manifest.json"),
        ):
            with self.assertRaises(ValueError):
                _ = feature_ablation_module._select_features_for_ablation(
                    config=config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                )

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.zeros(2, 4)], "manifest.json"),
        ):
            selected, candidate_count, samples = (
                feature_ablation_module._select_features_for_ablation(
                    config=config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                )
            )
        self.assertEqual(len(selected), 2)
        self.assertGreaterEqual(int(candidate_count), 2)
        self.assertEqual(int(samples), 2)

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.zeros(1, 4), torch.zeros(1, 4)], "manifest.json"),
        ):
            _, _, limited_samples = (
                feature_ablation_module._select_features_for_ablation(
                    config=config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                )
            )
        self.assertEqual(int(limited_samples), 1)

        config_no_alive_payload = _base_config(
            output_dir="/tmp", data_files=["dummy.jsonl"]
        )
        config_no_alive_payload["evaluation"]["alive_threshold"] = 1000.0
        config_no_alive_payload["evaluation"]["alive_min_fraction"] = 0.5
        config_no_alive_payload["feature_ablation"] = {
            "enabled": True,
            "num_features": 2,
            "selection": "top_mean_abs",
            "max_eval_batches": 1,
            "seed": 0,
        }
        config_no_alive = LLMBenchmarkConfig.model_validate(config_no_alive_payload)
        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.ones(2, 4)], "manifest.json"),
        ):
            selected_none, candidate_count_none, _ = (
                feature_ablation_module._select_features_for_ablation(
                    config=config_no_alive,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                )
            )
        self.assertEqual(selected_none, [])
        self.assertEqual(int(candidate_count_none), 0)

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.ones(2, 4)], "manifest.json"),
        ):
            with self.assertRaises(ValueError):
                _ = feature_ablation_module._select_features_for_ablation(
                    config=config_no_alive,
                    model=_FeatureModel(d_in=4, d_hidden=3, invalid_latent_rank=True),
                    subrun=subrun,
                    output_dir="/tmp",
                )

    def test_feature_ablation_selection_ordering_is_deterministic(self) -> None:
        # given
        class _OrderedFeatureModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.d_hidden = 5
                self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                latents = torch.tensor(
                    [[[0.0, 8.0, 1.0, 7.0, 9.0], [0.0, 8.0, 1.0, 7.0, 9.0]]],
                    dtype=torch.float32,
                    device=x.device,
                )
                return {"x_hat": x.unsqueeze(0), "latents": latents}

        model = _OrderedFeatureModel()
        top_payload = _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])
        top_payload["feature_ablation"] = {
            "enabled": True,
            "num_features": 3,
            "selection": "top_mean_abs",
            "max_eval_batches": 1,
            "seed": 0,
        }
        top_config = LLMBenchmarkConfig.model_validate(top_payload)
        random_payload = _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])
        random_payload["feature_ablation"] = {
            "enabled": True,
            "num_features": 3,
            "selection": "random_alive",
            "max_eval_batches": 1,
            "seed": 19,
        }
        random_config = LLMBenchmarkConfig.model_validate(random_payload)
        subrun = {"seed": 0}

        # when
        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.zeros(2, 4)], "manifest.json"),
        ):
            top_selected, top_candidate_count, top_samples = (
                feature_ablation_module._select_features_for_ablation(
                    config=top_config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                )
            )
            random_selected, random_candidate_count, random_samples = (
                feature_ablation_module._select_features_for_ablation(
                    config=random_config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                )
            )

        # then
        self.assertEqual(top_selected, [4, 1, 3])
        self.assertEqual(int(top_candidate_count), 5)
        self.assertEqual(int(top_samples), 2)
        expected_generator = torch.Generator()
        expected_generator.manual_seed(19)
        expected_random = torch.randperm(5, generator=expected_generator)[:3].tolist()
        self.assertEqual(random_selected, expected_random)
        self.assertEqual(int(random_candidate_count), 5)
        self.assertEqual(int(random_samples), 2)

    def test_feature_ablation_top_mean_abs_tie_boundary_prefers_lower_indices(
        self,
    ) -> None:
        # given
        class _TieBoundaryFeatureModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.d_hidden = 4
                self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                latents = torch.ones(
                    1,
                    int(x.shape[0]),
                    self.d_hidden,
                    dtype=torch.float32,
                    device=x.device,
                )
                return {"x_hat": x.unsqueeze(0), "latents": latents}

        model = _TieBoundaryFeatureModel()
        config_payload = _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])
        config_payload["feature_ablation"] = {
            "enabled": True,
            "num_features": 2,
            "selection": "top_mean_abs",
            "max_eval_batches": 1,
            "seed": 0,
        }
        config = LLMBenchmarkConfig.model_validate(config_payload)
        subrun = {"seed": 0}

        # when
        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.zeros(3, 4)], "manifest.json"),
        ):
            selected, candidate_count, samples = (
                feature_ablation_module._select_features_for_ablation(
                    config=config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                )
            )

        # then
        self.assertEqual(selected, [0, 1])
        self.assertEqual(int(candidate_count), 4)
        self.assertEqual(int(samples), 3)

    def test_feature_ablation_metric_helpers_cover_branches(self) -> None:
        # given
        config_payload = _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])
        config_payload["feature_ablation"] = {
            "enabled": True,
            "num_features": 3,
            "selection": "top_mean_abs",
            "max_eval_batches": 1,
            "seed": 0,
            "metric": "delta_lm_loss",
            "lm_max_features": 2,
        }
        config_payload["loss_recovered"] = {"enabled": True}
        config = LLMBenchmarkConfig.model_validate(config_payload)
        model = _FeatureModel(d_in=4, d_hidden=3)
        subrun = {"seed": 0}

        # when / then
        impacts_empty, baseline_empty, count_empty = (
            feature_ablation_module._compute_delta_mse_impacts(
                config=config,
                model=model,
                subrun=subrun,
                output_dir="/tmp",
                selected_features=[],
            )
        )
        self.assertEqual(impacts_empty, [])
        self.assertEqual(float(baseline_empty), 0.0)
        self.assertEqual(int(count_empty), 0)

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([], "manifest.json"),
        ):
            with self.assertRaises(ValueError):
                _ = feature_ablation_module._compute_delta_mse_impacts(
                    config=config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                    selected_features=[0],
                )

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.ones(2, 4)], "manifest.json"),
        ):
            with self.assertRaises(ValueError):
                _ = feature_ablation_module._compute_delta_mse_impacts(
                    config=config,
                    model=_FeatureModel(d_in=4, d_hidden=3, invalid_latent_rank=True),
                    subrun=subrun,
                    output_dir="/tmp",
                    selected_features=[0],
                )

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.ones(2, 4)], "manifest.json"),
        ):
            impacts, baseline, count = (
                feature_ablation_module._compute_delta_mse_impacts(
                    config=config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                    selected_features=[0, 1],
                )
            )
        self.assertEqual(len(impacts), 2)
        self.assertGreaterEqual(float(baseline), 0.0)
        self.assertGreater(int(count), 0)

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation._iter_val_batches",
            return_value=([torch.ones(1, 4), torch.ones(1, 4)], "manifest.json"),
        ):
            limited_impacts, _, limited_count = (
                feature_ablation_module._compute_delta_mse_impacts(
                    config=config,
                    model=model,
                    subrun=subrun,
                    output_dir="/tmp",
                    selected_features=[0],
                )
            )
        self.assertEqual(len(limited_impacts), 1)
        self.assertEqual(int(limited_count), 4)

        self.assertEqual(
            feature_ablation_module._compute_delta_lm_loss_impacts(
                config=config,
                output_dir="/tmp",
                model=model,
                selected_features=[],
            ),
            ([], 0),
        )

        with patch(
            "sgnsae.experiments.llm_benchmark.feature_ablation.compute_reconstructed_loss",
            side_effect=[(2.0, 5), (2.4, 5), (2.6, 5)],
        ):
            lm_impacts, lm_count = (
                feature_ablation_module._compute_delta_lm_loss_impacts(
                    config=config,
                    output_dir="/tmp",
                    model=model,
                    selected_features=[0, 1, 2],
                )
            )
        self.assertEqual(lm_impacts, [0.3999999999999999, 0.6000000000000001, None])
        self.assertEqual(int(lm_count), 2)

    def test_compute_and_write_feature_ablation_covers_disabled_reuse_and_lm_metric(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun = {
                "variant_name": "variant",
                "variant_slug": "variant",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
            }
            subrun_dir = os.path.join(
                tmpdir, "models", "variant", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            model = _FeatureModel(d_in=4, d_hidden=4)

            disabled_config = LLMBenchmarkConfig.model_validate(
                _base_config(output_dir=tmpdir, data_files=["dummy.jsonl"])
            )

            # when / then
            self.assertIsNone(
                feature_ablation_module.compute_and_write_feature_ablation(
                    config=disabled_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    subrun_dir=subrun_dir,
                    model=model,
                    overwrite=True,
                )
            )

            output_path = feature_ablation_module.subrun_feature_ablation_path(
                subrun_dir
            )
            with open(output_path, "w") as file:
                json.dump({"cached": True}, file)
            enabled_payload = _base_config(
                output_dir=tmpdir, data_files=["dummy.jsonl"]
            )
            enabled_payload["feature_ablation"] = {"enabled": True}
            enabled_config = LLMBenchmarkConfig.model_validate(enabled_payload)
            cached = feature_ablation_module.compute_and_write_feature_ablation(
                config=enabled_config,
                subrun=subrun,
                output_dir=tmpdir,
                subrun_dir=subrun_dir,
                model=model,
                overwrite=False,
            )
            self.assertEqual(cached, {"cached": True})

            lm_payload = _base_config(output_dir=tmpdir, data_files=["dummy.jsonl"])
            lm_payload["feature_ablation"] = {
                "enabled": True,
                "metric": "delta_lm_loss",
                "num_features": 3,
                "lm_max_features": 2,
                "selection": "top_mean_abs",
                "max_eval_batches": 1,
                "seed": 0,
            }
            lm_payload["loss_recovered"] = {"enabled": True}
            lm_config = LLMBenchmarkConfig.model_validate(lm_payload)
            with patch(
                "sgnsae.experiments.llm_benchmark.feature_ablation._select_features_for_ablation",
                return_value=([1, 2], 2, 8),
            ), patch(
                "sgnsae.experiments.llm_benchmark.feature_ablation._compute_delta_lm_loss_impacts",
                return_value=([0.1, 0.2], 2),
            ):
                payload = feature_ablation_module.compute_and_write_feature_ablation(
                    config=lm_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    subrun_dir=subrun_dir,
                    model=model,
                    overwrite=True,
                )
            self.assertEqual(payload["metric"], "delta_lm_loss")
            self.assertIsNone(payload["eval"]["baseline_mse"])
            self.assertEqual(int(payload["eval"]["num_features_evaluated"]), 2)


class TestPlottingAdditionalCoverage(unittest.TestCase):
    def _summary_with_subruns(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "metadata": {},
            "subruns": [
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "mean_l0": 15.0,
                    "mse": 0.2,
                    "relative_dir": "models/variant_a/seed_0/lambda_1e-03",
                },
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-4,
                    "mean_l0": 65.0,
                    "mse": 0.1,
                    "relative_dir": "models/variant_a/seed_0/lambda_1e-04",
                },
                {
                    "variant_slug": "variant_b",
                    "seed": 1,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "mean_l0": 64.0,
                    "mse": 0.3,
                    "relative_dir": "models/variant_b/seed_1/lambda_1e-03",
                },
            ],
            "aggregates": [
                {
                    "variant_slug": "variant_a",
                    "mean_l0_mean": 16.0,
                    "mse_mean": 0.2,
                    "loss_recovered_mean": 0.4,
                    "dead_fraction_mean": 0.1,
                },
                {
                    "variant_slug": "variant_a",
                    "mean_l0_mean": 64.0,
                    "mse_mean": 0.1,
                    "loss_recovered_mean": 0.6,
                    "dead_fraction_mean": 0.2,
                },
                {
                    "variant_slug": "variant_b",
                    "mean_l0_mean": 64.0,
                    "mse_mean": 0.3,
                    "loss_recovered_mean": None,
                    "dead_fraction_mean": 0.3,
                },
                {
                    "variant_slug": 1,
                    "mean_l0_mean": 10.0,
                    "mse_mean": 1.0,
                },
            ],
        }

    def test_load_json_metric_and_selection_helpers_cover_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing.json")
            non_dict_path = os.path.join(tmpdir, "list.json")
            with open(non_dict_path, "w") as file:
                json.dump([1, 2, 3], file)

            summary = self._summary_with_subruns()
            summary["subruns"].append("not-a-dict")
            summary["subruns"].append(
                {
                    "variant_slug": "variant_invalid_seed",
                    "seed": True,
                    "mean_l0": 32.0,
                    "mse": 0.5,
                }
            )
            selected = plotting_module._select_matched_l0_entries(
                summary, targets=[16, 64]
            )
            empty_selected = plotting_module._select_matched_l0_entries(
                {"subruns": "bad"}, targets=[16]
            )

            output_path = os.path.join(tmpdir, "metric.png")
            no_data_path = os.path.join(tmpdir, "no_data.png")

            # when
            loaded_missing = plotting_module._load_json_if_exists(missing_path)
            loaded_non_dict = plotting_module._load_json_if_exists(non_dict_path)
            plotting_module._plot_metric_vs_l0(
                output_path=output_path,
                aggregates=summary["aggregates"],
                y_key="mse_mean",
                y_label="MSE",
                title="Reconstruction Error vs Mean L0",
            )
            plotting_module._plot_metric_vs_l0(
                output_path=no_data_path,
                aggregates=[],
                y_key="missing",
                y_label="missing",
                title="Missing",
            )

            # then
            self.assertIsNone(loaded_missing)
            self.assertIsNone(loaded_non_dict)
            self.assertTrue(os.path.isfile(output_path))
            self.assertTrue(os.path.isfile(no_data_path))
            self.assertGreaterEqual(len(selected), 2)
            self.assertEqual(empty_selected, [])

    def test_select_matched_l0_entries_uses_stable_tertiary_tiebreak(self) -> None:
        # given
        summary = {
            "subruns": [
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 0.001,
                    "mean_l0": 15.0,
                    "mse": 0.1,
                    "relative_dir": "models/variant_a/seed_0/lambda_1e-03",
                },
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 0.0003,
                    "mean_l0": 17.0,
                    "mse": 0.1,
                    "relative_dir": "models/variant_a/seed_0/lambda_3e-04",
                },
            ]
        }

        # when
        selected = plotting_module._select_matched_l0_entries(summary, targets=[16])

        # then
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            str(selected[0]["relative_dir"]),
            "models/variant_a/seed_0/lambda_3e-04",
        )

    def test_select_matched_l0_entries_treats_non_finite_values_as_worst(self) -> None:
        # given
        summary = {
            "subruns": [
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 0.001,
                    "mean_l0": 16.0,
                    "mse": float("nan"),
                    "relative_dir": "models/variant_a/seed_0/lambda_nan",
                },
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 0.0003,
                    "mean_l0": 16.0,
                    "mse": 0.1,
                    "relative_dir": "models/variant_a/seed_0/lambda_finite",
                },
            ]
        }

        # when
        selected = plotting_module._select_matched_l0_entries(summary, targets=[16])

        # then
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            str(selected[0]["relative_dir"]),
            "models/variant_a/seed_0/lambda_finite",
        )

    def test_select_matched_l0_entries_handles_tie_break_edge_value_types(self) -> None:
        # given
        summary = {
            "subruns": [
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": True,
                    "mean_l0": float("nan"),
                    "mse": 0.2,
                    "relative_dir": "models/variant_a/seed_0/lambda_bool",
                },
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": "invalid",
                    "mean_l0": 16.0,
                    "mse": 0.1,
                    "relative_dir": "models/variant_a/seed_0/lambda_invalid",
                },
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": float("nan"),
                    "mean_l0": 16.0,
                    "mse": 0.1,
                    "relative_dir": "models/variant_a/seed_0/lambda_nan",
                },
                {
                    "variant_slug": "variant_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 0.001,
                    "mean_l0": 16.0,
                    "mse": 0.1,
                    "relative_dir": "models/variant_a/seed_0/lambda_valid",
                },
            ]
        }

        # when
        selected = plotting_module._select_matched_l0_entries(summary, targets=[16])

        # then
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            str(selected[0]["relative_dir"]),
            "models/variant_a/seed_0/lambda_valid",
        )

    def test_threshold_helper_extracts_aligned_step_value_pairs(self) -> None:
        # given
        history = [
            {"step": 1, "delta_pos_mean": 0.1},
            {"step": 2},
            {"step": None, "delta_pos_mean": 0.2},
            {"step": 4, "delta_pos_mean": 0.3},
            {"step": True, "delta_pos_mean": 0.5},
            {"step": 6, "delta_pos_mean": False},
            {"step": 7, "delta_pos_mean": "invalid"},
        ]

        # when
        steps, values = plotting_module._extract_step_value_pairs(
            history=history,
            metric_key="delta_pos_mean",
        )

        # then
        self.assertEqual(steps, [1.0, 4.0])
        self.assertEqual(values, [0.1, 0.3])

    def test_matched_l0_ensure_helper_reuses_or_augments_summary(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_without_matches = self._summary_with_subruns()
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(summary_without_matches, file, indent=2)

            # when
            augmented = plotting_module.ensure_summary_has_matched_l0_anchors(
                output_dir=tmpdir,
                summary=summary_without_matches,
            )

            # then
            self.assertIn("matched_l0_anchors", augmented)
            self.assertIn("targets", augmented["matched_l0_anchors"])
            self.assertIn("selected", augmented["matched_l0_anchors"])

            # given
            summary_with_matches = self._summary_with_subruns()
            summary_with_matches["matched_l0_anchors"] = {
                "targets": [16, 64],
                "selected": [{"target": 64, "relative_dir": "models/example"}],
            }

            # when
            reused = plotting_module.ensure_summary_has_matched_l0_anchors(
                output_dir=tmpdir,
                summary=summary_with_matches,
            )

            # then
            self.assertEqual(
                reused["matched_l0_anchors"], summary_with_matches["matched_l0_anchors"]
            )

            # given
            summary_with_non_default_targets = self._summary_with_subruns()
            summary_with_non_default_targets["matched_l0_anchors"] = {
                "targets": [32],
                "selected": [{"target": 32, "relative_dir": "models/example"}],
            }

            # when
            refreshed = plotting_module.ensure_summary_has_matched_l0_anchors(
                output_dir=tmpdir,
                summary=summary_with_non_default_targets,
            )

            # then
            self.assertEqual(refreshed["matched_l0_anchors"]["targets"], [16, 64])

            # given
            summary_with_invalid_target_type = self._summary_with_subruns()
            summary_with_invalid_target_type["matched_l0_anchors"] = {
                "targets": [16, True],
                "selected": [{"target": 16, "relative_dir": "models/example"}],
            }

            # when
            refreshed_from_invalid_targets = (
                plotting_module.ensure_summary_has_matched_l0_anchors(
                    output_dir=tmpdir,
                    summary=summary_with_invalid_target_type,
                )
            )

            # then
            self.assertEqual(
                refreshed_from_invalid_targets["matched_l0_anchors"]["targets"],
                [16, 64],
            )

    def test_alive_index_resolution_for_duplication_covers_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            relative_dir = "models/variant/seed_0/lambda_1e-03"
            subrun_dir = os.path.join(tmpdir, relative_dir)
            os.makedirs(subrun_dir, exist_ok=True)

            # when / then
            self.assertIsNone(
                plotting_module._resolve_alive_feature_indices_for_duplication(
                    output_dir=tmpdir,
                    relative_dir=relative_dir,
                    d_hidden=6,
                )
            )

            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump({"alive_indices": []}, file, indent=2)
            empty_indices = (
                plotting_module._resolve_alive_feature_indices_for_duplication(
                    output_dir=tmpdir,
                    relative_dir=relative_dir,
                    d_hidden=6,
                )
            )
            self.assertIsNotNone(empty_indices)
            self.assertEqual(int(empty_indices.numel()), 0)

            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump({"alive_indices": "bad"}, file, indent=2)
            self.assertIsNone(
                plotting_module._resolve_alive_feature_indices_for_duplication(
                    output_dir=tmpdir,
                    relative_dir=relative_dir,
                    d_hidden=6,
                )
            )

            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump({"alive_indices": [None, True, -1, 8]}, file, indent=2)
            self.assertIsNone(
                plotting_module._resolve_alive_feature_indices_for_duplication(
                    output_dir=tmpdir,
                    relative_dir=relative_dir,
                    d_hidden=6,
                )
            )

            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump({"alive_indices": [4, 1, 1, 3]}, file, indent=2)
            resolved = plotting_module._resolve_alive_feature_indices_for_duplication(
                output_dir=tmpdir,
                relative_dir=relative_dir,
                d_hidden=6,
            )

            # then
            self.assertTrue(torch.equal(resolved, torch.tensor([1, 3, 4])))

    def test_stable_duplication_seed_is_deterministic(self) -> None:
        # given
        relative_dir_a = "models/variant_a/seed_0/lambda_1e-03"
        relative_dir_b = "models/variant_b/seed_0/lambda_1e-03"

        # when
        first_seed_a = plotting_module._stable_duplication_seed(relative_dir_a)
        second_seed_a = plotting_module._stable_duplication_seed(relative_dir_a)
        seed_b = plotting_module._stable_duplication_seed(relative_dir_b)

        # then
        self.assertEqual(first_seed_a, second_seed_a)
        self.assertNotEqual(first_seed_a, seed_b)

    def test_augment_and_selected_dir_helpers_cover_invalid_inputs(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = self._summary_with_subruns()
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(summary, file, indent=2)

            # when
            updated = plotting_module._augment_summary_with_matched_l0(tmpdir, summary)
            target_dirs = list(
                plotting_module._iter_selected_subrun_dirs_for_target(
                    updated, target=64
                )
            )
            not_dict_dirs = list(
                plotting_module._iter_selected_subrun_dirs_for_target(
                    {"matched_l0_anchors": "bad"}, target=64
                )
            )
            not_list_dirs = list(
                plotting_module._iter_selected_subrun_dirs_for_target(
                    {"matched_l0_anchors": {"selected": "bad"}}, target=64
                )
            )
            mixed_entry_dirs = list(
                plotting_module._iter_selected_subrun_dirs_for_target(
                    {
                        "matched_l0_anchors": {
                            "selected": [
                                "not-a-dict",
                                {"target": 64, "relative_dir": "models/example"},
                            ]
                        }
                    },
                    target=64,
                )
            )

            # then
            self.assertIn("matched_l0_anchors", updated)
            self.assertGreaterEqual(len(target_dirs), 1)
            self.assertEqual(not_dict_dirs, [])
            self.assertEqual(not_list_dirs, [])
            self.assertEqual(mixed_entry_dirs, ["models/example"])

    def test_gamma_and_polarity_histograms_cover_data_and_no_data_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = self._summary_with_subruns()
            summary = plotting_module._augment_summary_with_matched_l0(tmpdir, summary)
            gamma_output = os.path.join(tmpdir, "gamma.png")
            polarity_output = os.path.join(tmpdir, "polarity.png")

            # when
            plotting_module._plot_gamma_histogram_for_anchor(
                output_dir=tmpdir,
                summary=summary,
                target=64,
                output_path=gamma_output,
            )
            plotting_module._plot_polarity_imbalance_histogram(
                output_dir=tmpdir,
                summary=summary,
                output_path=polarity_output,
                anchor_target=64,
            )

            # then
            self.assertTrue(os.path.isfile(gamma_output))
            self.assertTrue(os.path.isfile(polarity_output))

            # given
            for relative_dir in plotting_module._iter_selected_subrun_dirs_for_target(
                summary, target=64
            ):
                subrun_dir = os.path.join(tmpdir, relative_dir)
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "gamma_metrics.json"), "w") as file:
                    json.dump(
                        {"gamma_pos_values": [0.2, None], "gamma_neg_values": [-0.2]},
                        file,
                        indent=2,
                    )
                with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                    json.dump(
                        {
                            "polarity_imbalance_by_latent": (
                                [0.1, None, 0.3]
                                if "variant_a" in relative_dir
                                else "not-a-list"
                            )
                        },
                        file,
                        indent=2,
                    )

            # when
            plotting_module._plot_gamma_histogram_for_anchor(
                output_dir=tmpdir,
                summary=summary,
                target=64,
                output_path=gamma_output,
            )
            plotting_module._plot_polarity_imbalance_histogram(
                output_dir=tmpdir,
                summary=summary,
                output_path=polarity_output,
                anchor_target=64,
            )

            # then
            self.assertTrue(os.path.isfile(gamma_output))
            self.assertTrue(os.path.isfile(polarity_output))

    def test_plotting_helpers_cover_remaining_gamma_duplication_and_summary_branches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_for_ensure = self._summary_with_subruns()
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(summary_for_ensure, file, indent=2)
            summary_with_invalid_selected = self._summary_with_subruns()
            summary_with_invalid_selected["matched_l0_anchors"] = {
                "targets": [16, 64],
                "selected": "bad",
            }

            # when
            refreshed_summary = plotting_module.ensure_summary_has_matched_l0_anchors(
                output_dir=tmpdir,
                summary=summary_with_invalid_selected,
            )

            # then
            self.assertIn("matched_l0_anchors", refreshed_summary)
            self.assertIsInstance(
                refreshed_summary["matched_l0_anchors"]["selected"], list
            )

            # given
            gamma_summary = {
                "matched_l0_anchors": {
                    "selected": [
                        {"target": 64, "relative_dir": "models/a"},
                        {"target": 64, "relative_dir": "models/b"},
                        {"target": 64, "relative_dir": ""},
                        {"target": 64, "relative_dir": None},
                    ]
                }
            }
            gamma_output = os.path.join(tmpdir, "gamma_remaining.png")
            for relative_dir in ("models/a", "models/b"):
                subrun_dir = os.path.join(tmpdir, relative_dir)
                os.makedirs(subrun_dir, exist_ok=True)
            with open(
                os.path.join(tmpdir, "models/a", "gamma_metrics.json"), "w"
            ) as file:
                json.dump(
                    {"gamma_pos_values": "bad", "gamma_neg_values": [None, -0.2]},
                    file,
                    indent=2,
                )
            with open(
                os.path.join(tmpdir, "models/b", "gamma_metrics.json"), "w"
            ) as file:
                json.dump(
                    {"gamma_pos_values": [0.3], "gamma_neg_values": "bad"},
                    file,
                    indent=2,
                )

            # when
            selected_dirs = list(
                plotting_module._iter_selected_subrun_dirs_for_target(
                    gamma_summary,
                    target=64,
                )
            )
            plotting_module._plot_gamma_histogram_for_anchor(
                output_dir=tmpdir,
                summary=gamma_summary,
                target=64,
                output_path=gamma_output,
            )

            # then
            self.assertEqual(selected_dirs, ["models/a", "models/b"])
            self.assertTrue(os.path.isfile(gamma_output))

            # given
            with open(
                os.path.join(tmpdir, "models/a", "gamma_metrics.json"), "w"
            ) as file:
                json.dump(
                    {"gamma_pos_values": [], "gamma_neg_values": [-0.4]},
                    file,
                    indent=2,
                )
            with open(
                os.path.join(tmpdir, "models/b", "gamma_metrics.json"), "w"
            ) as file:
                json.dump(
                    {"gamma_pos_values": "bad", "gamma_neg_values": "bad"},
                    file,
                    indent=2,
                )

            # when
            plotting_module._plot_gamma_histogram_for_anchor(
                output_dir=tmpdir,
                summary=gamma_summary,
                target=64,
                output_path=gamma_output,
            )

            # then
            self.assertTrue(os.path.isfile(gamma_output))

            # given
            with open(
                os.path.join(tmpdir, "models/a", "gamma_metrics.json"), "w"
            ) as file:
                json.dump(
                    {"gamma_pos_values": [0.5], "gamma_neg_values": []},
                    file,
                    indent=2,
                )
            with open(
                os.path.join(tmpdir, "models/b", "gamma_metrics.json"), "w"
            ) as file:
                json.dump(
                    {"gamma_pos_values": "bad", "gamma_neg_values": "bad"},
                    file,
                    indent=2,
                )

            # when
            plotting_module._plot_gamma_histogram_for_anchor(
                output_dir=tmpdir,
                summary=gamma_summary,
                target=64,
                output_path=gamma_output,
            )

            # then
            self.assertTrue(os.path.isfile(gamma_output))

            # given
            duplication_output = os.path.join(tmpdir, "duplication_remaining.png")
            duplication_summary = {
                "matched_l0_anchors": {
                    "selected": [
                        {"target": 64, "relative_dir": "models/a"},
                        {"target": 64, "relative_dir": "models/b"},
                    ]
                }
            }

            # when
            with patch.object(
                plotting_module,
                "_load_or_compute_duplication_metrics",
                side_effect=[
                    {
                        "min_cosine_values": "bad",
                        "counts_below_thresholds": "bad",
                        "pair_count": "bad",
                    },
                    {
                        "min_cosine_values": [-0.5],
                        "counts_below_thresholds": {"-0.9": "bad", "-0.95": 1},
                        "pair_count": 2.5,
                    },
                ],
            ):
                plotting_module._plot_duplication_histogram(
                    output_dir=tmpdir,
                    summary=duplication_summary,
                    output_path=duplication_output,
                    anchor_target=64,
                )

            # then
            self.assertTrue(os.path.isfile(duplication_output))

    def test_duplication_helpers_cover_compute_and_load_paths(self) -> None:
        # given
        decoder_rows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        # when
        sampled_same = plotting_module._sample_decoder_rows(
            decoder_rows, sample_size=5, seed=0
        )
        sampled_subset = plotting_module._sample_decoder_rows(
            decoder_rows, sample_size=2, seed=0
        )
        metrics_single = plotting_module._compute_duplication_metrics_for_decoder(
            decoder_rows[:1]
        )
        metrics_multi = plotting_module._compute_duplication_metrics_for_decoder(
            decoder_rows
        )

        # then
        self.assertEqual(tuple(sampled_same.shape), tuple(decoder_rows.shape))
        self.assertEqual(tuple(sampled_subset.shape), (2, 2))
        self.assertEqual(metrics_single["pair_count"], 0)
        self.assertGreater(metrics_multi["pair_count"], 0)
        with patch(
            "sgnsae.experiments.llm_benchmark.plotting._DUPLICATION_CHUNK_SIZE", 2
        ):
            chunked_metrics = plotting_module._compute_duplication_metrics_for_decoder(
                decoder_rows
            )
        self.assertGreater(chunked_metrics["pair_count"], 0)

        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            relative_dir = "models/variant/seed_0/lambda_1e-03"
            subrun_dir = os.path.join(tmpdir, relative_dir)
            os.makedirs(subrun_dir, exist_ok=True)
            metrics_path = os.path.join(subrun_dir, "duplication_metrics.json")
            with open(metrics_path, "w") as file:
                json.dump({"loaded": True}, file)

            # when
            loaded = plotting_module._load_or_compute_duplication_metrics(
                output_dir=tmpdir, relative_dir=relative_dir
            )

            # then
            self.assertEqual(loaded, {"loaded": True})

            # given
            os.remove(metrics_path)

            # when / then
            self.assertIsNone(
                plotting_module._load_or_compute_duplication_metrics(
                    output_dir=tmpdir, relative_dir=relative_dir
                )
            )

            checkpoint_path = os.path.join(subrun_dir, "model.pt")
            torch.save([1, 2, 3], checkpoint_path)
            self.assertIsNone(
                plotting_module._load_or_compute_duplication_metrics(
                    output_dir=tmpdir, relative_dir=relative_dir
                )
            )

            torch.save({"state_dict": []}, checkpoint_path)
            self.assertIsNone(
                plotting_module._load_or_compute_duplication_metrics(
                    output_dir=tmpdir, relative_dir=relative_dir
                )
            )

            torch.save({"state_dict": {"W_dec": torch.ones(2, 2)}}, checkpoint_path)
            self.assertIsNone(
                plotting_module._load_or_compute_duplication_metrics(
                    output_dir=tmpdir, relative_dir=relative_dir
                )
            )

            torch.save(
                {"state_dict": {"W_dec": torch.randn(1, 4, 6, dtype=torch.float32)}},
                checkpoint_path,
            )
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump({"alive_indices": [1, 3]}, file, indent=2)
            computed = plotting_module._load_or_compute_duplication_metrics(
                output_dir=tmpdir, relative_dir=relative_dir
            )
            self.assertIsNotNone(computed)
            self.assertTrue(os.path.isfile(metrics_path))
            self.assertEqual(int(computed["candidate_feature_count"]), 2)
            self.assertTrue(bool(computed["alive_filter_applied"]))
            self.assertEqual(int(computed["sample_size"]), 2)
            self.assertEqual(
                int(computed["seed"]),
                int(plotting_module._stable_duplication_seed(relative_dir)),
            )

            with open(metrics_path, "r") as file:
                persisted = json.load(file)
            self.assertEqual(int(persisted["candidate_feature_count"]), 2)
            self.assertTrue(bool(persisted["alive_filter_applied"]))

            relative_dir_no_filter = "models/variant_no_filter/seed_0/lambda_1e-03"
            subrun_dir_no_filter = os.path.join(tmpdir, relative_dir_no_filter)
            os.makedirs(subrun_dir_no_filter, exist_ok=True)
            checkpoint_path_no_filter = os.path.join(subrun_dir_no_filter, "model.pt")
            torch.save(
                {"state_dict": {"W_dec": torch.randn(1, 4, 6, dtype=torch.float32)}},
                checkpoint_path_no_filter,
            )
            computed_no_filter = plotting_module._load_or_compute_duplication_metrics(
                output_dir=tmpdir, relative_dir=relative_dir_no_filter
            )
            self.assertIsNotNone(computed_no_filter)
            self.assertEqual(int(computed_no_filter["candidate_feature_count"]), 6)
            self.assertFalse(bool(computed_no_filter["alive_filter_applied"]))

            relative_dir_empty_alive = "models/variant_empty_alive/seed_0/lambda_1e-03"
            subrun_dir_empty_alive = os.path.join(tmpdir, relative_dir_empty_alive)
            os.makedirs(subrun_dir_empty_alive, exist_ok=True)
            checkpoint_path_empty_alive = os.path.join(
                subrun_dir_empty_alive, "model.pt"
            )
            torch.save(
                {"state_dict": {"W_dec": torch.randn(1, 4, 6, dtype=torch.float32)}},
                checkpoint_path_empty_alive,
            )
            with open(
                os.path.join(subrun_dir_empty_alive, "evaluation.json"), "w"
            ) as file:
                json.dump({"alive_indices": []}, file, indent=2)
            computed_empty_alive = plotting_module._load_or_compute_duplication_metrics(
                output_dir=tmpdir, relative_dir=relative_dir_empty_alive
            )
            self.assertIsNotNone(computed_empty_alive)
            self.assertEqual(int(computed_empty_alive["candidate_feature_count"]), 0)
            self.assertTrue(bool(computed_empty_alive["alive_filter_applied"]))

    def test_duplication_histogram_threshold_stability_and_generate_plots(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = self._summary_with_subruns()
            summary = plotting_module._augment_summary_with_matched_l0(tmpdir, summary)
            duplication_output = os.path.join(tmpdir, "duplication.png")
            threshold_output = os.path.join(tmpdir, "threshold.png")

            # when
            plotting_module._plot_duplication_histogram(
                output_dir=tmpdir,
                summary=summary,
                output_path=duplication_output,
                anchor_target=64,
            )
            plotting_module._plot_threshold_stability(
                output_dir=tmpdir,
                summary=summary,
                output_path=threshold_output,
                anchor_target=64,
            )

            # then
            self.assertTrue(os.path.isfile(duplication_output))
            self.assertTrue(os.path.isfile(threshold_output))

            # given
            selected_dirs = list(
                plotting_module._iter_selected_subrun_dirs_for_target(
                    summary, target=64
                )
            )
            self.assertGreaterEqual(len(selected_dirs), 2)
            first_subrun_dir = os.path.join(tmpdir, selected_dirs[0])
            os.makedirs(first_subrun_dir, exist_ok=True)
            with open(
                os.path.join(first_subrun_dir, "training_metrics.json"), "w"
            ) as file:
                json.dump({"history": {}}, file, indent=2)
            second_subrun_dir = os.path.join(tmpdir, selected_dirs[1])
            os.makedirs(second_subrun_dir, exist_ok=True)
            with open(
                os.path.join(second_subrun_dir, "training_metrics.json"), "w"
            ) as file:
                json.dump({"history": [1, 2]}, file, indent=2)
            plotting_module._plot_threshold_stability(
                output_dir=tmpdir,
                summary=summary,
                output_path=threshold_output,
                anchor_target=64,
            )

            # given
            for relative_dir in plotting_module._iter_selected_subrun_dirs_for_target(
                summary, target=64
            ):
                subrun_dir = os.path.join(tmpdir, relative_dir)
                os.makedirs(subrun_dir, exist_ok=True)
                with open(
                    os.path.join(subrun_dir, "duplication_metrics.json"), "w"
                ) as file:
                    json.dump(
                        {
                            "min_cosine_values": [-0.99, -0.3],
                            "counts_below_thresholds": {"-0.9": 1, "-0.95": 1},
                            "pair_count": 2,
                        },
                        file,
                        indent=2,
                    )
                with open(
                    os.path.join(subrun_dir, "training_metrics.json"), "w"
                ) as file:
                    json.dump(
                        {
                            "history": [
                                {
                                    "step": 1,
                                    "delta_pos_mean": 0.1,
                                    "delta_neg_mean": 0.2,
                                    "theta_mean": 0.3,
                                    "jump_mean": 0.4,
                                    "dead_zone_frac_mean": 0.05,
                                },
                                {"step": 2, "delta_pos_mean": 0.15},
                            ]
                        },
                        file,
                        indent=2,
                    )

            # when
            plotting_module._plot_duplication_histogram(
                output_dir=tmpdir,
                summary=summary,
                output_path=duplication_output,
                anchor_target=64,
            )
            plotting_module._plot_threshold_stability(
                output_dir=tmpdir,
                summary=summary,
                output_path=threshold_output,
                anchor_target=64,
            )

            # then
            self.assertTrue(os.path.isfile(duplication_output))
            self.assertTrue(os.path.isfile(threshold_output))

            # when / then
            comparison_path = os.path.join(tmpdir, "comparison_summary.json")
            Path(comparison_path).unlink(missing_ok=True)
            with self.assertRaises(ValueError):
                _ = plotting_module.generate_plots_for_run(output_dir=tmpdir)

            summary_for_generate = self._summary_with_subruns()
            summary_for_generate["aggregates"] = "invalid"
            with open(comparison_path, "w") as file:
                json.dump(summary_for_generate, file, indent=2)

            result = plotting_module.generate_plots_for_run(output_dir=tmpdir)
            self.assertEqual(
                int(result["plots_written"]),
                len(plotting_module.canonical_plot_paths(tmpdir)),
            )


class TestOverheadAdditionalCoverage(unittest.TestCase):
    def test_dtype_and_topk_helpers_cover_error_paths(self) -> None:
        # given / when / then
        self.assertEqual(
            overhead_module.estimate_parameter_memory_bytes(10, "float16"), 20
        )
        self.assertEqual(
            overhead_module.estimate_parameter_memory_bytes(10, "bfloat16"), 20
        )
        self.assertEqual(
            overhead_module.estimate_parameter_memory_bytes(10, "float32"), 40
        )
        with self.assertRaises(ValueError):
            _ = overhead_module.estimate_parameter_memory_bytes(10, "int8")  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            _ = overhead_module._resolve_topk_k_value({"topk_k": []}, d_hidden=8)
        self.assertEqual(
            overhead_module._resolve_topk_k_value({"topk_k": True}, d_hidden=8), 8
        )
        self.assertEqual(
            overhead_module._resolve_topk_k_value({"topk_k": "bad"}, d_hidden=8), 8
        )
        self.assertEqual(
            overhead_module._resolve_topk_k_value({"topk_k": 0}, d_hidden=8), 1
        )
        self.assertEqual(
            overhead_module._resolve_topk_k_value({"topk_k": 100}, d_hidden=8), 8
        )
        self.assertEqual(
            overhead_module._resolve_topk_k_value({"topk_k": [4, 8]}, d_hidden=8), 4
        )

    def test_build_variant_model_covers_all_supported_classes(self) -> None:
        # given
        variants = [
            {"sae_class": "SAE", "d_hidden": 8, "sign_aware": False},
            {"sae_class": "GatedSAE", "d_hidden": 8, "sign_aware": False},
            {"sae_class": "SignedSoftThresholdSAE", "d_hidden": 8, "sign_aware": False},
            {"sae_class": "JumpReLUSAE", "d_hidden": 8, "sign_aware": False},
            {"sae_class": "TopKSAE", "d_hidden": 8, "sign_aware": False, "topk_k": 4},
            {
                "sae_class": "AbsTopKSAE",
                "d_hidden": 8,
                "sign_aware": False,
                "topk_k": [4],
            },
            {
                "sae_class": "AbsTopKGatedMagnitudeSAE",
                "d_hidden": 8,
                "sign_aware": False,
                "topk_k": [4],
            },
        ]

        # when
        built_models = [
            overhead_module.build_variant_model(
                variant_config=variant,
                d_in=6,
                device="cpu",
                resample_latents_threshold=1,
            )
            for variant in variants
        ]

        # then
        self.assertEqual(len(built_models), len(variants))
        for model in built_models:
            self.assertGreater(overhead_module.count_parameters(model), 0)

        with self.assertRaises(ValueError):
            _ = overhead_module.build_variant_model(
                variant_config={"sae_class": "Unknown", "d_hidden": 8},
                d_in=6,
            )

        summary = overhead_module.summarize_variant_parameters(
            variant_config={
                "name": "variant",
                "variant_slug": "variant",
                "sae_class": "SAE",
                "d_hidden": 8,
                "sign_aware": False,
            },
            d_in=6,
        )
        self.assertEqual(summary["variant_slug"], "variant")
        self.assertIn("float32", summary["estimated_memory_bytes"])

    def test_run_parameter_summary_validates_inputs_and_deduplicates(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            manifest_dir = os.path.join(tmpdir, "activations")
            manifest_path = os.path.join(manifest_dir, "activations_manifest.json")
            os.makedirs(manifest_dir, exist_ok=True)

            # when / then
            with self.assertRaises(ValueError):
                _ = overhead_module._load_resolved_run_config(tmpdir)

            with open(config_path, "w") as file:
                json.dump([1, 2], file)
            with self.assertRaises(ValueError):
                _ = overhead_module._load_resolved_run_config(tmpdir)

            with self.assertRaises(ValueError):
                _ = overhead_module._load_manifest_d_in(tmpdir)

            with open(manifest_path, "w") as file:
                json.dump([1, 2], file)
            with self.assertRaises(ValueError):
                _ = overhead_module._load_manifest_d_in(tmpdir)

            with open(manifest_path, "w") as file:
                json.dump({"activation": "bad"}, file)
            with self.assertRaises(ValueError):
                _ = overhead_module._load_manifest_d_in(tmpdir)

            with open(manifest_path, "w") as file:
                json.dump({"activation": {"d_in": 0}}, file)
            with self.assertRaises(ValueError):
                _ = overhead_module._load_manifest_d_in(tmpdir)

            with open(config_path, "w") as file:
                json.dump({"variants": "bad"}, file)
            with open(manifest_path, "w") as file:
                json.dump({"activation": {"d_in": 6}}, file)
            with self.assertRaises(ValueError):
                _ = overhead_module.summarize_run_parameters(tmpdir)

            with open(config_path, "w") as file:
                json.dump({"variants": ["bad"]}, file)
            with self.assertRaises(ValueError):
                _ = overhead_module.summarize_run_parameters(tmpdir)

            with open(config_path, "w") as file:
                json.dump(
                    {
                        "variants": [
                            {
                                "name": "a",
                                "variant_slug": "a",
                                "sae_class": "SAE",
                                "d_hidden": 8,
                                "sign_aware": False,
                            },
                            {
                                "name": "a_dup",
                                "variant_slug": "a",
                                "sae_class": "SAE",
                                "d_hidden": 8,
                                "sign_aware": False,
                            },
                            {
                                "name": "b",
                                "variant_slug": "b",
                                "sae_class": "TopKSAE",
                                "d_hidden": 8,
                                "sign_aware": False,
                                "topk_k": 4,
                            },
                        ]
                    },
                    file,
                    indent=2,
                )
            summary = overhead_module.summarize_run_parameters(tmpdir)
            self.assertEqual(int(summary["d_in"]), 6)
            self.assertEqual(len(summary["variants"]), 2)


class TestConfigMetricsAndRunnerAdditionalCoverage(unittest.TestCase):
    def test_config_validators_cover_seed_split_and_denominator_paths(self) -> None:
        # given
        base = _base_config(output_dir="/tmp", data_files=["dummy.jsonl"])

        # when / then
        bad_seed = dict(base)
        bad_seed["loss_recovered"] = {"seed": True}
        with self.assertRaises(ValidationError):
            _ = LLMBenchmarkConfig.model_validate(bad_seed)

        bad_dataset_split = dict(base)
        bad_dataset_split["loss_recovered"] = {"dataset_split": ""}
        with self.assertRaises(ValidationError):
            _ = LLMBenchmarkConfig.model_validate(bad_dataset_split)

        good_dataset_split = dict(base)
        good_dataset_split["loss_recovered"] = {"dataset_split": None, "seed": 5}
        config = LLMBenchmarkConfig.model_validate(good_dataset_split)
        self.assertIsNone(config.loss_recovered.dataset_split)
        self.assertEqual(int(config.loss_recovered.seed), 5)

        bad_denominator_type = dict(base)
        bad_denominator_type["loss_recovered"] = {"denominator_epsilon": True}
        with self.assertRaises(ValidationError):
            _ = LLMBenchmarkConfig.model_validate(bad_denominator_type)

        bad_feature_seed = dict(base)
        bad_feature_seed["feature_ablation"] = {"seed": True}
        with self.assertRaises(ValidationError):
            _ = LLMBenchmarkConfig.model_validate(bad_feature_seed)

    def test_metrics_reuse_and_optional_artifacts_cover_remaining_branches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _base_config(output_dir=tmpdir, data_files=["dummy.jsonl"])
            config_payload["loss_recovered"] = {"enabled": True}
            config_payload["feature_ablation"] = {"enabled": True}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = {
                "subruns": [
                    {
                        "variant_name": "variant",
                        "variant_slug": "variant",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_subrun_dir": "models/variant/seed_0/lambda_1e-03",
                    }
                ]
            }
            subrun = resolved_config["subruns"][0]
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump({"ok": True}, file)
            with open(os.path.join(subrun_dir, "gamma_metrics.json"), "w") as file:
                json.dump({"ok": True}, file)
            with open(os.path.join(subrun_dir, "model.pt"), "w") as file:
                file.write("dummy")

            class _FakeModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.anchor = torch.nn.Parameter(
                        torch.zeros(1, dtype=torch.float32)
                    )

                def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # type: ignore[override]
                    del state_dict
                    return None

                def to(self, device: Any) -> "_FakeModel":
                    del device
                    return self

                def eval(self) -> "_FakeModel":
                    return self

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.metrics.resolve_manifest_path",
                return_value=os.path.join(
                    tmpdir, "activations", "activations_manifest.json"
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.load_activation_manifest",
                return_value={"activation": {"d_in": 4}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.resolve_runtime_device",
                return_value="cpu",
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.build_model_for_subrun",
                return_value=(
                    _FakeModel(),
                    None,
                    None,
                    {"d_hidden": 4, "sign_aware": False},
                ),
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.torch.load",
                return_value={"state_dict": {}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.compute_and_write_subrun_loss_recovered",
                return_value={"ok": True},
            ) as loss_writer, patch(
                "sgnsae.experiments.llm_benchmark.metrics.compute_and_write_feature_ablation",
                return_value={"ok": True},
            ) as ablation_writer:
                result = run_evaluation_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            self.assertIn("artifact_paths", result)
            self.assertEqual(loss_writer.call_count, 1)
            self.assertEqual(ablation_writer.call_count, 1)

            # given
            with open(os.path.join(subrun_dir, "loss_recovered.json"), "w") as file:
                json.dump({"ok": True}, file)
            with open(os.path.join(subrun_dir, "feature_ablation.json"), "w") as file:
                json.dump({"ok": True}, file)

            # when
            reused = run_evaluation_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            self.assertTrue(bool(reused["reused_all"]))

    def test_comparison_summary_and_runner_failure_paths_cover_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(
                    {
                        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                        "hook": {
                            "hook_name": "mlp_out",
                            "layer": 0,
                            "module_path_resolved": "transformer.h.0.mlp",
                        },
                        "dataset": {"dataset_id": "fixture", "split": "train"},
                        "tokenization": {"max_seq_len": 8},
                    },
                    file,
                    indent=2,
                )
            subrun_dir = os.path.join(
                tmpdir, "models", "variant", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "mse": 1.0,
                        "r2": 0.1,
                        "mean_l0": 2.0,
                        "dead_fraction": 0.2,
                        "gamma_summary": {"coverage": {"pos_fraction_valid": 0.5}},
                    },
                    file,
                    indent=2,
                )
            with open(os.path.join(subrun_dir, "feature_ablation.json"), "w") as file:
                json.dump({"summary": []}, file, indent=2)
            resolved_config = {
                "llm": {"model_id": "dummy-model"},
                "hook": {"hook_name": "mlp_out", "layer": 0},
                "subruns": [
                    {
                        "variant_name": "variant",
                        "variant_slug": "variant",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_subrun_dir": "models/variant/seed_0/lambda_1e-03",
                    }
                ],
            }

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(summary["subruns"][0]["feature_ablation_mean_delta"], None)

            # given
            output_dir = os.path.join(tmpdir, "runner_output")
            config_path = os.path.join(tmpdir, "runner_config.json")
            with open(config_path, "w") as file:
                json.dump(_runner_config(output_dir=output_dir), file)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._run_plot_stage",
                side_effect=RuntimeError("plot failed"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = runner_module.run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["plot"],
                    )

            with patch(
                "sgnsae.experiments.llm_benchmark.runner._run_summarize_stage",
                side_effect=RuntimeError("summarize failed"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = runner_module.run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["summarize"],
                    )

            with patch(
                "sgnsae.experiments.llm_benchmark.runner._effective_stages",
                return_value=["mystery"],
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.resolve_benchmark_config",
                return_value={
                    "output_dir": output_dir,
                    "config_fingerprint": "fingerprint",
                    "subruns": [],
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.persist_environment_metadata",
                return_value=os.path.join(output_dir, "environment.json"),
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._load_existing_fingerprint",
                return_value=None,
            ):
                with self.assertRaises(NotImplementedError):
                    _ = runner_module.run_llm_sae_benchmark(config_path=config_path)

            # given
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            plot_paths = plotting_module.plot_stage_artifact_paths(output_dir)
            for path in plot_paths.values():
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as file:
                    file.write("plot")

            # when
            runner_module._delete_plot_files_if_present(plot_paths)

            # then
            for path in plot_paths.values():
                self.assertFalse(os.path.isfile(path))

            # given
            config = LLMBenchmarkConfig.model_validate(
                _runner_config(output_dir=output_dir)
            )
            eval_config_payload = _runner_config(output_dir=output_dir)
            eval_config_payload["loss_recovered"] = {"enabled": True}
            eval_config = LLMBenchmarkConfig.model_validate(eval_config_payload)
            subrun_dir = os.path.join(
                output_dir, "models", "variant", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)
            for path in plot_paths.values():
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as file:
                    file.write("plot")
            with open(os.path.join(output_dir, "comparison_summary.json"), "w") as file:
                json.dump({"aggregates": [], "subruns": [], "metadata": {}}, file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.ensure_run_level_loss_recovered_artifacts",
                return_value={"enabled": True},
            ) as ensure_lr, patch(
                "sgnsae.experiments.llm_benchmark.runner._are_artifacts_present",
                return_value=False,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": []},
            ):
                eval_stage = runner_module._run_eval_stage(
                    config=eval_config,
                    resolved_config={
                        "subruns": [
                            {
                                "relative_subrun_dir": "models/variant/seed_0/lambda_1e-03"
                            }
                        ]
                    },
                    output_dir=output_dir,
                )

            # then
            self.assertEqual(ensure_lr.call_count, 1)
            self.assertEqual(int(eval_stage["evaluated_subruns"]), 1)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._are_artifacts_present",
                return_value=True,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"metadata": {}, "aggregates": [], "subruns": []},
            ):
                reused_plot = runner_module._run_plot_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=output_dir,
                )

            # then
            self.assertTrue(bool(reused_plot["reused"]))

            # when
            config_overwrite = config.model_copy(update={"overwrite": True})
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._delete_plot_stage_artifacts_if_present"
            ) as delete_plots, patch(
                "sgnsae.experiments.llm_benchmark.runner._are_artifacts_present",
                return_value=False,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_plots_for_run",
                return_value={"plots_written": 1},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"aggregates": [], "subruns": [], "metadata": {}},
            ):
                _ = runner_module._run_plot_stage(
                    config=config_overwrite,
                    resolved_config={"subruns": []},
                    output_dir=output_dir,
                )

            # then
            self.assertEqual(delete_plots.call_count, 1)

            # when
            comparison_summary_path = os.path.join(
                output_dir, "comparison_summary.json"
            )
            Path(comparison_summary_path).unlink(missing_ok=True)
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._are_artifacts_present",
                return_value=True,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"aggregates": [], "subruns": [], "metadata": {}},
            ) as build_summary:
                reused_missing_summary = runner_module._run_plot_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=output_dir,
                )

            # then
            self.assertTrue(bool(reused_missing_summary["reused"]))
            self.assertEqual(build_summary.call_count, 1)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._are_artifacts_present",
                return_value=False,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_plots_for_run",
                return_value={"plots_written": 5},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"aggregates": [], "subruns": [], "metadata": {}},
            ):
                generated_plot = runner_module._run_plot_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=output_dir,
                )

            # then
            self.assertFalse(bool(generated_plot["reused"]))
            self.assertEqual(int(generated_plot["plots_written"]), 5)

            # when
            best_min = runner_module._select_best_metric_entry(
                aggregates=[
                    {
                        "variant_slug": "variant",
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "mse_mean": 0.2,
                        "num_seeds": 2,
                    },
                    {
                        "variant_slug": "variant",
                        "sweep_param": "lambda",
                        "sweep_value": 1e-4,
                        "mse_mean": 0.1,
                        "num_seeds": 2,
                    },
                    {"variant_slug": 1, "mse_mean": 0.5},
                    {"variant_slug": "variant", "mse_mean": None},
                ],
                metric_name="mse_mean",
                maximize=False,
            )
            best_max = runner_module._select_best_metric_entry(
                aggregates=[
                    {"variant_slug": "a", "score": 1.0},
                    {"variant_slug": "a", "score": 2.0},
                ],
                metric_name="score",
                maximize=True,
            )

            # then
            self.assertEqual(float(best_min[0]["mse_mean"]), 0.1)
            self.assertEqual(float(best_max[0]["score"]), 2.0)

            # given
            with open(os.path.join(output_dir, "comparison_summary.json"), "w") as file:
                json.dump(["invalid"], file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"metadata": {}, "aggregates": [], "subruns": []},
            ):
                summarize_result = runner_module._run_summarize_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=output_dir,
                )

            # then
            self.assertTrue(os.path.isfile(summarize_result["results_summary_path"]))

            # given
            with open(os.path.join(output_dir, "comparison_summary.json"), "w") as file:
                json.dump({"metadata": {}, "aggregates": [], "subruns": []}, file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={
                    "metadata": {},
                    "aggregates": [],
                    "subruns": [],
                },
            ):
                summarize_from_dict = runner_module._run_summarize_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=output_dir,
                )

            # then
            self.assertTrue(os.path.isfile(summarize_from_dict["results_summary_path"]))
            with open(summarize_from_dict["results_summary_path"], "r") as file:
                summarize_payload = json.load(file)
            self.assertIn("matched_l0_anchors", summarize_payload)
            self.assertEqual(
                summarize_payload["matched_l0_anchors"]["targets"], [16, 64]
            )

            # when
            Path(os.path.join(output_dir, "comparison_summary.json")).unlink(
                missing_ok=True
            )
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"metadata": {}, "aggregates": "bad", "subruns": []},
            ):
                summarize_missing_file = runner_module._run_summarize_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=output_dir,
                )

            # then
            self.assertTrue(
                os.path.isfile(summarize_missing_file["results_summary_path"])
            )

    def test_comparison_summary_excludes_stale_evaluation_for_incomplete_training(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(
                    {
                        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                        "hook": {"hook_name": "mlp_out", "layer": 0},
                        "dataset": {"dataset_id": "fixture", "split": "train"},
                        "tokenization": {"max_seq_len": 8},
                    },
                    file,
                    indent=2,
                )
            resolved_config = {
                "llm": {"model_id": "dummy-model"},
                "hook": {"hook_name": "mlp_out", "layer": 0},
                "subruns": [],
            }

            for index, training_metrics_payload in enumerate(
                [
                    {},
                    {"status": "failed"},
                    {"status": "running"},
                    {"status": "completed", "terminated_early": True},
                    {"status": "completed", "failure": {"type": "exception"}},
                    "[]",
                    "[",
                ]
            ):
                relative_dir = f"models/variant/seed_0/lambda_{index}"
                subrun_dir = os.path.join(tmpdir, relative_dir)
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                    json.dump(
                        {
                            "mse": float(index),
                            "r2": 0.1,
                            "mean_l0": 2.0,
                            "dead_fraction": 0.2,
                        },
                        file,
                        indent=2,
                    )
                with open(
                    os.path.join(subrun_dir, "training_metrics.json"), "w"
                ) as file:
                    if isinstance(training_metrics_payload, str):
                        file.write(training_metrics_payload)
                    else:
                        json.dump(training_metrics_payload, file, indent=2)
                resolved_config["subruns"].append(
                    {
                        "variant_name": "variant",
                        "variant_slug": "variant",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": index,
                        "relative_subrun_dir": relative_dir,
                    }
                )

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(summary["subruns"], [])

    def test_generate_plots_for_run_overwrites_stale_matched_points_payload(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(
                    {
                        "metadata": {},
                        "aggregates": [],
                        "subruns": [],
                    },
                    file,
                    indent=2,
                )
            with open(os.path.join(tmpdir, "matched_points.json"), "w") as file:
                json.dump(
                    {
                        "anchors": {"l0": [16]},
                        "per_seed": [
                            {
                                "variant_slug": "variant",
                                "seed": 0,
                                "anchor_type": "l0",
                                "anchor_value": 16.0,
                                "method": "stale",
                                "distance_to_target": 0.0,
                                "metrics": {"mean_l0": 16.0, "mse": 0.1},
                            }
                        ],
                        "aggregate": [
                            {
                                "variant_slug": "variant",
                                "anchor_type": "l0",
                                "anchor_value": 16.0,
                                "mean_l0": 16.0,
                                "mse": 0.1,
                            }
                        ],
                    },
                    file,
                    indent=2,
                )

            # when
            result = plotting_module.generate_plots_for_run(output_dir=tmpdir)

            # then
            self.assertEqual(
                int(result["plots_written"]),
                len(plotting_module.canonical_plot_paths(output_dir=tmpdir)),
            )
            with open(os.path.join(tmpdir, "matched_points.json"), "r") as file:
                matched_points_payload = json.load(file)
            self.assertEqual(matched_points_payload["per_seed"], [])
            self.assertEqual(matched_points_payload["aggregate"], [])


class TestTinyGPT2SmokeIntegration(unittest.TestCase):
    def test_tiny_gpt2_smoke_exercises_loss_recovered_and_plotting(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "texts.jsonl")
            _write_local_data(
                data_path,
                [
                    "Tiny GPT2 smoke test document number one.",
                    "Tiny GPT2 smoke test document number two.",
                    "Tiny GPT2 smoke test document number three.",
                ],
            )
            config_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_payload["llm"] = {"model_id": "sshleifer/tiny-gpt2"}
            config_payload["loss_recovered"] = {
                "enabled": True,
                "max_eval_sequences": 2,
                "llm_batch_size": 1,
                "persist_tokens": False,
                "denominator_epsilon": 1e-12,
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)

            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(
                    {
                        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                        "activation": {"d_in": 2},
                        "hook": {
                            "module_path_resolved": "transformer.h.0.mlp.c_proj",
                            "layer": 0,
                        },
                        "split": {
                            "method": "shard_ranges",
                            "train_shard_indices": [0],
                            "val_shard_indices": [0],
                            "test_shard_indices": [],
                        },
                        "shards": [
                            {
                                "filename": "shard_00000.pt",
                                "num_vectors": 1,
                                "start_vector_index": 0,
                                "end_vector_index": 1,
                            }
                        ],
                    },
                    file,
                    indent=2,
                )

            try:
                _ = loss_recovered_module.ensure_run_level_loss_recovered_artifacts(
                    config=config,
                    output_dir=tmpdir,
                    overwrite=True,
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - only for unavailable external model artifacts
                self.skipTest(
                    f"Unable to run tiny-gpt2 smoke test in this environment: {exc}"
                )

            sae_model = SAE(
                n_instances=1,
                d_in=2,
                d_hidden=4,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            subrun = {
                "variant_name": "smoke_variant",
                "variant_slug": "smoke_variant",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/smoke_variant/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)

            # when
            loss_payload = (
                loss_recovered_module.compute_and_write_subrun_loss_recovered(
                    config=config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    subrun_dir=subrun_dir,
                    sae_model=sae_model,
                    overwrite=True,
                )
            )
            with open(os.path.join(subrun_dir, "gamma_metrics.json"), "w") as file:
                json.dump(
                    {"gamma_pos_values": [0.1, None], "gamma_neg_values": [-0.1]},
                    file,
                    indent=2,
                )
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "polarity_imbalance_by_latent": [0.2, None],
                        "gamma_summary": {"coverage": {"pos_fraction_valid": 1.0}},
                        "mse": 0.5,
                        "r2": 0.1,
                        "mean_l0": 16.0,
                        "dead_fraction": 0.2,
                    },
                    file,
                    indent=2,
                )
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
                json.dump(
                    {
                        "history": [
                            {
                                "step": 1,
                                "delta_pos_mean": 0.1,
                                "delta_neg_mean": 0.2,
                                "theta_mean": 0.3,
                                "jump_mean": 0.4,
                                "dead_zone_frac_mean": 0.1,
                            }
                        ]
                    },
                    file,
                    indent=2,
                )
            torch.save(
                {"state_dict": {"W_dec": sae_model.W_dec.detach().cpu()}},
                os.path.join(subrun_dir, "model.pt"),
            )
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(
                    {
                        "schema_version": 1,
                        "metadata": {},
                        "subruns": [
                            {
                                "variant_name": "smoke_variant",
                                "variant_slug": "smoke_variant",
                                "seed": 0,
                                "sweep_param": "lambda",
                                "sweep_value": 1e-3,
                                "mean_l0": 16.0,
                                "mse": 0.5,
                                "relative_dir": subrun["relative_subrun_dir"],
                            }
                        ],
                        "aggregates": [
                            {
                                "variant_slug": "smoke_variant",
                                "sweep_param": "lambda",
                                "sweep_value": 1e-3,
                                "num_seeds": 1,
                                "mean_l0_mean": 16.0,
                                "mse_mean": 0.5,
                                "dead_fraction_mean": 0.2,
                                "loss_recovered_mean": loss_payload["loss_recovered"],
                            }
                        ],
                    },
                    file,
                    indent=2,
                )
            plot_result = plotting_module.generate_plots_for_run(output_dir=tmpdir)

            # then
            self.assertIsNotNone(loss_payload)
            self.assertEqual(
                int(plot_result["plots_written"]),
                len(plotting_module.canonical_plot_paths(tmpdir)),
            )
