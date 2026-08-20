import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import torch

from sgnsae.experiments.llm_benchmark.config import LLMBenchmarkConfig
from sgnsae.experiments.llm_benchmark.feature_ablation import (
    compute_and_write_feature_ablation,
)
from sgnsae.experiments.llm_benchmark.loss_recovered import (
    _build_eval_tokens,
    _loss_for_mode,
)
from sgnsae.experiments.llm_benchmark.overhead import summarize_run_parameters
from sgnsae.experiments.llm_benchmark.plotting import (
    canonical_plot_paths,
    generate_plots_for_run,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)
from sgnsae.models import SAE


class _FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 99
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [int((ord(character) % 37) + 1) for character in str(text)]


class _TinyHookedLM(torch.nn.Module):
    def __init__(self, vocab_size: int, d_in: int) -> None:
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
        embedded = self.embedding(input_ids)
        hidden = self.hook_module(embedded)
        logits = self.lm_head(hidden)
        return {"logits": logits}


class _IdentityToySAE(torch.nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.W_dec = torch.nn.Parameter(torch.eye(d_in).unsqueeze(0))
        self.b_dec = torch.nn.Parameter(torch.zeros(1, d_in))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "x_hat": x.unsqueeze(0),
            "latents": x.unsqueeze(0),
        }


def _base_config(output_dir: str, data_files: List[str]) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": 0,
        "llm": {"model_id": "dummy-model"},
        "data": {
            "backend": "local_jsonl",
            "data_files": data_files,
            "text_field": "text",
        },
        "cache": {"max_seq_len": 8},
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "training": {"batch_size": 4, "mixed_precision": False},
        "variants": [
            {
                "name": "variant",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 6,
                "sparsity_coeff": 1e-3,
            }
        ],
    }


def _write_activation_cache(
    output_dir: str, d_in: int, train_count: int, val_count: int
) -> None:
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)
    train_acts = torch.randn(train_count, d_in, dtype=torch.float32)
    val_acts = torch.randn(val_count, d_in, dtype=torch.float32)
    torch.save({"acts": train_acts}, os.path.join(activations_dir, "shard_00000.pt"))
    torch.save({"acts": val_acts}, os.path.join(activations_dir, "shard_00001.pt"))
    manifest = {
        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
        "activation": {"d_in": int(d_in)},
        "hook": {"module_path_resolved": "hook_module", "layer": 0},
        "split": {
            "method": "shard_ranges",
            "train_shard_indices": [0],
            "val_shard_indices": [1],
            "test_shard_indices": [],
        },
        "shards": [
            {
                "filename": "shard_00000.pt",
                "num_vectors": int(train_count),
                "start_vector_index": 0,
                "end_vector_index": int(train_count),
            },
            {
                "filename": "shard_00001.pt",
                "num_vectors": int(val_count),
                "start_vector_index": int(train_count),
                "end_vector_index": int(train_count + val_count),
            },
        ],
    }
    with open(os.path.join(activations_dir, "activations_manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)


class TestLLMBenchmarkTasksNineToTwelve(unittest.TestCase):
    def test_loss_recovered_eval_token_generation_is_deterministic(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "texts.jsonl")
            with open(data_path, "w") as file:
                for line in (
                    "alpha beta gamma",
                    "delta epsilon",
                    "zeta eta theta iota",
                ):
                    file.write(json.dumps({"text": line}) + "\n")
            config_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_payload["loss_recovered"] = {
                "enabled": True,
                "max_eval_sequences": 4,
                "llm_batch_size": 2,
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.loss_recovered._load_tokenizer",
                return_value=_FakeTokenizer(),
            ):
                first_payload, _ = _build_eval_tokens(config=config)
                second_payload, _ = _build_eval_tokens(config=config)

            # then
            self.assertTrue(
                torch.equal(first_payload["input_ids"], second_payload["input_ids"])
            )
            self.assertTrue(
                torch.equal(
                    first_payload["attention_mask"], second_payload["attention_mask"]
                )
            )

    def test_loss_recovered_hook_patching_smoke(self) -> None:
        # given
        torch.manual_seed(0)
        model = _TinyHookedLM(vocab_size=17, d_in=8)
        sae_model = _IdentityToySAE(d_in=8)
        input_ids = torch.randint(0, 17, (4, 12), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # when
        original_loss, _ = _loss_for_mode(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_batch_size=2,
            mode="none",
        )
        ablated_loss, _ = _loss_for_mode(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_batch_size=2,
            hook_module=model.hook_module,
            mode="ablate",
        )
        reconstructed_loss, _ = _loss_for_mode(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_batch_size=2,
            hook_module=model.hook_module,
            mode="reconstruct",
            sae_model=sae_model,
        )

        # then
        self.assertTrue(torch.isfinite(torch.tensor(original_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(ablated_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(reconstructed_loss)))
        self.assertAlmostEqual(original_loss, reconstructed_loss, places=6)
        self.assertGreater(abs(ablated_loss - original_loss), 1e-6)

    def test_feature_ablation_uses_selected_subset_only(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "texts.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "placeholder"}) + "\n")
            _write_activation_cache(
                output_dir=tmpdir, d_in=4, train_count=8, val_count=8
            )
            config_payload = _base_config(output_dir=tmpdir, data_files=[data_path])
            config_payload["evaluation"] = {
                "alive_threshold": 0.0,
                "alive_min_fraction": 0.0,
                "min_gamma_samples": 1,
                "gamma_denominator_epsilon": 1e-12,
                "r2_denominator_epsilon": 1e-12,
            }
            config_payload["feature_ablation"] = {
                "enabled": True,
                "num_features": 2,
                "max_eval_batches": 2,
                "metric": "delta_mse",
                "selection": "top_mean_abs",
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)
            model = SAE(
                n_instances=1,
                d_in=4,
                d_hidden=6,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            subrun = {
                "variant_name": "variant",
                "variant_slug": "variant",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/variant/seed_0/lambda_1e-03",
            }
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)

            # when
            payload = compute_and_write_feature_ablation(
                config=config,
                subrun=subrun,
                output_dir=tmpdir,
                subrun_dir=subrun_dir,
                model=model,
                overwrite=True,
            )

            # then
            self.assertIsNotNone(payload)
            self.assertEqual(len(payload["features"]), 2)
            self.assertEqual(len(payload["impacts"]["delta_mse"]), 2)
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "feature_ablation.json"))
            )

    def test_plotting_stage_writes_canonical_files_even_without_data(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison_summary = {
                "schema_version": 1,
                "metadata": {},
                "subruns": [],
                "aggregates": [],
            }
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump(comparison_summary, file, indent=2)

            # when
            result = generate_plots_for_run(output_dir=tmpdir)

            # then
            self.assertEqual(
                int(result["plots_written"]), len(canonical_plot_paths(tmpdir))
            )
            for path in canonical_plot_paths(tmpdir).values():
                self.assertTrue(os.path.isfile(path))
            with open(os.path.join(tmpdir, "comparison_summary.json"), "r") as file:
                updated_summary = json.load(file)
            self.assertIn("matched_l0_anchors", updated_summary)

    def test_overhead_summary_reads_run_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "activations"), exist_ok=True)
            resolved_config = {
                "variants": [
                    {
                        "name": "variant_a",
                        "variant_slug": "variant_a",
                        "sae_class": "SAE",
                        "sign_aware": False,
                        "d_hidden": 8,
                        "sparsity_coeff": [1e-3, 1e-4],
                    },
                    {
                        "name": "variant_b",
                        "variant_slug": "variant_b",
                        "sae_class": "TopKSAE",
                        "sign_aware": False,
                        "d_hidden": 8,
                        "topk_k": [4, 8],
                    },
                ]
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as file:
                json.dump(resolved_config, file, indent=2)
            manifest = {"activation": {"d_in": 16}}
            with open(
                os.path.join(tmpdir, "activations", "activations_manifest.json"), "w"
            ) as file:
                json.dump(manifest, file, indent=2)

            # when
            summary = summarize_run_parameters(output_dir=tmpdir)

            # then
            self.assertEqual(int(summary["schema_version"]), 1)
            self.assertEqual(int(summary["d_in"]), 16)
            self.assertEqual(len(summary["variants"]), 2)
            for variant_summary in summary["variants"]:
                self.assertGreater(int(variant_summary["param_count"]), 0)
                self.assertIn("float16", variant_summary["estimated_memory_bytes"])
