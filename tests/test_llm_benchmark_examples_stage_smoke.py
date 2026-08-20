import json
import os
import tempfile
import types
import unittest
from typing import Any, Dict, Optional
from unittest.mock import patch

import torch

import sgnsae.experiments.llm_benchmark.examples as examples_module
from sgnsae.experiments.llm_benchmark.config import (
    LLMBenchmarkConfig,
    resolve_benchmark_config,
)
from sgnsae.experiments.llm_benchmark.runner import _run_examples_stage
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
    PAIRED_DUPLICATE_CASE_STUDIES_SCHEMA_VERSION,
)
from sgnsae.experiments.llm_benchmark.training import build_model_for_subrun
from sgnsae.models import SAE


class DummyTokenizer:
    def decode(
        self,
        token_ids: Any,
        clean_up_tokenization_spaces: Optional[bool] = None,
    ) -> str:
        del clean_up_tokenization_spaces
        if isinstance(token_ids, list) and token_ids:
            return f" t{int(token_ids[0])}"
        return ""


def _base_config(output_dir: str, data_path: str, sign_aware: bool) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": 0,
        "llm": {"model_id": "sshleifer/tiny-gpt2"},
        "data": {
            "backend": "local_jsonl",
            "data_files": [data_path],
            "text_field": "text",
        },
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "cache": {
            "max_seq_len": 4,
            "vectors_per_shard": 8,
            "store_tokens": True,
            "store_positions": True,
            "store_sequence_ids": True,
            "split_method": "shard_ranges",
        },
        "training": {
            "batch_size": 2,
            "train_steps": 2,
            "mixed_precision": False,
            "resample_dead_latents": False,
        },
        "latent_examples": {
            "enabled": True,
            "sample_size": 8,
            "max_latents": 4,
            "max_examples_per_latent": 2,
            "activation_threshold": 0.0,
            "latent_selection": "top_by_max_abs",
            "target_subruns": "explicit",
            "highlight_markers": ["<<", ">>"],
        },
        "variants": [
            {
                "name": "variant",
                "sae_class": "SAE",
                "sign_aware": bool(sign_aware),
                "d_hidden": 2,
                "sparsity_coeff": 1e-3,
            }
        ],
    }


def _write_manifest_and_shards(
    output_dir: str,
    train_acts: torch.Tensor,
    val_acts: torch.Tensor,
    val_tokens: torch.Tensor,
    include_tokens: bool = True,
    include_positions: bool = True,
    include_sequence_ids: bool = True,
) -> Dict[str, Any]:
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    torch.save({"acts": train_acts}, os.path.join(activations_dir, "shard_00000.pt"))
    val_payload: Dict[str, Any] = {"acts": val_acts}
    if include_tokens:
        val_payload["tokens"] = val_tokens
    if include_positions:
        val_payload["positions"] = torch.arange(
            int(val_tokens.numel()), dtype=torch.int64
        )
    if include_sequence_ids:
        val_payload["sequence_id"] = torch.zeros(
            int(val_tokens.numel()), dtype=torch.int64
        )
    torch.save(val_payload, os.path.join(activations_dir, "shard_00001.pt"))

    manifest = {
        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
        "activation": {"d_in": int(train_acts.shape[1])},
        "split": {
            "method": "shard_ranges",
            "train_shard_indices": [0],
            "val_shard_indices": [1],
            "test_shard_indices": [],
        },
        "store_tokens": bool(include_tokens),
        "store_positions": bool(include_positions),
        "store_sequence_ids": bool(include_sequence_ids),
        "shards": [
            {
                "filename": "shard_00000.pt",
                "num_vectors": int(train_acts.shape[0]),
                "start_vector_index": 0,
                "end_vector_index": int(train_acts.shape[0]),
            },
            {
                "filename": "shard_00001.pt",
                "num_vectors": int(val_acts.shape[0]),
                "start_vector_index": int(train_acts.shape[0]),
                "end_vector_index": int(train_acts.shape[0] + val_acts.shape[0]),
            },
        ],
    }
    with open(os.path.join(activations_dir, "activations_manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def _build_config_and_resolved(
    output_dir: str,
    data_path: str,
    sign_aware: bool,
) -> Dict[str, Any]:
    config = LLMBenchmarkConfig.model_validate(
        _base_config(output_dir=output_dir, data_path=data_path, sign_aware=sign_aware)
    )
    resolved = resolve_benchmark_config(
        config=config,
        stages_override=["examples"],
        validate_only=False,
    )
    return {"config": config, "resolved": resolved, "subrun": resolved["subruns"][0]}


def _write_completed_training_metrics(output_dir: str, subrun: Dict[str, Any]) -> None:
    subrun_dir = os.path.join(output_dir, str(subrun["relative_subrun_dir"]))
    os.makedirs(subrun_dir, exist_ok=True)
    with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
        json.dump({"status": "completed"}, file, indent=2)


def _write_checkpoint_for_subrun(
    config: LLMBenchmarkConfig,
    resolved: Dict[str, Any],
    subrun: Dict[str, Any],
    manifest: Dict[str, Any],
    output_dir: str,
    sign_aware: bool,
) -> str:
    model, _, _, _ = build_model_for_subrun(
        config=config,
        resolved_config=resolved,
        subrun=subrun,
        manifest=manifest,
        device="cpu",
    )
    with torch.no_grad():
        model.W_dec.fill_(0.0)
        if sign_aware:
            model.W_dec[0, 0, 0] = 1.0
            model.W_dec[0, 1, 1] = 1.0
            model.b_enc.fill_(0.0)
            model.b_dec.fill_(0.0)
            if hasattr(model, "act_offset"):
                model.act_offset.fill_(0.0)
            if hasattr(model, "act_offset_right"):
                model.act_offset_right.fill_(0.0)
            if hasattr(model, "act_offset_left"):
                model.act_offset_left.fill_(0.0)
        else:
            model.W_dec[0, 0, 0] = 1.0
            model.W_dec[0, 0, 1] = -1.0
            model.b_enc.fill_(0.0)
            model.b_dec.fill_(0.0)

    subrun_dir = os.path.join(output_dir, str(subrun["relative_subrun_dir"]))
    os.makedirs(subrun_dir, exist_ok=True)
    _write_completed_training_metrics(output_dir=output_dir, subrun=subrun)
    checkpoint_path = os.path.join(subrun_dir, "model.pt")
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


class TestExamplesStageSmoke(unittest.TestCase):
    def test_dummy_tokenizer_decode_returns_empty_string_for_empty_token_ids(
        self,
    ) -> None:
        # given
        tokenizer = DummyTokenizer()

        # when
        decoded = tokenizer.decode([])

        # then
        self.assertEqual(decoded, "")

    def test_write_manifest_and_shards_can_skip_optional_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # given
            train_acts = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
            val_acts = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
            val_tokens = torch.tensor([7], dtype=torch.int64)

            # when
            _ = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=train_acts,
                val_acts=val_acts,
                val_tokens=val_tokens,
                include_tokens=False,
                include_positions=False,
                include_sequence_ids=False,
            )
            shard_payload = torch.load(
                os.path.join(tmpdir, "activations", "shard_00001.pt"),
                map_location="cpu",
            )

            # then
            self.assertNotIn("tokens", shard_payload)
            self.assertNotIn("positions", shard_payload)
            self.assertNotIn("sequence_id", shard_payload)

    def test_write_checkpoint_for_subrun_supports_single_activation_offset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # given
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config_payload = _base_config(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=True,
            )
            config_payload["variants"][0]["single_activation_offset"] = True
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(
                config=config,
                stages_override=["examples"],
                validate_only=False,
            )
            subrun = resolved["subruns"][0]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor([[0.1, -0.1]], dtype=torch.float32),
                val_acts=torch.tensor([[0.2, -0.2]], dtype=torch.float32),
                val_tokens=torch.tensor([1], dtype=torch.int64),
            )

            # when
            checkpoint_path = _write_checkpoint_for_subrun(
                config=config,
                resolved=resolved,
                subrun=subrun,
                manifest=manifest,
                output_dir=tmpdir,
                sign_aware=True,
            )

            # then
            self.assertTrue(os.path.isfile(checkpoint_path))

    def test_read_json_and_output_subdir_path_error_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = os.path.join(tmpdir, "payload.json")
            with open(payload_path, "w") as file:
                json.dump({"ok": True}, file)

            # when
            payload = examples_module._read_json(payload_path)

            # then
            self.assertEqual(payload, {"ok": True})

            # when / then
            with self.assertRaises(ValueError):
                _ = examples_module._output_subdir_path(tmpdir, "")
            with self.assertRaises(ValueError):
                _ = examples_module._output_subdir_path(tmpdir, "../escape")
            with patch(
                "sgnsae.experiments.llm_benchmark.examples.os.path.commonpath",
                side_effect=ValueError("different drives"),
            ):
                with self.assertRaises(ValueError):
                    _ = examples_module._output_subdir_path(tmpdir, "models/subrun")

    def test_resolve_val_locations_handles_short_circuits_and_errors(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_base = {
                "split": {"method": "shard_ranges", "val_shard_indices": [0]},
                "shards": [
                    {
                        "filename": "shard_00000.pt",
                        "num_vectors": 4,
                        "start_vector_index": 0,
                        "end_vector_index": 4,
                    }
                ],
            }

            # when
            empty = examples_module._resolve_val_locations(
                manifest=manifest_base,
                output_dir=tmpdir,
                max_samples=0,
            )

            # then
            self.assertEqual(empty, [])

            # when
            limited = examples_module._resolve_val_locations(
                manifest=manifest_base,
                output_dir=tmpdir,
                max_samples=2,
            )

            # then
            self.assertEqual(limited, [(0, 0), (0, 1)])

            # given
            bad_manifest = {
                "split": {"method": "bad"},
                "shards": [],
            }

            # when / then
            with self.assertRaises(ValueError):
                _ = examples_module._resolve_val_locations(
                    manifest=bad_manifest,
                    output_dir=tmpdir,
                    max_samples=1,
                )

    def test_validate_examples_cache_metadata_returns_for_non_text(self) -> None:
        # given
        config = types.SimpleNamespace(modality="image")

        # when
        examples_module.validate_examples_cache_metadata(config=config, output_dir="/")

        # then
        self.assertTrue(True)

    def test_validate_examples_cache_metadata_raises_without_tokens(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            torch.save(
                {"acts": torch.zeros((2, 2), dtype=torch.float32)},
                os.path.join(activations_dir, "shard_00000.pt"),
            )
            manifest = {
                "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                "activation": {"d_in": 2},
                "split": {
                    "method": "shard_ranges",
                    "train_shard_indices": [],
                    "val_shard_indices": [0],
                    "test_shard_indices": [],
                },
                "store_tokens": False,
                "shards": [
                    {
                        "filename": "shard_00000.pt",
                        "num_vectors": 2,
                        "start_vector_index": 0,
                        "end_vector_index": 2,
                    }
                ],
            }
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(manifest, file, indent=2)
            config = types.SimpleNamespace(modality="text")

            # when / then
            with self.assertRaises(ValueError) as context:
                examples_module.validate_examples_cache_metadata(
                    config=config,
                    output_dir=tmpdir,
                )
            self.assertIn("cache.store_tokens=true", str(context.exception))

    def test_generate_latent_examples_reuses_existing_output(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "hello"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=True,
            )
            config = payload["config"]
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            subrun_dir = os.path.join(tmpdir, str(subrun["relative_subrun_dir"]))
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(output_dir=tmpdir, subrun=subrun)
            with open(os.path.join(subrun_dir, "latent_examples.json"), "w") as file:
                json.dump({"already": "present"}, file)

            # when
            result = examples_module.generate_latent_examples(
                config=config,
                resolved_config=resolved,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            self.assertTrue(bool(result["reused"]))
            self.assertTrue(os.path.isfile(result["latent_examples_path"]))

    def test_generate_latent_examples_outputs_sign_aware_examples(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=True,
            )
            config = payload["config"]
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor(
                    [[0.1, -0.1], [0.2, -0.2]], dtype=torch.float32
                ),
                val_acts=torch.tensor(
                    [[2.0, -2.0], [-1.0, 1.0], [3.0, -3.0], [-2.0, 2.0]],
                    dtype=torch.float32,
                ),
                val_tokens=torch.tensor([11, 12, 13, 14], dtype=torch.int64),
            )
            _ = _write_checkpoint_for_subrun(
                config=config,
                resolved=resolved,
                subrun=subrun,
                manifest=manifest,
                output_dir=tmpdir,
                sign_aware=True,
            )

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                result = examples_module.generate_latent_examples(
                    config=config,
                    resolved_config=resolved,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=True,
                )

            # then
            self.assertFalse(bool(result["reused"]))
            with open(result["latent_examples_path"], "r") as file:
                payload_json = json.load(file)
            self.assertEqual(payload_json["examples"]["kind"], "sign_aware")
            self.assertTrue(bool(payload_json["examples"]["latents"]))
            first_entry = payload_json["examples"]["latents"][0]
            self.assertIn("positive_examples", first_entry)
            self.assertIn("negative_examples", first_entry)
            contexts = [
                entry["context"]
                for entry in first_entry["positive_examples"]
                + first_entry["negative_examples"]
            ]
            highlighted_contexts = [
                context for context in contexts if "<<" in context and ">>" in context
            ]
            self.assertTrue(bool(highlighted_contexts))
            histogram_entries = payload_json["assets"]["activation_histograms"]
            self.assertTrue(bool(histogram_entries))
            for histogram_entry in histogram_entries:
                histogram_path = os.path.join(
                    result["subrun_dir"], str(histogram_entry["path"])
                )
                self.assertTrue(os.path.isfile(histogram_path))

    def test_generate_latent_examples_outputs_non_negative_duplicate_pairs(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=False,
            )
            config = payload["config"]
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor([[0.1, 0.0], [0.2, 0.0]], dtype=torch.float32),
                val_acts=torch.tensor(
                    [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
                    dtype=torch.float32,
                ),
                val_tokens=torch.tensor([21, 22, 23, 24], dtype=torch.int64),
            )
            _ = _write_checkpoint_for_subrun(
                config=config,
                resolved=resolved,
                subrun=subrun,
                manifest=manifest,
                output_dir=tmpdir,
                sign_aware=False,
            )

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                result = examples_module.generate_latent_examples(
                    config=config,
                    resolved_config=resolved,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=True,
                )

            # then
            with open(result["latent_examples_path"], "r") as file:
                payload_json = json.load(file)
            self.assertEqual(payload_json["examples"]["kind"], "non_negative")
            self.assertTrue(bool(payload_json["examples"]["duplicate_pairs"]))
            first_pair = payload_json["examples"]["duplicate_pairs"][0]
            self.assertIn("latent_a_examples", first_pair)
            self.assertIn("latent_b_examples", first_pair)
            self.assertLessEqual(
                len(payload_json["selection"]["selected_latents"]),
                int(payload_json["selection"]["max_latents"]),
            )

    def test_generate_latent_examples_handles_missing_optional_token_metadata(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=True,
            )
            config = payload["config"]
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor(
                    [[0.1, -0.1], [0.2, -0.2]], dtype=torch.float32
                ),
                val_acts=torch.tensor(
                    [[2.0, -2.0], [-1.0, 1.0], [3.0, -3.0], [-2.0, 2.0]],
                    dtype=torch.float32,
                ),
                val_tokens=torch.tensor([11, 12, 13, 14], dtype=torch.int64),
                include_positions=False,
                include_sequence_ids=False,
            )
            _ = _write_checkpoint_for_subrun(
                config=config,
                resolved=resolved,
                subrun=subrun,
                manifest=manifest,
                output_dir=tmpdir,
                sign_aware=True,
            )

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                result = examples_module.generate_latent_examples(
                    config=config,
                    resolved_config=resolved,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=True,
                )

            # then
            with open(result["latent_examples_path"], "r") as file:
                payload_json = json.load(file)
            first_entry = payload_json["examples"]["latents"][0]["positive_examples"][0]
            self.assertIsNone(first_entry["position"])
            self.assertIsNone(first_entry["sequence_id"])
            self.assertIn("<<", first_entry["context"])

    def test_generate_latent_examples_non_negative_requires_at_least_two_latents(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=False,
            )
            config = payload["config"].model_copy(
                update={
                    "latent_examples": payload["config"].latent_examples.model_copy(
                        update={"max_latents": 1}
                    )
                }
            )
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor([[0.1, 0.0], [0.2, 0.0]], dtype=torch.float32),
                val_acts=torch.tensor(
                    [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
                    dtype=torch.float32,
                ),
                val_tokens=torch.tensor([21, 22, 23, 24], dtype=torch.int64),
            )
            _ = _write_checkpoint_for_subrun(
                config=config,
                resolved=resolved,
                subrun=subrun,
                manifest=manifest,
                output_dir=tmpdir,
                sign_aware=False,
            )

            # when / then
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError) as context:
                    _ = examples_module.generate_latent_examples(
                        config=config,
                        resolved_config=resolved,
                        subrun=subrun,
                        output_dir=tmpdir,
                        overwrite=True,
                    )
            self.assertIn("max_latents must be >= 2", str(context.exception))

    def test_generate_latent_examples_non_negative_no_pairs_is_empty_but_valid(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=False,
            )
            config = payload["config"]
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor([[0.1, 0.0], [0.2, 0.0]], dtype=torch.float32),
                val_acts=torch.tensor(
                    [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
                    dtype=torch.float32,
                ),
                val_tokens=torch.tensor([21, 22, 23, 24], dtype=torch.int64),
            )
            model, _, _, _ = build_model_for_subrun(
                config=config,
                resolved_config=resolved,
                subrun=subrun,
                manifest=manifest,
                device="cpu",
            )
            with torch.no_grad():
                model.W_dec.fill_(0.0)
                model.W_dec[0, 0, 0] = 1.0
                model.W_dec[0, 1, 1] = 1.0
                model.b_enc.fill_(0.0)
                model.b_dec.fill_(0.0)
            subrun_dir = os.path.join(tmpdir, str(subrun["relative_subrun_dir"]))
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(output_dir=tmpdir, subrun=subrun)
            checkpoint_path = os.path.join(subrun_dir, "model.pt")
            torch.save({"state_dict": model.state_dict()}, checkpoint_path)

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                result = examples_module.generate_latent_examples(
                    config=config,
                    resolved_config=resolved,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=True,
                )

            # then
            with open(result["latent_examples_path"], "r") as file:
                payload_json = json.load(file)
            self.assertEqual(payload_json["examples"]["kind"], "non_negative")
            self.assertEqual(payload_json["examples"]["duplicate_pairs"], [])
            self.assertTrue(bool(payload_json["examples"]["messages"]))

    def test_generate_latent_examples_delegates_non_text_to_legacy_path(self) -> None:
        # given
        config = types.SimpleNamespace(modality="image")

        # when
        with patch(
            "sgnsae.experiments.llm_benchmark.examples.generate_legacy_latent_examples_for_subrun",
            return_value={"delegated": True},
        ) as legacy_call:
            result = examples_module.generate_latent_examples(
                config=config,
                resolved_config={},
                subrun={},
                output_dir="/tmp",
                overwrite=False,
            )

        # then
        self.assertEqual(result, {"delegated": True})
        self.assertEqual(legacy_call.call_count, 1)

    def test_resolve_val_locations_stored_indices_branch(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            indices_path = os.path.join(activations_dir, "split_indices.pt")
            torch.save(
                {
                    "train": torch.tensor([0, 1], dtype=torch.int64),
                    "val": torch.tensor([2, 3], dtype=torch.int64),
                    "test": torch.tensor([], dtype=torch.int64),
                },
                indices_path,
            )
            manifest = {
                "activation": {"d_in": 2},
                "split": {
                    "method": "stored_indices",
                    "indices_filename": "split_indices.pt",
                },
                "shards": [
                    {
                        "filename": "shard_00000.pt",
                        "num_vectors": 2,
                        "start_vector_index": 0,
                        "end_vector_index": 2,
                    },
                    {
                        "filename": "shard_00001.pt",
                        "num_vectors": 2,
                        "start_vector_index": 2,
                        "end_vector_index": 4,
                    },
                ],
            }

            # when
            locations = examples_module._resolve_val_locations(
                manifest=manifest,
                output_dir=tmpdir,
                max_samples=8,
            )

            # then
            self.assertEqual(locations, [(1, 0), (1, 1)])

            # when
            truncated = examples_module._resolve_val_locations(
                manifest=manifest,
                output_dir=tmpdir,
                max_samples=1,
            )

            # then
            self.assertEqual(truncated, [(1, 0)])

            # given
            torch.save({"train": torch.tensor([0], dtype=torch.int64)}, indices_path)

            # when / then
            with self.assertRaises(ValueError):
                _ = examples_module._resolve_val_locations(
                    manifest=manifest,
                    output_dir=tmpdir,
                    max_samples=8,
                )

    def test_load_text_records_covers_empty_missing_tokens_and_optional_metadata(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            torch.save(
                {"acts": torch.zeros((2, 2), dtype=torch.float32)},
                os.path.join(activations_dir, "shard_00000.pt"),
            )
            manifest = {
                "activation": {"d_in": 2},
                "split": {"method": "shard_ranges", "val_shard_indices": [0]},
                "shards": [
                    {
                        "filename": "shard_00000.pt",
                        "num_vectors": 2,
                        "start_vector_index": 0,
                        "end_vector_index": 2,
                    }
                ],
            }

            # when / then
            with self.assertRaises(ValueError):
                _ = examples_module._load_text_records(
                    manifest=manifest,
                    output_dir=tmpdir,
                    max_samples=1,
                )

            # given
            torch.save(
                {
                    "acts": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                    "tokens": torch.tensor([10, 11], dtype=torch.int64),
                },
                os.path.join(activations_dir, "shard_00000.pt"),
            )

            # when
            activations, token_ids, positions, sequence_ids = (
                examples_module._load_text_records(
                    manifest=manifest,
                    output_dir=tmpdir,
                    max_samples=2,
                )
            )

            # then
            self.assertEqual(tuple(activations.shape), (2, 2))
            self.assertEqual(token_ids, [10, 11])
            self.assertEqual(positions, [None, None])
            self.assertEqual(sequence_ids, [None, None])

            # when
            empty_activations, _, _, _ = examples_module._load_text_records(
                manifest=manifest,
                output_dir=tmpdir,
                max_samples=0,
            )

            # then
            self.assertEqual(tuple(empty_activations.shape), (0, 2))

    def test_flatten_and_compute_latents_error_branches(self) -> None:
        # given
        tensor_2d = torch.zeros((2, 3), dtype=torch.float32)
        tensor_4d = torch.zeros((1, 2, 3, 4), dtype=torch.float32)

        # when
        flattened_2d = examples_module._flatten_single_instance_latents(tensor_2d)
        flattened_4d = examples_module._flatten_single_instance_latents(tensor_4d)

        # then
        self.assertEqual(tuple(flattened_2d.shape), (2, 3))
        self.assertEqual(tuple(flattened_4d.shape), (6, 4))

        # given
        class WrongWidthModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {
                    "latents": torch.zeros((1, int(x.shape[0]), 4), dtype=torch.float32)
                }

        # when / then
        with self.assertRaises(ValueError):
            _ = examples_module._compute_latents(
                model=WrongWidthModel(),
                activations=torch.ones((2, 2), dtype=torch.float32),
                runtime_device="cpu",
                d_hidden=2,
            )

        # when
        empty = examples_module._compute_latents(
            model=WrongWidthModel(),
            activations=torch.zeros((0, 2), dtype=torch.float32),
            runtime_device="cpu",
            d_hidden=2,
        )

        # then
        self.assertEqual(tuple(empty.shape), (0, 2))

    def test_select_latents_and_decode_fallback_branches(self) -> None:
        # given
        config_payload = {
            "output_dir": "/tmp/out",
            "device": "cpu",
            "seed": 0,
            "llm": {"model_id": "sshleifer/tiny-gpt2"},
            "data": {
                "backend": "local_jsonl",
                "data_files": ["/tmp/dummy.jsonl"],
                "text_field": "text",
            },
            "hook": {"hook_name": "mlp_out", "layer": 0},
            "variants": [
                {
                    "name": "variant",
                    "sae_class": "SAE",
                    "sign_aware": True,
                    "d_hidden": 3,
                    "sparsity_coeff": 1e-3,
                }
            ],
            "latent_examples": {
                "enabled": True,
                "latent_selection": "explicit",
                "explicit_latents": [0, 2],
                "max_latents": 2,
                "max_examples_per_latent": 2,
                "sample_size": 8,
                "activation_threshold": 0.1,
                "target_subruns": "explicit",
            },
        }
        config = LLMBenchmarkConfig.model_validate(config_payload)
        latents = torch.tensor([[0.0, 0.2, 0.3], [0.0, 0.5, 0.6]], dtype=torch.float32)

        # when
        explicit = examples_module._select_latents(
            config=config,
            latents=latents,
            alive_indices=[0, 2],
            d_hidden=3,
        )

        # then
        self.assertEqual(explicit, [0, 2])

        # when
        top_freq_config = config.model_copy(
            update={
                "latent_examples": config.latent_examples.model_copy(
                    update={"latent_selection": "top_by_freq", "explicit_latents": None}
                )
            }
        )
        top_freq = examples_module._select_latents(
            config=top_freq_config,
            latents=latents,
            alive_indices=None,
            d_hidden=3,
        )

        # then
        self.assertTrue(bool(top_freq))

        # given
        comparison_latents = torch.tensor(
            [[9.0, 3.0], [0.0, 3.0], [0.0, 3.0], [0.0, 3.0]], dtype=torch.float32
        )
        top_by_mean_config = config.model_copy(
            update={
                "latent_examples": config.latent_examples.model_copy(
                    update={
                        "latent_selection": "top_by_mean_abs",
                        "explicit_latents": None,
                        "max_latents": 1,
                        "activation_threshold": 0.0,
                    }
                )
            }
        )
        top_by_max_config = config.model_copy(
            update={
                "latent_examples": config.latent_examples.model_copy(
                    update={
                        "latent_selection": "top_by_max_abs",
                        "explicit_latents": None,
                        "max_latents": 1,
                        "activation_threshold": 0.0,
                    }
                )
            }
        )

        # when
        selected_by_mean = examples_module._select_latents(
            config=top_by_mean_config,
            latents=comparison_latents,
            alive_indices=[0, 1],
            d_hidden=2,
        )
        selected_by_max = examples_module._select_latents(
            config=top_by_max_config,
            latents=comparison_latents,
            alive_indices=[0, 1],
            d_hidden=2,
        )

        # then
        self.assertEqual(selected_by_mean, [1])
        self.assertEqual(selected_by_max, [0])

        # when
        max_zero_config = config.model_copy(
            update={
                "latent_examples": config.latent_examples.model_copy(
                    update={"max_latents": 0}
                )
            }
        )
        selected_none = examples_module._select_latents(
            config=max_zero_config,
            latents=latents,
            alive_indices=None,
            d_hidden=3,
        )

        # then
        self.assertEqual(selected_none, [])

        # given
        class FallbackTokenizer:
            def decode(self, token_ids: Any) -> str:
                return f"tok_{int(token_ids[0])}"

        # when
        decoded = examples_module._decode_token(FallbackTokenizer(), 5, {})

        # then
        self.assertEqual(decoded, "tok_5")

        # given
        class EmptyTokenizer:
            def decode(
                self, token_ids: Any, clean_up_tokenization_spaces: Any = None
            ) -> str:
                del token_ids
                del clean_up_tokenization_spaces
                return ""

        class BrokenTokenizer:
            def decode(
                self, token_ids: Any, clean_up_tokenization_spaces: Any = None
            ) -> str:
                del token_ids
                del clean_up_tokenization_spaces
                raise RuntimeError("decode failed")

        # when
        empty_decoded = examples_module._decode_token(EmptyTokenizer(), 9, {})
        broken_decoded = examples_module._decode_token(BrokenTokenizer(), 10, {})

        # then
        self.assertEqual(empty_decoded, "<tok:9>")
        self.assertEqual(broken_decoded, "<tok:10>")

    def test_context_and_collect_examples_branches(self) -> None:
        # given
        token_ids = [1, 2, 3, 4]
        sequence_ids = [0, 0, 1, 1]
        positions = [0, 1, 0, 1]
        latents = torch.tensor(
            [[1.0, -1.0], [0.0, 1.5], [-2.0, 0.1], [1.2, -1.3]],
            dtype=torch.float32,
        )

        # when
        context = examples_module._context_text(
            tokenizer=DummyTokenizer(),
            token_ids=token_ids,
            sequence_ids=sequence_ids,
            center_index=1,
            markers=("<<", ">>"),
            decode_cache={},
        )
        entries_no_negative, _ = examples_module._collect_examples_for_latents(
            tokenizer=DummyTokenizer(),
            token_ids=token_ids,
            positions=positions,
            sequence_ids=sequence_ids,
            latents=latents,
            latent_indices=[0],
            max_examples_per_latent=2,
            activation_threshold=0.0,
            include_negative=False,
            markers=("<<", ">>"),
        )

        # then
        self.assertIn("<<", context)
        self.assertEqual(entries_no_negative[0]["negative_examples"], [])

        # when
        context_without_sequence = examples_module._context_text(
            tokenizer=DummyTokenizer(),
            token_ids=token_ids,
            sequence_ids=[None, None, None, None],
            center_index=1,
            markers=("<<", ">>"),
            decode_cache={},
        )

        # then
        self.assertIn("<<", context_without_sequence)

    def test_find_duplicate_pairs_branch_coverage(self) -> None:
        # given
        class NoDecoder(torch.nn.Module):
            pass

        # when
        none_pairs = examples_module._find_duplicate_pairs(
            model=NoDecoder(),
            alive_indices=None,
            max_pairs=4,
        )

        # then
        self.assertEqual(none_pairs, [])

        # given
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.W_dec = torch.nn.Parameter(
                    torch.zeros((1, 2, 2), dtype=torch.float32)
                )
                with torch.no_grad():
                    self.W_dec[0, 0, 0] = 1.0
                    self.W_dec[0, 0, 1] = 0.99

        # when
        near_pairs = examples_module._find_duplicate_pairs(
            model=TinyModel(),
            alive_indices=[0],
            max_pairs=2,
        )

        # then
        self.assertEqual(near_pairs, [])

        # when
        no_duplicate_pairs = examples_module._find_duplicate_pairs(
            model=TinyModel(),
            alive_indices=None,
            max_pairs=2,
        )

        # then
        self.assertEqual(no_duplicate_pairs, [])

        # when
        original_limit = examples_module._MAX_DUPLICATE_CANDIDATES
        examples_module._MAX_DUPLICATE_CANDIDATES = 1
        try:
            truncated_pairs = examples_module._find_duplicate_pairs(
                model=TinyModel(),
                alive_indices=None,
                max_pairs=2,
            )
        finally:
            examples_module._MAX_DUPLICATE_CANDIDATES = original_limit

        # then
        self.assertEqual(truncated_pairs, [])

    def test_examples_helper_branches_for_histograms_and_case_studies(self) -> None:
        # given
        class TypeErrorThenCrashTokenizer:
            def decode(self, token_ids: Any) -> str:
                del token_ids
                raise RuntimeError("decode failed")

        decoded = examples_module._decode_token(TypeErrorThenCrashTokenizer(), 42, {})
        self.assertEqual(decoded, "<tok:42>")

        empty_values = examples_module._finite_values(
            torch.zeros((0,), dtype=torch.float32)
        )
        self.assertEqual(int(empty_values.numel()), 0)
        self.assertEqual(
            examples_module._resolve_histogram_range(
                torch.zeros((0,), dtype=torch.float32),
                sign_aware=True,
            ),
            (-1.0, 1.0),
        )
        self.assertEqual(
            examples_module._resolve_histogram_range(
                torch.zeros((0,), dtype=torch.float32),
                sign_aware=False,
            ),
            (0.0, 1.0),
        )
        self.assertEqual(
            examples_module._resolve_histogram_range(
                torch.zeros((3,), dtype=torch.float32),
                sign_aware=True,
            ),
            (-1.0, 1.0),
        )
        low, high = examples_module._resolve_histogram_range(
            torch.full((5,), 3.0, dtype=torch.float32),
            sign_aware=False,
        )
        self.assertLess(low, high)
        empty_summary = examples_module._activation_summary(
            torch.zeros((0,), dtype=torch.float32)
        )
        self.assertEqual(empty_summary["count"], 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=False,
            )
            config = payload["config"]
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor([[0.1, 0.0]], dtype=torch.float32),
                val_acts=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                val_tokens=torch.tensor([1], dtype=torch.int64),
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

            with patch(
                "builtins.__import__",
                side_effect=_import_with_missing_transformers,
            ):
                with self.assertRaises(RuntimeError):
                    _ = examples_module._load_text_tokenizer(config=config)

            with self.assertRaises(ValueError):
                _ = examples_module._decoder_columns(model=torch.nn.Linear(1, 1))
            with self.assertRaises(ValueError):
                _ = examples_module._normalized_decoder_column(
                    model=torch.nn.Linear(1, 1),
                    latent_index=0,
                )
            with self.assertRaises(ValueError):
                _ = examples_module._normalized_decoder_columns_subset(
                    model=torch.nn.Linear(1, 1),
                    latent_indices=[0],
                )

            decoder_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=4,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            with torch.no_grad():
                decoder_model.W_dec.fill_(0.0)
                decoder_model.W_dec[0, 0, 0] = 1.0
                decoder_model.W_dec[0, 1, 2] = 2.0
            with self.assertRaises(ValueError):
                _ = examples_module._normalized_decoder_column(
                    model=decoder_model,
                    latent_index=6,
                )
            column = examples_module._normalized_decoder_column(
                model=decoder_model,
                latent_index=2,
            )
            self.assertTrue(torch.allclose(column.norm(), torch.tensor(1.0), atol=1e-6))
            subset_columns, subset_indices = (
                examples_module._normalized_decoder_columns_subset(
                    model=decoder_model,
                    latent_indices=[2, 0, 2, 7, True],  # type: ignore[list-item]
                )
            )
            self.assertEqual(subset_indices, [0, 2])
            self.assertEqual(tuple(subset_columns.shape), (2, 3))
            self.assertTrue(
                torch.allclose(
                    torch.linalg.vector_norm(subset_columns, dim=1),
                    torch.ones((2,), dtype=torch.float32),
                    atol=1e-6,
                )
            )
            empty_subset, empty_indices = (
                examples_module._normalized_decoder_columns_subset(
                    model=decoder_model,
                    latent_indices=[9],
                )
            )
            self.assertEqual(tuple(empty_subset.shape), (0, 3))
            self.assertEqual(empty_indices, [])

            with self.assertRaises(ValueError):
                _ = examples_module._load_checkpointed_subrun_model(
                    config=config,
                    resolved_config=resolved,
                    subrun=subrun,
                    output_dir=tmpdir,
                    manifest=manifest,
                    runtime_device="cpu",
                )

            subrun_dir = os.path.join(tmpdir, str(subrun["relative_subrun_dir"]))
            os.makedirs(subrun_dir, exist_ok=True)
            checkpoint_path = os.path.join(subrun_dir, "model.pt")
            torch.save({}, checkpoint_path)
            with self.assertRaises(ValueError):
                _ = examples_module._load_checkpointed_subrun_model(
                    config=config,
                    resolved_config=resolved,
                    subrun=subrun,
                    output_dir=tmpdir,
                    manifest=manifest,
                    runtime_device="cpu",
                )

            invalid_examples_path = os.path.join(subrun_dir, "latent_examples.json")
            with open(invalid_examples_path, "w") as file:
                json.dump([], file)
            self.assertIsNone(
                examples_module._load_subrun_examples_payload(
                    output_dir=tmpdir,
                    subrun=subrun,
                )
            )

            self.assertEqual(
                examples_module._eligible_signed_latent_entries(
                    sign_aware_payload={"examples": []}
                ),
                {},
            )
            self.assertEqual(
                examples_module._eligible_signed_latent_entries(
                    sign_aware_payload={"examples": {"latents": "bad"}}
                ),
                {},
            )
            self.assertEqual(
                examples_module._eligible_signed_latent_entries(
                    sign_aware_payload={
                        "examples": {
                            "latents": [
                                "bad",
                                {"latent_index": True},
                                {
                                    "latent_index": 0,
                                    "positive_examples": "bad",
                                    "negative_examples": [],
                                },
                                {
                                    "latent_index": 1,
                                    "positive_examples": [],
                                    "negative_examples": [],
                                },
                            ]
                        }
                    }
                ),
                {},
            )

            direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
            columns = torch.eye(2, dtype=torch.float32)
            self.assertIsNone(
                examples_module._find_best_sign_latent_for_direction(
                    direction=direction,
                    sign_columns=columns,
                    eligible_latent_indices=[],
                )
            )
            self.assertIsNone(
                examples_module._find_best_sign_latent_for_direction(
                    direction=direction,
                    sign_columns=columns,
                    eligible_latent_indices=[9],
                )
            )

            non_text_result = examples_module.build_and_persist_paired_duplicate_case_studies(
                config=types.SimpleNamespace(modality="image"),  # type: ignore[arg-type]
                resolved_config={},
                selected_subruns=[],
                output_dir=tmpdir,
            )
            self.assertEqual(int(non_text_result["num_case_studies"]), 0)

    def test_paired_duplicate_case_studies_handles_invalid_duplicate_entries(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config_payload = _base_config(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=True,
            )
            config_payload["variants"] = [
                {
                    "name": "baseline",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 2,
                    "sparsity_coeff": 1e-3,
                },
                {
                    "name": "signed",
                    "sae_class": "SAE",
                    "sign_aware": True,
                    "d_hidden": 2,
                    "sparsity_coeff": 1e-3,
                },
                {
                    "name": "signed_single",
                    "sae_class": "SAE",
                    "sign_aware": True,
                    "single_activation_offset": True,
                    "d_hidden": 2,
                    "sparsity_coeff": 1e-3,
                },
            ]
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(
                config=config,
                stages_override=["examples"],
                validate_only=False,
            )
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor([[0.1, 0.0]], dtype=torch.float32),
                val_acts=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                val_tokens=torch.tensor([1], dtype=torch.int64),
            )
            for subrun in resolved["subruns"]:
                model, _, _, variant_config = build_model_for_subrun(
                    config=config,
                    resolved_config=resolved,
                    subrun=subrun,
                    manifest=manifest,
                    device="cpu",
                )
                with torch.no_grad():
                    model.W_dec.fill_(0.0)
                    model.W_dec[0, 0, 0] = 1.0
                    model.W_dec[0, 0, 1] = -1.0
                    model.b_enc.fill_(0.0)
                    model.b_dec.fill_(0.0)
                    if bool(variant_config.get("sign_aware", False)):
                        if hasattr(model, "act_offset"):
                            model.act_offset.fill_(0.0)
                        if hasattr(model, "act_offset_right"):
                            model.act_offset_right.fill_(0.0)
                        if hasattr(model, "act_offset_left"):
                            model.act_offset_left.fill_(0.0)
                subrun_dir = os.path.join(tmpdir, str(subrun["relative_subrun_dir"]))
                os.makedirs(subrun_dir, exist_ok=True)
                torch.save(
                    {"state_dict": model.state_dict()},
                    os.path.join(subrun_dir, "model.pt"),
                )

            baseline_subrun = [
                subrun
                for subrun in resolved["subruns"]
                if str(subrun["variant_name"]) == "baseline"
            ][0]
            signed_subrun = [
                subrun
                for subrun in resolved["subruns"]
                if str(subrun["variant_name"]) == "signed"
            ][0]
            baseline_subrun_dir = os.path.join(
                tmpdir, str(baseline_subrun["relative_subrun_dir"])
            )
            signed_subrun_dir = os.path.join(
                tmpdir, str(signed_subrun["relative_subrun_dir"])
            )
            with open(
                os.path.join(baseline_subrun_dir, "latent_examples.json"), "w"
            ) as file:
                json.dump(
                    {
                        "examples": {
                            "kind": "non_negative",
                            "duplicate_pairs": [
                                "bad",
                                {"latent_a": True, "latent_b": 1},
                                {"latent_a": 9, "latent_b": 1},
                                {"latent_a": 0, "latent_b": 9},
                                {"latent_a": 0, "latent_b": 1},
                            ],
                        }
                    },
                    file,
                    indent=2,
                )
            with open(
                os.path.join(signed_subrun_dir, "latent_examples.json"), "w"
            ) as file:
                json.dump(
                    {
                        "examples": {
                            "kind": "sign_aware",
                            "latents": [
                                {
                                    "latent_index": 0,
                                    "positive_examples": [{"context": "only_pos"}],
                                    "negative_examples": [],
                                }
                            ],
                        }
                    },
                    file,
                    indent=2,
                )

            # when
            result = examples_module.build_and_persist_paired_duplicate_case_studies(
                config=config,
                resolved_config=resolved,
                selected_subruns=resolved["subruns"],
                output_dir=tmpdir,
            )

            # then
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertEqual(payload["case_studies"], [])
            self.assertTrue(bool(payload["notes"]))

    def test_paired_duplicate_case_studies_skips_invalid_payloads_before_valid_matches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline_a", "sign_aware": False},
                    {"variant_slug": "baseline_b", "sign_aware": False},
                    {"variant_slug": "signed_a", "sign_aware": True},
                    {"variant_slug": "signed_b", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline_a",
                    "variant_slug": "baseline_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline_a/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "baseline_b",
                    "variant_slug": "baseline_b",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 2e-3,
                    "relative_subrun_dir": "models/baseline_b/seed_0/lambda_2e-03",
                },
                {
                    "variant_name": "signed_a",
                    "variant_slug": "signed_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed_a/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed_b",
                    "variant_slug": "signed_b",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 2e-3,
                    "relative_subrun_dir": "models/signed_b/seed_0/lambda_2e-03",
                },
            ]

            baseline_model = SAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            sign_model = SAE(
                n_instances=1,
                d_in=2,
                d_hidden=1,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                device="cpu",
            )
            with torch.no_grad():
                baseline_model.W_dec.fill_(0.0)
                baseline_model.W_dec[0, 0, 0] = 1.0
                baseline_model.W_dec[0, 0, 1] = -1.0
                sign_model.W_dec.fill_(0.0)
                sign_model.W_dec[0, 0, 0] = 1.0

            payload_side_effects: List[Any] = [
                {"examples": {"duplicate_pairs": []}},
                {
                    "examples": {
                        "duplicate_pairs": [
                            {
                                "latent_a": 0,
                                "latent_b": 1,
                                "decoder_cosine": -0.99,
                                "latent_a_examples": [{"context": "a"}],
                                "latent_b_examples": [{"context": "b"}],
                            }
                        ]
                    }
                },
                "invalid_payload",
                {
                    "examples": {
                        "latents": [
                            {
                                "latent_index": 0,
                                "positive_examples": [{"context": "pos"}],
                                "negative_examples": [{"context": "neg"}],
                            }
                        ]
                    }
                },
            ]

            def _load_model_side_effect(*args: Any, **kwargs: Any) -> Any:
                del args
                subrun = kwargs["subrun"]
                return (
                    baseline_model
                    if str(subrun["variant_slug"]).startswith("baseline")
                    else sign_model
                )

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=payload_side_effects,
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
                side_effect=_load_model_side_effect,
            ), patch.object(
                examples_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "manifest.json"),
            ), patch.object(
                examples_module,
                "load_activation_manifest",
                return_value={},
            ), patch.object(
                examples_module,
                "resolve_runtime_device",
                return_value="cpu",
            ):
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(int(result["num_case_studies"]), 1)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertEqual(len(payload["case_studies"]), 1)

    def test_paired_duplicate_case_studies_early_exit_avoids_checkpoint_loading(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]

            payload_side_effects = [
                {
                    "examples": {
                        "duplicate_pairs": [{"latent_a": 0, "latent_b": 1}],
                    }
                },
                {
                    "examples": {
                        "latents": [
                            {
                                "latent_index": 0,
                                "positive_examples": [{"context": "only_pos"}],
                                "negative_examples": [],
                            }
                        ]
                    }
                },
            ]

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=payload_side_effects,
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
            ) as load_model_call:
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(load_model_call.call_count, 0)
            self.assertEqual(int(result["num_case_studies"]), 0)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertEqual(
                int(payload["schema_version"]),
                int(PAIRED_DUPLICATE_CASE_STUDIES_SCHEMA_VERSION),
            )
            self.assertTrue(bool(payload["notes"]))

    def test_paired_duplicate_case_studies_early_exit_preserves_known_subrun_references(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": [{"latent_a": 0, "latent_b": 1}]}},
                    None,
                ],
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
            ) as load_model_call:
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(load_model_call.call_count, 0)
            self.assertEqual(int(result["num_case_studies"]), 0)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertEqual(
                payload["baseline_subrun"]["variant_slug"],
                "baseline",
            )
            self.assertIsNone(payload["sign_aware_subrun"])
            self.assertIn(
                "No sign-aware examples artifact was available.",
                payload["notes"],
            )

    def test_paired_duplicate_case_studies_early_exit_preserves_sign_subrun_references_when_baseline_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")

            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": []}},
                    {"examples": {"latents": []}},
                ],
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
            ) as load_model_call:
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(load_model_call.call_count, 0)
            self.assertEqual(int(result["num_case_studies"]), 0)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertIsNone(payload["baseline_subrun"])
            self.assertEqual(payload["sign_aware_subrun"]["variant_slug"], "signed")
            self.assertIn(
                "No non-negative examples artifact with duplicate pairs was available.",
                payload["notes"],
            )

    def test_paired_duplicate_case_studies_uses_subset_matching_and_sanitizes_lists(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]
            baseline_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            sign_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=1,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                device="cpu",
            )
            with torch.no_grad():
                baseline_model.W_dec.fill_(0.0)
                baseline_model.W_dec[0, 0, 0] = 1.0
                baseline_model.W_dec[0, 0, 1] = -1.0
                sign_model.W_dec.fill_(0.0)
                sign_model.W_dec[0, 0, 0] = 1.0

            payload_side_effects = [
                {
                    "examples": {
                        "duplicate_pairs": [
                            {
                                "latent_a": 0,
                                "latent_b": 1,
                                "decoder_cosine": -0.99,
                                "latent_a_examples": "oops",
                                "latent_b_examples": None,
                            }
                        ]
                    }
                },
                {"examples": {"latents": []}},
            ]

            def _load_model_side_effect(*args: Any, **kwargs: Any) -> SAE:
                del args
                subrun = kwargs["subrun"]
                if str(subrun["variant_slug"]) == "baseline":
                    return baseline_model
                return sign_model

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=payload_side_effects,
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
                side_effect=_load_model_side_effect,
            ), patch.object(
                examples_module,
                "_eligible_signed_latent_entries",
                return_value={
                    0: {
                        "positive_examples": "bad",
                        "negative_examples": None,
                    }
                },
            ), patch.object(
                examples_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "manifest.json"),
            ), patch.object(
                examples_module,
                "load_activation_manifest",
                return_value={},
            ), patch.object(
                examples_module,
                "_decoder_columns",
                side_effect=AssertionError("_decoder_columns should not be called"),
            ):
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(int(result["num_case_studies"]), 1)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertEqual(
                int(payload["schema_version"]),
                int(PAIRED_DUPLICATE_CASE_STUDIES_SCHEMA_VERSION),
            )
            case_study = payload["case_studies"][0]
            self.assertEqual(case_study["baseline_pair"]["latent_a_examples"], [])
            self.assertEqual(case_study["baseline_pair"]["latent_b_examples"], [])
            self.assertEqual(
                case_study["matched_sign_aware_latent"]["positive_examples"], []
            )
            self.assertEqual(
                case_study["matched_sign_aware_latent"]["negative_examples"], []
            )

    def test_decoder_columns_and_direction_matching_cover_success_paths(self) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=3,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.fill_(0.0)
            model.W_dec[0, 0, 0] = 1.0
            model.W_dec[0, 0, 1] = -1.0
            model.W_dec[0, 1, 2] = 1.0

        # when
        columns = examples_module._decoder_columns(model=model)
        matched = examples_module._find_best_sign_latent_for_direction(
            direction=columns[0],
            sign_columns=columns,
            eligible_latent_indices=[0, 1, 2],
        )

        # then
        self.assertEqual(tuple(columns.shape), (3, 2))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(columns, dim=1),
                torch.ones((3,), dtype=torch.float32),
                atol=1e-6,
            )
        )
        self.assertIsNotNone(matched)
        assert matched is not None
        self.assertEqual(int(matched[0]), 0)
        self.assertAlmostEqual(float(matched[1]), 1.0, places=6)

    def test_paired_duplicate_case_studies_handles_empty_duplicate_pairs_after_filter(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": [{"latent_a": 0, "latent_b": 1}]}},
                    {"examples": {"latents": []}},
                ],
            ), patch.object(
                examples_module,
                "_eligible_signed_latent_entries",
                return_value={
                    0: {
                        "positive_examples": [{"context": "p"}],
                        "negative_examples": [{"context": "n"}],
                    }
                },
            ), patch.object(
                examples_module,
                "_list_or_empty",
                return_value=[],
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
            ) as load_model_call:
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(load_model_call.call_count, 0)
            self.assertEqual(int(result["num_case_studies"]), 0)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertIn(
                "No duplicate pairs were available to match against.",
                payload["notes"],
            )

    def test_paired_duplicate_case_studies_handles_subset_empty_after_index_filter(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]
            baseline_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            sign_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=1,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                device="cpu",
            )
            with torch.no_grad():
                baseline_model.W_dec.fill_(0.0)
                baseline_model.W_dec[0, 0, 0] = 1.0
                baseline_model.W_dec[0, 0, 1] = -1.0
                sign_model.W_dec.fill_(0.0)
                sign_model.W_dec[0, 0, 0] = 1.0

            def _load_model_side_effect(*args: Any, **kwargs: Any) -> SAE:
                del args
                return (
                    baseline_model
                    if kwargs["subrun"]["variant_slug"] == "baseline"
                    else sign_model
                )

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": [{"latent_a": 0, "latent_b": 1}]}},
                    {"examples": {"latents": []}},
                ],
            ), patch.object(
                examples_module,
                "_eligible_signed_latent_entries",
                return_value={
                    9: {
                        "positive_examples": [{"context": "p"}],
                        "negative_examples": [{"context": "n"}],
                    }
                },
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
                side_effect=_load_model_side_effect,
            ), patch.object(
                examples_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "manifest.json"),
            ), patch.object(
                examples_module,
                "load_activation_manifest",
                return_value={},
            ):
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(int(result["num_case_studies"]), 0)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertIn(
                "Eligible sign-aware latent indices were empty after filtering to model bounds.",
                payload["notes"],
            )

    def test_paired_duplicate_case_studies_raises_for_invalid_baseline_decoder(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]
            sign_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=1,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                device="cpu",
            )
            with torch.no_grad():
                sign_model.W_dec.fill_(0.0)
                sign_model.W_dec[0, 0, 0] = 1.0

            def _load_model_side_effect(*args: Any, **kwargs: Any) -> Any:
                del args
                if kwargs["subrun"]["variant_slug"] == "baseline":
                    return torch.nn.Linear(2, 2)
                return sign_model

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": [{"latent_a": 0, "latent_b": 1}]}},
                    {"examples": {"latents": []}},
                ],
            ), patch.object(
                examples_module,
                "_eligible_signed_latent_entries",
                return_value={
                    0: {
                        "positive_examples": [{"context": "p"}],
                        "negative_examples": [{"context": "n"}],
                    }
                },
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
                side_effect=_load_model_side_effect,
            ), patch.object(
                examples_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "manifest.json"),
            ), patch.object(
                examples_module,
                "load_activation_manifest",
                return_value={},
            ):
                with self.assertRaises(ValueError):
                    _ = examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )

    def test_paired_duplicate_case_studies_raises_for_decoder_dimension_mismatch(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]
            baseline_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            sign_model = SAE(
                n_instances=1,
                d_in=4,
                d_hidden=1,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                device="cpu",
            )
            with torch.no_grad():
                baseline_model.W_dec.fill_(0.0)
                baseline_model.W_dec[0, 0, 0] = 1.0
                baseline_model.W_dec[0, 1, 1] = 1.0
                sign_model.W_dec.fill_(0.0)
                sign_model.W_dec[0, 0, 0] = 1.0

            def _load_model_side_effect(*args: Any, **kwargs: Any) -> SAE:
                del args
                return (
                    baseline_model
                    if kwargs["subrun"]["variant_slug"] == "baseline"
                    else sign_model
                )

            # when / then
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": [{"latent_a": 0, "latent_b": 1}]}},
                    {"examples": {"latents": []}},
                ],
            ), patch.object(
                examples_module,
                "_eligible_signed_latent_entries",
                return_value={
                    0: {
                        "positive_examples": [{"context": "p"}],
                        "negative_examples": [{"context": "n"}],
                    }
                },
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
                side_effect=_load_model_side_effect,
            ), patch.object(
                examples_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "manifest.json"),
            ), patch.object(
                examples_module,
                "load_activation_manifest",
                return_value={},
            ):
                with self.assertRaises(ValueError) as context:
                    _ = examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
            message = str(context.exception)
            self.assertIn("Decoder dimension mismatch", message)
            self.assertIn("baseline d_in=", message)
            self.assertIn("sign-aware d_in=", message)
            self.assertIn("variant=baseline", message)
            self.assertIn("subrun=models/baseline/seed_0/lambda_1e-03", message)
            self.assertIn("variant=signed", message)
            self.assertIn("subrun=models/signed/seed_0/lambda_1e-03", message)

    def test_paired_duplicate_case_studies_raises_for_decoder_dimension_mismatch_even_when_duplicate_entries_invalid(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")

            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]
            baseline_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            sign_model = SAE(
                n_instances=1,
                d_in=4,
                d_hidden=1,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                device="cpu",
            )
            with torch.no_grad():
                baseline_model.W_dec.fill_(0.0)
                baseline_model.W_dec[0, 0, 0] = 1.0
                sign_model.W_dec.fill_(0.0)
                sign_model.W_dec[0, 0, 0] = 1.0

            def _load_model_side_effect(*args: Any, **kwargs: Any) -> SAE:
                del args
                return (
                    baseline_model
                    if kwargs["subrun"]["variant_slug"] == "baseline"
                    else sign_model
                )

            invalid_pairs = [
                {"latent_a": True, "latent_b": 1},
                {"latent_a": "bad", "latent_b": 1},
                {"latent_a": 0, "latent_b": 0},
            ]

            # when / then
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": invalid_pairs}},
                    {"examples": {"latents": []}},
                ],
            ), patch.object(
                examples_module,
                "_eligible_signed_latent_entries",
                return_value={
                    0: {
                        "positive_examples": [{"context": "p"}],
                        "negative_examples": [{"context": "n"}],
                    }
                },
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
                side_effect=_load_model_side_effect,
            ), patch.object(
                examples_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "manifest.json"),
            ), patch.object(
                examples_module,
                "load_activation_manifest",
                return_value={},
            ):
                with self.assertRaisesRegex(ValueError, "Decoder dimension mismatch"):
                    _ = examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )

    def test_paired_duplicate_case_studies_loop_branches_cover_invalid_entries_and_cache(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]
            baseline_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            sign_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=1,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                device="cpu",
            )
            with torch.no_grad():
                baseline_model.W_dec.fill_(0.0)
                baseline_model.W_dec[0, 0, 0] = 1.0
                baseline_model.W_dec[0, 0, 1] = -1.0
                sign_model.W_dec.fill_(0.0)
                sign_model.W_dec[0, 0, 0] = 1.0

            duplicate_pairs = [
                "bad",
                {"latent_a": True, "latent_b": 1},
                {"latent_a": "bad", "latent_b": 1},
                {"latent_a": 9, "latent_b": 1},
                {"latent_a": 0, "latent_b": 9},
                {"latent_a": 0, "latent_b": 0},
                {"latent_a": 0, "latent_b": 1},
                {"latent_a": 0, "latent_b": 1},
                {"latent_a": 0, "latent_b": 1},
            ]

            def _load_model_side_effect(*args: Any, **kwargs: Any) -> SAE:
                del args
                return (
                    baseline_model
                    if kwargs["subrun"]["variant_slug"] == "baseline"
                    else sign_model
                )

            original_decoder_column = examples_module._normalized_decoder_column
            call_counter = {"count": 0}

            def _normalized_decoder_side_effect(
                model: torch.nn.Module, latent_index: int
            ) -> torch.Tensor:
                call_counter["count"] += 1
                if int(call_counter["count"]) == 1:
                    raise ValueError("simulated direction failure")
                return original_decoder_column(model=model, latent_index=latent_index)

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": duplicate_pairs}},
                    {"examples": {"latents": []}},
                ],
            ), patch.object(
                examples_module,
                "_eligible_signed_latent_entries",
                return_value={
                    0: {
                        "positive_examples": [{"context": "p"}],
                        "negative_examples": [{"context": "n"}],
                    }
                },
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
                side_effect=_load_model_side_effect,
            ), patch.object(
                examples_module,
                "_normalized_decoder_column",
                side_effect=_normalized_decoder_side_effect,
            ), patch.object(
                examples_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "manifest.json"),
            ), patch.object(
                examples_module,
                "load_activation_manifest",
                return_value={},
            ):
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(int(result["num_case_studies"]), 2)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertEqual(len(payload["case_studies"]), 2)

    def test_paired_duplicate_case_studies_adds_note_when_no_valid_case_studies(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline",
                    "variant_slug": "baseline",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]
            baseline_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            sign_model = SAE(
                n_instances=1,
                d_in=3,
                d_hidden=1,
                resample_latents_threshold=1,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                device="cpu",
            )
            with torch.no_grad():
                baseline_model.W_dec.fill_(0.0)
                baseline_model.W_dec[0, 0, 0] = 1.0
                baseline_model.W_dec[0, 0, 1] = -1.0
                sign_model.W_dec.fill_(0.0)
                sign_model.W_dec[0, 0, 0] = 1.0

            def _load_model_side_effect(*args: Any, **kwargs: Any) -> SAE:
                del args
                return (
                    baseline_model
                    if kwargs["subrun"]["variant_slug"] == "baseline"
                    else sign_model
                )

            # when
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    {"examples": {"duplicate_pairs": ["bad"]}},
                    {"examples": {"latents": []}},
                ],
            ), patch.object(
                examples_module,
                "_eligible_signed_latent_entries",
                return_value={
                    0: {
                        "positive_examples": [{"context": "p"}],
                        "negative_examples": [{"context": "n"}],
                    }
                },
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
                side_effect=_load_model_side_effect,
            ), patch.object(
                examples_module,
                "resolve_manifest_path",
                return_value=os.path.join(tmpdir, "manifest.json"),
            ), patch.object(
                examples_module,
                "load_activation_manifest",
                return_value={},
            ):
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertEqual(int(result["num_case_studies"]), 0)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertIn(
                "No valid paired duplicate case studies could be constructed.",
                payload["notes"],
            )

    def test_paired_duplicate_case_studies_handles_missing_and_invalid_payloads(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config = LLMBenchmarkConfig.model_validate(
                _base_config(
                    output_dir=tmpdir,
                    data_path=data_path,
                    sign_aware=True,
                )
            )
            resolved = {
                "variants": [
                    {"variant_slug": "baseline_a", "sign_aware": False},
                    {"variant_slug": "baseline_b", "sign_aware": False},
                    {"variant_slug": "signed", "sign_aware": True},
                ]
            }
            selected_subruns = [
                {
                    "variant_name": "baseline_a",
                    "variant_slug": "baseline_a",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/baseline_a/seed_0/lambda_1e-03",
                },
                {
                    "variant_name": "baseline_b",
                    "variant_slug": "baseline_b",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 2e-3,
                    "relative_subrun_dir": "models/baseline_b/seed_0/lambda_2e-03",
                },
                {
                    "variant_name": "signed",
                    "variant_slug": "signed",
                    "seed": 0,
                    "sweep_param": "lambda",
                    "sweep_value": 1e-3,
                    "relative_subrun_dir": "models/signed/seed_0/lambda_1e-03",
                },
            ]

            # when
            missing_payload = examples_module._load_subrun_examples_payload(
                output_dir=tmpdir,
                subrun=selected_subruns[0],
            )
            with patch.object(
                examples_module,
                "_load_subrun_examples_payload",
                side_effect=[
                    "invalid",
                    {"examples": {"duplicate_pairs": [{"latent_a": 0, "latent_b": 1}]}},
                    None,
                ],
            ), patch.object(
                examples_module,
                "_load_checkpointed_subrun_model",
            ) as load_model_call:
                result = (
                    examples_module.build_and_persist_paired_duplicate_case_studies(
                        config=config,
                        resolved_config=resolved,
                        selected_subruns=selected_subruns,
                        output_dir=tmpdir,
                    )
                )

            # then
            self.assertIsNone(missing_payload)
            self.assertEqual(load_model_call.call_count, 0)
            self.assertEqual(int(result["num_case_studies"]), 0)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertIn(
                "No sign-aware examples artifact was available.", payload["notes"]
            )

    def test_load_alive_indices_and_select_latents_remaining_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = os.path.join(tmpdir, "evaluation.json")
            with open(eval_path, "w") as file:
                json.dump({"alive_indices": [0, 2, -1, "bad"]}, file)

            # when
            alive_indices = examples_module._load_alive_indices(
                subrun_dir=tmpdir,
                d_hidden=3,
            )

            # then
            self.assertEqual(alive_indices, [0, 2])

            # given
            with open(eval_path, "w") as file:
                json.dump([], file)

            # when
            missing_alive_indices = examples_module._load_alive_indices(
                subrun_dir=tmpdir,
                d_hidden=3,
            )

            # then
            self.assertIsNone(missing_alive_indices)

            # given
            config_payload = {
                "output_dir": "/tmp/out",
                "device": "cpu",
                "seed": 0,
                "llm": {"model_id": "sshleifer/tiny-gpt2"},
                "data": {
                    "backend": "local_jsonl",
                    "data_files": ["/tmp/dummy.jsonl"],
                    "text_field": "text",
                },
                "hook": {"hook_name": "mlp_out", "layer": 0},
                "variants": [
                    {
                        "name": "variant",
                        "sae_class": "SAE",
                        "sign_aware": True,
                        "d_hidden": 3,
                        "sparsity_coeff": 1e-3,
                    }
                ],
                "latent_examples": {
                    "enabled": True,
                    "latent_selection": "top_by_max_abs",
                    "max_latents": 3,
                    "max_examples_per_latent": 2,
                    "sample_size": 8,
                    "activation_threshold": 0.0,
                    "target_subruns": "explicit",
                },
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)
            latents = torch.zeros((0, 3), dtype=torch.float32)

            # when
            selected_none = examples_module._select_latents(
                config=config,
                latents=latents,
                alive_indices=[-1, 1, 99],
                d_hidden=3,
            )

            # then
            self.assertEqual(selected_none, [])

    def test_generate_latent_examples_error_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=True,
            )
            config = payload["config"]
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor(
                    [[0.1, -0.1], [0.2, -0.2]], dtype=torch.float32
                ),
                val_acts=torch.tensor([[2.0, -2.0]], dtype=torch.float32),
                val_tokens=torch.tensor([41], dtype=torch.int64),
            )
            subrun_dir = os.path.join(tmpdir, str(subrun["relative_subrun_dir"]))
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(output_dir=tmpdir, subrun=subrun)

            # when / then
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = examples_module.generate_latent_examples(
                        config=config,
                        resolved_config=resolved,
                        subrun=subrun,
                        output_dir=tmpdir,
                        overwrite=True,
                    )

            # given
            checkpoint_path = os.path.join(subrun_dir, "model.pt")
            torch.save({}, checkpoint_path)

            # when / then
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                with self.assertRaises(ValueError):
                    _ = examples_module.generate_latent_examples(
                        config=config,
                        resolved_config=resolved,
                        subrun=subrun,
                        output_dir=tmpdir,
                        overwrite=True,
                    )

            # given
            _ = _write_checkpoint_for_subrun(
                config=config,
                resolved=resolved,
                subrun=subrun,
                manifest=manifest,
                output_dir=tmpdir,
                sign_aware=True,
            )
            explicit_config = config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"latent_selection": "explicit", "explicit_latents": [0]}
                    )
                }
            )

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                result = examples_module.generate_latent_examples_for_subrun(
                    config=explicit_config,
                    resolved_config=resolved,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=True,
                )

            # then
            self.assertTrue(os.path.isfile(result["latent_examples_path"]))

    def test_generate_latent_examples_rejects_non_text_data_config(self) -> None:
        # given
        config = types.SimpleNamespace(modality="text", data=object())

        # when / then
        with self.assertRaises(ValueError):
            _ = examples_module.generate_latent_examples(
                config=config,
                resolved_config={},
                subrun={},
                output_dir="/tmp",
                overwrite=False,
            )

    def test_paired_duplicate_case_study_matches_highest_cosine_latent(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            config_payload = _base_config(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=True,
            )
            config_payload["variants"] = [
                {
                    "name": "baseline",
                    "sae_class": "SAE",
                    "sign_aware": False,
                    "d_hidden": 3,
                    "sparsity_coeff": 1e-3,
                },
                {
                    "name": "signed",
                    "sae_class": "SAE",
                    "sign_aware": True,
                    "d_hidden": 3,
                    "sparsity_coeff": 1e-3,
                },
                {
                    "name": "signed_single",
                    "sae_class": "SAE",
                    "sign_aware": True,
                    "single_activation_offset": True,
                    "d_hidden": 3,
                    "sparsity_coeff": 1e-3,
                },
            ]
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(
                config=config,
                stages_override=["examples"],
                validate_only=False,
            )
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor([[0.1, -0.1]], dtype=torch.float32),
                val_acts=torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32),
                val_tokens=torch.tensor([11, 12], dtype=torch.int64),
            )
            subruns = resolved["subruns"]
            for subrun in subruns:
                model, _, _, variant_config = build_model_for_subrun(
                    config=config,
                    resolved_config=resolved,
                    subrun=subrun,
                    manifest=manifest,
                    device="cpu",
                )
                with torch.no_grad():
                    model.W_dec.fill_(0.0)
                    model.b_enc.fill_(0.0)
                    model.b_dec.fill_(0.0)
                    if bool(variant_config.get("sign_aware", False)):
                        model.W_dec[0, 0, 0] = 0.6
                        model.W_dec[0, 1, 0] = 0.8
                        model.W_dec[0, 1, 1] = 1.0
                        model.W_dec[0, 0, 2] = 1.0
                        if hasattr(model, "act_offset"):
                            model.act_offset.fill_(0.0)
                        if hasattr(model, "act_offset_right"):
                            model.act_offset_right.fill_(0.0)
                        if hasattr(model, "act_offset_left"):
                            model.act_offset_left.fill_(0.0)
                    else:
                        model.W_dec[0, 0, 0] = 1.0
                        model.W_dec[0, 0, 1] = -1.0
                        model.W_dec[0, 1, 2] = 1.0
                subrun_dir = os.path.join(tmpdir, str(subrun["relative_subrun_dir"]))
                os.makedirs(subrun_dir, exist_ok=True)
                torch.save(
                    {"state_dict": model.state_dict()},
                    os.path.join(subrun_dir, "model.pt"),
                )

            baseline_subrun = [
                subrun
                for subrun in subruns
                if str(subrun["variant_name"]) == "baseline"
            ][0]
            signed_subrun = [
                subrun for subrun in subruns if str(subrun["variant_name"]) == "signed"
            ][0]
            baseline_subrun_dir = os.path.join(
                tmpdir, str(baseline_subrun["relative_subrun_dir"])
            )
            signed_subrun_dir = os.path.join(
                tmpdir, str(signed_subrun["relative_subrun_dir"])
            )
            with open(
                os.path.join(baseline_subrun_dir, "latent_examples.json"), "w"
            ) as file:
                json.dump(
                    {
                        "examples": {
                            "kind": "non_negative",
                            "duplicate_pairs": [
                                {
                                    "latent_a": 0,
                                    "latent_b": 1,
                                    "decoder_cosine": -0.99,
                                    "latent_a_examples": [{"context": "A"}],
                                    "latent_b_examples": [{"context": "B"}],
                                }
                            ],
                        }
                    },
                    file,
                    indent=2,
                )
            with open(
                os.path.join(signed_subrun_dir, "latent_examples.json"), "w"
            ) as file:
                json.dump(
                    {
                        "examples": {
                            "kind": "sign_aware",
                            "latents": [
                                {
                                    "latent_index": 0,
                                    "positive_examples": [{"context": "p0"}],
                                    "negative_examples": [{"context": "n0"}],
                                },
                                {
                                    "latent_index": 2,
                                    "positive_examples": [{"context": "p2"}],
                                    "negative_examples": [{"context": "n2"}],
                                },
                            ],
                        }
                    },
                    file,
                    indent=2,
                )

            # when
            result = examples_module.build_and_persist_paired_duplicate_case_studies(
                config=config,
                resolved_config=resolved,
                selected_subruns=subruns,
                output_dir=tmpdir,
            )

            # then
            self.assertTrue(os.path.isfile(str(result["case_studies_path"])))
            self.assertEqual(int(result["num_case_studies"]), 1)
            with open(str(result["case_studies_path"]), "r") as file:
                payload = json.load(file)
            self.assertEqual(
                int(
                    payload["case_studies"][0]["matched_sign_aware_latent"][
                        "latent_index"
                    ]
                ),
                2,
            )

    def test_run_examples_stage_smoke_with_store_tokens_true(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.jsonl")
            with open(data_path, "w") as file:
                file.write(json.dumps({"text": "alpha beta"}) + "\n")
            payload = _build_config_and_resolved(
                output_dir=tmpdir,
                data_path=data_path,
                sign_aware=True,
            )
            config = payload["config"]
            resolved = payload["resolved"]
            subrun = payload["subrun"]
            manifest = _write_manifest_and_shards(
                output_dir=tmpdir,
                train_acts=torch.tensor(
                    [[0.1, -0.1], [0.2, -0.2]], dtype=torch.float32
                ),
                val_acts=torch.tensor(
                    [[2.0, -2.0], [-1.0, 1.0], [3.0, -3.0], [-2.0, 2.0]],
                    dtype=torch.float32,
                ),
                val_tokens=torch.tensor([31, 32, 33, 34], dtype=torch.int64),
            )
            _ = _write_checkpoint_for_subrun(
                config=config,
                resolved=resolved,
                subrun=subrun,
                manifest=manifest,
                output_dir=tmpdir,
                sign_aware=True,
            )

            # when
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                result = _run_examples_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(int(result["selected_subruns"]), 1)
            self.assertEqual(int(result["generated_subruns"]), 1)
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "latent_examples_index.json"))
            )
            self.assertTrue(
                os.path.isfile(str(result["paired_duplicate_case_studies_path"]))
            )
