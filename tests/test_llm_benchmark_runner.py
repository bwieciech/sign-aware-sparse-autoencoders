import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional

from pydantic import ValidationError
from unittest.mock import patch
import torch

from sgnsae.experiments.llm_benchmark.runner import (
    _check_resume_no_prior_run,
    _delete_run_level_downstream_artifacts,
    _delete_plot_files_if_present,
    _effective_stages,
    _ensure_model_config_for_reused_subrun,
    _load_existing_fingerprint,
    _load_existing_rng_fingerprint,
    _load_existing_run_status,
    _load_training_metrics_if_present,
    _output_subdir_path,
    _run_eval_stage,
    _run_examples_stage,
    _run_plot_stage,
    _run_summarize_stage,
    _run_train_stage,
    _select_best_metric_entry,
    _training_completed_successfully,
    _version_existing_config,
    run_llm_sae_benchmark,
)
from sgnsae.experiments.llm_benchmark.config import (
    LLMBenchmarkConfig,
    resolve_benchmark_config,
)
from sgnsae.experiments.llm_benchmark.matched_points import matched_points_path
from sgnsae.experiments.llm_benchmark.plotting import (
    canonical_plot_paths,
    plot_stage_artifact_paths,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)
from sgnsae.experiments.llm_benchmark.training import (
    run_training_for_subrun as execute_training_for_subrun,
)
from tests.llm_benchmark_test_helpers import legacy_success_training_metrics_payload


def _minimal_valid_config(output_dir: str) -> Dict[str, Any]:
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


def _minimal_image_config(output_dir: str) -> Dict[str, Any]:
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "modality": "image",
        "seed": 0,
        "vision": {"model_id": "facebook/dinov2-giant"},
        "data": {
            "backend": "hf",
            "dataset_id": "ILSVRC/imagenet-1k",
            "split": "train",
            "image_field": "image",
        },
        "hook": {"hook_name": "hidden_state", "layer": 0},
        "cache": {
            "image_batch_size": 2,
            "image_size": 224,
            "max_images": 8,
            "vectors_per_shard": 64,
            "split_method": "stored_indices",
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "token_selection": "patch_only",
            "representation": "hidden_state",
        },
        "variants": [
            {
                "name": "baseline_relu",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": 1e-3,
            }
        ],
        "loss_recovered": {"enabled": False},
    }


def _write_activation_cache(output_dir: str) -> None:
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    train_activations = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    val_activations = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    torch.save(
        {"acts": train_activations}, os.path.join(activations_dir, "shard_00000.pt")
    )
    torch.save(
        {"acts": val_activations}, os.path.join(activations_dir, "shard_00001.pt")
    )

    manifest = {
        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
        "cache_fingerprint": "cache-fingerprint",
        "activation": {"d_in": 4},
        "hook": {"module_path_resolved": "transformer.h.0.mlp"},
        "dataset": {"dataset_id": None, "split": None},
        "tokenization": {"max_seq_len": 8},
        "split": {
            "method": "shard_ranges",
            "train_shard_indices": [0],
            "val_shard_indices": [1],
            "test_shard_indices": [],
        },
        "shards": [
            {
                "filename": "shard_00000.pt",
                "num_vectors": 12,
                "start_vector_index": 0,
                "end_vector_index": 12,
            },
            {
                "filename": "shard_00001.pt",
                "num_vectors": 8,
                "start_vector_index": 12,
                "end_vector_index": 20,
            },
        ],
    }
    manifest_path = os.path.join(activations_dir, "activations_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _copy_activation_cache(source_output_dir: str, target_output_dir: str) -> None:
    source_dir = os.path.join(source_output_dir, "activations")
    target_dir = os.path.join(target_output_dir, "activations")
    shutil.copytree(source_dir, target_dir)


def _create_run_level_downstream_artifacts(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    paths = [
        os.path.join(output_dir, "comparison_summary.json"),
        os.path.join(output_dir, "results_summary.json"),
        matched_points_path(output_dir=output_dir),
        os.path.join(output_dir, "latent_examples_index.json"),
        os.path.join(output_dir, "paired_duplicate_case_studies.json"),
        *plot_stage_artifact_paths(output_dir=output_dir).values(),
    ]
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.write("{}")


def _write_training_metrics(
    subrun_dir: str,
    status: Optional[str] = None,
    payload: Optional[Any] = None,
) -> None:
    os.makedirs(subrun_dir, exist_ok=True)
    with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
        if payload is None:
            payload = {"status": status}
        if isinstance(payload, str):
            file.write(payload)
        else:
            json.dump(payload, file, indent=2)


class TestLLMBenchmarkRunner(unittest.TestCase):
    def test_ensure_model_config_for_reused_subrun_returns_when_model_config_exists(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            resolved_config = resolve_benchmark_config(config)
            variant = dict(resolved_config["variants"][0])
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(output_dir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            model_config_path = os.path.join(subrun_dir, "model_config.json")
            with open(model_config_path, "w") as f:
                json.dump({"sentinel": "keep"}, f, indent=2)

            # when
            _ensure_model_config_for_reused_subrun(
                output_dir=output_dir,
                subrun_dir=subrun_dir,
                variant=variant,
                subrun=subrun,
                config=config,
            )

            # then
            with open(model_config_path, "r") as f:
                payload = json.load(f)
            self.assertEqual(payload, {"sentinel": "keep"})

    def test_ensure_model_config_for_reused_subrun_handles_manifest_and_metrics_failures(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            resolved_config = resolve_benchmark_config(config)
            variant = dict(resolved_config["variants"][0])
            subrun = dict(resolved_config["subruns"][0])
            subrun["sweep_index"] = 123
            subrun_dir = os.path.join(output_dir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as f:
                f.write("{")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.resolve_manifest_path",
                side_effect=ValueError("broken-manifest"),
            ):
                _ensure_model_config_for_reused_subrun(
                    output_dir=output_dir,
                    subrun_dir=subrun_dir,
                    variant=variant,
                    subrun=subrun,
                    config=config,
                )

            # then
            with open(os.path.join(subrun_dir, "model_config.json"), "r") as f:
                model_config = json.load(f)
            self.assertIsNone(model_config["manifest_cache_fingerprint"])
            self.assertEqual(model_config["resample_latents_threshold"], 0)
            self.assertNotIn("sweep_index", model_config["subrun"])
            self.assertEqual(
                model_config["loss_knobs"],
                {
                    "aux_loss_coeff": 1.0,
                    "dead_zone_reg_coeff": 0.0,
                    "dead_zone_reg_type": "l2",
                },
            )
            self.assertEqual(
                model_config["manifest_path"],
                os.path.join(output_dir, "activations", "activations_manifest.json"),
            )

    def test_ensure_model_config_for_reused_subrun_reads_manifest_and_threshold(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            _write_activation_cache(output_dir)
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            resolved_config = resolve_benchmark_config(config)
            variant = dict(resolved_config["variants"][0])
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(output_dir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as f:
                json.dump({"resample_latents_threshold": 17}, f, indent=2)

            # when
            _ensure_model_config_for_reused_subrun(
                output_dir=output_dir,
                subrun_dir=subrun_dir,
                variant=variant,
                subrun=subrun,
                config=config,
            )

            # then
            with open(os.path.join(subrun_dir, "model_config.json"), "r") as f:
                model_config = json.load(f)
            self.assertEqual(
                model_config["manifest_path"],
                os.path.join(output_dir, "activations", "activations_manifest.json"),
            )
            self.assertEqual(
                model_config["manifest_cache_fingerprint"], "cache-fingerprint"
            )
            self.assertEqual(model_config["resample_latents_threshold"], 17)
            self.assertEqual(
                model_config["loss_knobs"],
                {
                    "aux_loss_coeff": 1.0,
                    "dead_zone_reg_coeff": 0.0,
                    "dead_zone_reg_type": "l2",
                },
            )

    def test_ensure_model_config_for_reused_subrun_ignores_non_dict_training_metrics(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            _write_activation_cache(output_dir)
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            resolved_config = resolve_benchmark_config(config)
            variant = dict(resolved_config["variants"][0])
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(output_dir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
                json.dump(["not", "a", "dict"], file, indent=2)

            # when
            _ensure_model_config_for_reused_subrun(
                output_dir=output_dir,
                subrun_dir=subrun_dir,
                variant=variant,
                subrun=subrun,
                config=config,
            )

            # then
            with open(os.path.join(subrun_dir, "model_config.json"), "r") as file:
                model_config = json.load(file)
            self.assertEqual(model_config["resample_latents_threshold"], 0)

    def test_ensure_model_config_for_reused_subrun_persists_custom_loss_knobs(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_payload = _minimal_valid_config(output_dir=output_dir)
            config_payload["variants"][0]["sae_class"] = "GatedSAE"
            config_payload["variants"][0]["sign_aware"] = True
            config_payload["variants"][0]["aux_loss_coeff"] = 0.7
            config_payload["variants"][0]["dead_zone_reg_coeff"] = 0.15
            config_payload["variants"][0]["dead_zone_reg_type"] = "l2"
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            variant = dict(resolved_config["variants"][0])
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(output_dir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)

            # when
            _ensure_model_config_for_reused_subrun(
                output_dir=output_dir,
                subrun_dir=subrun_dir,
                variant=variant,
                subrun=subrun,
                config=config,
            )

            # then
            with open(os.path.join(subrun_dir, "model_config.json"), "r") as file:
                model_config = json.load(file)
            self.assertEqual(
                model_config["loss_knobs"],
                {
                    "aux_loss_coeff": 0.7,
                    "dead_zone_reg_coeff": 0.15,
                    "dead_zone_reg_type": "l2",
                },
            )

    def test_runner_helpers_cover_error_and_validation_branches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            os.makedirs(output_dir, exist_ok=True)
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )

            bad_config_path = os.path.join(output_dir, "config.json")
            with open(bad_config_path, "w") as f:
                f.write("{")

            bad_status_path = os.path.join(output_dir, "run_status.json")
            with open(bad_status_path, "w") as f:
                f.write("{")

            # when / then
            self.assertIsNone(_load_existing_fingerprint(output_dir))
            self.assertIsNone(_load_existing_rng_fingerprint(output_dir))
            self.assertIsNone(_load_existing_run_status(output_dir))

            with open(bad_config_path, "w") as f:
                json.dump({"config_fingerprint": ""}, f)
            self.assertIsNone(_load_existing_fingerprint(output_dir))
            self.assertIsNone(_load_existing_rng_fingerprint(output_dir))

            with open(bad_config_path, "w") as f:
                json.dump({"config_fingerprint": "x", "rng_fingerprint": "rng"}, f)
            self.assertEqual(_load_existing_rng_fingerprint(output_dir), "rng")

            with self.assertRaises(ValueError):
                _ = _effective_stages(
                    config=config,
                    stages_override=["train"],
                    validate_only=True,
                )
            with self.assertRaises(ValueError):
                _ = _effective_stages(
                    config=config,
                    stages_override=["unknown"],
                    validate_only=False,
                )

            with self.assertRaises(ValueError):
                _ = _output_subdir_path(output_dir=output_dir, relative_dir="/")
            with self.assertRaises(ValueError):
                _ = _output_subdir_path(output_dir=output_dir, relative_dir="../x")

            with patch(
                "sgnsae.experiments.llm_benchmark.runner.os.path.commonpath",
                side_effect=ValueError("bad commonpath"),
            ):
                with self.assertRaises(ValueError):
                    _ = _output_subdir_path(
                        output_dir=output_dir, relative_dir="models/subrun"
                    )

            with patch(
                "sgnsae.experiments.llm_benchmark.runner.os.path.commonpath",
                return_value=os.path.join(tmpdir, "other_root"),
            ):
                with self.assertRaises(ValueError):
                    _ = _output_subdir_path(
                        output_dir=output_dir, relative_dir="models/subrun"
                    )

    def test_load_training_metrics_if_present_returns_none_for_invalid_json(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "training_metrics.json"), "w") as file:
                file.write("{")

            # when
            result = _load_training_metrics_if_present(tmpdir)

            # then
            self.assertIsNone(result)

    def test_training_completed_successfully_returns_false_for_invalid_metrics(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.pt"), "w") as file:
                file.write("")
            with open(os.path.join(tmpdir, "training_metrics.json"), "w") as file:
                file.write("{")

            # when
            result = _training_completed_successfully(tmpdir)

            # then
            self.assertFalse(result)

    def test_training_completed_successfully_returns_false_when_metrics_file_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.pt"), "w") as file:
                file.write("")

            # when
            result = _training_completed_successfully(tmpdir)

            # then
            self.assertFalse(result)

    def test_training_completed_successfully_requires_completed_status(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.pt"), "w") as file:
                file.write("")
            with open(os.path.join(tmpdir, "training_metrics.json"), "w") as file:
                json.dump({"status": "failed"}, file)

            # when
            result = _training_completed_successfully(tmpdir)

            # then
            self.assertFalse(result)

    def test_training_completed_successfully_rejects_contradictory_completed_payloads(
        self,
    ) -> None:
        # given
        for payload in (
            {},
            {"status": "completed", "terminated_early": True},
            {"status": "completed", "failure": {"type": "exception"}},
        ):
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with open(os.path.join(tmpdir, "model.pt"), "w") as file:
                        file.write("")
                    with open(
                        os.path.join(tmpdir, "training_metrics.json"), "w"
                    ) as file:
                        json.dump(payload, file, indent=2)

                    # when
                    result = _training_completed_successfully(tmpdir)

                    # then
                    self.assertFalse(result)

    def test_effective_stages_accepts_examples_override(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=os.path.join(tmpdir, "run_root"))
            )

            # when
            stages = _effective_stages(
                config=config,
                stages_override=["cache", "examples", "plot"],
                validate_only=False,
            )

            # then
            self.assertEqual(stages, ["cache", "examples", "plot"])

    def test_runner_rejects_invalid_resolved_subrun_relative_dir(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            resolved_config = {
                "output_dir": output_dir,
                "config_fingerprint": "fingerprint",
                "subruns": [{"relative_subrun_dir": ""}],
            }

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.resolve_benchmark_config",
                return_value=resolved_config,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.persist_environment_metadata",
                return_value=os.path.join(output_dir, "environment.json"),
            ):
                with self.assertRaises(ValueError):
                    _ = run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["train"],
                    )

    def test_train_stage_failure_updates_run_status_and_raises(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                side_effect=RuntimeError("train exploded"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["train"],
                    )

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["status"], "failed")
            self.assertEqual(status["stages"]["train"]["message"], "train exploded")

    def test_validate_stage_must_be_alone_rejects_before_writes(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when / then
            with self.assertRaises(ValueError):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["validate", "cache"],
                )

            self.assertFalse(os.path.exists(output_dir))

    def test_runner_allows_image_modality_full_execution(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "image_config.json")
            config = _minimal_image_config(output_dir=output_dir)
            config["stages"] = ["cache", "train", "eval", "plot", "summarize"]
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_cache_stage",
                return_value={"message": "Image activation cache generated."},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": [], "aggregates": [], "metadata": {}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._run_plot_stage",
                return_value={"message": "Plot stage completed.", "plots_written": 8},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._run_summarize_stage",
                return_value={
                    "message": "Summarize stage completed.",
                    "results_summary_path": os.path.join(
                        output_dir, "results_summary.json"
                    ),
                },
            ):
                written_dir = run_llm_sae_benchmark(config_path=config_path)

            # then
            self.assertEqual(written_dir, output_dir)
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["cache"]["status"], "done")
            self.assertEqual(status["stages"]["train"]["status"], "done")
            self.assertEqual(status["stages"]["eval"]["status"], "done")
            self.assertEqual(status["stages"]["plot"]["status"], "done")
            self.assertEqual(status["stages"]["summarize"]["status"], "done")

    def test_runner_allows_image_modality_validate_only(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "image_config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_image_config(output_dir=output_dir), f)

            # when
            written_dir = run_llm_sae_benchmark(
                config_path=config_path, validate_only=True
            )

            # then
            self.assertEqual(written_dir, output_dir)
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["requested_stages"], ["validate"])
            self.assertEqual(status["stages"]["validate"]["status"], "done")

    def test_runner_allows_image_modality_cache_only(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "image_config.json")
            config = _minimal_image_config(output_dir=output_dir)
            config["stages"] = ["cache"]
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_cache_stage",
                return_value={"message": "Image activation cache generated."},
            ):
                written_dir = run_llm_sae_benchmark(config_path=config_path)

            # then
            self.assertEqual(written_dir, output_dir)
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["requested_stages"], ["cache"])
            self.assertEqual(status["stages"]["cache"]["status"], "done")

    def test_validate_only_writes_run_level_artifacts_and_sets_invariant_statuses(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when
            written_dir = run_llm_sae_benchmark(
                config_path=config_path,
                validate_only=True,
            )

            # then
            self.assertEqual(written_dir, output_dir)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "config.json")))
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "environment.json"))
            )
            status_path = os.path.join(output_dir, "run_status.json")
            self.assertTrue(os.path.isfile(status_path))
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["requested_stages"], ["validate"])
            self.assertEqual(status["stages"]["validate"]["status"], "done")
            for stage in ("cache", "train", "eval", "examples", "plot", "summarize"):
                self.assertEqual(status["stages"][stage]["status"], "not_requested")

    def test_non_validate_run_creates_subrun_dirs_and_completes_all_stages(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["seed"] = [0, 1]
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_cache_stage",
                return_value={"message": "Activation cache generated."},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": [], "aggregates": [], "metadata": {}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._run_plot_stage",
                return_value={"message": "Plot stage completed.", "plots_written": 8},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._run_summarize_stage",
                return_value={
                    "message": "Summarize stage completed.",
                    "results_summary_path": os.path.join(
                        output_dir, "results_summary.json"
                    ),
                },
            ):
                _ = run_llm_sae_benchmark(config_path=config_path)

            # then
            expected_dirs = [
                os.path.join(
                    output_dir, "models", "baseline_relu", "seed_0", "lambda_1e-03"
                ),
                os.path.join(
                    output_dir, "models", "baseline_relu", "seed_0", "lambda_1e-04"
                ),
                os.path.join(
                    output_dir, "models", "baseline_relu", "seed_1", "lambda_1e-03"
                ),
                os.path.join(
                    output_dir, "models", "baseline_relu", "seed_1", "lambda_1e-04"
                ),
            ]
            for directory in expected_dirs:
                self.assertTrue(os.path.isdir(directory))
                self.assertFalse(
                    os.path.exists(os.path.join(directory, "training_metrics.json"))
                )

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["cache"]["status"], "done")
            self.assertEqual(status["stages"]["train"]["status"], "done")
            self.assertEqual(status["stages"]["eval"]["status"], "done")
            self.assertEqual(status["stages"]["plot"]["status"], "done")
            self.assertEqual(status["stages"]["summarize"]["status"], "done")

    def test_cache_only_stage_completes_successfully(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_cache_stage",
                return_value={"message": "Reused existing activation cache."},
            ) as cache_stage:
                written_dir = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["cache"],
                )

            # then
            self.assertEqual(written_dir, output_dir)
            self.assertEqual(cache_stage.call_count, 1)

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["cache"]["status"], "done")
            self.assertEqual(status["stages"]["train"]["status"], "not_requested")

    def test_cache_stage_failure_sets_failed_and_blocks_following_stages(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_cache_stage",
                side_effect=RuntimeError("cache exploded"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["cache", "train"],
                    )

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["cache"]["status"], "failed")
            self.assertEqual(status["stages"]["cache"]["message"], "cache exploded")
            self.assertEqual(status["stages"]["train"]["status"], "pending")

    def test_plot_stage_failure_sets_failed_and_raises(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._run_plot_stage",
                side_effect=RuntimeError("plot exploded"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["plot", "summarize"],
                    )

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["plot"]["status"], "failed")
            self.assertEqual(status["stages"]["plot"]["message"], "plot exploded")
            self.assertEqual(status["stages"]["summarize"]["status"], "pending")

    def test_summarize_stage_failure_sets_failed_and_raises(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._run_summarize_stage",
                side_effect=RuntimeError("summarize exploded"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["summarize"],
                    )

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["summarize"]["status"], "failed")
            self.assertEqual(
                status["stages"]["summarize"]["message"], "summarize exploded"
            )

    def test_unknown_stage_marks_not_implemented_and_raises(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            resolved_config = {
                "schema_version": 1,
                "config_fingerprint": "fingerprint",
                "rng_fingerprint": "rng-fingerprint",
                "output_dir": output_dir,
                "device": "cpu",
                "stages": ["mystery"],
                "resume": True,
                "overwrite": False,
                "seeds": [0],
                "llm": {},
                "data": {},
                "hook": {},
                "cache": {},
                "training": {},
                "variants": [],
                "evaluation": {},
                "loss_recovered": {},
                "feature_ablation": {},
                "subruns": [],
            }

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._effective_stages",
                return_value=["mystery"],
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.resolve_benchmark_config",
                return_value=resolved_config,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.persist_environment_metadata",
                return_value=os.path.join(output_dir, "environment.json"),
            ):
                with self.assertRaises(NotImplementedError):
                    _ = run_llm_sae_benchmark(config_path=config_path)

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["mystery"]["status"], "not_implemented")

    def test_resume_mismatch_does_not_overwrite_config_or_environment_and_writes_run_status(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            modified = _minimal_valid_config(output_dir=output_dir)
            modified["llm"]["model_id"] = "different-model-id"
            with open(config_path, "w") as f:
                json.dump(modified, f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["resume_check"]["status"], "passed")
            self.assertFalse(status["resume_check"]["fingerprints_match"])
            self.assertIn("existing_fingerprint", status["resume_check"])
            self.assertIn("new_fingerprint", status["resume_check"])

    def test_resume_allows_config_mismatch_when_rng_fingerprint_matches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)
            with open(os.path.join(output_dir, "config.json"), "r") as f:
                existing_resolved_config = json.load(f)

            modified = _minimal_valid_config(output_dir=output_dir)
            modified["training"] = {"multi_sparsity_max_instances": 1}
            with open(config_path, "w") as f:
                json.dump(modified, f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            with open(os.path.join(output_dir, "run_status.json"), "r") as f:
                status = json.load(f)
            self.assertEqual(status["resume_check"]["status"], "passed")
            self.assertFalse(status["resume_check"]["fingerprints_match"])
            self.assertNotEqual(
                status["resume_check"]["existing_fingerprint"],
                status["resume_check"]["new_fingerprint"],
            )

            with open(os.path.join(output_dir, "config.json"), "r") as f:
                new_resolved_config = json.load(f)
            self.assertNotEqual(
                existing_resolved_config["config_fingerprint"],
                new_resolved_config["config_fingerprint"],
            )
            self.assertEqual(
                existing_resolved_config["rng_fingerprint"],
                new_resolved_config["rng_fingerprint"],
            )

    def test_validate_only_then_full_run_does_not_trigger_fingerprint_mismatch(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_cache_stage",
                return_value={"message": "Activation cache generated."},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": [], "aggregates": [], "metadata": {}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._run_plot_stage",
                return_value={"message": "Plot stage completed.", "plots_written": 8},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._run_summarize_stage",
                return_value={
                    "message": "Summarize stage completed.",
                    "results_summary_path": os.path.join(
                        output_dir, "results_summary.json"
                    ),
                },
            ):
                _ = run_llm_sae_benchmark(config_path=config_path)

            # then
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["resume_check"]["status"], "passed")

    def test_stage_override_does_not_trigger_fingerprint_mismatch(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_cache_stage",
                return_value={"message": "Activation cache generated."},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": [], "aggregates": [], "metadata": {}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._run_plot_stage",
                return_value={"message": "Plot stage completed.", "plots_written": 8},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner._run_summarize_stage",
                return_value={
                    "message": "Summarize stage completed.",
                    "results_summary_path": os.path.join(
                        output_dir, "results_summary.json"
                    ),
                },
            ):
                _ = run_llm_sae_benchmark(config_path=config_path)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._run_plot_stage",
                return_value={"message": "Plot stage completed.", "plots_written": 8},
            ):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["plot"],
                )

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["resume_check"]["status"], "passed")

    def test_overwrite_rewrites_run_level_files_without_deleting_models(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            sentinel_dir = os.path.join(output_dir, "models")
            os.makedirs(sentinel_dir, exist_ok=True)
            sentinel_path = os.path.join(sentinel_dir, "sentinel.txt")
            with open(sentinel_path, "w") as f:
                f.write("do_not_delete")

            modified = _minimal_valid_config(output_dir=output_dir)
            modified["overwrite"] = True
            modified["resume"] = False
            modified["llm"]["model_id"] = "different-model-id"
            with open(config_path, "w") as f:
                json.dump(modified, f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            self.assertTrue(os.path.isfile(sentinel_path))

    def test_dataset_backend_validation_hf_requires_dataset_id_and_split(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _minimal_valid_config(output_dir=os.path.join(tmpdir, "run_root"))
            config["data"] = {"backend": "hf"}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(config)

    def test_train_stage_runs_requested_subruns(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["seed"] = [0, 1]
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train, patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                return_value={},
            ) as scalar_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 2)
            self.assertEqual(scalar_train.call_count, 0)
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["status"], "done")
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 4)
            self.assertEqual(status["stages"]["train"]["reused_subruns"], 0)

    def test_train_stage_chunks_lambda_groups_when_max_instances_is_set(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["seed"] = [0, 1]
            config["training"] = {"multi_sparsity_max_instances": 1}
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 4)
            for call in lambda_chunk_train.call_args_list:
                self.assertEqual(len(call.kwargs["chunk_subruns"]), 1)

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 4)
            self.assertEqual(status["stages"]["train"]["reused_subruns"], 0)

    def test_train_stage_passes_stable_sweep_indices_to_lambda_chunk_trainer(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 1)
            chunk_subruns = lambda_chunk_train.call_args.kwargs["chunk_subruns"]
            self.assertEqual(
                [int(subrun["sweep_index"]) for subrun in chunk_subruns],
                [0, 1],
            )
            self.assertEqual(
                [float(subrun["sweep_value"]) for subrun in chunk_subruns],
                [1e-3, 1e-4],
            )

    def test_run_train_stage_is_chunking_invariant_without_manual_seed(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            full_output_dir = os.path.join(tmpdir, "full")
            chunked_output_dir = os.path.join(tmpdir, "chunked")
            os.makedirs(full_output_dir, exist_ok=True)
            os.makedirs(chunked_output_dir, exist_ok=True)
            _write_activation_cache(full_output_dir)
            _copy_activation_cache(full_output_dir, chunked_output_dir)

            full_payload = _minimal_valid_config(output_dir=full_output_dir)
            full_payload["training"] = {
                "train_steps": 4,
                "batch_size": 4,
                "log_every_n_steps": 2,
                "resample_dead_latents": False,
                "mixed_precision": False,
            }
            full_config = LLMBenchmarkConfig.model_validate(full_payload)
            full_resolved_config = resolve_benchmark_config(full_config)

            chunked_payload = _minimal_valid_config(output_dir=chunked_output_dir)
            chunked_payload["training"] = {
                "train_steps": 4,
                "batch_size": 4,
                "log_every_n_steps": 2,
                "resample_dead_latents": False,
                "mixed_precision": False,
                "multi_sparsity_max_instances": 1,
            }
            chunked_config = LLMBenchmarkConfig.model_validate(chunked_payload)
            chunked_resolved_config = resolve_benchmark_config(chunked_config)

            self.assertNotEqual(
                full_resolved_config["config_fingerprint"],
                chunked_resolved_config["config_fingerprint"],
            )
            self.assertEqual(
                full_resolved_config["rng_fingerprint"],
                chunked_resolved_config["rng_fingerprint"],
            )

            # when
            _ = _run_train_stage(
                config=full_config,
                resolved_config=full_resolved_config,
                output_dir=full_output_dir,
            )
            _ = _run_train_stage(
                config=chunked_config,
                resolved_config=chunked_resolved_config,
                output_dir=chunked_output_dir,
            )

            # then
            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                full_model_path = os.path.join(
                    full_output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    sweep_dir,
                    "model.pt",
                )
                chunked_model_path = os.path.join(
                    chunked_output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    sweep_dir,
                    "model.pt",
                )
                full_state_dict = torch.load(full_model_path, map_location="cpu")[
                    "state_dict"
                ]
                chunked_state_dict = torch.load(chunked_model_path, map_location="cpu")[
                    "state_dict"
                ]
                self.assertEqual(
                    set(full_state_dict.keys()), set(chunked_state_dict.keys())
                )
                for parameter_name in full_state_dict:
                    torch.testing.assert_close(
                        full_state_dict[parameter_name],
                        chunked_state_dict[parameter_name],
                        rtol=0.0,
                        atol=1e-8,
                    )

    def test_train_stage_uses_scalar_fallback_when_multi_sparsity_is_disabled(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["seed"] = [0, 1]
            config["training"] = {"multi_sparsity_enabled": False}
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train, patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                return_value={},
            ) as scalar_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 0)
            self.assertEqual(scalar_train.call_count, 4)

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 4)
            self.assertEqual(status["stages"]["train"]["reused_subruns"], 0)

    def test_train_stage_reuses_existing_artifacts_when_resume_enabled(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)
            _write_activation_cache(output_dir)

            subrun_dir = os.path.join(
                output_dir, "models", "baseline_relu", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                f.write("{}")
            _write_training_metrics(subrun_dir, status="completed")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 1)
            self.assertEqual(
                len(lambda_chunk_train.call_args.kwargs["chunk_subruns"]), 1
            )
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 1)
            self.assertEqual(status["stages"]["train"]["reused_subruns"], 1)
            model_config_path = os.path.join(subrun_dir, "model_config.json")
            self.assertTrue(os.path.isfile(model_config_path))
            with open(model_config_path, "r") as f:
                model_config_payload = json.load(f)
            self.assertNotIn("sweep_index", model_config_payload["subrun"])

    def test_run_train_stage_does_not_reuse_terminated_early_subrun(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            failed_subrun = dict(resolved_config["subruns"][0])
            failed_subrun_dir = os.path.join(
                tmpdir, failed_subrun["relative_subrun_dir"]
            )
            os.makedirs(failed_subrun_dir, exist_ok=True)
            with open(os.path.join(failed_subrun_dir, "model.pt"), "w") as file:
                file.write("")
            with open(
                os.path.join(failed_subrun_dir, "training_metrics.json"), "w"
            ) as file:
                json.dump({"terminated_early": True}, file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(result["reused_subruns"], 0)
            self.assertEqual(result["trained_subruns"], 2)
            self.assertEqual(lambda_chunk_train.call_count, 1)
            self.assertEqual(
                len(lambda_chunk_train.call_args.kwargs["chunk_subruns"]),
                2,
            )

    def test_run_train_stage_does_not_reuse_failed_status_subrun(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            failed_subrun = dict(resolved_config["subruns"][0])
            failed_subrun_dir = os.path.join(
                tmpdir, failed_subrun["relative_subrun_dir"]
            )
            os.makedirs(failed_subrun_dir, exist_ok=True)
            with open(os.path.join(failed_subrun_dir, "model.pt"), "w") as file:
                file.write("")
            with open(
                os.path.join(failed_subrun_dir, "training_metrics.json"), "w"
            ) as file:
                json.dump({"status": "failed", "terminated_early": True}, file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(result["reused_subruns"], 0)
            self.assertEqual(result["trained_subruns"], 2)
            self.assertEqual(lambda_chunk_train.call_count, 1)

    def test_run_train_stage_does_not_reuse_statusless_metrics_payload(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            failed_subrun = dict(resolved_config["subruns"][0])
            failed_subrun_dir = os.path.join(
                tmpdir, failed_subrun["relative_subrun_dir"]
            )
            os.makedirs(failed_subrun_dir, exist_ok=True)
            with open(os.path.join(failed_subrun_dir, "model.pt"), "w") as file:
                file.write("")
            _write_training_metrics(failed_subrun_dir, payload={})

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(result["reused_subruns"], 0)
            self.assertEqual(result["trained_subruns"], 2)
            self.assertEqual(lambda_chunk_train.call_count, 1)

    def test_run_train_stage_raises_for_legacy_success_metrics_without_status(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            legacy_subrun = dict(resolved_config["subruns"][0])
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            os.makedirs(legacy_subrun_dir, exist_ok=True)
            with open(os.path.join(legacy_subrun_dir, "model.pt"), "w") as file:
                file.write("")
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                side_effect=AssertionError("training should not run"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_train_stage(
                        config=config,
                        resolved_config=resolved_config,
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))

    def test_run_train_stage_overwrite_allows_regeneration_of_legacy_success_metrics(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(update={"overwrite": True})
            resolved_config = resolve_benchmark_config(config)
            legacy_subrun = dict(resolved_config["subruns"][0])
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            os.makedirs(legacy_subrun_dir, exist_ok=True)
            with open(os.path.join(legacy_subrun_dir, "model.pt"), "w") as file:
                file.write("")
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 1)
            self.assertEqual(result["trained_subruns"], 2)

    def test_run_train_stage_scalar_branch_raises_for_legacy_success_metrics_without_status(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["training"] = {"multi_sparsity_enabled": False}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            legacy_subrun = dict(resolved_config["subruns"][0])
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            os.makedirs(legacy_subrun_dir, exist_ok=True)
            with open(os.path.join(legacy_subrun_dir, "model.pt"), "w") as file:
                file.write("")
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                side_effect=AssertionError("training should not run"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_train_stage(
                        config=config,
                        resolved_config=resolved_config,
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))

    def test_run_train_stage_scalar_preflights_legacy_subruns_before_training_anything(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["training"] = {"multi_sparsity_enabled": False}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            legacy_subrun = dict(resolved_config["subruns"][1])
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            os.makedirs(legacy_subrun_dir, exist_ok=True)
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                side_effect=AssertionError("training should not run"),
            ) as scalar_train:
                with self.assertRaises(ValueError) as error:
                    _ = _run_train_stage(
                        config=config,
                        resolved_config=resolved_config,
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            self.assertEqual(scalar_train.call_count, 0)

    def test_run_train_stage_vectorized_preflights_legacy_subruns_before_training_any_group(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["seed"] = [0, 1]
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            legacy_subrun = next(
                subrun
                for subrun in resolved_config["subruns"]
                if int(subrun["seed"]) == 1
            )
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            os.makedirs(legacy_subrun_dir, exist_ok=True)
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                side_effect=AssertionError("vectorized training should not run"),
            ) as lambda_chunk_train:
                with self.assertRaises(ValueError) as error:
                    _ = _run_train_stage(
                        config=config,
                        resolved_config=resolved_config,
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            self.assertEqual(lambda_chunk_train.call_count, 0)

    def test_train_stage_reuses_all_completed_vectorized_subruns(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)
            _write_activation_cache(output_dir)

            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                subrun_dir = os.path.join(
                    output_dir, "models", "baseline_relu", "seed_0", sweep_dir
                )
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                    f.write("{}")
                _write_training_metrics(subrun_dir, status="completed")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 0)

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 0)
            self.assertEqual(status["stages"]["train"]["reused_subruns"], 2)
            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                model_config_path = os.path.join(
                    output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    sweep_dir,
                    "model_config.json",
                )
                self.assertTrue(os.path.isfile(model_config_path))

    def test_train_stage_scalar_fallback_reuses_complete_subrun_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["training"] = {"multi_sparsity_enabled": False}
            with open(config_path, "w") as f:
                json.dump(config, f)

            subrun_dir = os.path.join(
                output_dir, "models", "baseline_relu", "seed_0", "lambda_1e-03"
            )
            os.makedirs(subrun_dir, exist_ok=True)
            for filename in ("model.pt", "model_config.json"):
                with open(os.path.join(subrun_dir, filename), "w") as f:
                    f.write("{}")
            _write_training_metrics(subrun_dir, status="completed")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                return_value={},
            ) as scalar_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(scalar_train.call_count, 1)

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 1)
            self.assertEqual(status["stages"]["train"]["reused_subruns"], 1)

    def test_run_train_stage_scalar_reuse_regenerates_missing_model_config(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["training"] = {"multi_sparsity_enabled": False}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            reused_subrun = dict(resolved_config["subruns"][0])
            reused_subrun_dir = os.path.join(
                tmpdir, reused_subrun["relative_subrun_dir"]
            )
            os.makedirs(reused_subrun_dir, exist_ok=True)
            with open(os.path.join(reused_subrun_dir, "model.pt"), "w") as file:
                file.write("{}")
            with open(
                os.path.join(reused_subrun_dir, "training_metrics.json"), "w"
            ) as file:
                json.dump({"status": "completed"}, file)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                return_value={},
            ) as scalar_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(result["trained_subruns"], 1)
            self.assertEqual(result["reused_subruns"], 1)
            self.assertEqual(scalar_train.call_count, 1)
            self.assertTrue(
                os.path.isfile(os.path.join(reused_subrun_dir, "model_config.json"))
            )

    def test_run_eval_stage_reuses_scalar_subrun_after_train_repairs_model_config(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["training"] = {"multi_sparsity_enabled": False}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            reused_subrun = dict(resolved_config["subruns"][0])
            reused_subrun_dir = os.path.join(
                tmpdir, reused_subrun["relative_subrun_dir"]
            )
            os.makedirs(reused_subrun_dir, exist_ok=True)
            with open(os.path.join(reused_subrun_dir, "model.pt"), "w") as file:
                file.write("{}")
            with open(
                os.path.join(reused_subrun_dir, "training_metrics.json"), "w"
            ) as file:
                json.dump({"status": "completed"}, file)
            for filename in ("evaluation.json", "gamma_metrics.json"):
                with open(os.path.join(reused_subrun_dir, filename), "w") as file:
                    file.write("{}")

            def _fake_scalar_train(**kwargs: Any) -> Dict[str, Any]:
                current_subrun = kwargs["subrun"]
                current_subrun_dir = os.path.join(
                    tmpdir, str(current_subrun["relative_subrun_dir"])
                )
                os.makedirs(current_subrun_dir, exist_ok=True)
                with open(os.path.join(current_subrun_dir, "model.pt"), "w") as file:
                    file.write("{}")
                with open(
                    os.path.join(current_subrun_dir, "model_config.json"), "w"
                ) as file:
                    json.dump({"subrun": current_subrun}, file, indent=2)
                _write_training_metrics(current_subrun_dir, status="completed")
                return {}

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                side_effect=_fake_scalar_train,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ) as eval_stage, patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": []},
            ):
                _ = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )
                result = _run_eval_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertTrue(
                os.path.isfile(os.path.join(reused_subrun_dir, "model_config.json"))
            )
            self.assertEqual(result["reused_subruns"], 1)
            self.assertEqual(eval_stage.call_count, 1)

    def test_train_stage_overwrite_retrains_all_targeted_lambda_subruns(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["overwrite"] = True
            config["resume"] = False
            with open(config_path, "w") as f:
                json.dump(config, f)

            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                subrun_dir = os.path.join(
                    output_dir, "models", "baseline_relu", "seed_0", sweep_dir
                )
                os.makedirs(subrun_dir, exist_ok=True)
                for filename in ("model.pt", "training_metrics.json"):
                    with open(os.path.join(subrun_dir, filename), "w") as f:
                        f.write("{}")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 1)
            self.assertEqual(
                len(lambda_chunk_train.call_args.kwargs["chunk_subruns"]), 2
            )
            self.assertTrue(bool(lambda_chunk_train.call_args.kwargs["overwrite"]))

            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 2)
            self.assertEqual(status["stages"]["train"]["reused_subruns"], 0)

    def test_run_train_stage_invalidates_run_level_outputs_before_retraining(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            _create_run_level_downstream_artifacts(tmpdir)

            def _assert_run_level_outputs_deleted(**kwargs: Any) -> Dict[str, Any]:
                del kwargs
                self.assertFalse(
                    os.path.isfile(os.path.join(tmpdir, "comparison_summary.json"))
                )
                self.assertFalse(
                    os.path.isfile(os.path.join(tmpdir, "results_summary.json"))
                )
                self.assertFalse(os.path.isfile(matched_points_path(output_dir=tmpdir)))
                self.assertFalse(
                    os.path.isfile(os.path.join(tmpdir, "latent_examples_index.json"))
                )
                self.assertFalse(
                    os.path.isfile(
                        os.path.join(tmpdir, "paired_duplicate_case_studies.json")
                    )
                )
                for path in plot_stage_artifact_paths(output_dir=tmpdir).values():
                    self.assertFalse(os.path.isfile(path))
                return {}

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                side_effect=_assert_run_level_outputs_deleted,
            ):
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(result["trained_subruns"], 2)
            self.assertEqual(result["reused_subruns"], 0)

    def test_train_stage_vectorized_subruns_keep_training_artifact_contract(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            def _fake_chunk_trainer(**kwargs: Any) -> Dict[str, Any]:
                chunk_subruns = kwargs["chunk_subruns"]
                run_output_dir = kwargs["output_dir"]
                for subrun in chunk_subruns:
                    subrun_dir = os.path.join(
                        run_output_dir,
                        str(subrun["relative_subrun_dir"]).replace("/", os.sep),
                    )
                    os.makedirs(subrun_dir, exist_ok=True)
                    with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                        f.write("{}")
                    with open(os.path.join(subrun_dir, "model_config.json"), "w") as f:
                        json.dump({"subrun": subrun}, f, indent=2)
                    with open(
                        os.path.join(subrun_dir, "training_metrics.json"), "w"
                    ) as f:
                        json.dump(
                            {
                                "schema_version": 1,
                                "variant_name": subrun["variant_name"],
                                "variant_slug": subrun["variant_slug"],
                                "seed": int(subrun["seed"]),
                                "sweep_param": subrun["sweep_param"],
                                "sweep_value": subrun["sweep_value"],
                                "history": [],
                            },
                            f,
                            indent=2,
                        )
                return {}

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                side_effect=_fake_chunk_trainer,
            ):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            for sweep_dir, expected_sweep_value in (
                ("lambda_1e-03", 1e-3),
                ("lambda_1e-04", 1e-4),
            ):
                subrun_dir = os.path.join(
                    output_dir, "models", "baseline_relu", "seed_0", sweep_dir
                )
                with open(os.path.join(subrun_dir, "training_metrics.json"), "r") as f:
                    training_metrics = json.load(f)
                self.assertIn("schema_version", training_metrics)
                self.assertEqual(int(training_metrics["schema_version"]), 1)
                self.assertEqual(str(training_metrics["variant_name"]), "baseline_relu")
                self.assertEqual(str(training_metrics["variant_slug"]), "baseline_relu")
                self.assertEqual(int(training_metrics["seed"]), 0)
                self.assertEqual(str(training_metrics["sweep_param"]), "lambda")
                self.assertAlmostEqual(
                    float(training_metrics["sweep_value"]),
                    float(expected_sweep_value),
                )
                self.assertIn("history", training_metrics)
                self.assertTrue(
                    os.path.isfile(os.path.join(subrun_dir, "model_config.json"))
                )
                self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model.pt")))

    def test_train_stage_rejects_unknown_variant_slug_in_resolved_config(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            resolved_config = {
                "output_dir": output_dir,
                "config_fingerprint": "fingerprint",
                "variants": [],
                "subruns": [
                    {
                        "variant_name": "baseline_relu",
                        "variant_slug": "missing_variant",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_subrun_dir": "models/missing_variant/seed_0/lambda_1e-03",
                    }
                ],
            }

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.resolve_benchmark_config",
                return_value=resolved_config,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.persist_environment_metadata",
                return_value=os.path.join(output_dir, "environment.json"),
            ):
                with self.assertRaises(ValueError):
                    _ = run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["train"],
                    )

    def test_eval_stage_runs_and_builds_comparison_summary(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            resolved_config = resolve_benchmark_config(config)
            for subrun in resolved_config["subruns"]:
                _write_training_metrics(
                    os.path.join(output_dir, str(subrun["relative_subrun_dir"])),
                    status="completed",
                )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ) as eval_stage, patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": [1, 2]},
            ) as summary_builder:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["eval"],
                )

            # then
            self.assertEqual(eval_stage.call_count, 2)
            self.assertEqual(summary_builder.call_count, 1)
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["eval"]["status"], "done")
            self.assertEqual(status["stages"]["eval"]["evaluated_subruns"], 2)
            self.assertEqual(status["stages"]["eval"]["reused_subruns"], 0)
            self.assertEqual(status["stages"]["eval"]["comparison_subruns"], 2)

    def test_run_eval_stage_rejects_loss_recovered_for_image_modality(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "modality": "image",
                    "loss_recovered": base_config.loss_recovered.model_copy(
                        update={"enabled": True}
                    ),
                }
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = _run_eval_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=tmpdir,
                )

    def test_run_eval_stage_rejects_delta_lm_loss_ablation_for_image_modality(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "modality": "image",
                    "feature_ablation": base_config.feature_ablation.model_copy(
                        update={"enabled": True, "metric": "delta_lm_loss"}
                    ),
                }
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = _run_eval_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=tmpdir,
                )

    def test_run_eval_stage_initializes_run_level_loss_recovered_for_text(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "loss_recovered": base_config.loss_recovered.model_copy(
                        update={"enabled": True}
                    )
                }
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.ensure_run_level_loss_recovered_artifacts",
                return_value={},
            ) as ensure_artifacts, patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": []},
            ):
                result = _run_eval_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(ensure_artifacts.call_count, 1)
            self.assertEqual(result["evaluated_subruns"], 0)
            self.assertEqual(result["reused_subruns"], 0)
            self.assertEqual(result["comparison_subruns"], 0)

    def test_eval_stage_reuse_requires_checkpoint_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            reused_subrun_dir = os.path.join(
                output_dir, "models", "baseline_relu", "seed_0", "lambda_1e-03"
            )
            os.makedirs(reused_subrun_dir, exist_ok=True)
            _write_training_metrics(reused_subrun_dir, status="completed")
            for filename in ("evaluation.json", "gamma_metrics.json"):
                with open(os.path.join(reused_subrun_dir, filename), "w") as f:
                    f.write("{}")

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ) as eval_stage, patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": [1]},
            ):
                with self.assertRaises(ValueError):
                    _ = run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["eval"],
                    )

            self.assertEqual(eval_stage.call_count, 0)

    def test_eval_stage_reuses_artifacts_when_checkpoint_exists(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            reused_subrun_dir = os.path.join(
                output_dir, "models", "baseline_relu", "seed_0", "lambda_1e-03"
            )
            os.makedirs(reused_subrun_dir, exist_ok=True)
            _write_training_metrics(reused_subrun_dir, status="completed")
            for filename in (
                "model.pt",
                "model_config.json",
                "evaluation.json",
                "gamma_metrics.json",
            ):
                with open(os.path.join(reused_subrun_dir, filename), "w") as f:
                    f.write("{}")
            _write_training_metrics(
                os.path.join(
                    output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    "lambda_1e-04",
                ),
                status="completed",
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ) as eval_stage, patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": [1, 2]},
            ):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["eval"],
                )

            # then
            self.assertEqual(eval_stage.call_count, 1)
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["eval"]["evaluated_subruns"], 1)
            self.assertEqual(status["stages"]["eval"]["reused_subruns"], 1)

    def test_run_eval_stage_skips_incomplete_subrun_and_deletes_stale_eval_artifacts(
        self,
    ) -> None:
        # given
        for training_metrics_payload in (
            {"status": "failed", "terminated_early": True},
            {},
            {"status": "completed", "terminated_early": True},
            {"status": "completed", "failure": {"type": "exception"}},
            "[",
            None,
        ):
            with self.subTest(training_metrics_payload=training_metrics_payload):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config = LLMBenchmarkConfig.model_validate(
                        _minimal_valid_config(output_dir=tmpdir)
                    )
                    resolved_config = resolve_benchmark_config(config)
                    failed_subrun = dict(resolved_config["subruns"][0])
                    failed_subrun_dir = os.path.join(
                        tmpdir, failed_subrun["relative_subrun_dir"]
                    )
                    os.makedirs(failed_subrun_dir, exist_ok=True)
                    if training_metrics_payload is not None:
                        _write_training_metrics(
                            failed_subrun_dir,
                            payload=training_metrics_payload,
                        )
                    stale_paths = [
                        os.path.join(failed_subrun_dir, "evaluation.json"),
                        os.path.join(failed_subrun_dir, "gamma_metrics.json"),
                    ]
                    for path in stale_paths:
                        with open(path, "w") as file:
                            file.write("{}")

                    # when
                    with patch(
                        "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                        return_value={},
                    ) as eval_stage, patch(
                        "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                        return_value={"subruns": []},
                    ):
                        result = _run_eval_stage(
                            config=config,
                            resolved_config={"subruns": [failed_subrun]},
                            output_dir=tmpdir,
                        )

                    # then
                    self.assertEqual(eval_stage.call_count, 0)
                    self.assertEqual(result["evaluated_subruns"], 0)
                    self.assertEqual(result["reused_subruns"], 0)
                    self.assertEqual(result["skipped_failed_subruns"], 1)
                    for path in stale_paths:
                        self.assertFalse(os.path.exists(path))

    def test_run_eval_stage_raises_for_legacy_success_metrics_without_status_and_preserves_eval_files(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            legacy_subrun = dict(resolved_config["subruns"][0])
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            os.makedirs(legacy_subrun_dir, exist_ok=True)
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )
            stale_paths = [
                os.path.join(legacy_subrun_dir, "evaluation.json"),
                os.path.join(legacy_subrun_dir, "gamma_metrics.json"),
            ]
            for path in stale_paths:
                with open(path, "w") as file:
                    file.write("{}")

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                side_effect=AssertionError("evaluation should not run"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_eval_stage(
                        config=config,
                        resolved_config={"subruns": [legacy_subrun]},
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            for path in stale_paths:
                self.assertTrue(os.path.isfile(path))

    def test_run_eval_stage_preflights_legacy_subruns_before_cleanup(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            failed_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            legacy_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-4,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-04",
            }
            failed_subrun_dir = os.path.join(
                tmpdir, failed_subrun["relative_subrun_dir"]
            )
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            _write_training_metrics(failed_subrun_dir, status="failed")
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )
            stale_evaluation_path = os.path.join(failed_subrun_dir, "evaluation.json")
            with open(stale_evaluation_path, "w") as file:
                file.write("{}")

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                side_effect=AssertionError("evaluation should not run"),
            ) as run_eval, patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                side_effect=AssertionError("summary should not rebuild"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_eval_stage(
                        config=config,
                        resolved_config={"subruns": [failed_subrun, legacy_subrun]},
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            self.assertEqual(run_eval.call_count, 0)
            self.assertTrue(os.path.isfile(stale_evaluation_path))

    def test_run_eval_stage_preflights_legacy_subruns_before_loss_recovered_setup(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(
                update={
                    "loss_recovered": base_config.loss_recovered.model_copy(
                        update={"enabled": True}
                    )
                }
            )
            legacy_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.ensure_run_level_loss_recovered_artifacts",
                side_effect=AssertionError("loss-recovered setup should not run"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_eval_stage(
                        config=config,
                        resolved_config={"subruns": [legacy_subrun]},
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))

    def test_run_eval_stage_invalidates_run_level_outputs_before_recomputing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            _create_run_level_downstream_artifacts(tmpdir)
            for subrun in resolved_config["subruns"]:
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                os.makedirs(subrun_dir, exist_ok=True)
                with open(
                    os.path.join(subrun_dir, "training_metrics.json"), "w"
                ) as file:
                    json.dump({"status": "completed"}, file)

            def _assert_run_level_outputs_deleted(**kwargs: Any) -> Dict[str, Any]:
                del kwargs
                self.assertFalse(
                    os.path.isfile(os.path.join(tmpdir, "comparison_summary.json"))
                )
                self.assertFalse(
                    os.path.isfile(os.path.join(tmpdir, "results_summary.json"))
                )
                self.assertFalse(os.path.isfile(matched_points_path(output_dir=tmpdir)))
                self.assertFalse(
                    os.path.isfile(os.path.join(tmpdir, "latent_examples_index.json"))
                )
                self.assertFalse(
                    os.path.isfile(
                        os.path.join(tmpdir, "paired_duplicate_case_studies.json")
                    )
                )
                for path in plot_stage_artifact_paths(output_dir=tmpdir).values():
                    self.assertFalse(os.path.isfile(path))
                return {}

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                side_effect=_assert_run_level_outputs_deleted,
            ) as eval_stage, patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": []},
            ):
                result = _run_eval_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(eval_stage.call_count, 2)
            self.assertEqual(result["evaluated_subruns"], 2)
            self.assertEqual(result["reused_subruns"], 0)

    def test_eval_stage_invalidation_forces_plot_regeneration_on_resume(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(update={"resume": True})
            resolved_config = resolve_benchmark_config(config)
            _create_run_level_downstream_artifacts(tmpdir)
            for subrun in resolved_config["subruns"]:
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                os.makedirs(subrun_dir, exist_ok=True)
                with open(
                    os.path.join(subrun_dir, "training_metrics.json"), "w"
                ) as file:
                    json.dump({"status": "completed"}, file)

            summary_payload = {"aggregates": [], "subruns": [], "metadata": {}}

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value=summary_payload,
            ):
                _ = _run_eval_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value=summary_payload,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_plots_for_run",
                return_value={"plots_written": 8},
            ) as generate_plots:
                result = _run_plot_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertFalse(result["reused"])
            self.assertEqual(generate_plots.call_count, 1)

    def test_eval_stage_persists_skipped_failed_subruns_in_run_status(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as file:
                json.dump(_minimal_valid_config(output_dir=output_dir), file)
            failed_subrun_dir = os.path.join(
                output_dir, "models", "baseline_relu", "seed_0", "lambda_1e-03"
            )
            os.makedirs(failed_subrun_dir, exist_ok=True)
            with open(
                os.path.join(failed_subrun_dir, "training_metrics.json"), "w"
            ) as file:
                json.dump({"status": "failed", "terminated_early": True}, file)
            _write_training_metrics(
                os.path.join(
                    output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    "lambda_1e-04",
                ),
                status="completed",
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_evaluation_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"subruns": []},
            ):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["eval"],
                )

            # then
            with open(os.path.join(output_dir, "run_status.json"), "r") as file:
                status = json.load(file)
            self.assertEqual(
                int(status["stages"]["eval"]["skipped_failed_subruns"]),
                1,
            )

    def test_run_examples_stage_skips_when_latent_examples_are_disabled(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )

            # when
            result = _run_examples_stage(
                config=config,
                resolved_config={"subruns": []},
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(result["selected_subruns"], 0)
            self.assertEqual(result["generated_subruns"], 0)
            self.assertEqual(result["reused_subruns"], 0)
            self.assertIsNone(result["index_path"])

    def test_run_examples_stage_generates_examples_and_writes_index(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            enabled_config = config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"enabled": True}
                    )
                }
            )
            selected_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            _write_training_metrics(
                os.path.join(output_dir, selected_subrun["relative_subrun_dir"]),
                status="completed",
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.select_target_subruns_for_examples",
                return_value=[selected_subrun],
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_latent_examples",
                return_value={
                    "reused": False,
                    "latent_examples_path": os.path.join(
                        output_dir,
                        selected_subrun["relative_subrun_dir"],
                        "latent_examples.json",
                    ),
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                return_value=None,
            ):
                result = _run_examples_stage(
                    config=enabled_config,
                    resolved_config={"subruns": [selected_subrun], "seeds": [0]},
                    output_dir=output_dir,
                )

            # then
            self.assertEqual(result["selected_subruns"], 1)
            self.assertEqual(result["generated_subruns"], 1)
            self.assertEqual(result["reused_subruns"], 0)
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "latent_examples_index.json"))
            )
            with open(os.path.join(output_dir, "latent_examples_index.json"), "r") as f:
                index_payload = json.load(f)
            self.assertEqual(len(index_payload["subruns"]), 1)
            self.assertEqual(
                index_payload["subruns"][0]["latent_examples_path"],
                "models/baseline_relu/seed_0/lambda_1e-03/latent_examples.json",
            )

    def test_run_examples_stage_counts_reused_subruns(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            enabled_config = config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"enabled": True}
                    )
                }
            )
            selected_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            _write_training_metrics(
                os.path.join(output_dir, selected_subrun["relative_subrun_dir"]),
                status="completed",
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.select_target_subruns_for_examples",
                return_value=[selected_subrun],
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_latent_examples",
                return_value={
                    "reused": True,
                    "latent_examples_path": os.path.join(
                        output_dir,
                        selected_subrun["relative_subrun_dir"],
                        "latent_examples.json",
                    ),
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                return_value=None,
            ):
                result = _run_examples_stage(
                    config=enabled_config,
                    resolved_config={"subruns": [selected_subrun], "seeds": [0]},
                    output_dir=output_dir,
                )

            # then
            self.assertEqual(result["selected_subruns"], 1)
            self.assertEqual(result["generated_subruns"], 0)
            self.assertEqual(result["reused_subruns"], 1)

    def test_run_examples_stage_skips_incomplete_subruns_in_explicit_mode(
        self,
    ) -> None:
        # given
        for failed_status in ("failed", "running"):
            with self.subTest(failed_status=failed_status):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_dir = os.path.join(tmpdir, "run_root")
                    base_config = LLMBenchmarkConfig.model_validate(
                        _minimal_valid_config(output_dir=output_dir)
                    )
                    config = base_config.model_copy(
                        update={
                            "latent_examples": base_config.latent_examples.model_copy(
                                update={"enabled": True, "target_subruns": "explicit"}
                            )
                        }
                    )
                    completed_subrun = {
                        "variant_name": "baseline_relu",
                        "variant_slug": "baseline_relu",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
                    }
                    failed_subrun = {
                        "variant_name": "baseline_relu",
                        "variant_slug": "baseline_relu",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-4,
                        "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-04",
                    }
                    completed_dir = os.path.join(
                        output_dir, completed_subrun["relative_subrun_dir"]
                    )
                    failed_dir = os.path.join(
                        output_dir, failed_subrun["relative_subrun_dir"]
                    )
                    _write_training_metrics(completed_dir, status="completed")
                    _write_training_metrics(failed_dir, status=failed_status)
                    with open(os.path.join(failed_dir, "model.pt"), "w") as file:
                        file.write("stale checkpoint")
                    with open(
                        os.path.join(failed_dir, "latent_examples.json"), "w"
                    ) as file:
                        json.dump({"stale": True}, file)
                    assets_dir = os.path.join(failed_dir, "latent_examples_assets")
                    os.makedirs(assets_dir, exist_ok=True)
                    with open(os.path.join(assets_dir, "asset.txt"), "w") as file:
                        file.write("stale")

                    called_relative_dirs: List[str] = []

                    def _generate_examples(**kwargs: Any) -> Dict[str, Any]:
                        current_subrun = kwargs["subrun"]
                        called_relative_dirs.append(
                            str(current_subrun["relative_subrun_dir"])
                        )
                        return {
                            "reused": False,
                            "latent_examples_path": os.path.join(
                                output_dir,
                                str(current_subrun["relative_subrun_dir"]),
                                "latent_examples.json",
                            ),
                        }

                    # when
                    with patch(
                        "sgnsae.experiments.llm_benchmark.runner.generate_latent_examples",
                        side_effect=_generate_examples,
                    ) as generate_examples, patch(
                        "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                        return_value=None,
                    ), patch(
                        "sgnsae.experiments.llm_benchmark.runner.build_and_persist_paired_duplicate_case_studies",
                        return_value={"case_studies_path": None, "num_case_studies": 0},
                    ):
                        result = _run_examples_stage(
                            config=config,
                            resolved_config={
                                "subruns": [completed_subrun, failed_subrun],
                                "seeds": [0],
                            },
                            output_dir=output_dir,
                        )

                    # then
                    self.assertEqual(generate_examples.call_count, 1)
                    self.assertEqual(
                        called_relative_dirs,
                        [str(completed_subrun["relative_subrun_dir"])],
                    )
                    self.assertEqual(result["selected_subruns"], 2)
                    self.assertEqual(result["eligible_subruns"], 1)
                    self.assertEqual(result["skipped_failed_subruns"], 1)
                    self.assertFalse(
                        os.path.isfile(os.path.join(failed_dir, "latent_examples.json"))
                    )
                    self.assertFalse(os.path.isdir(assets_dir))

    def test_run_examples_stage_succeeds_when_all_selected_subruns_are_ineligible(
        self,
    ) -> None:
        # given
        for training_metrics_payload in (
            {"status": "failed"},
            "[",
        ):
            with self.subTest(training_metrics_payload=training_metrics_payload):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_dir = os.path.join(tmpdir, "run_root")
                    config_payload = _minimal_valid_config(output_dir=output_dir)
                    config_payload["cache"] = {"store_tokens": False}
                    base_config = LLMBenchmarkConfig.model_validate(config_payload)
                    config = base_config.model_copy(
                        update={
                            "latent_examples": base_config.latent_examples.model_copy(
                                update={"enabled": True, "target_subruns": "explicit"}
                            )
                        }
                    )
                    failed_subrun = {
                        "variant_name": "baseline_relu",
                        "variant_slug": "baseline_relu",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 1e-3,
                        "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
                    }
                    failed_dir = os.path.join(
                        output_dir, failed_subrun["relative_subrun_dir"]
                    )
                    _write_training_metrics(
                        failed_dir,
                        payload=training_metrics_payload,
                    )
                    with open(
                        os.path.join(failed_dir, "latent_examples.json"), "w"
                    ) as file:
                        json.dump({"stale": True}, file)
                    assets_dir = os.path.join(failed_dir, "latent_examples_assets")
                    os.makedirs(assets_dir, exist_ok=True)
                    with open(os.path.join(assets_dir, "asset.txt"), "w") as file:
                        file.write("stale")

                    # when
                    with patch(
                        "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                        side_effect=AssertionError("validation should be skipped"),
                    ), patch(
                        "sgnsae.experiments.llm_benchmark.runner.generate_latent_examples",
                        side_effect=AssertionError("generation should be skipped"),
                    ), patch(
                        "sgnsae.experiments.llm_benchmark.runner.build_and_persist_paired_duplicate_case_studies",
                        return_value={
                            "case_studies_path": os.path.join(
                                output_dir, "paired_duplicate_case_studies.json"
                            ),
                            "num_case_studies": 0,
                        },
                    ):
                        result = _run_examples_stage(
                            config=config,
                            resolved_config={"subruns": [failed_subrun], "seeds": [0]},
                            output_dir=output_dir,
                        )

                    # then
                    self.assertEqual(result["selected_subruns"], 1)
                    self.assertEqual(result["eligible_subruns"], 0)
                    self.assertEqual(result["skipped_failed_subruns"], 1)
                    self.assertTrue(
                        os.path.isfile(
                            os.path.join(output_dir, "latent_examples_index.json")
                        )
                    )
                    with open(
                        os.path.join(output_dir, "latent_examples_index.json"), "r"
                    ) as file:
                        index_payload = json.load(file)
                    self.assertEqual(index_payload["subruns"], [])
                    self.assertFalse(
                        os.path.isfile(os.path.join(failed_dir, "latent_examples.json"))
                    )
                    self.assertFalse(os.path.isdir(assets_dir))

    def test_run_examples_stage_raises_for_legacy_success_metrics_without_status_and_preserves_examples_files(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"enabled": True, "target_subruns": "explicit"}
                    )
                }
            )
            legacy_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            legacy_dir = os.path.join(output_dir, legacy_subrun["relative_subrun_dir"])
            _write_training_metrics(
                legacy_dir,
                payload=legacy_success_training_metrics_payload(),
            )
            latent_examples_path = os.path.join(legacy_dir, "latent_examples.json")
            with open(latent_examples_path, "w") as file:
                json.dump({"legacy": True}, file)
            assets_dir = os.path.join(legacy_dir, "latent_examples_assets")
            os.makedirs(assets_dir, exist_ok=True)
            with open(os.path.join(assets_dir, "asset.txt"), "w") as file:
                file.write("legacy")

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                side_effect=AssertionError("validation should not run"),
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_latent_examples",
                side_effect=AssertionError("generation should not run"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_examples_stage(
                        config=config,
                        resolved_config={"subruns": [legacy_subrun], "seeds": [0]},
                        output_dir=output_dir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            self.assertTrue(os.path.isfile(latent_examples_path))
            self.assertTrue(os.path.isdir(assets_dir))

    def test_run_examples_stage_preflights_legacy_subruns_before_failed_subrun_cleanup(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=output_dir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"enabled": True, "target_subruns": "explicit"}
                    )
                }
            )
            failed_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            legacy_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-4,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-04",
            }
            failed_dir = os.path.join(output_dir, failed_subrun["relative_subrun_dir"])
            legacy_dir = os.path.join(output_dir, legacy_subrun["relative_subrun_dir"])
            _write_training_metrics(failed_dir, status="failed")
            _write_training_metrics(
                legacy_dir,
                payload=legacy_success_training_metrics_payload(),
            )
            failed_examples_path = os.path.join(failed_dir, "latent_examples.json")
            with open(failed_examples_path, "w") as file:
                json.dump({"stale": True}, file)
            failed_assets_dir = os.path.join(failed_dir, "latent_examples_assets")
            os.makedirs(failed_assets_dir, exist_ok=True)
            with open(os.path.join(failed_assets_dir, "asset.txt"), "w") as file:
                file.write("stale")

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.select_target_subruns_for_examples",
                return_value=[failed_subrun, legacy_subrun],
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_latent_examples",
                side_effect=AssertionError("generation should not run"),
            ) as generate_examples, patch(
                "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                side_effect=AssertionError("validation should not run"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_examples_stage(
                        config=config,
                        resolved_config={
                            "subruns": [failed_subrun, legacy_subrun],
                            "seeds": [0],
                        },
                        output_dir=output_dir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            self.assertEqual(generate_examples.call_count, 0)
            self.assertTrue(os.path.isfile(failed_examples_path))
            self.assertTrue(os.path.isdir(failed_assets_dir))

    def test_run_examples_stage_skips_cache_validation_when_no_subruns_are_selected(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_payload = _minimal_valid_config(output_dir=output_dir)
            config_payload["cache"] = {"store_tokens": False}
            base_config = LLMBenchmarkConfig.model_validate(config_payload)
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"enabled": True}
                    )
                }
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.select_target_subruns_for_examples",
                return_value=[],
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                side_effect=AssertionError("validation should be skipped"),
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_paired_duplicate_case_studies",
                return_value={
                    "case_studies_path": os.path.join(
                        output_dir, "paired_duplicate_case_studies.json"
                    ),
                    "num_case_studies": 0,
                },
            ):
                result = _run_examples_stage(
                    config=config,
                    resolved_config={"subruns": [], "seeds": [0]},
                    output_dir=output_dir,
                )

            # then
            self.assertEqual(result["selected_subruns"], 0)
            self.assertEqual(result["eligible_subruns"], 0)
            self.assertEqual(result["skipped_failed_subruns"], 0)
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "latent_examples_index.json"))
            )

    def test_run_examples_stage_does_not_reuse_stale_examples_after_retraining(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["variants"][0]["sparsity_coeff"] = 1e-3
            config_payload["training"] = {
                "train_steps": 1,
                "batch_size": 4,
                "log_every_n_steps": 1,
                "resample_dead_latents": False,
                "mixed_precision": False,
            }
            config = LLMBenchmarkConfig.model_validate(config_payload)
            config = config.model_copy(
                update={
                    "latent_examples": config.latent_examples.model_copy(
                        update={"enabled": True}
                    )
                }
            )
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            latent_examples_path = os.path.join(subrun_dir, "latent_examples.json")
            with open(latent_examples_path, "w") as file:
                json.dump({"stale": True}, file)

            _ = execute_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            def _examples_result(**kwargs: Any) -> Dict[str, Any]:
                current_subrun = kwargs["subrun"]
                current_dir = os.path.join(
                    tmpdir, str(current_subrun["relative_subrun_dir"])
                )
                current_path = os.path.join(current_dir, "latent_examples.json")
                return {
                    "reused": os.path.isfile(current_path),
                    "latent_examples_path": current_path,
                }

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.select_target_subruns_for_examples",
                return_value=[subrun],
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_latent_examples",
                side_effect=_examples_result,
            ) as generate_examples, patch(
                "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                return_value=None,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_paired_duplicate_case_studies",
                return_value={"case_studies_path": None, "num_case_studies": 0},
            ):
                result = _run_examples_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(generate_examples.call_count, 1)
            self.assertEqual(result["generated_subruns"], 1)
            self.assertEqual(result["reused_subruns"], 0)

    def test_run_examples_stage_skips_paired_case_studies_for_image_modality(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_image_config(output_dir=output_dir)
            )
            config = base_config.model_copy(
                update={
                    "latent_examples": base_config.latent_examples.model_copy(
                        update={"enabled": True}
                    )
                }
            )
            selected_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            _write_training_metrics(
                os.path.join(output_dir, selected_subrun["relative_subrun_dir"]),
                status="completed",
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.validate_examples_cache_metadata",
                return_value=None,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.select_target_subruns_for_examples",
                return_value=[selected_subrun],
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_latent_examples",
                return_value={"reused": False},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_paired_duplicate_case_studies",
            ) as paired_case_studies:
                result = _run_examples_stage(
                    config=config,
                    resolved_config={"subruns": [selected_subrun], "seeds": [0]},
                    output_dir=output_dir,
                )

            # then
            self.assertEqual(result["paired_duplicate_case_studies_path"], None)
            self.assertEqual(int(result["paired_duplicate_case_studies_count"]), 0)
            self.assertEqual(paired_case_studies.call_count, 0)

    def test_examples_stage_updates_run_status_in_runner(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config_payload = _minimal_valid_config(output_dir=output_dir)
            config_payload["latent_examples"] = {"enabled": True}
            with open(config_path, "w") as f:
                json.dump(config_payload, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._run_examples_stage",
                return_value={
                    "message": "Examples stage completed.",
                    "generated_subruns": 1,
                    "reused_subruns": 0,
                    "selected_subruns": 1,
                    "eligible_subruns": 1,
                    "skipped_failed_subruns": 1,
                    "index_path": os.path.join(
                        output_dir, "latent_examples_index.json"
                    ),
                },
            ):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["examples"],
                )

            # then
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["examples"]["status"], "done")
            self.assertEqual(status["stages"]["examples"]["selected_subruns"], 1)
            self.assertEqual(status["stages"]["examples"]["generated_subruns"], 1)
            self.assertEqual(status["stages"]["examples"]["eligible_subruns"], 1)
            self.assertEqual(status["stages"]["examples"]["skipped_failed_subruns"], 1)

    def test_examples_stage_failure_updates_run_status_in_runner(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config_payload = _minimal_valid_config(output_dir=output_dir)
            config_payload["latent_examples"] = {"enabled": True}
            with open(config_path, "w") as f:
                json.dump(config_payload, f)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._run_examples_stage",
                side_effect=RuntimeError("examples exploded"),
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_llm_sae_benchmark(
                        config_path=config_path,
                        stages_override=["examples", "plot"],
                    )

            # then
            status_path = os.path.join(output_dir, "run_status.json")
            with open(status_path, "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["examples"]["status"], "failed")
            self.assertIn("examples exploded", status["stages"]["examples"]["message"])
            self.assertEqual(status["stages"]["plot"]["status"], "pending")

    def test_dataset_backend_validation_local_jsonl_requires_glob_or_files(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _minimal_valid_config(output_dir=os.path.join(tmpdir, "run_root"))
            config["data"] = {"backend": "local_jsonl"}

            # when / then
            with self.assertRaises(ValidationError):
                _ = LLMBenchmarkConfig.model_validate(config)

    def test_delete_plot_files_if_present_removes_existing_files(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = os.path.join(tmpdir, "first.png")
            second_path = os.path.join(tmpdir, "second.png")
            for path in (first_path, second_path):
                with open(path, "w") as file:
                    file.write("plot")

            # when
            _delete_plot_files_if_present({"first": first_path, "second": second_path})

            # then
            self.assertFalse(os.path.isfile(first_path))
            self.assertFalse(os.path.isfile(second_path))

    def test_delete_plot_files_if_present_ignores_missing_paths(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_path = os.path.join(tmpdir, "existing.png")
            missing_path = os.path.join(tmpdir, "missing.png")
            with open(existing_path, "w") as file:
                file.write("plot")

            # when
            _delete_plot_files_if_present(
                {"existing": existing_path, "missing": missing_path}
            )

            # then
            self.assertFalse(os.path.isfile(existing_path))
            self.assertFalse(os.path.isfile(missing_path))

    def test_delete_run_level_downstream_artifacts_removes_cached_outputs(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_run_level_downstream_artifacts(tmpdir)

            # when
            _delete_run_level_downstream_artifacts(tmpdir)

            # then
            self.assertFalse(
                os.path.isfile(os.path.join(tmpdir, "comparison_summary.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(tmpdir, "results_summary.json"))
            )
            self.assertFalse(os.path.isfile(matched_points_path(output_dir=tmpdir)))
            self.assertFalse(
                os.path.isfile(os.path.join(tmpdir, "latent_examples_index.json"))
            )
            self.assertFalse(
                os.path.isfile(
                    os.path.join(tmpdir, "paired_duplicate_case_studies.json")
                )
            )
            for path in plot_stage_artifact_paths(output_dir=tmpdir).values():
                self.assertFalse(os.path.isfile(path))

    def test_run_plot_stage_overwrite_deletes_existing_plots(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(update={"overwrite": True})

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._delete_plot_stage_artifacts_if_present"
            ) as delete_plots, patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"aggregates": [], "subruns": [], "metadata": {}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_plots_for_run",
                return_value={"plots_written": 8},
            ):
                result = _run_plot_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(delete_plots.call_count, 1)
            self.assertFalse(result["reused"])
            self.assertEqual(result["plots_written"], 8)

    def test_run_plot_stage_raises_for_legacy_success_metrics_when_summary_and_all_plot_artifacts_exist(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(update={"resume": True})
            legacy_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )
            with open(os.path.join(tmpdir, "comparison_summary.json"), "w") as file:
                json.dump({"subruns": [], "aggregates": [], "metadata": {}}, file)
            for path in plot_stage_artifact_paths(output_dir=tmpdir).values():
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as file:
                    file.write("{}")

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                side_effect=AssertionError("summary should not rebuild"),
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_plots_for_run",
                side_effect=AssertionError("plot generation should not run"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_plot_stage(
                        config=config,
                        resolved_config={"subruns": [legacy_subrun]},
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))

    def test_run_plot_stage_overwrite_preserves_existing_plot_artifacts_when_legacy_metrics_are_present(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(update={"overwrite": True})
            legacy_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            legacy_subrun_dir = os.path.join(
                tmpdir, legacy_subrun["relative_subrun_dir"]
            )
            _write_training_metrics(
                legacy_subrun_dir,
                payload=legacy_success_training_metrics_payload(),
            )
            artifact_paths = plot_stage_artifact_paths(output_dir=tmpdir)
            for path in artifact_paths.values():
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as file:
                    file.write("{}")

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._delete_plot_stage_artifacts_if_present",
                side_effect=AssertionError("plot deletion should not run"),
            ):
                with self.assertRaises(ValueError) as error:
                    _ = _run_plot_stage(
                        config=config,
                        resolved_config={"subruns": [legacy_subrun]},
                        output_dir=tmpdir,
                    )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))
            for path in artifact_paths.values():
                self.assertTrue(os.path.isfile(path))

    def test_run_plot_stage_reuses_plots_and_rebuilds_missing_summary(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner._are_artifacts_present",
                return_value=True,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"aggregates": [], "subruns": [], "metadata": {}},
            ) as summary_builder:
                result = _run_plot_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=tmpdir,
                )

            # then
            self.assertTrue(result["reused"])
            self.assertEqual(summary_builder.call_count, 1)

    def test_run_plot_stage_regenerates_when_plot_table_artifact_is_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(update={"resume": True})
            artifact_paths = plot_stage_artifact_paths(output_dir=tmpdir)
            for key, path in artifact_paths.items():
                if key == "matched_l0_points_table":
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as file:
                    file.write("{}")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"aggregates": [], "subruns": [], "metadata": {}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_plots_for_run",
                return_value={"plots_written": 8},
            ) as generate_plots:
                result = _run_plot_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=tmpdir,
                )

            # then
            self.assertFalse(result["reused"])
            self.assertEqual(generate_plots.call_count, 1)

    def test_run_plot_stage_regenerates_when_matched_points_artifact_is_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(update={"resume": True})
            artifact_paths = plot_stage_artifact_paths(output_dir=tmpdir)
            for key, path in artifact_paths.items():
                if key == "matched_points":
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as file:
                    file.write("{}")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value={"aggregates": [], "subruns": [], "metadata": {}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_plots_for_run",
                return_value={"plots_written": 8},
            ) as generate_plots:
                result = _run_plot_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=tmpdir,
                )

            # then
            self.assertFalse(result["reused"])
            self.assertEqual(generate_plots.call_count, 1)

    def test_run_plot_stage_overwrite_recomputes_matched_points_from_current_summary(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            base_config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            config = base_config.model_copy(update={"overwrite": True})
            failed_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            failed_subrun_dir = os.path.join(
                tmpdir, failed_subrun["relative_subrun_dir"]
            )
            _write_training_metrics(failed_subrun_dir, status="failed")
            with open(os.path.join(failed_subrun_dir, "evaluation.json"), "w") as file:
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
            with open(matched_points_path(output_dir=tmpdir), "w") as file:
                json.dump(
                    {
                        "anchors": {"l0": [16]},
                        "per_seed": [
                            {
                                "variant_slug": "baseline_relu",
                                "seed": 0,
                                "anchor_type": "l0",
                                "anchor_value": 16.0,
                                "method": "stale",
                                "distance_to_target": 0.0,
                                "metrics": {
                                    "mean_l0": 16.0,
                                    "mse": 0.1,
                                    "loss_recovered": 0.9,
                                    "dead_fraction": 0.01,
                                },
                            }
                        ],
                        "aggregate": [
                            {
                                "variant_slug": "baseline_relu",
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
            result = _run_plot_stage(
                config=config,
                resolved_config={
                    "llm": {"model_id": "dummy-model"},
                    "hook": {"hook_name": "mlp_out", "layer": 0},
                    "subruns": [failed_subrun],
                },
                output_dir=tmpdir,
            )

            # then
            self.assertFalse(result["reused"])
            with open(matched_points_path(output_dir=tmpdir), "r") as file:
                matched_points_payload = json.load(file)
            self.assertEqual(matched_points_payload["per_seed"], [])
            self.assertEqual(matched_points_payload["aggregate"], [])
            with open(
                plot_stage_artifact_paths(output_dir=tmpdir)["matched_l0_points_table"],
                "r",
            ) as file:
                self.assertEqual(len(file.read().strip().splitlines()), 1)

    def test_select_best_metric_entry_filters_and_selects_by_metric_direction(
        self,
    ) -> None:
        # given
        aggregates = [
            {
                "variant_slug": "variant_b",
                "sweep_param": "lambda",
                "sweep_value": 0.1,
                "mse_mean": 0.6,
                "loss_recovered_mean": 0.2,
                "num_seeds": 1,
            },
            {
                "variant_slug": "variant_a",
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "mse_mean": 0.4,
                "loss_recovered_mean": 0.5,
                "num_seeds": 2,
            },
            {
                "variant_slug": "variant_a",
                "sweep_param": "lambda",
                "sweep_value": 0.02,
                "mse_mean": 0.9,
                "loss_recovered_mean": 0.7,
                "num_seeds": 2,
            },
            {
                "variant_slug": "variant_c",
                "sweep_param": "lambda",
                "sweep_value": 0.03,
                "mse_mean": None,
                "loss_recovered_mean": None,
                "num_seeds": 1,
            },
            {"variant_slug": 7, "mse_mean": 1.0, "loss_recovered_mean": 1.0},
        ]

        # when
        best_mse = _select_best_metric_entry(
            aggregates=aggregates,
            metric_name="mse_mean",
            maximize=False,
        )
        best_loss_recovered = _select_best_metric_entry(
            aggregates=aggregates,
            metric_name="loss_recovered_mean",
            maximize=True,
        )

        # then
        self.assertEqual(
            [entry["variant_slug"] for entry in best_mse], ["variant_a", "variant_b"]
        )
        variant_a_mse = [
            entry for entry in best_mse if entry["variant_slug"] == "variant_a"
        ][0]
        self.assertEqual(variant_a_mse["sweep_value"], 0.01)
        variant_a_loss = [
            entry
            for entry in best_loss_recovered
            if entry["variant_slug"] == "variant_a"
        ][0]
        self.assertEqual(variant_a_loss["sweep_value"], 0.02)

    def test_run_summarize_stage_rebuilds_summary_from_current_subrun_files(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            comparison_summary_path = os.path.join(tmpdir, "comparison_summary.json")
            with open(comparison_summary_path, "w") as file:
                json.dump(
                    {
                        "metadata": {"stale": True},
                        "aggregates": [],
                        "subruns": [
                            {"relative_dir": "models/variant_a/seed_0/lambda_1e-03"},
                            {"relative_dir": "models/variant_a/seed_0/lambda_1e-04"},
                        ],
                    },
                    file,
                )

            rebuilt_summary = {
                "metadata": {},
                "aggregates": "bad",
                "subruns": [{"relative_dir": "models/variant_a/seed_0/lambda_1e-03"}],
            }

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.build_and_persist_comparison_summary",
                return_value=rebuilt_summary,
            ) as rebuild_summary, patch(
                "sgnsae.experiments.llm_benchmark.runner.ensure_summary_has_matched_l0_anchors",
                side_effect=lambda output_dir, summary: {
                    **summary,
                    "matched_l0_anchors": {},
                },
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.compute_and_persist_matched_points",
                return_value={"aggregate": [], "per_seed": [], "anchors": {}},
            ):
                result = _run_summarize_stage(
                    config=config,
                    resolved_config={"subruns": []},
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(rebuild_summary.call_count, 1)
            self.assertTrue(os.path.isfile(result["results_summary_path"]))
            with open(result["results_summary_path"], "r") as file:
                results_summary = json.load(file)
            self.assertEqual(int(results_summary["num_subruns"]), 1)
            self.assertEqual(results_summary["best_mse_by_variant"], [])

    def test_run_summarize_stage_excludes_failed_subrun_with_stale_evaluation(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            failed_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            failed_subrun_dir = os.path.join(
                tmpdir, failed_subrun["relative_subrun_dir"]
            )
            _write_training_metrics(failed_subrun_dir, status="failed")
            with open(os.path.join(failed_subrun_dir, "evaluation.json"), "w") as file:
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
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.ensure_summary_has_matched_l0_anchors",
                side_effect=lambda output_dir, summary: summary,
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner.compute_and_persist_matched_points",
                return_value={"aggregate": [], "per_seed": [], "anchors": {}},
            ):
                result = _run_summarize_stage(
                    config=config,
                    resolved_config={
                        "llm": {"model_id": "dummy-model"},
                        "hook": {"hook_name": "mlp_out", "layer": 0},
                        "subruns": [failed_subrun],
                    },
                    output_dir=tmpdir,
                )

            # then
            with open(result["results_summary_path"], "r") as file:
                summary = json.load(file)
            self.assertEqual(int(summary["num_subruns"]), 0)

    def test_run_plot_stage_rebuild_excludes_failed_subrun_with_stale_evaluation(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            failed_subrun = {
                "variant_name": "baseline_relu",
                "variant_slug": "baseline_relu",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 1e-3,
                "relative_subrun_dir": "models/baseline_relu/seed_0/lambda_1e-03",
            }
            failed_subrun_dir = os.path.join(
                tmpdir, failed_subrun["relative_subrun_dir"]
            )
            _write_training_metrics(failed_subrun_dir, status="failed")
            with open(os.path.join(failed_subrun_dir, "evaluation.json"), "w") as file:
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

            def _assert_summary_excludes_failed(**kwargs: Any) -> Dict[str, Any]:
                del kwargs
                with open(os.path.join(tmpdir, "comparison_summary.json"), "r") as file:
                    comparison_summary = json.load(file)
                self.assertEqual(comparison_summary["subruns"], [])
                return {"plots_written": len(canonical_plot_paths(output_dir=tmpdir))}

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.generate_plots_for_run",
                side_effect=_assert_summary_excludes_failed,
            ) as generate_plots:
                result = _run_plot_stage(
                    config=config,
                    resolved_config={
                        "llm": {"model_id": "dummy-model"},
                        "hook": {"hook_name": "mlp_out", "layer": 0},
                        "subruns": [failed_subrun],
                    },
                    output_dir=tmpdir,
                )

            # then
            self.assertFalse(result["reused"])
            self.assertEqual(generate_plots.call_count, 1)

    def test_train_stage_routes_k_sweep_through_multi_sparsity_path(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["variants"] = [
                {
                    "name": "topk_variant",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": [2, 4],
                }
            ]
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train, patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                return_value={},
            ) as scalar_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 1)
            chunk_subruns = lambda_chunk_train.call_args.kwargs["chunk_subruns"]
            self.assertEqual(len(chunk_subruns), 2)
            self.assertEqual(
                [int(subrun["sweep_value"]) for subrun in chunk_subruns],
                [2, 4],
            )
            self.assertTrue(
                all(subrun["sweep_param"] == "k" for subrun in chunk_subruns),
            )
            self.assertEqual(scalar_train.call_count, 0)

    def test_train_stage_routes_scalar_topk_k_through_single_subrun_path(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["variants"] = [
                {
                    "name": "topk_scalar",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": 4,
                }
            ]
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train, patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_subrun",
                return_value={},
            ) as scalar_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 0)
            self.assertEqual(scalar_train.call_count, 1)

    def test_train_stage_resumes_k_sweep_skipping_completed_subruns(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["variants"] = [
                {
                    "name": "topk_variant",
                    "sae_class": "TopKSAE",
                    "sign_aware": False,
                    "d_hidden": 8,
                    "topk_k": [2, 4],
                }
            ]
            with open(config_path, "w") as f:
                json.dump(config, f)
            _write_activation_cache(output_dir)

            completed_subrun_dir = os.path.join(
                output_dir, "models", "topk_variant", "seed_0", "k_2"
            )
            os.makedirs(completed_subrun_dir, exist_ok=True)
            with open(os.path.join(completed_subrun_dir, "model.pt"), "w") as f:
                f.write("{}")
            _write_training_metrics(completed_subrun_dir, status="completed")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_training_for_lambda_chunk",
                return_value={},
            ) as lambda_chunk_train:
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then
            self.assertEqual(lambda_chunk_train.call_count, 1)
            chunk_subruns = lambda_chunk_train.call_args.kwargs["chunk_subruns"]
            self.assertEqual(len(chunk_subruns), 1)
            self.assertEqual(int(chunk_subruns[0]["sweep_value"]), 4)
            self.assertEqual(chunk_subruns[0]["sweep_param"], "k")


def _write_completed_subrun(
    subrun_dir: str,
    train_steps: Optional[int] = None,
) -> None:
    os.makedirs(subrun_dir, exist_ok=True)
    with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
        f.write("{}")
    payload: Dict[str, Any] = {"status": "completed"}
    if train_steps is not None:
        payload["train_steps"] = train_steps
    with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as f:
        json.dump(payload, f)


class TestRunTrainStageContinuation(unittest.TestCase):
    def _single_subrun_config(
        self, tmpdir: str, train_steps: int = 10
    ) -> Dict[str, Any]:
        config_payload = _minimal_valid_config(output_dir=tmpdir)
        config_payload["variants"] = [
            {
                "name": "baseline_relu",
                "sae_class": "SAE",
                "sign_aware": False,
                "d_hidden": 8,
                "sparsity_coeff": 1e-3,
            }
        ]
        config_payload["training"] = {
            "multi_sparsity_enabled": False,
            "train_steps": train_steps,
        }
        return config_payload

    def test_continuation_detected_for_single_subrun(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = self._single_subrun_config(tmpdir, train_steps=10)
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            subrun = resolved["subruns"][0]
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _write_completed_subrun(subrun_dir, train_steps=5)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner" ".run_training_for_subrun",
                return_value={},
            ) as mock_train:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(mock_train.call_count, 1)
            self.assertEqual(
                mock_train.call_args.kwargs["continuation_from_steps"],
                5,
            )
            self.assertFalse(mock_train.call_args.kwargs["overwrite"])

    def test_error_on_train_steps_reduction_single_subrun(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = self._single_subrun_config(tmpdir, train_steps=5)
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            subrun = resolved["subruns"][0]
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _write_completed_subrun(subrun_dir, train_steps=10)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )
            self.assertIn("10", str(ctx.exception))
            self.assertIn("5", str(ctx.exception))

    def test_skip_when_train_steps_matches(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = self._single_subrun_config(tmpdir, train_steps=10)
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            subrun = resolved["subruns"][0]
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _write_completed_subrun(subrun_dir, train_steps=10)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner" ".run_training_for_subrun",
                return_value={},
            ) as mock_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )

            # then
            mock_train.assert_not_called()
            self.assertEqual(result["reused_subruns"], 1)

    def test_skip_when_completed_steps_is_none(self) -> None:
        # given — completed but no train_steps key
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = self._single_subrun_config(tmpdir, train_steps=10)
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            subrun = resolved["subruns"][0]
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _write_completed_subrun(subrun_dir, train_steps=None)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner" ".run_training_for_subrun",
                return_value={},
            ) as mock_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )

            # then
            mock_train.assert_not_called()
            self.assertEqual(result["reused_subruns"], 1)

    def test_continuation_detected_for_multi_sparsity_group(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["training"] = {"train_steps": 10}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            for subrun in resolved["subruns"]:
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                _write_completed_subrun(subrun_dir, train_steps=5)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_chunk:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )

            # then
            self.assertEqual(mock_chunk.call_count, 1)
            self.assertEqual(
                mock_chunk.call_args.kwargs["continuation_from_steps"],
                5,
            )

    def test_error_on_mixed_continuation_points_in_sweep(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["training"] = {"train_steps": 10}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            subruns = resolved["subruns"]
            sd0 = os.path.join(tmpdir, subruns[0]["relative_subrun_dir"])
            sd1 = os.path.join(tmpdir, subruns[1]["relative_subrun_dir"])
            _write_completed_subrun(sd0, train_steps=5)
            _write_completed_subrun(sd1, train_steps=8)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )
            self.assertIn("Mixed continuation", str(ctx.exception))

    def test_error_on_mixed_fresh_and_continuation_in_sweep(
        self,
    ) -> None:
        # given — one completed at 5, one not present
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["training"] = {"train_steps": 10}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            subruns = resolved["subruns"]
            sd0 = os.path.join(tmpdir, subruns[0]["relative_subrun_dir"])
            _write_completed_subrun(sd0, train_steps=5)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )
            self.assertIn("mix fresh", str(ctx.exception).lower())

    def test_error_on_train_steps_reduction_multi_sparsity(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _minimal_valid_config(output_dir=tmpdir)
            config_payload["training"] = {"train_steps": 5}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            for subrun in resolved["subruns"]:
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                _write_completed_subrun(subrun_dir, train_steps=10)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved,
                    output_dir=tmpdir,
                )
            self.assertIn("10", str(ctx.exception))
            self.assertIn("5", str(ctx.exception))


class TestStateMachineGuards(unittest.TestCase):
    def test_resume_true_overwrite_true_raises_value_error(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            config = _minimal_valid_config(output_dir=output_dir)
            config["resume"] = True
            config["overwrite"] = True
            with open(config_path, "w") as f:
                json.dump(config, f)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                run_llm_sae_benchmark(config_path=config_path)
            self.assertIn("mutually exclusive", str(ctx.exception))

    def test_resume_true_clean_subrun_dir_exists_raises_value_error(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                    pre_existing_subrun_dirs={subrun_dir},
                )
            self.assertIn("no prior training run", str(ctx.exception))

    def test_resume_true_corrupt_training_metrics_raises_value_error(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as f:
                f.write("{{{invalid json")

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                    pre_existing_subrun_dirs={subrun_dir},
                )
            self.assertIn("corrupt", str(ctx.exception))

    def test_resume_true_new_subrun_dir_does_not_exist_runs_fresh(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                    pre_existing_subrun_dirs=set(),
                )

            # then
            self.assertGreater(result["trained_subruns"], 0)
            self.assertTrue(mock_train.called)

    def test_resume_true_multi_sparsity_one_new_subrun_runs_fresh(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _minimal_valid_config(output_dir=tmpdir)
            raw["training"] = {"train_steps": 5}
            config = LLMBenchmarkConfig.model_validate(raw)
            resolved_config = resolve_benchmark_config(config)
            subruns = list(resolved_config["subruns"])
            self.assertEqual(len(subruns), 2)

            first_subrun_dir = os.path.join(tmpdir, subruns[0]["relative_subrun_dir"])
            os.makedirs(first_subrun_dir, exist_ok=True)
            with open(os.path.join(first_subrun_dir, "model.pt"), "w") as f:
                f.write("{}")
            _write_training_metrics(
                first_subrun_dir,
                payload={"status": "completed", "train_steps": 5},
            )

            second_subrun_dir = os.path.join(tmpdir, subruns[1]["relative_subrun_dir"])
            os.makedirs(second_subrun_dir, exist_ok=True)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                    pre_existing_subrun_dirs={first_subrun_dir},
                )

            # then
            self.assertEqual(result["reused_subruns"], 1)
            self.assertEqual(result["trained_subruns"], 1)
            self.assertTrue(mock_train.called)
            chunk_subruns = mock_train.call_args.kwargs["chunk_subruns"]
            self.assertEqual(len(chunk_subruns), 1)

    def test_resume_true_incomplete_prior_run_reruns_from_scratch(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _minimal_valid_config(output_dir=tmpdir)
            )
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _write_training_metrics(subrun_dir, status="failed")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                    pre_existing_subrun_dirs={subrun_dir},
                )

            # then
            self.assertGreater(result["trained_subruns"], 0)
            self.assertTrue(mock_train.called)

    def test_no_resume_no_overwrite_completed_run_raises_value_error(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _minimal_valid_config(output_dir=tmpdir)
            raw["resume"] = False
            raw["overwrite"] = False
            config = LLMBenchmarkConfig.model_validate(raw)
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                f.write("{}")
            _write_training_metrics(subrun_dir, status="completed")

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )
            self.assertIn("already exists", str(ctx.exception))

    def test_no_resume_no_overwrite_completed_scalar_subrun_raises_value_error(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _minimal_valid_config(output_dir=tmpdir)
            raw["resume"] = False
            raw["overwrite"] = False
            raw["variants"][0]["sparsity_coeff"] = 1e-3
            config = LLMBenchmarkConfig.model_validate(raw)
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                f.write("{}")
            _write_training_metrics(subrun_dir, status="completed")

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )
            self.assertIn("already exists", str(ctx.exception))

    def test_no_resume_no_overwrite_clean_dir_runs_fresh(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _minimal_valid_config(output_dir=tmpdir)
            raw["resume"] = False
            raw["overwrite"] = False
            config = LLMBenchmarkConfig.model_validate(raw)
            resolved_config = resolve_benchmark_config(config)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertGreater(result["trained_subruns"], 0)
            self.assertTrue(mock_train.called)

    def test_no_resume_no_overwrite_incomplete_prior_run_reruns(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _minimal_valid_config(output_dir=tmpdir)
            raw["resume"] = False
            raw["overwrite"] = False
            config = LLMBenchmarkConfig.model_validate(raw)
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _write_training_metrics(subrun_dir, status="failed")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_train:
                result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertGreater(result["trained_subruns"], 0)
            self.assertTrue(mock_train.called)

    def test_version_existing_config_produces_v1_on_first_resume(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"key": "original"}, f)

            # when
            result = _version_existing_config(tmpdir)

            # then
            self.assertEqual(
                result,
                os.path.join(tmpdir, "config-v1-deprecated.json"),
            )
            self.assertFalse(os.path.isfile(config_path))
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "config-v1-deprecated.json"))
            )
            with open(os.path.join(tmpdir, "config-v1-deprecated.json"), "r") as f:
                self.assertEqual(json.load(f), {"key": "original"})

    def test_version_existing_config_increments_version(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config-v1-deprecated.json"), "w") as f:
                json.dump({"key": "v1"}, f)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"key": "current"}, f)

            # when
            result = _version_existing_config(tmpdir)

            # then
            self.assertEqual(
                result,
                os.path.join(tmpdir, "config-v2-deprecated.json"),
            )
            self.assertFalse(os.path.isfile(config_path))
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "config-v2-deprecated.json"))
            )

    def test_version_existing_config_is_gap_tolerant(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config-v1-deprecated.json"), "w") as f:
                f.write("{}")
            with open(os.path.join(tmpdir, "config-v3-deprecated.json"), "w") as f:
                f.write("{}")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"key": "current"}, f)

            # when
            result = _version_existing_config(tmpdir)

            # then
            self.assertEqual(
                result,
                os.path.join(tmpdir, "config-v4-deprecated.json"),
            )
            self.assertFalse(os.path.isfile(config_path))

    def test_version_existing_config_returns_none_when_no_config(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            result = _version_existing_config(tmpdir)

            # then
            self.assertIsNone(result)

    def test_no_config_versioning_when_resume_false(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = False
            with open(config_path, "w") as f:
                json.dump(raw, f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "config.json")))
            deprecated_files = [
                f
                for f in os.listdir(output_dir)
                if f.startswith("config-v") and f.endswith("-deprecated.json")
            ]
            self.assertEqual(len(deprecated_files), 0)

    def test_config_fingerprint_mismatch_with_resume_does_not_raise(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            modified = _minimal_valid_config(output_dir=output_dir)
            modified["llm"]["model_id"] = "completely-different-model"
            with open(config_path, "w") as f:
                json.dump(modified, f)

            # when / then — no ValueError raised
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

    def test_fingerprints_still_computed_and_stored_in_config_json(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            with open(os.path.join(output_dir, "config.json"), "r") as f:
                resolved = json.load(f)
            self.assertIn("config_fingerprint", resolved)
            self.assertIn("rng_fingerprint", resolved)
            self.assertTrue(len(resolved["config_fingerprint"]) > 0)
            self.assertTrue(len(resolved["rng_fingerprint"]) > 0)

    def test_resume_check_records_fingerprints_with_status_passed(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(_minimal_valid_config(output_dir=output_dir), f)
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            modified = _minimal_valid_config(output_dir=output_dir)
            modified["llm"]["model_id"] = "another-model-id"
            with open(config_path, "w") as f:
                json.dump(modified, f)

            # when
            _ = run_llm_sae_benchmark(config_path=config_path, validate_only=True)

            # then
            with open(os.path.join(output_dir, "run_status.json"), "r") as f:
                status = json.load(f)
            rc = status["resume_check"]
            self.assertEqual(rc["status"], "passed")
            self.assertIn("existing_fingerprint", rc)
            self.assertIn("new_fingerprint", rc)
            self.assertIn("fingerprints_match", rc)
            self.assertFalse(rc["fingerprints_match"])
            self.assertNotIn("overwrite_applied", status)

    def test_state_machine_matrix_resume_false_overwrite_false_clean(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = False
            raw["overwrite"] = False
            with open(config_path, "w") as f:
                json.dump(raw, f)
            _write_activation_cache(output_dir)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ):
                result = run_llm_sae_benchmark(
                    config_path=config_path, stages_override=["train"]
                )

            # then
            with open(os.path.join(result, "run_status.json"), "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["status"], "done")

    def test_state_machine_matrix_resume_false_overwrite_false_existing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = False
            raw["overwrite"] = False
            with open(config_path, "w") as f:
                json.dump(raw, f)
            _write_activation_cache(output_dir)

            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                subrun_dir = os.path.join(
                    output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    sweep_dir,
                )
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                    f.write("{}")
                _write_training_metrics(subrun_dir, status="completed")

            # when / then
            with self.assertRaises(ValueError) as ctx:
                run_llm_sae_benchmark(
                    config_path=config_path, stages_override=["train"]
                )
            self.assertIn("already exists", str(ctx.exception))

    def test_state_machine_matrix_resume_true_overwrite_false_clean(
        self,
    ) -> None:
        # given — clean dir that pre-existed (no training_metrics)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = True
            raw["overwrite"] = False
            with open(config_path, "w") as f:
                json.dump(raw, f)
            _write_activation_cache(output_dir)

            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                subrun_dir = os.path.join(
                    output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    sweep_dir,
                )
                os.makedirs(subrun_dir, exist_ok=True)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                run_llm_sae_benchmark(
                    config_path=config_path, stages_override=["train"]
                )
            self.assertIn("no prior training run", str(ctx.exception))

    def test_state_machine_matrix_resume_true_overwrite_false_existing(
        self,
    ) -> None:
        # given — completed subruns at matching train_steps → skip
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = True
            raw["overwrite"] = False
            with open(config_path, "w") as f:
                json.dump(raw, f)
            _write_activation_cache(output_dir)

            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                subrun_dir = os.path.join(
                    output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    sweep_dir,
                )
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                    f.write("{}")
                _write_training_metrics(
                    subrun_dir,
                    payload={
                        "status": "completed",
                        "train_steps": 1000,
                    },
                )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_train:
                result = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then — both subruns reused, none trained
            with open(os.path.join(result, "run_status.json"), "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["reused_subruns"], 2)
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 0)

    def test_state_machine_matrix_resume_false_overwrite_true_clean(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = False
            raw["overwrite"] = True
            with open(config_path, "w") as f:
                json.dump(raw, f)
            _write_activation_cache(output_dir)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ):
                result = run_llm_sae_benchmark(
                    config_path=config_path, stages_override=["train"]
                )

            # then
            with open(os.path.join(result, "run_status.json"), "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["status"], "done")

    def test_state_machine_matrix_resume_false_overwrite_true_existing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = False
            raw["overwrite"] = True
            with open(config_path, "w") as f:
                json.dump(raw, f)
            _write_activation_cache(output_dir)

            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                subrun_dir = os.path.join(
                    output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    sweep_dir,
                )
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                    f.write("{}")
                _write_training_metrics(subrun_dir, status="completed")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_train:
                result = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then — overwrite retrains all
            self.assertTrue(mock_train.called)
            with open(os.path.join(result, "run_status.json"), "r") as f:
                status = json.load(f)
            self.assertEqual(status["stages"]["train"]["trained_subruns"], 2)

    def test_state_machine_matrix_resume_true_overwrite_true_raises(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = True
            raw["overwrite"] = True
            with open(config_path, "w") as f:
                json.dump(raw, f)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                run_llm_sae_benchmark(config_path=config_path)
            self.assertIn("mutually exclusive", str(ctx.exception))

    def test_overwrite_does_not_touch_subruns_outside_current_config(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = False
            raw["overwrite"] = True
            with open(config_path, "w") as f:
                json.dump(raw, f)
            _write_activation_cache(output_dir)

            # Pre-populate a subrun for seed_3 (not in config)
            orphan_dir = os.path.join(
                output_dir,
                "models",
                "baseline_relu",
                "seed_3",
                "lambda_1e-03",
            )
            os.makedirs(orphan_dir, exist_ok=True)
            orphan_model = os.path.join(orphan_dir, "model.pt")
            orphan_metrics = os.path.join(orphan_dir, "training_metrics.json")
            with open(orphan_model, "w") as f:
                f.write("{}")
            _write_training_metrics(orphan_dir, status="completed")

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ):
                _ = run_llm_sae_benchmark(
                    config_path=config_path, stages_override=["train"]
                )

            # then — orphan artifacts untouched
            self.assertTrue(os.path.isfile(orphan_model))
            self.assertTrue(os.path.isfile(orphan_metrics))

    def test_overwrite_eval_stage_does_not_delete_training_artifacts(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = False
            raw["overwrite"] = True
            raw["variants"][0]["sparsity_coeff"] = 1e-3
            with open(config_path, "w") as f:
                json.dump(raw, f)

            config = LLMBenchmarkConfig.model_validate(raw)
            resolved = resolve_benchmark_config(config)
            subrun = dict(resolved["subruns"][0])
            subrun_dir = os.path.join(output_dir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            model_path = os.path.join(subrun_dir, "model.pt")
            metrics_path = os.path.join(subrun_dir, "training_metrics.json")
            eval_path = os.path.join(subrun_dir, "evaluation.json")
            with open(model_path, "w") as f:
                f.write("{}")
            _write_training_metrics(subrun_dir, status="completed")
            with open(eval_path, "w") as f:
                json.dump({"old": True}, f)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner" ".run_evaluation_for_subrun",
                return_value={},
            ), patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".build_and_persist_comparison_summary",
                return_value={
                    "subruns": [],
                    "aggregates": [],
                    "metadata": {},
                },
            ):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["eval"],
                )

            # then — training artifacts still exist
            self.assertTrue(os.path.isfile(model_path))
            self.assertTrue(os.path.isfile(metrics_path))

    def test_resume_true_clean_scalar_subrun_raises_value_error(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _minimal_valid_config(output_dir=tmpdir)
            raw["variants"][0]["sparsity_coeff"] = 1e-3
            config = LLMBenchmarkConfig.model_validate(raw)
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                    pre_existing_subrun_dirs={subrun_dir},
                )
            self.assertIn("no prior training run", str(ctx.exception))

    def test_continuation_with_config_versioning_end_to_end(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["resume"] = True
            raw["overwrite"] = False
            raw["training"] = {"train_steps": 1000}
            with open(config_path, "w") as f:
                json.dump(raw, f)
            _write_activation_cache(output_dir)

            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump({"old": "config"}, f)

            for sweep_dir in ("lambda_1e-03", "lambda_1e-04"):
                subrun_dir = os.path.join(
                    output_dir,
                    "models",
                    "baseline_relu",
                    "seed_0",
                    sweep_dir,
                )
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "model.pt"), "w") as f:
                    f.write("{}")
                _write_training_metrics(
                    subrun_dir,
                    payload={
                        "status": "completed",
                        "train_steps": 500,
                    },
                )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.runner"
                ".run_training_for_lambda_chunk",
                return_value={},
            ) as mock_train:
                result = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )

            # then — config versioned
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "config-v1-deprecated.json"))
            )
            # then — continuation dispatched
            self.assertTrue(mock_train.called)
            call_kwargs = mock_train.call_args.kwargs
            self.assertEqual(call_kwargs["continuation_from_steps"], 500)

    def test_resume_true_completed_metrics_missing_model_warns(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = _minimal_valid_config(output_dir=tmpdir)
            raw["variants"][0]["sparsity_coeff"] = 1e-3
            config = LLMBenchmarkConfig.model_validate(raw)
            resolved_config = resolve_benchmark_config(config)
            subrun = dict(resolved_config["subruns"][0])
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _write_training_metrics(subrun_dir, status="completed")

            # when
            import warnings as _w

            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                with patch(
                    "sgnsae.experiments.llm_benchmark.runner"
                    ".run_training_for_subrun",
                    return_value={},
                ):
                    _run_train_stage(
                        config=config,
                        resolved_config=resolved_config,
                        output_dir=tmpdir,
                        pre_existing_subrun_dirs={subrun_dir},
                    )

            # then
            messages = [str(w.message) for w in caught]
            self.assertTrue(
                any("model.pt is missing" in m for m in messages),
                f"Expected model.pt warning, got: {messages}",
            )
            warning_match = [
                w for w in caught if "model.pt is missing" in str(w.message)
            ]
            self.assertIs(warning_match[0].category, UserWarning)

    def test_empty_variants_allowed_for_cache_only_stage(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["variants"] = []
            raw["resume"] = False
            with open(config_path, "w") as f:
                json.dump(raw, f)

            # when / then — no error
            with patch(
                "sgnsae.experiments.llm_benchmark.runner.run_cache_stage",
                return_value={"message": "done"},
            ):
                _ = run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["cache"],
                )

    def test_empty_variants_raises_for_train_stage(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "run_root")
            config_path = os.path.join(tmpdir, "config.json")
            raw = _minimal_valid_config(output_dir=output_dir)
            raw["variants"] = []
            raw["resume"] = False
            with open(config_path, "w") as f:
                json.dump(raw, f)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                run_llm_sae_benchmark(
                    config_path=config_path,
                    stages_override=["train"],
                )
            self.assertIn("variant", str(ctx.exception))
