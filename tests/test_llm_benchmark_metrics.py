import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import torch

from sgnsae.experiments.llm_benchmark.config import (
    LLMBenchmarkConfig,
    resolve_benchmark_config,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)
from sgnsae.experiments.llm_benchmark.metrics import (
    _delete_file_if_exists,
    _flatten_single_instance,
    _resolve_reconstruction_output,
    _safe_mean,
    _safe_standard_error,
    _summary_statistics,
    build_and_persist_comparison_summary,
    eval_artifact_paths,
    run_level_matched_points_path,
    run_evaluation_for_subrun,
)
from sgnsae.experiments.llm_benchmark.training import (
    build_model_for_subrun,
    load_activation_manifest,
    resolve_manifest_path,
)
from tests.llm_benchmark_test_helpers import legacy_success_training_metrics_payload


def _build_config(output_dir: str, sae_class: str) -> Dict[str, Any]:
    variant: Dict[str, Any] = {
        "name": "eval_variant",
        "sae_class": sae_class,
        "sign_aware": False,
        "d_hidden": 2,
        "sparsity_coeff": 1e-3,
    }
    if sae_class == "SignedSoftThresholdSAE":
        variant["theta_init"] = 0.0
    if sae_class == "JumpReLUSAE":
        variant["theta_init"] = 0.01
        variant["jump_init"] = 0.02
    return {
        "output_dir": output_dir,
        "device": "cpu",
        "seed": [0],
        "llm": {"model_id": "dummy-model"},
        "data": {
            "backend": "local_jsonl",
            "data_files": ["dummy.jsonl"],
            "text_field": "text",
        },
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "training": {
            "train_steps": 2,
            "batch_size": 2,
            "mixed_precision": False,
        },
        "evaluation": {
            "alive_threshold": 1e-6,
            "alive_min_fraction": 0.5,
            "min_gamma_samples": 2,
            "gamma_denominator_epsilon": 1e-12,
            "r2_denominator_epsilon": 1e-12,
        },
        "variants": [variant],
    }


def _build_resolved_comparison_subruns(output_dir: str) -> Dict[str, Any]:
    return {
        "llm": {"model_id": "dummy-model"},
        "hook": {"hook_name": "mlp_out", "layer": 0},
        "subruns": [
            {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "relative_subrun_dir": "models/variant_a/seed_0/lambda_0p001",
            },
            {
                "variant_name": "variant_a",
                "variant_slug": "variant_a",
                "seed": 1,
                "sweep_param": "lambda",
                "sweep_value": 0.001,
                "relative_subrun_dir": "models/variant_a/seed_1/lambda_0p001",
            },
            {
                "variant_name": "variant_b",
                "variant_slug": "variant_b",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.01,
                "relative_subrun_dir": "models/variant_b/seed_0/lambda_0p01",
            },
            {
                "variant_name": "variant_missing",
                "variant_slug": "variant_missing",
                "seed": 0,
                "sweep_param": "lambda",
                "sweep_value": 0.1,
                "relative_subrun_dir": "models/variant_missing/seed_0/lambda_0p1",
            },
        ],
    }


def _write_activation_cache(output_dir: str, val_acts: torch.Tensor) -> None:
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)
    train_acts = val_acts.clone()
    torch.save({"acts": train_acts}, os.path.join(activations_dir, "shard_00000.pt"))
    torch.save({"acts": val_acts}, os.path.join(activations_dir, "shard_00001.pt"))
    manifest = {
        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
        "cache_fingerprint": "cache-fingerprint",
        "activation": {"d_in": int(val_acts.shape[1])},
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
    manifest_path = os.path.join(activations_dir, "activations_manifest.json")
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)


def _write_completed_training_metrics(subrun_dir: str) -> None:
    with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
        json.dump({"status": "completed"}, file, indent=2)


def _write_empty_val_activation_cache(output_dir: str, d_in: int) -> None:
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)
    train_acts = torch.randn(4, d_in, dtype=torch.float32)
    torch.save({"acts": train_acts}, os.path.join(activations_dir, "shard_00000.pt"))
    manifest = {
        "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
        "cache_fingerprint": "cache-fingerprint",
        "activation": {"d_in": int(d_in)},
        "hook": {"module_path_resolved": "transformer.h.0.mlp"},
        "dataset": {"dataset_id": "fixture", "split": "train"},
        "tokenization": {"max_seq_len": 8},
        "split": {
            "method": "shard_ranges",
            "train_shard_indices": [0],
            "val_shard_indices": [],
            "test_shard_indices": [],
        },
        "shards": [
            {
                "filename": "shard_00000.pt",
                "num_vectors": int(train_acts.shape[0]),
                "start_vector_index": 0,
                "end_vector_index": int(train_acts.shape[0]),
            }
        ],
    }
    manifest_path = os.path.join(activations_dir, "activations_manifest.json")
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)


def _write_identity_checkpoint(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    subrun: Dict[str, Any],
    output_dir: str,
    theta_zero: bool,
) -> str:
    manifest_path = resolve_manifest_path(output_dir=output_dir)
    manifest = load_activation_manifest(manifest_path)
    model, _, _, _ = build_model_for_subrun(
        config=config,
        resolved_config=resolved_config,
        subrun=subrun,
        manifest=manifest,
        device="cpu",
    )
    with torch.no_grad():
        model.W_dec.data = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        model.b_enc.data.zero_()
        model.b_dec.data.zero_()
        if theta_zero and hasattr(model, "theta"):
            model.theta.data.zero_()
    subrun_dir = os.path.join(output_dir, subrun["relative_subrun_dir"])
    os.makedirs(subrun_dir, exist_ok=True)
    checkpoint_path = os.path.join(subrun_dir, "model.pt")
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


class TestLLMBenchmarkMetrics(unittest.TestCase):
    def test_run_level_matched_points_path_returns_run_level_artifact_path(
        self,
    ) -> None:
        # given
        output_dir = os.path.join("tmp", "run")

        # when
        path = run_level_matched_points_path(output_dir=output_dir)

        # then
        self.assertEqual(path, os.path.join("tmp", "run", "matched_points.json"))

    def test_eval_metrics_for_relu_sae_match_expected_values(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            val_acts = torch.tensor(
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=torch.float32
            )
            _write_activation_cache(tmpdir, val_acts=val_acts)
            config = LLMBenchmarkConfig.model_validate(_build_config(tmpdir, "SAE"))
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            _ = _write_identity_checkpoint(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                theta_zero=False,
            )

            # when
            _ = run_evaluation_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            with open(os.path.join(subrun_dir, "evaluation.json"), "r") as file:
                evaluation = json.load(file)
            with open(os.path.join(subrun_dir, "gamma_metrics.json"), "r") as file:
                gamma_metrics = json.load(file)

            self.assertAlmostEqual(float(evaluation["mse"]), 0.0, places=8)
            self.assertAlmostEqual(float(evaluation["r2"]), 1.0, places=8)
            self.assertAlmostEqual(float(evaluation["mean_l0"]), 1.0, places=8)
            self.assertAlmostEqual(float(evaluation["mean_l1"]), 2.5, places=8)
            self.assertEqual(int(evaluation["alive_count"]), 1)
            self.assertEqual(int(evaluation["dead_count"]), 1)
            self.assertEqual(evaluation["alive_indices"], [0])
            self.assertAlmostEqual(float(evaluation["alive_fraction"]), 0.5, places=8)
            self.assertAlmostEqual(float(evaluation["dead_fraction"]), 0.5, places=8)
            self.assertAlmostEqual(
                float(gamma_metrics["gamma_pos_values"][0]), 1.0, places=8
            )
            self.assertIsNone(gamma_metrics["gamma_pos_values"][1])

    def test_eval_metrics_for_signed_soft_threshold_include_sign_split_outputs(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            val_acts = torch.tensor(
                [[1.0, -1.0], [2.0, -2.0], [-1.0, 1.0], [-2.0, 2.0]],
                dtype=torch.float32,
            )
            _write_activation_cache(tmpdir, val_acts=val_acts)
            config = LLMBenchmarkConfig.model_validate(
                _build_config(tmpdir, "SignedSoftThresholdSAE")
            )
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            _ = _write_identity_checkpoint(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                theta_zero=True,
            )

            # when
            _ = run_evaluation_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            with open(os.path.join(subrun_dir, "evaluation.json"), "r") as file:
                evaluation = json.load(file)
            with open(os.path.join(subrun_dir, "gamma_metrics.json"), "r") as file:
                gamma_metrics = json.load(file)

            self.assertEqual(int(evaluation["alive_count"]), 2)
            self.assertEqual(int(evaluation["dead_count"]), 0)
            self.assertEqual(int(evaluation["rare_count"]), 0)
            self.assertEqual(evaluation["alive_indices"], [0, 1])
            self.assertIn("polarity_imbalance_summary", evaluation)
            self.assertIn("polarity_imbalance_by_latent", evaluation)
            self.assertAlmostEqual(
                float(evaluation["polarity_imbalance_summary"]["mean"]), 0.0, places=8
            )
            self.assertIn("gamma_neg_values", gamma_metrics)
            self.assertAlmostEqual(
                float(gamma_metrics["gamma_pos_values"][0]), 1.0, places=8
            )
            self.assertAlmostEqual(
                float(gamma_metrics["gamma_neg_values"][0]), 1.0, places=8
            )
            self.assertAlmostEqual(
                float(gamma_metrics["summary"]["coverage"]["both_fraction_valid"]),
                1.0,
                places=8,
            )

    def test_metric_helpers_cover_edge_cases(self) -> None:
        # given
        three_dimensional = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32
        )
        two_dimensional = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        config_payload = _build_config(output_dir="/tmp", sae_class="SAE")
        config_payload["loss_recovered"] = {"enabled": True}
        config_payload["feature_ablation"] = {"enabled": True}
        config_with_optional_eval_artifacts = LLMBenchmarkConfig.model_validate(
            config_payload
        )

        # when
        artifacts = eval_artifact_paths("/tmp/subrun")
        artifacts_with_optional = eval_artifact_paths(
            "/tmp/subrun", config=config_with_optional_eval_artifacts
        )
        flattened_single = _flatten_single_instance(three_dimensional)
        flattened_general = _flatten_single_instance(two_dimensional)
        empty_summary = _summary_statistics([])
        resolved_output = _resolve_reconstruction_output({"y_pred": three_dimensional})
        mean_empty = _safe_mean([])
        standard_error_empty = _safe_standard_error([])
        standard_error_single = _safe_standard_error([5.0])

        # then
        self.assertEqual(
            artifacts,
            {
                "evaluation": "/tmp/subrun/evaluation.json",
                "gamma_metrics": "/tmp/subrun/gamma_metrics.json",
            },
        )
        self.assertEqual(
            artifacts_with_optional,
            {
                "evaluation": "/tmp/subrun/evaluation.json",
                "gamma_metrics": "/tmp/subrun/gamma_metrics.json",
                "loss_recovered": "/tmp/subrun/loss_recovered.json",
                "feature_ablation": "/tmp/subrun/feature_ablation.json",
            },
        )
        self.assertTrue(torch.equal(flattened_single, three_dimensional[0]))
        self.assertEqual(tuple(flattened_general.shape), (2, 2))
        self.assertEqual(int(empty_summary["num_valid"]), 0)
        self.assertTrue(torch.equal(resolved_output, three_dimensional))
        self.assertIsNone(mean_empty)
        self.assertIsNone(standard_error_empty)
        self.assertEqual(float(standard_error_single), 0.0)
        with self.assertRaises(ValueError):
            _ = _resolve_reconstruction_output({})

    def test_delete_file_if_exists_removes_existing_file(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "temporary.json")
            with open(path, "w") as file:
                file.write("temporary")

            # when
            _delete_file_if_exists(path)

            # then
            self.assertFalse(os.path.isfile(path))

    def test_run_evaluation_for_subrun_overwrite_rewrites_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            val_acts = torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32
            )
            _write_activation_cache(tmpdir, val_acts=val_acts)
            config = LLMBenchmarkConfig.model_validate(_build_config(tmpdir, "SAE"))
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            _ = _write_identity_checkpoint(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                theta_zero=False,
            )
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                file.write("stale")
            with open(os.path.join(subrun_dir, "gamma_metrics.json"), "w") as file:
                file.write("stale")

            # when
            result = run_evaluation_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=True,
            )

            # then
            with open(result["artifact_paths"]["evaluation"], "r") as file:
                evaluation = json.load(file)
            with open(result["artifact_paths"]["gamma_metrics"], "r") as file:
                gamma_metrics = json.load(file)
            self.assertEqual(int(evaluation["schema_version"]), 1)
            self.assertEqual(int(gamma_metrics["schema_version"]), 1)

    def test_run_evaluation_for_subrun_raises_when_checkpoint_is_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            val_acts = torch.randn(4, 2, dtype=torch.float32)
            _write_activation_cache(tmpdir, val_acts=val_acts)
            config = LLMBenchmarkConfig.model_validate(_build_config(tmpdir, "SAE"))
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            # when / then
            with self.assertRaises(ValueError):
                _ = run_evaluation_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

    def test_run_evaluation_for_subrun_rejects_invalid_checkpoint_payloads(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            val_acts = torch.randn(4, 2, dtype=torch.float32)
            _write_activation_cache(tmpdir, val_acts=val_acts)
            config = LLMBenchmarkConfig.model_validate(_build_config(tmpdir, "SAE"))
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            checkpoint_path = os.path.join(subrun_dir, "model.pt")

            # when / then
            torch.save([1, 2, 3], checkpoint_path)
            with self.assertRaises(ValueError):
                _ = run_evaluation_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            torch.save({"weights": {}}, checkpoint_path)
            with self.assertRaises(ValueError):
                _ = run_evaluation_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

    def test_run_evaluation_for_subrun_returns_r2_none_for_constant_targets(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            val_acts = torch.ones((4, 2), dtype=torch.float32)
            _write_activation_cache(tmpdir, val_acts=val_acts)
            config_payload = _build_config(tmpdir, "SAE")
            config_payload["evaluation"]["l0_is_nonzero"] = False
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            _ = _write_identity_checkpoint(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                theta_zero=False,
            )

            # when
            result = run_evaluation_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            with open(result["artifact_paths"]["evaluation"], "r") as file:
                evaluation = json.load(file)
            self.assertIsNone(evaluation["r2"])
            self.assertEqual(int(evaluation["num_val_samples"]), 4)

    def test_run_evaluation_for_subrun_raises_when_val_split_is_empty(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_empty_val_activation_cache(tmpdir, d_in=2)
            config = LLMBenchmarkConfig.model_validate(_build_config(tmpdir, "SAE"))
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            _ = _write_identity_checkpoint(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                theta_zero=False,
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = run_evaluation_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

    def test_build_and_persist_comparison_summary_aggregates_existing_subruns(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            manifest = {
                "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                "hook": {
                    "hook_name": "mlp_out",
                    "layer": 7,
                    "module_path_resolved": "transformer.h.7.mlp",
                },
                "dataset": {"dataset_id": "fixture", "split": "train"},
                "tokenization": {"max_seq_len": 16},
            }
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(manifest, file, indent=2)

            resolved_config = _build_resolved_comparison_subruns(tmpdir)
            subrun_metrics: Dict[str, Dict[str, Optional[float]]] = {
                "models/variant_a/seed_0/lambda_0p001": {
                    "mse": 1.0,
                    "r2": 0.5,
                    "mean_l0": 2.0,
                    "dead_fraction": 0.1,
                    "gamma_pos_fraction_valid": 0.7,
                },
                "models/variant_a/seed_1/lambda_0p001": {
                    "mse": 3.0,
                    "r2": 0.7,
                    "mean_l0": 4.0,
                    "dead_fraction": 0.3,
                    "gamma_pos_fraction_valid": 0.9,
                },
                "models/variant_b/seed_0/lambda_0p01": {
                    "mse": 2.0,
                    "r2": None,
                    "mean_l0": None,
                    "dead_fraction": 0.2,
                    "gamma_pos_fraction_valid": None,
                },
            }
            for relative_dir, metrics in subrun_metrics.items():
                subrun_dir = os.path.join(tmpdir, relative_dir)
                os.makedirs(subrun_dir, exist_ok=True)
                _write_completed_training_metrics(subrun_dir)
                evaluation = {
                    "mse": metrics["mse"],
                    "r2": metrics["r2"],
                    "mean_l0": metrics["mean_l0"],
                    "dead_fraction": metrics["dead_fraction"],
                    "gamma_summary": {
                        "coverage": {
                            "pos_fraction_valid": metrics["gamma_pos_fraction_valid"]
                        }
                    },
                }
                with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                    json.dump(evaluation, file, indent=2)
                if relative_dir in {
                    "models/variant_a/seed_0/lambda_0p001",
                    "models/variant_a/seed_1/lambda_0p001",
                }:
                    loss_recovered_value = (
                        0.4 if relative_dir.endswith("seed_0/lambda_0p001") else 0.6
                    )
                    with open(
                        os.path.join(subrun_dir, "loss_recovered.json"), "w"
                    ) as file:
                        json.dump(
                            {
                                "loss_recovered": loss_recovered_value,
                                "loss_original": 1.0,
                                "loss_ablated": 2.0,
                                "loss_reconstructed": 1.5,
                            },
                            file,
                            indent=2,
                        )
                    feature_mean = (
                        0.01 if relative_dir.endswith("seed_0/lambda_0p001") else 0.03
                    )
                    with open(
                        os.path.join(subrun_dir, "feature_ablation.json"), "w"
                    ) as file:
                        json.dump(
                            {
                                "summary": {
                                    "mean": feature_mean,
                                    "p90": feature_mean + 0.01,
                                }
                            },
                            file,
                            indent=2,
                        )

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(int(summary["schema_version"]), 1)
            self.assertEqual(int(len(summary["subruns"])), 3)
            self.assertEqual(int(len(summary["aggregates"])), 2)
            self.assertEqual(int(summary["metadata"]["layer"]), 7)
            self.assertEqual(str(summary["metadata"]["hook_name"]), "mlp_out")
            self.assertEqual(
                str(summary["metadata"]["module_path"]), "transformer.h.7.mlp"
            )

            aggregate_by_variant = {
                str(entry["variant_slug"]): entry for entry in summary["aggregates"]
            }
            variant_a = aggregate_by_variant["variant_a"]
            self.assertEqual(int(variant_a["num_seeds"]), 2)
            self.assertAlmostEqual(float(variant_a["mse_mean"]), 2.0, places=8)
            self.assertGreater(float(variant_a["mse_se"]), 0.0)
            self.assertAlmostEqual(float(variant_a["r2_mean"]), 0.6, places=8)
            self.assertAlmostEqual(
                float(variant_a["loss_recovered_mean"]), 0.5, places=8
            )
            self.assertEqual(int(variant_a["num_seeds_loss_recovered"]), 2)
            self.assertAlmostEqual(
                float(variant_a["feature_ablation_mean_delta"]), 0.02, places=8
            )
            self.assertEqual(int(variant_a["num_seeds_feature_ablation"]), 2)

            variant_b = aggregate_by_variant["variant_b"]
            self.assertEqual(int(variant_b["num_seeds"]), 1)
            self.assertAlmostEqual(float(variant_b["mse_mean"]), 2.0, places=8)
            self.assertEqual(float(variant_b["mse_se"]), 0.0)
            self.assertIsNone(variant_b["r2_mean"])
            self.assertIsNone(variant_b["r2_se"])
            self.assertIsNone(variant_b["loss_recovered_mean"])
            self.assertEqual(int(variant_b["num_seeds_loss_recovered"]), 0)

    def test_comparison_summary_prefers_manifest_hook_layer_over_config_layer(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            manifest = {
                "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                "hook": {
                    "hook_name": "mlp_out",
                    "layer": 5,
                    "module_path_resolved": "transformer.h.5.mlp",
                },
                "dataset": {"dataset_id": "fixture", "split": "train"},
                "tokenization": {"max_seq_len": 16},
            }
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(manifest, file, indent=2)

            resolved_config = _build_resolved_comparison_subruns(tmpdir)
            resolved_config["hook"]["layer"] = None
            subrun_dir = os.path.join(tmpdir, "models/variant_a/seed_0/lambda_0p001")
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "mse": 1.0,
                        "r2": 0.5,
                        "mean_l0": 2.0,
                        "dead_fraction": 0.1,
                        "gamma_summary": {"coverage": {"pos_fraction_valid": 0.7}},
                    },
                    file,
                    indent=2,
                )

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(int(summary["metadata"]["layer"]), 5)

    def test_comparison_summary_skips_completed_subruns_without_evaluation(
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
                        "tokenization": {"max_seq_len": 16},
                    },
                    file,
                    indent=2,
                )
            resolved_config = _build_resolved_comparison_subruns(tmpdir)
            subrun_dir = os.path.join(tmpdir, "models/variant_a/seed_0/lambda_0p001")
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(summary["subruns"], [])

    def test_comparison_summary_raises_for_legacy_success_metrics_without_status(
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
                        "tokenization": {"max_seq_len": 16},
                    },
                    file,
                    indent=2,
                )
            resolved_config = _build_resolved_comparison_subruns(tmpdir)
            subrun_dir = os.path.join(tmpdir, "models/variant_a/seed_0/lambda_0p001")
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
                json.dump(legacy_success_training_metrics_payload(), file, indent=2)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "mse": 1.0,
                        "r2": 0.5,
                        "mean_l0": 2.0,
                        "dead_fraction": 0.1,
                    },
                    file,
                    indent=2,
                )

            # when / then
            with self.assertRaises(ValueError) as error:
                _ = build_and_persist_comparison_summary(
                    resolved_config=resolved_config,
                    output_dir=tmpdir,
                )

            # then
            self.assertIn("migrate_llm_training_status.py", str(error.exception))

    def test_comparison_summary_falls_back_to_layer_selector_when_layer_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            manifest = {
                "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                "hook": {
                    "hook_name": "mlp_out",
                    "module_path_resolved": "transformer.h.5.mlp",
                },
                "dataset": {"dataset_id": "fixture", "split": "train"},
                "tokenization": {"max_seq_len": 16},
            }
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(manifest, file, indent=2)

            resolved_config = _build_resolved_comparison_subruns(tmpdir)
            resolved_config["hook"]["layer"] = None
            resolved_config["hook"]["layer_selector"] = "mid"
            subrun_dir = os.path.join(tmpdir, "models/variant_a/seed_0/lambda_0p001")
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "mse": 1.0,
                        "r2": 0.5,
                        "mean_l0": 2.0,
                        "dead_fraction": 0.1,
                        "gamma_summary": {"coverage": {"pos_fraction_valid": 0.7}},
                    },
                    file,
                    indent=2,
                )

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(str(summary["metadata"]["layer"]), "mid")

    def test_comparison_summary_handles_non_dict_hook_payloads(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            manifest = {
                "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                "hook": "not-a-dict",
                "dataset": {"dataset_id": "fixture", "split": "train"},
                "tokenization": {"max_seq_len": 16},
            }
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(manifest, file, indent=2)

            resolved_config = _build_resolved_comparison_subruns(tmpdir)
            resolved_config["hook"] = "not-a-dict"
            subrun_dir = os.path.join(tmpdir, "models/variant_a/seed_0/lambda_0p001")
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "mse": 1.0,
                        "r2": 0.5,
                        "mean_l0": 2.0,
                        "dead_fraction": 0.1,
                        "gamma_summary": {"coverage": {"pos_fraction_valid": 0.7}},
                    },
                    file,
                    indent=2,
                )

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertIsNone(summary["metadata"]["hook_name"])
            self.assertIsNone(summary["metadata"]["layer"])

    def test_comparison_summary_uses_image_metadata_for_image_modality(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            manifest = {
                "schema_version": ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
                "modality": "image",
                "vision_model_id": "hf-internal-testing/tiny-random-vit",
                "hook": {
                    "hook_name": "hidden_state",
                    "layer": 1,
                    "module_path_resolved": "vision.encoder.layer.1.output",
                },
                "dataset": {"dataset_id": "local/images_tiny", "split": "train"},
                "image_size": 64,
                "patch_size": 32,
                "token_selection": "patch_only",
            }
            with open(
                os.path.join(activations_dir, "activations_manifest.json"), "w"
            ) as file:
                json.dump(manifest, file, indent=2)

            resolved_config = {
                "modality": "image",
                "vision": {"model_id": "hf-internal-testing/tiny-random-vit"},
                "hook": {"hook_name": "hidden_state", "layer": 1},
                "subruns": [
                    {
                        "variant_name": "variant_a",
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 0.001,
                        "relative_subrun_dir": "models/variant_a/seed_0/lambda_0p001",
                    }
                ],
            }
            subrun_dir = os.path.join(tmpdir, "models/variant_a/seed_0/lambda_0p001")
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "mse": 1.0,
                        "r2": 0.5,
                        "mean_l0": 2.0,
                        "dead_fraction": 0.1,
                        "gamma_summary": {"coverage": {"pos_fraction_valid": 0.7}},
                    },
                    file,
                    indent=2,
                )

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(summary["metadata"]["modality"], "image")
            self.assertEqual(
                summary["metadata"]["vision_model_id"],
                "hf-internal-testing/tiny-random-vit",
            )
            self.assertEqual(int(summary["metadata"]["image_size"]), 64)
            self.assertEqual(int(summary["metadata"]["patch_size"]), 32)
            self.assertEqual(summary["metadata"]["token_selection"], "patch_only")
            self.assertNotIn("llm_id", summary["metadata"])

    def test_run_evaluation_for_subrun_reuses_all_when_eval_artifacts_exist(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(_build_config(tmpdir, "SAE"))
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            for filename in ("evaluation.json", "gamma_metrics.json"):
                with open(os.path.join(subrun_dir, filename), "w") as file:
                    json.dump({}, file)

            # when
            result = run_evaluation_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            self.assertTrue(result["reused_all"])
            self.assertEqual(result["subrun_dir"], subrun_dir)

    def test_run_evaluation_for_subrun_runs_optional_artifact_writers(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _build_config(tmpdir, "SAE")
            config_payload["loss_recovered"] = {"enabled": True}
            config_payload["feature_ablation"] = {"enabled": True}
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            for filename in ("evaluation.json", "gamma_metrics.json"):
                with open(os.path.join(subrun_dir, filename), "w") as file:
                    json.dump({}, file)
            checkpoint_path = os.path.join(subrun_dir, "model.pt")
            torch.save({"state_dict": {}}, checkpoint_path)

            class DummyModel:
                def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
                    del state_dict

                def to(self, device: str) -> "DummyModel":
                    del device
                    return self

                def eval(self) -> "DummyModel":
                    return self

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.metrics.resolve_manifest_path",
                return_value=os.path.join(tmpdir, "activations_manifest.json"),
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.load_activation_manifest",
                return_value={"activation": {"d_in": 2}},
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.resolve_runtime_device",
                return_value="cpu",
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.build_model_for_subrun",
                return_value=(DummyModel(), 0.0, 0, {"d_hidden": 2}),
            ), patch(
                "sgnsae.experiments.llm_benchmark.metrics.compute_and_write_subrun_loss_recovered",
                return_value={},
            ) as loss_recovered_writer, patch(
                "sgnsae.experiments.llm_benchmark.metrics.compute_and_write_feature_ablation",
                return_value={},
            ) as feature_ablation_writer:
                _ = run_evaluation_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            self.assertEqual(loss_recovered_writer.call_count, 1)
            self.assertEqual(feature_ablation_writer.call_count, 1)

    def test_comparison_summary_handles_non_dict_text_metadata_payloads(
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
                        "hook": {"hook_name": "mlp_out", "layer": 1},
                        "dataset": "invalid",
                        "tokenization": "invalid",
                    },
                    file,
                    indent=2,
                )

            resolved_config = {
                "modality": "text",
                "llm": "invalid",
                "hook": {"hook_name": "mlp_out", "layer": 1},
                "subruns": [
                    {
                        "variant_name": "variant_a",
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 0.001,
                        "relative_subrun_dir": "models/variant_a/seed_0/lambda_0p001",
                    }
                ],
            }
            subrun_dir = os.path.join(tmpdir, "models/variant_a/seed_0/lambda_0p001")
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "mse": 1.0,
                        "r2": 0.5,
                        "mean_l0": 2.0,
                        "dead_fraction": 0.1,
                        "gamma_summary": {"coverage": {"pos_fraction_valid": 0.7}},
                    },
                    file,
                    indent=2,
                )
            with open(os.path.join(subrun_dir, "feature_ablation.json"), "w") as file:
                json.dump({"summary": "invalid"}, file, indent=2)

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(summary["metadata"]["modality"], "text")
            self.assertIsNone(summary["metadata"]["dataset_id"])
            self.assertIsNone(summary["metadata"]["max_seq_len"])
            self.assertIsNone(summary["metadata"]["llm_id"])

    def test_comparison_summary_image_mode_handles_non_dict_vision(self) -> None:
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
                        "modality": "image",
                        "vision_model_id": "manifest-vision-id",
                        "hook": {"hook_name": "hidden_state", "layer": 0},
                        "dataset": {"dataset_id": "fixture", "split": "train"},
                    },
                    file,
                    indent=2,
                )

            resolved_config = {
                "modality": "image",
                "vision": "invalid",
                "hook": {"hook_name": "hidden_state", "layer": 0},
                "subruns": [
                    {
                        "variant_name": "variant_a",
                        "variant_slug": "variant_a",
                        "seed": 0,
                        "sweep_param": "lambda",
                        "sweep_value": 0.001,
                        "relative_subrun_dir": "models/variant_a/seed_0/lambda_0p001",
                    }
                ],
            }
            subrun_dir = os.path.join(tmpdir, "models/variant_a/seed_0/lambda_0p001")
            os.makedirs(subrun_dir, exist_ok=True)
            _write_completed_training_metrics(subrun_dir)
            with open(os.path.join(subrun_dir, "evaluation.json"), "w") as file:
                json.dump(
                    {
                        "mse": 1.0,
                        "r2": 0.5,
                        "mean_l0": 2.0,
                        "dead_fraction": 0.1,
                        "gamma_summary": {"coverage": {"pos_fraction_valid": 0.7}},
                    },
                    file,
                    indent=2,
                )

            # when
            summary = build_and_persist_comparison_summary(
                resolved_config=resolved_config,
                output_dir=tmpdir,
            )

            # then
            self.assertEqual(summary["metadata"]["modality"], "image")
            self.assertEqual(
                summary["metadata"]["vision_model_id"], "manifest-vision-id"
            )
