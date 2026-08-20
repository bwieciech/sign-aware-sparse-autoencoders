import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import torch

from sgnsae.experiments.llm_benchmark.config import (
    LLMBenchmarkConfig,
    TrainingConfig,
    resolve_benchmark_config,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    ACTIVATIONS_MANIFEST_SCHEMA_VERSION,
)
from sgnsae.experiments.llm_benchmark.training import (
    _apply_initialization_overrides,
    _build_resample_generators_for_subruns,
    _collect_threshold_gradient_norms,
    _collect_threshold_gradient_norms_per_instance,
    _collect_polarity_flip_statistics,
    _collect_polarity_flip_statistics_per_instance,
    _collect_polarity_usage_statistics,
    _collect_polarity_usage_statistics_per_instance,
    _collect_threshold_statistics,
    _collect_threshold_statistics_per_instance,
    _copy_instance_0_parameters_to_all_instances,
    _delete_file_if_exists,
    _distribution_summary,
    _distribution_summary_per_instance,
    _dump_training_metrics_snapshot,
    _load_continuation_state,
    _next_activation_batch,
    _reassemble_state_dict_from_instances,
    _resolve_reconstruction_output,
    _resolve_manifest_d_in,
    _resolve_resample_latents_threshold,
    _resolve_split_sample_count,
    _run_optimizer_warmup,
    _should_use_mixed_precision,
    _slice_state_dict_to_single_instance,
    _dominant_polarity_per_instance,
    _latents_with_instance_dimension,
    _threshold_parameters,
    build_model_for_subrun,
    instantiate_sae_model,
    load_activation_manifest,
    resolve_manifest_path,
    resolve_runtime_device,
    run_training_for_lambda_chunk,
    run_training_for_subrun,
)
from sgnsae.models.sae_base import SAEBase
from sgnsae.models import (
    AbsTopKGatedMagnitudeSAE,
    AbsTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    SAE,
    SignedSoftThresholdSAE,
    TopKSAE,
)

_DOWNSTREAM_EVAL_FILENAMES = (
    "evaluation.json",
    "gamma_metrics.json",
    "loss_recovered.json",
    "feature_ablation.json",
    "duplication_metrics.json",
    "latent_examples.json",
)

_DOWNSTREAM_SUBRUN_DIRNAMES = ("latent_examples_assets",)


def _build_config(
    output_dir: str,
    sae_class: str = "JumpReLUSAE",
    sign_aware: bool = False,
) -> Dict[str, Any]:
    variant: Dict[str, Any] = {
        "name": "variant",
        "sae_class": sae_class,
        "sign_aware": sign_aware,
        "d_hidden": 6,
    }
    if sae_class in {"TopKSAE", "AbsTopKSAE", "AbsTopKGatedMagnitudeSAE"}:
        variant["topk_k"] = 2
    else:
        variant["sparsity_coeff"] = 1e-3
    if sae_class == "JumpReLUSAE":
        variant["theta_init"] = 0.01
        variant["jump_init"] = 0.02
    if sae_class == "SignedSoftThresholdSAE":
        variant["theta_init"] = 0.01
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
            "train_steps": 5,
            "batch_size": 4,
            "log_every_n_steps": 2,
            "resample_dead_latents": True,
            "resample_latents_steps": 4,
            "threshold_warmup_steps": 1,
            "mixed_precision": False,
        },
        "variants": [variant],
    }


def _build_lambda_sweep_config(
    output_dir: str,
    sae_class: str = "SAE",
    sign_aware: bool = False,
    resample_dead_latents: bool = True,
) -> Dict[str, Any]:
    config_payload = _build_config(
        output_dir=output_dir,
        sae_class=sae_class,
        sign_aware=sign_aware,
    )
    config_payload["variants"][0]["sparsity_coeff"] = [1e-3, 1e-4]
    config_payload["training"]["resample_dead_latents"] = bool(resample_dead_latents)
    config_payload["training"]["mixed_precision"] = False
    return config_payload


def _build_k_sweep_config(
    output_dir: str,
    sae_class: str = "TopKSAE",
    topk_k_list: Optional[List[int]] = None,
) -> Dict[str, Any]:
    config_payload = _build_config(output_dir=output_dir, sae_class=sae_class)
    config_payload["variants"][0]["topk_k"] = topk_k_list or [2, 4]
    config_payload["training"]["resample_dead_latents"] = False
    config_payload["training"]["mixed_precision"] = False
    return config_payload


def _write_activation_cache(output_dir: str) -> None:
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    train_acts = torch.randn(12, 4, dtype=torch.float32)
    val_acts = torch.randn(8, 4, dtype=torch.float32)
    torch.save({"acts": train_acts}, os.path.join(activations_dir, "shard_00000.pt"))
    torch.save({"acts": val_acts}, os.path.join(activations_dir, "shard_00001.pt"))

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
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)


def _copy_activation_cache(source_output_dir: str, target_output_dir: str) -> None:
    source_dir = os.path.join(source_output_dir, "activations")
    target_dir = os.path.join(target_output_dir, "activations")
    shutil.copytree(source_dir, target_dir)


def _create_downstream_eval_artifacts(subrun_dir: str) -> None:
    os.makedirs(subrun_dir, exist_ok=True)
    for filename in _DOWNSTREAM_EVAL_FILENAMES:
        with open(os.path.join(subrun_dir, filename), "w") as file:
            file.write("{}")
    for dirname in _DOWNSTREAM_SUBRUN_DIRNAMES:
        assets_dir = os.path.join(subrun_dir, dirname)
        os.makedirs(assets_dir, exist_ok=True)
        with open(os.path.join(assets_dir, "artifact.txt"), "w") as file:
            file.write("{}")


def _resolve_single_subrun(
    config_payload: Dict[str, Any],
) -> tuple[LLMBenchmarkConfig, Dict[str, Any], Dict[str, Any]]:
    config = LLMBenchmarkConfig.model_validate(config_payload)
    resolved_config = resolve_benchmark_config(config)
    return config, resolved_config, resolved_config["subruns"][0]


def _resolve_variant_and_chunk_subruns(
    resolved_config: Dict[str, Any],
    seed: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    variant = dict(resolved_config["variants"][0])
    subruns_for_seed = []
    for sweep_index, subrun in enumerate(resolved_config["subruns"]):
        if int(subrun["seed"]) != int(seed):
            continue
        subrun_with_index = dict(subrun)
        subrun_with_index["sweep_index"] = int(sweep_index)
        subruns_for_seed.append(subrun_with_index)
    return variant, subruns_for_seed


class _FakeInfiniteLossModel(torch.nn.Module):
    def __init__(self, d_hidden: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.d_hidden = int(d_hidden)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "x_hat": x.unsqueeze(0),
            "latents": torch.zeros(
                (1, int(x.shape[0]), int(self.d_hidden)),
                dtype=torch.float32,
                device=x.device,
            ),
        }

    def calculate_losses(
        self,
        y_true: torch.Tensor,
        sparsity_coeffs: torch.Tensor,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del y_true
        del sparsity_coeffs
        del output
        losses = torch.tensor(
            [float("inf")], dtype=torch.float32, device=self.scale.device
        )
        return {
            "losses": losses,
            "reconstruction_losses": losses,
        }

    def resample_dead_latents(
        self, inactive_latents_count: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(inactive_latents_count, dtype=torch.bool)


class _FakeFiniteNoSparsityLossesModel(torch.nn.Module):
    def __init__(self, n_instances: int, d_hidden: int) -> None:
        super().__init__()
        self.n_instances = int(n_instances)
        self.d_hidden = int(d_hidden)
        self.scale = torch.nn.Parameter(
            torch.ones((self.n_instances,), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_hat = x.unsqueeze(0) * self.scale.reshape(self.n_instances, 1, 1)
        latents = torch.zeros(
            (self.n_instances, int(x.shape[0]), int(self.d_hidden)),
            dtype=torch.float32,
            device=x.device,
        )
        if int(self.d_hidden) > 0:
            latents[:, :, 0] = x[:, 0].unsqueeze(0).expand(self.n_instances, -1)
        if int(self.d_hidden) > 1:
            latents[:, :, 1] = -x[:, 0].unsqueeze(0).expand(self.n_instances, -1)
        return {
            "x_hat": x_hat,
            "latents": latents,
        }

    def calculate_losses(
        self,
        y_true: torch.Tensor,
        sparsity_coeffs: torch.Tensor,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del sparsity_coeffs
        reconstruction_losses = ((output["x_hat"] - y_true) ** 2).mean(dim=(1, 2))
        losses = reconstruction_losses + self.scale * 0.0
        return {
            "losses": losses,
            "reconstruction_losses": reconstruction_losses,
        }

    def resample_dead_latents(
        self,
        inactive_latents_count: torch.Tensor,
        generators: Any = None,
    ) -> torch.Tensor:
        del generators
        return torch.zeros_like(inactive_latents_count, dtype=torch.bool)


class _FakeWarmupExceptionModel(torch.nn.Module):
    def __init__(self, n_instances: int, d_hidden: int) -> None:
        super().__init__()
        self.n_instances = int(n_instances)
        self.d_hidden = int(d_hidden)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "x_hat": x.unsqueeze(0).expand(self.n_instances, -1, -1),
            "latents": torch.zeros(
                (self.n_instances, int(x.shape[0]), int(self.d_hidden)),
                dtype=torch.float32,
                device=x.device,
            ),
        }

    def calculate_losses(
        self,
        y_true: torch.Tensor,
        sparsity_coeffs: torch.Tensor,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del y_true
        del sparsity_coeffs
        del output
        raise RuntimeError("warmup exploded")

    def resample_dead_latents(
        self,
        inactive_latents_count: torch.Tensor,
        generators: Any = None,
    ) -> torch.Tensor:
        del generators
        return torch.zeros_like(inactive_latents_count, dtype=torch.bool)


class _FakeEnabledGradScaler:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args
        del kwargs

    def is_enabled(self) -> bool:
        return True

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


class TestLLMBenchmarkTraining(unittest.TestCase):
    def test_next_activation_batch_raises_when_dataset_yields_no_batches(self) -> None:
        # given
        train_dataset: List[torch.Tensor] = []
        train_iterator = iter(train_dataset)

        # when / then
        with self.assertRaises(ValueError):
            _ = _next_activation_batch(train_dataset, train_iterator)

    def test_run_optimizer_warmup_returns_failure_payload_for_non_finite_loss(
        self,
    ) -> None:
        # given
        class _DisabledScaler:
            def is_enabled(self) -> bool:
                return False

        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        train_dataset = [torch.ones((1, 2), dtype=torch.float32)]

        # when
        failure = _run_optimizer_warmup(
            warmup_steps=1,
            train_dataset=train_dataset,
            model=model,
            optimizer=optimizer,
            scaler=_DisabledScaler(),
            grad_clip_norm=1.0,
            use_mixed_precision=False,
            mixed_precision_dtype=torch.float16,
            runtime_device="cpu",
            compute_total_loss=lambda x: torch.tensor(float("inf"), device=x.device),
        )

        # then
        self.assertIsNotNone(failure)
        self.assertEqual(str(failure["phase"]), "warmup")
        self.assertEqual(int(failure["warmup_step"]), 1)
        self.assertEqual(float(optimizer.param_groups[0]["lr"]), 0.1)

    def test_run_optimizer_warmup_uses_disabled_scaler_grad_clip_path(self) -> None:
        # given
        class _DisabledScaler:
            def is_enabled(self) -> bool:
                return False

        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        initial_weight = model.weight.detach().clone()
        train_dataset = [torch.tensor([[1.0, 2.0]], dtype=torch.float32)]

        # when
        result = _run_optimizer_warmup(
            warmup_steps=1,
            train_dataset=train_dataset,
            model=model,
            optimizer=optimizer,
            scaler=_DisabledScaler(),
            grad_clip_norm=1.0,
            use_mixed_precision=False,
            mixed_precision_dtype=torch.float16,
            runtime_device="cpu",
            compute_total_loss=lambda x: model(x).sum(),
        )

        # then
        self.assertIsNone(result)
        self.assertTrue(torch.equal(model.weight.detach(), initial_weight))
        self.assertEqual(float(optimizer.param_groups[0]["lr"]), 0.1)
        self.assertTrue(bool(optimizer.state))

    def test_run_optimizer_warmup_uses_enabled_scaler_grad_clip_path(self) -> None:
        # given
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        initial_weight = model.weight.detach().clone()
        train_dataset = [torch.tensor([[1.0, 2.0]], dtype=torch.float32)]

        # when
        result = _run_optimizer_warmup(
            warmup_steps=1,
            train_dataset=train_dataset,
            model=model,
            optimizer=optimizer,
            scaler=_FakeEnabledGradScaler(),
            grad_clip_norm=1.0,
            use_mixed_precision=False,
            mixed_precision_dtype=torch.float16,
            runtime_device="cpu",
            compute_total_loss=lambda x: model(x).sum(),
        )

        # then
        self.assertIsNone(result)
        self.assertTrue(torch.equal(model.weight.detach(), initial_weight))
        self.assertEqual(float(optimizer.param_groups[0]["lr"]), 0.1)
        self.assertTrue(bool(optimizer.state))

    def test_collect_threshold_gradient_norms_per_instance_skips_missing_grads(
        self,
    ) -> None:
        # given
        class _NoGradientModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.jump_param = torch.nn.Parameter(torch.tensor([1.0]))

        model = _NoGradientModel()

        # when
        gradient_stats = _collect_threshold_gradient_norms_per_instance(model)

        # then
        self.assertEqual(len(gradient_stats), 1)
        self.assertNotIn("grad_norm_jump", gradient_stats[0])

    def test_resolve_variant_and_chunk_subruns_skips_non_matching_seeds(self) -> None:
        # given
        resolved_config = {
            "variants": [{"variant_slug": "variant_a"}],
            "subruns": [
                {"seed": 0, "variant_slug": "variant_a"},
                {"seed": 1, "variant_slug": "variant_a"},
            ],
        }

        # when
        variant, subruns = _resolve_variant_and_chunk_subruns(
            resolved_config=resolved_config,
            seed=0,
        )

        # then
        self.assertEqual(variant["variant_slug"], "variant_a")
        self.assertEqual(len(subruns), 1)
        self.assertEqual(int(subruns[0]["seed"]), 0)

    def test_fake_loss_models_cover_forward_and_resample_branches(self) -> None:
        # given
        base_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        model_hidden_zero = _FakeFiniteNoSparsityLossesModel(n_instances=1, d_hidden=0)
        model_hidden_one = _FakeFiniteNoSparsityLossesModel(n_instances=1, d_hidden=1)
        inactive_latents = torch.zeros((1, 2), dtype=torch.int64)
        infinite_model = _FakeInfiniteLossModel(d_hidden=2)

        # when
        output_hidden_zero = model_hidden_zero.forward(base_input)
        output_hidden_one = model_hidden_one.forward(base_input)
        resampled_finite = model_hidden_one.resample_dead_latents(
            inactive_latents_count=inactive_latents,
            generators=object(),
        )
        resampled_infinite = infinite_model.resample_dead_latents(inactive_latents)

        # then
        self.assertEqual(tuple(output_hidden_zero["latents"].shape), (1, 2, 0))
        self.assertEqual(tuple(output_hidden_one["latents"].shape), (1, 2, 1))
        self.assertTrue(
            bool(torch.equal(resampled_finite, torch.zeros_like(resampled_finite)))
        )
        self.assertTrue(
            bool(torch.equal(resampled_infinite, torch.zeros_like(resampled_infinite)))
        )

    def test_run_training_for_subrun_writes_artifacts_and_metrics(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config = LLMBenchmarkConfig.model_validate(_build_config(tmpdir))
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            # when
            result = run_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            subrun_dir = result["subrun_dir"]
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model.pt")))
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "model_config.json"))
            )
            metrics_path = os.path.join(subrun_dir, "training_metrics.json")
            self.assertTrue(os.path.isfile(metrics_path))

            with open(metrics_path, "r") as file:
                metrics = json.load(file)
            self.assertEqual(metrics["schema_version"], 1)
            self.assertEqual(str(metrics["status"]), "completed")
            self.assertNotIn("terminated_early", metrics)
            self.assertEqual(metrics["train_steps"], 5)
            self.assertEqual(metrics["thresholds_unfrozen_at_step"], 2)
            self.assertEqual(metrics["resample_latents_threshold"], 4)
            self.assertEqual(len(metrics["history"]), 3)
            first_entry = metrics["history"][0]
            self.assertEqual(metrics["history"][-1]["step"], 5)
            self.assertIn("loss_total", metrics["history"][-1])
            self.assertIn("loss_recon", metrics["history"][-1])
            self.assertIn("mean_l0", metrics["history"][-1])
            self.assertIn("polarity_positive_fraction_nonzero", metrics["history"][-1])
            self.assertIn("polarity_negative_fraction_nonzero", metrics["history"][-1])
            self.assertIn(
                "polarity_bidirectional_usage_fraction", metrics["history"][-1]
            )
            self.assertIn("polarity_dominance_flip_rate", metrics["history"][-1])
            self.assertIn("polarity_dominance_flip_indicator", metrics["history"][-1])
            self.assertIn("polarity_dominance_flip_rate", first_entry)
            self.assertIn("polarity_dominance_flip_indicator", first_entry)
            self.assertEqual(
                float(first_entry["polarity_dominance_flip_rate"]),
                0.0,
            )
            self.assertEqual(
                float(first_entry["polarity_dominance_flip_indicator"]),
                0.0,
            )

            model_config_path = os.path.join(subrun_dir, "model_config.json")
            with open(model_config_path, "r") as file:
                model_config = json.load(file)
            self.assertEqual(model_config["resample_latents_threshold"], 4)

    def test_run_training_for_subrun_restarts_training_after_warmup(self) -> None:
        # given
        class RecordingWarmupModel(torch.nn.Module):
            def __init__(self, d_hidden: int) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.d_hidden = int(d_hidden)
                self.act_offset = torch.nn.Parameter(
                    torch.zeros((1, int(d_hidden)), dtype=torch.float32)
                )
                self.seen_batches: List[torch.Tensor] = []
                self.seen_weights: List[float] = []
                self.resample_calls = 0

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                self.seen_batches.append(x.detach().cpu().clone())
                self.seen_weights.append(float(self.weight.detach().item()))
                return {
                    "x_hat": x.unsqueeze(0) + 0.0 * self.weight,
                    "latents": torch.zeros(
                        (1, int(x.shape[0]), int(self.d_hidden)),
                        dtype=torch.float32,
                        device=x.device,
                    ),
                }

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del y_true
                del sparsity_coeffs
                del output
                loss = (self.weight - 1.0) ** 2
                return {
                    "losses": loss.reshape(1),
                    "reconstruction_losses": loss.reshape(1),
                    "sparsity_losses": torch.zeros(1, device=self.weight.device),
                }

            def resample_dead_latents(
                self,
                inactive_latents_count: torch.Tensor,
                generators: object = None,
            ) -> torch.Tensor:
                del generators
                self.resample_calls += 1
                return torch.zeros_like(inactive_latents_count, dtype=torch.bool)

        class RecordingDataset:
            def __iter__(self) -> Any:
                batches = [
                    torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0, 0.0],
                            [4.0, 0.0, 0.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        [
                            [5.0, 0.0, 0.0, 0.0],
                            [6.0, 0.0, 0.0, 0.0],
                            [7.0, 0.0, 0.0, 0.0],
                            [8.0, 0.0, 0.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                ]
                return iter(batches)

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir)
            config_payload["training"]["train_steps"] = 2
            config_payload["training"]["log_every_n_steps"] = 1
            config_payload["training"]["warmup_steps"] = 3
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            variant_config = resolved_config["variants"][0]
            model = RecordingWarmupModel(d_hidden=int(variant_config["d_hidden"]))

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.build_model_for_subrun",
                return_value=(model, 1e-3, 4, variant_config),
            ), patch(
                "sgnsae.experiments.llm_benchmark.training._create_train_activation_dataset",
                return_value=RecordingDataset(),
            ):
                result = run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            with open(result["artifact_paths"]["training_metrics"], "r") as file:
                metrics = json.load(file)
            self.assertEqual(metrics["warmup_steps"], 3)
            self.assertEqual(metrics["thresholds_unfrozen_at_step"], 2)
            self.assertEqual(len(metrics["history"]), 2)
            self.assertEqual(model.resample_calls, 2)
            self.assertEqual(len(model.seen_batches), 5)
            self.assertTrue(torch.equal(model.seen_batches[0], model.seen_batches[2]))
            self.assertTrue(torch.equal(model.seen_batches[0], model.seen_batches[3]))
            self.assertTrue(torch.equal(model.seen_batches[1], model.seen_batches[4]))
            self.assertEqual(model.seen_weights[0], model.seen_weights[1])
            self.assertEqual(model.seen_weights[0], model.seen_weights[2])
            self.assertEqual(model.seen_weights[0], model.seen_weights[3])

    def test_run_training_for_lambda_chunk_writes_artifacts_and_slices_state_dict(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )

            # when
            result = run_training_for_lambda_chunk(
                config=config,
                resolved_config=resolved_config,
                variant=variant,
                seed=0,
                chunk_subruns=chunk_subruns,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            self.assertEqual(len(result["subrun_dirs"]), 2)
            for subrun in chunk_subruns:
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model.pt")))
                self.assertTrue(
                    os.path.isfile(os.path.join(subrun_dir, "model_config.json"))
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(subrun_dir, "training_metrics.json"))
                )

                with open(
                    os.path.join(subrun_dir, "training_metrics.json"), "r"
                ) as file:
                    metrics = json.load(file)
                self.assertEqual(str(metrics["status"]), "completed")
                self.assertNotIn("terminated_early", metrics)
                self.assertEqual(metrics["sweep_value"], subrun["sweep_value"])
                self.assertGreaterEqual(len(metrics["history"]), 1)
                first_entry = metrics["history"][0]
                history_entry = metrics["history"][-1]
                self.assertIsInstance(history_entry["loss_total"], float)
                self.assertIsInstance(history_entry["loss_recon"], float)
                self.assertIsInstance(history_entry["mean_l0"], float)
                self.assertIsInstance(history_entry["mean_l1"], float)
                self.assertIsInstance(
                    history_entry["polarity_positive_fraction_nonzero"], float
                )
                self.assertIsInstance(
                    history_entry["polarity_negative_fraction_nonzero"], float
                )
                self.assertIsInstance(
                    history_entry["polarity_bidirectional_usage_fraction"], float
                )
                self.assertIsInstance(
                    history_entry["polarity_dominance_flip_rate"], float
                )
                self.assertIsInstance(
                    history_entry["polarity_dominance_flip_indicator"], float
                )
                self.assertIn("polarity_dominance_flip_rate", first_entry)
                self.assertIn("polarity_dominance_flip_indicator", first_entry)
                self.assertEqual(
                    float(first_entry["polarity_dominance_flip_rate"]),
                    0.0,
                )
                self.assertEqual(
                    float(first_entry["polarity_dominance_flip_indicator"]),
                    0.0,
                )
                with open(os.path.join(subrun_dir, "model_config.json"), "r") as file:
                    model_config = json.load(file)
                self.assertNotIn("sweep_index", model_config["subrun"])

                model_payload = torch.load(
                    os.path.join(subrun_dir, "model.pt"),
                    map_location="cpu",
                )
                state_dict = model_payload["state_dict"]
                for tensor in list(state_dict.values()) + [torch.tensor(1.0)]:
                    self.assertEqual(tensor.device.type, "cpu")
                    if tensor.ndim >= 1:
                        self.assertEqual(int(tensor.shape[0]), 1)

    def test_run_training_for_subrun_persists_effective_loss_knobs(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            payload = _build_config(
                output_dir=tmpdir,
                sae_class="GatedSAE",
                sign_aware=True,
            )
            payload["variants"][0]["aux_loss_coeff"] = 0.75
            payload["variants"][0]["dead_zone_reg_coeff"] = 0.2
            payload["variants"][0]["dead_zone_reg_type"] = "l2"
            config = LLMBenchmarkConfig.model_validate(payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            # when
            result = run_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            with open(result["artifact_paths"]["model_config"], "r") as file:
                model_config = json.load(file)
            with open(result["artifact_paths"]["training_metrics"], "r") as file:
                training_metrics = json.load(file)
            self.assertEqual(
                model_config["loss_knobs"],
                {
                    "aux_loss_coeff": 0.75,
                    "dead_zone_reg_coeff": 0.2,
                    "dead_zone_reg_type": "l2",
                },
            )
            self.assertEqual(
                training_metrics["loss_knobs"],
                {
                    "aux_loss_coeff": 0.75,
                    "dead_zone_reg_coeff": 0.2,
                    "dead_zone_reg_type": "l2",
                },
            )

    def test_run_training_for_lambda_chunk_checkpoint_and_dump(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(output_dir=tmpdir)
            config_payload["training"]["checkpoint_model_steps"] = 2
            config_payload["training"]["dump_training_metrics_steps"] = 3
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training"
                "._dump_training_metrics_snapshot",
                wraps=_dump_training_metrics_snapshot,
            ) as mock_dump:
                result = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=chunk_subruns,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            n_instances = len(chunk_subruns)
            for subrun_dir in result["subrun_dirs"]:
                self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model_2.pt")))
                self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model_4.pt")))
                self.assertFalse(os.path.isfile(os.path.join(subrun_dir, "model_5.pt")))
            self.assertEqual(mock_dump.call_count, n_instances)

    def test_run_training_for_lambda_chunk_persists_effective_loss_knobs(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            payload = _build_lambda_sweep_config(
                output_dir=tmpdir,
                sae_class="GatedSAE",
                sign_aware=True,
            )
            payload["variants"][0]["aux_loss_coeff"] = 0.6
            payload["variants"][0]["dead_zone_reg_coeff"] = 0.1
            payload["variants"][0]["dead_zone_reg_type"] = "l2"
            config = LLMBenchmarkConfig.model_validate(payload)
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )

            # when
            result = run_training_for_lambda_chunk(
                config=config,
                resolved_config=resolved_config,
                variant=variant,
                seed=0,
                chunk_subruns=chunk_subruns,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            for artifact_paths in result["artifact_paths_by_subrun"]:
                with open(artifact_paths["model_config"], "r") as file:
                    model_config = json.load(file)
                with open(artifact_paths["training_metrics"], "r") as file:
                    training_metrics = json.load(file)
                self.assertEqual(
                    model_config["loss_knobs"],
                    {
                        "aux_loss_coeff": 0.6,
                        "dead_zone_reg_coeff": 0.1,
                        "dead_zone_reg_type": "l2",
                    },
                )
                self.assertEqual(
                    training_metrics["loss_knobs"],
                    {
                        "aux_loss_coeff": 0.6,
                        "dead_zone_reg_coeff": 0.1,
                        "dead_zone_reg_type": "l2",
                    },
                )

    def test_run_training_for_lambda_chunk_logs_threshold_diagnostics_per_instance(
        self,
    ) -> None:
        # given
        thresholded_configs = [
            ("JumpReLUSAE", False, ["theta_mean", "jump_mean"]),
            ("SignedSoftThresholdSAE", False, ["theta_mean"]),
            ("GatedSAE", True, ["delta_pos_mean", "delta_neg_mean"]),
        ]

        for sae_class, sign_aware, expected_keys in thresholded_configs:
            with self.subTest(sae_class=sae_class):
                with tempfile.TemporaryDirectory() as tmpdir:
                    _write_activation_cache(tmpdir)
                    config = LLMBenchmarkConfig.model_validate(
                        _build_lambda_sweep_config(
                            output_dir=tmpdir,
                            sae_class=sae_class,
                            sign_aware=sign_aware,
                        )
                    )
                    resolved_config = resolve_benchmark_config(config)
                    variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                        resolved_config, seed=0
                    )

                    # when
                    _ = run_training_for_lambda_chunk(
                        config=config,
                        resolved_config=resolved_config,
                        variant=variant,
                        seed=0,
                        chunk_subruns=chunk_subruns,
                        output_dir=tmpdir,
                        overwrite=False,
                    )

                    # then
                    for subrun in chunk_subruns:
                        with open(
                            os.path.join(
                                tmpdir,
                                subrun["relative_subrun_dir"],
                                "training_metrics.json",
                            ),
                            "r",
                        ) as file:
                            metrics = json.load(file)
                        self.assertGreaterEqual(len(metrics["history"]), 1)
                        history_entry = metrics["history"][-1]
                        self.assertIn("dead_zone_frac_mean", history_entry)
                        self.assertIsInstance(
                            history_entry["dead_zone_frac_mean"], float
                        )
                        for key in expected_keys:
                            self.assertIn(key, history_entry)
                            self.assertIsInstance(history_entry[key], float)

    def test_run_training_for_lambda_chunk_is_chunking_invariant_without_manual_seed(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            full_chunk_dir = os.path.join(tmpdir, "full_chunk")
            single_chunk_dir = os.path.join(tmpdir, "single_chunk")
            os.makedirs(full_chunk_dir, exist_ok=True)
            os.makedirs(single_chunk_dir, exist_ok=True)
            _write_activation_cache(full_chunk_dir)
            _copy_activation_cache(full_chunk_dir, single_chunk_dir)

            full_config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=full_chunk_dir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            full_resolved = resolve_benchmark_config(full_config)
            full_variant, full_subruns = _resolve_variant_and_chunk_subruns(
                full_resolved, seed=0
            )

            single_config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=single_chunk_dir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            single_resolved = resolve_benchmark_config(single_config)
            single_variant, single_subruns = _resolve_variant_and_chunk_subruns(
                single_resolved, seed=0
            )

            # when
            _ = run_training_for_lambda_chunk(
                config=full_config,
                resolved_config=full_resolved,
                variant=full_variant,
                seed=0,
                chunk_subruns=full_subruns,
                output_dir=full_chunk_dir,
                overwrite=False,
            )
            for subrun in single_subruns:
                _ = run_training_for_lambda_chunk(
                    config=single_config,
                    resolved_config=single_resolved,
                    variant=single_variant,
                    seed=0,
                    chunk_subruns=[subrun],
                    output_dir=single_chunk_dir,
                    overwrite=False,
                )

            # then
            for subrun in full_subruns:
                full_model_path = os.path.join(
                    full_chunk_dir, subrun["relative_subrun_dir"], "model.pt"
                )
                single_model_path = os.path.join(
                    single_chunk_dir, subrun["relative_subrun_dir"], "model.pt"
                )
                full_state_dict = torch.load(full_model_path, map_location="cpu")[
                    "state_dict"
                ]
                single_state_dict = torch.load(single_model_path, map_location="cpu")[
                    "state_dict"
                ]
                self.assertEqual(
                    set(full_state_dict.keys()), set(single_state_dict.keys())
                )
                for parameter_name in full_state_dict:
                    torch.testing.assert_close(
                        full_state_dict[parameter_name],
                        single_state_dict[parameter_name],
                        rtol=0.0,
                        atol=1e-8,
                    )

    def test_run_training_for_lambda_chunk_passes_stable_instance_local_generators(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            first_run_dir = os.path.join(tmpdir, "run_one")
            second_run_dir = os.path.join(tmpdir, "run_two")
            os.makedirs(first_run_dir, exist_ok=True)
            os.makedirs(second_run_dir, exist_ok=True)
            _write_activation_cache(first_run_dir)
            _copy_activation_cache(first_run_dir, second_run_dir)

            first_config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=first_run_dir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=True,
                )
            )
            first_resolved = resolve_benchmark_config(first_config)
            first_variant, first_subruns = _resolve_variant_and_chunk_subruns(
                first_resolved, seed=0
            )

            second_config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=second_run_dir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=True,
                )
            )
            second_resolved = resolve_benchmark_config(second_config)
            second_variant, second_subruns = _resolve_variant_and_chunk_subruns(
                second_resolved, seed=0
            )

            original_resample_dead_latents = SAEBase.resample_dead_latents
            first_run_generator_seeds: List[Tuple[int, ...]] = []
            second_run_generator_seeds: List[Tuple[int, ...]] = []

            def _capture_first_run_generator_seeds(
                model: SAEBase,
                inactive_latents_count: torch.Tensor,
                generators: Any = None,
            ) -> torch.Tensor:
                self.assertIsNotNone(generators)
                assert generators is not None
                first_run_generator_seeds.append(
                    tuple(int(generator.initial_seed()) for generator in generators)
                )
                return original_resample_dead_latents(
                    model,
                    inactive_latents_count,
                    generators=generators,
                )

            def _capture_second_run_generator_seeds(
                model: SAEBase,
                inactive_latents_count: torch.Tensor,
                generators: Any = None,
            ) -> torch.Tensor:
                self.assertIsNotNone(generators)
                assert generators is not None
                second_run_generator_seeds.append(
                    tuple(int(generator.initial_seed()) for generator in generators)
                )
                return original_resample_dead_latents(
                    model,
                    inactive_latents_count,
                    generators=generators,
                )

            # when
            with patch(
                "sgnsae.models.sae_base.SAEBase.resample_dead_latents",
                autospec=True,
                side_effect=_capture_first_run_generator_seeds,
            ):
                _ = run_training_for_lambda_chunk(
                    config=first_config,
                    resolved_config=first_resolved,
                    variant=first_variant,
                    seed=0,
                    chunk_subruns=first_subruns,
                    output_dir=first_run_dir,
                    overwrite=False,
                )
            with patch(
                "sgnsae.models.sae_base.SAEBase.resample_dead_latents",
                autospec=True,
                side_effect=_capture_second_run_generator_seeds,
            ):
                _ = run_training_for_lambda_chunk(
                    config=second_config,
                    resolved_config=second_resolved,
                    variant=second_variant,
                    seed=0,
                    chunk_subruns=second_subruns,
                    output_dir=second_run_dir,
                    overwrite=False,
                )

            # then
            self.assertGreaterEqual(len(first_run_generator_seeds), 1)
            self.assertGreaterEqual(len(second_run_generator_seeds), 1)
            self.assertEqual(
                len(first_run_generator_seeds[0]),
                len(first_subruns),
            )
            self.assertEqual(
                len(second_run_generator_seeds[0]),
                len(second_subruns),
            )
            self.assertNotEqual(
                first_run_generator_seeds[0][0],
                first_run_generator_seeds[0][1],
            )
            self.assertEqual(
                first_run_generator_seeds[0],
                second_run_generator_seeds[0],
            )

    def test_instance_copy_and_threshold_parameter_helpers_cover_guard_branches(
        self,
    ) -> None:
        # given
        class _NoInstanceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        class _MixedInstanceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 2
                self.instance_param = torch.nn.Parameter(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                )
                self.non_instance_param = torch.nn.Parameter(
                    torch.tensor([5.0, 6.0, 7.0])
                )

        no_instance_model = _NoInstanceModel()
        mixed_instance_model = _MixedInstanceModel()
        no_instance_before = no_instance_model.weight.detach().clone()
        non_instance_before = mixed_instance_model.non_instance_param.detach().clone()

        # when
        _copy_instance_0_parameters_to_all_instances(no_instance_model)
        _copy_instance_0_parameters_to_all_instances(mixed_instance_model)
        no_threshold_parameters = _threshold_parameters(torch.nn.Linear(2, 2))

        # then
        self.assertTrue(
            torch.equal(no_instance_model.weight.detach(), no_instance_before)
        )
        self.assertTrue(
            torch.equal(
                mixed_instance_model.instance_param[0].detach(),
                mixed_instance_model.instance_param[1].detach(),
            )
        )
        self.assertTrue(
            torch.equal(
                mixed_instance_model.non_instance_param.detach(),
                non_instance_before,
            )
        )
        self.assertEqual(no_threshold_parameters, [])

    def test_instantiate_sae_model_requires_sweep_value_for_topk_models(self) -> None:
        # given
        topk_variant = {
            "sae_class": "TopKSAE",
            "d_hidden": 8,
        }
        abstopk_variant = {
            "sae_class": "AbsTopKSAE",
            "d_hidden": 8,
        }
        hybrid_variant = {
            "sae_class": "AbsTopKGatedMagnitudeSAE",
            "d_hidden": 8,
        }
        unknown_variant = {
            "sae_class": "UnknownSAE",
            "d_hidden": 8,
        }

        # when / then
        with self.assertRaises(ValueError):
            _ = instantiate_sae_model(
                variant_config=topk_variant,
                n_instances=1,
                d_in=4,
                resample_latents_threshold=2,
                device="cpu",
                sweep_param="k",
                sweep_value=None,
            )
        with self.assertRaises(ValueError):
            _ = instantiate_sae_model(
                variant_config=abstopk_variant,
                n_instances=1,
                d_in=4,
                resample_latents_threshold=2,
                device="cpu",
                sweep_param="k",
                sweep_value=None,
            )
        with self.assertRaises(ValueError):
            _ = instantiate_sae_model(
                variant_config=hybrid_variant,
                n_instances=1,
                d_in=4,
                resample_latents_threshold=2,
                device="cpu",
                sweep_param="k",
                sweep_value=None,
            )
        with self.assertRaises(ValueError):
            _ = instantiate_sae_model(
                variant_config=unknown_variant,
                n_instances=1,
                d_in=4,
                resample_latents_threshold=2,
                device="cpu",
                sweep_param="lambda",
                sweep_value=1e-3,
            )

    def test_instantiate_sae_model_with_list_sweep_value_creates_per_instance_topk(
        self,
    ) -> None:
        for sae_class_name in ["TopKSAE", "AbsTopKSAE", "AbsTopKGatedMagnitudeSAE"]:
            with self.subTest(sae_class=sae_class_name):
                # given
                variant = {"sae_class": sae_class_name, "d_hidden": 8}

                # when
                model = instantiate_sae_model(
                    variant_config=variant,
                    n_instances=3,
                    d_in=4,
                    resample_latents_threshold=2,
                    device="cpu",
                    sweep_param="k",
                    sweep_value=[2, 4, 6],
                )

                # then
                self.assertEqual(model.topk_k_per_instance, [2, 4, 6])
                self.assertEqual(model.topk_k, 2)

    def test_instantiate_sae_model_with_scalar_sweep_value_for_topk(self) -> None:
        # given
        variant = {"sae_class": "TopKSAE", "d_hidden": 8}

        # when
        model = instantiate_sae_model(
            variant_config=variant,
            n_instances=1,
            d_in=4,
            resample_latents_threshold=2,
            device="cpu",
            sweep_param="k",
            sweep_value=4,
        )

        # then
        self.assertEqual(model.topk_k, 4)
        self.assertIsNone(model.topk_k_per_instance)

    def test_instantiate_sae_model_rejects_k_sweep_param_for_lambda_sweep_classes(
        self,
    ) -> None:
        for sae_class_name in [
            "SAE",
            "GatedSAE",
            "JumpReLUSAE",
            "SignedSoftThresholdSAE",
        ]:
            with self.subTest(sae_class=sae_class_name):
                # given
                variant = {"sae_class": sae_class_name, "d_hidden": 8}

                # when / then
                with self.assertRaises(ValueError):
                    instantiate_sae_model(
                        variant_config=variant,
                        n_instances=1,
                        d_in=4,
                        resample_latents_threshold=2,
                        device="cpu",
                        sweep_param="k",
                        sweep_value=4,
                    )

    def test_instantiate_gated_sae_threads_loss_knobs(self) -> None:
        # given
        variant = {
            "sae_class": "GatedSAE",
            "d_hidden": 8,
            "sign_aware": True,
            "symmetric_magnitude": False,
            "aux_loss_coeff": 0.7,
            "dead_zone_reg_coeff": 0.15,
            "dead_zone_reg_type": "l2",
            "reset_activation_offsets_on_resample": True,
            "single_activation_offset": False,
        }

        # when
        model = instantiate_sae_model(
            variant_config=variant,
            n_instances=1,
            d_in=4,
            resample_latents_threshold=2,
            device="cpu",
            sweep_param="lambda",
            sweep_value=1e-3,
        )

        # then
        self.assertIsInstance(model, GatedSAE)
        self.assertAlmostEqual(float(model.aux_loss_coeff), 0.7, places=12)
        self.assertAlmostEqual(float(model.dead_zone_reg_coeff), 0.15, places=12)
        self.assertEqual(str(model.dead_zone_reg_type), "l2")

    def test_distribution_summary_per_instance_covers_validation_and_empty_paths(
        self,
    ) -> None:
        # given
        scalar_values = torch.tensor(1.0)
        empty_instances = torch.empty((0, 3), dtype=torch.float32)
        empty_features = torch.empty((2, 0), dtype=torch.float32)

        # when / then
        with self.assertRaises(ValueError):
            _ = _distribution_summary_per_instance(scalar_values, "delta")
        self.assertEqual(
            _distribution_summary_per_instance(empty_instances, "delta"), []
        )
        self.assertEqual(
            _distribution_summary_per_instance(empty_features, "delta"),
            [{}, {}],
        )

    def test_per_instance_threshold_and_gradient_helpers_cover_additional_branches(
        self,
    ) -> None:
        # given
        single_offset_model = SAE(
            n_instances=2,
            d_in=3,
            d_hidden=4,
            resample_latents_threshold=2,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )

        class _GradientModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 2
                self.theta = torch.tensor([1.0, 2.0], dtype=torch.float32)
                self.jump_param = torch.nn.Parameter(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
                )

        class _SingleInstanceScalarGradientModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.jump_param = torch.nn.Parameter(torch.tensor(1.0))

        gradient_model = _GradientModel()
        single_instance_gradient_model = _SingleInstanceScalarGradientModel()
        gradient_model.jump_param.sum().backward()
        (single_instance_gradient_model.jump_param * 2.0).backward()

        # when
        threshold_stats = _collect_threshold_statistics_per_instance(
            single_offset_model
        )
        gradient_stats = _collect_threshold_gradient_norms_per_instance(gradient_model)
        single_instance_gradient_stats = _collect_threshold_gradient_norms_per_instance(
            single_instance_gradient_model
        )

        # then
        self.assertEqual(len(threshold_stats), 2)
        self.assertIn("delta_pos_mean", threshold_stats[0])
        self.assertIn("delta_neg_mean", threshold_stats[0])
        self.assertNotIn("grad_norm_theta", gradient_stats[0])
        self.assertIn("grad_norm_jump", gradient_stats[0])
        self.assertIn("grad_norm_jump", single_instance_gradient_stats[0])

    def test_polarity_usage_and_flip_helpers_cover_branches(self) -> None:
        # given
        latents_per_instance = torch.tensor(
            [
                [[1.0, -1.0], [1.0, -1.0], [-1.0, 0.0]],
                [[2.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        latents_single_instance_2d = torch.tensor(
            [[1.0, -1.0], [0.0, 0.0]], dtype=torch.float32
        )
        previous_dominance = torch.tensor([[1, -1], [1, 0]], dtype=torch.int8)

        # when
        expanded_from_2d = _latents_with_instance_dimension(latents_single_instance_2d)
        expanded_from_3d = _latents_with_instance_dimension(latents_per_instance)
        usage_stats_per_instance = _collect_polarity_usage_statistics_per_instance(
            latents_per_instance
        )
        usage_stats_single = _collect_polarity_usage_statistics(
            latents_single_instance_2d
        )
        current_dominance = _dominant_polarity_per_instance(latents_per_instance)
        flip_stats_none = _collect_polarity_flip_statistics_per_instance(
            previous_dominant_polarity=None,
            current_dominant_polarity=current_dominance,
        )
        flip_stats_shape_mismatch = _collect_polarity_flip_statistics_per_instance(
            previous_dominant_polarity=torch.tensor([[1, -1]], dtype=torch.int8),
            current_dominant_polarity=current_dominance,
        )
        flip_stats = _collect_polarity_flip_statistics_per_instance(
            previous_dominant_polarity=previous_dominance,
            current_dominant_polarity=current_dominance,
        )
        flip_stats_with_change = _collect_polarity_flip_statistics_per_instance(
            previous_dominant_polarity=torch.tensor(
                [[-1, -1], [1, 0]], dtype=torch.int8
            ),
            current_dominant_polarity=current_dominance,
        )
        flip_stats_without_comparable_latents = (
            _collect_polarity_flip_statistics_per_instance(
                previous_dominant_polarity=torch.tensor([[1, -1]], dtype=torch.int8),
                current_dominant_polarity=torch.tensor([[0, 0]], dtype=torch.int8),
            )
        )
        single_flip_stats = _collect_polarity_flip_statistics(
            previous_dominant_polarity=previous_dominance[:1],
            current_dominant_polarity=current_dominance[:1],
        )
        empty_usage_stats = _collect_polarity_usage_statistics(
            torch.zeros((0, 2, 2), dtype=torch.float32)
        )
        empty_flip_stats = _collect_polarity_flip_statistics(
            previous_dominant_polarity=torch.zeros((0, 2), dtype=torch.int8),
            current_dominant_polarity=torch.zeros((0, 2), dtype=torch.int8),
        )

        # then
        self.assertEqual(tuple(expanded_from_2d.shape), (1, 2, 2))
        self.assertEqual(tuple(expanded_from_3d.shape), (2, 3, 2))
        self.assertAlmostEqual(
            float(usage_stats_per_instance[0]["polarity_positive_fraction_nonzero"]),
            0.4,
            places=7,
        )
        self.assertAlmostEqual(
            float(usage_stats_per_instance[0]["polarity_negative_fraction_nonzero"]),
            0.6,
            places=7,
        )
        self.assertAlmostEqual(
            float(usage_stats_per_instance[0]["polarity_bidirectional_usage_fraction"]),
            0.5,
            places=8,
        )
        self.assertAlmostEqual(
            float(usage_stats_per_instance[1]["polarity_positive_fraction_nonzero"]),
            1.0,
            places=8,
        )
        self.assertAlmostEqual(
            float(usage_stats_per_instance[1]["polarity_bidirectional_usage_fraction"]),
            0.0,
            places=8,
        )
        self.assertAlmostEqual(
            float(usage_stats_single["polarity_positive_fraction_nonzero"]),
            0.5,
            places=8,
        )
        self.assertEqual(tuple(current_dominance.shape), (2, 2))
        self.assertEqual(int(current_dominance[0, 0].item()), 1)
        self.assertEqual(int(current_dominance[0, 1].item()), -1)
        self.assertEqual(int(current_dominance[1, 0].item()), 1)
        self.assertEqual(int(current_dominance[1, 1].item()), 0)
        self.assertEqual(
            float(flip_stats_none[0]["polarity_dominance_flip_rate"]),
            0.0,
        )
        self.assertEqual(
            float(flip_stats_shape_mismatch[1]["polarity_dominance_flip_indicator"]),
            0.0,
        )
        self.assertEqual(float(flip_stats[0]["polarity_dominance_flip_rate"]), 0.0)
        self.assertEqual(float(single_flip_stats["polarity_dominance_flip_rate"]), 0.0)
        self.assertEqual(
            float(flip_stats[1]["polarity_dominance_flip_indicator"]),
            0.0,
        )
        self.assertEqual(
            float(
                flip_stats_without_comparable_latents[0]["polarity_dominance_flip_rate"]
            ),
            0.0,
        )
        self.assertEqual(
            float(
                flip_stats_without_comparable_latents[0][
                    "polarity_dominance_flip_indicator"
                ]
            ),
            0.0,
        )
        self.assertAlmostEqual(
            float(flip_stats_with_change[0]["polarity_dominance_flip_rate"]),
            0.5,
            places=8,
        )
        self.assertEqual(
            float(flip_stats_with_change[0]["polarity_dominance_flip_indicator"]),
            1.0,
        )
        self.assertEqual(
            empty_usage_stats,
            {
                "polarity_positive_fraction_nonzero": 0.0,
                "polarity_negative_fraction_nonzero": 0.0,
                "polarity_bidirectional_usage_fraction": 0.0,
            },
        )
        self.assertEqual(
            empty_flip_stats,
            {
                "polarity_dominance_flip_rate": 0.0,
                "polarity_dominance_flip_indicator": 0.0,
            },
        )
        with self.assertRaises(ValueError):
            _ = _latents_with_instance_dimension(torch.zeros((1, 2, 3, 4)))

    def test_resample_generator_and_state_dict_slicing_helpers_cover_error_paths(
        self,
    ) -> None:
        # given
        state_dict = {
            "W_dec": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "bias": torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32),
        }

        # when / then
        with self.assertRaises(ValueError):
            _ = _build_resample_generators_for_subruns(
                runtime_device="cpu",
                rng_fingerprint="fingerprint",
                variant_slug="variant",
                seed=0,
                subruns=[{"sweep_value": 1e-3}],
            )
        sliced_state_dict = _slice_state_dict_to_single_instance(
            state_dict=state_dict,
            n_instances=2,
            instance_index=1,
        )
        self.assertEqual(sliced_state_dict["W_dec"].shape, (1, 2))
        self.assertEqual(sliced_state_dict["bias"].shape, (3,))
        self.assertEqual(sliced_state_dict["W_dec"].device.type, "cpu")
        self.assertEqual(sliced_state_dict["bias"].device.type, "cpu")

    def test_run_training_for_lambda_chunk_rejects_invalid_chunk_inputs(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            valid_subrun = dict(chunk_subruns[0])

            # when / then
            with self.assertRaises(ValueError):
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=[],
                    output_dir=tmpdir,
                    overwrite=False,
                )

            invalid_sweep_param_subrun = dict(valid_subrun)
            invalid_sweep_param_subrun["sweep_param"] = "k"
            with self.assertRaises(ValueError):
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=[invalid_sweep_param_subrun],
                    output_dir=tmpdir,
                    overwrite=False,
                )

            invalid_seed_subrun = dict(valid_subrun)
            invalid_seed_subrun["seed"] = 123
            with self.assertRaises(ValueError):
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=[invalid_seed_subrun],
                    output_dir=tmpdir,
                    overwrite=False,
                )

            invalid_variant_subrun = dict(valid_subrun)
            invalid_variant_subrun["variant_slug"] = "other_variant"
            with self.assertRaises(ValueError):
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=[invalid_variant_subrun],
                    output_dir=tmpdir,
                    overwrite=False,
                )

    def test_run_training_for_sweep_chunk_rejects_mixed_sweep_params(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            subrun_lambda = dict(chunk_subruns[0])
            subrun_k = dict(chunk_subruns[1])
            subrun_k["sweep_param"] = "k"

            # when / then
            with self.assertRaises(ValueError) as context:
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=[subrun_lambda, subrun_k],
                    output_dir=tmpdir,
                    overwrite=False,
                )
            self.assertIn("share the same sweep_param", str(context.exception))

    def test_run_training_for_sweep_chunk_rejects_unsupported_sweep_param(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            subrun_unknown = dict(chunk_subruns[0])
            subrun_unknown["sweep_param"] = "unknown"

            # when / then
            with self.assertRaises(ValueError) as context:
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=[subrun_unknown],
                    output_dir=tmpdir,
                    overwrite=False,
                )
            self.assertIn('"lambda", "k"', str(context.exception))

    def test_run_training_for_lambda_chunk_overwrite_cleans_stale_files_on_setup_failure(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            target_subrun = chunk_subruns[0]
            subrun_dir = os.path.join(tmpdir, target_subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            stale_files = [
                "model.pt",
                "model_config.json",
                "evaluation.json",
                "gamma_metrics.json",
                "loss_recovered.json",
                "feature_ablation.json",
                "duplication_metrics.json",
                "latent_examples.json",
            ]
            for filename in stale_files:
                with open(os.path.join(subrun_dir, filename), "w") as file:
                    file.write("stale")
            assets_dir = os.path.join(subrun_dir, "latent_examples_assets")
            os.makedirs(assets_dir, exist_ok=True)
            with open(os.path.join(assets_dir, "artifact.txt"), "w") as file:
                file.write("stale")
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
                file.write("stale")

            # when / then
            with self.assertRaises(ValueError):
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=[target_subrun],
                    output_dir=tmpdir,
                    overwrite=True,
                )
            for filename in stale_files:
                self.assertFalse(os.path.exists(os.path.join(subrun_dir, filename)))
            self.assertFalse(os.path.exists(assets_dir))
            with open(os.path.join(subrun_dir, "training_metrics.json"), "r") as file:
                metrics = json.load(file)
            self.assertEqual(str(metrics["status"]), "failed")
            self.assertEqual(str(metrics["failure"]["type"]), "exception")

    def test_run_training_for_lambda_chunk_requires_rng_fingerprint_for_resampling(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=True,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            resolved_config["rng_fingerprint"] = ""
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )

            # when / then
            with self.assertRaises(ValueError):
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=chunk_subruns,
                    output_dir=tmpdir,
                    overwrite=False,
                )

    def test_run_training_for_lambda_chunk_raises_when_dataset_yields_no_batches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            for subrun in chunk_subruns:
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "model.pt"), "w") as file:
                    file.write("stale")
                with open(
                    os.path.join(subrun_dir, "training_metrics.json"), "w"
                ) as file:
                    json.dump({"status": "completed"}, file)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.training.create_activation_dataset",
                return_value=[],
            ):
                with self.assertRaises(ValueError):
                    _ = run_training_for_lambda_chunk(
                        config=config,
                        resolved_config=resolved_config,
                        variant=variant,
                        seed=0,
                        chunk_subruns=chunk_subruns,
                        output_dir=tmpdir,
                        overwrite=False,
                    )
            for subrun in chunk_subruns:
                metrics_path = os.path.join(
                    tmpdir,
                    subrun["relative_subrun_dir"],
                    "training_metrics.json",
                )
                with open(metrics_path, "r") as file:
                    metrics = json.load(file)
                self.assertEqual(str(metrics["status"]), "failed")
                self.assertEqual(str(metrics["failure"]["type"]), "exception")

    def test_run_training_for_lambda_chunk_marks_failed_status_on_warmup_exception(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(
                output_dir=tmpdir,
                sae_class="SAE",
                sign_aware=False,
                resample_dead_latents=False,
            )
            config_payload["training"]["warmup_steps"] = 1
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            for subrun in chunk_subruns:
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                os.makedirs(subrun_dir, exist_ok=True)
                with open(os.path.join(subrun_dir, "model.pt"), "w") as file:
                    file.write("stale")
                with open(
                    os.path.join(subrun_dir, "training_metrics.json"), "w"
                ) as file:
                    json.dump({"status": "completed"}, file)
            fake_model = _FakeWarmupExceptionModel(
                n_instances=len(chunk_subruns),
                d_hidden=int(variant["d_hidden"]),
            )

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.training.instantiate_sae_model",
                return_value=fake_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.create_activation_dataset",
                return_value=[torch.zeros((2, 4), dtype=torch.float32)],
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_training_for_lambda_chunk(
                        config=config,
                        resolved_config=resolved_config,
                        variant=variant,
                        seed=0,
                        chunk_subruns=chunk_subruns,
                        output_dir=tmpdir,
                        overwrite=False,
                    )
            for subrun in chunk_subruns:
                metrics_path = os.path.join(
                    tmpdir,
                    subrun["relative_subrun_dir"],
                    "training_metrics.json",
                )
                with open(metrics_path, "r") as file:
                    metrics = json.load(file)
                self.assertEqual(str(metrics["status"]), "failed")
                self.assertEqual(str(metrics["failure"]["type"]), "exception")
                self.assertEqual(str(metrics["failure"]["message"]), "warmup exploded")

    def test_run_training_for_lambda_chunk_records_non_finite_loss_failure(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            fake_model = _FakeInfiniteLossModel(d_hidden=int(variant["d_hidden"]))

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.instantiate_sae_model",
                return_value=fake_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.create_activation_dataset",
                return_value=[torch.zeros((2, 4), dtype=torch.float32)],
            ):
                result = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=chunk_subruns,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            self.assertTrue(bool(result["terminated_early"]))
            for subrun in chunk_subruns:
                metrics_path = os.path.join(
                    tmpdir,
                    subrun["relative_subrun_dir"],
                    "training_metrics.json",
                )
                with open(metrics_path, "r") as file:
                    metrics = json.load(file)
                self.assertEqual(str(metrics["status"]), "failed")
                self.assertTrue(bool(metrics["terminated_early"]))
                self.assertEqual(str(metrics["failure"]["type"]), "non_finite_loss")

    def test_run_training_for_lambda_chunk_records_warmup_non_finite_loss_failure(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(
                output_dir=tmpdir,
                sae_class="SAE",
                sign_aware=False,
                resample_dead_latents=False,
            )
            config_payload["training"]["warmup_steps"] = 1
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            for subrun in chunk_subruns:
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                _create_downstream_eval_artifacts(subrun_dir)
            fake_model = _FakeInfiniteLossModel(d_hidden=int(variant["d_hidden"]))

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.instantiate_sae_model",
                return_value=fake_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.create_activation_dataset",
                return_value=[torch.zeros((2, 4), dtype=torch.float32)],
            ):
                result = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=chunk_subruns,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            self.assertTrue(bool(result["terminated_early"]))
            for subrun in chunk_subruns:
                metrics_path = os.path.join(
                    tmpdir,
                    subrun["relative_subrun_dir"],
                    "training_metrics.json",
                )
                with open(metrics_path, "r") as file:
                    metrics = json.load(file)
                self.assertEqual(str(metrics["status"]), "failed")
                self.assertTrue(bool(metrics["terminated_early"]))
                self.assertEqual(str(metrics["failure"]["type"]), "non_finite_loss")
                self.assertEqual(str(metrics["failure"]["phase"]), "warmup")
                for filename in _DOWNSTREAM_EVAL_FILENAMES:
                    self.assertFalse(
                        os.path.exists(
                            os.path.join(
                                tmpdir, subrun["relative_subrun_dir"], filename
                            )
                        )
                    )
                for dirname in _DOWNSTREAM_SUBRUN_DIRNAMES:
                    self.assertFalse(
                        os.path.exists(
                            os.path.join(tmpdir, subrun["relative_subrun_dir"], dirname)
                        )
                    )

    def test_run_training_for_lambda_chunk_covers_scaler_and_grad_clip_paths(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_dir = os.path.join(tmpdir, "scaler_enabled")
            grad_clip_dir = os.path.join(tmpdir, "grad_clip_disabled_scaler")
            os.makedirs(scaler_dir, exist_ok=True)
            os.makedirs(grad_clip_dir, exist_ok=True)
            _write_activation_cache(scaler_dir)
            _copy_activation_cache(scaler_dir, grad_clip_dir)

            scaler_config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=scaler_dir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            scaler_config.training.grad_clip_norm = 1.0
            scaler_resolved = resolve_benchmark_config(scaler_config)
            scaler_variant, scaler_subruns = _resolve_variant_and_chunk_subruns(
                scaler_resolved, seed=0
            )

            grad_clip_payload = _build_lambda_sweep_config(
                output_dir=grad_clip_dir,
                sae_class="SAE",
                sign_aware=False,
                resample_dead_latents=False,
            )
            grad_clip_payload["training"]["grad_clip_norm"] = 1.0
            grad_clip_config = LLMBenchmarkConfig.model_validate(grad_clip_payload)
            grad_clip_resolved = resolve_benchmark_config(grad_clip_config)
            grad_clip_variant, grad_clip_subruns = _resolve_variant_and_chunk_subruns(
                grad_clip_resolved, seed=0
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.torch.amp.GradScaler",
                _FakeEnabledGradScaler,
            ):
                scaler_result = run_training_for_lambda_chunk(
                    config=scaler_config,
                    resolved_config=scaler_resolved,
                    variant=scaler_variant,
                    seed=0,
                    chunk_subruns=scaler_subruns,
                    output_dir=scaler_dir,
                    overwrite=False,
                )
            grad_clip_result = run_training_for_lambda_chunk(
                config=grad_clip_config,
                resolved_config=grad_clip_resolved,
                variant=grad_clip_variant,
                seed=0,
                chunk_subruns=grad_clip_subruns,
                output_dir=grad_clip_dir,
                overwrite=False,
            )

            # then
            self.assertFalse(bool(scaler_result["terminated_early"]))
            self.assertFalse(bool(grad_clip_result["terminated_early"]))

    def test_run_training_for_lambda_chunk_handles_no_grad_clip_and_no_sparsity_losses(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(
                output_dir=tmpdir,
                sae_class="SAE",
                sign_aware=False,
                resample_dead_latents=False,
            )
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )
            fake_model = _FakeFiniteNoSparsityLossesModel(
                n_instances=len(chunk_subruns),
                d_hidden=int(variant["d_hidden"]),
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.instantiate_sae_model",
                return_value=fake_model,
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.torch.amp.GradScaler",
                _FakeEnabledGradScaler,
            ):
                result = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=0,
                    chunk_subruns=chunk_subruns,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            self.assertFalse(bool(result["terminated_early"]))
            first_subrun = chunk_subruns[0]
            with open(
                os.path.join(
                    tmpdir,
                    first_subrun["relative_subrun_dir"],
                    "training_metrics.json",
                ),
                "r",
            ) as file:
                metrics = json.load(file)
            history_entry = metrics["history"][-1]
            self.assertNotIn("loss_sparsity", history_entry)
            self.assertNotIn("loss_aux", history_entry)

    def test_run_training_for_lambda_chunk_raises_when_training_loop_is_skipped(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config = LLMBenchmarkConfig.model_validate(
                _build_lambda_sweep_config(
                    output_dir=tmpdir,
                    sae_class="SAE",
                    sign_aware=False,
                    resample_dead_latents=False,
                )
            )
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.training.range",
                return_value=[],
            ):
                with self.assertRaises(ValueError):
                    _ = run_training_for_lambda_chunk(
                        config=config,
                        resolved_config=resolved_config,
                        variant=variant,
                        seed=0,
                        chunk_subruns=chunk_subruns,
                        output_dir=tmpdir,
                        overwrite=False,
                    )

    def test_resolve_manifest_path_and_load_activation_manifest_validate_inputs(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            activations_dir = os.path.join(tmpdir, "activations")
            os.makedirs(activations_dir, exist_ok=True)
            invalid_manifest_path = os.path.join(
                activations_dir, "activations_manifest.json"
            )
            with open(invalid_manifest_path, "w") as file:
                json.dump([], file)

            # when / then
            with self.assertRaises(ValueError):
                _ = load_activation_manifest(invalid_manifest_path)

            os.remove(invalid_manifest_path)
            with self.assertRaises(ValueError):
                _ = resolve_manifest_path(output_dir=tmpdir)

            manifest_without_schema = {
                "activation": {"d_in": 4},
                "split": {"method": "stored_indices", "train_count": 4},
            }
            with open(invalid_manifest_path, "w") as file:
                json.dump(manifest_without_schema, file, indent=2)
            with self.assertRaises(ValueError):
                _ = load_activation_manifest(invalid_manifest_path)

            manifest_wrong_schema = {
                "schema_version": int(ACTIVATIONS_MANIFEST_SCHEMA_VERSION) + 1,
                "activation": {"d_in": 4},
                "split": {"method": "stored_indices", "train_count": 4},
            }
            with open(invalid_manifest_path, "w") as file:
                json.dump(manifest_wrong_schema, file, indent=2)
            with self.assertRaises(ValueError):
                _ = load_activation_manifest(invalid_manifest_path)

        # given
        invalid_manifests = [
            {},
            {"activation": {"d_in": 0}},
        ]

        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest):
                # when / then
                with self.assertRaises(ValueError):
                    _ = _resolve_manifest_d_in(manifest)

    def test_resolve_runtime_device_handles_auto_cuda_and_cuda_indices(self) -> None:
        # given
        with patch(
            "sgnsae.experiments.device.torch.cuda.is_available",
            return_value=False,
        ):
            # when
            auto_resolved = resolve_runtime_device("auto")

            # then
            self.assertEqual(auto_resolved, "cpu")
            with self.assertRaises(ValueError):
                _ = resolve_runtime_device("cuda")
            with self.assertRaises(ValueError):
                _ = resolve_runtime_device("cuda:0")

        # given
        with patch(
            "sgnsae.experiments.device.torch.cuda.is_available",
            return_value=True,
        ), patch("sgnsae.experiments.device.torch.cuda.device_count", return_value=2):
            # when
            auto_resolved = resolve_runtime_device("auto")
            cuda_resolved = resolve_runtime_device("cuda")
            cuda_index_resolved = resolve_runtime_device("cuda:1")

            # then
            self.assertEqual(auto_resolved, "cuda")
            self.assertEqual(cuda_resolved, "cuda")
            self.assertEqual(cuda_index_resolved, "cuda:1")

        # given
        with patch(
            "sgnsae.experiments.device.torch.cuda.is_available",
            return_value=True,
        ), patch("sgnsae.experiments.device.torch.cuda.device_count", return_value=1):
            # when & then
            with self.assertRaises(ValueError):
                _ = resolve_runtime_device("cuda:1")

    def test_should_use_mixed_precision_treats_cuda_index_as_cuda(self) -> None:
        # given / when / then
        self.assertTrue(
            _should_use_mixed_precision(
                runtime_device="cuda:1",
                mixed_precision_enabled=True,
            )
        )
        self.assertFalse(
            _should_use_mixed_precision(
                runtime_device="cuda:1",
                mixed_precision_enabled=False,
            )
        )
        self.assertFalse(
            _should_use_mixed_precision(
                runtime_device="cpu",
                mixed_precision_enabled=True,
            )
        )

    def test_resolve_split_sample_count_covers_stored_and_shard_modes(self) -> None:
        # given
        stored_manifest = {"split": {"method": "stored_indices", "train_count": 10}}
        shard_manifest = {
            "split": {"method": "shard_ranges", "train_shard_indices": [0, 1]},
            "shards": [{"num_vectors": 3}, {"num_vectors": 4}],
        }

        # when
        stored_count = _resolve_split_sample_count(
            manifest=stored_manifest,
            split="train",
            max_samples=4,
        )
        shard_count = _resolve_split_sample_count(
            manifest=shard_manifest,
            split="train",
            max_samples=None,
        )

        # then
        self.assertEqual(stored_count, 4)
        self.assertEqual(shard_count, 7)

    def test_resolve_split_sample_count_rejects_invalid_manifests(self) -> None:
        # given
        invalid_manifests = [
            {"split": None},
            {"split": {"method": "stored_indices", "train_count": True}},
            {"split": {"method": "stored_indices", "train_count": -1}},
            {"split": {"method": "shard_ranges", "train_shard_indices": [0]}},
            {
                "split": {"method": "shard_ranges", "train_shard_indices": "bad"},
                "shards": [{"num_vectors": 1}],
            },
            {
                "split": {"method": "shard_ranges", "train_shard_indices": [True]},
                "shards": [{"num_vectors": 1}],
            },
            {
                "split": {"method": "shard_ranges", "train_shard_indices": [2]},
                "shards": [{"num_vectors": 1}],
            },
            {
                "split": {"method": "shard_ranges", "train_shard_indices": [0]},
                "shards": ["bad"],
            },
            {
                "split": {"method": "shard_ranges", "train_shard_indices": [0]},
                "shards": [{"num_vectors": "bad"}],
            },
            {"split": {"method": "unknown"}},
        ]

        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest):
                # when / then
                with self.assertRaises(ValueError):
                    _ = _resolve_split_sample_count(
                        manifest=manifest,
                        split="train",
                        max_samples=None,
                    )

    def test_resolve_resample_latents_threshold_returns_steps_when_set(
        self,
    ) -> None:
        # given
        training_config = TrainingConfig(batch_size=4, resample_latents_steps=42)

        # when
        result = _resolve_resample_latents_threshold(
            training_config=training_config,
        )

        # then
        self.assertEqual(result, 42)

    def test_resolve_resample_latents_threshold_returns_fallback_when_none(
        self,
    ) -> None:
        # given
        training_config = TrainingConfig(batch_size=4, train_steps=100)

        # when
        result = _resolve_resample_latents_threshold(
            training_config=training_config,
        )

        # then
        self.assertEqual(result, 101)

    def test_checkpoint_model_steps_saves_intermediate_checkpoints(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir)
            config_payload["training"]["checkpoint_model_steps"] = 2
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            # when
            result = run_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            subrun_dir = result["subrun_dir"]
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model_2.pt")))
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model_4.pt")))
            self.assertFalse(os.path.isfile(os.path.join(subrun_dir, "model_5.pt")))

    def test_overwrite_deletes_stale_checkpoint_files(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir)
            config_payload["training"]["checkpoint_model_steps"] = 2
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]
            result = run_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )
            subrun_dir = result["subrun_dir"]
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model_2.pt")))

            # when
            config_payload["training"]["checkpoint_model_steps"] = 5
            config2 = LLMBenchmarkConfig.model_validate(config_payload)
            resolved2 = resolve_benchmark_config(config2)
            subrun2 = resolved2["subruns"][0]
            run_training_for_subrun(
                config=config2,
                resolved_config=resolved2,
                subrun=subrun2,
                output_dir=tmpdir,
                overwrite=True,
            )

            # then
            self.assertFalse(os.path.isfile(os.path.join(subrun_dir, "model_2.pt")))
            self.assertFalse(os.path.isfile(os.path.join(subrun_dir, "model_4.pt")))
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model_5.pt")))

    def test_dump_training_metrics_steps_writes_intermediate_metrics(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir)
            config_payload["training"]["dump_training_metrics_steps"] = 3
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training"
                "._dump_training_metrics_snapshot",
                wraps=_dump_training_metrics_snapshot,
            ) as mock_dump:
                result = run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            self.assertEqual(mock_dump.call_count, 1)
            subrun_dir = result["subrun_dir"]
            metrics_path = os.path.join(subrun_dir, "training_metrics.json")
            with open(metrics_path, "r") as file:
                metrics = json.load(file)
            self.assertEqual(metrics["status"], "completed")

    def test_apply_initialization_overrides_sets_threshold_parameters(self) -> None:
        # given
        single_offset_model = SAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            resample_latents_threshold=2,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )
        bilateral_offset_model = SAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            resample_latents_threshold=2,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device="cpu",
        )
        theta_model = SignedSoftThresholdSAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            resample_latents_threshold=2,
            device="cpu",
        )
        jump_model = JumpReLUSAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            resample_latents_threshold=2,
            device="cpu",
        )

        # when / then
        with self.assertRaises(ValueError):
            _apply_initialization_overrides(
                model=single_offset_model,
                variant_config={
                    "sign_aware": True,
                    "single_activation_offset": True,
                    "delta_pos_init": 0.2,
                    "delta_neg_init": 0.3,
                },
            )

        _apply_initialization_overrides(
            model=single_offset_model,
            variant_config={
                "sign_aware": True,
                "single_activation_offset": True,
                "delta_init": 0.25,
            },
        )
        _apply_initialization_overrides(
            model=single_offset_model,
            variant_config={
                "sign_aware": True,
                "single_activation_offset": True,
                "delta_pos_init": 0.15,
            },
        )
        _apply_initialization_overrides(
            model=single_offset_model,
            variant_config={
                "sign_aware": True,
                "single_activation_offset": True,
                "delta_pos_init": 0.12,
                "delta_neg_init": 0.12,
            },
        )
        _apply_initialization_overrides(
            model=single_offset_model,
            variant_config={
                "sign_aware": True,
                "single_activation_offset": True,
                "delta_neg_init": 0.18,
            },
        )
        _apply_initialization_overrides(
            model=bilateral_offset_model,
            variant_config={
                "sign_aware": True,
                "single_activation_offset": False,
                "delta_pos_init": 0.4,
                "delta_neg_init": 0.5,
            },
        )
        _apply_initialization_overrides(
            model=theta_model,
            variant_config={"theta_init": 0.35},
        )
        _apply_initialization_overrides(
            model=jump_model,
            variant_config={"theta_init": 0.15, "jump_init": 0.65},
        )
        with self.assertRaises(ValueError):
            _apply_initialization_overrides(
                model=jump_model,
                variant_config={"theta_init": 0.0},
            )
        with self.assertRaises(ValueError):
            _apply_initialization_overrides(
                model=jump_model,
                variant_config={"jump_init": 0.0},
            )

        self.assertTrue(
            torch.allclose(
                single_offset_model.act_offset.detach(),
                torch.full_like(single_offset_model.act_offset.detach(), 0.18),
            )
        )
        self.assertTrue(
            torch.allclose(
                bilateral_offset_model.act_offset_right.detach(),
                torch.full_like(
                    bilateral_offset_model.act_offset_right.detach(),
                    0.4,
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                bilateral_offset_model.act_offset_left.detach(),
                torch.full_like(
                    bilateral_offset_model.act_offset_left.detach(),
                    0.5,
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                theta_model.theta.detach(),
                torch.full_like(theta_model.theta.detach(), 0.35),
            )
        )
        self.assertGreater(
            float(torch.nn.functional.softplus(jump_model.theta_param).mean()), 0.0
        )
        self.assertGreater(
            float(torch.nn.functional.softplus(jump_model.jump_param).mean()), 0.0
        )

    def test_build_model_for_subrun_instantiates_supported_classes(self) -> None:
        # given
        manifest = {
            "activation": {"d_in": 4},
            "split": {"method": "stored_indices", "train_count": 12},
        }
        cases = [
            ("SAE", SAE, 0.001),
            ("GatedSAE", GatedSAE, 0.001),
            ("SignedSoftThresholdSAE", SignedSoftThresholdSAE, 0.001),
            ("JumpReLUSAE", JumpReLUSAE, 0.001),
            ("TopKSAE", TopKSAE, 0.0),
            ("AbsTopKSAE", AbsTopKSAE, 0.0),
            ("AbsTopKGatedMagnitudeSAE", AbsTopKGatedMagnitudeSAE, 0.0),
        ]

        for sae_class, expected_type, expected_sparsity in cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_payload = _build_config(
                    output_dir=tmpdir,
                    sae_class=sae_class,
                    sign_aware=sae_class == "GatedSAE",
                )
                config, resolved_config, subrun = _resolve_single_subrun(config_payload)

                # when
                model, sparsity_coeff, threshold, _ = build_model_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    manifest=manifest,
                    device="cpu",
                )

                # then
                self.assertIsInstance(model, expected_type)
                self.assertAlmostEqual(
                    float(sparsity_coeff), float(expected_sparsity), places=8
                )
                self.assertGreater(int(threshold), 0)

    def test_build_model_for_subrun_rejects_invalid_configurations(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config, resolved_config, subrun = _resolve_single_subrun(
                _build_config(tmpdir, "SAE")
            )
            manifest = {
                "activation": {"d_in": 4},
                "split": {"method": "stored_indices", "train_count": 8},
            }

            # when / then
            invalid_variant_subrun = dict(subrun)
            invalid_variant_subrun["variant_slug"] = "missing"
            with self.assertRaises(ValueError):
                _ = build_model_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=invalid_variant_subrun,
                    manifest=manifest,
                    device="cpu",
                )

            invalid_sweep_subrun = dict(subrun)
            invalid_sweep_subrun["sweep_param"] = "unknown"
            with self.assertRaises(ValueError):
                _ = build_model_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=invalid_sweep_subrun,
                    manifest=manifest,
                    device="cpu",
                )

            invalid_lambda_class_sweep_subrun = dict(subrun)
            invalid_lambda_class_sweep_subrun["sweep_param"] = "k"
            with self.assertRaises(ValueError):
                _ = build_model_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=invalid_lambda_class_sweep_subrun,
                    manifest=manifest,
                    device="cpu",
                )

            resolved_missing_fingerprint = dict(resolved_config)
            resolved_missing_fingerprint["rng_fingerprint"] = ""
            with self.assertRaises(ValueError):
                _ = build_model_for_subrun(
                    config=config,
                    resolved_config=resolved_missing_fingerprint,
                    subrun=subrun,
                    manifest=manifest,
                    device="cpu",
                )

            resolved_unknown = dict(resolved_config)
            resolved_unknown["variants"] = [dict(resolved_config["variants"][0])]
            resolved_unknown["variants"][0]["sae_class"] = "UnknownSAE"
            with self.assertRaises(ValueError):
                _ = build_model_for_subrun(
                    config=config,
                    resolved_config=resolved_unknown,
                    subrun=subrun,
                    manifest=manifest,
                    device="cpu",
                )

            topk_config, topk_resolved, topk_subrun = _resolve_single_subrun(
                _build_config(tmpdir, "TopKSAE")
            )
            bad_topk_subrun = dict(topk_subrun)
            bad_topk_subrun["sweep_param"] = "lambda"
            with self.assertRaises(ValueError):
                _ = build_model_for_subrun(
                    config=topk_config,
                    resolved_config=topk_resolved,
                    subrun=bad_topk_subrun,
                    manifest=manifest,
                    device="cpu",
                )

            abstopk_config, abstopk_resolved, abstopk_subrun = _resolve_single_subrun(
                _build_config(tmpdir, "AbsTopKSAE")
            )
            bad_abstopk_subrun = dict(abstopk_subrun)
            bad_abstopk_subrun["sweep_param"] = "lambda"
            with self.assertRaises(ValueError):
                _ = build_model_for_subrun(
                    config=abstopk_config,
                    resolved_config=abstopk_resolved,
                    subrun=bad_abstopk_subrun,
                    manifest=manifest,
                    device="cpu",
                )

    def test_reconstruction_and_threshold_helpers_cover_edge_cases(self) -> None:
        # given
        y_pred = torch.randn(1, 2, 3)
        single_offset_model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=4,
            resample_latents_threshold=2,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )
        bilateral_offset_model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=4,
            resample_latents_threshold=2,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device="cpu",
        )
        theta_model = SignedSoftThresholdSAE(
            n_instances=1,
            d_in=3,
            d_hidden=4,
            resample_latents_threshold=2,
            device="cpu",
        )
        jump_model = JumpReLUSAE(
            n_instances=1,
            d_in=3,
            d_hidden=4,
            resample_latents_threshold=2,
            device="cpu",
        )

        dummy_model = torch.nn.Module()
        dummy_model.act_offset = torch.nn.Parameter(
            torch.tensor([2.0], dtype=torch.float32)
        )
        dummy_model.act_offset_right = torch.tensor([1.0], dtype=torch.float32)
        dummy_model.act_offset_left = torch.nn.Parameter(
            torch.tensor([3.0], dtype=torch.float32)
        )
        dummy_model.act_offset.grad = torch.tensor([4.0], dtype=torch.float32)

        # when
        resolved = _resolve_reconstruction_output({"y_pred": y_pred})
        empty_distribution = _distribution_summary(torch.empty(0), "delta")
        single_stats = _collect_threshold_statistics(single_offset_model)
        bilateral_stats = _collect_threshold_statistics(bilateral_offset_model)
        theta_stats = _collect_threshold_statistics(theta_model)
        jump_stats = _collect_threshold_statistics(jump_model)
        gradient_norms = _collect_threshold_gradient_norms(dummy_model)

        # then
        self.assertTrue(torch.equal(resolved, y_pred))
        self.assertEqual(empty_distribution, {})
        self.assertIn("delta_pos_mean", single_stats)
        self.assertIn("delta_neg_mean", single_stats)
        self.assertIn("delta_pos_mean", bilateral_stats)
        self.assertIn("delta_neg_mean", bilateral_stats)
        self.assertIn("theta_mean", theta_stats)
        self.assertIn("theta_mean", jump_stats)
        self.assertIn("jump_mean", jump_stats)
        self.assertIn("grad_norm_delta", gradient_norms)
        self.assertNotIn("grad_norm_delta_pos", gradient_norms)
        self.assertNotIn("grad_norm_delta_neg", gradient_norms)
        with self.assertRaises(ValueError):
            _ = _resolve_reconstruction_output({})

    def test_delete_file_if_exists_deletes_existing_file(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "temporary.txt")
            with open(file_path, "w") as file:
                file.write("temporary")

            # when
            _delete_file_if_exists(file_path)

            # then
            self.assertFalse(os.path.isfile(file_path))

    def test_run_training_for_subrun_overwrite_clears_eval_artifacts_and_logs_aux_loss(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(
                output_dir=tmpdir,
                sae_class="GatedSAE",
                sign_aware=True,
            )
            config_payload["training"]["grad_clip_norm"] = 1.0
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            for filename in (
                "model.pt",
                "model_config.json",
                "training_metrics.json",
                "evaluation.json",
                "gamma_metrics.json",
                "loss_recovered.json",
                "feature_ablation.json",
                "duplication_metrics.json",
            ):
                with open(os.path.join(subrun_dir, filename), "w") as file:
                    file.write("stale")

            # when
            result = run_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=True,
            )

            # then
            self.assertTrue(os.path.isfile(result["artifact_paths"]["model"]))
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "evaluation.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "gamma_metrics.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "loss_recovered.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "feature_ablation.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "duplication_metrics.json"))
            )
            with open(result["artifact_paths"]["training_metrics"], "r") as file:
                metrics = json.load(file)
            self.assertEqual(str(metrics["status"]), "completed")
            self.assertNotIn("terminated_early", metrics)
            history_with_aux_loss = [
                entry for entry in metrics["history"] if "loss_aux" in entry
            ]
            self.assertTrue(bool(history_with_aux_loss))

    def test_run_training_for_subrun_always_clears_eval_artifacts(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(output_dir=tmpdir, sae_class="SAE")
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _create_downstream_eval_artifacts(subrun_dir)

            # when
            _ = run_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "evaluation.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "gamma_metrics.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "loss_recovered.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "feature_ablation.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "duplication_metrics.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "latent_examples.json"))
            )
            self.assertFalse(
                os.path.isdir(os.path.join(subrun_dir, "latent_examples_assets"))
            )

    def test_run_training_for_subrun_preserves_eval_artifacts_when_setup_fails(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _build_config(output_dir=tmpdir, sae_class="SAE")
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _create_downstream_eval_artifacts(subrun_dir)

            # when / then
            with self.assertRaises(ValueError):
                _ = run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "evaluation.json")))
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "gamma_metrics.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "loss_recovered.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "feature_ablation.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "duplication_metrics.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(subrun_dir, "latent_examples.json"))
            )
            self.assertTrue(
                os.path.isdir(os.path.join(subrun_dir, "latent_examples_assets"))
            )
            with open(os.path.join(subrun_dir, "training_metrics.json"), "r") as file:
                metrics = json.load(file)
            self.assertEqual(str(metrics["status"]), "failed")
            self.assertEqual(str(metrics["failure"]["type"]), "exception")

    def test_run_training_for_subrun_overwrite_deletes_eval_artifacts_on_setup_failure(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config_payload = _build_config(output_dir=tmpdir, sae_class="SAE")
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            subrun = resolved_config["subruns"][0]

            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _create_downstream_eval_artifacts(subrun_dir)

            # when / then
            with self.assertRaises(ValueError):
                _ = run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=True,
                )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "evaluation.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "gamma_metrics.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "loss_recovered.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "feature_ablation.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "duplication_metrics.json"))
            )
            self.assertFalse(
                os.path.isfile(os.path.join(subrun_dir, "latent_examples.json"))
            )
            self.assertFalse(
                os.path.isdir(os.path.join(subrun_dir, "latent_examples_assets"))
            )
            with open(os.path.join(subrun_dir, "training_metrics.json"), "r") as file:
                metrics = json.load(file)
            self.assertEqual(str(metrics["status"]), "failed")
            self.assertEqual(str(metrics["failure"]["type"]), "exception")

    def test_run_training_for_subrun_raises_when_dataset_yields_no_batches(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir, sae_class="SAE")
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)
            stub_model = torch.nn.Linear(4, 4)
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "model.pt"), "w") as file:
                file.write("stale")
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
                json.dump({"status": "completed"}, file)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.training.build_model_for_subrun",
                return_value=(stub_model, 0.0, 2, {"d_hidden": 4}),
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.create_activation_dataset",
                return_value=[],
            ):
                with self.assertRaises(ValueError):
                    _ = run_training_for_subrun(
                        config=config,
                        resolved_config=resolved_config,
                        subrun=subrun,
                        output_dir=tmpdir,
                        overwrite=False,
                    )
            with open(os.path.join(subrun_dir, "training_metrics.json"), "r") as file:
                metrics = json.load(file)
            self.assertEqual(str(metrics["status"]), "failed")
            self.assertEqual(str(metrics["failure"]["type"]), "exception")

    def test_run_training_for_subrun_marks_failed_status_on_warmup_exception(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir, sae_class="SAE")
            config_payload["training"]["warmup_steps"] = 1
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            os.makedirs(subrun_dir, exist_ok=True)
            with open(os.path.join(subrun_dir, "model.pt"), "w") as file:
                file.write("stale")
            with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as file:
                json.dump({"status": "completed"}, file)
            fake_model = _FakeWarmupExceptionModel(n_instances=1, d_hidden=4)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.training.build_model_for_subrun",
                return_value=(fake_model, 0.0, 2, {"d_hidden": 4}),
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.create_activation_dataset",
                return_value=[torch.zeros((2, 4), dtype=torch.float32)],
            ):
                with self.assertRaises(RuntimeError):
                    _ = run_training_for_subrun(
                        config=config,
                        resolved_config=resolved_config,
                        subrun=subrun,
                        output_dir=tmpdir,
                        overwrite=False,
                    )
            with open(os.path.join(subrun_dir, "training_metrics.json"), "r") as file:
                metrics = json.load(file)
            self.assertEqual(str(metrics["status"]), "failed")
            self.assertEqual(str(metrics["failure"]["type"]), "exception")
            self.assertEqual(str(metrics["failure"]["message"]), "warmup exploded")

    def test_run_training_for_subrun_records_non_finite_loss_failure(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir, sae_class="SAE")
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)
            fake_model = _FakeInfiniteLossModel(d_hidden=4)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.build_model_for_subrun",
                return_value=(fake_model, 0.0, 2, {"d_hidden": 4}),
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.create_activation_dataset",
                return_value=[torch.zeros((2, 4), dtype=torch.float32)],
            ):
                result = run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            self.assertTrue(bool(result["terminated_early"]))
            with open(result["artifact_paths"]["training_metrics"], "r") as file:
                metrics = json.load(file)
            self.assertEqual(str(metrics["status"]), "failed")
            self.assertTrue(bool(metrics["terminated_early"]))
            self.assertEqual(str(metrics["failure"]["type"]), "non_finite_loss")

    def test_run_training_for_subrun_records_warmup_non_finite_loss_failure(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir, sae_class="SAE")
            config_payload["training"]["warmup_steps"] = 1
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)
            subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
            _create_downstream_eval_artifacts(subrun_dir)
            fake_model = _FakeInfiniteLossModel(d_hidden=4)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.build_model_for_subrun",
                return_value=(fake_model, 0.0, 2, {"d_hidden": 4}),
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.create_activation_dataset",
                return_value=[torch.zeros((2, 4), dtype=torch.float32)],
            ):
                result = run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            self.assertTrue(bool(result["terminated_early"]))
            with open(result["artifact_paths"]["training_metrics"], "r") as file:
                metrics = json.load(file)
            self.assertEqual(str(metrics["status"]), "failed")
            self.assertTrue(bool(metrics["terminated_early"]))
            self.assertEqual(str(metrics["failure"]["type"]), "non_finite_loss")
            self.assertEqual(str(metrics["failure"]["phase"]), "warmup")
            for filename in _DOWNSTREAM_EVAL_FILENAMES:
                self.assertFalse(os.path.exists(os.path.join(subrun_dir, filename)))
            for dirname in _DOWNSTREAM_SUBRUN_DIRNAMES:
                self.assertFalse(os.path.exists(os.path.join(subrun_dir, dirname)))

    def test_run_training_for_subrun_uses_enabled_grad_scaler_branch(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir, sae_class="JumpReLUSAE")
            config_payload["training"]["grad_clip_norm"] = 1.0
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.torch.amp.GradScaler",
                _FakeEnabledGradScaler,
            ):
                result = run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            with open(result["artifact_paths"]["training_metrics"], "r") as file:
                metrics = json.load(file)
            self.assertGreaterEqual(len(metrics["history"]), 1)
            self.assertIn("step", metrics["history"][-1])

    def test_run_training_for_subrun_handles_no_grad_clip_and_no_sparsity_losses(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir, sae_class="SAE")
            config_payload["training"]["resample_dead_latents"] = False
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)
            fake_model = _FakeFiniteNoSparsityLossesModel(
                n_instances=1,
                d_hidden=4,
            )

            # when
            with patch(
                "sgnsae.experiments.llm_benchmark.training.build_model_for_subrun",
                return_value=(fake_model, 0.0, 2, {"d_hidden": 4}),
            ), patch(
                "sgnsae.experiments.llm_benchmark.training.torch.amp.GradScaler",
                _FakeEnabledGradScaler,
            ):
                result = run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=False,
                )

            # then
            with open(result["artifact_paths"]["training_metrics"], "r") as file:
                metrics = json.load(file)
            history_entry = metrics["history"][-1]
            self.assertNotIn("loss_sparsity", history_entry)
            self.assertNotIn("loss_aux", history_entry)

    def test_run_training_for_subrun_raises_when_training_loop_is_skipped(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir, sae_class="JumpReLUSAE")
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)

            # when / then
            with patch(
                "sgnsae.experiments.llm_benchmark.training.range",
                return_value=[],
            ):
                with self.assertRaises(ValueError):
                    _ = run_training_for_subrun(
                        config=config,
                        resolved_config=resolved_config,
                        subrun=subrun,
                        output_dir=tmpdir,
                        overwrite=False,
                    )

    def _run_k_sweep_integration(self, sae_class: str, topk_k_list: List[int]) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # given
            _write_activation_cache(tmpdir)
            config_payload = _build_k_sweep_config(
                output_dir=tmpdir,
                sae_class=sae_class,
                topk_k_list=topk_k_list,
            )
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved_config = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved_config, seed=0
            )

            # when
            result = run_training_for_lambda_chunk(
                config=config,
                resolved_config=resolved_config,
                variant=variant,
                seed=0,
                chunk_subruns=chunk_subruns,
                output_dir=tmpdir,
                overwrite=False,
            )

            # then
            self.assertEqual(len(result["subrun_dirs"]), len(topk_k_list))
            mean_l0_values = []
            model_cls = {
                "TopKSAE": TopKSAE,
                "AbsTopKSAE": AbsTopKSAE,
                "AbsTopKGatedMagnitudeSAE": AbsTopKGatedMagnitudeSAE,
            }[sae_class]
            for i, subrun in enumerate(chunk_subruns):
                subrun_dir = os.path.join(tmpdir, subrun["relative_subrun_dir"])
                self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model.pt")))
                self.assertTrue(
                    os.path.isfile(os.path.join(subrun_dir, "model_config.json"))
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(subrun_dir, "training_metrics.json"))
                )

                with open(
                    os.path.join(subrun_dir, "training_metrics.json"), "r"
                ) as file:
                    metrics = json.load(file)
                self.assertEqual(str(metrics["status"]), "completed")
                self.assertGreaterEqual(len(metrics["history"]), 1)
                mean_l0_values.append(float(metrics["history"][-1]["mean_l0"]))

                loaded_model = model_cls(
                    n_instances=1,
                    d_in=4,
                    d_hidden=6,
                    topk_k=topk_k_list[i],
                    resample_latents_threshold=1_000_000,
                    device="cpu",
                )
                payload = torch.load(
                    os.path.join(subrun_dir, "model.pt"), map_location="cpu"
                )
                loaded_model.load_state_dict(payload["state_dict"])
                output = loaded_model(torch.randn(2, 4))
                self.assertEqual(output["latents"].shape, (1, 2, 6))

            self.assertNotAlmostEqual(mean_l0_values[0], mean_l0_values[1], places=1)

    def test_run_training_for_sweep_chunk_k_sweep_topk_sae(self) -> None:
        self._run_k_sweep_integration("TopKSAE", [2, 4])

    def test_run_training_for_sweep_chunk_k_sweep_abstopk_sae(self) -> None:
        self._run_k_sweep_integration("AbsTopKSAE", [2, 4])

    def test_run_training_for_sweep_chunk_k_sweep_abstopk_gated_magnitude_sae(
        self,
    ) -> None:
        self._run_k_sweep_integration("AbsTopKGatedMagnitudeSAE", [2, 4])

    def test_run_training_for_sweep_chunk_k_sweep_chunking_invariance(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            full_dir = os.path.join(tmpdir, "full")
            split_dir = os.path.join(tmpdir, "split")
            os.makedirs(full_dir, exist_ok=True)
            os.makedirs(split_dir, exist_ok=True)
            _write_activation_cache(full_dir)
            _copy_activation_cache(full_dir, split_dir)

            full_config = LLMBenchmarkConfig.model_validate(
                _build_k_sweep_config(output_dir=full_dir)
            )
            full_resolved = resolve_benchmark_config(full_config)
            full_variant, full_subruns = _resolve_variant_and_chunk_subruns(
                full_resolved, seed=0
            )

            split_config = LLMBenchmarkConfig.model_validate(
                _build_k_sweep_config(output_dir=split_dir)
            )
            split_resolved = resolve_benchmark_config(split_config)
            split_variant, split_subruns = _resolve_variant_and_chunk_subruns(
                split_resolved, seed=0
            )

            # when
            _ = run_training_for_lambda_chunk(
                config=full_config,
                resolved_config=full_resolved,
                variant=full_variant,
                seed=0,
                chunk_subruns=full_subruns,
                output_dir=full_dir,
                overwrite=False,
            )
            for subrun in split_subruns:
                _ = run_training_for_lambda_chunk(
                    config=split_config,
                    resolved_config=split_resolved,
                    variant=split_variant,
                    seed=0,
                    chunk_subruns=[subrun],
                    output_dir=split_dir,
                    overwrite=False,
                )

            # then
            for subrun in full_subruns:
                full_sd = torch.load(
                    os.path.join(full_dir, subrun["relative_subrun_dir"], "model.pt"),
                    map_location="cpu",
                )["state_dict"]
                split_sd = torch.load(
                    os.path.join(split_dir, subrun["relative_subrun_dir"], "model.pt"),
                    map_location="cpu",
                )["state_dict"]
                self.assertEqual(set(full_sd.keys()), set(split_sd.keys()))
                for param_name in full_sd:
                    torch.testing.assert_close(
                        full_sd[param_name],
                        split_sd[param_name],
                        rtol=0.0,
                        atol=1e-8,
                    )


class TestLoadContinuationState(unittest.TestCase):
    def _write_checkpoint(
        self,
        subrun_dir: str,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        os.makedirs(subrun_dir, exist_ok=True)
        torch.save(
            {"state_dict": state_dict},
            os.path.join(subrun_dir, "model.pt"),
        )

    def _write_metrics(
        self,
        subrun_dir: str,
        payload: Dict[str, Any],
    ) -> None:
        os.makedirs(subrun_dir, exist_ok=True)
        with open(os.path.join(subrun_dir, "training_metrics.json"), "w") as f:
            json.dump(payload, f)

    def test_returns_state_dict_history_and_thresholds_unfrozen_at_step(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "subrun")
            state_dict = {"W_enc": torch.randn(2, 4)}
            self._write_checkpoint(subrun_dir, state_dict)
            self._write_metrics(
                subrun_dir,
                {
                    "status": "completed",
                    "history": [{"step": 50, "loss": 0.5}],
                    "thresholds_unfrozen_at_step": 201,
                },
            )

            # when
            sd, history, tuas = _load_continuation_state(subrun_dir)

            # then
            torch.testing.assert_close(sd["W_enc"], state_dict["W_enc"])
            self.assertEqual(history, [{"step": 50, "loss": 0.5}])
            self.assertEqual(tuas, 201)

    def test_returns_none_thresholds_unfrozen_at_step_when_absent(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "subrun")
            self._write_checkpoint(subrun_dir, {"W": torch.zeros(1)})
            self._write_metrics(
                subrun_dir,
                {
                    "status": "completed",
                    "history": [],
                },
            )

            # when
            _, _, tuas = _load_continuation_state(subrun_dir)

            # then
            self.assertIsNone(tuas)

    def test_raises_when_model_pt_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "empty")
            os.makedirs(subrun_dir, exist_ok=True)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _load_continuation_state(subrun_dir)
            self.assertIn("model.pt not found", str(ctx.exception))

    def test_raises_when_checkpoint_has_invalid_format(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "bad")
            os.makedirs(subrun_dir, exist_ok=True)
            torch.save(
                {"wrong_key": torch.zeros(1)},
                os.path.join(subrun_dir, "model.pt"),
            )

            # when / then
            with self.assertRaises(ValueError) as ctx:
                _load_continuation_state(subrun_dir)
            self.assertIn("state_dict", str(ctx.exception))

    def test_returns_empty_history_when_metrics_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "subrun")
            self._write_checkpoint(subrun_dir, {"W": torch.zeros(1)})

            # when
            _, history, tuas = _load_continuation_state(subrun_dir)

            # then
            self.assertEqual(history, [])
            self.assertIsNone(tuas)

    def test_returns_empty_history_when_status_running(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "subrun")
            self._write_checkpoint(subrun_dir, {"W": torch.zeros(1)})
            self._write_metrics(
                subrun_dir,
                {
                    "status": "running",
                    "history": [{"step": 1}],
                    "thresholds_unfrozen_at_step": 5,
                },
            )

            # when
            _, history, tuas = _load_continuation_state(subrun_dir)

            # then
            self.assertEqual(history, [])
            self.assertIsNone(tuas)

    def test_returns_none_thresholds_unfrozen_at_step_when_boolean(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            subrun_dir = os.path.join(tmpdir, "subrun")
            self._write_checkpoint(subrun_dir, {"W": torch.zeros(1)})
            self._write_metrics(
                subrun_dir,
                {
                    "status": "completed",
                    "history": [],
                    "thresholds_unfrozen_at_step": True,
                },
            )

            # when
            _, _, tuas = _load_continuation_state(subrun_dir)

            # then
            self.assertIsNone(tuas)


class TestReassembleStateDictFromInstances(unittest.TestCase):
    def test_concatenates_instance_dimensioned_tensors(self) -> None:
        # given
        sd_0 = {"W": torch.tensor([[1.0, 2.0]]), "b": torch.tensor([10.0])}
        sd_1 = {"W": torch.tensor([[3.0, 4.0]]), "b": torch.tensor([10.0])}

        # when
        result = _reassemble_state_dict_from_instances([sd_0, sd_1], n_instances=2)

        # then
        self.assertEqual(result["W"].shape, (2, 2))
        torch.testing.assert_close(
            result["W"],
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

    def test_non_instance_tensors_taken_from_first(self) -> None:
        # given
        shared = torch.tensor([5.0, 6.0])
        sd_0 = {"W": torch.tensor([[1.0]]), "bias": shared.clone()}
        sd_1 = {"W": torch.tensor([[2.0]]), "bias": shared.clone()}

        # when
        result = _reassemble_state_dict_from_instances([sd_0, sd_1], n_instances=2)

        # then
        torch.testing.assert_close(result["bias"], shared)

    def test_raises_when_non_instance_tensors_differ(self) -> None:
        # given
        sd_0 = {"W": torch.tensor([[1.0]]), "bias": torch.tensor([5.0, 6.0])}
        sd_1 = {"W": torch.tensor([[2.0]]), "bias": torch.tensor([7.0, 8.0])}

        # when / then
        with self.assertRaises(ValueError) as ctx:
            _reassemble_state_dict_from_instances([sd_0, sd_1], n_instances=2)
        self.assertIn("bias", str(ctx.exception))

    def test_round_trip_with_slice(self) -> None:
        # given
        n_instances = 3
        d_hidden = 4
        d_in = 8
        original_sd = {
            "W_enc": torch.randn(n_instances, d_hidden, d_in),
            "b_dec": torch.randn(n_instances, d_in),
        }
        sliced = [
            _slice_state_dict_to_single_instance(original_sd, n_instances, i)
            for i in range(n_instances)
        ]

        # when
        reassembled = _reassemble_state_dict_from_instances(sliced, n_instances)

        # then
        for name in original_sd:
            torch.testing.assert_close(reassembled[name], original_sd[name])

    def test_round_trip_single_instance(self) -> None:
        # given
        original_sd = {
            "W_enc": torch.randn(1, 4, 8),
            "b_dec": torch.randn(1, 8),
        }
        sliced = [_slice_state_dict_to_single_instance(original_sd, 1, 0)]

        # when
        reassembled = _reassemble_state_dict_from_instances(sliced, n_instances=1)

        # then
        for name in original_sd:
            torch.testing.assert_close(reassembled[name], original_sd[name])

    def test_raises_on_empty_list(self) -> None:
        # when / then
        with self.assertRaises(ValueError) as ctx:
            _reassemble_state_dict_from_instances([], n_instances=1)
        self.assertIn("at least one", str(ctx.exception))

    def test_raises_on_length_mismatch(self) -> None:
        # given
        sd = {"W": torch.tensor([[1.0]])}

        # when / then
        with self.assertRaises(ValueError) as ctx:
            _reassemble_state_dict_from_instances([sd], n_instances=2)
        self.assertIn("Expected 2", str(ctx.exception))

    def test_shape_one_scalar_treated_as_instance_dimensioned(self) -> None:
        # given — a parameter of shape (1,) for a non-instance reason
        sd_0 = {"scalar_bias": torch.tensor([5.0])}
        sd_1 = {"scalar_bias": torch.tensor([7.0])}

        # when
        result = _reassemble_state_dict_from_instances([sd_0, sd_1], n_instances=2)

        # then — heuristic treats shape[0]==1 as instance-dimensioned
        self.assertEqual(result["scalar_bias"].shape, (2,))
        torch.testing.assert_close(
            result["scalar_bias"],
            torch.tensor([5.0, 7.0]),
        )

    def test_raises_on_key_mismatch(self) -> None:
        # given
        sd_0 = {"W": torch.tensor([[1.0]]), "b": torch.tensor([2.0])}
        sd_1 = {"W": torch.tensor([[3.0]])}

        # when / then
        with self.assertRaises(ValueError) as ctx:
            _reassemble_state_dict_from_instances([sd_0, sd_1], n_instances=2)
        self.assertIn("key mismatch", str(ctx.exception))


class TestRunTrainingForSubrunContinuation(unittest.TestCase):
    def _run_fresh(
        self,
        tmpdir: str,
        train_steps: int = 5,
        threshold_warmup_steps: int = 1,
        warmup_steps: int = 0,
        log_every_n_steps: int = 2,
        checkpoint_model_steps: Optional[int] = None,
        resample_latents_steps: Optional[int] = 4,
    ) -> Dict[str, Any]:
        _write_activation_cache(tmpdir)
        config_payload = _build_config(tmpdir)
        config_payload["training"]["train_steps"] = train_steps
        config_payload["training"]["threshold_warmup_steps"] = threshold_warmup_steps
        config_payload["training"]["warmup_steps"] = warmup_steps
        config_payload["training"]["log_every_n_steps"] = log_every_n_steps
        if checkpoint_model_steps is not None:
            config_payload["training"][
                "checkpoint_model_steps"
            ] = checkpoint_model_steps
        if resample_latents_steps is not None:
            config_payload["training"][
                "resample_latents_steps"
            ] = resample_latents_steps
        else:
            config_payload["training"].pop("resample_latents_steps", None)
        config, resolved_config, subrun = _resolve_single_subrun(config_payload)
        result = run_training_for_subrun(
            config=config,
            resolved_config=resolved_config,
            subrun=subrun,
            output_dir=tmpdir,
            overwrite=False,
        )
        return result

    def _run_continuation(
        self,
        tmpdir: str,
        continuation_from_steps: int,
        train_steps: int = 10,
        threshold_warmup_steps: int = 1,
        warmup_steps: int = 0,
        log_every_n_steps: int = 2,
        checkpoint_model_steps: Optional[int] = None,
        resample_latents_steps: Optional[int] = 4,
    ) -> Dict[str, Any]:
        config_payload = _build_config(tmpdir)
        config_payload["training"]["train_steps"] = train_steps
        config_payload["training"]["threshold_warmup_steps"] = threshold_warmup_steps
        config_payload["training"]["warmup_steps"] = warmup_steps
        config_payload["training"]["log_every_n_steps"] = log_every_n_steps
        if checkpoint_model_steps is not None:
            config_payload["training"][
                "checkpoint_model_steps"
            ] = checkpoint_model_steps
        if resample_latents_steps is not None:
            config_payload["training"][
                "resample_latents_steps"
            ] = resample_latents_steps
        else:
            config_payload["training"].pop("resample_latents_steps", None)
        config, resolved_config, subrun = _resolve_single_subrun(config_payload)
        result = run_training_for_subrun(
            config=config,
            resolved_config=resolved_config,
            subrun=subrun,
            output_dir=tmpdir,
            overwrite=False,
            continuation_from_steps=continuation_from_steps,
        )
        return result

    def _load_metrics(self, subrun_dir: str) -> Dict[str, Any]:
        path = os.path.join(subrun_dir, "training_metrics.json")
        with open(path, "r") as f:
            return json.load(f)

    def test_continuation_produces_merged_history_and_boundary(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(tmpdir, train_steps=5)
            subrun_dir = result_1["subrun_dir"]
            metrics_1 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_1["status"], "completed")
            self.assertEqual(metrics_1["train_steps"], 5)
            original_history_len = len(metrics_1["history"])
            self.assertGreater(original_history_len, 0)

            # when
            with self.assertWarns(UserWarning) as ctx:
                result_2 = self._run_continuation(
                    tmpdir,
                    continuation_from_steps=5,
                    train_steps=10,
                )
            self.assertIn("warmup_steps=0", str(ctx.warning))

            # then
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_2["status"], "completed")
            self.assertEqual(metrics_2["train_steps"], 10)
            self.assertEqual(metrics_2["continuation_boundary_step"], 6)
            self.assertGreater(len(metrics_2["history"]), original_history_len)
            self.assertEqual(metrics_2["history"][-1]["step"], 10)
            self.assertTrue(os.path.isfile(os.path.join(subrun_dir, "model.pt")))
            model_config_path = os.path.join(subrun_dir, "model_config.json")
            with open(model_config_path, "r") as f:
                model_config = json.load(f)
            self.assertNotIn("continuation_from_steps", model_config)

    def test_threshold_warmup_completes_during_continuation(
        self,
    ) -> None:
        # given — warmup=8, train=5 → warmup does NOT complete
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(
                tmpdir,
                train_steps=5,
                threshold_warmup_steps=8,
            )
            subrun_dir = result_1["subrun_dir"]
            metrics_1 = self._load_metrics(subrun_dir)
            self.assertNotIn("thresholds_unfrozen_at_step", metrics_1)

            # when — continue to 12, warmup completes at step 9
            result_2 = self._run_continuation(
                tmpdir,
                continuation_from_steps=5,
                train_steps=12,
                threshold_warmup_steps=8,
            )

            # then
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_2["status"], "completed")
            self.assertEqual(metrics_2["thresholds_unfrozen_at_step"], 9)

    def test_threshold_warmup_preserved_from_original_run(
        self,
    ) -> None:
        # given — warmup=2, train=5 → warmup completes at step 3
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(
                tmpdir,
                train_steps=5,
                threshold_warmup_steps=2,
            )
            subrun_dir = result_1["subrun_dir"]
            metrics_1 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_1["thresholds_unfrozen_at_step"], 3)

            # when — continue to 10
            result_2 = self._run_continuation(
                tmpdir,
                continuation_from_steps=5,
                train_steps=10,
                threshold_warmup_steps=2,
            )

            # then — original thresholds_unfrozen_at_step preserved
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_2["thresholds_unfrozen_at_step"], 3)

    def test_downstream_artifacts_invalidated_during_continuation(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(tmpdir, train_steps=5)
            subrun_dir = result_1["subrun_dir"]
            _create_downstream_eval_artifacts(subrun_dir)
            for filename in _DOWNSTREAM_EVAL_FILENAMES:
                self.assertTrue(os.path.isfile(os.path.join(subrun_dir, filename)))

            # when
            self._run_continuation(
                tmpdir,
                continuation_from_steps=5,
                train_steps=10,
            )

            # then
            for filename in _DOWNSTREAM_EVAL_FILENAMES:
                self.assertFalse(
                    os.path.isfile(os.path.join(subrun_dir, filename)),
                    f"{filename} should be deleted",
                )

    def test_continuation_with_warmup_steps(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(tmpdir, train_steps=5, warmup_steps=2)
            subrun_dir = result_1["subrun_dir"]
            metrics_1 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_1["status"], "completed")

            # when
            result_2 = self._run_continuation(
                tmpdir,
                continuation_from_steps=5,
                train_steps=10,
                warmup_steps=2,
            )

            # then
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_2["status"], "completed")
            self.assertEqual(metrics_2["train_steps"], 10)
            self.assertGreater(len(metrics_2["history"]), 0)

    def test_continuation_with_overwrite_raises_value_error(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir)
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                run_training_for_subrun(
                    config=config,
                    resolved_config=resolved_config,
                    subrun=subrun,
                    output_dir=tmpdir,
                    overwrite=True,
                    continuation_from_steps=5,
                )
            self.assertIn("mutually exclusive", str(ctx.exception))

    def test_failure_during_continuation_preserves_history(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(tmpdir, train_steps=5)
            subrun_dir = result_1["subrun_dir"]
            metrics_1 = self._load_metrics(subrun_dir)
            original_history_len = len(metrics_1["history"])

            # when — patch model forward to raise after 2 steps
            call_count = {"n": 0}
            original_forward = JumpReLUSAE.forward

            # Threshold > warmup forward calls so the
            # failure fires during the continuation loop.
            def _failing_forward(self_model, x):
                call_count["n"] += 1
                if call_count["n"] > 4:
                    raise RuntimeError("Injected failure")
                return original_forward(self_model, x)

            with patch.object(JumpReLUSAE, "forward", _failing_forward):
                with self.assertRaises(RuntimeError):
                    self._run_continuation(
                        tmpdir,
                        continuation_from_steps=5,
                        train_steps=10,
                    )

            # then
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_2["status"], "failed")
            self.assertGreaterEqual(len(metrics_2["history"]), original_history_len)
            self.assertIn("continuation_boundary_step", metrics_2)

    def test_intermediate_checkpoints_preserved_after_continuation(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(
                tmpdir,
                train_steps=5,
                checkpoint_model_steps=3,
            )
            subrun_dir = result_1["subrun_dir"]
            checkpoint_path = os.path.join(subrun_dir, "model_3.pt")
            self.assertTrue(os.path.isfile(checkpoint_path))

            # when
            self._run_continuation(
                tmpdir,
                continuation_from_steps=5,
                train_steps=10,
                checkpoint_model_steps=3,
            )

            # then — original checkpoint still exists
            self.assertTrue(os.path.isfile(checkpoint_path))

    def test_resample_latents_threshold_recomputed_after_continuation(
        self,
    ) -> None:
        # given — resample_latents_steps=None → threshold = train_steps + 1
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(
                tmpdir,
                train_steps=5,
                resample_latents_steps=None,
            )
            subrun_dir = result_1["subrun_dir"]
            metrics_1 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_1["resample_latents_threshold"], 6)

            # when
            self._run_continuation(
                tmpdir,
                continuation_from_steps=5,
                train_steps=10,
                resample_latents_steps=None,
            )

            # then — threshold recomputed from new train_steps
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_2["resample_latents_threshold"], 11)

    def test_polarity_flip_rate_zero_at_continuation_boundary(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(
                tmpdir,
                train_steps=5,
                log_every_n_steps=1,
            )
            subrun_dir = result_1["subrun_dir"]

            # when
            self._run_continuation(
                tmpdir,
                continuation_from_steps=5,
                train_steps=10,
                log_every_n_steps=1,
            )

            # then
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertIn("continuation_boundary_step", metrics_2)
            boundary = metrics_2["continuation_boundary_step"]
            self.assertEqual(boundary, 6)
            boundary_entry = next(
                e for e in metrics_2["history"] if e["step"] == boundary
            )
            self.assertEqual(
                float(boundary_entry["polarity_dominance_flip_rate"]),
                0.0,
            )

    def test_periodic_snapshot_includes_continuation_boundary_step(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_config(tmpdir)
            config_payload["training"]["train_steps"] = 5
            config, resolved_config, subrun = _resolve_single_subrun(config_payload)
            result_1 = run_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=tmpdir,
                overwrite=False,
            )
            subrun_dir = result_1["subrun_dir"]

            # when — continue with dump_training_metrics_steps=3
            config_payload["training"]["train_steps"] = 10
            config_payload["training"]["dump_training_metrics_steps"] = 3
            config2, resolved2, subrun2 = _resolve_single_subrun(config_payload)
            run_training_for_subrun(
                config=config2,
                resolved_config=resolved2,
                subrun=subrun2,
                output_dir=tmpdir,
                overwrite=False,
                continuation_from_steps=5,
            )

            # then — final metrics has continuation_boundary_step
            metrics = self._load_metrics(subrun_dir)
            self.assertEqual(metrics["status"], "completed")
            self.assertEqual(metrics["continuation_boundary_step"], 6)

    def test_chained_continuation_records_most_recent_boundary(
        self,
    ) -> None:
        # given — train 3, continue to 6, continue to 10
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(tmpdir, train_steps=3, log_every_n_steps=1)
            subrun_dir = result_1["subrun_dir"]
            metrics_1 = self._load_metrics(subrun_dir)
            len_1 = len(metrics_1["history"])

            result_2 = self._run_continuation(
                tmpdir,
                continuation_from_steps=3,
                train_steps=6,
                log_every_n_steps=1,
            )
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_2["continuation_boundary_step"], 4)
            len_2 = len(metrics_2["history"])
            self.assertGreater(len_2, len_1)

            # when — second continuation
            result_3 = self._run_continuation(
                tmpdir,
                continuation_from_steps=6,
                train_steps=10,
                log_every_n_steps=1,
            )

            # then — boundary is most recent (7), full history
            metrics_3 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_3["status"], "completed")
            self.assertEqual(metrics_3["continuation_boundary_step"], 7)
            self.assertEqual(metrics_3["train_steps"], 10)
            self.assertGreater(len(metrics_3["history"]), len_2)
            self.assertEqual(
                [e["step"] for e in metrics_3["history"]],
                list(range(1, 11)),
            )

    def test_single_step_continuation(self) -> None:
        # given — train 4, continue to 5 (exactly 1 step)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(tmpdir, train_steps=4, log_every_n_steps=1)
            subrun_dir = result_1["subrun_dir"]
            metrics_1 = self._load_metrics(subrun_dir)
            original_len = len(metrics_1["history"])
            self.assertEqual(metrics_1["history"][-1]["step"], 4)

            # when
            result_2 = self._run_continuation(
                tmpdir,
                continuation_from_steps=4,
                train_steps=5,
                log_every_n_steps=1,
            )

            # then — last step logged via "force log on last step"
            metrics_2 = self._load_metrics(subrun_dir)
            self.assertEqual(metrics_2["status"], "completed")
            self.assertEqual(metrics_2["train_steps"], 5)
            self.assertEqual(metrics_2["continuation_boundary_step"], 5)
            self.assertEqual(len(metrics_2["history"]), original_len + 1)
            self.assertEqual(metrics_2["history"][-1]["step"], 5)

    def test_continuation_from_steps_ge_train_steps_raises(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            result_1 = self._run_fresh(tmpdir, train_steps=5)

            # when / then
            with self.assertRaises(ValueError) as ctx:
                self._run_continuation(
                    tmpdir,
                    continuation_from_steps=5,
                    train_steps=5,
                )
            self.assertIn("continuation_from_steps", str(ctx.exception))


class TestRunTrainingForSweepChunkContinuation(unittest.TestCase):
    def _load_metrics(self, subrun_dir: str) -> Dict[str, Any]:
        path = os.path.join(subrun_dir, "training_metrics.json")
        with open(path, "r") as f:
            return json.load(f)

    def test_sweep_chunk_continuation_produces_merged_history(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(tmpdir)
            config_payload["training"]["train_steps"] = 5
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved, seed=0
            )
            result_1 = run_training_for_lambda_chunk(
                config=config,
                resolved_config=resolved,
                variant=variant,
                seed=0,
                chunk_subruns=chunk_subruns,
                output_dir=tmpdir,
                overwrite=False,
            )
            for sd in result_1["subrun_dirs"]:
                m = self._load_metrics(sd)
                self.assertEqual(m["status"], "completed")
                self.assertEqual(m["train_steps"], 5)
            original_history_lens = {
                sd: len(self._load_metrics(sd)["history"])
                for sd in result_1["subrun_dirs"]
            }

            # when — continue to 10
            config_payload["training"]["train_steps"] = 10
            config2 = LLMBenchmarkConfig.model_validate(config_payload)
            resolved2 = resolve_benchmark_config(config2)
            variant2, chunk_subruns2 = _resolve_variant_and_chunk_subruns(
                resolved2, seed=0
            )
            result_2 = run_training_for_lambda_chunk(
                config=config2,
                resolved_config=resolved2,
                variant=variant2,
                seed=0,
                chunk_subruns=chunk_subruns2,
                output_dir=tmpdir,
                overwrite=False,
                continuation_from_steps=5,
            )

            # then
            for sd in result_2["subrun_dirs"]:
                m = self._load_metrics(sd)
                self.assertEqual(m["status"], "completed")
                self.assertEqual(m["train_steps"], 10)
                self.assertEqual(m["continuation_boundary_step"], 6)
                self.assertEqual(m["history"][-1]["step"], 10)
                self.assertGreater(
                    len(m["history"]),
                    original_history_lens[sd],
                )
                mc_path = os.path.join(sd, "model_config.json")
                with open(mc_path, "r") as f:
                    mc = json.load(f)
                self.assertNotIn("continuation_from_steps", mc)

    def test_sweep_chunk_continuation_round_trip_with_real_model(
        self,
    ) -> None:
        # given — train 2-instance JumpReLUSAE sweep, slice, reassemble
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(tmpdir, sae_class="JumpReLUSAE")
            config_payload["variants"][0]["theta_init"] = 0.01
            config_payload["variants"][0]["jump_init"] = 0.02
            config_payload["training"]["train_steps"] = 3
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved, seed=0
            )
            result = run_training_for_lambda_chunk(
                config=config,
                resolved_config=resolved,
                variant=variant,
                seed=0,
                chunk_subruns=chunk_subruns,
                output_dir=tmpdir,
                overwrite=False,
            )

            # when — load per-instance checkpoints and reassemble
            instance_sds = []
            for sd_path in result["subrun_dirs"]:
                cp = torch.load(
                    os.path.join(sd_path, "model.pt"),
                    map_location="cpu",
                    weights_only=True,
                )
                instance_sds.append(cp["state_dict"])
            assembled = _reassemble_state_dict_from_instances(
                instance_sds, n_instances=2
            )

            # then — load into a fresh 2-instance model
            model = JumpReLUSAE(
                n_instances=2,
                d_in=4,
                d_hidden=6,
                resample_latents_threshold=4,
                device="cpu",
            )
            model.load_state_dict(assembled)

    def test_sweep_chunk_continuation_with_overwrite_raises(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(tmpdir)
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved, seed=0
            )

            # when / then
            with self.assertRaises(ValueError) as ctx:
                run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved,
                    variant=variant,
                    seed=0,
                    chunk_subruns=chunk_subruns,
                    output_dir=tmpdir,
                    overwrite=True,
                    continuation_from_steps=5,
                )
            self.assertIn("mutually exclusive", str(ctx.exception))

    def test_sweep_chunk_continuation_ge_train_steps_raises(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(tmpdir)
            config_payload["training"]["train_steps"] = 5
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved, seed=0
            )
            run_training_for_lambda_chunk(
                config=config,
                resolved_config=resolved,
                variant=variant,
                seed=0,
                chunk_subruns=chunk_subruns,
                output_dir=tmpdir,
                overwrite=False,
            )

            # when / then — continue with same train_steps
            with self.assertRaises(ValueError) as ctx:
                run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved,
                    variant=variant,
                    seed=0,
                    chunk_subruns=chunk_subruns,
                    output_dir=tmpdir,
                    overwrite=False,
                    continuation_from_steps=5,
                )
            self.assertIn(
                "continuation_from_steps",
                str(ctx.exception),
            )

    def test_sweep_chunk_inconsistent_thresholds_raises(
        self,
    ) -> None:
        # given — train 2 instances, then tamper with one's metrics
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(tmpdir)
            config_payload["training"]["train_steps"] = 3
            config_payload["training"]["threshold_warmup_steps"] = 1
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved, seed=0
            )
            result = run_training_for_lambda_chunk(
                config=config,
                resolved_config=resolved,
                variant=variant,
                seed=0,
                chunk_subruns=chunk_subruns,
                output_dir=tmpdir,
                overwrite=False,
            )
            # tamper: set different thresholds_unfrozen_at_step
            sd0 = result["subrun_dirs"][0]
            sd1 = result["subrun_dirs"][1]
            m0_path = os.path.join(sd0, "training_metrics.json")
            m1_path = os.path.join(sd1, "training_metrics.json")
            with open(m0_path, "r") as f:
                m0 = json.load(f)
            with open(m1_path, "r") as f:
                m1 = json.load(f)
            m0["thresholds_unfrozen_at_step"] = 2
            m1["thresholds_unfrozen_at_step"] = 99
            with open(m0_path, "w") as f:
                json.dump(m0, f)
            with open(m1_path, "w") as f:
                json.dump(m1, f)

            # when / then
            config_payload["training"]["train_steps"] = 10
            config2 = LLMBenchmarkConfig.model_validate(config_payload)
            resolved2 = resolve_benchmark_config(config2)
            variant2, chunk_subruns2 = _resolve_variant_and_chunk_subruns(
                resolved2, seed=0
            )
            with self.assertRaises(ValueError) as ctx:
                run_training_for_lambda_chunk(
                    config=config2,
                    resolved_config=resolved2,
                    variant=variant2,
                    seed=0,
                    chunk_subruns=chunk_subruns2,
                    output_dir=tmpdir,
                    overwrite=False,
                    continuation_from_steps=3,
                )
            self.assertIn("Inconsistent", str(ctx.exception))

    def test_sweep_chunk_failure_during_continuation_preserves_history(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_activation_cache(tmpdir)
            config_payload = _build_lambda_sweep_config(tmpdir)
            config_payload["training"]["train_steps"] = 3
            config = LLMBenchmarkConfig.model_validate(config_payload)
            resolved = resolve_benchmark_config(config)
            variant, chunk_subruns = _resolve_variant_and_chunk_subruns(
                resolved, seed=0
            )
            result_1 = run_training_for_lambda_chunk(
                config=config,
                resolved_config=resolved,
                variant=variant,
                seed=0,
                chunk_subruns=chunk_subruns,
                output_dir=tmpdir,
                overwrite=False,
            )
            original_history_lens = []
            for sd in result_1["subrun_dirs"]:
                m = self._load_metrics(sd)
                original_history_lens.append(len(m["history"]))

            # when — patch forward to fail during continuation
            call_count = {"n": 0}
            original_forward = SAE.forward

            # Threshold > warmup forward calls so the
            # failure fires during the continuation loop.
            def _failing_forward(self_model, x):
                call_count["n"] += 1
                if call_count["n"] > 4:
                    raise RuntimeError("Injected failure")
                return original_forward(self_model, x)

            config_payload["training"]["train_steps"] = 10
            config2 = LLMBenchmarkConfig.model_validate(config_payload)
            resolved2 = resolve_benchmark_config(config2)
            variant2, chunk_subruns2 = _resolve_variant_and_chunk_subruns(
                resolved2, seed=0
            )
            with patch.object(SAE, "forward", _failing_forward):
                with self.assertRaises(RuntimeError):
                    run_training_for_lambda_chunk(
                        config=config2,
                        resolved_config=resolved2,
                        variant=variant2,
                        seed=0,
                        chunk_subruns=chunk_subruns2,
                        output_dir=tmpdir,
                        overwrite=False,
                        continuation_from_steps=3,
                    )

            # then
            for idx, sd in enumerate(result_1["subrun_dirs"]):
                m = self._load_metrics(sd)
                self.assertEqual(m["status"], "failed")
                self.assertGreaterEqual(
                    len(m["history"]),
                    original_history_lens[idx],
                )
                self.assertIn("continuation_boundary_step", m)
