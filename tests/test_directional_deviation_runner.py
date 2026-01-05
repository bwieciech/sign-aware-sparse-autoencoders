import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from typing import Any, Dict, List
from unittest.mock import patch

import torch

from sgnsae.experiments.directional_deviation import runner as runner_module
from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationAnomalyConfig,
    DirectionalDeviationDataConfig,
    DirectionalDeviationExperimentConfig,
    DirectionalDeviationModelVariantConfig,
    DirectionalDeviationScoringConfig,
    DirectionalDeviationThresholdConfig,
)


class TestDirectionalDeviationRunner(unittest.TestCase):
    def _build_minimal_experiment_config(
        self, output_dir: str, device: str, run_id: str
    ) -> DirectionalDeviationExperimentConfig:
        data_config = DirectionalDeviationDataConfig(
            seed=1,
            num_train=2,
            num_val=2,
            num_test_normal=1,
            num_test_high_per_channel=1,
            num_test_low_per_channel=1,
            num_channels=4,
        )
        anomaly_config = DirectionalDeviationAnomalyConfig(
            delta=1.0, shift_mode="single_channel"
        )
        threshold_config = DirectionalDeviationThresholdConfig(
            latent_quantile=0.99,
            score_fpr=0.01,
            min_validation_samples=1,
            deterministic_algorithms=False,
        )
        model_config = DirectionalDeviationModelVariantConfig(
            name="relu_sae",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        return DirectionalDeviationExperimentConfig(
            output_dir=output_dir,
            device=device,  # type: ignore[arg-type]
            data=data_config,
            anomaly=anomaly_config,
            thresholds=threshold_config,
            models=[model_config],
            run_id=run_id,
            wandb_project="project",
            wandb_api_key="secret",
        )

    def _build_minimal_run_state(
        self, num_channels: int
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        normal_data = {
            "x_train": torch.zeros((2, num_channels), dtype=torch.float32),
            "x_val": torch.zeros((2, num_channels), dtype=torch.float32),
            "x_test_normal": torch.zeros((1, num_channels), dtype=torch.float32),
        }
        anomaly_data = {
            "anomaly_manifest": {
                "shift_mode": "single_channel",
                "num_channels": num_channels,
                "channels_to_evaluate": list(range(num_channels)),
            }
        }
        return normal_data, anomaly_data

    def test_resolve_device_returns_cpu_when_cuda_not_available(self) -> None:
        # given
        with patch.object(
            runner_module.torch.cuda, "is_available", return_value=False
        ), patch("builtins.print"):
            # when
            device = runner_module._resolve_device("cuda")

            # then
            self.assertEqual(device, "cpu")

    def test_resolve_device_returns_device_when_no_resolution_needed(self) -> None:
        # given
        # when
        device = runner_module._resolve_device("cpu")

        # then
        self.assertEqual(device, "cpu")

    def test_resolve_device_auto_resolves_to_cpu_or_cuda(self) -> None:
        # given
        with patch.object(runner_module.torch.cuda, "is_available", return_value=True):
            # when
            device = runner_module._resolve_device("auto")

            # then
            self.assertEqual(device, "cuda")

        with patch.object(runner_module.torch.cuda, "is_available", return_value=False):
            # when
            device = runner_module._resolve_device("auto")

            # then
            self.assertEqual(device, "cpu")

    def test_configure_determinism_enables_cudnn_flags(self) -> None:
        # given
        with patch.object(
            runner_module.torch, "use_deterministic_algorithms"
        ) as deterministic_mock, patch.object(
            runner_module.torch.backends, "cudnn"
        ) as cudnn_mock:
            # when
            runner_module._configure_determinism(True)

            # then
            deterministic_mock.assert_called_once_with(True)
            self.assertTrue(cudnn_mock.deterministic)
            self.assertFalse(cudnn_mock.benchmark)

    def test_iterate_tensor_batches_splits_tensor_into_expected_batches(self) -> None:
        # given
        x = torch.arange(10, dtype=torch.float32).reshape(5, 2)

        # when
        batches = list(runner_module._iterate_tensor_batches(x, batch_size=2))

        # then
        self.assertEqual([batch.shape[0] for batch in batches], [2, 2, 1])
        self.assertTrue(torch.equal(batches[0], x[:2]))
        self.assertTrue(torch.equal(batches[1], x[2:4]))
        self.assertTrue(torch.equal(batches[2], x[4:5]))

    def test_iterate_tensor_batches_raises_for_non_positive_batch_size(self) -> None:
        # given
        x = torch.zeros((1, 2), dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = list(runner_module._iterate_tensor_batches(x, batch_size=0))

    def test_stable_model_seed_is_deterministic_and_non_negative(self) -> None:
        # given
        base_seed = 123
        model_name = "relu_sae"

        # when
        seed_a = runner_module._stable_model_seed(base_seed, model_name)
        seed_b = runner_module._stable_model_seed(base_seed, model_name)

        # then
        self.assertEqual(seed_a, seed_b)
        self.assertGreaterEqual(seed_a, 0)
        self.assertLess(seed_a, 2**63)

    def test_determine_scoring_mode_returns_expected_mode(self) -> None:
        # given
        base_config = DirectionalDeviationModelVariantConfig(
            name="model",
            sae_class="SAE",
            sign_aware=False,
            single_activation_offset=False,
            reset_activation_offsets_on_resample=False,
            d_hidden=2,
            sparsity_coeff=0.0,
            learning_rate=0.01,
            num_epochs=1,
            batch_size=2,
            adam_betas=(0.9, 0.999),
        )
        sign_aware_config = replace(base_config, sign_aware=True)

        # when
        baseline_mode = runner_module._determine_scoring_mode(base_config)
        sign_aware_mode = runner_module._determine_scoring_mode(sign_aware_config)

        # then
        self.assertEqual(baseline_mode, "baseline")
        self.assertEqual(sign_aware_mode, "sign_aware")

    def test_extract_validation_signal_cpu_uses_detection_signal_for_sign_aware(
        self,
    ) -> None:
        # given
        x_val = torch.zeros((3, 2), dtype=torch.float32)

        def extract_side_effect(
            model: torch.nn.Module, batch: torch.Tensor
        ) -> torch.Tensor:
            return torch.full((batch.shape[0], 1), 1.0, dtype=torch.float32)

        # when
        with patch.object(
            runner_module, "extract_detection_signal", side_effect=extract_side_effect
        ) as extract_mock:
            output = runner_module._extract_validation_signal_cpu(
                model=torch.nn.Identity(),
                scoring_mode="sign_aware",
                x_val=x_val,
                batch_size=2,
            )

        # then
        self.assertEqual(extract_mock.call_count, 2)
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertTrue(torch.allclose(output, torch.ones((3, 1), dtype=torch.float32)))

    def test_extract_validation_signal_cpu_uses_one_sided_evidence_for_baseline(
        self,
    ) -> None:
        # given
        x_val = torch.zeros((3, 2), dtype=torch.float32)

        def evidence_side_effect(
            model: torch.nn.Module, batch: torch.Tensor
        ) -> torch.Tensor:
            return torch.full((batch.shape[0], 1), 2.0, dtype=torch.float32)

        # when
        with patch.object(
            runner_module,
            "extract_one_sided_evidence",
            side_effect=evidence_side_effect,
        ) as evidence_mock, patch.object(
            runner_module,
            "extract_detection_signal",
        ) as detection_mock:
            output = runner_module._extract_validation_signal_cpu(
                model=torch.nn.Identity(),
                scoring_mode="baseline",
                x_val=x_val,
                batch_size=2,
            )

        # then
        self.assertEqual(evidence_mock.call_count, 2)
        detection_mock.assert_not_called()
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertTrue(
            torch.allclose(output, torch.full((3, 1), 2.0, dtype=torch.float32))
        )

    def test_extract_validation_signal_cpu_raises_for_empty_validation_set(
        self,
    ) -> None:
        # given
        x_val = torch.empty((0, 2), dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = runner_module._extract_validation_signal_cpu(
                model=torch.nn.Identity(),
                scoring_mode="baseline",
                x_val=x_val,
                batch_size=2,
            )

    def test_calibrate_and_persist_thresholds_raises_for_unknown_shift_mode(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            model_dir = os.path.join(tmpdir, "models", "relu_sae")
            os.makedirs(model_dir, exist_ok=True)

            # when & then
            with self.assertRaises(ValueError):
                _ = runner_module._calibrate_and_persist_thresholds(
                    experiment_config=config,
                    model=torch.nn.Identity(),
                    scoring_mode="baseline",
                    anomaly_manifest={"shift_mode": "bogus"},
                    x_val=torch.zeros((2, 2), dtype=torch.float32),
                    model_dir=model_dir,
                    batch_size=2,
                )

    def test_calibrate_and_persist_thresholds_persists_sign_aware_threshold_files(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            model_dir = os.path.join(tmpdir, "models", "relu_sae")
            os.makedirs(model_dir, exist_ok=True)
            anomaly_manifest = {"shift_mode": "single_channel", "subset_indices": None}
            pi_val = torch.tensor(
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32
            )

            # when
            with patch.object(
                runner_module,
                "_extract_validation_signal_cpu",
                return_value=pi_val,
            ):
                (
                    score_thresholds,
                    latent_thresholds,
                    baseline_thresholds,
                    zscore_stats,
                ) = runner_module._calibrate_and_persist_thresholds(
                    experiment_config=config,
                    model=torch.nn.Identity(),
                    scoring_mode="sign_aware",
                    anomaly_manifest=anomaly_manifest,
                    x_val=torch.zeros((3, 1), dtype=torch.float32),
                    model_dir=model_dir,
                    batch_size=2,
                )

            # then
            self.assertIsNone(baseline_thresholds)
            self.assertIsNotNone(latent_thresholds)
            self.assertEqual(latent_thresholds.tau_plus.shape[0], 2)  # type: ignore[union-attr]
            self.assertIsNone(zscore_stats)
            self.assertIsInstance(score_thresholds.theta_high, float)
            thresholds_dir = os.path.join(model_dir, "thresholds")
            self.assertTrue(os.path.isfile(os.path.join(thresholds_dir, "tau_plus.pt")))
            self.assertTrue(
                os.path.isfile(os.path.join(thresholds_dir, "tau_minus.pt"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(thresholds_dir, "score_thresholds.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(thresholds_dir, "thresholds_metadata.json"))
            )

    def test_calibrate_and_persist_thresholds_zscore_normalization_persists_stats(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            config = replace(
                config,
                scoring=DirectionalDeviationScoringConfig(
                    pooling="sum",
                    signal_normalization="zscore",
                    normalization_eps=1e-6,
                ),
            )
            model_dir = os.path.join(tmpdir, "models", "relu_sae")
            os.makedirs(model_dir, exist_ok=True)
            anomaly_manifest = {"shift_mode": "single_channel", "subset_indices": None}
            pi_val = torch.tensor(
                [[-2.0, 1.0], [0.0, 2.0], [2.0, 3.0], [4.0, 4.0]],
                dtype=torch.float32,
            )

            # when
            with patch.object(
                runner_module,
                "_extract_validation_signal_cpu",
                return_value=pi_val,
            ):
                (
                    score_thresholds,
                    latent_thresholds,
                    baseline_thresholds,
                    zscore_stats,
                ) = runner_module._calibrate_and_persist_thresholds(
                    experiment_config=config,
                    model=torch.nn.Identity(),
                    scoring_mode="sign_aware",
                    anomaly_manifest=anomaly_manifest,
                    x_val=torch.zeros((4, 1), dtype=torch.float32),
                    model_dir=model_dir,
                    batch_size=2,
                )

            # then
            self.assertIsNotNone(latent_thresholds)
            self.assertIsNone(baseline_thresholds)
            self.assertIsNotNone(zscore_stats)
            self.assertTrue(math.isfinite(score_thresholds.theta_high))
            self.assertTrue(math.isfinite(score_thresholds.theta_low))
            thresholds_dir = os.path.join(model_dir, "thresholds")
            self.assertTrue(
                os.path.isfile(os.path.join(thresholds_dir, "signal_normalization.pt"))
            )
            with open(
                os.path.join(thresholds_dir, "thresholds_metadata.json"), "r"
            ) as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["signal_normalization"]["mode"], "zscore")
            self.assertEqual(
                metadata["scoring_config"]["signal_normalization"], "zscore"
            )
            self.assertEqual(len(metadata["signal_normalization"]["mean"]), 2)
            self.assertEqual(len(metadata["signal_normalization"]["std"]), 2)

    def test_calibrate_and_persist_thresholds_baseline_zscore_normalization_persists_stats(
        self,
    ) -> None:
        # given
        from sgnsae.models import SAE

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            config = replace(
                config,
                scoring=DirectionalDeviationScoringConfig(
                    pooling="sum",
                    signal_normalization="zscore",
                    normalization_eps=1e-6,
                ),
            )
            model_dir = os.path.join(tmpdir, "models", "relu_sae")
            os.makedirs(model_dir, exist_ok=True)
            anomaly_manifest = {"shift_mode": "single_channel", "subset_indices": None}
            evidence_val = torch.tensor(
                [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0]],
                dtype=torch.float32,
            )
            model = SAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            with torch.no_grad():
                model.W_dec[0] = torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
                )

            # when
            with patch.object(
                runner_module,
                "_extract_validation_signal_cpu",
                return_value=evidence_val,
            ):
                (
                    score_thresholds,
                    latent_thresholds,
                    baseline_thresholds,
                    zscore_stats,
                ) = runner_module._calibrate_and_persist_thresholds(
                    experiment_config=config,
                    model=model,
                    scoring_mode="baseline",
                    anomaly_manifest=anomaly_manifest,
                    x_val=torch.zeros((4, 2), dtype=torch.float32),
                    model_dir=model_dir,
                    batch_size=2,
                )

            # then
            self.assertIsNone(latent_thresholds)
            self.assertIsNotNone(baseline_thresholds)
            self.assertIsNotNone(zscore_stats)
            self.assertTrue(math.isfinite(score_thresholds.theta_high))
            self.assertTrue(math.isfinite(score_thresholds.theta_low))
            thresholds_dir = os.path.join(model_dir, "thresholds")
            self.assertTrue(
                os.path.isfile(os.path.join(thresholds_dir, "signal_normalization.pt"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(thresholds_dir, "latent_directions.pt"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(thresholds_dir, "evidence_tau.pt"))
            )
            with open(
                os.path.join(thresholds_dir, "thresholds_metadata.json"), "r"
            ) as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["signal_normalization"]["mode"], "zscore")
            self.assertEqual(
                metadata["scoring_config"]["signal_normalization"], "zscore"
            )

    def test_calibrate_and_persist_thresholds_target_restricted_scoring_generates_targets(
        self,
    ) -> None:
        # given
        from sgnsae.models import SAE

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            config = replace(
                config,
                scoring=DirectionalDeviationScoringConfig(
                    pooling="sum", target_restricted_scoring=True
                ),
            )
            model_dir = os.path.join(tmpdir, "models", "relu_sae")
            os.makedirs(model_dir, exist_ok=True)
            anomaly_manifest = {
                "shift_mode": "single_channel",
                "subset_indices": None,
                "channels_to_evaluate": [0, 1],
            }
            pi_val = torch.tensor(
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32
            )
            model = SAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=1,
                sign_aware=False,
                device="cpu",
            )
            with torch.no_grad():
                model.W_dec[0] = torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
                )

            # when
            with patch.object(
                runner_module,
                "_extract_validation_signal_cpu",
                return_value=pi_val,
            ):
                (
                    score_thresholds,
                    latent_thresholds,
                    baseline_thresholds,
                    zscore_stats,
                ) = runner_module._calibrate_and_persist_thresholds(
                    experiment_config=config,
                    model=model,
                    scoring_mode="sign_aware",
                    anomaly_manifest=anomaly_manifest,
                    x_val=torch.zeros((3, 2), dtype=torch.float32),
                    model_dir=model_dir,
                    batch_size=2,
                )

            # then
            self.assertIsNotNone(latent_thresholds)
            self.assertIsNone(baseline_thresholds)
            self.assertIsNone(zscore_stats)
            thresholds_dir = os.path.join(model_dir, "thresholds")
            with open(
                os.path.join(thresholds_dir, "thresholds_metadata.json"), "r"
            ) as f:
                metadata = json.load(f)
            self.assertTrue(metadata["scoring_config"]["target_restricted_scoring"])
            self.assertEqual(metadata["scoring_config"]["pooling"], "sum")
            self.assertIsInstance(score_thresholds.theta_high, float)

    def test_calibrate_and_persist_thresholds_target_restricted_scoring_raises_when_channels_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            config = replace(
                config,
                scoring=DirectionalDeviationScoringConfig(
                    pooling="sum", target_restricted_scoring=True
                ),
            )
            model_dir = os.path.join(tmpdir, "models", "relu_sae")
            os.makedirs(model_dir, exist_ok=True)

            # when & then
            with self.assertRaises(ValueError):
                _ = runner_module._calibrate_and_persist_thresholds(
                    experiment_config=config,
                    model=torch.nn.Identity(),
                    scoring_mode="sign_aware",
                    anomaly_manifest={"shift_mode": "single_channel"},
                    x_val=torch.zeros((2, 2), dtype=torch.float32),
                    model_dir=model_dir,
                    batch_size=2,
                )

    def test_prepare_run_directory_uses_run_id_when_provided(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="test_run"
            )

            # when
            run_dir, run_id = runner_module._prepare_run_directory(config)

            # then
            self.assertEqual(run_id, "test_run")
            self.assertEqual(
                run_dir, os.path.join(tmpdir, "directional_deviation", "test_run")
            )
            self.assertTrue(os.path.isdir(run_dir))

    def test_persist_run_config_redacts_wandb_api_key(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="test_run"
            )
            run_dir = os.path.join(tmpdir, "directional_deviation", "test_run")
            os.makedirs(run_dir, exist_ok=True)

            # when
            runner_module._persist_run_config(config, run_dir)

            # then
            config_path = os.path.join(run_dir, "config.json")
            with open(config_path, "r") as f:
                persisted = json.load(f)
            self.assertIsNone(persisted["wandb_api_key"])

    def test_run_directional_deviation_experiment_persists_config_and_returns_run_dir(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="auto", run_id="test_run"
            )

            # when
            with patch.object(
                runner_module,
                "load_experiment_config",
                return_value=config,
            ), patch.object(
                runner_module,
                "_resolve_device",
                return_value="cpu",
            ), patch.object(
                runner_module,
                "_run_experiment_dispatch",
                return_value={},
            ) as dispatch_mock:
                run_dir = runner_module.run_directional_deviation_experiment(
                    config_path=os.path.join(tmpdir, "config.json")
                )

            # then
            self.assertTrue(os.path.isdir(run_dir))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, "config.json")))
            dispatch_mock.assert_called_once()

    def test_run_experiment_dispatch_prefers_width_sweep(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            sweep_config = replace(config, width_sweep=[1])

            # when
            with patch.object(
                runner_module, "_run_width_sweep", return_value={"sentinel": "width"}
            ) as width_mock:
                result = runner_module._run_experiment_dispatch(
                    config=sweep_config, run_dir=tmpdir, batch_size=8
                )

            # then
            width_mock.assert_called_once()
            self.assertEqual(result, {"sentinel": "width"})

    def test_run_experiment_dispatch_prefers_seed_sweep_over_leaf(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            seed_config = replace(config, data=replace(config.data, seeds=[1, 2]))

            # when
            with patch.object(
                runner_module, "_run_seed_sweep", return_value={"sentinel": "seed"}
            ) as seed_mock:
                result = runner_module._run_experiment_dispatch(
                    config=seed_config, run_dir=tmpdir, batch_size=8
                )

            # then
            seed_mock.assert_called_once()
            self.assertEqual(result, {"sentinel": "seed"})

    def test_run_experiment_dispatch_runs_leaf_when_no_sweep_configured(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )

            # when
            with patch.object(
                runner_module, "_run_leaf_experiment", return_value={"sentinel": "leaf"}
            ) as leaf_mock:
                result = runner_module._run_experiment_dispatch(
                    config=config, run_dir=tmpdir, batch_size=8
                )

            # then
            leaf_mock.assert_called_once()
            self.assertEqual(result, {"sentinel": "leaf"})

    def test_run_width_sweep_persists_subrun_configs_and_summary(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            sweep_config = replace(config, width_sweep=[1, 2])
            run_dir = os.path.join(tmpdir, "directional_deviation", "root")
            os.makedirs(run_dir, exist_ok=True)

            def dispatch_side_effect(
                config: DirectionalDeviationExperimentConfig,
                run_dir: str,
                batch_size: int,
            ) -> Dict[str, Any]:
                return {
                    "models": {
                        "relu_sae": {
                            "recall_high": float(config.models[0].d_hidden),
                        }
                    }
                }

            # when
            with patch.object(
                runner_module,
                "_run_experiment_dispatch",
                side_effect=dispatch_side_effect,
            ):
                summary = runner_module._run_width_sweep(
                    config=sweep_config, run_dir=run_dir, batch_size=8
                )

            # then
            summary_path = os.path.join(run_dir, "comparison_summary.json")
            self.assertTrue(os.path.isfile(summary_path))
            with open(summary_path, "r") as f:
                persisted = json.load(f)

            self.assertEqual(persisted["sweep"]["type"], "width")
            self.assertEqual(set(persisted["per_width"].keys()), {"1", "2"})
            self.assertEqual(set(summary["per_width"].keys()), {"1", "2"})

            per_width_metrics_path = os.path.join(run_dir, "per-width metrics.json")
            self.assertTrue(os.path.isfile(per_width_metrics_path))
            with open(per_width_metrics_path, "r") as f:
                per_width_metrics = json.load(f)
            self.assertEqual(per_width_metrics["widths"], [1, 2])
            self.assertEqual(
                set(per_width_metrics["per_width"].keys()),
                {"1", "2"},
            )

            for width in sweep_config.width_sweep or []:
                width_dir = os.path.join(run_dir, f"width_{width}")
                self.assertTrue(os.path.isdir(width_dir))
                config_path = os.path.join(width_dir, "config.json")
                self.assertTrue(os.path.isfile(config_path))
                metrics_path = os.path.join(width_dir, "metrics.json")
                self.assertTrue(os.path.isfile(metrics_path))
                with open(config_path, "r") as f:
                    width_config = json.load(f)
                self.assertIsNone(width_config["wandb_api_key"])
                self.assertIsNone(width_config["width_sweep"])
                self.assertEqual(width_config["models"][0]["d_hidden"], width)

    def test_extract_models_metrics_for_metrics_json_supports_aggregate_over_seeds(
        self,
    ) -> None:
        # given
        summary = {
            "aggregate_over_seeds": {
                "models": {
                    "relu_sae": {
                        "recall_high_mean": 0.1,
                        "recall_high_std": 0.0,
                        "recall_high_n": 1,
                    }
                }
            }
        }

        # when
        models = runner_module._extract_models_metrics_for_metrics_json(summary)

        # then
        self.assertIn("relu_sae", models)

    def test_run_seed_sweep_raises_when_seeds_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )

            # when & then
            with self.assertRaises(ValueError):
                _ = runner_module._run_seed_sweep(
                    config=config, run_dir=tmpdir, batch_size=8
                )

    def test_run_width_sweep_raises_when_width_sweep_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )

            # when & then
            with self.assertRaises(ValueError):
                _ = runner_module._run_width_sweep(
                    config=config, run_dir=tmpdir, batch_size=8
                )

    def test_run_seed_sweep_persists_subrun_configs_and_aggregates_metrics(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            seed_config = replace(
                base_config, data=replace(base_config.data, seeds=[10, 11])
            )
            run_dir = os.path.join(tmpdir, "directional_deviation", "root")
            os.makedirs(run_dir, exist_ok=True)

            def leaf_side_effect(
                config: DirectionalDeviationExperimentConfig,
                run_dir: str,
                batch_size: int,
            ) -> Dict[str, Any]:
                if config.data.seed == 10:
                    recall_high = 0.1
                    recall_low = 0.2
                    covered_count = 2
                    covered_fraction = 0.5
                else:
                    recall_high = 0.3
                    recall_low = 0.4
                    covered_count = 1
                    covered_fraction = 0.25
                return {
                    "models": {
                        "relu_sae": {
                            "recall_high": recall_high,
                            "recall_low": recall_low,
                            "fpr_high": 0.01,
                            "fpr_low": 0.02,
                            "covered_channels_fraction": covered_fraction,
                            "covered_channels_count": covered_count,
                            "error": None,
                        }
                    }
                }

            # when
            with patch.object(
                runner_module, "_run_leaf_experiment", side_effect=leaf_side_effect
            ):
                summary = runner_module._run_seed_sweep(
                    config=seed_config, run_dir=run_dir, batch_size=8
                )

            # then
            self.assertEqual(summary["sweep"]["type"], "seed")
            self.assertEqual(set(summary["per_seed"].keys()), {"10", "11"})

            summary_path = os.path.join(run_dir, "comparison_summary.json")
            self.assertTrue(os.path.isfile(summary_path))
            with open(summary_path, "r") as f:
                persisted = json.load(f)
            aggregate = persisted["aggregate_over_seeds"]["models"]["relu_sae"]
            self.assertAlmostEqual(aggregate["recall_high_mean"], 0.2)
            self.assertAlmostEqual(aggregate["recall_high_std"], 0.1)
            self.assertEqual(aggregate["recall_high_n"], 2)
            self.assertAlmostEqual(aggregate["covered_channels_count_mean"], 1.5)
            self.assertAlmostEqual(aggregate["covered_channels_count_std"], 0.5)
            self.assertEqual(aggregate["covered_channels_count_n"], 2)

            for seed in (10, 11):
                seed_dir = os.path.join(run_dir, f"seed_{seed}")
                self.assertTrue(os.path.isdir(seed_dir))
                config_path = os.path.join(seed_dir, "config.json")
                self.assertTrue(os.path.isfile(config_path))
                with open(config_path, "r") as f:
                    seed_persisted = json.load(f)
                self.assertIsNone(seed_persisted["wandb_api_key"])
                self.assertEqual(seed_persisted["data"]["seed"], seed)
                self.assertIsNone(seed_persisted["data"]["seeds"])

    def test_aggregate_seed_sweep_skips_seed_summaries_without_models_dict(
        self,
    ) -> None:
        # given
        per_seed_summaries = {
            0: {"models": None},
            1: {"models": {"relu_sae": {"recall_high": 0.1}}},
        }

        # when
        aggregate = runner_module._aggregate_seed_sweep(per_seed_summaries)

        # then
        relu_summary = aggregate["models"]["relu_sae"]
        self.assertAlmostEqual(relu_summary["recall_high_mean"], 0.1)
        self.assertAlmostEqual(relu_summary["recall_high_std"], 0.0)
        self.assertEqual(relu_summary["recall_high_n"], 1)

    def test_aggregate_seed_sweep_skips_non_dict_model_metrics(self) -> None:
        # given
        per_seed_summaries = {
            0: {"models": {"relu_sae": "invalid"}},
            1: {"models": {"relu_sae": {"recall_high": 0.1}}},
        }

        # when
        aggregate = runner_module._aggregate_seed_sweep(per_seed_summaries)

        # then
        relu_summary = aggregate["models"]["relu_sae"]
        self.assertAlmostEqual(relu_summary["recall_high_mean"], 0.1)
        self.assertEqual(relu_summary["recall_high_n"], 1)

    def test_aggregate_seed_sweep_skips_none_metric_values(self) -> None:
        # given
        per_seed_summaries = {
            0: {
                "models": {
                    "relu_sae": {
                        "recall_high": None,
                        "recall_low": 0.2,
                    }
                }
            }
        }

        # when
        aggregate = runner_module._aggregate_seed_sweep(per_seed_summaries)

        # then
        relu_summary = aggregate["models"]["relu_sae"]
        self.assertIsNone(relu_summary["recall_high_mean"])
        self.assertEqual(relu_summary["recall_high_n"], 0)
        self.assertAlmostEqual(relu_summary["recall_low_mean"], 0.2)
        self.assertEqual(relu_summary["recall_low_n"], 1)

    def test_run_single_model_persists_error_evaluation_on_failure(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            normal_data, anomaly_data = self._build_minimal_run_state(
                num_channels=config.data.num_channels
            )

            # when
            with patch.object(
                runner_module,
                "run_training_for_variant",
                side_effect=RuntimeError("boom"),
            ):
                evaluation = runner_module._run_single_model(
                    experiment_config=config,
                    run_dir=tmpdir,
                    model_config=config.models[0],
                    normal_data=normal_data,
                    anomaly_data=anomaly_data,
                    batch_size=8,
                )

            # then
            self.assertIsNone(evaluation["metrics"])
            self.assertIn("RuntimeError: boom", evaluation["error"])
            evaluation_path = os.path.join(
                tmpdir, "models", config.models[0].name, "evaluation.json"
            )
            self.assertTrue(os.path.isfile(evaluation_path))

    def test_run_single_model_success_path_calls_dependencies(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            normal_data, anomaly_data = self._build_minimal_run_state(
                num_channels=config.data.num_channels
            )
            score_thresholds = runner_module.ScoreThresholds(
                theta_high=0.5,
                theta_low=0.25,
                score_fpr=0.01,
                score_quantile=0.99,
            )
            expected = {
                "model_name": config.models[0].name,
                "metrics": {"recall_high": 1.0},
            }

            # when
            with patch.object(
                runner_module,
                "run_training_for_variant",
                return_value={
                    "model": torch.nn.Linear(4, 2),
                    "training_metrics": {"loss_total": []},
                },
            ) as train_mock, patch.object(
                runner_module, "persist_model_artifacts"
            ) as persist_mock, patch.object(
                runner_module,
                "_calibrate_and_persist_thresholds",
                return_value=(score_thresholds, None, None, None),
            ) as calibrate_mock, patch.object(
                runner_module,
                "evaluate_and_persist_directional_deviation",
                return_value=expected,
            ) as evaluation_mock:
                result = runner_module._run_single_model(
                    experiment_config=config,
                    run_dir=tmpdir,
                    model_config=config.models[0],
                    normal_data=normal_data,
                    anomaly_data=anomaly_data,
                    batch_size=8,
                )

            # then
            train_mock.assert_called_once()
            persist_mock.assert_called_once()
            calibrate_mock.assert_called_once()
            evaluation_mock.assert_called_once()
            self.assertEqual(result, expected)
            called_inputs = evaluation_mock.call_args.kwargs["inputs"]
            self.assertEqual(called_inputs.model_name, config.models[0].name)
            self.assertEqual(called_inputs.evaluation_batch_size, 8)
            self.assertEqual(
                called_inputs.coverage_recall_target, config.coverage_recall_target
            )

    def test_run_leaf_experiment_feature_direction_trains_reference_model_before_building_anomalies(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_config = DirectionalDeviationDataConfig(
                seed=1,
                num_train=2,
                num_val=2,
                num_test_normal=1,
                num_test_high_per_channel=1,
                num_test_low_per_channel=1,
                num_channels=4,
            )
            anomaly_config = DirectionalDeviationAnomalyConfig(
                delta=1.0,
                shift_mode="feature_direction",
                direction_source="reference_model_decoder",
                num_directions=2,
                reference_model_name="sign_aware_gated_sae",
                direction_selection="top_purity",
            )
            threshold_config = DirectionalDeviationThresholdConfig(
                latent_quantile=0.99,
                score_fpr=0.01,
                min_validation_samples=1,
                deterministic_algorithms=False,
            )
            relu_config = DirectionalDeviationModelVariantConfig(
                name="relu_sae",
                sae_class="SAE",
                sign_aware=False,
                single_activation_offset=False,
                reset_activation_offsets_on_resample=False,
                d_hidden=2,
                sparsity_coeff=0.0,
                learning_rate=0.01,
                num_epochs=1,
                batch_size=2,
                adam_betas=(0.9, 0.999),
            )
            reference_config = DirectionalDeviationModelVariantConfig(
                name="sign_aware_gated_sae",
                sae_class="GatedSAE",
                sign_aware=True,
                single_activation_offset=False,
                reset_activation_offsets_on_resample=False,
                d_hidden=3,
                sparsity_coeff=0.0,
                learning_rate=0.01,
                num_epochs=1,
                batch_size=2,
                adam_betas=(0.9, 0.999),
            )
            config = DirectionalDeviationExperimentConfig(
                output_dir=tmpdir,
                device="cpu",
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[relu_config, reference_config],
                run_id="root",
            )
            normal_data = {
                "x_train": torch.zeros((2, 4), dtype=torch.float32),
                "x_val": torch.zeros((2, 4), dtype=torch.float32),
                "x_test_normal": torch.zeros((1, 4), dtype=torch.float32),
            }
            anomaly_directions = torch.eye(4, dtype=torch.float32)[:2]
            direction_metadata: Dict[str, Any] = {
                "direction_selection": "top_purity",
                "direction_latent_indices": [0, 1],
            }
            anomaly_data = {
                "anomaly_manifest": {
                    "shift_mode": "feature_direction",
                    "num_channels": 4,
                    "num_directions": 2,
                    "direction_indices_to_evaluate": [0, 1],
                }
            }
            reference_training_output = {
                "model": torch.nn.Identity(),
                "training_metrics": {"loss_total": []},
            }

            single_model_calls: List[Dict[str, Any]] = []

            def run_single_model_side_effect(**kwargs: Any) -> Dict[str, Any]:
                single_model_calls.append(kwargs)
                return {
                    "model_name": kwargs["model_config"].name,
                    "metrics": {"recall_high": 0.0, "recall_low": 0.0},
                    "per_channel": None,
                    "per_direction": None,
                    "error": None,
                }

            # when
            with patch.object(
                runner_module, "generate_normal_data", return_value=normal_data
            ), patch.object(
                runner_module,
                "run_training_for_variant",
                return_value=reference_training_output,
            ) as train_mock, patch.object(
                runner_module,
                "build_feature_directions_from_decoder",
                return_value=(anomaly_directions, direction_metadata),
            ) as directions_mock, patch.object(
                runner_module,
                "generate_anomaly_test_data",
                return_value=anomaly_data,
            ) as anomaly_mock, patch.object(
                runner_module,
                "export_data_summary",
            ), patch.object(
                runner_module,
                "_run_single_model",
                side_effect=run_single_model_side_effect,
            ) as single_model_mock, patch.object(
                runner_module,
                "persist_comparison_summary",
            ), patch.object(
                runner_module,
                "generate_plots_for_run",
            ):
                _ = runner_module._run_leaf_experiment(
                    config=config, run_dir=tmpdir, batch_size=8
                )

            # then
            train_mock.assert_called_once()
            train_kwargs = train_mock.call_args.kwargs
            self.assertEqual(train_kwargs["model_config"].name, "sign_aware_gated_sae")

            directions_mock.assert_called_once()
            anomaly_mock.assert_called_once()
            anomaly_kwargs = anomaly_mock.call_args.kwargs
            self.assertTrue(
                torch.equal(anomaly_kwargs["anomaly_directions"], anomaly_directions)
            )
            self.assertEqual(anomaly_kwargs["direction_metadata"], direction_metadata)

            self.assertEqual(single_model_mock.call_count, 2)
            by_name = {call["model_config"].name: call for call in single_model_calls}
            self.assertIn("relu_sae", by_name)
            self.assertIn("sign_aware_gated_sae", by_name)
            self.assertIsNone(by_name["relu_sae"].get("training_output"))
            self.assertIs(
                by_name["sign_aware_gated_sae"].get("training_output"),
                reference_training_output,
            )

    def test_run_leaf_experiment_raises_for_per_model_decoder_direction_source(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            feature_anomaly = DirectionalDeviationAnomalyConfig(
                delta=1.0,
                shift_mode="feature_direction",
                direction_source="per_model_decoder",
                num_directions=1,
            )
            feature_config = replace(config, anomaly=feature_anomaly)

            # when & then
            with patch.object(
                runner_module, "generate_normal_data", return_value={}
            ), self.assertRaises(ValueError):
                _ = runner_module._run_leaf_experiment(
                    config=feature_config, run_dir=tmpdir, batch_size=8
                )

    def test_run_leaf_experiment_raises_when_reference_model_name_missing_after_mutation(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            data_config = DirectionalDeviationDataConfig(
                seed=1,
                num_train=2,
                num_val=2,
                num_test_normal=1,
                num_test_high_per_channel=1,
                num_test_low_per_channel=1,
                num_channels=4,
            )
            anomaly_config = DirectionalDeviationAnomalyConfig(
                delta=1.0,
                shift_mode="feature_direction",
                direction_source="reference_model_decoder",
                num_directions=1,
                reference_model_name="sign_aware_gated_sae",
            )
            threshold_config = DirectionalDeviationThresholdConfig(
                latent_quantile=0.99,
                score_fpr=0.01,
                min_validation_samples=1,
                deterministic_algorithms=False,
            )
            model_config = DirectionalDeviationModelVariantConfig(
                name="sign_aware_gated_sae",
                sae_class="SAE",
                sign_aware=False,
                single_activation_offset=False,
                reset_activation_offsets_on_resample=False,
                d_hidden=2,
                sparsity_coeff=0.0,
                learning_rate=0.01,
                num_epochs=1,
                batch_size=2,
                adam_betas=(0.9, 0.999),
            )
            config = DirectionalDeviationExperimentConfig(
                output_dir=tmpdir,
                device="cpu",
                data=data_config,
                anomaly=anomaly_config,
                thresholds=threshold_config,
                models=[model_config],
                run_id="root",
            )
            config.anomaly.reference_model_name = "missing"  # type: ignore[assignment]

            # when & then
            with patch.object(
                runner_module, "generate_normal_data", return_value={}
            ), self.assertRaises(ValueError):
                _ = runner_module._run_leaf_experiment(
                    config=config, run_dir=tmpdir, batch_size=8
                )

    def test_run_leaf_experiment_catches_plot_generation_failure_when_verbose(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            normal_data, anomaly_data = self._build_minimal_run_state(
                num_channels=config.data.num_channels
            )
            evaluation = {"metrics": {"recall_high": 0.0, "recall_low": 0.0}}

            # when
            with patch.object(
                runner_module, "generate_normal_data", return_value=normal_data
            ), patch.object(
                runner_module,
                "generate_anomaly_test_data",
                return_value=anomaly_data,
            ), patch.object(
                runner_module, "export_data_summary"
            ), patch.object(
                runner_module, "_run_single_model", return_value=evaluation
            ), patch.object(
                runner_module, "persist_comparison_summary"
            ), patch.object(
                runner_module,
                "generate_plots_for_run",
                side_effect=RuntimeError("plot_fail"),
            ), patch(
                "builtins.print"
            ) as print_mock:
                _ = runner_module._run_leaf_experiment(
                    config=config, run_dir=tmpdir, batch_size=8
                )

            # then
            self.assertTrue(
                any("plot_fail" in str(call) for call in print_mock.call_args_list)
            )

    def test_run_seed_sweep_catches_plot_generation_failure_when_verbose(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            sweep_config = replace(config, data=replace(config.data, seeds=[0]))
            run_dir = os.path.join(tmpdir, "directional_deviation", "root")
            os.makedirs(run_dir, exist_ok=True)

            # when
            with patch.object(
                runner_module, "_run_leaf_experiment", return_value={"models": {}}
            ), patch.object(
                runner_module,
                "generate_plots_for_run",
                side_effect=RuntimeError("plot_fail"),
            ), patch(
                "builtins.print"
            ) as print_mock:
                _ = runner_module._run_seed_sweep(
                    config=sweep_config, run_dir=run_dir, batch_size=8
                )

            # then
            self.assertTrue(
                any("plot_fail" in str(call) for call in print_mock.call_args_list)
            )

    def test_run_width_sweep_catches_plot_generation_failure_when_verbose(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._build_minimal_experiment_config(
                output_dir=tmpdir, device="cpu", run_id="root"
            )
            sweep_config = replace(config, width_sweep=[1])
            run_dir = os.path.join(tmpdir, "directional_deviation", "root")
            os.makedirs(run_dir, exist_ok=True)

            # when
            with patch.object(
                runner_module,
                "_run_experiment_dispatch",
                return_value={"models": {"relu_sae": {"recall_high": 0.0}}},
            ), patch.object(runner_module, "_persist_per_width_metrics"), patch.object(
                runner_module,
                "generate_plots_for_run",
                side_effect=RuntimeError("plot_fail"),
            ), patch(
                "builtins.print"
            ) as print_mock:
                _ = runner_module._run_width_sweep(
                    config=sweep_config, run_dir=run_dir, batch_size=8
                )

            # then
            self.assertTrue(
                any("plot_fail" in str(call) for call in print_mock.call_args_list)
            )

    def test_extract_models_metrics_for_metrics_json_returns_empty_when_no_models(
        self,
    ) -> None:
        # given
        summary = {"no_models": True}

        # when
        models = runner_module._extract_models_metrics_for_metrics_json(summary)

        # then
        self.assertEqual(models, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
