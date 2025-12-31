import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sgnsae.experiments.directional_deviation import runner as runner_module
from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationAnomalyConfig,
    DirectionalDeviationDataConfig,
    DirectionalDeviationExperimentConfig,
    DirectionalDeviationModelVariantConfig,
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

    def test_resolve_device_returns_cpu_when_cuda_not_available(self) -> None:
        # given
        with patch.object(runner_module.torch.cuda, "is_available", return_value=False), patch(
            "builtins.print"
        ):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
