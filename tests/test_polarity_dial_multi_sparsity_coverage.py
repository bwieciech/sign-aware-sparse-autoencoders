import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

import torch

from sgnsae.experiments.polarity_dial import plotting as plotting_module
from sgnsae.experiments.polarity_dial import runner as runner_module
from sgnsae.experiments.polarity_dial import training as training_module
from sgnsae.experiments.polarity_dial.config import PolarityDialModelVariantConfig
from sgnsae.models import GatedSAE, SAE


class TestPolarityDialMultiSparsityCoverage(unittest.TestCase):
    def test_copy_instance_0_parameters_skips_non_instance_parameters(self) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 2
                self.instance_param = torch.nn.Parameter(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                )
                self.shared_param = torch.nn.Parameter(torch.tensor([5.0, 6.0, 7.0]))

        model = DummyModel()
        shared_before = model.shared_param.detach().clone()

        # when
        training_module._copy_instance_0_parameters_to_all_instances(model)

        # then
        torch.testing.assert_close(
            model.instance_param.detach(),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        )
        torch.testing.assert_close(model.shared_param.detach(), shared_before)

    def test_slice_trained_model_instance_raises_when_model_missing_n_instances(
        self,
    ) -> None:
        # given
        model = torch.nn.Linear(2, 2)
        model_config = PolarityDialModelVariantConfig(
            name="model",
            sae_class=SAE.__name__,
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

        # when & then
        with self.assertRaises(ValueError):
            _ = training_module.slice_trained_model_instance(
                model=model,
                model_config=model_config,
                instance_index=0,
                input_dim=2,
                steps_per_epoch=1,
                device="cpu",
            )

    def test_slice_trained_model_instance_raises_on_out_of_bounds_index(self) -> None:
        # given
        model = SAE(
            n_instances=2,
            d_in=4,
            d_hidden=2,
            resample_latents_threshold=10,
            sign_aware=False,
            device="cpu",
        )
        model_config = PolarityDialModelVariantConfig(
            name="model",
            sae_class=SAE.__name__,
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

        # when & then
        with self.assertRaises(ValueError):
            _ = training_module.slice_trained_model_instance(
                model=model,
                model_config=model_config,
                instance_index=2,
                input_dim=4,
                steps_per_epoch=1,
                device="cpu",
            )

    def test_slice_trained_model_instance_raises_when_parameter_missing_in_source_model(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=2,
            d_in=4,
            d_hidden=2,
            resample_latents_threshold=10,
            sign_aware=False,
            device="cpu",
        )
        model_config = PolarityDialModelVariantConfig(
            name="model",
            sae_class=GatedSAE.__name__,
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

        # when & then
        with self.assertRaises(ValueError):
            _ = training_module.slice_trained_model_instance(
                model=model,
                model_config=model_config,
                instance_index=0,
                input_dim=4,
                steps_per_epoch=1,
                device="cpu",
            )

    def test_slice_trained_model_instance_copies_non_instance_parameter(self) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self, n_instances: int) -> None:
                super().__init__()
                self.n_instances = n_instances
                self.instance_param = torch.nn.Parameter(
                    torch.arange(float(n_instances) * 2).reshape(n_instances, 2)
                )
                self.shared_param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

        model = DummyModel(n_instances=2)
        model_config = PolarityDialModelVariantConfig(
            name="model",
            sae_class=SAE.__name__,
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

        def fake_instantiate_model(
            model_config: PolarityDialModelVariantConfig,
            n_instances: int,
            input_dim: int,
            steps_per_epoch: int,
            device: str,
        ) -> torch.nn.Module:
            return DummyModel(n_instances=n_instances)

        # when
        with patch.object(
            training_module, "_instantiate_model", fake_instantiate_model
        ):
            sliced = training_module.slice_trained_model_instance(
                model=model,
                model_config=model_config,
                instance_index=1,
                input_dim=4,
                steps_per_epoch=1,
                device="cpu",
            )

        # then
        torch.testing.assert_close(
            sliced.instance_param.detach(),
            model.instance_param.detach()[1:2],
        )
        torch.testing.assert_close(
            sliced.shared_param.detach(),
            model.shared_param.detach(),
        )

    def test_resolve_subrun_count_raises_for_empty_models(self) -> None:
        # given
        models = []

        # when & then
        with self.assertRaises(ValueError):
            _ = runner_module._resolve_subrun_count(models)

    def test_copy_seed_data_exports_returns_when_source_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_run_dir = os.path.join(tmpdir, "seed")
            subrun_dir = os.path.join(tmpdir, "subrun")
            os.makedirs(seed_run_dir, exist_ok=True)
            os.makedirs(subrun_dir, exist_ok=True)

            # when
            runner_module._copy_seed_data_exports(seed_run_dir, subrun_dir)

            # then
            self.assertFalse(os.path.isdir(os.path.join(subrun_dir, "data")))

    def test_training_metrics_for_instance_returns_input_when_not_multi_instance(
        self,
    ) -> None:
        # given
        training_metrics: Dict[str, Any] = {
            "epoch": [1],
            "loss_total": [1.0],
            "loss_reconstruction": [0.5],
            "loss_sparsity": [0.25],
        }

        # when
        result = runner_module._training_metrics_for_instance(
            training_metrics, instance_index=0
        )

        # then
        self.assertIs(result, training_metrics)

    def test_training_metrics_for_instance_includes_terminated_early(self) -> None:
        # given
        training_metrics: Dict[str, Any] = {
            "epoch": [1],
            "loss_total_by_instance": [[1.0], [2.0]],
            "loss_reconstruction_by_instance": [[0.5], [1.0]],
            "loss_sparsity_by_instance": [[0.25], [0.75]],
            "terminated_early": True,
        }

        # when
        result = runner_module._training_metrics_for_instance(
            training_metrics, instance_index=0
        )

        # then
        self.assertTrue(result.get("terminated_early"))

    def test_final_metric_values_raises_for_non_list_by_instance_series(self) -> None:
        # given
        training_metrics = {"loss_reconstruction_by_instance": "bad"}

        # when & then
        with self.assertRaises(TypeError):
            _ = plotting_module._final_metric_values_for_sparsity_coeff_sweep(
                training_metrics=training_metrics,
                scalar_metric_key="loss_reconstruction",
                metric_by_instance_key="loss_reconstruction_by_instance",
                n_instances=2,
            )

    def test_final_metric_values_falls_back_to_scalar_series(self) -> None:
        # given
        training_metrics = {"loss_reconstruction": [0.5]}

        # when
        values = plotting_module._final_metric_values_for_sparsity_coeff_sweep(
            training_metrics=training_metrics,
            scalar_metric_key="loss_reconstruction",
            metric_by_instance_key="loss_reconstruction_by_instance",
            n_instances=3,
        )

        # then
        self.assertEqual(values, [0.5, 0.5, 0.5])

    def test_plot_multi_sparsity_loss_summaries_raises_for_empty_coeff_list(
        self,
    ) -> None:
        # given
        training_metrics: Dict[str, Any] = {}

        # when & then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                plotting_module.plot_multi_sparsity_loss_summaries(
                    training_metrics=training_metrics,
                    model_name="model",
                    seed_run_dir=tmpdir,
                    sparsity_coeff_list=[],
                )

    def test_plot_multi_sparsity_loss_summaries_raises_for_length_mismatch(
        self,
    ) -> None:
        # given
        training_metrics: Dict[str, Any] = {
            "loss_reconstruction_by_instance": [[1.0]],
            "loss_sparsity_by_instance": [[0.5]],
        }

        # when & then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                plotting_module.plot_multi_sparsity_loss_summaries(
                    training_metrics=training_metrics,
                    model_name="model",
                    seed_run_dir=tmpdir,
                    sparsity_coeff_list=[0.1, 0.2],
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
