import unittest
from typing import Dict, Optional

import torch

from sgnsae.experiments.llm_benchmark.training import (
    _collect_threshold_statistics_per_instance,
)
from sgnsae.models import (
    AbsTopKGatedMagnitudeSAE,
    AbsTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    SAE,
    SignedSoftThresholdSAE,
    TopKSAE,
)


class TestModelInstanceGradientIndependence(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.n_instances = 2
        self.batch_size = 4
        self.d_in = 8
        self.d_hidden = 6
        self.resample_latents_threshold = 1_000_000
        self.device = "cpu"

        self.x = torch.randn(self.batch_size, self.d_in, device=self.device)
        self.y_true_by_instance = torch.stack([self.x, self.x + 0.1], dim=0)
        self.y_true_broadcasted = self.x + 0.1
        self.sparsity_coeffs = torch.tensor(
            [0.1, 0.5], dtype=torch.float32, device=self.device
        )

    def _collect_instance_parameter_grads(
        self,
        model: torch.nn.Module,
        y_true: torch.Tensor,
        loss_tensor_key: str,
        instance_index: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)

        output = model(self.x)
        losses_dict = model.calculate_losses(y_true, self.sparsity_coeffs, output)
        if loss_tensor_key not in losses_dict:
            raise KeyError(
                f"{model.__class__.__name__} does not provide '{loss_tensor_key}'."
            )

        loss_tensor = losses_dict[loss_tensor_key]
        if instance_index is None:
            loss_tensor.sum().backward()
        else:
            loss_tensor[instance_index].backward()

        if self.n_instances > 1:
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.ndim == 0 or parameter.shape[0] != self.n_instances:
                    raise AssertionError(
                        "Found shared trainable parameter without leading "
                        f"n_instances dimension: {name} shape={tuple(parameter.shape)}"
                    )

        grads_by_parameter = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if parameter.grad.ndim == 0 or parameter.grad.shape[0] != self.n_instances:
                continue
            grads_by_parameter[name] = parameter.grad.detach().clone()
        return grads_by_parameter

    def _assert_instance_gradients_do_not_mix(
        self,
        model: torch.nn.Module,
        y_true: torch.Tensor,
        loss_tensor_key: str,
    ) -> None:
        # given
        grads_all_instances = self._collect_instance_parameter_grads(
            model,
            y_true,
            loss_tensor_key=loss_tensor_key,
            instance_index=None,
        )

        # when / then
        for instance_index in range(self.n_instances):
            grads_isolated_instance = self._collect_instance_parameter_grads(
                model,
                y_true,
                loss_tensor_key=loss_tensor_key,
                instance_index=instance_index,
            )

            for name, grads in grads_isolated_instance.items():
                torch.testing.assert_close(
                    grads[instance_index], grads_all_instances[name][instance_index]
                )
                for other_instance_index in range(self.n_instances):
                    if other_instance_index == instance_index:
                        continue
                    torch.testing.assert_close(
                        grads[other_instance_index],
                        torch.zeros_like(grads[other_instance_index]),
                    )

    def _assert_model_gradients_do_not_mix_for_loss_components(
        self,
        model: torch.nn.Module,
        y_true: torch.Tensor,
        include_aux_loss: bool,
        include_sparsity_loss: bool = True,
    ) -> None:
        loss_keys = ["losses", "reconstruction_losses"]
        if include_sparsity_loss:
            loss_keys.append("sparsity_losses")
        if include_aux_loss:
            loss_keys.append("aux_loss")
        for loss_key in loss_keys:
            self._assert_instance_gradients_do_not_mix(
                model, y_true=y_true, loss_tensor_key=loss_key
            )

    def test_sae_relu_instances_have_independent_gradients(self) -> None:
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
        )

    def test_sae_sign_aware_single_offset_instances_have_independent_gradients(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
        )

    def test_sae_sign_aware_dual_offset_instances_have_independent_gradients(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
        )

    def test_gated_sae_instances_have_independent_gradients(self) -> None:
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=True,
        )

    def test_gated_sae_sign_aware_instances_have_independent_gradients(
        self,
    ) -> None:
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            symmetric_magnitude=False,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=True,
        )

    def test_gated_sae_sign_aware_symmetric_magnitude_instances_have_independent_gradients(
        self,
    ) -> None:
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            symmetric_magnitude=True,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=True,
        )

    def test_signed_soft_threshold_instances_have_independent_gradients(self) -> None:
        # given
        model = SignedSoftThresholdSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
        )

    def test_sae_relu_instances_have_independent_gradients_with_broadcasted_y_true(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when & then
        self._assert_instance_gradients_do_not_mix(
            model,
            y_true=self.y_true_broadcasted,
            loss_tensor_key="losses",
        )

    def test_jump_relu_instances_have_independent_gradients(self) -> None:
        # given
        model = JumpReLUSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
        )

    def test_topk_instances_have_independent_gradients(self) -> None:
        # given
        model = TopKSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            topk_k=4,
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
            include_sparsity_loss=False,
        )

    def test_abstopk_instances_have_independent_gradients(self) -> None:
        # given
        model = AbsTopKSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            topk_k=4,
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
            include_sparsity_loss=False,
        )

    def test_baseline_parameters_have_leading_n_instances_dimension(self) -> None:
        # given
        models = {
            "SAE": SAE(
                n_instances=self.n_instances,
                d_in=self.d_in,
                d_hidden=self.d_hidden,
                resample_latents_threshold=self.resample_latents_threshold,
                sign_aware=False,
                device=self.device,
            ),
            "GatedSAE": GatedSAE(
                n_instances=self.n_instances,
                d_in=self.d_in,
                d_hidden=self.d_hidden,
                resample_latents_threshold=self.resample_latents_threshold,
                sign_aware=False,
                device=self.device,
            ),
            "SignedSoftThresholdSAE": SignedSoftThresholdSAE(
                n_instances=self.n_instances,
                d_in=self.d_in,
                d_hidden=self.d_hidden,
                resample_latents_threshold=self.resample_latents_threshold,
                device=self.device,
            ),
            "JumpReLUSAE": JumpReLUSAE(
                n_instances=self.n_instances,
                d_in=self.d_in,
                d_hidden=self.d_hidden,
                resample_latents_threshold=self.resample_latents_threshold,
                device=self.device,
            ),
            "TopKSAE": TopKSAE(
                n_instances=self.n_instances,
                d_in=self.d_in,
                d_hidden=self.d_hidden,
                topk_k=4,
                resample_latents_threshold=self.resample_latents_threshold,
                device=self.device,
            ),
            "AbsTopKSAE": AbsTopKSAE(
                n_instances=self.n_instances,
                d_in=self.d_in,
                d_hidden=self.d_hidden,
                topk_k=4,
                resample_latents_threshold=self.resample_latents_threshold,
                device=self.device,
            ),
            "AbsTopKGatedMagnitudeSAE": AbsTopKGatedMagnitudeSAE(
                n_instances=self.n_instances,
                d_in=self.d_in,
                d_hidden=self.d_hidden,
                topk_k=4,
                resample_latents_threshold=self.resample_latents_threshold,
                device=self.device,
            ),
        }

        # when / then
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                for parameter_name, parameter in model.named_parameters():
                    self.assertGreaterEqual(parameter.ndim, 1)
                    self.assertEqual(
                        parameter.shape[0],
                        self.n_instances,
                        msg=(
                            f"{model_name}.{parameter_name} has shape "
                            f"{tuple(parameter.shape)}"
                        ),
                    )

    def test_threshold_statistics_per_instance_do_not_mix_values_between_instances(
        self,
    ) -> None:
        # given
        model = JumpReLUSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )
        theta_values = torch.tensor(
            [
                [0.1] * self.d_hidden,
                [0.3] * self.d_hidden,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        jump_values = torch.tensor(
            [
                [0.2] * self.d_hidden,
                [0.4] * self.d_hidden,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            model.theta_param.copy_(torch.log(torch.expm1(theta_values)))
            model.jump_param.copy_(torch.log(torch.expm1(jump_values)))

        # when
        statistics_by_instance = _collect_threshold_statistics_per_instance(model)

        # then
        self.assertEqual(len(statistics_by_instance), self.n_instances)
        self.assertAlmostEqual(
            statistics_by_instance[0]["theta_mean"], float(theta_values[0, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[0]["theta_p10"], float(theta_values[0, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[0]["theta_p90"], float(theta_values[0, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[1]["theta_mean"], float(theta_values[1, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[1]["theta_p10"], float(theta_values[1, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[1]["theta_p90"], float(theta_values[1, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[0]["jump_mean"], float(jump_values[0, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[0]["jump_p10"], float(jump_values[0, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[0]["jump_p90"], float(jump_values[0, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[1]["jump_mean"], float(jump_values[1, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[1]["jump_p10"], float(jump_values[1, 0].item())
        )
        self.assertAlmostEqual(
            statistics_by_instance[1]["jump_p90"], float(jump_values[1, 0].item())
        )
        self.assertNotEqual(
            statistics_by_instance[0]["theta_mean"],
            statistics_by_instance[1]["theta_mean"],
        )
        self.assertNotEqual(
            statistics_by_instance[0]["jump_mean"],
            statistics_by_instance[1]["jump_mean"],
        )

    def test_collect_instance_parameter_grads_raises_for_missing_loss_key(self) -> None:
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when / then
        with self.assertRaises(KeyError):
            _ = self._collect_instance_parameter_grads(
                model=model,
                y_true=self.y_true_by_instance,
                loss_tensor_key="missing_loss_key",
                instance_index=0,
            )

    def test_collect_instance_parameter_grads_raises_for_shared_parameters(
        self,
    ) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self, n_instances: int, d_hidden: int):
                super().__init__()
                self.instance_parameter = torch.nn.Parameter(
                    torch.ones(n_instances, d_hidden)
                )
                self.shared_parameter = torch.nn.Parameter(torch.ones(d_hidden))

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"x": x}

            def calculate_losses(
                self,
                y_true: torch.Tensor,
                sparsity_coeffs: torch.Tensor,
                output: Dict[str, torch.Tensor],
            ) -> Dict[str, torch.Tensor]:
                del y_true
                del sparsity_coeffs
                del output
                shared_term = self.shared_parameter.sum()
                per_instance_term = self.instance_parameter.pow(2).sum(dim=1)
                return {
                    "losses": per_instance_term + shared_term,
                }

        model = DummyModel(n_instances=self.n_instances, d_hidden=self.d_hidden)

        # when / then
        with self.assertRaises(AssertionError):
            _ = self._collect_instance_parameter_grads(
                model=model,
                y_true=self.y_true_by_instance,
                loss_tensor_key="losses",
                instance_index=0,
            )

    def test_gated_sae_instances_have_independent_gradients_with_broadcasted_y_true(
        self,
    ) -> None:
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when & then
        self._assert_instance_gradients_do_not_mix(
            model,
            y_true=self.y_true_broadcasted,
            loss_tensor_key="losses",
        )

    def test_signed_soft_threshold_instances_have_independent_gradients_with_broadcasted_y_true(
        self,
    ) -> None:
        # given
        model = SignedSoftThresholdSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_instance_gradients_do_not_mix(
            model,
            y_true=self.y_true_broadcasted,
            loss_tensor_key="losses",
        )

    def test_abstopk_gated_magnitude_instances_have_independent_gradients(
        self,
    ) -> None:
        # given
        model = AbsTopKGatedMagnitudeSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            topk_k=4,
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
            include_sparsity_loss=False,
        )

    def test_topk_per_instance_k_instances_have_independent_gradients(self) -> None:
        # given
        model = TopKSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            topk_k=[2, 4],
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
            include_sparsity_loss=False,
        )

    def test_abstopk_per_instance_k_instances_have_independent_gradients(
        self,
    ) -> None:
        # given
        model = AbsTopKSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            topk_k=[2, 4],
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
            include_sparsity_loss=False,
        )

    def test_abstopk_gated_magnitude_per_instance_k_instances_have_independent_gradients(
        self,
    ) -> None:
        # given
        model = AbsTopKGatedMagnitudeSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            topk_k=[2, 4],
            resample_latents_threshold=self.resample_latents_threshold,
            device=self.device,
        )

        # when & then
        self._assert_model_gradients_do_not_mix_for_loss_components(
            model,
            y_true=self.y_true_by_instance,
            include_aux_loss=False,
            include_sparsity_loss=False,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
