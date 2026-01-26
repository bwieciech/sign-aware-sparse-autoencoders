import unittest
from typing import Dict, Optional

import torch

from sgnsae.models import GatedSAE, SAE, SignedSoftThresholdSAE


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
    ) -> None:
        loss_keys = ["losses", "reconstruction_losses", "sparsity_losses"]
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

