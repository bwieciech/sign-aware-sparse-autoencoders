from typing import Dict, Optional, Sequence

import einops
import torch
from torch import Tensor

from sgnsae.models.sae_base import SAEBase


class JumpReLUSAE(SAEBase):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        device: str = "cpu",
        detach_sparsity_source: bool = True,
    ):
        super().__init__(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            device=device,
            sign_aware=False,
        )
        self.detach_sparsity_source = detach_sparsity_source
        initial_threshold = torch.tensor(1e-3, device=device)
        initial_parameter_value = torch.log(torch.expm1(initial_threshold)).item()
        self.theta_param = torch.nn.Parameter(
            torch.full(
                (n_instances, d_hidden),
                initial_parameter_value,
                device=device,
            )
        )
        self.jump_param = torch.nn.Parameter(
            torch.full(
                (n_instances, d_hidden),
                initial_parameter_value,
                device=device,
            )
        )
        self._initial_parameter_value = initial_parameter_value

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError(f"Invalid dimension count of input: {x.ndim}")
        x = x.to(self.device)
        while x.ndim < 3:
            x = x.unsqueeze(0)

        W_dec_norm = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)
        x_centered = x - self.b_dec.unsqueeze(1)
        pre_activations = (
            einops.einsum(
                x_centered,
                W_dec_norm,
                "n_instances batch d_in, n_instances d_in d_hidden -> batch n_instances d_hidden",
            )
            + self.b_enc
        )

        theta = torch.nn.functional.softplus(self.theta_param)
        jump = torch.nn.functional.softplus(self.jump_param)
        gate = (pre_activations > theta).to(pre_activations.dtype)
        latents = gate * (torch.relu(pre_activations - theta) + jump)
        x_hat = (
            einops.einsum(
                W_dec_norm,
                latents,
                "n_instances d_in d_hidden, batch n_instances d_hidden -> batch n_instances d_in",
            )
            + self.b_dec
        )
        x_hat_transposed = x_hat.transpose(0, 1)
        return {
            "x_hat": x_hat_transposed,
            "y_pred": x_hat_transposed,
            "latents": latents.transpose(0, 1),
            "pre_activations": pre_activations.transpose(0, 1),
        }

    def calculate_losses(
        self, y_true: Tensor, sparsity_coeffs: Tensor, output: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        reconstruction_loss = (output["x_hat"] - y_true).pow(2).mean(dim=(-1, -2))
        pre_activations = output["pre_activations"]
        if self.detach_sparsity_source:
            pre_activations = pre_activations.detach()
        theta = torch.nn.functional.softplus(self.theta_param).unsqueeze(1)
        sparsity_source = torch.relu(pre_activations - theta)
        sparsity_loss = sparsity_source.sum(dim=-1).mean(-1)
        loss = reconstruction_loss + sparsity_coeffs * sparsity_loss
        return {
            "losses": loss,
            "reconstruction_losses": reconstruction_loss,
            "sparsity_losses": sparsity_loss,
        }

    def resample_dead_latents(
        self,
        inactive_latents_count: Tensor,
        generators: Optional[Sequence[torch.Generator]] = None,
    ) -> Tensor:
        resample_latents_mask = super().resample_dead_latents(
            inactive_latents_count, generators=generators
        )
        if int(resample_latents_mask.sum().item()) == 0:
            return resample_latents_mask

        with torch.no_grad():
            self.theta_param[resample_latents_mask] = self._initial_parameter_value
            self.jump_param[resample_latents_mask] = self._initial_parameter_value
        return resample_latents_mask
