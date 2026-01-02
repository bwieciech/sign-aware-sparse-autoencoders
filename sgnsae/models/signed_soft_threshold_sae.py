from typing import Dict

import einops
import torch
from torch import Tensor

from sgnsae.models.sae_base import SAEBase


class SignedSoftThresholdSAE(SAEBase):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        device: str,
    ):
        super().__init__(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            device=device,
            sign_aware=False,
        )
        self.signed_latents = True
        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.zeros(n_instances, d_hidden, d_in, device=device)
            )
        )
        self.theta = torch.nn.Parameter(
            torch.zeros(n_instances, d_hidden, device=device)
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError(f"Invalid dimension count of input: {x.ndim}")
        x = x.to(self.device)
        while x.ndim < 3:
            x = x.unsqueeze(0)

        z = (
            einops.einsum(
                x - self.b_dec.unsqueeze(1),
                self.W_enc,
                "n_instances batch d_in, n_instances d_hidden d_in -> batch n_instances d_hidden",
            )
            + self.b_enc
        )
        theta = torch.relu(self.theta)
        latents = torch.sign(z) * torch.relu(z.abs() - theta)

        x_hat = (
            einops.einsum(
                self.W_dec,
                latents,
                "n_instances d_in d_hidden, batch n_instances d_hidden -> batch n_instances d_in",
            )
            + self.b_dec
        )
        return {
            "x_hat": x_hat.transpose(0, 1),
            "latents": latents.transpose(0, 1),
        }

    def calculate_losses(
        self, y_true: Tensor, sparsity_coeffs: Tensor, output: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        reconstruction_loss = (output["x_hat"] - y_true).pow(2).mean(dim=(-1, -2))
        sparsity_loss = output["latents"].abs().sum(dim=-1).mean(-1)
        loss = reconstruction_loss + sparsity_coeffs * sparsity_loss
        return {
            "losses": loss,
            "reconstruction_losses": reconstruction_loss,
            "sparsity_losses": sparsity_loss,
        }

    def resample_dead_latents(self, inactive_latents_count: Tensor) -> Tensor:
        resample_latents_mask = super().resample_dead_latents(inactive_latents_count)
        resample_latents_count = int(resample_latents_mask.sum().item())
        if resample_latents_count == 0:
            return resample_latents_mask

        new_weights = torch.rand(resample_latents_count, self.d_in, device=self.device)
        new_weights = (
            1e-2 * new_weights / (new_weights.norm(dim=1, keepdim=True) + 1e-8)
        )
        self.W_enc.data[resample_latents_mask] = new_weights
        self.theta.data[resample_latents_mask] = 0.0
        return resample_latents_mask
