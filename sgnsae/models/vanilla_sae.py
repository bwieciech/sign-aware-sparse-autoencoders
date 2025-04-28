from typing import Dict

import einops
import torch
from torch import Tensor

from sgnsae.models import SAEBase


class VanillaSAE(SAEBase):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        device: str,
    ):
        super().__init__(
            n_instances, d_in, d_hidden, resample_latents_threshold, device
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Tensor]:
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError(f"Invalid dimension count of input: {x.ndim}")
        x = x.to(self.device)
        while x.ndim < 3:
            x = x.unsqueeze(0)

        x = (
            einops.einsum(
                self.W_enc,
                x - self.b_dec.unsqueeze(1),
                "n_instances d_hidden d_in, n_instances batch d_in -> batch n_instances d_hidden",
            )
            + self.b_enc
        )
        latents = torch.relu(x)

        W_dec_norm = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)
        x_hat = (
            einops.einsum(
                W_dec_norm,
                latents,
                "n_instances d_in d_hidden, batch n_instances d_hidden -> batch n_instances d_in",
            )
            + self.b_dec
        )
        return {
            "y_pred": x_hat.transpose(0, 1),
            "latents": latents.transpose(0, 1),
        }

    def calculate_losses(
        self, y_true: Tensor, sparsity_coeffs: Tensor, output: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        reconstruction_loss = (output["y_pred"] - y_true).pow(2).mean(dim=(-1, -2))
        sparsity_loss = output["latents"].abs().sum(dim=-1).mean(-1)
        loss = reconstruction_loss + sparsity_coeffs * sparsity_loss
        return {
            "losses": loss,
            "reconstruction_losses": reconstruction_loss,
            "sparsity_losses": sparsity_loss,
        }
