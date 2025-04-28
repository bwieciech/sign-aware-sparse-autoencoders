from typing import Dict

import einops
import torch
from torch import Tensor

from sgnsae.models import SAEBase


class PosNegSAE(SAEBase):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        reset_activation_offsets_on_resample: bool,
        allow_activation_discontinuity: bool,
        device: str,
    ):
        super().__init__(
            n_instances, d_in, d_hidden, resample_latents_threshold, device
        )
        self.act_offset = torch.nn.Parameter(
            1e-3 * torch.ones(n_instances, d_hidden, device=device)
        )
        self.reset_activation_offsets_on_resample = reset_activation_offsets_on_resample
        self.register_buffer(
            "allow_activation_discontinuity",
            torch.tensor(allow_activation_discontinuity, device=device),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
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
        if self.allow_activation_discontinuity:
            latents = (x.abs() > self.act_offset) * x
        else:
            latents = (x.abs() > self.act_offset) * torch.where(
                x > 0, x - self.act_offset, x + self.act_offset
            )

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

    def resample_dead_latents(
        self,
        inactive_latents_count: Tensor,
    ) -> Tensor:
        resample_latents_mask = super().resample_dead_latents(inactive_latents_count)
        if not self.reset_activation_offsets_on_resample:
            return resample_latents_mask
        self.act_offset.data[resample_latents_mask] = 1e-3 * torch.ones(
            resample_latents_mask.sum(), device=self.device
        )
        return resample_latents_mask
