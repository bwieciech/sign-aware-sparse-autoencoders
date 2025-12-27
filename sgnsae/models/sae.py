from typing import Dict, Optional

import einops
import torch
from torch import Tensor


from sgnsae.models.sae_base import SAEBase


class SAE(SAEBase):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        sign_aware: bool,
        reset_activation_offsets_on_resample: Optional[bool] = None,
        single_activation_offset: Optional[bool] = None,
        device: str = "cpu",
    ):
        super().__init__(
            n_instances,
            d_in,
            d_hidden,
            resample_latents_threshold,
            device,
            sign_aware=sign_aware,
            reset_activation_offsets_on_resample=reset_activation_offsets_on_resample,
            single_activation_offset=single_activation_offset,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError(f"Invalid dimension count of input: {x.ndim}")
        x = x.to(self.device)
        while x.ndim < 3:
            x = x.unsqueeze(0)

        W_dec_norm = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)

        # Shared-weights projection: t(x) = D^T (x - b_dec), then add encoder bias.
        x = (
            einops.einsum(
                x - self.b_dec.unsqueeze(1),
                W_dec_norm,
                "n_instances batch d_in, n_instances d_in d_hidden -> batch n_instances d_hidden",
            )
            + self.b_enc
        )

        if self.sign_aware:
            if self.single_activation_offset:
                delta = torch.relu(self.act_offset)
                latents = torch.where(
                    x > delta,
                    x - delta,
                    torch.where(x < -delta, x + delta, 0.0),
                )
            else:
                delta_pos = torch.relu(self.act_offset_right)
                delta_neg = torch.relu(self.act_offset_left)
                latents = torch.where(
                    x > delta_pos,
                    x - delta_pos,
                    torch.where(
                        x < -delta_neg,
                        x + delta_neg,
                        0.0,
                    ),
                )
        else:
            latents = torch.relu(x)

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
