from typing import Dict, Optional

import einops
import torch
from torch import Tensor

from sgnsae.models import SAEBase


class GatedSAE(SAEBase):
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
        self.r_mag = torch.nn.Parameter(
            torch.zeros(n_instances, d_hidden, device=device)
        )
        self.b_mag = torch.nn.Parameter(
            torch.zeros(n_instances, d_hidden, device=device)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Tensor]:
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError(f"Invalid dimension count of input: {x.ndim}")
        x = x.to(self.device)
        while x.ndim < 3:
            x = x.unsqueeze(0)

        W_mag = einops.einsum(
            self.r_mag.exp(),
            self.W_enc,
            "n_instances d_hidden, n_instances d_hidden d_in -> n_instances d_hidden d_in",
        )
        magnitude_input = einops.einsum(
            W_mag,
            x - self.b_dec.unsqueeze(1),
            "n_instances d_hidden d_in, n_instances batch d_in -> batch n_instances d_hidden",
        )
        if self.sign_aware:
            magnitude_input = magnitude_input.abs()
        f_mag = torch.relu(magnitude_input + self.b_mag)
        pi_gate = (
            einops.einsum(
                self.W_enc,
                x - self.b_dec.unsqueeze(1),
                "n_instances d_hidden d_in, n_instances batch d_in -> batch n_instances d_hidden",
            )
            + self.b_enc
        )
        if self.sign_aware:
            f_gate = torch.where(
                pi_gate
                > torch.relu(
                    self.act_offset
                    if self.single_activation_offset
                    else self.act_offset_right
                ),
                1.0,
                torch.where(
                    pi_gate
                    < -torch.relu(
                        self.act_offset
                        if self.single_activation_offset
                        else self.act_offset_left
                    ),
                    -1.0,
                    0.0,
                ),
            ).float()
        else:
            f_gate = (pi_gate > 0).float()
        latents = f_gate * f_mag
        W_dec_norm = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)
        x_hat = (
            einops.einsum(
                W_dec_norm,
                latents,
                "n_instances d_in d_hidden, batch n_instances d_hidden -> batch n_instances d_in",
            )
            + self.b_dec
        )
        x_hat_frozen = (
            einops.einsum(
                W_dec_norm.detach(),
                torch.relu(pi_gate),
                "n_instances d_in d_hidden, batch n_instances d_hidden -> batch n_instances d_in",
            )
            + self.b_dec.detach()
        )
        return {
            "x_hat": x_hat.transpose(0, 1),
            "pi_gate": pi_gate.transpose(0, 1),
            "x_hat_frozen": x_hat_frozen.transpose(0, 1),
            "latents": latents.transpose(0, 1),
        }

    def calculate_losses(
        self, y_true: Tensor, sparsity_coeffs: Tensor, output: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        reconstruction_loss = (output["x_hat"] - y_true).pow(2).mean(dim=(-1, -2))
        sparsity_source = (
            output["pi_gate"].abs() if self.sign_aware else torch.relu(output["pi_gate"])
        )
        sparsity_loss = sparsity_source.sum(dim=-1).mean(-1)
        aux_loss = (output["x_hat_frozen"] - y_true).pow(2).mean(dim=(-1, -2))
        loss = reconstruction_loss + sparsity_coeffs * sparsity_loss + aux_loss
        return {
            "losses": loss,
            "reconstruction_losses": reconstruction_loss,
            "sparsity_losses": sparsity_loss,
            "aux_loss": aux_loss,
        }
