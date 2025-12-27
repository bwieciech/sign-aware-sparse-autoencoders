from typing import Dict, Optional

import einops
import torch
from torch import Tensor


from sgnsae.models.sae_base import SAEBase


class GatedSAE(SAEBase):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        sign_aware: bool,
        symmetric_magnitude: bool = False,
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
        if symmetric_magnitude and not sign_aware:
            raise ValueError("symmetric_magnitude=True requires sign_aware=True.")
        self.symmetric_magnitude = symmetric_magnitude
        # Gate parameters for paper form: pi = alpha * t + beta
        # We store log(alpha) for stability and positivity.
        self.log_alpha_gate = torch.nn.Parameter(
            torch.zeros(self.n_instances, self.d_hidden, device=self.device)
        )

        self.r_mag = torch.nn.Parameter(
            torch.zeros(n_instances, d_hidden, device=device)
        )
        if self.sign_aware and not self.symmetric_magnitude:
            self.r_mag_neg = torch.nn.Parameter(
                torch.zeros(self.n_instances, self.d_hidden, device=self.device)
            )
        self.b_mag = torch.nn.Parameter(
            torch.zeros(n_instances, d_hidden, device=device)
        )

        self._exp_clamp = 20.0

    def forward(self, x: torch.Tensor) -> Dict[str, Tensor]:
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError(f"Invalid dimension count of input: {x.ndim}")
        x = x.to(self.device)
        while x.ndim < 3:
            x = x.unsqueeze(0)

        W_dec_norm = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)
        x_centered = x - self.b_dec.unsqueeze(1)

        # Shared projection: t = D^T (x - b_dec)
        t = einops.einsum(
            x_centered,
            W_dec_norm,
            "n_instances batch d_in, n_instances d_in d_hidden -> batch n_instances d_hidden",
        )

        # Gate logits pi = alpha * t + beta, but detach t so gate/aux gradients don't flow into D.
        alpha = torch.exp(
            self.log_alpha_gate.clamp(min=-self._exp_clamp, max=self._exp_clamp)
        )
        pi_gate = alpha * t.detach() + self.b_enc

        # Magnitude scales
        scale_pos = torch.exp(self.r_mag.clamp(min=-self._exp_clamp, max=self._exp_clamp))
        if self.sign_aware:
            if self.single_activation_offset:
                delta_pos = delta_neg = torch.relu(self.act_offset)
            else:
                delta_pos = torch.relu(self.act_offset_right)
                delta_neg = torch.relu(self.act_offset_left)

            s = torch.where(
                pi_gate > delta_pos,
                1.0,
                torch.where(pi_gate < -delta_neg, -1.0, 0.0),
            )

            # Option (2): shared additive bias b_mag in BOTH polarities
            a_pos = torch.relu(scale_pos * t + self.b_mag)
            if self.symmetric_magnitude:
                scale_neg = scale_pos
            else:
                scale_neg = torch.exp(
                    self.r_mag_neg.clamp(min=-self._exp_clamp, max=self._exp_clamp)
                )
            a_neg = -torch.relu(-scale_neg * t + self.b_mag)

            latents = torch.where(
                s > 0, a_pos, torch.where(s < 0, a_neg, torch.zeros_like(a_pos))
            )

            gate_cont = torch.relu(pi_gate - delta_pos) - torch.relu(
                -pi_gate - delta_neg
            )
        else:
            latents = (pi_gate > 0).float() * torch.relu(scale_pos * t + self.b_mag)
            gate_cont = torch.relu(pi_gate)

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
                gate_cont,
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

        pi = output["pi_gate"]
        if self.sign_aware:
            if self.single_activation_offset:
                delta_pos = torch.relu(self.act_offset).unsqueeze(1)
                delta_neg = delta_pos
            else:
                delta_pos = torch.relu(self.act_offset_right).unsqueeze(1)
                delta_neg = torch.relu(self.act_offset_left).unsqueeze(1)
            sparsity_source = torch.relu(pi - delta_pos) + torch.relu(-pi - delta_neg)
        else:
            sparsity_source = torch.relu(pi)

        sparsity_loss = sparsity_source.sum(dim=-1).mean(-1)
        aux_loss = (output["x_hat_frozen"] - y_true).pow(2).mean(dim=(-1, -2))

        loss = reconstruction_loss + sparsity_coeffs * sparsity_loss + aux_loss
        return {
            "losses": loss,
            "reconstruction_losses": reconstruction_loss,
            "sparsity_losses": sparsity_loss,
            "aux_loss": aux_loss,
        }
