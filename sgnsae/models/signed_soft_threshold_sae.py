from typing import Dict, Optional, Sequence

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
        # NOTE:
        #   We parameterize the soft-threshold as theta = relu(theta_param).
        #   If theta_param is initialized to exactly 0, ReLU's gradient at 0 is 0,
        #   so theta cannot increase during training and the model stays dense.
        #   Initialize slightly positive (mirrors SAEBase's activation offsets).
        self.theta = torch.nn.Parameter(
            1e-3 * torch.ones(n_instances, d_hidden, device=device)
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError(f"Invalid dimension count of input: {x.ndim}")
        x = x.to(self.device)
        while x.ndim < 3:
            x = x.unsqueeze(0)

        W_dec_norm = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)
        x_centered = x - self.b_dec.unsqueeze(1)
        z = torch.einsum("nbd,ndh->bnh", x_centered, W_dec_norm) + self.b_enc
        theta = torch.relu(self.theta)
        latents = torch.sign(z) * torch.relu(z.abs() - theta)

        x_hat = torch.einsum("ndh,bnh->bnd", W_dec_norm, latents) + self.b_dec
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

        # Reset to a small positive value so the threshold remains learnable after
        # resampling. Resetting to 0 would freeze the parameter under the ReLU
        # parameterization used in forward().
        self.theta.data[resample_latents_mask] = 1e-3
        return resample_latents_mask
