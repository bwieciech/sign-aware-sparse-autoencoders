import abc
from typing import Dict

import torch
from torch import Tensor


class SAE(abc.ABC, torch.nn.Module):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        device: str,
    ):
        super().__init__()

        self.n_instances = n_instances
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.resample_latents_threshold = resample_latents_threshold
        self.device = device

    def forward(self, x: torch.Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError()

    def calculate_losses(
        self, y_true: Tensor, sparsity_coeffs: Tensor, output: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        raise NotImplementedError()

    def resample_dead_latents(
        self,
        inactive_latents_count: Tensor,
    ) -> Tensor:
        resample_latents_mask = inactive_latents_count > self.resample_latents_threshold
        resample_latents_count = resample_latents_mask.sum()
        if resample_latents_count == 0:
            return resample_latents_mask

        new_weights = torch.rand(resample_latents_count, self.d_in, device=self.device)
        new_weights = (
            1e-2 * new_weights / (new_weights.norm(dim=-1, keepdim=True) + 1e-8)
        )

        self.W_enc.data[resample_latents_mask] = new_weights
        self.b_enc.data[resample_latents_mask] = torch.zeros(
            resample_latents_count, device=self.device
        )
        self.W_dec.transpose(-1, -2).data[resample_latents_mask] = new_weights

        return resample_latents_mask
