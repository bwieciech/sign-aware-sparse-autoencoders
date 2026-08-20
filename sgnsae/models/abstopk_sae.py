from typing import Dict, List, Optional, Union

import einops
import torch
from torch import Tensor

from sgnsae.models.topk_utils import (
    deterministic_topk_mask,
    deterministic_topk_mask_per_instance,
    validate_and_prepare_topk_k,
)
from sgnsae.models.sae_base import SAEBase


class AbsTopKSAE(SAEBase):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        topk_k: Union[int, List[int]],
        resample_latents_threshold: int,
        device: str = "cpu",
    ):
        super().__init__(
            n_instances=n_instances,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            device=device,
            sign_aware=False,
        )
        resolved = validate_and_prepare_topk_k(topk_k, n_instances, d_hidden, device)
        self.topk_k: int = resolved[0]
        self.topk_k_per_instance: Optional[List[int]] = resolved[1]
        self._topk_k_tensor: Optional[Tensor] = resolved[2]
        self.signed_latents = True

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
        scores = pre_activations.detach().abs().float()
        if self._topk_k_tensor is not None:
            mask = deterministic_topk_mask_per_instance(
                scores, self._topk_k_tensor.to(device=scores.device)
            ).to(pre_activations.dtype)
        else:
            mask = deterministic_topk_mask(scores, self.topk_k).to(
                pre_activations.dtype
            )
        latents = pre_activations * mask
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
        }

    def calculate_losses(
        self, y_true: Tensor, sparsity_coeffs: Tensor, output: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        reconstruction_loss = (output["x_hat"] - y_true).pow(2).mean(dim=(-1, -2))
        sparsity_loss = torch.zeros_like(reconstruction_loss)
        return {
            "losses": reconstruction_loss,
            "reconstruction_losses": reconstruction_loss,
            "sparsity_losses": sparsity_loss,
        }
