import abc
from typing import Dict, Optional

import torch.nn
from torch import Tensor



class SAEBase(abc.ABC, torch.nn.Module):
    def __init__(
        self,
        n_instances: int,
        d_in: int,
        d_hidden: int,
        resample_latents_threshold: int,
        device: str,
        sign_aware: bool = False,
        reset_activation_offsets_on_resample: Optional[bool] = None,
        single_activation_offset: Optional[bool] = None,
    ):
        super().__init__()

        self.n_instances = n_instances
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.resample_latents_threshold = resample_latents_threshold
        self.device = device
        self.sign_aware = sign_aware

        if sign_aware:
            if reset_activation_offsets_on_resample is None:
                raise ValueError(
                    "reset_activation_offsets_on_resample must be specified when sign_aware is 'true'"
                )
            if single_activation_offset is None:
                raise ValueError(
                    "single_activation_offset must be specified when sign_aware is 'true'"
                )
            if single_activation_offset:
                self.act_offset = torch.nn.Parameter(
                    1e-3 * torch.ones(n_instances, d_hidden, device=device)
                )
            else:
                self.act_offset_left = torch.nn.Parameter(
                    1e-3 * torch.ones(n_instances, d_hidden, device=device)
                )
                self.act_offset_right = torch.nn.Parameter(
                    1e-3 * torch.ones(n_instances, d_hidden, device=device)
                )
            self.reset_activation_offsets_on_resample = (
                reset_activation_offsets_on_resample
            )
            self.single_activation_offset = single_activation_offset

        self.b_enc = torch.nn.Parameter(
            torch.zeros(n_instances, d_hidden, device=device)
        )
        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.zeros(n_instances, d_in, d_hidden, device=device)
            )
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(n_instances, d_in, device=device))

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
            1e-2 * new_weights / (new_weights.norm(dim=1, keepdim=True) + 1e-8)
        )

        self.b_enc.data[resample_latents_mask] = torch.zeros(
            resample_latents_count, device=self.device
        )
        self.W_dec.transpose(-1, -2).data[resample_latents_mask] = new_weights

        if self.sign_aware and self.reset_activation_offsets_on_resample:
            if self.single_activation_offset:
                self.act_offset.data[resample_latents_mask] = 1e-3 * torch.ones(
                    resample_latents_count, device=self.device
                )
            else:
                self.act_offset_left.data[resample_latents_mask] = 1e-3 * torch.ones(
                    resample_latents_count, device=self.device
                )
                self.act_offset_right.data[resample_latents_mask] = 1e-3 * torch.ones(
                    resample_latents_count, device=self.device
                )

        # Optional: reset any gated-SAE specific per-latent parameters if they exist.
        for name in ("r_mag", "r_mag_neg", "b_mag", "log_alpha_gate"):
            if hasattr(self, name):
                p = getattr(self, name)
                if isinstance(p, torch.nn.Parameter) and p.shape == self.b_enc.shape:
                    p.data[resample_latents_mask] = 0.0

        return resample_latents_mask
