from typing import Tuple

import torch
from torch import Tensor


def calculate_directional_scores_from_one_sided_evidence(
    one_sided_evidence: Tensor,
    evidence_thresholds: Tensor,
    latent_directions: Tensor,
) -> Tuple[Tensor, Tensor]:
    if one_sided_evidence.ndim != 2:
        raise ValueError(
            f"Expected one_sided_evidence.ndim == 2, got {one_sided_evidence.ndim}"
        )
    if evidence_thresholds.ndim != 1:
        raise ValueError(
            f"Expected evidence_thresholds.ndim == 1, got {evidence_thresholds.ndim}"
        )
    if latent_directions.ndim != 1:
        raise ValueError(
            f"Expected latent_directions.ndim == 1, got {latent_directions.ndim}"
        )

    batch_size, num_latents = one_sided_evidence.shape
    if evidence_thresholds.shape[0] != num_latents:
        raise ValueError(
            "Expected evidence_thresholds.shape[0] == one_sided_evidence.shape[1], got "
            f"{evidence_thresholds.shape[0]} vs {num_latents}"
        )
    if latent_directions.shape[0] != num_latents:
        raise ValueError(
            "Expected latent_directions.shape[0] == one_sided_evidence.shape[1], got "
            f"{latent_directions.shape[0]} vs {num_latents}"
        )

    if not torch.all((latent_directions == 1) | (latent_directions == -1)):
        raise ValueError("Expected latent_directions entries to be in {-1, +1}.")

    evidence_excess = torch.relu(one_sided_evidence - evidence_thresholds.unsqueeze(0))

    high_mask = latent_directions > 0
    low_mask = latent_directions < 0

    score_high = (
        evidence_excess[:, high_mask].sum(dim=1)
        if torch.any(high_mask)
        else torch.zeros(batch_size, device=evidence_excess.device)
    )
    score_low = (
        evidence_excess[:, low_mask].sum(dim=1)
        if torch.any(low_mask)
        else torch.zeros(batch_size, device=evidence_excess.device)
    )
    return score_high, score_low

