"""Common training utilities for sparse autoencoder experiments."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor

from sgnsae.data.toy2d import Toy2DTensorDataset, batched_iterator

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Hyperparameters for the Adam optimizer."""

    learning_rate: float = 1e-3
    beta1: float = 0.0
    beta2: float = 0.999
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration shared across SAE training runs."""

    steps: int
    batch_size: int
    sparsity_coeff: float
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    log_every: int = 100
    resample_check_interval: int = 50
    latent_activity_epsilon: float = 1e-6
    grad_clip: Optional[float] = None
    shuffle: bool = True
    seed: int = 0


@dataclass
class TrainingLogEntry:
    """Single log entry recorded during training."""

    step: int
    loss: float
    reconstruction_loss: float
    sparsity_loss: float
    aux_loss: Optional[float]
    l0: float


@dataclass
class TrainingResult:
    """Return object of :func:`train_model`."""

    model: torch.nn.Module
    history: List[TrainingLogEntry]
    inactivity_counters: Tensor


def _cycle_batches(
    dataset: Toy2DTensorDataset, config: TrainingConfig
) -> Iterable[Dict[str, Tensor]]:
    iterator = batched_iterator(
        dataset,
        config.batch_size,
        shuffle=config.shuffle,
        seed=config.seed,
    )
    return iterator


def _prepare_targets(observations: Tensor, n_instances: int) -> Tensor:
    targets = observations.unsqueeze(0)
    if n_instances > 1:
        targets = targets.expand(n_instances, -1, -1)
    return targets


def _compute_l0(latents: Tensor, threshold: float) -> float:
    active = (latents.abs() > threshold).float().sum(dim=-1)
    return active.mean().item()


def train_model(
    model: torch.nn.Module,
    dataset: Toy2DTensorDataset,
    config: TrainingConfig,
) -> TrainingResult:
    """Train ``model`` on ``dataset`` according to ``config``.

    The function assumes that ``model`` implements the API of
    :class:`sgnsae.models.SAEBase`.
    """

    model.train()
    device = next(model.parameters()).device
    iterator = _cycle_batches(dataset, config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )

    sparsity_coeffs = torch.full(
        (model.n_instances,),
        fill_value=config.sparsity_coeff,
        device=device,
        dtype=torch.float32,
    )
    inactivity_counters = torch.zeros(
        (model.n_instances, model.d_hidden), device=device, dtype=torch.long
    )

    history: List[TrainingLogEntry] = []

    for step in range(1, config.steps + 1):
        batch = next(iterator)
        observations = batch["observation"].to(device)
        targets = _prepare_targets(observations, model.n_instances).to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs: Dict[str, Tensor] = model(observations)
        losses = model.calculate_losses(targets, sparsity_coeffs, outputs)
        loss = losses["losses"].mean()
        loss.backward()

        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        with torch.no_grad():
            if hasattr(model, "W_dec"):
                model.W_dec.data = model.W_dec.data / (
                    model.W_dec.data.norm(dim=1, keepdim=True) + 1e-8
                )

        latents = outputs["latents"].detach()
        l0 = _compute_l0(latents, config.latent_activity_epsilon)

        if step % config.log_every == 0 or step == 1 or step == config.steps:
            aux_loss = losses.get("aux_loss")
            aux_value = aux_loss.mean().item() if aux_loss is not None else None
            entry = TrainingLogEntry(
                step=step,
                loss=loss.item(),
                reconstruction_loss=losses["reconstruction_losses"].mean().item(),
                sparsity_loss=losses["sparsity_losses"].mean().item(),
                aux_loss=aux_value,
                l0=l0,
            )
            history.append(entry)
            LOGGER.debug("step=%d loss=%.6f l0=%.3f", step, entry.loss, entry.l0)

        # Update inactivity counters used for dead latent resampling.
        active_mask = (latents.abs() > config.latent_activity_epsilon).any(dim=1)
        inactivity_counters[~active_mask] += 1
        inactivity_counters[active_mask] = 0

        if step % config.resample_check_interval == 0:
            with torch.no_grad():
                resampled_mask = model.resample_dead_latents(inactivity_counters)
                if resampled_mask.any():
                    inactivity_counters[resampled_mask] = 0

    return TrainingResult(
        model=model,
        history=history,
        inactivity_counters=inactivity_counters,
    )


def detect_device(preferred: Optional[str] = None) -> str:
    """Return the device string to use for training.

    Args:
        preferred: Optional explicit device string (e.g. ``"cuda:1"``).

    Returns:
        A device string. If ``preferred`` is provided it is returned directly;
        otherwise the function chooses ``"cuda"`` when CUDA is available and
        ``"cpu"`` otherwise.
    """

    if preferred is not None:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = [
    "OptimizerConfig",
    "TrainingConfig",
    "TrainingLogEntry",
    "TrainingResult",
    "detect_device",
    "train_model",
]
