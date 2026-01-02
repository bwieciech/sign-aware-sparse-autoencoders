import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from sgnsae.experiments.polarity_dial.config import (
    PolarityDialExperimentConfig,
    PolarityDialModelVariantConfig,
)
from sgnsae.models import GatedSAE, SAE, SignedSoftThresholdSAE


def _instantiate_model(
    model_config: PolarityDialModelVariantConfig,
    input_dim: int,
    steps_per_epoch: int,
    device: str,
) -> torch.nn.Module:
    resample_latents_threshold = int(
        max(1.0, steps_per_epoch * model_config.resample_latents_epochs)
    )
    if model_config.sae_class == SAE.__name__:
        return SAE(
            n_instances=1,
            d_in=input_dim,
            d_hidden=model_config.d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            sign_aware=model_config.sign_aware,
            reset_activation_offsets_on_resample=model_config.reset_activation_offsets_on_resample,
            single_activation_offset=model_config.single_activation_offset,
            device=device,
        )
    if model_config.sae_class == GatedSAE.__name__:
        return GatedSAE(
            n_instances=1,
            d_in=input_dim,
            d_hidden=model_config.d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            sign_aware=model_config.sign_aware,
            symmetric_magnitude=model_config.symmetric_magnitude,
            reset_activation_offsets_on_resample=model_config.reset_activation_offsets_on_resample,
            single_activation_offset=model_config.single_activation_offset,
            device=device,
        )
    if model_config.sae_class == SignedSoftThresholdSAE.__name__:
        return SignedSoftThresholdSAE(
            n_instances=1,
            d_in=input_dim,
            d_hidden=model_config.d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            device=device,
        )
    raise ValueError(f"Unknown SAE class: {model_config.sae_class}")


def _log_step_progress(
    epoch: int,
    num_epochs: int,
    step: int,
    total_steps: int,
    losses_dict: Dict[str, Tensor],
    mean_active_latents: float,
    model_name: str,
) -> None:
    s = (
        f"[{model_name}] Epoch {epoch + 1}/{num_epochs} "
        f"Step {step + 1}/{total_steps} "
        f"Loss {losses_dict['losses'].mean().item():.6f} "
        f"Recon {losses_dict['reconstruction_losses'].mean().item():.6f} "
        f"Sparsity {losses_dict['sparsity_losses'].mean().item():.6f} "
        f"MeanActive {mean_active_latents:.2f}"
    )
    sys.stdout.write(f"\r{s}")
    sys.stdout.flush()


def _log_epoch_summary(
    epoch: int,
    epoch_mean_losses: Dict[str, float],
    model_config: PolarityDialModelVariantConfig,
) -> None:
    header = (
        f"[{model_config.name}] Epoch {epoch + 1} "
        f"Loss {epoch_mean_losses['losses']:.6f} "
        f"Recon {epoch_mean_losses['reconstruction_losses']:.6f} "
        f"Sparsity {epoch_mean_losses['sparsity_losses']:.6f}"
    )
    if "aux_loss" in epoch_mean_losses:
        header = f"{header} Aux {epoch_mean_losses['aux_loss']:.6f}"
    print(f"\n{header}")


def run_training_for_variant(
    experiment_config: PolarityDialExperimentConfig,
    data: Dict[str, Tensor],
    model_config: PolarityDialModelVariantConfig,
    device: str,
) -> Dict[str, Any]:
    dataloader = DataLoader(
        TensorDataset(data["x_train"]),
        batch_size=model_config.batch_size,
        shuffle=True,
    )
    steps_per_epoch = len(dataloader)
    model = _instantiate_model(
        model_config, data["x_train"].shape[1], steps_per_epoch, device
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_config.learning_rate, betas=model_config.adam_betas
    )
    sparsity_coeffs = torch.tensor(
        [model_config.sparsity_coeff], device=device, dtype=torch.float32
    )
    inactive_latents_count = torch.zeros(1, model_config.d_hidden, device=device)

    training_metrics: Dict[str, Any] = {
        "epoch": [],
        "loss_total": [],
        "loss_reconstruction": [],
        "loss_sparsity": [],
    }
    terminated_early = False

    for epoch in range(model_config.num_epochs):
        model.train()
        epoch_totals = defaultdict(float)
        for step, (batch,) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(batch)
            losses_dict = model.calculate_losses(batch, sparsity_coeffs, output)
            if torch.isnan(losses_dict["losses"]).any():
                terminated_early = True
                break
            losses_dict["losses"].sum().backward()
            optimizer.step()

            latents_active = output["latents"].abs().detach() > 1e-8
            inactive_latents_count = torch.where(
                latents_active.any(dim=1),
                torch.zeros_like(inactive_latents_count),
                inactive_latents_count + 1,
            )
            resampled_latents_mask = model.resample_dead_latents(inactive_latents_count)
            inactive_latents_count = torch.where(
                resampled_latents_mask,
                torch.zeros_like(inactive_latents_count),
                inactive_latents_count,
            )
            mean_active_latents = latents_active.sum(dim=-1).float().mean().item()

            for loss_name, loss_values in losses_dict.items():
                epoch_totals[loss_name] += loss_values.mean().item()
            if experiment_config.verbose and (
                (step + 1) % experiment_config.log_every_n_steps == 0
                or step + 1 == steps_per_epoch
            ):
                _log_step_progress(
                    epoch,
                    model_config.num_epochs,
                    step,
                    steps_per_epoch,
                    losses_dict,
                    mean_active_latents,
                    model_config.name,
                )
        if terminated_early:
            print(
                f"\n[{model_config.name}] NaN detected in losses at epoch {epoch + 1}; terminating early."
            )
            break
        epoch_mean_losses = {
            name: total / steps_per_epoch for name, total in epoch_totals.items()
        }
        if experiment_config.verbose and (
            (epoch + 1) % experiment_config.log_every_n_epochs == 0
            or epoch + 1 == model_config.num_epochs
        ):
            _log_epoch_summary(epoch, epoch_mean_losses, model_config)

        training_metrics["epoch"].append(epoch + 1)
        training_metrics["loss_total"].append(epoch_mean_losses["losses"])
        training_metrics["loss_reconstruction"].append(
            epoch_mean_losses["reconstruction_losses"]
        )
        training_metrics["loss_sparsity"].append(epoch_mean_losses["sparsity_losses"])
        if "aux_loss" in epoch_mean_losses:
            training_metrics.setdefault("loss_aux", []).append(
                epoch_mean_losses["aux_loss"]
            )

    if terminated_early:
        training_metrics["terminated_early"] = True

    return {"model": model, "training_metrics": training_metrics}


def persist_model_artifacts(
    model: torch.nn.Module,
    training_metrics: Dict[str, Any],
    model_config: PolarityDialModelVariantConfig,
    model_dir: str,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model, model_path)
    model_config_path = os.path.join(model_dir, "model_config.json")
    with open(model_config_path, "w") as f:
        serialized_config = asdict(model_config)
        serialized_config["parameter_count"] = int(
            sum(parameter.numel() for parameter in model.parameters())
        )
        serialized_config["model_class"] = model.__class__.__name__
        if hasattr(model, "d_in"):
            serialized_config["d_in"] = int(getattr(model, "d_in"))
        if hasattr(model, "d_hidden"):
            serialized_config["d_hidden_effective"] = int(getattr(model, "d_hidden"))
        json.dump(serialized_config, f, indent=2)
    training_metrics_path = os.path.join(model_dir, "training_metrics.json")
    with open(training_metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
