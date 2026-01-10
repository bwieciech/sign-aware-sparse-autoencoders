import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DExperimentConfig,
    Toy2DModelVariantConfig,
)
from sgnsae.models import GatedSAE, SAE


def _effective_sparsity_coeff(
    base_coeff: float,
    epoch_index: int,
    warmup_epochs: int,
) -> float:
    if warmup_epochs <= 0:
        return float(base_coeff)
    t = min(1.0, float(epoch_index + 1) / float(warmup_epochs))
    return float(base_coeff) * t


def _instantiate_model(
    model_config: Toy2DModelVariantConfig,
    d_in: int,
    steps_per_epoch: int,
    device: str,
) -> torch.nn.Module:
    resample_latents_threshold = int(
        max(1.0, float(steps_per_epoch) * float(model_config.resample_latents_epochs))
    )
    if model_config.sae_class == SAE.__name__:
        return SAE(
            n_instances=1,
            d_in=d_in,
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
            d_in=d_in,
            d_hidden=model_config.d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            sign_aware=model_config.sign_aware,
            symmetric_magnitude=model_config.symmetric_magnitude,
            reset_activation_offsets_on_resample=model_config.reset_activation_offsets_on_resample,
            single_activation_offset=model_config.single_activation_offset,
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
    message = (
        f"[{model_name}] Epoch {epoch + 1}/{num_epochs} "
        f"Step {step + 1}/{total_steps} "
        f"Loss {losses_dict['losses'].mean().item():.6f} "
        f"Recon {losses_dict['reconstruction_losses'].mean().item():.6f} "
        f"Sparsity {losses_dict['sparsity_losses'].mean().item():.6f} "
        f"MeanActive {mean_active_latents:.2f}"
    )
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()


def _log_epoch_summary(
    epoch: int,
    epoch_mean_losses: Dict[str, float],
    model_name: str,
) -> None:
    header = (
        f"[{model_name}] Epoch {epoch + 1} "
        f"Loss {epoch_mean_losses['losses']:.6f} "
        f"Recon {epoch_mean_losses['reconstruction_losses']:.6f} "
        f"Sparsity {epoch_mean_losses['sparsity_losses']:.6f}"
    )
    if "aux_loss" in epoch_mean_losses:
        header = f"{header} Aux {epoch_mean_losses['aux_loss']:.6f}"
    print(f"\n{header}")


def run_training_for_variant(
    experiment_config: Toy2DExperimentConfig,
    data: Dict[str, Tensor],
    model_config: Toy2DModelVariantConfig,
    device: str,
    dataloader_generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    if int(data["x_train"].shape[0]) == 0:
        raise ValueError("x_train must contain at least one sample.")

    dataloader = DataLoader(
        TensorDataset(data["x_train"]),
        batch_size=model_config.batch_size,
        shuffle=True,
        generator=dataloader_generator,
    )
    steps_per_epoch = len(dataloader)

    model = _instantiate_model(
        model_config=model_config,
        d_in=int(data["x_train"].shape[1]),
        steps_per_epoch=steps_per_epoch,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_config.learning_rate, betas=model_config.adam_betas
    )
    inactive_latents_count = torch.zeros(1, model_config.d_hidden, device=device)

    training_metrics: Dict[str, Any] = {
        "epoch": [],
        "loss_total": [],
        "loss_reconstruction": [],
        "loss_sparsity": [],
        "mean_active_latents": [],
        "sparsity_coeff_effective": [],
    }
    terminated_early = False

    for epoch in range(model_config.num_epochs):
        model.train()
        eff_sparsity_coeff = _effective_sparsity_coeff(
            model_config.sparsity_coeff,
            epoch_index=epoch,
            warmup_epochs=model_config.sparsity_coeff_warmup_epochs,
        )
        sparsity_coeffs = torch.tensor(
            [eff_sparsity_coeff], device=device, dtype=torch.float32
        )
        epoch_totals = defaultdict(float)
        epoch_active_latents_total = 0.0
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
            epoch_active_latents_total += mean_active_latents

            for loss_name, loss_values in losses_dict.items():
                epoch_totals[loss_name] += loss_values.mean().item()

            if experiment_config.verbose and (
                (step + 1) % experiment_config.log_every_n_steps == 0
                or step + 1 == steps_per_epoch
            ):
                _log_step_progress(
                    epoch=epoch,
                    num_epochs=model_config.num_epochs,
                    step=step,
                    total_steps=steps_per_epoch,
                    losses_dict=losses_dict,
                    mean_active_latents=mean_active_latents,
                    model_name=model_config.name,
                )

        if terminated_early:
            print(
                f"\n[{model_config.name}] NaN detected in losses at epoch {epoch + 1}; terminating early."
            )
            break

        epoch_mean_losses = {
            name: total / steps_per_epoch for name, total in epoch_totals.items()
        }
        epoch_mean_active_latents = epoch_active_latents_total / float(steps_per_epoch)

        if experiment_config.verbose and (
            (epoch + 1) % experiment_config.log_every_n_epochs == 0
            or epoch + 1 == model_config.num_epochs
        ):
            _log_epoch_summary(
                epoch=epoch,
                epoch_mean_losses=epoch_mean_losses,
                model_name=model_config.name,
            )

        training_metrics["epoch"].append(epoch + 1)
        training_metrics["loss_total"].append(epoch_mean_losses["losses"])
        training_metrics["loss_reconstruction"].append(
            epoch_mean_losses["reconstruction_losses"]
        )
        training_metrics["loss_sparsity"].append(epoch_mean_losses["sparsity_losses"])
        training_metrics["mean_active_latents"].append(epoch_mean_active_latents)
        training_metrics["sparsity_coeff_effective"].append(eff_sparsity_coeff)
        if "aux_loss" in epoch_mean_losses:
            training_metrics.setdefault("loss_aux", []).append(
                epoch_mean_losses["aux_loss"]
            )

    if terminated_early:
        training_metrics["terminated_early"] = True

    return {"model": model, "training_metrics": training_metrics}


def plot_training_loss(
    training_metrics: Dict[str, Any],
    model_name: str,
    model_dir: str,
) -> None:
    try:
        import matplotlib

        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to plot training loss for Toy Model Geometry Validation."
        ) from exc

    plots_dir = os.path.join(model_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs = training_metrics.get(
        "epoch", list(range(1, len(training_metrics.get("loss_total", [])) + 1))
    )
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, training_metrics["loss_total"], label="Total Loss")
    plt.plot(
        epochs,
        training_metrics["loss_reconstruction"],
        label="Reconstruction Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {model_name}")
    plt.legend()
    plt.grid(axis="y", color="#e0e0e0")
    output_path = os.path.join(plots_dir, f"training_loss_{model_name}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def persist_model_artifacts(
    model: torch.nn.Module,
    training_metrics: Dict[str, Any],
    model_config: Toy2DModelVariantConfig,
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

    plot_training_loss(training_metrics, model_config.name, model_dir)
