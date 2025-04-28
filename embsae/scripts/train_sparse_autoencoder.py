import json
import os
import sys
from argparse import Namespace, ArgumentParser
from collections import defaultdict
from typing import Optional, Any, Dict, Tuple, List

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from embsae.config import SAEConfig
from embsae.models import PosNegSAE
from embsae.models.vanilla_sae import VanillaSAE


def parse_arguements() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
    )
    return parser.parse_args()


def instantiate_dataloaders(
    config: SAEConfig,
    embeddings: Dict[str, List[float]],
    device: str,
) -> Tuple[DataLoader, DataLoader]:
    training_embeddings = torch.tensor(
        [v for k, v in embeddings.items() if "/train/" in k], device=device
    )
    validation_embeddings = torch.tensor(
        [v for k, v in embeddings.items() if "/test/" in k], device=device
    )

    training_dataset = TensorDataset(training_embeddings)
    validation_dataset = TensorDataset(validation_embeddings)

    training_dataloader = DataLoader(
        training_dataset, batch_size=config.train_batch_size, shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=config.validation_batch_size, shuffle=False
    )
    return training_dataloader, validation_dataloader


def instantiate_model(
    config: SAEConfig, d_in: int, training_steps_per_epoch: int, device: str
) -> torch.nn.Module:
    if config.sae_class == "VanillaSAE":
        return VanillaSAE(
            n_instances=config.sparsity_coeffs_steps,
            d_in=d_in,
            d_hidden=config.d_hidden,
            resample_latents_threshold=int(
                training_steps_per_epoch * config.resample_latents_epochs
            ),
            device=device,
        )
    if config.sae_class == "PosNegSAE":
        return PosNegSAE(
            n_instances=config.sparsity_coeffs_steps,
            d_in=d_in,
            d_hidden=config.d_hidden,
            resample_latents_threshold=int(
                training_steps_per_epoch * config.resample_latents_epochs
            ),
            allow_activation_discontinuity=config.allow_activation_discontinuity,
            device=device,
        )
    raise ValueError(f"Unknown SAE class: {config.sae_class}")


def print_progress(
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    training_loss: float,
    training_reconstruction_loss: float,
    training_sparsity_loss: float,
    validation_loss: Optional[float] = None,
    validation_reconstruction_loss: Optional[float] = None,
    validation_sparsity_loss: Optional[float] = None,
) -> None:
    s = (
        f"Epoch: {epoch + 1}/{total_epochs}, Step: {step + 1}/{total_steps}, "
        f"Loss: {training_loss:.6f}, Reconstruction Loss: {training_reconstruction_loss:.6f}, "
        f"Sparsity Loss: {training_sparsity_loss:.6f}"
    )
    if validation_loss is not None:
        s += (
            f", Validation Loss: {validation_loss:.6f}, "
            f"Validation Reconstruction Loss: {validation_reconstruction_loss:.6f}, "
            f"Validation Sparsity Loss: {validation_sparsity_loss:.6f}"
        )
    sys.stdout.write(f"\r{s}")
    sys.stdout.flush()


def update_metrics(
    metrics: Dict[str, Any],
    sparsity_coeffs: Tensor,
    losses_dict: Dict[str, Tensor],
    epoch: int,
    epoch_step: int,
    is_training: bool,
) -> None:
    for loss_name, loss_values in losses_dict.items():
        for loss_value, sparsity_coeff in zip(
            loss_values.cpu().detach().numpy(), sparsity_coeffs.numpy()
        ):
            metrics[loss_name].append(
                {
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                    "value": loss_value.item(),
                    "sparsity_coeff": sparsity_coeff.item(),
                    "step_type": "training" if is_training else "validation",
                }
            )


def run_training_step(
    model: torch.nn.Module,
    batch: Tensor,
    optimizer: torch.optim.Optimizer,
    sparsity_coeffs: Tensor,
    metrics: Dict[str, Any],
    epoch: int,
    step: int,
    steps_per_epoch: int,
    val_losses_dict: Dict[str, Any],
    inactive_latents_counter: Tensor,
) -> Tuple[Dict[str, Tensor], Tensor]:
    optimizer.zero_grad()
    output = model(batch)
    losses_dict = model.calculate_losses(batch, sparsity_coeffs, output)
    losses_dict["losses"].sum().backward()
    optimizer.step()

    latents_active = (output["latents"].abs().detach() > 1e-8).any(dim=1)
    inactive_latents_counter = torch.where(
        latents_active, 0, inactive_latents_counter + 1
    )
    resampled_latents_mask = model.resample_dead_latents(inactive_latents_counter)
    inactive_latents_counter = torch.where(
        resampled_latents_mask, 0, inactive_latents_counter
    )
    update_metrics(
        metrics,
        sparsity_coeffs,
        losses_dict,
        epoch,
        step,
        is_training=True,
    )
    print_progress(
        epoch,
        config.num_epochs,
        step,
        steps_per_epoch,
        losses_dict["losses"].mean().item(),
        losses_dict["reconstruction_losses"].mean().item(),
        losses_dict["sparsity_losses"].mean().item(),
        None if "losses" not in val_losses_dict else val_losses_dict["losses"].mean(),
        None
        if "reconstruction_losses" not in val_losses_dict
        else val_losses_dict["reconstruction_losses"].mean(),
        None
        if "sparsity_losses" not in val_losses_dict
        else val_losses_dict["sparsity_losses"].mean(),
    )
    return losses_dict, inactive_latents_counter


def run_validation_step(
    model: torch.nn.Module,
    batch: Tensor,
    sparsity_coeffs: Tensor,
    metrics: Dict[str, Any],
    epoch: int,
    step: int,
    steps_per_epoch: int,
    losses_dict: Dict[str, Any],
) -> Tensor:
    output = model(batch)
    val_losses_dict = model.calculate_losses(batch, sparsity_coeffs, output)
    update_metrics(
        metrics,
        sparsity_coeffs,
        val_losses_dict,
        epoch,
        step,
        is_training=False,
    )
    print_progress(
        epoch,
        config.num_epochs,
        step,
        steps_per_epoch,
        losses_dict["losses"].mean().item(),
        losses_dict["reconstruction_losses"].mean().item(),
        losses_dict["sparsity_losses"].mean().item(),
        val_losses_dict["losses"].mean().item(),
        val_losses_dict["reconstruction_losses"].mean().item(),
        val_losses_dict["sparsity_losses"].mean().item(),
    )
    return val_losses_dict


if __name__ == "__main__":
    args = parse_arguements()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(args.config_path, "r") as f:
        config = json.load(f)
    print(f"Running training script with config: {config}")
    config = SAEConfig(**config)

    with open(config.embeddings_path, "r") as f:
        embeddings = json.load(f)

    sparsity_coeffs = torch.logspace(
        config.sparsity_coeffs_min_exponent,
        config.sparsity_coeffs_max_exponent,
        config.sparsity_coeffs_steps,
        device=device,
    )

    training_dataloader, validation_dataloader = instantiate_dataloaders(
        config, embeddings, device
    )

    metrics = defaultdict(list)
    inactive_latents_counter = torch.zeros(
        config.sparsity_coeffs_steps, config.d_hidden, device=device
    )
    total_steps_per_epoch = len(training_dataloader) + len(validation_dataloader)
    model = instantiate_model(
        config,
        d_in=training_dataloader.dataset[0][0].numel(),
        training_steps_per_epoch=len(training_dataloader),
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, betas=config.adam_betas
    )
    val_losses_dict = {}
    for epoch in range(config.num_epochs):
        model = model.train()
        for training_step, (batch,) in enumerate(training_dataloader):
            losses_dict, inactive_latents_counter = run_training_step(
                model,
                batch,
                optimizer,
                sparsity_coeffs,
                metrics,
                epoch,
                training_step,
                total_steps_per_epoch,
                val_losses_dict,
                inactive_latents_counter,
            )

        model = model.eval()
        with torch.no_grad():
            for validation_step, (batch,) in enumerate(validation_dataloader):
                val_losses_dict = run_validation_step(
                    model,
                    batch,
                    sparsity_coeffs,
                    metrics,
                    epoch,
                    training_step + validation_step,
                    total_steps_per_epoch,
                    losses_dict,
                )

    os.makedirs(config.output_path, exist_ok=True)
    with open(os.path.join(config.output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f)
