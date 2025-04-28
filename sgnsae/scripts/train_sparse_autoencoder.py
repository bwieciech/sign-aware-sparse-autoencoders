import dataclasses
import json
import os
import sys
from argparse import Namespace, ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Optional, Any, Dict, List

import torch
import wandb
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from wandb.apis.public import Run

from sgnsae.config import TrainingConfig
from sgnsae.models import PosNegSAE, GatedSAE, VanillaSAE


def parse_arguements() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--wandb-key",
        required=False,
        type=str,
    )
    return parser.parse_args()


def get_wandb_run(config: TrainingConfig, wandb_key: str) -> Optional[Run]:
    return (
        wandb.init(
            project=config.wandb_project,
            config=dataclasses.asdict(config),
        )
        if wandb_key is not None
        else None
    )


def instantiate_dataloader(
    config: TrainingConfig,
    embeddings: Dict[str, List[float]],
    device: str,
) -> DataLoader:
    dataset = TensorDataset(torch.tensor(list(embeddings.values()), device=device))
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    return dataloader


def instantiate_model(
    config: TrainingConfig, d_in: int, steps_per_epoch: int, device: str
) -> torch.nn.Module:
    if config.sae_class == VanillaSAE.__name__:
        return VanillaSAE(
            n_instances=config.sparsity_coeffs_steps,
            d_in=d_in,
            d_hidden=config.d_hidden,
            resample_latents_threshold=int(
                steps_per_epoch * config.resample_latents_epochs
            ),
            device=device,
        )
    if config.sae_class == PosNegSAE.__name__:
        return PosNegSAE(
            n_instances=config.sparsity_coeffs_steps,
            d_in=d_in,
            d_hidden=config.d_hidden,
            resample_latents_threshold=int(
                steps_per_epoch * config.resample_latents_epochs
            ),
            reset_activation_offsets_on_resample=config.reset_activation_offsets_on_resample,
            allow_activation_discontinuity=config.allow_activation_discontinuity,
            device=device,
        )
    if config.sae_class == GatedSAE.__name__:
        return GatedSAE(
            n_instances=config.sparsity_coeffs_steps,
            d_in=d_in,
            d_hidden=config.d_hidden,
            resample_latents_threshold=int(
                steps_per_epoch * config.resample_latents_epochs
            ),
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
    mean_active_latents: float,
) -> None:
    s = (
        f"Epoch: {epoch + 1}/{total_epochs}, Step: {step + 1}/{total_steps}, "
        f"Loss: {training_loss:.6f}, Reconstruction Loss: {training_reconstruction_loss:.6f}, "
        f"Sparsity Loss: {training_sparsity_loss:.6f}, Mean Active Latents: {mean_active_latents:.6f}"
    )
    sys.stdout.write(f"\r{s}")
    sys.stdout.flush()


def log_metrics(
    metrics: Dict[str, Any],
    sparsity_coeffs: Tensor,
    losses_dict: Dict[str, Tensor],
    mean_active_latents_per_instance: Tensor,
    epoch: int,
    epoch_step: int,
    wandb_run: Optional[Run],
) -> None:
    for loss_name, loss_values in losses_dict.items():
        for loss_value, sparsity_coeff in zip(
            loss_values.cpu().detach().numpy(), sparsity_coeffs.cpu().numpy()
        ):
            metrics[loss_name].append(
                {
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                    "value": loss_value.item(),
                    "sparsity_coeff": sparsity_coeff.item(),
                }
            )
    for active_latents, sparsity_coeff in zip(
        mean_active_latents_per_instance.cpu().detach().numpy(),
        sparsity_coeffs.cpu().numpy(),
    ):
        metrics["active_latents"].append(
            {
                "epoch": epoch,
                "epoch_step": epoch_step,
                "value": active_latents.item(),
                "sparsity_coeff": sparsity_coeff.item(),
            }
        )
    if wandb_run is not None:
        wandb_run.log(
            {
                "epoch": epoch,
                "epoch_step": epoch_step,
                "active_latents": mean_active_latents_per_instance.mean(),
                **{
                    f"{loss_name.removesuffix('es')}": loss_values.mean()
                    for loss_name, loss_values in losses_dict.items()
                },
            }
        )


def print_epoch_summary(
    epoch: int,
    epoch_mean_losses: Dict[str, float],
) -> None:
    epoch_header = f"==================== EPOCH {epoch + 1} ===================="
    print(f"\n{epoch_header}")
    print(f"Loss: {epoch_mean_losses['losses']};")
    print(f"Reconstruction Loss: {epoch_mean_losses['reconstruction_losses']};")
    print(f"Sparsity Loss: {epoch_mean_losses['sparsity_losses']};")
    print("=" * len(epoch_header))


def run_training_step(
    model: torch.nn.Module,
    batch: Tensor,
    optimizer: torch.optim.Optimizer,
    sparsity_coeffs: Tensor,
    metrics: Dict[str, Any],
    epoch: int,
    num_epochs: int,
    step: int,
    steps_per_epoch: int,
    inactive_latents_count: Tensor,
    wandb_run: Optional[Run],
) -> Tensor:
    optimizer.zero_grad()
    output = model(batch)
    losses_dict = model.calculate_losses(batch, sparsity_coeffs, output)
    losses_dict["losses"].sum().backward()
    optimizer.step()

    latents_active = output["latents"].abs().detach() > 1e-8
    inactive_latents_count = torch.where(
        latents_active.any(dim=1), 0, inactive_latents_count + 1
    )
    resampled_latents_mask = model.resample_dead_latents(inactive_latents_count)
    if wandb_run is not None:
        wandb_run.log(
            {
                "epoch": epoch,
                "resampled_latents_count": resampled_latents_mask.sum(),
            }
        )
    inactive_latents_count = torch.where(
        resampled_latents_mask, 0, inactive_latents_count
    )
    mean_active_latents_per_instance = latents_active.sum(dim=-1).float().mean(dim=-1)
    log_metrics(
        metrics,
        sparsity_coeffs,
        losses_dict,
        mean_active_latents_per_instance,
        epoch,
        step,
        wandb_run,
    )
    print_progress(
        epoch,
        num_epochs,
        step,
        steps_per_epoch,
        losses_dict["losses"].mean().item(),
        losses_dict["reconstruction_losses"].mean().item(),
        losses_dict["sparsity_losses"].mean().item(),
        mean_active_latents_per_instance.mean().item(),
    )
    return losses_dict, inactive_latents_count


def persist_metadata(
    metadata: Dict[str, Any], config: TrainingConfig, wandb_run: Optional[Run]
) -> None:
    os.makedirs(config.output_path, exist_ok=True)
    metadata_filename = (
        "metadata.json" if wandb_run is None else f"metadata-{wandb_run.name}.json"
    )
    metadata_path = os.path.join(config.output_path, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    if wandb_run is not None:
        artifact = wandb.Artifact(name=Path(metadata_path).stem, type="metrics")
        artifact.add_file(local_path=metadata_path)
        wandb_run.log_artifact(artifact)


def persist_metrics(
    metrics: Dict[str, Any], config: TrainingConfig, wandb_run: Optional[Run]
) -> None:
    os.makedirs(config.output_path, exist_ok=True)
    metrics_filename = (
        "metrics.json" if wandb_run is None else f"metrics-{wandb_run.name}.json"
    )
    metrics_path = os.path.join(config.output_path, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    if wandb_run is not None:
        artifact = wandb.Artifact(name=Path(metrics_path).stem, type="metrics")
        artifact.add_file(local_path=metrics_path)
        wandb_run.log_artifact(artifact)


def persist_model(
    model: torch.nn.Module, config: TrainingConfig, wandb_run: Optional[Run]
) -> None:
    os.makedirs(config.output_path, exist_ok=True)
    model_filename = "model.pt" if wandb_run is None else f"model-{wandb_run.name}.pt"
    model_path = os.path.join(config.output_path, model_filename)
    torch.save(model, model_path)
    if wandb_run is not None:
        artifact = wandb.Artifact(name=Path(model_path).stem, type="model")
        artifact.add_file(local_path=os.path.join(config.output_path, model_filename))
        wandb_run.log_artifact(artifact)


def train_sparse_autoencoder(
    config: TrainingConfig,
    wandb_key: Optional[str],
) -> None:
    if wandb_key is not None:
        wandb.login(key=wandb_key)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    metadata = {"config": dataclasses.asdict(config)}
    wandb_run = get_wandb_run(config, wandb_key)

    with open(config.embeddings_path, "r") as f:
        embeddings_file = json.load(f)
    if "embeddings" not in embeddings_file is None:
        raise ValueError(
            f"Invalid embeddings file: {config.embeddings_path}. "
            "Expected a JSON file with an 'embeddings' key."
        )
    metadata["embedding_model_id"] = embeddings_file.get("embedding_model_id")
    metadata["dataset"] = embeddings_file.get("dataset")
    persist_metadata(metadata, config, wandb_run)

    sparsity_coeffs = torch.logspace(
        config.sparsity_coeffs_min_exponent,
        config.sparsity_coeffs_max_exponent,
        config.sparsity_coeffs_steps,
        device=device,
    )

    dataloader = instantiate_dataloader(config, embeddings_file["embeddings"], device)

    metrics = defaultdict(list)
    inactive_latents_count = torch.zeros(
        config.sparsity_coeffs_steps, config.d_hidden, device=device
    )
    steps_per_epoch = len(dataloader)
    model = instantiate_model(
        config,
        d_in=dataloader.dataset[0][0].numel(),
        steps_per_epoch=steps_per_epoch,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, betas=config.adam_betas
    )
    best_loss = float("inf")
    epochs_since_best_loss = 0
    for epoch in range(config.num_epochs):
        model = model.train()
        epoch_total_losses = defaultdict(int)
        for training_step, (batch,) in enumerate(dataloader):
            losses_dict, inactive_latents_count = run_training_step(
                model,
                batch,
                optimizer,
                sparsity_coeffs,
                metrics,
                epoch,
                config.num_epochs,
                training_step,
                steps_per_epoch,
                inactive_latents_count,
                wandb_run,
            )
            for loss_name, loss_values in losses_dict.items():
                epoch_total_losses[loss_name] += loss_values.mean().item()
        epoch_mean_losses = {
            name: value / steps_per_epoch for name, value in epoch_total_losses.items()
        }
        print_epoch_summary(epoch, epoch_mean_losses)
        if config.early_stopping_patience is not None:
            min_improvement = config.early_stopping_min_delta or 0
            if best_loss - epoch_mean_losses["losses"] >= min_improvement:
                best_loss = epoch_mean_losses["losses"]
                epochs_since_best_loss = 0
            else:
                epochs_since_best_loss += 1
                if epochs_since_best_loss > config.early_stopping_patience:
                    print(
                        f"Early stopping triggered after {epochs_since_best_loss} epochs without improvement defined by"
                        f" min_improvement = {min_improvement}"
                    )
                    break

    persist_metrics(metrics, config, wandb_run)
    persist_model(model, config, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_arguements()
    with open(args.config_path, "r") as f:
        config_json = json.load(f)
    print(f"Running training script with config: {config_json}")
    config = TrainingConfig(**config_json)
    train_sparse_autoencoder(config, args.wandb_key)
