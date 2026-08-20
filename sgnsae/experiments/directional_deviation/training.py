import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from sgnsae.experiments.directional_deviation.config import (
    DirectionalDeviationAnomalyConfig,
    DirectionalDeviationDataConfig,
    DirectionalDeviationExperimentConfig,
    DirectionalDeviationModelVariantConfig,
)
from sgnsae.models import GatedSAE, SAE

_TORCH_SEED_MODULUS = 2**63


def _normalize_torch_seed(seed: int) -> int:
    return int(seed % _TORCH_SEED_MODULUS)


def _create_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_normalize_torch_seed(seed))
    return generator


def _build_training_dataloader(
    x_train: torch.Tensor,
    batch_size: int,
    dataloader_seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
        generator=_create_cpu_generator(dataloader_seed),
    )


def _run_optimizer_warmup_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    warmup_steps: int,
    sparsity_coeffs: torch.Tensor,
    dataloader_factory: Optional[Callable[[], DataLoader]] = None,
) -> bool:
    if warmup_steps <= 0:
        return False

    original_learning_rates = [
        float(param_group["lr"]) for param_group in optimizer.param_groups
    ]
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.0
    try:
        steps_completed = 0
        current_dataloader = dataloader
        while steps_completed < warmup_steps:
            steps_this_pass = 0
            for (batch,) in current_dataloader:
                steps_this_pass += 1
                steps_completed += 1
                optimizer.zero_grad()
                output = model(batch)
                losses_dict = model.calculate_losses(batch, sparsity_coeffs, output)
                if not bool(torch.isfinite(losses_dict["losses"]).all()):
                    return True
                losses_dict["losses"].sum().backward()
                optimizer.step()
                if steps_completed >= warmup_steps:
                    return False
            if steps_this_pass == 0:
                raise ValueError("Training dataloader produced zero steps.")
            if dataloader_factory is not None:
                current_dataloader = dataloader_factory()
    finally:
        for param_group, learning_rate in zip(
            optimizer.param_groups, original_learning_rates
        ):
            param_group["lr"] = learning_rate


def _resolve_channels_to_evaluate(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
) -> List[int]:
    if anomaly_config.channels_to_evaluate is None:
        return list(range(data_config.num_channels))
    return list(anomaly_config.channels_to_evaluate)


def _select_subset_indices(
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
    seed: int,
) -> List[int]:
    if anomaly_config.shift_mode != "subset":
        raise ValueError("subset indices are only valid for shift_mode='subset'.")
    if anomaly_config.subset_size is None:
        raise ValueError("subset_size must be provided when shift_mode='subset'.")

    candidate_channels = _resolve_channels_to_evaluate(data_config, anomaly_config)
    if anomaly_config.subset_size > len(candidate_channels):
        raise ValueError(
            "subset_size must not exceed the number of candidate channels; received "
            f"subset_size={anomaly_config.subset_size} and num_candidates={len(candidate_channels)}."
        )

    generator = _create_cpu_generator(seed)
    candidate_tensor = torch.tensor(candidate_channels, dtype=torch.int64, device="cpu")
    perm = torch.randperm(candidate_tensor.numel(), generator=generator, device="cpu")
    chosen = candidate_tensor[perm[: anomaly_config.subset_size]].tolist()
    return sorted(int(value) for value in chosen)


def _prepare_training_tensor(
    x_train: Tensor,
    data_config: DirectionalDeviationDataConfig,
    anomaly_config: DirectionalDeviationAnomalyConfig,
) -> Tuple[Tensor, int, int]:
    anomaly_fraction = float(data_config.train_anomaly_fraction)
    if anomaly_fraction <= 0.0:
        return x_train, 0, _normalize_torch_seed(data_config.seed + 5)

    num_samples = int(x_train.shape[0])
    num_anomalies = int(round(num_samples * anomaly_fraction))
    num_anomalies = min(max(num_anomalies, 0), num_samples)
    if num_anomalies == 0:
        return x_train, 0, _normalize_torch_seed(data_config.seed + 5)

    training_anomaly_seed = _normalize_torch_seed(data_config.seed + 5)
    generator = _create_cpu_generator(training_anomaly_seed)
    perm = torch.randperm(num_samples, generator=generator, device="cpu")
    anomaly_indices = perm[:num_anomalies]

    signs = torch.randint(
        low=0, high=2, size=(num_anomalies,), generator=generator, device="cpu"
    )
    signs = (signs * 2 - 1).to(dtype=torch.float32)
    delta = float(anomaly_config.delta)

    x_train_mixed = x_train.clone()
    anomaly_indices_device = anomaly_indices.to(device=x_train_mixed.device)
    signs_device = signs.to(device=x_train_mixed.device)

    if anomaly_config.shift_mode == "single_channel":
        candidate_channels = _resolve_channels_to_evaluate(data_config, anomaly_config)
        candidate_tensor = torch.tensor(
            candidate_channels, dtype=torch.int64, device="cpu"
        )
        channel_choices = torch.randint(
            low=0,
            high=candidate_tensor.numel(),
            size=(num_anomalies,),
            generator=generator,
            device="cpu",
        )
        channels = candidate_tensor[channel_choices].to(device=x_train_mixed.device)
        x_train_mixed[anomaly_indices_device, channels] = (
            x_train_mixed[anomaly_indices_device, channels] + signs_device * delta
        )
        return x_train_mixed, num_anomalies, training_anomaly_seed

    subset_indices = _select_subset_indices(
        data_config, anomaly_config, data_config.seed + 3
    )
    subset_tensor = torch.tensor(
        subset_indices, dtype=torch.int64, device=x_train_mixed.device
    )
    x_train_mixed[anomaly_indices_device.unsqueeze(1), subset_tensor.unsqueeze(0)] = (
        x_train_mixed[anomaly_indices_device.unsqueeze(1), subset_tensor.unsqueeze(0)]
        + signs_device.unsqueeze(1) * delta
    )
    return x_train_mixed, num_anomalies, training_anomaly_seed


def _instantiate_model(
    model_config: DirectionalDeviationModelVariantConfig,
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
    model_config: DirectionalDeviationModelVariantConfig,
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
    experiment_config: DirectionalDeviationExperimentConfig,
    data: Dict[str, Tensor],
    model_config: DirectionalDeviationModelVariantConfig,
    device: str,
) -> Dict[str, Any]:
    x_train_for_training, train_anomaly_count, train_anomaly_seed = (
        _prepare_training_tensor(
            x_train=data["x_train"],
            data_config=experiment_config.data,
            anomaly_config=experiment_config.anomaly,
        )
    )
    dataloader_seed = int(torch.initial_seed())
    dataloader = _build_training_dataloader(
        x_train=x_train_for_training,
        batch_size=model_config.batch_size,
        dataloader_seed=dataloader_seed,
    )
    steps_per_epoch = len(dataloader)
    model = _instantiate_model(
        model_config, x_train_for_training.shape[1], steps_per_epoch, device
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
        "mean_active_latents": [],
        "train_anomaly_fraction": float(experiment_config.data.train_anomaly_fraction),
        "train_anomaly_count": int(train_anomaly_count),
        "train_anomaly_seed": int(train_anomaly_seed),
        "warmup_steps": int(model_config.warmup_steps),
    }
    terminated_early = _run_optimizer_warmup_steps(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        warmup_steps=int(model_config.warmup_steps),
        sparsity_coeffs=sparsity_coeffs,
        dataloader_factory=lambda: _build_training_dataloader(
            x_train=x_train_for_training,
            batch_size=model_config.batch_size,
            dataloader_seed=dataloader_seed,
        ),
    )
    if terminated_early:
        print(
            f"\n[{model_config.name}] Non-finite loss detected in warmup; terminating early."
        )
        training_metrics["terminated_early"] = True
        return {"model": model, "training_metrics": training_metrics}

    dataloader = _build_training_dataloader(
        x_train=x_train_for_training,
        batch_size=model_config.batch_size,
        dataloader_seed=dataloader_seed,
    )
    terminated_early = False

    for epoch in range(model_config.num_epochs):
        model.train()
        epoch_totals = defaultdict(float)
        epoch_active_latents_total = 0.0
        epoch_steps = 0
        for step, (batch,) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(batch)
            losses_dict = model.calculate_losses(batch, sparsity_coeffs, output)
            if not bool(torch.isfinite(losses_dict["losses"]).all()):
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
            epoch_steps += 1

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
                f"\n[{model_config.name}] Non-finite loss detected at epoch {epoch + 1}; terminating early."
            )
            break
        if epoch_steps == 0:
            raise ValueError("Training dataloader produced zero steps.")

        epoch_mean_losses = {
            name: total / steps_per_epoch for name, total in epoch_totals.items()
        }
        epoch_mean_active_latents = epoch_active_latents_total / epoch_steps
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
        training_metrics["mean_active_latents"].append(epoch_mean_active_latents)
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
    model_config: DirectionalDeviationModelVariantConfig,
    model_dir: str,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model, model_path)
    model_config_path = os.path.join(model_dir, "model_config.json")
    with open(model_config_path, "w") as f:
        json.dump(asdict(model_config), f, indent=2)
    training_metrics_path = os.path.join(model_dir, "training_metrics.json")
    with open(training_metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
