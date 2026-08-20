import json
import os
import sys
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

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
    n_instances: int,
    input_dim: int,
    steps_per_epoch: int,
    device: str,
) -> torch.nn.Module:
    resample_latents_threshold = int(
        max(1.0, steps_per_epoch * model_config.resample_latents_epochs)
    )
    if model_config.sae_class == SAE.__name__:
        return SAE(
            n_instances=n_instances,
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
            n_instances=n_instances,
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
            n_instances=n_instances,
            d_in=input_dim,
            d_hidden=model_config.d_hidden,
            resample_latents_threshold=resample_latents_threshold,
            device=device,
        )
    raise ValueError(f"Unknown SAE class: {model_config.sae_class}")


def _resolve_sparsity_coeff_list(
    model_config: PolarityDialModelVariantConfig,
) -> List[float]:
    sparsity_coeff = model_config.sparsity_coeff
    if isinstance(sparsity_coeff, list):
        return [float(value) for value in sparsity_coeff]
    return [float(sparsity_coeff)]


def _resolve_dataloader_seed(
    dataloader_generator: Optional[torch.Generator],
) -> int:
    if dataloader_generator is not None:
        return int(dataloader_generator.initial_seed())
    return int(torch.initial_seed())


def _build_dataloader_generator(dataloader_seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(dataloader_seed))


def _build_training_dataloader(
    x_train: torch.Tensor,
    batch_size: int,
    dataloader_seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
        generator=_build_dataloader_generator(dataloader_seed),
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


def _copy_instance_0_parameters_to_all_instances(model: torch.nn.Module) -> None:
    if not hasattr(model, "n_instances"):
        return
    n_instances = int(getattr(model, "n_instances"))
    if n_instances <= 1:
        return
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim < 1 or parameter.shape[0] != n_instances:
                continue
            parameter.copy_(parameter[0:1].expand_as(parameter))


def slice_trained_model_instance(
    model: torch.nn.Module,
    model_config: PolarityDialModelVariantConfig,
    instance_index: int,
    input_dim: int,
    steps_per_epoch: int,
    device: str,
) -> torch.nn.Module:
    if not hasattr(model, "n_instances"):
        raise ValueError("Model does not expose n_instances.")
    n_instances = int(getattr(model, "n_instances"))
    if instance_index < 0 or instance_index >= n_instances:
        raise ValueError(
            f"instance_index must be within [0, {n_instances}); received {instance_index}."
        )

    sliced_model = _instantiate_model(
        model_config=model_config,
        n_instances=1,
        input_dim=input_dim,
        steps_per_epoch=steps_per_epoch,
        device=device,
    ).to(device)
    source_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, parameter in sliced_model.named_parameters():
            if name not in source_parameters:
                raise ValueError(f"Missing parameter '{name}' in source model.")
            source_parameter = source_parameters[name]
            if source_parameter.ndim >= 1 and source_parameter.shape[0] == n_instances:
                parameter.copy_(source_parameter[instance_index : instance_index + 1])
            else:
                parameter.copy_(source_parameter)
    return sliced_model


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
    dataloader_generator: Optional[torch.Generator] = None,
    model_seed: Optional[int] = None,
) -> Dict[str, Any]:
    dataloader_seed = _resolve_dataloader_seed(dataloader_generator)
    dataloader = _build_training_dataloader(
        x_train=data["x_train"],
        batch_size=model_config.batch_size,
        dataloader_seed=dataloader_seed,
    )
    steps_per_epoch = len(dataloader)
    sparsity_coeff_list = _resolve_sparsity_coeff_list(model_config)
    n_instances = len(sparsity_coeff_list)
    model = _instantiate_model(
        model_config, n_instances, data["x_train"].shape[1], steps_per_epoch, device
    ).to(device)
    _copy_instance_0_parameters_to_all_instances(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_config.learning_rate, betas=model_config.adam_betas
    )
    sparsity_coeffs = torch.tensor(
        sparsity_coeff_list, device=device, dtype=torch.float32
    )
    inactive_latents_count = torch.zeros(
        n_instances, model_config.d_hidden, device=device
    )
    uses_multiple_instances = n_instances > 1
    resample_generators: Optional[List[torch.Generator]] = None
    if uses_multiple_instances:
        base_seed = int(torch.initial_seed()) if model_seed is None else int(model_seed)
        resample_generators = [
            torch.Generator(device=device).manual_seed(base_seed + instance_index)
            for instance_index in range(n_instances)
        ]
    training_metrics: Dict[str, Any] = {"epoch": []}
    if uses_multiple_instances:
        training_metrics["sparsity_coeff_list"] = sparsity_coeff_list
        training_metrics["loss_total_by_instance"] = [[] for _ in range(n_instances)]
        training_metrics["loss_reconstruction_by_instance"] = [
            [] for _ in range(n_instances)
        ]
        training_metrics["loss_sparsity_by_instance"] = [[] for _ in range(n_instances)]
        training_metrics["mean_active_latents_by_instance"] = [
            [] for _ in range(n_instances)
        ]
    else:
        training_metrics["loss_total"] = []
        training_metrics["loss_reconstruction"] = []
        training_metrics["loss_sparsity"] = []
        training_metrics["mean_active_latents"] = []
    training_metrics["warmup_steps"] = int(model_config.warmup_steps)
    terminated_early = _run_optimizer_warmup_steps(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        warmup_steps=int(model_config.warmup_steps),
        sparsity_coeffs=sparsity_coeffs,
        dataloader_factory=lambda: _build_training_dataloader(
            x_train=data["x_train"],
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
        x_train=data["x_train"],
        batch_size=model_config.batch_size,
        dataloader_seed=dataloader_seed,
    )
    terminated_early = False

    for epoch in range(model_config.num_epochs):
        model.train()
        epoch_totals_by_loss: Dict[str, Tensor] = {}
        epoch_total_mean_active_latents_by_instance = torch.zeros(
            n_instances, dtype=torch.float32
        )
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
            if resample_generators is None:
                resampled_latents_mask = model.resample_dead_latents(
                    inactive_latents_count
                )
            else:
                resampled_latents_mask = model.resample_dead_latents(
                    inactive_latents_count, generators=resample_generators
                )
            inactive_latents_count = torch.where(
                resampled_latents_mask,
                torch.zeros_like(inactive_latents_count),
                inactive_latents_count,
            )
            mean_active_latents_by_instance = (
                latents_active.sum(dim=-1)
                .float()
                .mean(dim=1)
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            epoch_total_mean_active_latents_by_instance += (
                mean_active_latents_by_instance
            )
            mean_active_latents = float(mean_active_latents_by_instance.mean().item())

            for loss_name, loss_values in losses_dict.items():
                loss_values_cpu = loss_values.detach().to(
                    device="cpu", dtype=torch.float32
                )
                if loss_name not in epoch_totals_by_loss:
                    epoch_totals_by_loss[loss_name] = torch.zeros_like(loss_values_cpu)
                epoch_totals_by_loss[loss_name] += loss_values_cpu
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
        epoch_mean_losses_by_loss = {
            name: totals / steps_per_epoch
            for name, totals in epoch_totals_by_loss.items()
        }
        epoch_mean_losses_for_logging = {
            name: float(values.mean().item())
            for name, values in epoch_mean_losses_by_loss.items()
        }
        if experiment_config.verbose and (
            (epoch + 1) % experiment_config.log_every_n_epochs == 0
            or epoch + 1 == model_config.num_epochs
        ):
            _log_epoch_summary(epoch, epoch_mean_losses_for_logging, model_config)

        training_metrics["epoch"].append(epoch + 1)
        epoch_mean_active_latents_by_instance = (
            epoch_total_mean_active_latents_by_instance / steps_per_epoch
        )
        if uses_multiple_instances:
            for instance_index in range(n_instances):
                training_metrics["loss_total_by_instance"][instance_index].append(
                    float(epoch_mean_losses_by_loss["losses"][instance_index].item())
                )
                training_metrics["loss_reconstruction_by_instance"][
                    instance_index
                ].append(
                    float(
                        epoch_mean_losses_by_loss["reconstruction_losses"][
                            instance_index
                        ].item()
                    )
                )
                training_metrics["loss_sparsity_by_instance"][instance_index].append(
                    float(
                        epoch_mean_losses_by_loss["sparsity_losses"][
                            instance_index
                        ].item()
                    )
                )
                training_metrics["mean_active_latents_by_instance"][
                    instance_index
                ].append(
                    float(epoch_mean_active_latents_by_instance[instance_index].item())
                )
            if "aux_loss" in epoch_mean_losses_by_loss:
                if "loss_aux_by_instance" not in training_metrics:
                    training_metrics["loss_aux_by_instance"] = [
                        [] for _ in range(n_instances)
                    ]
                for instance_index in range(n_instances):
                    training_metrics["loss_aux_by_instance"][instance_index].append(
                        float(
                            epoch_mean_losses_by_loss["aux_loss"][instance_index].item()
                        )
                    )
        else:
            training_metrics["loss_total"].append(
                float(epoch_mean_losses_by_loss["losses"][0].item())
            )
            training_metrics["loss_reconstruction"].append(
                float(epoch_mean_losses_by_loss["reconstruction_losses"][0].item())
            )
            training_metrics["loss_sparsity"].append(
                float(epoch_mean_losses_by_loss["sparsity_losses"][0].item())
            )
            training_metrics["mean_active_latents"].append(
                float(epoch_mean_active_latents_by_instance[0].item())
            )
            if "aux_loss" in epoch_mean_losses_by_loss:
                training_metrics.setdefault("loss_aux", []).append(
                    float(epoch_mean_losses_by_loss["aux_loss"][0].item())
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
