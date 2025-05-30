from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrainingConfig:
    sae_class: str
    embeddings_path: str
    output_path: str
    d_hidden: int
    sign_aware: bool
    sparsity_coeffs_min_exponent: float = -6.0
    sparsity_coeffs_max_exponent: float = 2.0
    sparsity_coeffs_steps: int = 80
    train_batch_size: int = 128
    validation_batch_size: int = 512
    num_epochs: int = 200
    learning_rate: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    resample_latents_epochs: float = 1.0
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: Optional[float] = None
    reset_activation_offsets_on_resample: Optional[bool] = None
    wandb_project: Optional[str] = None
