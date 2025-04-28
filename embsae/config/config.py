from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class SAEConfig:
    sae_class: str
    embeddings_path: str
    output_path: str
    d_hidden: int
    sparsity_coeffs_min_exponent: float = -5.0
    sparsity_coeffs_max_exponent: float = 1.5
    sparsity_coeffs_steps: int = 128
    train_batch_size: int = 128
    validation_batch_size: int = 512
    num_epochs: int = 128
    learning_rate: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    resample_latents_epochs: float = 1.0
    allow_activation_discontinuity: Optional[bool] = None
