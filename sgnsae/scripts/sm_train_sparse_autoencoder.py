import dataclasses
import os
from argparse import Namespace, ArgumentParser

from sgnsae.config import TrainingConfig
from sgnsae.models import PosNegSAE
from sgnsae.scripts.train_sparse_autoencoder import train_sparse_autoencoder


def parse_arguements() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-path", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EMBEDDINGS_DIR"),
    )
    parser.add_argument("--embeddings-filename", type=str, required=True)
    parser.add_argument("--sae-class", type=str, required=True)
    parser.add_argument("--d-hidden", type=int, required=True)
    parser.add_argument("--sparsity-coeffs-min-exponent", type=float, default=-6.0)
    parser.add_argument("--sparsity-coeffs-max-exponent", type=float, default=2.0)
    parser.add_argument("--sparsity-coeffs-steps", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-betas-1", type=float, default=0.9)
    parser.add_argument("--adam-betas-2", type=float, default=0.999)
    parser.add_argument("--resample-latents-epochs", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--allow-activation-discontinuity", type=int, default=0)
    parser.add_argument("--reset-activation-offsets-on-resample", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument(
        "--wandb-key",
        required=False,
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguements()
    config = TrainingConfig(
        sae_class=args.sae_class,
        embeddings_path=os.path.join(args.embeddings_dir, args.embeddings_filename),
        output_path=args.output_path,
        d_hidden=args.d_hidden,
        sparsity_coeffs_min_exponent=args.sparsity_coeffs_min_exponent,
        sparsity_coeffs_max_exponent=args.sparsity_coeffs_max_exponent,
        sparsity_coeffs_steps=args.sparsity_coeffs_steps,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        adam_betas=(args.adam_betas_1, args.adam_betas_2),
        resample_latents_epochs=args.resample_latents_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        allow_activation_discontinuity=bool(args.allow_activation_discontinuity)
        if args.sae_class == PosNegSAE.__name__
        else None,
        reset_activation_offsets_on_resample=bool(
            args.reset_activation_offsets_on_resample
        )
        if args.sae_class == PosNegSAE.__name__
        else None,
        wandb_project=args.wandb_project,
    )
    print(f"Running training script with config: {dataclasses.asdict(config)}")
    train_sparse_autoencoder(
        config=config,
        wandb_key=args.wandb_key,
    )
