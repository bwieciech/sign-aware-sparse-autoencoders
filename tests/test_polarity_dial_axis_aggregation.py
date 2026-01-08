import unittest

import einops
import torch

from sgnsae.experiments.polarity_dial.config import PolarityDialEvaluationConfig
from sgnsae.experiments.polarity_dial.evaluation import (
    align_decoder_and_features,
    compute_alignment_and_calibration,
)


class DummyTwoAtomNonNegativeModel(torch.nn.Module):
    def __init__(self, decoder_directions: torch.Tensor, latents: torch.Tensor) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(
            decoder_directions.transpose(0, 1).unsqueeze(0).contiguous()
        )
        self.latents = latents.unsqueeze(0)
        self.sign_aware = False

    def forward(self, x: torch.Tensor) -> dict:
        x_hat = einops.einsum(
            self.W_dec,
            self.latents,
            "n_instances d_in d_hidden, n_instances batch d_hidden -> n_instances batch d_in",
        )
        return {"latents": self.latents, "x_hat": x_hat}


class TestNonNegativeAxisAggregation(unittest.TestCase):
    def test_axis_aggregation_recovers_signed_coefficients(self) -> None:
        # given
        evaluation_config = PolarityDialEvaluationConfig(
            axis_aggregation_for_nonnegative=True,
            min_regime_samples=5,
            max_points_per_regime_scatter=10,
        )
        feature_directions = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        decoder_directions = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32
        )
        coefficients = torch.tensor(
            [[1.0], [1.0], [1.0], [1.0], [1.0], [-2.0], [-2.0], [-2.0], [-2.0], [-2.0]],
            dtype=torch.float32,
        )
        latents = torch.cat(
            [
                torch.stack([coefficients[:5, 0], torch.zeros(5)], dim=1),
                torch.stack([torch.zeros(5), -coefficients[5:, 0]], dim=1),
            ],
            dim=0,
        )
        model = DummyTwoAtomNonNegativeModel(decoder_directions, latents)
        x_test = einops.einsum(
            coefficients,
            feature_directions,
            "batch num_features, num_features d_in -> batch d_in",
        )

        # when
        alignment, axis_alignment, _ = align_decoder_and_features(
            model, feature_directions, evaluation_config
        )
        calibration, _ = compute_alignment_and_calibration(
            model,
            x_test,
            coefficients,
            alignment,
            axis_alignment,
            evaluation_config,
        )

        # then
        self.assertEqual(calibration["summary"]["num_pairs"], 1)
        first_entry = calibration["per_pair"][0]
        self.assertAlmostEqual(first_entry["gamma_plus"], 1.0, places=6)
        self.assertAlmostEqual(first_entry["gamma_minus"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
