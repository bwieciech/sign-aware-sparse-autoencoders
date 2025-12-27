import unittest

import einops
import torch

from sgnsae.experiments.polarity_dial.evaluation import (
    align_decoder_and_features,
    compute_alignment_and_calibration,
)


class DummyAlignedModel(torch.nn.Module):
    def __init__(self, feature_directions: torch.Tensor, latents: torch.Tensor) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(
            feature_directions.transpose(0, 1).unsqueeze(0).contiguous()
        )
        self.latents = latents.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> dict:
        x_hat = einops.einsum(
            self.W_dec,
            self.latents,
            "n_instances d_in d_hidden, n_instances batch d_hidden -> n_instances batch d_in",
        )
        return {"latents": self.latents, "x_hat": x_hat}


class TestPolarityDialEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        feature_directions = torch.eye(3)
        coefficients = torch.tensor(
            [
                [1.0, 0.0, -0.5],
                [1.0, 0.0, -0.5],
                [1.0, 0.0, -0.5],
                [1.0, 0.0, -0.5],
                [1.0, 0.0, -0.5],
                [-0.7, 0.3, 0.0],
                [-0.7, 0.3, 0.0],
                [-0.7, 0.3, 0.0],
                [-0.7, 0.3, 0.0],
                [-0.7, 0.3, 0.0],
            ]
        )
        self.feature_directions = feature_directions
        self.coefficients = coefficients
        self.model = DummyAlignedModel(feature_directions, coefficients)

    def test_alignment_and_gamma_close_to_one(self) -> None:
        # given
        x_test = torch.zeros(self.coefficients.shape[0], self.feature_directions.shape[1])

        # when
        alignment, decoder_directions = align_decoder_and_features(
            self.model, self.feature_directions
        )
        calibration, _ = compute_alignment_and_calibration(
            self.model,
            x_test,
            self.coefficients,
            self.feature_directions,
            alignment,
            decoder_directions,
            max_scatter_points_per_regime=20,
        )

        # then
        gamma_pairs = {(p["latent_index"], p["feature_index"]): p for p in calibration["per_pair"]}
        self.assertAlmostEqual(gamma_pairs[(0, 0)]["gamma_plus"], 1.0, places=5)
        self.assertAlmostEqual(gamma_pairs[(0, 0)]["gamma_minus"], 1.0, places=5)
        self.assertIsNone(gamma_pairs[(1, 1)]["gamma_minus"])
        self.assertAlmostEqual(gamma_pairs[(1, 1)]["gamma_plus"], 1.0, places=5)
        summary = calibration["summary"]
        self.assertGreaterEqual(summary["num_pairs"], 3)
        self.assertIn("test_reconstruction_mse", summary)
        self.assertIn("test_l0", summary)

    def test_gamma_skips_when_insufficient_samples(self) -> None:
        # given
        coefficients = torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        )
        model = DummyAlignedModel(self.feature_directions, coefficients)
        x_test = torch.zeros(coefficients.shape[0], self.feature_directions.shape[1])

        # when
        alignment, decoder_directions = align_decoder_and_features(
            model, self.feature_directions
        )
        calibration, _ = compute_alignment_and_calibration(
            model,
            x_test,
            coefficients,
            self.feature_directions,
            alignment,
            decoder_directions,
            max_scatter_points_per_regime=10,
            min_gamma_count=3,
        )

        # then
        first_entry = calibration["per_pair"][0]
        self.assertIsNotNone(first_entry["gamma_plus"])
        self.assertIsNone(first_entry["gamma_minus"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
