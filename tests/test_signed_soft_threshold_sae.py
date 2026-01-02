import unittest

import torch

from sgnsae.experiments.polarity_dial.config import PolarityDialEvaluationConfig
from sgnsae.experiments.polarity_dial.evaluation import (
    align_decoder_and_features,
    compute_alignment_and_calibration,
)
from sgnsae.models import SignedSoftThresholdSAE


class TestSignedSoftThresholdSAE(unittest.TestCase):
    def test_forward_raises_for_invalid_input_dimensions(self) -> None:
        # given
        model = SignedSoftThresholdSAE(
            n_instances=1,
            d_in=2,
            d_hidden=1,
            resample_latents_threshold=1,
            device="cpu",
        )
        x = torch.tensor([1.0, 2.0])

        # when & then
        with self.assertRaises(ValueError):
            _ = model(x)

    def test_forward_can_produce_negative_latents(self) -> None:
        # given
        model = SignedSoftThresholdSAE(
            n_instances=1,
            d_in=2,
            d_hidden=1,
            resample_latents_threshold=1,
            device="cpu",
        )
        model.W_enc.data = torch.tensor([[[1.0, 0.0]]])
        model.W_dec.data = torch.tensor([[[1.0], [0.0]]])
        model.b_enc.data = torch.zeros_like(model.b_enc.data)
        model.b_dec.data = torch.zeros_like(model.b_dec.data)
        model.theta.data = torch.zeros_like(model.theta.data)
        x = torch.tensor([[-2.0, 0.0]], dtype=torch.float32)

        # when
        output = model(x)

        # then
        self.assertLess(output["latents"][0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(output["latents"][0, 0, 0].item(), -2.0, places=6)

    def test_calculate_losses_returns_expected_keys(self) -> None:
        # given
        model = SignedSoftThresholdSAE(
            n_instances=1,
            d_in=2,
            d_hidden=1,
            resample_latents_threshold=1,
            device="cpu",
        )
        model.W_enc.data = torch.zeros_like(model.W_enc.data)
        model.W_dec.data = torch.zeros_like(model.W_dec.data)
        model.b_enc.data = torch.zeros_like(model.b_enc.data)
        model.b_dec.data = torch.zeros_like(model.b_dec.data)
        model.theta.data = torch.zeros_like(model.theta.data)
        x = torch.zeros(3, 2, dtype=torch.float32)
        sparsity_coeffs = torch.tensor([0.1], dtype=torch.float32)

        # when
        output = model(x)
        losses = model.calculate_losses(x, sparsity_coeffs, output)

        # then
        self.assertIn("losses", losses)
        self.assertIn("reconstruction_losses", losses)
        self.assertIn("sparsity_losses", losses)

    def test_resample_dead_latents_resets_theta(self) -> None:
        # given
        torch.manual_seed(0)
        model = SignedSoftThresholdSAE(
            n_instances=1,
            d_in=2,
            d_hidden=1,
            resample_latents_threshold=0,
            device="cpu",
        )
        model.theta.data = torch.full_like(model.theta.data, 1.23)
        inactive_latents_count = torch.tensor([[1]])

        # when
        mask = model.resample_dead_latents(inactive_latents_count)

        # then
        self.assertTrue(mask[0, 0].item())
        self.assertAlmostEqual(model.theta[0, 0].item(), 0.0, places=6)

    def test_resample_dead_latents_noop_when_below_threshold(self) -> None:
        # given
        model = SignedSoftThresholdSAE(
            n_instances=1,
            d_in=2,
            d_hidden=1,
            resample_latents_threshold=10,
            device="cpu",
        )
        model.theta.data = torch.full_like(model.theta.data, 1.23)
        inactive_latents_count = torch.tensor([[1]])

        # when
        mask = model.resample_dead_latents(inactive_latents_count)

        # then
        self.assertFalse(mask[0, 0].item())
        self.assertAlmostEqual(model.theta[0, 0].item(), 1.23, places=6)

    def test_evaluation_runs_without_crashing(self) -> None:
        # given
        evaluation_config = PolarityDialEvaluationConfig(
            min_regime_samples=1,
            max_points_per_regime_scatter=20,
        )
        feature_directions = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        coefficients = torch.tensor(
            [[1.0], [2.0], [3.0], [-1.0], [-2.0], [-3.0]],
            dtype=torch.float32,
        )
        x_test = coefficients @ feature_directions
        model = SignedSoftThresholdSAE(
            n_instances=1,
            d_in=2,
            d_hidden=1,
            resample_latents_threshold=1,
            device="cpu",
        )
        model.W_enc.data = torch.tensor([[[1.0, 0.0]]])
        model.W_dec.data = torch.tensor([[[1.0], [0.0]]])
        model.b_enc.data = torch.zeros_like(model.b_enc.data)
        model.b_dec.data = torch.zeros_like(model.b_dec.data)
        model.theta.data = torch.zeros_like(model.theta.data)

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
        entry = calibration["per_pair"][0]
        self.assertAlmostEqual(entry["gamma_plus"], 1.0, places=6)
        self.assertAlmostEqual(entry["gamma_minus"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
