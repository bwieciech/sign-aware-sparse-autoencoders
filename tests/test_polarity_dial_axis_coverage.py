import unittest

import torch

from sgnsae.experiments.polarity_dial.config import PolarityDialEvaluationConfig
from sgnsae.experiments.polarity_dial.evaluation import align_decoder_and_features


class DummyDecoderModel(torch.nn.Module):
    def __init__(self, W_dec: torch.Tensor) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(W_dec)


class TestPolarityDialAxisCoverage(unittest.TestCase):
    def test_random_decoder_below_threshold_has_no_axis_coverage(self) -> None:
        # given
        evaluation_config = PolarityDialEvaluationConfig(min_axis_abs_cos=0.9)
        feature_directions = torch.eye(4, dtype=torch.float32)[:3]
        model = DummyDecoderModel(W_dec=torch.zeros(1, 4, 5, dtype=torch.float32))

        # when
        _, axis_alignment, _ = align_decoder_and_features(
            model, feature_directions, evaluation_config
        )

        # then
        coverage_summary = axis_alignment["coverage_summary"]
        self.assertEqual(coverage_summary["total_axes"], 3)
        self.assertEqual(coverage_summary["axes_with_pos"], 0)
        self.assertEqual(coverage_summary["axes_with_neg"], 0)
        self.assertEqual(coverage_summary["axes_with_both"], 0)
        self.assertEqual(coverage_summary["axes_with_neither"], 3)

        for axis in axis_alignment["axes"]:
            self.assertIsNone(axis["latent_pos"])
            self.assertIsNone(axis["latent_neg"])
            self.assertIsNone(axis["signed_cos_pos"])
            self.assertIsNone(axis["signed_cos_neg"])
            self.assertIsNone(axis["axis_cos_pos"])
            self.assertIsNone(axis["axis_cos_neg"])

    def test_imperfect_negative_side_below_threshold_is_excluded_from_coverage(
        self,
    ) -> None:
        # given
        evaluation_config = PolarityDialEvaluationConfig(min_axis_abs_cos=0.9)
        feature_directions = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        W_dec = torch.tensor(
            [[[1.0, -0.8], [0.0, 0.6]]],
            dtype=torch.float32,
        )
        model = DummyDecoderModel(W_dec=W_dec)

        # when
        _, axis_alignment, _ = align_decoder_and_features(
            model, feature_directions, evaluation_config
        )

        # then
        coverage_summary = axis_alignment["coverage_summary"]
        self.assertEqual(coverage_summary["total_axes"], 1)
        self.assertEqual(coverage_summary["axes_with_pos"], 1)
        self.assertEqual(coverage_summary["axes_with_neg"], 0)
        self.assertEqual(coverage_summary["axes_with_both"], 0)
        self.assertEqual(coverage_summary["axes_with_neither"], 0)

        axis = axis_alignment["axes"][0]
        self.assertEqual(axis["latent_pos"], 0)
        self.assertIsNone(axis["latent_neg"])
        self.assertAlmostEqual(axis["signed_cos_pos"], 1.0, places=6)
        self.assertIsNone(axis["signed_cos_neg"])
        self.assertAlmostEqual(axis["axis_cos_pos"], 1.0, places=6)
        self.assertIsNone(axis["axis_cos_neg"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
