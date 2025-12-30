import unittest

import torch

from sgnsae.experiments.directional_deviation.baseline_scoring import (
    calculate_directional_scores_from_one_sided_evidence,
)


class TestDirectionalDeviationBaselineLeakagePrevention(unittest.TestCase):
    def test_baseline_scoring_does_not_use_negative_tail_of_signed_pre_activations(
        self,
    ) -> None:
        # given
        pi_gate = torch.tensor([[2.0], [-2.0]])
        one_sided_evidence = torch.relu(pi_gate)
        evidence_thresholds = torch.tensor([1.0])

        latent_directions = torch.tensor([1])

        # when
        score_high, score_low = calculate_directional_scores_from_one_sided_evidence(
            one_sided_evidence=one_sided_evidence,
            evidence_thresholds=evidence_thresholds,
            latent_directions=latent_directions,
        )

        score_high_leaky = torch.relu(pi_gate - torch.tensor([1.0])).sum(dim=1)
        score_low_leaky = torch.relu(-pi_gate - torch.tensor([1.0])).sum(dim=1)

        # then
        self.assertGreater(score_high[0].item(), 0.0)
        self.assertEqual(score_low[0].item(), 0.0)

        self.assertEqual(score_high[1].item(), 0.0)
        self.assertEqual(score_low[1].item(), 0.0)

        self.assertGreater(score_high_leaky[0].item(), 0.0)
        self.assertEqual(score_low_leaky[0].item(), 0.0)

        self.assertEqual(score_high_leaky[1].item(), 0.0)
        self.assertGreater(score_low_leaky[1].item(), 0.0)

    def test_baseline_scoring_returns_zero_score_when_no_latents_assigned_to_direction(
        self,
    ) -> None:
        # given
        one_sided_evidence = torch.tensor([[3.0], [0.0]])
        evidence_thresholds = torch.tensor([1.0])

        # when
        score_high, score_low = calculate_directional_scores_from_one_sided_evidence(
            one_sided_evidence=one_sided_evidence,
            evidence_thresholds=evidence_thresholds,
            latent_directions=torch.tensor([-1]),
        )

        # then
        self.assertTrue(torch.all(score_high == 0.0))
        self.assertAlmostEqual(score_low[0].item(), 2.0, places=7)
        self.assertAlmostEqual(score_low[1].item(), 0.0, places=7)

    def test_baseline_scoring_raises_for_non_2d_evidence(self) -> None:
        # given
        one_sided_evidence = torch.tensor([1.0, 2.0])

        # when & then
        with self.assertRaises(ValueError):
            _ = calculate_directional_scores_from_one_sided_evidence(
                one_sided_evidence=one_sided_evidence,
                evidence_thresholds=torch.tensor([1.0, 1.0]),
                latent_directions=torch.tensor([1, -1]),
            )

    def test_baseline_scoring_raises_for_non_1d_thresholds(self) -> None:
        # given
        one_sided_evidence = torch.zeros(2, 1)
        evidence_thresholds = torch.zeros(1, 1)

        # when & then
        with self.assertRaises(ValueError):
            _ = calculate_directional_scores_from_one_sided_evidence(
                one_sided_evidence=one_sided_evidence,
                evidence_thresholds=evidence_thresholds,
                latent_directions=torch.tensor([1]),
            )

    def test_baseline_scoring_raises_for_non_1d_directions(self) -> None:
        # given
        one_sided_evidence = torch.zeros(2, 1)
        evidence_thresholds = torch.zeros(1)
        latent_directions = torch.tensor([[1]])

        # when & then
        with self.assertRaises(ValueError):
            _ = calculate_directional_scores_from_one_sided_evidence(
                one_sided_evidence=one_sided_evidence,
                evidence_thresholds=evidence_thresholds,
                latent_directions=latent_directions,
            )

    def test_baseline_scoring_raises_for_threshold_length_mismatch(self) -> None:
        # given
        one_sided_evidence = torch.zeros(2, 2)
        evidence_thresholds = torch.zeros(3)

        # when & then
        with self.assertRaises(ValueError):
            _ = calculate_directional_scores_from_one_sided_evidence(
                one_sided_evidence=one_sided_evidence,
                evidence_thresholds=evidence_thresholds,
                latent_directions=torch.tensor([1, -1]),
            )

    def test_baseline_scoring_raises_for_direction_length_mismatch(self) -> None:
        # given
        one_sided_evidence = torch.zeros(2, 2)
        latent_directions = torch.tensor([1])

        # when & then
        with self.assertRaises(ValueError):
            _ = calculate_directional_scores_from_one_sided_evidence(
                one_sided_evidence=one_sided_evidence,
                evidence_thresholds=torch.zeros(2),
                latent_directions=latent_directions,
            )

    def test_baseline_scoring_raises_for_invalid_direction_values(self) -> None:
        # given
        one_sided_evidence = torch.zeros(2, 1)
        evidence_thresholds = torch.zeros(1)
        latent_directions = torch.tensor([0])

        # when & then
        with self.assertRaises(ValueError):
            _ = calculate_directional_scores_from_one_sided_evidence(
                one_sided_evidence=one_sided_evidence,
                evidence_thresholds=evidence_thresholds,
                latent_directions=latent_directions,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
