import math
import unittest
from typing import Any, Dict

import torch

from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DEvaluationConfig,
)
from sgnsae.experiments.toy_model_geometry_validation import evaluation as eval_module


class _FixedLatentsModel(torch.nn.Module):
    def __init__(self, decoder_directions: torch.Tensor, latents: torch.Tensor):
        super().__init__()
        self.W_dec = torch.nn.Parameter(decoder_directions.unsqueeze(0))
        self._latents = latents.to(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"latents": self._latents.unsqueeze(0)}


def _signed_coeffs_to_coeffs(signed_coeffs: torch.Tensor) -> torch.Tensor:
    if signed_coeffs.ndim != 2:
        raise ValueError("signed_coeffs must be 2D.")
    num_samples, num_pairs = (int(value) for value in signed_coeffs.shape)
    coeffs = torch.zeros((num_samples, 2 * num_pairs), dtype=torch.float32)
    plus_coeffs = coeffs[:, 0::2]
    minus_coeffs = coeffs[:, 1::2]

    positive_mask = signed_coeffs > 0
    negative_mask = signed_coeffs < 0
    plus_coeffs[positive_mask] = signed_coeffs[positive_mask].to(dtype=torch.float32)
    minus_coeffs[negative_mask] = (-signed_coeffs[negative_mask]).to(
        dtype=torch.float32
    )
    return coeffs


class TestToyModelGeometryValidationConsolidation(unittest.TestCase):
    def test_perfect_sign_aware_model_consolidates_all_pairs(self) -> None:
        # given
        num_pairs = 3
        regime_values = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float32)
        signed_coeffs = torch.zeros((num_pairs * 4, num_pairs), dtype=torch.float32)
        for pair_index in range(num_pairs):
            start = pair_index * 4
            signed_coeffs[start : start + 4, pair_index] = regime_values

        coeffs_test = _signed_coeffs_to_coeffs(signed_coeffs)
        x_test = torch.zeros((int(coeffs_test.shape[0]), 2), dtype=torch.float32)
        pair_active_mask_test = torch.ones(
            (int(coeffs_test.shape[0]), num_pairs), dtype=torch.bool
        )

        angles = torch.tensor(
            [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0],
            dtype=torch.float32,
        )
        pair_axis_directions = torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=1
        ).to(dtype=torch.float32)
        decoder_directions = pair_axis_directions.transpose(0, 1)
        model = _FixedLatentsModel(
            decoder_directions=decoder_directions, latents=signed_coeffs
        )

        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.9,
            min_activation_corr=0.9,
            min_regime_samples=2,
            hist_num_bins=20,
        )
        data: Dict[str, Any] = {
            "x_test": x_test,
            "coeffs_test": coeffs_test,
            "pair_active_mask_test": pair_active_mask_test,
            "pair_axis_directions": pair_axis_directions,
        }

        # when
        evaluation = eval_module.evaluate_toy2d_model(
            model=model,
            data=data,
            evaluation_config=evaluation_config,
            model_name="perfect",
            sign_aware=True,
            device="cpu",
        )

        # then
        self.assertEqual(evaluation["status"], "ok")
        self.assertEqual(evaluation["num_pairs_total"], num_pairs)
        self.assertEqual(evaluation["num_pairs_evaluable"], num_pairs)
        self.assertEqual(evaluation["num_pairs_consolidated"], num_pairs)
        self.assertAlmostEqual(evaluation["pair_consolidation_rate"], 1.0)
        self.assertAlmostEqual(
            evaluation["pair_consolidation_rate_over_evaluable_pairs"], 1.0
        )
        self.assertEqual(
            evaluation["rate_over_evaluable_pairs"],
            evaluation["pair_consolidation_rate_over_evaluable_pairs"],
        )
        self.assertAlmostEqual(evaluation["evaluable_pair_fraction"], 1.0)
        self.assertIsNotNone(evaluation["histogram_latent"])
        self.assertEqual(evaluation["histogram_latent"]["pair_index"], 0)
        self.assertEqual(evaluation["histogram_latent"]["latent_index"], 0)

    def test_non_negative_model_cannot_consolidate_due_to_sign_condition(self) -> None:
        # given
        num_pairs = 2
        signed_coeffs = torch.tensor(
            [
                [-2.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, -2.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
        )
        coeffs_test = _signed_coeffs_to_coeffs(signed_coeffs)
        x_test = torch.zeros((int(coeffs_test.shape[0]), 2), dtype=torch.float32)
        pair_active_mask_test = torch.ones(
            (int(coeffs_test.shape[0]), num_pairs), dtype=torch.bool
        )
        angles = torch.tensor([0.0, math.pi / 2.0], dtype=torch.float32)
        pair_axis_directions = torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=1
        ).to(dtype=torch.float32)
        decoder_directions = pair_axis_directions.transpose(0, 1)

        non_negative_latents = torch.relu(signed_coeffs)
        model = _FixedLatentsModel(
            decoder_directions=decoder_directions, latents=non_negative_latents
        )

        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.9,
            min_activation_corr=0.0,
            min_regime_samples=2,
            hist_num_bins=10,
        )
        data: Dict[str, Any] = {
            "x_test": x_test,
            "coeffs_test": coeffs_test,
            "pair_active_mask_test": pair_active_mask_test,
            "pair_axis_directions": pair_axis_directions,
        }

        # when
        evaluation = eval_module.evaluate_toy2d_model(
            model=model,
            data=data,
            evaluation_config=evaluation_config,
            model_name="baseline",
            sign_aware=False,
            device="cpu",
        )

        # then
        self.assertEqual(evaluation["status"], "ok")
        self.assertEqual(evaluation["num_pairs_total"], num_pairs)
        self.assertEqual(evaluation["num_pairs_evaluable"], num_pairs)
        self.assertEqual(evaluation["num_pairs_consolidated"], 0)
        self.assertAlmostEqual(evaluation["pair_consolidation_rate"], 0.0)
        self.assertAlmostEqual(
            evaluation["pair_consolidation_rate_over_evaluable_pairs"], 0.0
        )
        self.assertEqual(
            evaluation["rate_over_evaluable_pairs"],
            evaluation["pair_consolidation_rate_over_evaluable_pairs"],
        )
        self.assertIsNone(evaluation["histogram_latent"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
