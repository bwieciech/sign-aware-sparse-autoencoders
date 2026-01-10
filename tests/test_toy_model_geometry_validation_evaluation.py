import os
import tempfile
import unittest
from typing import Any, Dict

import torch

from sgnsae.experiments.toy_model_geometry_validation.config import (
    Toy2DEvaluationConfig,
)
from sgnsae.experiments.toy_model_geometry_validation import evaluation as eval_module


class _LatentOnlyModel(torch.nn.Module):
    def __init__(
        self,
        decoder_directions: torch.Tensor,
        signed: bool,
        latent_offset: float = 0.0,
        bad_latents_shape: bool = False,
        missing_latents: bool = False,
        drop_last_sample: bool = False,
    ):
        super().__init__()
        self.W_dec = torch.nn.Parameter(decoder_directions.unsqueeze(0))
        self._signed = signed
        self._latent_offset = float(latent_offset)
        self._bad_latents_shape = bad_latents_shape
        self._missing_latents = missing_latents
        self._drop_last_sample = drop_last_sample

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._missing_latents:
            return {}

        signed_coeff = x[:, 0].detach()
        if not self._signed:
            signed_coeff = torch.relu(signed_coeff)
        if self._drop_last_sample:
            signed_coeff = signed_coeff[:-1]

        latents = (
            (signed_coeff + self._latent_offset)
            .unsqueeze(-1)
            .repeat(1, int(self.W_dec.shape[2]))
        )
        if self._bad_latents_shape:
            return {"latents": latents}
        return {"latents": latents.unsqueeze(0)}


class _NoDecoderModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"latents": x.unsqueeze(0)}


class _FixedLatentsModel(torch.nn.Module):
    def __init__(self, decoder_directions: torch.Tensor, latents: torch.Tensor):
        super().__init__()
        self.W_dec = torch.nn.Parameter(decoder_directions.unsqueeze(0))
        self._latents = latents.to(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"latents": self._latents.unsqueeze(0)}


class TestToyModelGeometryValidationEvaluation(unittest.TestCase):
    def _build_data(
        self,
        signed_coeffs: torch.Tensor,
        pair_axis_direction: torch.Tensor,
        pair_active_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        coeffs = torch.zeros((int(signed_coeffs.shape[0]), 2), dtype=torch.float32)
        positive_mask = signed_coeffs > 0
        negative_mask = signed_coeffs < 0
        coeffs[positive_mask, 0] = signed_coeffs[positive_mask]
        coeffs[negative_mask, 1] = (-signed_coeffs[negative_mask]).to(
            dtype=torch.float32
        )
        x = torch.zeros((int(signed_coeffs.shape[0]), 2), dtype=torch.float32)
        x[:, 0] = signed_coeffs.to(dtype=torch.float32)
        if pair_active_mask is None:
            pair_active_mask = torch.ones(
                (int(signed_coeffs.shape[0]), 1), dtype=torch.bool
            )
        return {
            "x_test": x,
            "coeffs_test": coeffs,
            "pair_active_mask_test": pair_active_mask,
            "pair_axis_directions": pair_axis_direction.unsqueeze(0),
        }

    def test_extract_signed_pair_coefficients_returns_plus_minus_difference(
        self,
    ) -> None:
        # given
        coeffs = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])

        # when
        result = eval_module.extract_signed_pair_coefficients(coeffs)

        # then
        expected = torch.tensor([[1.0], [-2.0], [-1.0]])
        torch.testing.assert_close(result, expected)

    def test_extract_signed_pair_coefficients_rejects_invalid_shapes(self) -> None:
        # given
        coeffs_bad_dim = torch.zeros((2, 1, 1))
        coeffs_bad_columns = torch.zeros((2, 3))

        # when / then
        with self.assertRaises(ValueError):
            _ = eval_module.extract_signed_pair_coefficients(coeffs_bad_dim)
        with self.assertRaises(ValueError):
            _ = eval_module.extract_signed_pair_coefficients(coeffs_bad_columns)

    def test_extract_decoder_directions_validates_required_attributes(self) -> None:
        # given
        no_decoder_model = _NoDecoderModel()
        wrong_type_model = torch.nn.Module()
        wrong_type_model.W_dec = "bad"
        wrong_ndim_model = torch.nn.Module()
        wrong_ndim_model.W_dec = torch.zeros((2, 2))
        wrong_instances_model = torch.nn.Module()
        wrong_instances_model.W_dec = torch.zeros((2, 2, 1))
        wrong_d_in_model = torch.nn.Module()
        wrong_d_in_model.W_dec = torch.zeros((1, 3, 1))

        # when / then
        with self.assertRaises(ValueError):
            _ = eval_module.extract_decoder_directions(no_decoder_model)
        with self.assertRaises(ValueError):
            _ = eval_module.extract_decoder_directions(wrong_type_model)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _ = eval_module.extract_decoder_directions(wrong_ndim_model)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _ = eval_module.extract_decoder_directions(wrong_instances_model)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _ = eval_module.extract_decoder_directions(wrong_d_in_model)  # type: ignore[arg-type]

    def test_pearsonr_returns_none_for_too_few_samples_or_zero_variance(self) -> None:
        # given
        one_sample_x = torch.tensor([1.0])
        one_sample_y = torch.tensor([2.0])
        constant_x = torch.tensor([1.0, 1.0, 1.0])
        variable_y = torch.tensor([1.0, 2.0, 3.0])

        # when
        one_sample_result = eval_module._pearsonr(one_sample_x, one_sample_y)
        constant_result = eval_module._pearsonr(constant_x, variable_y)

        # then
        self.assertIsNone(one_sample_result)
        self.assertIsNone(constant_result)

    def test_pearsonr_rejects_shape_mismatches(self) -> None:
        # given
        x = torch.zeros((2,))
        y = torch.zeros((3,))
        z = torch.zeros((2, 1))

        # when / then
        with self.assertRaises(ValueError):
            _ = eval_module._pearsonr(x, y)
        with self.assertRaises(ValueError):
            _ = eval_module._pearsonr(z, x)  # type: ignore[arg-type]

    def test_select_histogram_latent_rejects_out_of_range_pair_index(self) -> None:
        # given
        dot_products = torch.zeros((1, 1))

        # when / then
        with self.assertRaises(ValueError):
            _ = eval_module._select_histogram_latent(dot_products, pair_index=2)

    def test_select_histogram_latent_returns_best_aligned_latent(self) -> None:
        # given
        dot_products = torch.tensor([[0.1, -0.2], [-0.9, 0.3], [0.4, -0.8]])

        # when
        output = eval_module._select_histogram_latent(dot_products, pair_index=0)

        # then
        self.assertEqual(output["pair_index"], 0)
        self.assertEqual(output["latent_index"], 1)
        self.assertAlmostEqual(output["axis_abs_cos"], 0.9)

    def test_evaluate_toy2d_model_computes_consolidation_for_perfect_signed_model(
        self,
    ) -> None:
        # given
        signed_coeffs = torch.tensor(
            [-2.0, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0]
        )
        data = self._build_data(
            signed_coeffs=signed_coeffs,
            pair_axis_direction=torch.tensor([1.0, 0.0]),
        )
        model = _LatentOnlyModel(
            decoder_directions=torch.tensor([[1.0], [0.0]]),
            signed=True,
        )
        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.9,
            min_activation_corr=0.9,
            min_regime_samples=2,
            hist_num_bins=10,
        )

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
        self.assertEqual(evaluation["num_pairs_total"], 1)
        self.assertEqual(evaluation["num_pairs_evaluable"], 1)
        self.assertEqual(evaluation["num_pairs_consolidated"], 1)
        self.assertEqual(evaluation["pair_consolidation_rate"], 1.0)
        self.assertAlmostEqual(
            evaluation["pair_consolidation_rate_over_evaluable_pairs"], 1.0
        )
        self.assertEqual(
            evaluation["rate_over_evaluable_pairs"],
            evaluation["pair_consolidation_rate_over_evaluable_pairs"],
        )
        self.assertAlmostEqual(evaluation["evaluable_pair_fraction"], 1.0)
        self.assertEqual(evaluation["histogram_latent"]["latent_index"], 0)
        self.assertAlmostEqual(evaluation["histogram_latent"]["axis_abs_cos"], 1.0)
        self.assertEqual(len(evaluation["consolidated_pairs"]), 1)
        self.assertEqual(evaluation["consolidated_pairs"][0]["pair_index"], 0)
        self.assertEqual(evaluation["consolidated_pairs"][0]["latent_index"], 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_module.persist_evaluation(evaluation=evaluation, model_dir=tmpdir)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "evaluation.json")))

    def test_evaluate_toy2d_model_rejects_latents_with_wrong_mean_signs_after_sign_agreement(
        self,
    ) -> None:
        # given
        signed_coeffs = torch.tensor(
            list(range(-1, -11, -1)) + list(range(1, 11)),
            dtype=torch.float32,
        )
        data = self._build_data(
            signed_coeffs=signed_coeffs,
            pair_axis_direction=torch.tensor([1.0, 0.0]),
        )
        latents = torch.tensor(
            [0.1]
            + [-0.01] * 9
            + [-0.1]
            + [0.01] * 9,
            dtype=torch.float32,
        ).unsqueeze(1)
        model = _FixedLatentsModel(
            decoder_directions=torch.tensor([[1.0], [0.0]]),
            latents=latents,
        )
        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.9,
            min_activation_corr=0.0,
            min_sign_agreement=0.89,
            min_regime_samples=1,
            hist_num_bins=10,
        )

        # when
        evaluation = eval_module.evaluate_toy2d_model(
            model=model,
            data=data,
            evaluation_config=evaluation_config,
            model_name="mean_flip",
            sign_aware=True,
            device="cpu",
        )

        # then
        self.assertEqual(evaluation["num_pairs_evaluable"], 1)
        self.assertEqual(evaluation["num_pairs_consolidated"], 0)

    def test_evaluate_toy2d_model_returns_zero_consolidation_for_non_negative_model(
        self,
    ) -> None:
        # given
        signed_coeffs = torch.tensor([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        data = self._build_data(
            signed_coeffs=signed_coeffs,
            pair_axis_direction=torch.tensor([1.0, 0.0]),
        )
        model = _LatentOnlyModel(
            decoder_directions=torch.tensor([[1.0], [0.0]]),
            signed=False,
        )
        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.9,
            min_activation_corr=0.99,
            min_regime_samples=1,
            hist_num_bins=10,
        )

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
        self.assertEqual(evaluation["num_pairs_evaluable"], 1)
        self.assertEqual(evaluation["num_pairs_consolidated"], 0)
        self.assertEqual(evaluation["pair_consolidation_rate"], 0.0)
        self.assertEqual(
            evaluation["pair_consolidation_rate_over_evaluable_pairs"], 0.0
        )
        self.assertEqual(
            evaluation["rate_over_evaluable_pairs"],
            evaluation["pair_consolidation_rate_over_evaluable_pairs"],
        )
        self.assertIsNone(evaluation["histogram_latent"])

    def test_evaluate_toy2d_model_marks_pairs_not_evaluable_when_regime_too_small(
        self,
    ) -> None:
        # given
        signed_coeffs = torch.tensor([-1.0, 1.0])
        data = self._build_data(
            signed_coeffs=signed_coeffs,
            pair_axis_direction=torch.tensor([1.0, 0.0]),
        )
        model = _LatentOnlyModel(
            decoder_directions=torch.tensor([[1.0], [0.0]]),
            signed=True,
        )
        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.9,
            min_activation_corr=0.9,
            min_regime_samples=5,
            hist_num_bins=10,
        )

        # when
        evaluation = eval_module.evaluate_toy2d_model(
            model=model,
            data=data,
            evaluation_config=evaluation_config,
            model_name="not_evaluable",
            sign_aware=True,
            device="cpu",
        )

        # then
        self.assertEqual(evaluation["num_pairs_evaluable"], 0)
        self.assertIsNone(evaluation["pair_consolidation_rate_over_evaluable_pairs"])
        self.assertAlmostEqual(evaluation["evaluable_pair_fraction"], 0.0)

    def test_evaluate_toy2d_model_skips_non_candidate_latents_and_rejects_same_sign_latents(
        self,
    ) -> None:
        # given
        signed_coeffs = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        data = self._build_data(
            signed_coeffs=signed_coeffs,
            pair_axis_direction=torch.tensor([1.0, 0.0]),
        )
        model = _LatentOnlyModel(
            decoder_directions=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            signed=True,
            latent_offset=10.0,
        )
        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.9,
            min_activation_corr=0.9,
            min_regime_samples=1,
            hist_num_bins=10,
        )

        # when
        evaluation = eval_module.evaluate_toy2d_model(
            model=model,
            data=data,
            evaluation_config=evaluation_config,
            model_name="offset",
            sign_aware=True,
            device="cpu",
        )

        # then
        self.assertEqual(evaluation["num_pairs_evaluable"], 1)
        self.assertEqual(evaluation["num_pairs_consolidated"], 0)

    def test_evaluate_toy2d_model_skips_pairs_without_aligned_latents(self) -> None:
        # given
        signed_coeffs = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        data = self._build_data(
            signed_coeffs=signed_coeffs,
            pair_axis_direction=torch.tensor([1.0, 0.0]),
        )
        model = _LatentOnlyModel(
            decoder_directions=torch.tensor([[0.0], [1.0]]),
            signed=True,
        )
        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.9,
            min_activation_corr=0.9,
            min_regime_samples=1,
            hist_num_bins=10,
        )

        # when
        evaluation = eval_module.evaluate_toy2d_model(
            model=model,
            data=data,
            evaluation_config=evaluation_config,
            model_name="orthogonal",
            sign_aware=True,
            device="cpu",
        )

        # then
        self.assertEqual(evaluation["num_pairs_evaluable"], 1)
        self.assertEqual(evaluation["num_pairs_consolidated"], 0)
        self.assertEqual(
            evaluation["pair_consolidation_rate_over_evaluable_pairs"], 0.0
        )

    def test_evaluate_toy2d_model_raises_on_missing_keys_or_invalid_output(
        self,
    ) -> None:
        # given
        data_missing = {"x_test": torch.zeros((1, 2))}
        signed_coeffs = torch.tensor([-1.0, 1.0])
        data = self._build_data(
            signed_coeffs=signed_coeffs,
            pair_axis_direction=torch.tensor([1.0, 0.0]),
        )
        decoder_directions = torch.tensor([[1.0], [0.0]])
        missing_latents_model = _LatentOnlyModel(
            decoder_directions=decoder_directions, signed=True, missing_latents=True
        )
        bad_shape_model = _LatentOnlyModel(
            decoder_directions=decoder_directions, signed=True, bad_latents_shape=True
        )
        mismatch_model = _LatentOnlyModel(
            decoder_directions=decoder_directions, signed=True, drop_last_sample=True
        )
        evaluation_config = Toy2DEvaluationConfig(
            min_axis_abs_cos=0.0,
            min_activation_corr=0.0,
            min_regime_samples=1,
            hist_num_bins=10,
        )

        # when / then
        with self.assertRaises(ValueError):
            _ = eval_module.evaluate_toy2d_model(
                model=missing_latents_model,
                data=data_missing,  # type: ignore[arg-type]
                evaluation_config=evaluation_config,
                model_name="missing",
                sign_aware=False,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            _ = eval_module.evaluate_toy2d_model(
                model=missing_latents_model,
                data=data,
                evaluation_config=evaluation_config,
                model_name="missing",
                sign_aware=False,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            _ = eval_module.evaluate_toy2d_model(
                model=bad_shape_model,
                data=data,
                evaluation_config=evaluation_config,
                model_name="bad_shape",
                sign_aware=False,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            _ = eval_module.evaluate_toy2d_model(
                model=mismatch_model,
                data=data,
                evaluation_config=evaluation_config,
                model_name="mismatch",
                sign_aware=False,
                device="cpu",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
