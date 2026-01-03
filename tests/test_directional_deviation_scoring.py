import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import torch

from sgnsae.experiments.directional_deviation.scoring import (
    BaselineEvidenceThresholds,
    ScoreThresholds,
    SignAwareLatentThresholds,
    assign_baseline_latent_directions,
    build_threshold_metadata,
    calibrate_baseline_evidence_thresholds,
    calibrate_score_thresholds,
    calibrate_sign_aware_latent_thresholds,
    calculate_baseline_directional_scores,
    calculate_sign_aware_directional_scores,
    extract_detection_signal,
    extract_one_sided_evidence,
    persist_baseline_threshold_artifacts,
    persist_sign_aware_threshold_artifacts,
)
from sgnsae.models import GatedSAE, SAE


class TestDirectionalDeviationScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"

    def _make_sae(
        self,
        sign_aware: bool,
        single_activation_offset: bool,
    ) -> SAE:
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=sign_aware,
            reset_activation_offsets_on_resample=False if sign_aware else None,
            single_activation_offset=single_activation_offset if sign_aware else None,
            device=self.device,
        )
        with torch.no_grad():
            model.W_dec.zero_()
            model.W_dec[0, 0, 0] = 1.0
            model.W_dec[0, 1, 1] = 1.0
            model.b_dec[0] = torch.tensor([0.5, -0.5])
            model.b_enc[0] = torch.tensor([0.1, 0.2])
            if sign_aware:
                if single_activation_offset:
                    model.act_offset[0] = torch.tensor([0.3, -0.2])
                else:
                    model.act_offset_right[0] = torch.tensor([0.3, -0.2])
                    model.act_offset_left[0] = torch.tensor([0.4, 0.0])
        return model

    def _make_gated(
        self,
        sign_aware: bool,
        single_activation_offset: bool,
    ) -> GatedSAE:
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=sign_aware,
            symmetric_magnitude=False,
            reset_activation_offsets_on_resample=False if sign_aware else None,
            single_activation_offset=single_activation_offset if sign_aware else None,
            device=self.device,
        )
        with torch.no_grad():
            model.W_dec.zero_()
            model.W_dec[0, 0, 0] = 1.0
            model.W_dec[0, 1, 1] = 1.0
            model.b_dec[0] = torch.tensor([0.5, -0.5])
            model.b_enc[0] = torch.tensor([0.1, 0.2])
            model.log_alpha_gate[0] = torch.tensor([0.0, 0.0])
            model.r_mag[0] = torch.tensor([0.0, 0.0])
            model.b_mag[0] = torch.tensor([0.0, 0.0])
            if sign_aware:
                if single_activation_offset:
                    model.act_offset[0] = torch.tensor([0.3, -0.2])
                else:
                    model.act_offset_right[0] = torch.tensor([0.3, -0.2])
                    model.act_offset_left[0] = torch.tensor([0.4, 0.0])
        return model

    def test_extract_detection_signal_sae_matches_expected_preactivations(self) -> None:
        # given
        model = self._make_sae(sign_aware=False, single_activation_offset=False)
        x = torch.tensor([[1.0, 2.0], [-1.0, 0.0]], dtype=torch.float32)

        # when
        pi = extract_detection_signal(model, x)

        # then
        expected = torch.tensor([[0.6, 2.7], [-1.4, 0.7]], dtype=torch.float32)
        self.assertTrue(torch.allclose(pi, expected, atol=1e-7))

    def test_extract_detection_signal_gated_matches_expected_pi_gate(self) -> None:
        # given
        model = self._make_gated(sign_aware=False, single_activation_offset=False)
        x = torch.tensor([[1.0, 2.0], [-1.0, 0.0]], dtype=torch.float32)

        # when
        pi = extract_detection_signal(model, x)

        # then
        expected = torch.tensor([[0.6, 2.7], [-1.4, 0.7]], dtype=torch.float32)
        self.assertTrue(torch.allclose(pi, expected, atol=1e-7))

    def test_extract_detection_signal_raises_for_invalid_input_dimension(self) -> None:
        # given
        model = self._make_sae(sign_aware=False, single_activation_offset=False)
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = extract_detection_signal(model, x)

    def test_extract_detection_signal_raises_for_unsupported_model_type(self) -> None:
        # given
        class UnknownModel(torch.nn.Module):
            pass

        x = torch.zeros(2, 2)

        # when & then
        with self.assertRaises(ValueError):
            _ = extract_detection_signal(UnknownModel(), x)

    def test_extract_detection_signal_raises_for_missing_pi_gate_key(self) -> None:
        # given
        class NoPiGateGatedSAE(GatedSAE):
            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {}

        model = NoPiGateGatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            symmetric_magnitude=False,
            device=self.device,
        )
        x = torch.zeros(2, 2)

        # when & then
        with self.assertRaises(ValueError):
            _ = extract_detection_signal(model, x)

    def test_extract_one_sided_evidence_matches_model_latents_for_sae(self) -> None:
        # given
        model = self._make_sae(sign_aware=False, single_activation_offset=False)
        x = torch.tensor([[1.0, 2.0], [-1.0, 0.0]], dtype=torch.float32)

        # when
        evidence = extract_one_sided_evidence(model, x)

        # then
        expected_preactivations = torch.tensor([[0.6, 2.7], [-1.4, 0.7]])
        expected = torch.relu(expected_preactivations)
        self.assertTrue(torch.allclose(evidence, expected, atol=1e-7))

    def test_extract_one_sided_evidence_raises_for_sign_aware_model(self) -> None:
        # given
        model = self._make_sae(sign_aware=True, single_activation_offset=True)
        x = torch.zeros(2, 2)

        # when & then
        with self.assertRaises(ValueError):
            _ = extract_one_sided_evidence(model, x)

    def test_extract_one_sided_evidence_raises_for_missing_latents_key(self) -> None:
        # given
        class NoLatents(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {}

        # when & then
        with self.assertRaises(ValueError):
            _ = extract_one_sided_evidence(NoLatents(), torch.zeros(2, 2))

    def test_extract_one_sided_evidence_raises_for_negative_evidence(self) -> None:
        # given
        class NegativeLatents(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                batch_size = x.shape[0]
                latents = torch.zeros(1, batch_size, 2, dtype=torch.float32)
                latents[0, 0, 0] = -0.1
                return {"latents": latents}

        model = NegativeLatents()
        x = torch.zeros(2, 2, dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = extract_one_sided_evidence(model, x)

    def test_assign_baseline_latent_directions_single_channel_expected(self) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device=self.device,
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor(
                [[0.1, 0.3], [-0.5, 0.2], [0.2, -0.1]], dtype=torch.float32
            )

        # when
        directions = assign_baseline_latent_directions(
            model, shift_mode="single_channel"
        )

        # then
        self.assertTrue(
            torch.equal(directions, torch.tensor([-1, 1], dtype=torch.int64))
        )

    def test_assign_baseline_latent_directions_single_channel_tie_breaks_and_maps_zero_to_plus_one(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device=self.device,
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=torch.float32
            )

        # when
        directions = assign_baseline_latent_directions(
            model, shift_mode="single_channel"
        )

        # then
        self.assertTrue(
            torch.equal(directions, torch.tensor([1, 1], dtype=torch.int64))
        )

    def test_assign_baseline_latent_directions_subset_expected(self) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device=self.device,
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor(
                [[1.0, -1.0], [0.0, 1.0], [-2.0, 0.5]], dtype=torch.float32
            )

        # when
        directions = assign_baseline_latent_directions(
            model, shift_mode="subset", subset_indices=[0, 2]
        )

        # then
        self.assertTrue(
            torch.equal(directions, torch.tensor([-1, -1], dtype=torch.int64))
        )

    def test_assign_baseline_latent_directions_subset_raises_for_missing_indices(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device=self.device,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = assign_baseline_latent_directions(model, shift_mode="subset")

    def test_assign_baseline_latent_directions_subset_raises_for_empty_indices(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device=self.device,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = assign_baseline_latent_directions(
                model,
                shift_mode="subset",
                subset_indices=[],
            )

    def test_assign_baseline_latent_directions_raises_for_unsupported_shift_mode(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device=self.device,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = assign_baseline_latent_directions(  # type: ignore[arg-type]
                model, shift_mode="other_mode"
            )

    def test_assign_baseline_latent_directions_raises_for_missing_decoder_attribute(
        self,
    ) -> None:
        # given
        class MissingDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1

        # when & then
        with self.assertRaises(ValueError):
            _ = assign_baseline_latent_directions(
                MissingDecoder(),
                shift_mode="single_channel",
            )

    def test_assign_baseline_latent_directions_raises_for_decoder_not_tensor(
        self,
    ) -> None:
        # given
        class DecoderNotTensor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.W_dec = "not_a_tensor"

        # when & then
        with self.assertRaises(ValueError):
            _ = assign_baseline_latent_directions(
                DecoderNotTensor(),
                shift_mode="single_channel",
            )

    def test_assign_baseline_latent_directions_raises_for_missing_instance_count(
        self,
    ) -> None:
        # given
        class MissingInstanceCount(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.W_dec = torch.zeros(1, 2, 2, dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = assign_baseline_latent_directions(
                MissingInstanceCount(),
                shift_mode="single_channel",
            )

    def test_assign_baseline_latent_directions_raises_for_multiple_instances(
        self,
    ) -> None:
        # given
        class MultipleInstances(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 2
                self.W_dec = torch.zeros(2, 2, 2, dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = assign_baseline_latent_directions(
                MultipleInstances(),
                shift_mode="single_channel",
            )

    def test_calibrate_sign_aware_latent_thresholds_matches_expected(self) -> None:
        # given
        pi_validation = torch.tensor(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]], dtype=torch.float32
        )

        # when
        thresholds = calibrate_sign_aware_latent_thresholds(
            pi_validation=pi_validation,
            latent_quantile=0.5,
            min_validation_samples=3,
        )

        # then
        self.assertTrue(
            torch.allclose(thresholds.tau_plus, torch.tensor([0.0, 1.0]), atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(thresholds.tau_minus, torch.tensor([0.0, -1.0]), atol=1e-7)
        )

    def test_calibrate_sign_aware_latent_thresholds_raises_for_insufficient_samples(
        self,
    ) -> None:
        # given
        pi_validation = torch.zeros(2, 3)

        # when & then
        with self.assertRaises(ValueError):
            _ = calibrate_sign_aware_latent_thresholds(
                pi_validation=pi_validation,
                latent_quantile=0.5,
                min_validation_samples=3,
            )

    def test_calibrate_sign_aware_latent_thresholds_raises_for_invalid_quantile(
        self,
    ) -> None:
        # given
        pi_validation = torch.zeros(3, 2)

        # when & then
        with self.assertRaises(ValueError):
            _ = calibrate_sign_aware_latent_thresholds(
                pi_validation=pi_validation,
                latent_quantile=1.0,
                min_validation_samples=1,
            )

    def test_calibrate_sign_aware_latent_thresholds_raises_for_invalid_pi_dimension(
        self,
    ) -> None:
        # given
        pi_validation = torch.zeros(3, 2, 2, dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = calibrate_sign_aware_latent_thresholds(
                pi_validation=pi_validation,
                latent_quantile=0.5,
                min_validation_samples=1,
            )

    def test_calibrate_baseline_evidence_thresholds_matches_expected(self) -> None:
        # given
        evidence_validation = torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32
        )
        latent_directions = torch.tensor([1, -1], dtype=torch.int64)

        # when
        thresholds = calibrate_baseline_evidence_thresholds(
            evidence_validation=evidence_validation,
            latent_directions=latent_directions,
            latent_quantile=0.5,
            min_validation_samples=3,
        )

        # then
        self.assertTrue(
            torch.allclose(thresholds.evidence_tau, torch.tensor([2.0, 3.0]), atol=1e-7)
        )
        self.assertTrue(torch.equal(thresholds.latent_directions, latent_directions))

    def test_calibrate_baseline_evidence_thresholds_raises_for_direction_length_mismatch(
        self,
    ) -> None:
        # given
        evidence_validation = torch.zeros(3, 2)
        latent_directions = torch.ones(3, dtype=torch.int64)

        # when & then
        with self.assertRaises(ValueError):
            _ = calibrate_baseline_evidence_thresholds(
                evidence_validation=evidence_validation,
                latent_directions=latent_directions,
                latent_quantile=0.5,
                min_validation_samples=1,
            )

    def test_calculate_sign_aware_directional_scores_matches_expected(self) -> None:
        # given
        pi = torch.tensor([[0.6, 2.7], [-1.4, 0.7]], dtype=torch.float32)
        thresholds = SignAwareLatentThresholds(
            tau_plus=torch.tensor([0.0, 1.0], dtype=torch.float32),
            tau_minus=torch.tensor([0.0, -1.0], dtype=torch.float32),
        )

        # when
        score_high, score_low = calculate_sign_aware_directional_scores(
            pi=pi, latent_thresholds=thresholds
        )

        # then
        self.assertTrue(torch.allclose(score_high, torch.tensor([2.3, 0.0]), atol=1e-7))
        self.assertTrue(torch.allclose(score_low, torch.tensor([0.0, 1.7]), atol=1e-7))

    def test_calculate_sign_aware_directional_scores_raises_for_width_mismatch(
        self,
    ) -> None:
        # given
        pi = torch.zeros(2, 2)
        thresholds = SignAwareLatentThresholds(
            tau_plus=torch.zeros(3),
            tau_minus=torch.zeros(2),
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = calculate_sign_aware_directional_scores(
                pi=pi, latent_thresholds=thresholds
            )

    def test_calculate_sign_aware_directional_scores_raises_for_tau_minus_width_mismatch(
        self,
    ) -> None:
        # given
        pi = torch.zeros(2, 2)
        thresholds = SignAwareLatentThresholds(
            tau_plus=torch.zeros(2),
            tau_minus=torch.zeros(3),
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = calculate_sign_aware_directional_scores(
                pi=pi, latent_thresholds=thresholds
            )

    def test_calculate_baseline_directional_scores_matches_expected(self) -> None:
        # given
        evidence = torch.tensor([[2.0, 0.5], [0.0, 4.0]], dtype=torch.float32)
        thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([1.0, 3.0], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )

        # when
        score_high, score_low = calculate_baseline_directional_scores(
            evidence, thresholds
        )

        # then
        self.assertTrue(torch.allclose(score_high, torch.tensor([1.0, 0.0]), atol=1e-7))
        self.assertTrue(torch.allclose(score_low, torch.tensor([0.0, 1.0]), atol=1e-7))

    def test_calibrate_score_thresholds_matches_expected_quantiles(self) -> None:
        # given
        scores_high = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        scores_low = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)

        # when
        thresholds = calibrate_score_thresholds(
            score_high_validation=scores_high,
            score_low_validation=scores_low,
            score_fpr=0.25,
            min_validation_samples=4,
        )

        # then
        self.assertAlmostEqual(thresholds.theta_high, 2.25, places=7)
        self.assertAlmostEqual(thresholds.theta_low, 2.25, places=7)
        self.assertAlmostEqual(thresholds.score_quantile, 0.75, places=7)

    def test_calibrate_score_thresholds_raises_for_invalid_fpr_and_length_mismatch(
        self,
    ) -> None:
        # given
        scores_high = torch.zeros(3)
        scores_low = torch.zeros(2)

        # when & then
        with self.assertRaises(ValueError):
            _ = calibrate_score_thresholds(
                score_high_validation=scores_high,
                score_low_validation=scores_low,
                score_fpr=0.5,
                min_validation_samples=1,
            )
        with self.assertRaises(ValueError):
            _ = calibrate_score_thresholds(
                score_high_validation=scores_high,
                score_low_validation=scores_high,
                score_fpr=1.0,
                min_validation_samples=1,
            )

    def test_build_threshold_metadata_includes_learned_thresholds_sign_aware(
        self,
    ) -> None:
        # given
        model = self._make_gated(sign_aware=True, single_activation_offset=False)
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when
        metadata = build_threshold_metadata(
            model=model,
            scoring_mode="sign_aware",
            latent_quantile=0.99,
            score_thresholds=score_thresholds,
            shift_mode="single_channel",
        )

        # then
        self.assertIsNotNone(metadata["learned_model_thresholds"])
        self.assertIn("delta_plus", metadata["learned_model_thresholds"])
        self.assertIn("delta_minus", metadata["learned_model_thresholds"])
        self.assertIsNone(metadata["baseline_direction_assignment"])

    def test_build_threshold_metadata_extracts_learned_thresholds_single_activation_offset(
        self,
    ) -> None:
        # given
        model = self._make_sae(sign_aware=True, single_activation_offset=True)
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when
        metadata = build_threshold_metadata(
            model=model,
            scoring_mode="sign_aware",
            latent_quantile=0.99,
            score_thresholds=score_thresholds,
            shift_mode="single_channel",
        )

        # then
        learned_thresholds = metadata["learned_model_thresholds"]
        self.assertIsNotNone(learned_thresholds)
        delta_plus = torch.tensor(learned_thresholds["delta_plus"], dtype=torch.float32)
        delta_minus = torch.tensor(
            learned_thresholds["delta_minus"], dtype=torch.float32
        )
        self.assertTrue(
            torch.allclose(delta_plus, torch.tensor([0.3, 0.0], dtype=torch.float32))
        )
        self.assertTrue(
            torch.allclose(delta_minus, torch.tensor([0.3, 0.0], dtype=torch.float32))
        )

    def test_build_threshold_metadata_raises_for_missing_act_offset_in_single_offset_sign_aware_model(
        self,
    ) -> None:
        # given
        class MissingActOffset(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sign_aware = True
                self.single_activation_offset = True

        model = MissingActOffset()
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = build_threshold_metadata(
                model=model,
                scoring_mode="sign_aware",
                latent_quantile=0.99,
                score_thresholds=score_thresholds,
                shift_mode="single_channel",
            )

    def test_build_threshold_metadata_raises_for_non_tensor_act_offset_in_single_offset_sign_aware_model(
        self,
    ) -> None:
        # given
        class WrongTypeActOffset(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sign_aware = True
                self.single_activation_offset = True
                self.act_offset = "not_a_tensor"

        model = WrongTypeActOffset()
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = build_threshold_metadata(
                model=model,
                scoring_mode="sign_aware",
                latent_quantile=0.99,
                score_thresholds=score_thresholds,
                shift_mode="single_channel",
            )

    def test_build_threshold_metadata_raises_for_missing_left_right_offsets_in_sign_aware_model(
        self,
    ) -> None:
        # given
        class MissingLeftRightOffsets(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sign_aware = True
                self.single_activation_offset = False

        model = MissingLeftRightOffsets()
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = build_threshold_metadata(
                model=model,
                scoring_mode="sign_aware",
                latent_quantile=0.99,
                score_thresholds=score_thresholds,
                shift_mode="single_channel",
            )

    def test_build_threshold_metadata_raises_for_non_tensor_left_right_offsets_in_sign_aware_model(
        self,
    ) -> None:
        # given
        class WrongTypeLeftRightOffsets(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sign_aware = True
                self.single_activation_offset = False
                self.act_offset_right = "not_a_tensor"
                self.act_offset_left = torch.zeros(1, 2, dtype=torch.float32)

        model = WrongTypeLeftRightOffsets()
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = build_threshold_metadata(
                model=model,
                scoring_mode="sign_aware",
                latent_quantile=0.99,
                score_thresholds=score_thresholds,
                shift_mode="single_channel",
            )

    def test_build_threshold_metadata_includes_baseline_direction_assignment(
        self,
    ) -> None:
        # given
        model = self._make_sae(sign_aware=False, single_activation_offset=False)
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when
        metadata = build_threshold_metadata(
            model=model,
            scoring_mode="baseline",
            latent_quantile=0.99,
            score_thresholds=score_thresholds,
            shift_mode="subset",
            subset_indices=[0, 1],
        )

        # then
        self.assertIsNone(metadata["learned_model_thresholds"])
        self.assertEqual(
            metadata["baseline_direction_assignment"]["rule"],
            "sign_of_decoder_dot_subset_mask",
        )
        self.assertEqual(
            metadata["baseline_direction_assignment"]["subset_indices"], [0, 1]
        )

    def test_build_threshold_metadata_baseline_single_channel_includes_direction_assignment(
        self,
    ) -> None:
        # given
        model = self._make_sae(sign_aware=False, single_activation_offset=False)
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when
        metadata = build_threshold_metadata(
            model=model,
            scoring_mode="baseline",
            latent_quantile=0.99,
            score_thresholds=score_thresholds,
            shift_mode="single_channel",
        )

        # then
        self.assertEqual(
            metadata["baseline_direction_assignment"]["rule"],
            "argmax_abs_decoder_component_sign",
        )
        self.assertIsNone(metadata["baseline_direction_assignment"]["subset_indices"])

    def test_build_threshold_metadata_baseline_raises_for_unsupported_shift_mode(
        self,
    ) -> None:
        # given
        model = self._make_sae(sign_aware=False, single_activation_offset=False)
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = build_threshold_metadata(  # type: ignore[arg-type]
                model=model,
                scoring_mode="baseline",
                latent_quantile=0.99,
                score_thresholds=score_thresholds,
                shift_mode="other_mode",
            )

    def test_build_threshold_metadata_raises_for_invalid_latent_quantile(self) -> None:
        # given
        model = self._make_sae(sign_aware=False, single_activation_offset=False)
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=2.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = build_threshold_metadata(
                model=model,
                scoring_mode="baseline",
                latent_quantile=0.0,
                score_thresholds=score_thresholds,
                shift_mode="single_channel",
            )

    def test_persist_threshold_artifacts_write_expected_files(self) -> None:
        # given
        sign_aware_latent_thresholds = SignAwareLatentThresholds(
            tau_plus=torch.tensor([1.0, 2.0], dtype=torch.float32),
            tau_minus=torch.tensor([3.0, 4.0], dtype=torch.float32),
        )
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([1.0, 2.0], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=1.25,
            theta_low=2.5,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        metadata: Dict[str, Any] = {"key": "value"}

        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            sign_dir = os.path.join(tmpdir, "sign")
            baseline_dir = os.path.join(tmpdir, "baseline")
            persist_sign_aware_threshold_artifacts(
                thresholds_dir=sign_dir,
                latent_thresholds=sign_aware_latent_thresholds,
                score_thresholds=score_thresholds,
                metadata=metadata,
            )
            persist_baseline_threshold_artifacts(
                thresholds_dir=baseline_dir,
                baseline_thresholds=baseline_thresholds,
                score_thresholds=score_thresholds,
                metadata=metadata,
            )

            # then
            self.assertTrue(os.path.exists(os.path.join(sign_dir, "tau_plus.pt")))
            self.assertTrue(os.path.exists(os.path.join(sign_dir, "tau_minus.pt")))
            self.assertTrue(
                os.path.exists(os.path.join(sign_dir, "score_thresholds.json"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(sign_dir, "thresholds_metadata.json"))
            )

            self.assertTrue(
                os.path.exists(os.path.join(baseline_dir, "evidence_tau.pt"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(baseline_dir, "latent_directions.pt"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(baseline_dir, "score_thresholds.json"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(baseline_dir, "thresholds_metadata.json"))
            )

            loaded_tau_plus = torch.load(os.path.join(sign_dir, "tau_plus.pt"))
            loaded_tau_minus = torch.load(os.path.join(sign_dir, "tau_minus.pt"))
            self.assertTrue(
                torch.equal(loaded_tau_plus, sign_aware_latent_thresholds.tau_plus)
            )
            self.assertTrue(
                torch.equal(loaded_tau_minus, sign_aware_latent_thresholds.tau_minus)
            )

            with open(os.path.join(sign_dir, "score_thresholds.json"), "r") as f:
                saved_thresholds = json.load(f)
            self.assertEqual(
                saved_thresholds["theta_high"], score_thresholds.theta_high
            )
            self.assertEqual(saved_thresholds["theta_low"], score_thresholds.theta_low)

            with open(os.path.join(sign_dir, "thresholds_metadata.json"), "r") as f:
                saved_metadata = json.load(f)
            self.assertEqual(saved_metadata, metadata)

            loaded_evidence_tau = torch.load(
                os.path.join(baseline_dir, "evidence_tau.pt")
            )
            loaded_latent_directions = torch.load(
                os.path.join(baseline_dir, "latent_directions.pt")
            )
            self.assertTrue(
                torch.allclose(loaded_evidence_tau, baseline_thresholds.evidence_tau)
            )
            self.assertTrue(
                torch.equal(
                    loaded_latent_directions, baseline_thresholds.latent_directions
                )
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
