import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import torch

from sgnsae.experiments.directional_deviation.data import (
    LABEL_TYPE_HIGH,
    LABEL_TYPE_LOW,
    DirectionalDeviationAnomalyBatch,
)
from sgnsae.experiments.directional_deviation.evaluation import (
    DirectionalDeviationEvaluationInputs,
    evaluate_and_persist_directional_deviation,
    evaluate_directional_deviation,
    persist_comparison_summary,
)
from sgnsae.experiments.directional_deviation.scoring import (
    BaselineEvidenceThresholds,
    ScoreThresholds,
    SignAwareLatentThresholds,
    calibrate_score_thresholds,
    calibrate_sign_aware_latent_thresholds,
    calculate_sign_aware_directional_scores,
    extract_detection_signal,
)
from sgnsae.models import GatedSAE


class TestDirectionalDeviationEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"

    def _make_gated_model(self) -> GatedSAE:
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            symmetric_magnitude=False,
            device=self.device,
        )
        with torch.no_grad():
            model.W_dec.zero_()
            model.W_dec[0, 0, 0] = 1.0
            model.W_dec[0, 1, 1] = 1.0
            model.b_dec[0] = torch.tensor([0.5, -0.5], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.1, 0.2], dtype=torch.float32)
            model.log_alpha_gate[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.r_mag[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.b_mag[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
        return model

    def test_evaluate_directional_deviation_sign_aware_materialized_returns_expected_metrics(
        self,
    ) -> None:
        # given
        model = self._make_gated_model()
        latent_thresholds = SignAwareLatentThresholds(
            tau_plus=torch.tensor([10.0, 10.0], dtype=torch.float32),
            tau_minus=torch.tensor([10.0, 10.0], dtype=torch.float32),
        )
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=1.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.tensor([[0.4, -0.7], [0.4, -0.7]], dtype=torch.float32)

        x_test_anomaly = torch.tensor(
            [
                [20.0, -0.7],
                [-20.0, -0.7],
                [0.4, 20.0],
                [0.4, -20.0],
            ],
            dtype=torch.float32,
        )
        labels_type = torch.tensor(
            [LABEL_TYPE_HIGH, LABEL_TYPE_LOW, LABEL_TYPE_HIGH, LABEL_TYPE_LOW],
            dtype=torch.int64,
        )
        labels_channel = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        anomaly_manifest: Dict[str, Any] = {
            "shift_mode": "single_channel",
            "channels_to_evaluate": [0, 1],
            "num_channels": 2,
        }
        anomaly_data = {
            "x_test_anomaly": x_test_anomaly,
            "labels_type": labels_type,
            "labels_channel": labels_channel,
            "anomaly_stream": None,
            "anomaly_manifest": anomaly_manifest,
        }

        # when
        evaluation = evaluate_directional_deviation(
            model=model,
            model_name="gated",
            scoring_mode="sign_aware",
            anomaly_data=anomaly_data,
            x_test_normal=x_test_normal,
            score_thresholds=score_thresholds,
            latent_thresholds=latent_thresholds,
            evaluation_batch_size=2,
            coverage_recall_target=0.8,
        )

        # then
        self.assertAlmostEqual(evaluation["metrics"]["recall_high"], 1.0, places=7)
        self.assertAlmostEqual(evaluation["metrics"]["recall_low"], 1.0, places=7)
        self.assertAlmostEqual(evaluation["metrics"]["fpr_high"], 0.0, places=7)
        self.assertAlmostEqual(evaluation["metrics"]["fpr_low"], 0.0, places=7)

        per_channel = evaluation["per_channel"]
        self.assertEqual(per_channel["channels"], [0, 1])
        self.assertEqual(per_channel["num_high"], [1, 1])
        self.assertEqual(per_channel["num_low"], [1, 1])
        self.assertEqual(per_channel["covered_channels_count"], 2)
        self.assertAlmostEqual(per_channel["covered_channels_fraction"], 1.0, places=7)
        self.assertEqual(per_channel["recall_high"], [1.0, 1.0])
        self.assertEqual(per_channel["recall_low"], [1.0, 1.0])

    def test_evaluate_directional_deviation_baseline_streaming_returns_expected_metrics(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.zeros(2, 2, dtype=torch.float32)

        anomaly_batches: List[DirectionalDeviationAnomalyBatch] = [
            DirectionalDeviationAnomalyBatch(
                x=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                labels_type=torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
                labels_channel=torch.tensor([0], dtype=torch.int64),
                labels_mask=None,
            ),
            DirectionalDeviationAnomalyBatch(
                x=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                labels_type=torch.tensor([LABEL_TYPE_LOW], dtype=torch.int64),
                labels_channel=torch.tensor([0], dtype=torch.int64),
                labels_mask=None,
            ),
            DirectionalDeviationAnomalyBatch(
                x=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                labels_type=torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
                labels_channel=torch.tensor([1], dtype=torch.int64),
                labels_mask=None,
            ),
            DirectionalDeviationAnomalyBatch(
                x=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                labels_type=torch.tensor([LABEL_TYPE_LOW], dtype=torch.int64),
                labels_channel=torch.tensor([1], dtype=torch.int64),
                labels_mask=None,
            ),
        ]
        anomaly_data = {
            "anomaly_stream": anomaly_batches,
            "anomaly_manifest": {
                "shift_mode": "single_channel",
                "channels_to_evaluate": [0, 1],
                "num_channels": 2,
            },
        }

        # when
        evaluation = evaluate_directional_deviation(
            model=model,
            model_name="baseline",
            scoring_mode="baseline",
            anomaly_data=anomaly_data,
            x_test_normal=x_test_normal,
            score_thresholds=score_thresholds,
            baseline_thresholds=baseline_thresholds,
            evaluation_batch_size=2,
            coverage_recall_target=0.8,
        )

        # then
        self.assertAlmostEqual(evaluation["metrics"]["recall_high"], 1.0, places=7)
        self.assertAlmostEqual(evaluation["metrics"]["recall_low"], 1.0, places=7)
        self.assertAlmostEqual(evaluation["metrics"]["fpr_high"], 0.0, places=7)
        self.assertAlmostEqual(evaluation["metrics"]["fpr_low"], 0.0, places=7)
        self.assertIsNotNone(evaluation["per_channel"])

    def test_evaluate_directional_deviation_sign_aware_calibrated_thresholds_matches_fpr_target_and_detects_anomalies(
        self,
    ) -> None:
        # given
        torch.manual_seed(0)
        model = GatedSAE(
            n_instances=1,
            d_in=1,
            d_hidden=1,
            resample_latents_threshold=1,
            sign_aware=False,
            symmetric_magnitude=False,
            device=self.device,
        )
        with torch.no_grad():
            model.W_dec.zero_()
            model.W_dec[0, 0, 0] = 1.0
            model.b_dec[0] = torch.tensor([0.0], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.0], dtype=torch.float32)
            model.log_alpha_gate[0] = torch.tensor([0.0], dtype=torch.float32)
            model.r_mag[0] = torch.tensor([0.0], dtype=torch.float32)
            model.b_mag[0] = torch.tensor([0.0], dtype=torch.float32)

        num_val = 5000
        num_test_normal = 5000
        x_val = torch.randn((num_val, 1), dtype=torch.float32)
        x_test_normal = torch.randn((num_test_normal, 1), dtype=torch.float32)

        latent_quantile = 0.2
        score_fpr = 0.05

        pi_val = extract_detection_signal(model, x_val)
        latent_thresholds = calibrate_sign_aware_latent_thresholds(
            pi_validation=pi_val,
            latent_quantile=latent_quantile,
            min_validation_samples=num_val,
        )
        score_high_val, score_low_val = calculate_sign_aware_directional_scores(
            pi=pi_val, latent_thresholds=latent_thresholds
        )
        score_thresholds = calibrate_score_thresholds(
            score_high_validation=score_high_val,
            score_low_validation=score_low_val,
            score_fpr=score_fpr,
            min_validation_samples=num_val,
        )

        num_high = 50
        num_low = 50
        x_test_anomaly = torch.cat(
            [
                torch.full((num_high, 1), 6.0, dtype=torch.float32),
                torch.full((num_low, 1), -6.0, dtype=torch.float32),
            ],
            dim=0,
        )
        labels_type = torch.tensor(
            [LABEL_TYPE_HIGH] * num_high + [LABEL_TYPE_LOW] * num_low, dtype=torch.int64
        )
        labels_channel = torch.zeros(num_high + num_low, dtype=torch.int64)
        anomaly_manifest: Dict[str, Any] = {
            "shift_mode": "single_channel",
            "channels_to_evaluate": [0],
            "num_channels": 1,
        }
        anomaly_data = {
            "x_test_anomaly": x_test_anomaly,
            "labels_type": labels_type,
            "labels_channel": labels_channel,
            "anomaly_stream": None,
            "anomaly_manifest": anomaly_manifest,
        }

        # when
        evaluation = evaluate_directional_deviation(
            model=model,
            model_name="gated",
            scoring_mode="sign_aware",
            anomaly_data=anomaly_data,
            x_test_normal=x_test_normal,
            score_thresholds=score_thresholds,
            latent_thresholds=latent_thresholds,
            evaluation_batch_size=512,
            coverage_recall_target=0.8,
        )

        # then
        self.assertAlmostEqual(evaluation["metrics"]["recall_high"], 1.0, places=7)
        self.assertAlmostEqual(evaluation["metrics"]["recall_low"], 1.0, places=7)
        self.assertAlmostEqual(
            evaluation["metrics"]["fpr_high"], score_fpr, delta=0.015
        )
        self.assertAlmostEqual(evaluation["metrics"]["fpr_low"], score_fpr, delta=0.015)

    def test_evaluate_directional_deviation_subset_mode_does_not_compute_per_channel(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.zeros(1, 2, dtype=torch.float32)
        x_test_anomaly = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels_type = torch.tensor([LABEL_TYPE_HIGH, LABEL_TYPE_LOW], dtype=torch.int64)
        labels_channel = torch.tensor([-1, -1], dtype=torch.int64)
        anomaly_data = {
            "x_test_anomaly": x_test_anomaly,
            "labels_type": labels_type,
            "labels_channel": labels_channel,
            "anomaly_stream": None,
            "anomaly_manifest": {"shift_mode": "subset", "num_channels": 2},
        }

        # when
        evaluation = evaluate_directional_deviation(
            model=model,
            model_name="baseline",
            scoring_mode="baseline",
            anomaly_data=anomaly_data,
            x_test_normal=x_test_normal,
            score_thresholds=score_thresholds,
            baseline_thresholds=baseline_thresholds,
            evaluation_batch_size=1,
            coverage_recall_target=0.8,
        )

        # then
        self.assertIsNone(evaluation["per_channel"])
        self.assertIsNone(evaluation["per_direction"])

    def test_evaluate_directional_deviation_feature_direction_computes_per_direction(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.zeros(2, 2, dtype=torch.float32)

        x_test_anomaly = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        labels_type = torch.tensor(
            [LABEL_TYPE_HIGH, LABEL_TYPE_LOW, LABEL_TYPE_HIGH, LABEL_TYPE_LOW],
            dtype=torch.int64,
        )
        labels_channel = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        anomaly_data = {
            "x_test_anomaly": x_test_anomaly,
            "labels_type": labels_type,
            "labels_channel": labels_channel,
            "anomaly_stream": None,
            "anomaly_manifest": {
                "shift_mode": "feature_direction",
                "num_channels": 2,
                "num_directions": 2,
                "direction_indices_to_evaluate": [0, 1],
            },
        }

        # when
        evaluation = evaluate_directional_deviation(
            model=model,
            model_name="baseline",
            scoring_mode="baseline",
            anomaly_data=anomaly_data,
            x_test_normal=x_test_normal,
            score_thresholds=score_thresholds,
            baseline_thresholds=baseline_thresholds,
            evaluation_batch_size=2,
            coverage_recall_target=0.8,
        )

        # then
        self.assertIsNone(evaluation["per_channel"])
        self.assertEqual(evaluation["target_kind"], "direction")
        self.assertEqual(evaluation["num_targets"], 2)

        per_direction = evaluation["per_direction"]
        self.assertEqual(per_direction["directions"], [0, 1])
        self.assertEqual(per_direction["num_high"], [1, 1])
        self.assertEqual(per_direction["num_low"], [1, 1])
        self.assertEqual(per_direction["covered_directions_count"], 2)
        self.assertAlmostEqual(
            per_direction["covered_directions_fraction"], 1.0, places=7
        )
        self.assertEqual(per_direction["recall_high"], [1.0, 1.0])
        self.assertEqual(per_direction["recall_low"], [1.0, 1.0])

    def test_evaluate_directional_deviation_feature_direction_raises_for_missing_num_directions(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.0], dtype=torch.float32),
            latent_directions=torch.tensor([1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.zeros(1, 1, dtype=torch.float32)
        anomaly_data = {
            "x_test_anomaly": torch.ones(1, 1, dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([0], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {
                "shift_mode": "feature_direction",
                "num_channels": 1,
                "num_directions": 0,
            },
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=x_test_normal,
                score_thresholds=score_thresholds,
                baseline_thresholds=baseline_thresholds,
                evaluation_batch_size=1,
                coverage_recall_target=0.8,
            )

    def test_evaluate_directional_deviation_feature_direction_raises_for_negative_direction_index(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.0], dtype=torch.float32),
            latent_directions=torch.tensor([1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.zeros(1, 1, dtype=torch.float32)
        anomaly_data = {
            "x_test_anomaly": torch.ones(1, 1, dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([-1], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {
                "shift_mode": "feature_direction",
                "num_channels": 1,
                "num_directions": 1,
                "direction_indices_to_evaluate": [0],
            },
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=x_test_normal,
                score_thresholds=score_thresholds,
                baseline_thresholds=baseline_thresholds,
                evaluation_batch_size=1,
                coverage_recall_target=0.8,
            )

    def test_evaluate_directional_deviation_feature_direction_raises_for_empty_direction_indices_to_evaluate(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.0], dtype=torch.float32),
            latent_directions=torch.tensor([1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.zeros(1, 1, dtype=torch.float32)
        anomaly_data = {
            "x_test_anomaly": torch.ones(1, 1, dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([0], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {
                "shift_mode": "feature_direction",
                "num_channels": 1,
                "num_directions": 1,
                "direction_indices_to_evaluate": [],
            },
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=x_test_normal,
                score_thresholds=score_thresholds,
                baseline_thresholds=baseline_thresholds,
                evaluation_batch_size=1,
                coverage_recall_target=0.8,
            )

    def test_evaluate_directional_deviation_raises_for_invalid_batch_size(self) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.zeros(1, 2, dtype=torch.float32)
        anomaly_data = {
            "x_test_anomaly": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([-1], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {"shift_mode": "subset", "num_channels": 2},
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=x_test_normal,
                score_thresholds=score_thresholds,
                baseline_thresholds=baseline_thresholds,
                evaluation_batch_size=0,
            )

    def test_evaluate_directional_deviation_raises_for_invalid_coverage_target(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        x_test_normal = torch.zeros(1, 2, dtype=torch.float32)
        anomaly_data = {
            "x_test_anomaly": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([-1], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {"shift_mode": "subset", "num_channels": 2},
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=x_test_normal,
                score_thresholds=score_thresholds,
                baseline_thresholds=baseline_thresholds,
                evaluation_batch_size=1,
                coverage_recall_target=1.1,
            )

    def test_evaluate_directional_deviation_raises_for_missing_anomaly_manifest(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data={"x_test_anomaly": torch.zeros(1, 2)},
                x_test_normal=torch.zeros(1, 2),
                score_thresholds=ScoreThresholds(
                    theta_high=0.1,
                    theta_low=0.1,
                    score_fpr=0.01,
                    score_quantile=0.99,
                ),
                baseline_thresholds=BaselineEvidenceThresholds(
                    evidence_tau=torch.tensor([0.0, 0.0]),
                    latent_directions=torch.tensor([1, -1]),
                ),
                evaluation_batch_size=1,
            )

    def test_evaluate_directional_deviation_raises_for_non_dict_anomaly_manifest(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data={"anomaly_manifest": "not_a_dict"},
                x_test_normal=torch.zeros(1, 2),
                score_thresholds=ScoreThresholds(
                    theta_high=0.1,
                    theta_low=0.1,
                    score_fpr=0.01,
                    score_quantile=0.99,
                ),
                baseline_thresholds=BaselineEvidenceThresholds(
                    evidence_tau=torch.tensor([0.0, 0.0]),
                    latent_directions=torch.tensor([1, -1]),
                ),
                evaluation_batch_size=1,
            )

    def test_evaluate_directional_deviation_raises_for_missing_anomaly_source(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data={
                    "anomaly_manifest": {"shift_mode": "subset", "num_channels": 2}
                },
                x_test_normal=torch.zeros(1, 2),
                score_thresholds=ScoreThresholds(
                    theta_high=0.1,
                    theta_low=0.1,
                    score_fpr=0.01,
                    score_quantile=0.99,
                ),
                baseline_thresholds=BaselineEvidenceThresholds(
                    evidence_tau=torch.tensor([0.0, 0.0]),
                    latent_directions=torch.tensor([1, -1]),
                ),
                evaluation_batch_size=1,
            )

    def test_evaluate_directional_deviation_raises_for_baseline_mode_without_thresholds(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        anomaly_data = {
            "x_test_anomaly": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([-1], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {"shift_mode": "subset", "num_channels": 2},
        }
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=torch.zeros(1, 2),
                score_thresholds=score_thresholds,
                baseline_thresholds=None,
                evaluation_batch_size=1,
            )

    def test_evaluate_directional_deviation_raises_for_unsupported_scoring_mode(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        anomaly_data = {
            "x_test_anomaly": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([-1], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {"shift_mode": "subset", "num_channels": 2},
        }
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(  # type: ignore[arg-type]
                model=model,
                model_name="baseline",
                scoring_mode="other_mode",
                anomaly_data=anomaly_data,
                x_test_normal=torch.zeros(1, 2),
                score_thresholds=score_thresholds,
                baseline_thresholds=BaselineEvidenceThresholds(
                    evidence_tau=torch.tensor([0.0, 0.0]),
                    latent_directions=torch.tensor([1, -1]),
                ),
                evaluation_batch_size=1,
            )

    def test_evaluate_directional_deviation_sets_fpr_to_none_when_no_normal_samples(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        anomaly_data = {
            "x_test_anomaly": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([-1], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {"shift_mode": "subset", "num_channels": 2},
        }

        # when
        evaluation = evaluate_directional_deviation(
            model=model,
            model_name="baseline",
            scoring_mode="baseline",
            anomaly_data=anomaly_data,
            x_test_normal=torch.empty((0, 2), dtype=torch.float32),
            score_thresholds=score_thresholds,
            baseline_thresholds=baseline_thresholds,
            evaluation_batch_size=1,
        )

        # then
        self.assertIsNone(evaluation["metrics"]["fpr_high"])
        self.assertIsNone(evaluation["metrics"]["fpr_low"])

    def test_evaluate_directional_deviation_raises_for_negative_labels_channel_in_single_channel_mode(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        anomaly_data = {
            "x_test_anomaly": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([-1], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {
                "shift_mode": "single_channel",
                "channels_to_evaluate": [0, 1],
                "num_channels": 2,
            },
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=torch.zeros(1, 2),
                score_thresholds=score_thresholds,
                baseline_thresholds=baseline_thresholds,
                evaluation_batch_size=1,
            )

    def test_evaluate_directional_deviation_raises_for_missing_channels_to_evaluate_in_single_channel_mode(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        anomaly_data = {
            "x_test_anomaly": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([0], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {"shift_mode": "single_channel", "num_channels": 2},
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=torch.zeros(1, 2),
                score_thresholds=score_thresholds,
                baseline_thresholds=baseline_thresholds,
                evaluation_batch_size=1,
            )

    def test_evaluate_directional_deviation_raises_for_channels_to_evaluate_not_list_in_single_channel_mode(
        self,
    ) -> None:
        # given
        class DummyBaselineModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyBaselineModel()
        baseline_thresholds = BaselineEvidenceThresholds(
            evidence_tau=torch.tensor([0.5, 0.5], dtype=torch.float32),
            latent_directions=torch.tensor([1, -1], dtype=torch.int64),
        )
        score_thresholds = ScoreThresholds(
            theta_high=0.1,
            theta_low=0.1,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        anomaly_data = {
            "x_test_anomaly": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "labels_type": torch.tensor([LABEL_TYPE_HIGH], dtype=torch.int64),
            "labels_channel": torch.tensor([0], dtype=torch.int64),
            "anomaly_stream": None,
            "anomaly_manifest": {
                "shift_mode": "single_channel",
                "channels_to_evaluate": "not_a_list",
                "num_channels": 2,
            },
        }

        # when & then
        with self.assertRaises(ValueError):
            _ = evaluate_directional_deviation(
                model=model,
                model_name="baseline",
                scoring_mode="baseline",
                anomaly_data=anomaly_data,
                x_test_normal=torch.zeros(1, 2),
                score_thresholds=score_thresholds,
                baseline_thresholds=baseline_thresholds,
                evaluation_batch_size=1,
            )

    def test_evaluate_and_persist_writes_evaluation_even_on_error(self) -> None:
        # given
        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_instances = 1
                self.sign_aware = False

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"latents": x.unsqueeze(0)}

        model = DummyModel()
        x_test_normal = torch.zeros(1, 2, dtype=torch.float32)
        score_thresholds = ScoreThresholds(
            theta_high=1.0,
            theta_low=1.0,
            score_fpr=0.01,
            score_quantile=0.99,
        )
        anomaly_data = {
            "anomaly_stream": [],
            "anomaly_manifest": {"shift_mode": "subset"},
        }
        inputs = DirectionalDeviationEvaluationInputs(
            model=model,
            model_name="broken",
            scoring_mode="sign_aware",
            anomaly_data=anomaly_data,
            x_test_normal=x_test_normal,
            score_thresholds=score_thresholds,
            evaluation_batch_size=1,
            coverage_recall_target=0.8,
            latent_thresholds=None,
            baseline_thresholds=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "models", "broken")

            # when
            evaluation = evaluate_and_persist_directional_deviation(
                model_dir=model_dir,
                inputs=inputs,
            )

            # then
            evaluation_path = os.path.join(model_dir, "evaluation.json")
            self.assertTrue(os.path.exists(evaluation_path))
            with open(evaluation_path, "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, evaluation)
            self.assertIsNotNone(evaluation["error"])
            self.assertIsNone(evaluation["metrics"]["recall_high"])

    def test_persist_comparison_summary_writes_expected_file(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            evaluations = {
                "model_a": {"metrics": {"recall_high": 1.0, "recall_low": 0.5}},
                "model_b": {"metrics": {"recall_high": 0.0, "recall_low": 1.0}},
            }

            # when
            persist_comparison_summary(run_dir, evaluations)

            # then
            summary_path = os.path.join(run_dir, "comparison_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r") as f:
                summary = json.load(f)
            self.assertIn("models", summary)
            self.assertIn("model_a", summary["models"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
