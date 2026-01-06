import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

import torch

from sgnsae.experiments.directional_deviation.data import (
    LABEL_TYPE_HIGH,
    LABEL_TYPE_LOW,
)
from sgnsae.experiments.directional_deviation.plotting import (
    _compute_directional_scores,
    _extract_metric_value,
    _extract_models_dict_from_summary,
    _load_pickled_model,
    _load_torch_if_present,
    _load_torch_tensor_if_present,
    _plot_decoder_purity_histograms,
    _plot_leaf_run_summary,
    _plot_score_histograms_grid_for_models,
    _plot_score_distributions_for_model,
    _plot_score_distributions_for_models,
    _plot_seed_sweep_summary,
    _plot_width_sweep_summary,
    generate_plots_for_run,
)
from sgnsae.models import GatedSAE, SAE


class TestDirectionalDeviationPlotting(unittest.TestCase):
    def test_generate_plots_for_run_returns_when_summary_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            generate_plots_for_run(tmpdir)

            # then
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "plots")))

    def test_plot_width_sweep_summary_returns_when_per_width_not_dict(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            summary: Dict[str, Any] = {
                "sweep": {"type": "width", "widths": [2]},
                "per_width": None,
            }

            # when
            _plot_width_sweep_summary(plots_dir=plots_dir, summary=summary)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_width_sweep_summary_returns_when_widths_missing_or_empty(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            summary: Dict[str, Any] = {
                "sweep": {"type": "width", "widths": []},
                "per_width": {},
            }

            # when
            _plot_width_sweep_summary(plots_dir=plots_dir, summary=summary)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_width_sweep_summary_returns_when_series_empty(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            summary: Dict[str, Any] = {
                "sweep": {"type": "width", "widths": [2]},
                "per_width": {"2": {"models": None}},
            }

            # when
            _plot_width_sweep_summary(plots_dir=plots_dir, summary=summary)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_width_sweep_summary_skips_non_dict_metrics_entries(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            summary: Dict[str, Any] = {
                "sweep": {"type": "width", "widths": [2]},
                "per_width": {"2": {"models": {"relu_sae": "not_a_dict"}}},
            }

            # when
            _plot_width_sweep_summary(plots_dir=plots_dir, summary=summary)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_width_sweep_summary_writes_recall_vs_width_plot(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            summary: Dict[str, Any] = {
                "sweep": {"type": "width", "widths": [2, 4]},
                "per_width": {
                    "2": {
                        "models": {
                            "relu_sae": {
                                "recall_high": 0.1,
                                "recall_low": 0.2,
                                "covered_channels_fraction": 0.0,
                                "fpr_high": 0.01,
                                "fpr_low": 0.01,
                            }
                        }
                    },
                    "4": {
                        "models": {
                            "relu_sae": {
                                "recall_high": 0.2,
                                "recall_low": 0.3,
                                "covered_channels_fraction": 0.0,
                                "fpr_high": 0.01,
                                "fpr_low": 0.01,
                            }
                        }
                    },
                },
            }

            # when
            _plot_width_sweep_summary(plots_dir=plots_dir, summary=summary)

            # then
            self.assertTrue(
                os.path.isfile(os.path.join(plots_dir, "recall_vs_width.png"))
            )

    def test_plot_seed_sweep_summary_returns_when_aggregate_models_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            summary: Dict[str, Any] = {
                "sweep": {"type": "seed", "seeds": [0, 1]},
                "aggregate_over_seeds": {"models": None},
            }

            # when
            _plot_seed_sweep_summary(plots_dir=plots_dir, summary=summary)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_seed_sweep_summary_returns_when_no_models(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            summary: Dict[str, Any] = {
                "sweep": {"type": "seed", "seeds": [0, 1]},
                "aggregate_over_seeds": {"models": {}},
            }

            # when
            _plot_seed_sweep_summary(plots_dir=plots_dir, summary=summary)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_leaf_run_summary_returns_when_models_empty(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # when
            _plot_leaf_run_summary(plots_dir=plots_dir, models={})

            # then
            self.assertFalse(
                os.path.isfile(os.path.join(plots_dir, "leaf_summary.png"))
            )

    def test_plot_decoder_purity_histograms_returns_when_models_dir_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # when
            _plot_decoder_purity_histograms(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertFalse(
                os.path.isfile(os.path.join(plots_dir, "decoder_purity_hist.png"))
            )

    def test_plot_decoder_purity_histograms_returns_when_no_models_present(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "models"), exist_ok=True)

            # when
            _plot_decoder_purity_histograms(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertFalse(
                os.path.isfile(os.path.join(plots_dir, "decoder_purity_hist.png"))
            )

    def test_plot_decoder_purity_histograms_skips_invalid_alignment_payloads(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            models_dir = os.path.join(tmpdir, "models")
            os.makedirs(models_dir, exist_ok=True)

            missing_alignment_dir = os.path.join(models_dir, "missing_alignment")
            os.makedirs(missing_alignment_dir, exist_ok=True)

            error_alignment_dir = os.path.join(models_dir, "error_alignment")
            os.makedirs(error_alignment_dir, exist_ok=True)
            with open(
                os.path.join(error_alignment_dir, "decoder_alignment.json"), "w"
            ) as f:
                json.dump({"error": "boom"}, f)

            empty_purity_dir = os.path.join(models_dir, "empty_purity")
            os.makedirs(empty_purity_dir, exist_ok=True)
            with open(
                os.path.join(empty_purity_dir, "decoder_alignment.json"), "w"
            ) as f:
                json.dump({"latent_purity": []}, f)

            nan_purity_dir = os.path.join(models_dir, "nan_purity")
            os.makedirs(nan_purity_dir, exist_ok=True)
            with open(os.path.join(nan_purity_dir, "decoder_alignment.json"), "w") as f:
                json.dump({"latent_purity": [float("nan")]}, f)

            # when
            _plot_decoder_purity_histograms(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertFalse(
                os.path.isfile(os.path.join(plots_dir, "decoder_purity_hist.png"))
            )

    def test_plot_decoder_purity_histograms_writes_plot_when_valid_alignment_present(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            models_dir = os.path.join(tmpdir, "models")
            os.makedirs(models_dir, exist_ok=True)

            model_dir = os.path.join(models_dir, "relu_sae")
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "decoder_alignment.json"), "w") as f:
                json.dump({"latent_purity": [0.1, 0.25, 0.9]}, f)

            # when
            _plot_decoder_purity_histograms(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertTrue(
                os.path.isfile(os.path.join(plots_dir, "decoder_purity_hist.png"))
            )

    def test_plot_score_distributions_for_models_returns_when_preview_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)
            with open(os.path.join(tmpdir, "data", "anomaly_manifest.json"), "w") as f:
                json.dump({"shift_mode": "single_channel", "num_channels": 1}, f)

            # when
            _plot_score_distributions_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_score_distributions_for_models_returns_when_manifest_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)
            torch.save(
                {
                    "x_test_normal": torch.zeros((2, 1), dtype=torch.float32),
                    "x_test_anomaly": torch.zeros((2, 1), dtype=torch.float32),
                    "labels_type": torch.zeros((2,), dtype=torch.int64),
                },
                os.path.join(tmpdir, "data", "sampled_data.pt"),
            )

            # when
            _plot_score_distributions_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_score_distributions_for_models_returns_when_models_dir_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)
            torch.save(
                {
                    "x_test_normal": torch.zeros((2, 1), dtype=torch.float32),
                    "x_test_anomaly": torch.zeros((2, 1), dtype=torch.float32),
                    "labels_type": torch.zeros((2,), dtype=torch.int64),
                },
                os.path.join(tmpdir, "data", "sampled_data.pt"),
            )
            with open(os.path.join(tmpdir, "data", "anomaly_manifest.json"), "w") as f:
                json.dump({"shift_mode": "single_channel", "num_channels": 1}, f)

            # when
            _plot_score_distributions_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_score_histograms_grid_returns_when_preview_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # when
            _plot_score_histograms_grid_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertFalse(
                os.path.exists(os.path.join(plots_dir, "score_histograms.png"))
            )

    def test_plot_score_histograms_grid_returns_when_manifest_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            torch.save(
                {
                    "x_test_normal": torch.zeros((2, 1), dtype=torch.float32),
                    "x_test_anomaly": torch.zeros((2, 1), dtype=torch.float32),
                    "labels_type": torch.zeros((2,), dtype=torch.int64),
                },
                os.path.join(data_dir, "sampled_data.pt"),
            )

            # when
            _plot_score_histograms_grid_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertFalse(
                os.path.exists(os.path.join(plots_dir, "score_histograms.png"))
            )

    def test_plot_score_histograms_grid_returns_when_models_dir_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            torch.save(
                {
                    "x_test_normal": torch.zeros((2, 1), dtype=torch.float32),
                    "x_test_anomaly": torch.zeros((2, 1), dtype=torch.float32),
                    "labels_type": torch.zeros((2,), dtype=torch.int64),
                },
                os.path.join(data_dir, "sampled_data.pt"),
            )
            with open(os.path.join(data_dir, "anomaly_manifest.json"), "w") as f:
                json.dump({"shift_mode": "single_channel", "num_channels": 1}, f)

            # when
            _plot_score_histograms_grid_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertFalse(
                os.path.exists(os.path.join(plots_dir, "score_histograms.png"))
            )

    def test_plot_score_histograms_grid_returns_when_required_tensors_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            torch.save(
                {
                    "x_test_normal": torch.zeros((2, 1), dtype=torch.float32),
                    "x_test_anomaly": torch.zeros((2, 1), dtype=torch.float32),
                },
                os.path.join(data_dir, "sampled_data.pt"),
            )
            with open(os.path.join(data_dir, "anomaly_manifest.json"), "w") as f:
                json.dump({"shift_mode": "single_channel", "num_channels": 1}, f)
            os.makedirs(os.path.join(tmpdir, "models"), exist_ok=True)

            # when
            _plot_score_histograms_grid_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertFalse(
                os.path.exists(os.path.join(plots_dir, "score_histograms.png"))
            )

    def test_plot_score_histograms_grid_skips_models_with_missing_or_invalid_artifacts(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            torch.save(
                {
                    "x_test_normal": torch.zeros((2, 1), dtype=torch.float32),
                    "x_test_anomaly": torch.zeros((2, 1), dtype=torch.float32),
                    "labels_type": torch.tensor(
                        [LABEL_TYPE_HIGH, LABEL_TYPE_LOW], dtype=torch.int64
                    ),
                },
                os.path.join(data_dir, "sampled_data.pt"),
            )
            with open(os.path.join(data_dir, "anomaly_manifest.json"), "w") as f:
                json.dump({"shift_mode": "single_channel", "num_channels": 1}, f)

            models_dir = os.path.join(tmpdir, "models")
            os.makedirs(models_dir, exist_ok=True)

            os.makedirs(os.path.join(models_dir, "no_eval"), exist_ok=True)

            bad_scoring_dir = os.path.join(models_dir, "bad_scoring")
            os.makedirs(bad_scoring_dir, exist_ok=True)
            with open(os.path.join(bad_scoring_dir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "unsupported"}, f)

            missing_model_dir = os.path.join(models_dir, "missing_model")
            os.makedirs(missing_model_dir, exist_ok=True)
            with open(os.path.join(missing_model_dir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)

            missing_thresholds_dir = os.path.join(models_dir, "missing_thresholds")
            os.makedirs(missing_thresholds_dir, exist_ok=True)
            with open(
                os.path.join(missing_thresholds_dir, "evaluation.json"), "w"
            ) as f:
                json.dump({"scoring_mode": "baseline"}, f)
            torch.save(
                torch.nn.Identity(),
                os.path.join(missing_thresholds_dir, "model.pt"),
            )

            invalid_theta_dir = os.path.join(models_dir, "invalid_theta")
            os.makedirs(invalid_theta_dir, exist_ok=True)
            with open(os.path.join(invalid_theta_dir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)
            torch.save(torch.nn.Identity(), os.path.join(invalid_theta_dir, "model.pt"))
            thresholds_dir = os.path.join(invalid_theta_dir, "thresholds")
            os.makedirs(thresholds_dir, exist_ok=True)
            with open(os.path.join(thresholds_dir, "score_thresholds.json"), "w") as f:
                json.dump({"theta_high": "bad", "theta_low": 0.1}, f)

            non_finite_theta_dir = os.path.join(models_dir, "non_finite_theta")
            os.makedirs(non_finite_theta_dir, exist_ok=True)
            with open(os.path.join(non_finite_theta_dir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)
            torch.save(
                torch.nn.Identity(),
                os.path.join(non_finite_theta_dir, "model.pt"),
            )
            thresholds_dir = os.path.join(non_finite_theta_dir, "thresholds")
            os.makedirs(thresholds_dir, exist_ok=True)
            with open(os.path.join(thresholds_dir, "score_thresholds.json"), "w") as f:
                json.dump({"theta_high": float("inf"), "theta_low": 0.1}, f)

            compute_error_dir = os.path.join(models_dir, "compute_error")
            os.makedirs(compute_error_dir, exist_ok=True)
            with open(os.path.join(compute_error_dir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "sign_aware"}, f)
            torch.save(torch.nn.Identity(), os.path.join(compute_error_dir, "model.pt"))
            thresholds_dir = os.path.join(compute_error_dir, "thresholds")
            os.makedirs(thresholds_dir, exist_ok=True)
            with open(os.path.join(thresholds_dir, "score_thresholds.json"), "w") as f:
                json.dump({"theta_high": 0.1, "theta_low": 0.1}, f)

            # when
            _plot_score_histograms_grid_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertFalse(
                os.path.exists(os.path.join(plots_dir, "score_histograms.png"))
            )

    def test_plot_score_distributions_for_models_catches_model_exceptions(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            torch.save(
                {
                    "x_test_normal": torch.zeros((2, 1), dtype=torch.float32),
                    "x_test_anomaly": torch.zeros((2, 1), dtype=torch.float32),
                    "labels_type": torch.tensor(
                        [LABEL_TYPE_HIGH, LABEL_TYPE_LOW], dtype=torch.int64
                    ),
                },
                os.path.join(data_dir, "sampled_data.pt"),
            )
            with open(os.path.join(data_dir, "anomaly_manifest.json"), "w") as f:
                json.dump({"shift_mode": "single_channel", "num_channels": 1}, f)

            model_dir = os.path.join(tmpdir, "models", "broken_model")
            thresholds_dir = os.path.join(model_dir, "thresholds")
            os.makedirs(thresholds_dir, exist_ok=True)
            with open(os.path.join(model_dir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)
            torch.save({"not": "a module"}, os.path.join(model_dir, "model.pt"))
            with open(os.path.join(thresholds_dir, "score_thresholds.json"), "w") as f:
                json.dump({"theta_high": 0.1, "theta_low": 0.1}, f)

            # when
            _plot_score_distributions_for_models(run_dir=tmpdir, plots_dir=plots_dir)

            # then
            self.assertEqual(os.listdir(plots_dir), [])

    def test_plot_score_distributions_for_model_returns_when_evaluation_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            # when
            _plot_score_distributions_for_model(
                model_dir=tmpdir,
                model_name="model",
                anomaly_manifest={},
                preview={},
                plots_root=tmpdir,
            )

            # then
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "model")))

    def test_plot_score_distributions_for_model_returns_when_scoring_mode_invalid(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "unknown"}, f)

            # when
            _plot_score_distributions_for_model(
                model_dir=tmpdir,
                model_name="model",
                anomaly_manifest={},
                preview={},
                plots_root=tmpdir,
            )

            # then
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "model")))

    def test_plot_score_distributions_for_model_returns_when_model_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)

            # when
            _plot_score_distributions_for_model(
                model_dir=tmpdir,
                model_name="model",
                anomaly_manifest={},
                preview={},
                plots_root=tmpdir,
            )

            # then
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "model")))

    def test_plot_score_distributions_for_model_returns_when_thresholds_missing(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)
            torch.save(torch.nn.Identity(), os.path.join(tmpdir, "model.pt"))

            # when
            _plot_score_distributions_for_model(
                model_dir=tmpdir,
                model_name="model",
                anomaly_manifest={},
                preview={},
                plots_root=tmpdir,
            )

            # then
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "model")))

    def test_plot_score_distributions_for_model_returns_when_thresholds_non_numeric(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            thresholds_dir = os.path.join(tmpdir, "thresholds")
            os.makedirs(thresholds_dir, exist_ok=True)
            with open(os.path.join(tmpdir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)
            torch.save(torch.nn.Identity(), os.path.join(tmpdir, "model.pt"))
            with open(os.path.join(thresholds_dir, "score_thresholds.json"), "w") as f:
                json.dump({"theta_high": "invalid", "theta_low": 0.1}, f)

            # when
            _plot_score_distributions_for_model(
                model_dir=tmpdir,
                model_name="model",
                anomaly_manifest={},
                preview={},
                plots_root=tmpdir,
            )

            # then
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "model")))

    def test_plot_score_distributions_for_model_returns_when_thresholds_non_finite(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            thresholds_dir = os.path.join(tmpdir, "thresholds")
            os.makedirs(thresholds_dir, exist_ok=True)
            with open(os.path.join(tmpdir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)
            torch.save(torch.nn.Identity(), os.path.join(tmpdir, "model.pt"))
            with open(os.path.join(thresholds_dir, "score_thresholds.json"), "w") as f:
                json.dump(
                    {"theta_high": float("nan"), "theta_low": float("inf")},
                    f,
                )

            # when
            _plot_score_distributions_for_model(
                model_dir=tmpdir,
                model_name="model",
                anomaly_manifest={},
                preview={},
                plots_root=tmpdir,
            )

            # then
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "model")))

    def test_plot_score_distributions_for_model_returns_when_preview_missing_keys(
        self,
    ) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            thresholds_dir = os.path.join(tmpdir, "thresholds")
            os.makedirs(thresholds_dir, exist_ok=True)
            with open(os.path.join(tmpdir, "evaluation.json"), "w") as f:
                json.dump({"scoring_mode": "baseline"}, f)
            torch.save(torch.nn.Identity(), os.path.join(tmpdir, "model.pt"))
            with open(os.path.join(thresholds_dir, "score_thresholds.json"), "w") as f:
                json.dump({"theta_high": 0.1, "theta_low": 0.1}, f)

            preview = {"x_test_normal": torch.zeros((1, 1), dtype=torch.float32)}

            # when
            _plot_score_distributions_for_model(
                model_dir=tmpdir,
                model_name="model",
                anomaly_manifest={},
                preview=preview,
                plots_root=tmpdir,
            )

            # then
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "model")))

    def test_compute_directional_scores_sign_aware_matches_expected_shapes(
        self,
    ) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=1,
            d_hidden=1,
            resample_latents_threshold=1,
            sign_aware=True,
            symmetric_magnitude=True,
            reset_activation_offsets_on_resample=False,
            single_activation_offset=True,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(1, dtype=torch.float32), os.path.join(tmpdir, "tau_plus.pt")
            )
            torch.save(
                torch.zeros(1, dtype=torch.float32),
                os.path.join(tmpdir, "tau_minus.pt"),
            )
            x = torch.randn((4, 1), dtype=torch.float32)

            # when
            score_high, score_low = _compute_directional_scores(
                model=model,
                scoring_mode="sign_aware",
                anomaly_manifest={},
                thresholds_dir=tmpdir,
                x=x,
                targets=None,
            )

            # then
            self.assertEqual(tuple(score_high.shape), (4,))
            self.assertEqual(tuple(score_low.shape), (4,))

    def test_compute_directional_scores_sign_aware_reads_pooling_config_from_metadata_and_ignores_targets_when_not_restricted(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            model.b_dec[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(2, dtype=torch.float32), os.path.join(tmpdir, "tau_plus.pt")
            )
            torch.save(
                torch.zeros(2, dtype=torch.float32),
                os.path.join(tmpdir, "tau_minus.pt"),
            )
            with open(os.path.join(tmpdir, "thresholds_metadata.json"), "w") as f:
                json.dump({"scoring_config": {"pooling": "topk", "topk_k": 1}}, f)

            x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

            # when
            score_high, score_low = _compute_directional_scores(
                model=model,
                scoring_mode="sign_aware",
                anomaly_manifest={},
                thresholds_dir=tmpdir,
                x=x,
                targets=torch.tensor([0], dtype=torch.int64),
            )

            # then
            self.assertTrue(torch.allclose(score_high, torch.tensor([2.0])))
            self.assertTrue(torch.allclose(score_low, torch.tensor([0.0])))

    def test_compute_directional_scores_sign_aware_applies_zscore_normalization_from_metadata(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            model.b_dec[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(2, dtype=torch.float32), os.path.join(tmpdir, "tau_plus.pt")
            )
            torch.save(
                torch.zeros(2, dtype=torch.float32),
                os.path.join(tmpdir, "tau_minus.pt"),
            )
            with open(os.path.join(tmpdir, "thresholds_metadata.json"), "w") as handle:
                json.dump(
                    {
                        "signal_normalization": {
                            "mode": "zscore",
                            "normalization_eps": 1e-6,
                            "mean": [1.0, 1.0],
                            "std": [1.0, 1.0],
                        }
                    },
                    handle,
                )

            x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

            # when
            score_high, score_low = _compute_directional_scores(
                model=model,
                scoring_mode="sign_aware",
                anomaly_manifest={},
                thresholds_dir=tmpdir,
                x=x,
                targets=None,
            )

            # then
            self.assertTrue(torch.allclose(score_high, torch.tensor([1.0])))
            self.assertTrue(torch.allclose(score_low, torch.tensor([0.0])))

    def test_compute_directional_scores_sign_aware_applies_zscore_normalization_from_file(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            model.b_dec[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(2, dtype=torch.float32), os.path.join(tmpdir, "tau_plus.pt")
            )
            torch.save(
                torch.zeros(2, dtype=torch.float32),
                os.path.join(tmpdir, "tau_minus.pt"),
            )
            with open(os.path.join(tmpdir, "thresholds_metadata.json"), "w") as handle:
                json.dump({}, handle)
            torch.save(
                {
                    "mode": "zscore",
                    "mean": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    "std": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    "normalization_eps": 1e-6,
                    "clamped_mask": torch.tensor([False, False]),
                },
                os.path.join(tmpdir, "signal_normalization.pt"),
            )

            x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

            # when
            score_high, score_low = _compute_directional_scores(
                model=model,
                scoring_mode="sign_aware",
                anomaly_manifest={},
                thresholds_dir=tmpdir,
                x=x,
                targets=None,
            )

            # then
            self.assertTrue(torch.allclose(score_high, torch.tensor([1.0])))
            self.assertTrue(torch.allclose(score_low, torch.tensor([0.0])))

    def test_compute_directional_scores_sign_aware_target_restricted_scoring_generates_targets_when_missing(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            model.b_dec[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(2, dtype=torch.float32), os.path.join(tmpdir, "tau_plus.pt")
            )
            torch.save(
                torch.zeros(2, dtype=torch.float32),
                os.path.join(tmpdir, "tau_minus.pt"),
            )
            with open(os.path.join(tmpdir, "thresholds_metadata.json"), "w") as f:
                json.dump({"scoring_config": {"target_restricted_scoring": True}}, f)

            x = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)

            # when
            score_high, _ = _compute_directional_scores(
                model=model,
                scoring_mode="sign_aware",
                anomaly_manifest={"channels_to_evaluate": [0, 1]},
                thresholds_dir=tmpdir,
                x=x,
                targets=None,
            )

            # then
            self.assertTrue(torch.allclose(score_high, torch.tensor([1.0, 2.0])))

    def test_compute_directional_scores_sign_aware_target_restricted_scoring_uses_provided_targets(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            model.b_dec[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(2, dtype=torch.float32), os.path.join(tmpdir, "tau_plus.pt")
            )
            torch.save(
                torch.zeros(2, dtype=torch.float32),
                os.path.join(tmpdir, "tau_minus.pt"),
            )
            with open(os.path.join(tmpdir, "thresholds_metadata.json"), "w") as f:
                json.dump({"scoring_config": {"target_restricted_scoring": True}}, f)

            x = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)

            # when
            score_high, _ = _compute_directional_scores(
                model=model,
                scoring_mode="sign_aware",
                anomaly_manifest={"channels_to_evaluate": [0, 1]},
                thresholds_dir=tmpdir,
                x=x,
                targets=torch.tensor([1, 0], dtype=torch.int64),
            )

            # then
            self.assertTrue(torch.allclose(score_high, torch.tensor([2.0, 1.0])))

    def test_compute_directional_scores_sign_aware_target_restricted_scoring_raises_when_channels_missing(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "thresholds_metadata.json"), "w") as f:
                json.dump({"scoring_config": {"target_restricted_scoring": True}}, f)

            # when & then
            with self.assertRaises(ValueError):
                _ = _compute_directional_scores(
                    model=model,
                    scoring_mode="sign_aware",
                    anomaly_manifest={},
                    thresholds_dir=tmpdir,
                    x=torch.zeros((1, 2), dtype=torch.float32),
                    targets=None,
                )

    def test_compute_directional_scores_baseline_assigns_latent_directions_when_missing(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor(
                [[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32
            )
            model.b_dec[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(2, dtype=torch.float32),
                os.path.join(tmpdir, "evidence_tau.pt"),
            )
            x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

            # when
            score_high, score_low = _compute_directional_scores(
                model=model,
                scoring_mode="baseline",
                anomaly_manifest={"shift_mode": "single_channel"},
                thresholds_dir=tmpdir,
                x=x,
                targets=None,
            )

            # then
            self.assertEqual(tuple(score_high.shape), (1,))
            self.assertEqual(tuple(score_low.shape), (1,))

    def test_compute_directional_scores_baseline_applies_zscore_normalization_from_metadata(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            model.b_dec[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            model.b_enc[0] = torch.tensor([0.0, 0.0], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(2, dtype=torch.float32),
                os.path.join(tmpdir, "evidence_tau.pt"),
            )
            torch.save(
                torch.tensor([1, -1], dtype=torch.int64),
                os.path.join(tmpdir, "latent_directions.pt"),
            )
            with open(os.path.join(tmpdir, "thresholds_metadata.json"), "w") as handle:
                json.dump(
                    {
                        "signal_normalization": {
                            "mode": "zscore",
                            "normalization_eps": 1e-6,
                            "mean": [1.0, 1.0],
                            "std": [1.0, 1.0],
                        }
                    },
                    handle,
                )
            x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

            # when
            score_high, score_low = _compute_directional_scores(
                model=model,
                scoring_mode="baseline",
                anomaly_manifest={"shift_mode": "single_channel"},
                thresholds_dir=tmpdir,
                x=x,
                targets=None,
            )

            # then
            self.assertTrue(torch.allclose(score_high, torch.tensor([0.0])))
            self.assertTrue(torch.allclose(score_low, torch.tensor([1.0])))

    def test_compute_directional_scores_baseline_raises_when_latent_directions_wrong_dim(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(2, dtype=torch.float32),
                os.path.join(tmpdir, "evidence_tau.pt"),
            )
            torch.save(
                torch.zeros((1, 2), dtype=torch.int64),
                os.path.join(tmpdir, "latent_directions.pt"),
            )
            x = torch.zeros((1, 2), dtype=torch.float32)

            # when & then
            with self.assertRaises(ValueError):
                _ = _compute_directional_scores(
                    model=model,
                    scoring_mode="baseline",
                    anomaly_manifest={"shift_mode": "single_channel"},
                    thresholds_dir=tmpdir,
                    x=x,
                    targets=None,
                )

    def test_compute_directional_scores_sign_aware_raises_when_tau_missing(
        self,
    ) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=1,
            d_hidden=1,
            resample_latents_threshold=1,
            sign_aware=False,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            x = torch.randn((2, 1), dtype=torch.float32)

            # when & then
            with self.assertRaises(ValueError):
                _ = _compute_directional_scores(
                    model=model,
                    scoring_mode="sign_aware",
                    anomaly_manifest={},
                    thresholds_dir=tmpdir,
                    x=x,
                    targets=None,
                )

    def test_compute_directional_scores_baseline_raises_when_evidence_tau_missing(
        self,
    ) -> None:
        # given
        model = torch.nn.Identity()
        with tempfile.TemporaryDirectory() as tmpdir:
            x = torch.zeros((1, 1), dtype=torch.float32)

            # when & then
            with self.assertRaises(ValueError):
                _ = _compute_directional_scores(
                    model=model,
                    scoring_mode="baseline",
                    anomaly_manifest={"shift_mode": "single_channel"},
                    thresholds_dir=tmpdir,
                    x=x,
                    targets=None,
                )

    def test_compute_directional_scores_baseline_raises_when_shift_mode_invalid(
        self,
    ) -> None:
        # given
        model = torch.nn.Identity()
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                torch.zeros(1, dtype=torch.float32),
                os.path.join(tmpdir, "evidence_tau.pt"),
            )
            x = torch.zeros((1, 1), dtype=torch.float32)

            # when & then
            with self.assertRaises(ValueError):
                _ = _compute_directional_scores(
                    model=model,
                    scoring_mode="baseline",
                    anomaly_manifest={"shift_mode": "bogus"},
                    thresholds_dir=tmpdir,
                    x=x,
                    targets=None,
                )

    def test_extract_models_dict_from_summary_supports_aggregate_over_seeds(
        self,
    ) -> None:
        # given
        summary = {
            "aggregate_over_seeds": {"models": {"relu_sae": {"recall_high_mean": 0.1}}}
        }

        # when
        models = _extract_models_dict_from_summary(summary)

        # then
        self.assertIn("relu_sae", models)

    def test_extract_models_dict_from_summary_returns_empty_for_non_dict(self) -> None:
        # given
        # when
        models = _extract_models_dict_from_summary("invalid")

        # then
        self.assertEqual(models, {})

    def test_extract_metric_value_returns_none_when_metrics_not_dict(self) -> None:
        # given
        # when
        value = _extract_metric_value(metrics="invalid", key="recall_high")

        # then
        self.assertIsNone(value)

    def test_extract_metric_value_falls_back_to_mean_suffix(self) -> None:
        # given
        metrics = {"recall_high_mean": 0.25}

        # when
        value = _extract_metric_value(metrics=metrics, key="recall_high")

        # then
        self.assertIsNotNone(value)
        self.assertAlmostEqual(value or 0.0, 0.25, places=7)

    def test_extract_metric_value_returns_none_when_key_has_suffix(self) -> None:
        # given
        metrics = {"recall_high_mean": "not_numeric"}

        # when
        value = _extract_metric_value(metrics=metrics, key="recall_high_mean")

        # then
        self.assertIsNone(value)

    def test_load_torch_helpers_return_none_when_file_missing(self) -> None:
        # given
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing.pt")

            # when
            data = _load_torch_if_present(missing_path)
            tensor = _load_torch_tensor_if_present(missing_path)

            # then
            self.assertIsNone(data)
            self.assertIsNone(tensor)

    def test_load_pickled_model_falls_back_when_weights_only_not_supported(
        self,
    ) -> None:
        # given
        def load_side_effect(*args: Any, **kwargs: Any) -> Any:
            if "weights_only" in kwargs:
                raise TypeError("weights_only not supported")
            return torch.nn.Identity()

        # when
        with patch(
            "sgnsae.experiments.directional_deviation.plotting.torch.load",
            side_effect=load_side_effect,
        ):
            model = _load_pickled_model("unused.pt")

        # then
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
