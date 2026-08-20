import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sgnsae.experiments.toy_model_geometry_validation import plotting as plotting_module


class _PlotModelMissingLatents(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.tensor([[[1.0], [0.0]]]))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}


class _PlotModelBadLatentsShape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.tensor([[[1.0], [0.0]]]))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"latents": torch.zeros((int(x.shape[0]), 1), dtype=torch.float32)}


class _HistogramModelMissingLatents(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}


class _HistogramModelBadLatentsShape(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"latents": torch.zeros((int(x.shape[0]), 1), dtype=torch.float32)}


class _HistogramModelZeroLatents(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"latents": torch.zeros((1, int(x.shape[0]), 1), dtype=torch.float32)}


class TestToyModelGeometryValidationPlotting(unittest.TestCase):
    def test_plot_unit_circle_raises_when_matplotlib_missing(self) -> None:
        # given
        model = _PlotModelMissingLatents()
        feature_directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        original_import = __import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "matplotlib":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("builtins.__import__", side_effect=guarded_import):
                with self.assertRaises(RuntimeError):
                    _ = plotting_module.plot_unit_circle(
                        model=model,
                        model_name="missing_matplotlib",
                        feature_directions=feature_directions,
                        model_dir=tmpdir,
                        x_test=x_test,
                    )

    def test_plot_unit_circle_ignores_matplotlib_use_failure(self) -> None:
        # given
        model = _PlotModelMissingLatents()
        feature_directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("matplotlib.use", side_effect=RuntimeError("boom")):
                with self.assertRaises(ValueError):
                    _ = plotting_module.plot_unit_circle(
                        model=model,
                        model_name="matplotlib_use_failure",
                        feature_directions=feature_directions,
                        model_dir=tmpdir,
                        x_test=x_test,
                    )

    def test_plot_unit_circle_raises_when_model_output_missing_latents(self) -> None:
        # given
        model = _PlotModelMissingLatents()
        feature_directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_unit_circle(
                    model=model,
                    model_name="missing_latents",
                    feature_directions=feature_directions,
                    model_dir=tmpdir,
                    x_test=x_test,
                )

    def test_plot_unit_circle_raises_when_model_output_latents_wrong_shape(self) -> None:
        # given
        model = _PlotModelBadLatentsShape()
        feature_directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_unit_circle(
                    model=model,
                    model_name="bad_latents_shape",
                    feature_directions=feature_directions,
                    model_dir=tmpdir,
                    x_test=x_test,
                )

    def test_plot_bijump_histogram_raises_when_model_output_missing_latents(self) -> None:
        # given
        model = _HistogramModelMissingLatents()
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_bijump_histogram(
                    model=model,
                    model_name="missing_latents",
                    x_test=x_test,
                    latent_index=0,
                    hist_num_bins=10,
                    model_dir=tmpdir,
                    device="cpu",
                )

    def test_plot_bijump_histogram_raises_when_latents_wrong_shape(self) -> None:
        # given
        model = _HistogramModelBadLatentsShape()
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_bijump_histogram(
                    model=model,
                    model_name="bad_shape",
                    x_test=x_test,
                    latent_index=0,
                    hist_num_bins=10,
                    model_dir=tmpdir,
                    device="cpu",
                )

    def test_plot_bijump_histogram_raises_when_latent_index_out_of_range(self) -> None:
        # given
        model = _HistogramModelZeroLatents()
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_bijump_histogram(
                    model=model,
                    model_name="out_of_range",
                    x_test=x_test,
                    latent_index=2,
                    hist_num_bins=10,
                    model_dir=tmpdir,
                    device="cpu",
                )

    def test_plot_bijump_histogram_accepts_zero_latents(self) -> None:
        # given
        model = _HistogramModelZeroLatents()
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plotting_module.plot_bijump_histogram(
                model=model,
                model_name="zero_latents",
                x_test=x_test,
                latent_index=0,
                hist_num_bins=10,
                model_dir=tmpdir,
                device="cpu",
            )

            # then
            self.assertTrue(output_path.endswith("bi_jump_hist_zero_latents_latent0.png"))

    def test_plot_bijump_histogram_logy_rejects_empty_values(self) -> None:
        # given
        values = np.array([], dtype=np.float64)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_bijump_histogram_logy(
                    values=values,
                    hist_num_bins=10,
                    title="empty",
                    output_path=os.path.join(tmpdir, "hist.png"),
                )

    def test_plot_bijump_histogram_logy_handles_all_zero_values(self) -> None:
        # given
        values = np.zeros((10,), dtype=np.float64)

        # when
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plotting_module.plot_bijump_histogram_logy(
                values=values,
                hist_num_bins=10,
                title="zeros",
                output_path=os.path.join(tmpdir, "hist.png"),
            )

            # then
            self.assertTrue(output_path.endswith("hist.png"))
            self.assertTrue(os.path.isfile(output_path))

    def test_plot_latent_vs_coeff_scatter_rejects_invalid_inputs(self) -> None:
        # given
        latent = np.zeros((2,), dtype=np.float64)
        signed_coeff = np.zeros((2,), dtype=np.float64)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_latent_vs_coeff_scatter(
                    latent=np.zeros((2,), dtype=np.float64),
                    signed_coeff=np.zeros((3,), dtype=np.float64),
                    title="bad",
                    output_path=os.path.join(tmpdir, "scatter.png"),
                )
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_latent_vs_coeff_scatter(
                    latent=np.zeros((2, 1), dtype=np.float64),
                    signed_coeff=np.zeros((2, 1), dtype=np.float64),
                    title="bad",
                    output_path=os.path.join(tmpdir, "scatter.png"),
                )
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_latent_vs_coeff_scatter(
                    latent=np.array([], dtype=np.float64),
                    signed_coeff=np.array([], dtype=np.float64),
                    title="bad",
                    output_path=os.path.join(tmpdir, "scatter.png"),
                )
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_latent_vs_coeff_scatter(
                    latent=latent,
                    signed_coeff=signed_coeff,
                    title="bad",
                    output_path=os.path.join(tmpdir, "scatter.png"),
                    max_points=0,
                )

    def test_plot_latent_vs_coeff_scatter_samples_when_too_many_points(self) -> None:
        # given
        latent = np.arange(5000, dtype=np.float64)
        signed_coeff = np.arange(5000, dtype=np.float64)

        # when
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plotting_module.plot_latent_vs_coeff_scatter(
                latent=latent,
                signed_coeff=signed_coeff,
                title="scatter",
                output_path=os.path.join(tmpdir, "scatter.png"),
                max_points=4000,
            )

            # then
            self.assertTrue(os.path.isfile(output_path))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
