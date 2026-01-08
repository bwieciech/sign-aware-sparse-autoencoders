import builtins
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

import torch

from sgnsae.experiments.toy_model_geometry_validation import plotting as plotting_module


class _DecoderOnlyModel(torch.nn.Module):
    def __init__(self, decoder_directions: torch.Tensor):
        super().__init__()
        self.W_dec = torch.nn.Parameter(decoder_directions.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise RuntimeError("Not used")


class _LatentsModel(torch.nn.Module):
    def __init__(
        self,
        decoder_directions: torch.Tensor,
        latents: torch.Tensor | None = None,
        missing_latents: bool = False,
        bad_latents_shape: bool = False,
    ):
        super().__init__()
        self.W_dec = torch.nn.Parameter(decoder_directions.unsqueeze(0))
        self._latents = latents
        self._missing_latents = missing_latents
        self._bad_latents_shape = bad_latents_shape

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._missing_latents:
            return {}
        if self._latents is None:
            latents = torch.zeros((int(x.shape[0]), 1), dtype=torch.float32)
        else:
            latents = self._latents.to(dtype=torch.float32)
        if self._bad_latents_shape:
            return {"latents": latents}
        return {"latents": latents.unsqueeze(0)}


class TestToyModelGeometryValidationPlotting(unittest.TestCase):
    def test_plot_unit_circle_handles_backend_selection_errors(self) -> None:
        # given
        model = _DecoderOnlyModel(decoder_directions=torch.tensor([[1.0], [0.0]]))
        feature_directions = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32
        )
        import matplotlib

        # when
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(matplotlib, "use", side_effect=RuntimeError("boom")):
                output_path = plotting_module.plot_unit_circle(
                    model=model,
                    model_name="relu_sae",
                    feature_directions=feature_directions,
                    model_dir=tmpdir,
                )

            # then
            self.assertTrue(os.path.isfile(output_path))

    def test_plot_unit_circle_writes_png(self) -> None:
        # given
        model = _DecoderOnlyModel(decoder_directions=torch.tensor([[1.0], [0.0]]))
        feature_directions = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32
        )

        # when
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plotting_module.plot_unit_circle(
                model=model,
                model_name="relu_sae",
                feature_directions=feature_directions,
                model_dir=tmpdir,
            )

            # then
            self.assertTrue(os.path.isfile(output_path))
            self.assertTrue(output_path.endswith("unit_circle_relu_sae.png"))

    def test_plot_bijump_histogram_writes_png(self) -> None:
        # given
        latents = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)
        model = _LatentsModel(
            decoder_directions=torch.tensor([[1.0], [0.0]]),
            latents=latents,
        )
        x_test = torch.zeros((3, 2), dtype=torch.float32)

        # when
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plotting_module.plot_bijump_histogram(
                model=model,
                model_name="sign_aware",
                x_test=x_test,
                latent_index=0,
                hist_num_bins=9,
                model_dir=tmpdir,
                device="cpu",
                axis_abs_cos=0.99,
            )

            # then
            self.assertTrue(os.path.isfile(output_path))
            self.assertTrue(output_path.endswith("bi_jump_hist_sign_aware_latent0.png"))

    def test_plot_bijump_histogram_handles_all_zero_latents(self) -> None:
        # given
        model = _LatentsModel(
            decoder_directions=torch.tensor([[1.0], [0.0]]),
            latents=torch.zeros((2, 1), dtype=torch.float32),
        )
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plotting_module.plot_bijump_histogram(
                model=model,
                model_name="zeros",
                x_test=x_test,
                latent_index=0,
                hist_num_bins=5,
                model_dir=tmpdir,
                device="cpu",
                axis_abs_cos=None,
            )

            # then
            self.assertTrue(os.path.isfile(output_path))

    def test_plot_bijump_histogram_raises_for_invalid_model_outputs(self) -> None:
        # given
        missing_latents_model = _LatentsModel(
            decoder_directions=torch.tensor([[1.0], [0.0]]),
            missing_latents=True,
        )
        bad_shape_model = _LatentsModel(
            decoder_directions=torch.tensor([[1.0], [0.0]]),
            bad_latents_shape=True,
        )
        x_test = torch.zeros((2, 2), dtype=torch.float32)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_bijump_histogram(
                    model=missing_latents_model,
                    model_name="missing",
                    x_test=x_test,
                    latent_index=0,
                    hist_num_bins=5,
                    model_dir=tmpdir,
                    device="cpu",
                )
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_bijump_histogram(
                    model=bad_shape_model,
                    model_name="bad",
                    x_test=x_test,
                    latent_index=0,
                    hist_num_bins=5,
                    model_dir=tmpdir,
                    device="cpu",
                )
            with self.assertRaises(ValueError):
                _ = plotting_module.plot_bijump_histogram(
                    model=_LatentsModel(
                        decoder_directions=torch.tensor([[1.0], [0.0]])
                    ),
                    model_name="range",
                    x_test=x_test,
                    latent_index=1,
                    hist_num_bins=5,
                    model_dir=tmpdir,
                    device="cpu",
                )

    def test_plotting_raises_when_matplotlib_import_fails(self) -> None:
        # given
        model = _DecoderOnlyModel(decoder_directions=torch.tensor([[1.0], [0.0]]))
        feature_directions = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32
        )
        real_import = builtins.__import__

        def fake_import(
            name: str,
            globals: Dict[str, Any] | None = None,
            locals: Dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("matplotlib unavailable")
            return real_import(name, globals, locals, fromlist, level)

        # when / then
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(builtins, "__import__", side_effect=fake_import):
                with self.assertRaises(RuntimeError):
                    _ = plotting_module.plot_unit_circle(
                        model=model,
                        model_name="relu_sae",
                        feature_directions=feature_directions,
                        model_dir=tmpdir,
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
