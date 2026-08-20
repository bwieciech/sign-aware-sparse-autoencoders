import math
import unittest

import torch

from sgnsae.experiments.directional_deviation.decoder_alignment import (
    _calculate_purity_summary,
    build_decoder_alignment_diagnostics,
    calculate_latent_argmax_channel_and_purity,
)


class _DummyDecoderModel(torch.nn.Module):
    def __init__(self, decoder: torch.Tensor):
        super().__init__()
        if decoder.ndim != 2:
            raise ValueError("decoder must be 2D.")
        self.n_instances = 1
        self.W_dec = torch.nn.Parameter(decoder.unsqueeze(0))


class TestDirectionalDeviationDecoderAlignment(unittest.TestCase):
    def test_calculate_latent_argmax_channel_and_purity_one_hot_columns_returns_purity_one(
        self,
    ):
        # given
        decoder = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )
        model = _DummyDecoderModel(decoder)

        # when
        argmax_channel, purity = calculate_latent_argmax_channel_and_purity(model)

        # then
        torch.testing.assert_close(
            argmax_channel.to(dtype=torch.int64),
            torch.tensor([0, 1, 3], dtype=torch.int64),
        )
        torch.testing.assert_close(
            purity.to(dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
        )

    def test_build_decoder_alignment_diagnostics_uniform_column_matches_expected_purity(
        self,
    ):
        # given
        num_channels = 4
        decoder = torch.ones((num_channels, 1), dtype=torch.float32)
        model = _DummyDecoderModel(decoder)
        expected_purity = 1.0 / math.sqrt(num_channels)

        # when
        diagnostics, summary = build_decoder_alignment_diagnostics(
            model=model, num_bins=10
        )

        # then
        self.assertEqual(diagnostics["latent_argmax_channel"], [0])
        self.assertEqual(diagnostics["unique_channels_claimed"], 1)
        self.assertAlmostEqual(
            diagnostics["latent_purity"][0], expected_purity, places=6
        )
        self.assertAlmostEqual(summary["purity_mean"], expected_purity, places=6)
        self.assertAlmostEqual(summary["purity_p50"], expected_purity, places=6)

    def test_build_decoder_alignment_diagnostics_raises_when_num_bins_not_positive(
        self,
    ) -> None:
        # given
        decoder = torch.eye(2, dtype=torch.float32)
        model = _DummyDecoderModel(decoder)

        # when / then
        with self.assertRaises(ValueError):
            _ = build_decoder_alignment_diagnostics(model=model, num_bins=0)

    def test_calculate_latent_argmax_channel_and_purity_raises_when_model_missing_n_instances(
        self,
    ) -> None:
        # given
        decoder = torch.eye(2, dtype=torch.float32).unsqueeze(0)

        class _NoInstances(torch.nn.Module):
            def __init__(self, w_dec: torch.Tensor):
                super().__init__()
                self.W_dec = torch.nn.Parameter(w_dec)

        model = _NoInstances(decoder)

        # when / then
        with self.assertRaises(ValueError):
            _ = calculate_latent_argmax_channel_and_purity(model)

    def test_calculate_latent_argmax_channel_and_purity_raises_when_model_has_multiple_instances(
        self,
    ) -> None:
        # given
        decoder = torch.eye(2, dtype=torch.float32)

        class _TwoInstances(torch.nn.Module):
            def __init__(self, w_dec: torch.Tensor):
                super().__init__()
                self.n_instances = 2
                self.W_dec = torch.nn.Parameter(w_dec.unsqueeze(0).repeat(2, 1, 1))

        model = _TwoInstances(decoder)

        # when / then
        with self.assertRaises(ValueError):
            _ = calculate_latent_argmax_channel_and_purity(model)

    def test_calculate_latent_argmax_channel_and_purity_raises_when_decoder_not_tensor(
        self,
    ) -> None:
        # given
        class _InvalidDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_instances = 1
                self.W_dec = "not_a_tensor"

        model = _InvalidDecoder()

        # when / then
        with self.assertRaises(ValueError):
            _ = calculate_latent_argmax_channel_and_purity(model)

    def test_calculate_latent_argmax_channel_and_purity_raises_when_decoder_has_wrong_ndim(
        self,
    ) -> None:
        # given
        class _WrongShapeDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_instances = 1
                self.W_dec = torch.nn.Parameter(
                    torch.zeros((2, 2), dtype=torch.float32)
                )

        model = _WrongShapeDecoder()

        # when / then
        with self.assertRaises(ValueError):
            _ = calculate_latent_argmax_channel_and_purity(model)

    def test_calculate_purity_summary_raises_for_non_vector_or_empty(self) -> None:
        # given
        non_vector = torch.zeros((2, 2), dtype=torch.float32).numpy()
        empty = torch.zeros((0,), dtype=torch.float32).numpy()

        # when / then
        with self.assertRaises(ValueError):
            _ = _calculate_purity_summary(non_vector)
        with self.assertRaises(ValueError):
            _ = _calculate_purity_summary(empty)
