import unittest

import torch

from sgnsae.models import AbsTopKGatedMagnitudeSAE


class TestAbsTopKGatedMagnitudeSAE(unittest.TestCase):
    def test_forward_shape_contract_for_rank_2_and_rank_3_input(self) -> None:
        # given
        model = AbsTopKGatedMagnitudeSAE(
            n_instances=2,
            d_in=4,
            d_hidden=5,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )
        rank_2_input = torch.randn(3, 4)
        rank_3_input = torch.randn(2, 3, 4)

        # when
        rank_2_output = model(rank_2_input)
        rank_3_output = model(rank_3_input)

        # then
        self.assertEqual(rank_2_output["latents"].shape, (2, 3, 5))
        self.assertEqual(rank_2_output["x_hat"].shape, (2, 3, 4))
        self.assertEqual(rank_2_output["y_pred"].shape, (2, 3, 4))
        self.assertEqual(rank_3_output["latents"].shape, (2, 3, 5))
        self.assertEqual(rank_3_output["x_hat"].shape, (2, 3, 4))
        self.assertEqual(rank_3_output["y_pred"].shape, (2, 3, 4))

    def test_forward_raises_for_invalid_input_rank(self) -> None:
        # given
        model = AbsTopKGatedMagnitudeSAE(
            n_instances=1,
            d_in=4,
            d_hidden=3,
            topk_k=1,
            resample_latents_threshold=3,
            device="cpu",
        )
        rank_1_input = torch.randn(4)
        rank_4_input = torch.randn(1, 2, 3, 4)

        # when / then
        with self.assertRaises(ValueError):
            _ = model(rank_1_input)
        with self.assertRaises(ValueError):
            _ = model(rank_4_input)

    def test_constructor_rejects_invalid_topk_k(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = AbsTopKGatedMagnitudeSAE(
                n_instances=1,
                d_in=4,
                d_hidden=3,
                topk_k=0,
                resample_latents_threshold=3,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            _ = AbsTopKGatedMagnitudeSAE(
                n_instances=1,
                d_in=4,
                d_hidden=3,
                topk_k=4,
                resample_latents_threshold=3,
                device="cpu",
            )

    def test_forward_is_deterministic_for_fixed_seed(self) -> None:
        # given
        torch.manual_seed(1234)
        input_tensor = torch.randn(5, 4)
        torch.manual_seed(1234)
        model_a = AbsTopKGatedMagnitudeSAE(
            n_instances=1,
            d_in=4,
            d_hidden=6,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )
        torch.manual_seed(1234)
        model_b = AbsTopKGatedMagnitudeSAE(
            n_instances=1,
            d_in=4,
            d_hidden=6,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )

        # when
        output_a = model_a(input_tensor)
        output_b = model_b(input_tensor)

        # then
        torch.testing.assert_close(output_a["latents"], output_b["latents"])
        torch.testing.assert_close(output_a["x_hat"], output_b["x_hat"])
        torch.testing.assert_close(output_a["y_pred"], output_b["y_pred"])

    def test_latents_preserve_both_signs_for_selected_features(self) -> None:
        # given
        model = AbsTopKGatedMagnitudeSAE(
            n_instances=1,
            d_in=3,
            d_hidden=3,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
            )
            model.b_dec.data.zero_()
            model.r_mag.data.zero_()
            model.r_mag_neg.data.zero_()
            model.b_mag.data.zero_()
        input_tensor = torch.tensor([[2.0, -3.0, 0.5]], dtype=torch.float32)

        # when
        output = model(input_tensor)
        latents = output["latents"][0, 0]

        # then
        self.assertGreater(float(latents[0].item()), 0.0)
        self.assertLess(float(latents[1].item()), 0.0)
        self.assertAlmostEqual(float(latents[2].item()), 0.0, places=12)

    def test_calculate_losses_returns_expected_keys_and_shapes(self) -> None:
        # given
        model = AbsTopKGatedMagnitudeSAE(
            n_instances=2,
            d_in=3,
            d_hidden=4,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )
        input_tensor = torch.randn(5, 3)
        sparsity_coeffs = torch.tensor([0.1, 0.2], dtype=torch.float32)

        # when
        output = model(input_tensor)
        losses = model.calculate_losses(input_tensor, sparsity_coeffs, output)

        # then
        self.assertEqual(losses["losses"].shape, (2,))
        self.assertEqual(losses["reconstruction_losses"].shape, (2,))
        self.assertEqual(losses["sparsity_losses"].shape, (2,))
        self.assertTrue(torch.isfinite(losses["losses"]).all())
        self.assertTrue(torch.equal(losses["losses"], losses["reconstruction_losses"]))
        self.assertTrue(
            torch.equal(losses["sparsity_losses"], torch.zeros_like(losses["losses"]))
        )


if __name__ == "__main__":
    unittest.main()
