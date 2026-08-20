import importlib
import unittest

import torch

from sgnsae.models.topk_utils import deterministic_topk_mask
from sgnsae.models import AbsTopKSAE, JumpReLUSAE, TopKSAE


def inverse_softplus(value: float) -> float:
    return float(
        torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32))).item()
    )


class TestBaselineModels(unittest.TestCase):
    def test_deterministic_topk_mask_treats_nan_as_negative_infinity(self) -> None:
        # given
        scores = torch.tensor(
            [[float("nan"), -float("inf"), 0.0, 1.0]], dtype=torch.float32
        )

        # when
        mask = deterministic_topk_mask(scores, topk_k=1)
        selected_indices = torch.nonzero(mask[0], as_tuple=False).squeeze(-1)

        # then
        self.assertEqual(selected_indices.tolist(), [3])

    def test_forward_shape_contract_for_rank_2_input(self) -> None:
        # given
        x = torch.randn(3, 4)
        models = [
            JumpReLUSAE(
                n_instances=2,
                d_in=4,
                d_hidden=5,
                resample_latents_threshold=3,
                device="cpu",
            ),
            TopKSAE(
                n_instances=2,
                d_in=4,
                d_hidden=5,
                topk_k=2,
                resample_latents_threshold=3,
                device="cpu",
            ),
            AbsTopKSAE(
                n_instances=2,
                d_in=4,
                d_hidden=5,
                topk_k=2,
                resample_latents_threshold=3,
                device="cpu",
            ),
        ]

        # when / then
        for model in models:
            output = model(x)
            self.assertEqual(output["latents"].shape, (2, 3, 5))
            self.assertEqual(output["x_hat"].shape, (2, 3, 4))
            self.assertEqual(output["y_pred"].shape, (2, 3, 4))

    def test_forward_shape_contract_for_rank_3_input(self) -> None:
        # given
        x = torch.randn(2, 3, 4)
        models = [
            JumpReLUSAE(
                n_instances=2,
                d_in=4,
                d_hidden=5,
                resample_latents_threshold=3,
                device="cpu",
            ),
            TopKSAE(
                n_instances=2,
                d_in=4,
                d_hidden=5,
                topk_k=2,
                resample_latents_threshold=3,
                device="cpu",
            ),
            AbsTopKSAE(
                n_instances=2,
                d_in=4,
                d_hidden=5,
                topk_k=2,
                resample_latents_threshold=3,
                device="cpu",
            ),
        ]

        # when / then
        for model in models:
            output = model(x)
            self.assertEqual(output["latents"].shape, (2, 3, 5))
            self.assertEqual(output["x_hat"].shape, (2, 3, 4))
            self.assertEqual(output["y_pred"].shape, (2, 3, 4))

    def test_forward_raises_for_invalid_input_rank(self) -> None:
        # given
        rank_1_input = torch.randn(4)
        rank_4_input = torch.randn(1, 2, 3, 4)
        models = [
            JumpReLUSAE(1, 4, 3, 3, "cpu"),
            TopKSAE(1, 4, 3, 1, 3, "cpu"),
            AbsTopKSAE(1, 4, 3, 1, 3, "cpu"),
        ]

        # when / then
        for model in models:
            with self.assertRaises(ValueError):
                _ = model(rank_1_input)
            with self.assertRaises(ValueError):
                _ = model(rank_4_input)

    def test_reconstruction_alias_matches_x_hat(self) -> None:
        # given
        x = torch.randn(4, 4)
        models = [
            JumpReLUSAE(1, 4, 3, 3, "cpu"),
            TopKSAE(1, 4, 3, 2, 3, "cpu"),
            AbsTopKSAE(1, 4, 3, 2, 3, "cpu"),
        ]

        # when / then
        for model in models:
            output = model(x)
            torch.testing.assert_close(output["x_hat"], output["y_pred"])

    def test_jump_relu_threshold_and_jump_semantics(self) -> None:
        # given
        model = JumpReLUSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=10,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
            model.b_enc.data.zero_()
            model.b_dec.data.zero_()
            model.theta_param.data.fill_(inverse_softplus(0.5))
            model.jump_param.data.fill_(inverse_softplus(0.25))
        x = torch.tensor([[0.25, 1.5]], dtype=torch.float32)

        # when
        output = model(x)
        first_latent = output["latents"][0, 0, 0].item()
        second_latent = output["latents"][0, 0, 1].item()

        # then
        self.assertAlmostEqual(first_latent, 0.0, places=6)
        self.assertAlmostEqual(second_latent, 1.25, places=5)

    def test_jump_relu_calculate_losses_returns_expected_keys_and_shapes(self) -> None:
        # given
        model = JumpReLUSAE(
            n_instances=2,
            d_in=3,
            d_hidden=4,
            resample_latents_threshold=10,
            device="cpu",
        )
        x = torch.randn(5, 3)
        sparsity_coeffs = torch.tensor([0.1, 0.2], dtype=torch.float32)

        # when
        output = model(x)
        losses = model.calculate_losses(x, sparsity_coeffs, output)

        # then
        self.assertIn("losses", losses)
        self.assertIn("reconstruction_losses", losses)
        self.assertIn("sparsity_losses", losses)
        self.assertEqual(losses["losses"].shape, (2,))
        self.assertEqual(losses["reconstruction_losses"].shape, (2,))
        self.assertEqual(losses["sparsity_losses"].shape, (2,))
        self.assertTrue(torch.isfinite(losses["losses"]).all())

    def test_jump_relu_supports_non_detached_sparsity_source(self) -> None:
        # given
        model = JumpReLUSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=10,
            device="cpu",
            detach_sparsity_source=False,
        )
        x = torch.tensor([[0.1, -0.2], [1.0, 0.5]], dtype=torch.float32)
        sparsity_coeffs = torch.tensor([0.05], dtype=torch.float32)

        # when
        output = model(x)
        losses = model.calculate_losses(x, sparsity_coeffs, output)

        # then
        self.assertTrue(torch.isfinite(losses["losses"]).all())

    def test_jump_relu_resample_resets_theta_and_jump_parameters(self) -> None:
        # given
        model = JumpReLUSAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=0,
            device="cpu",
        )
        with torch.no_grad():
            model.theta_param.data = torch.tensor([[4.0, 5.0]])
            model.jump_param.data = torch.tensor([[6.0, 7.0]])
        inactive_latents_count = torch.tensor([[1, 0]])

        # when
        mask = model.resample_dead_latents(inactive_latents_count)

        # then
        self.assertTrue(mask[0, 0].item())
        self.assertFalse(mask[0, 1].item())
        self.assertAlmostEqual(
            torch.nn.functional.softplus(model.theta_param[0, 0]).item(),
            1e-3,
            places=6,
        )
        self.assertAlmostEqual(
            torch.nn.functional.softplus(model.jump_param[0, 0]).item(),
            1e-3,
            places=6,
        )
        self.assertAlmostEqual(model.theta_param[0, 1].item(), 5.0, places=6)
        self.assertAlmostEqual(model.jump_param[0, 1].item(), 7.0, places=6)

    def test_jump_relu_resample_noop_when_below_threshold(self) -> None:
        # given
        model = JumpReLUSAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=10,
            device="cpu",
        )
        with torch.no_grad():
            model.theta_param.data = torch.tensor([[2.0, 3.0]])
            model.jump_param.data = torch.tensor([[4.0, 5.0]])
        inactive_latents_count = torch.tensor([[1, 1]])

        # when
        mask = model.resample_dead_latents(inactive_latents_count)

        # then
        self.assertFalse(mask.any().item())
        self.assertAlmostEqual(model.theta_param[0, 0].item(), 2.0, places=6)
        self.assertAlmostEqual(model.theta_param[0, 1].item(), 3.0, places=6)
        self.assertAlmostEqual(model.jump_param[0, 0].item(), 4.0, places=6)
        self.assertAlmostEqual(model.jump_param[0, 1].item(), 5.0, places=6)

    def test_topk_non_negativity_and_exact_k_with_positive_preactivations(self) -> None:
        # given
        model = TopKSAE(
            n_instances=1,
            d_in=5,
            d_hidden=5,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.eye(5, dtype=torch.float32).unsqueeze(0)
            model.b_enc.data.zero_()
            model.b_dec.data.zero_()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])

        # when
        output = model(x)
        latents = output["latents"][0]
        nonzero_counts = (latents > 0).sum(dim=-1)

        # then
        self.assertGreaterEqual(latents.min().item(), 0.0)
        torch.testing.assert_close(nonzero_counts, torch.tensor([2, 2]))

    def test_abstopk_sign_preservation_and_absolute_value_selection(self) -> None:
        # given
        model = AbsTopKSAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            model.b_enc.data.zero_()
            model.b_dec.data.zero_()
        x = torch.tensor([[-5.0, 4.0, 1.0, -0.5]], dtype=torch.float32)

        # when
        output = model(x)
        latents = output["latents"][0, 0]
        selected_indices = torch.nonzero(latents != 0.0).squeeze(-1).tolist()

        # then
        self.assertEqual(selected_indices, [0, 1])
        self.assertAlmostEqual(latents[0].item(), -5.0, places=6)
        self.assertAlmostEqual(latents[1].item(), 4.0, places=6)
        self.assertTrue(model.signed_latents)

    def test_topk_tie_breaking_is_deterministic_and_prefers_lower_indices(self) -> None:
        # given
        model = TopKSAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            model.b_enc.data.zero_()
            model.b_dec.data.zero_()
        x = torch.ones(1, 4, dtype=torch.float32)

        # when
        output_1 = model(x)
        output_2 = model(x)
        indices_1 = torch.nonzero(output_1["latents"][0, 0] > 0.0).squeeze(-1).tolist()
        indices_2 = torch.nonzero(output_2["latents"][0, 0] > 0.0).squeeze(-1).tolist()

        # then
        self.assertEqual(indices_1, [0, 1])
        self.assertEqual(indices_2, [0, 1])

    def test_topk_non_tie_large_hidden_keeps_true_high_index_winner(self) -> None:
        # given
        d_hidden = 16_384
        model = TopKSAE(
            n_instances=1,
            d_in=1,
            d_hidden=d_hidden,
            topk_k=1,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.b_enc.zero_()
            model.b_dec.zero_()
            model.b_enc[0, 0] = 0.015
            model.b_enc[0, d_hidden - 1] = 0.020
        x = torch.zeros(1, 1, dtype=torch.float32)

        # when
        output = model(x)
        selected_indices = torch.nonzero(
            output["latents"][0, 0] > 0.0, as_tuple=False
        ).squeeze(-1)

        # then
        self.assertEqual(selected_indices.tolist(), [d_hidden - 1])

    def test_topk_non_zero_tie_large_hidden_prefers_lower_indices(self) -> None:
        # given
        d_hidden = 16_384
        topk_k = 2
        model = TopKSAE(
            n_instances=1,
            d_in=1,
            d_hidden=d_hidden,
            topk_k=topk_k,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.b_enc.zero_()
            model.b_dec.zero_()
            model.b_enc[0, : topk_k + 2] = 1.0
        x = torch.zeros(1, 1, dtype=torch.float32)

        # when
        output = model(x)
        selected_indices = torch.nonzero(
            output["latents"][0, 0] > 0.0, as_tuple=False
        ).squeeze(-1)

        # then
        self.assertEqual(selected_indices.tolist(), [0, 1])

    def test_topk_boundary_tie_does_not_admit_close_below_values(self) -> None:
        # given
        d_hidden = 4_096
        topk_k = 32
        tie_count = 128
        model = TopKSAE(
            n_instances=1,
            d_in=1,
            d_hidden=d_hidden,
            topk_k=topk_k,
            resample_latents_threshold=3,
            device="cpu",
        )
        one = torch.tensor(1.0, dtype=torch.float32)
        next_lower = torch.nextafter(
            one, torch.tensor(-float("inf"), dtype=torch.float32)
        ).item()
        with torch.no_grad():
            model.b_enc.zero_()
            model.b_dec.zero_()
            model.b_enc[0, :tie_count] = 1.0
            model.b_enc[0, tie_count:] = next_lower
        x = torch.zeros(1, 1, dtype=torch.float32)

        # when
        output = model(x)
        selected_indices = torch.nonzero(
            output["latents"][0, 0] > 0.0, as_tuple=False
        ).squeeze(-1)

        # then
        self.assertTrue((selected_indices < tie_count).all().item())
        self.assertEqual(selected_indices.tolist(), list(range(topk_k)))

    def test_topk_tie_breaking_prefers_lower_indices_in_bfloat16(self) -> None:
        # given
        model = TopKSAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        ).to(dtype=torch.bfloat16)
        with torch.no_grad():
            model.W_dec.copy_(torch.eye(4, dtype=torch.bfloat16).unsqueeze(0))
            model.b_enc.zero_()
            model.b_dec.zero_()
        x = torch.ones(1, 4, dtype=torch.bfloat16)

        # when
        try:
            output = model(x)
        except (RuntimeError, NotImplementedError) as exception:
            self.skipTest(
                "bfloat16 forward is not supported in this environment: " f"{exception}"
            )
        selected_indices = torch.nonzero(
            output["latents"][0, 0] > 0.0, as_tuple=False
        ).squeeze(-1)

        # then
        self.assertEqual(selected_indices.tolist(), [0, 1])

    def test_abstopk_tie_breaking_is_deterministic_and_prefers_lower_indices(
        self,
    ) -> None:
        # given
        model = AbsTopKSAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            model.b_enc.data.zero_()
            model.b_dec.data.zero_()
        x = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float32)

        # when
        output_1 = model(x)
        output_2 = model(x)
        latents_1 = output_1["latents"][0, 0]
        latents_2 = output_2["latents"][0, 0]
        indices_1 = torch.nonzero(latents_1 != 0.0).squeeze(-1).tolist()
        indices_2 = torch.nonzero(latents_2 != 0.0).squeeze(-1).tolist()

        # then
        self.assertEqual(indices_1, [0, 1])
        self.assertEqual(indices_2, [0, 1])

    def test_abstopk_non_tie_large_hidden_keeps_true_high_index_winner(self) -> None:
        # given
        d_hidden = 16_384
        model = AbsTopKSAE(
            n_instances=1,
            d_in=1,
            d_hidden=d_hidden,
            topk_k=1,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.b_enc.zero_()
            model.b_dec.zero_()
            model.b_enc[0, 0] = 0.015
            model.b_enc[0, d_hidden - 1] = -0.020
        x = torch.zeros(1, 1, dtype=torch.float32)

        # when
        output = model(x)
        selected_indices = torch.nonzero(
            output["latents"][0, 0] != 0.0, as_tuple=False
        ).squeeze(-1)

        # then
        self.assertEqual(selected_indices.tolist(), [d_hidden - 1])

    def test_abstopk_non_zero_tie_large_hidden_prefers_lower_indices(self) -> None:
        # given
        d_hidden = 16_384
        topk_k = 2
        model = AbsTopKSAE(
            n_instances=1,
            d_in=1,
            d_hidden=d_hidden,
            topk_k=topk_k,
            resample_latents_threshold=3,
            device="cpu",
        )
        with torch.no_grad():
            model.b_enc.zero_()
            model.b_dec.zero_()
            model.b_enc[0, : topk_k + 2] = -1.0
        x = torch.zeros(1, 1, dtype=torch.float32)

        # when
        output = model(x)
        selected_indices = torch.nonzero(
            output["latents"][0, 0] != 0.0, as_tuple=False
        ).squeeze(-1)

        # then
        self.assertEqual(selected_indices.tolist(), [0, 1])

    def test_abstopk_boundary_tie_does_not_admit_close_below_values(self) -> None:
        # given
        d_hidden = 4_096
        topk_k = 32
        tie_count = 128
        model = AbsTopKSAE(
            n_instances=1,
            d_in=1,
            d_hidden=d_hidden,
            topk_k=topk_k,
            resample_latents_threshold=3,
            device="cpu",
        )
        one = torch.tensor(1.0, dtype=torch.float32)
        next_lower = torch.nextafter(
            one, torch.tensor(-float("inf"), dtype=torch.float32)
        ).item()
        with torch.no_grad():
            model.b_enc.zero_()
            model.b_dec.zero_()
            model.b_enc[0, :tie_count] = -1.0
            model.b_enc[0, tie_count:] = -next_lower
        x = torch.zeros(1, 1, dtype=torch.float32)

        # when
        output = model(x)
        selected_indices = torch.nonzero(
            output["latents"][0, 0] != 0.0, as_tuple=False
        ).squeeze(-1)

        # then
        self.assertTrue((selected_indices < tie_count).all().item())
        self.assertEqual(selected_indices.tolist(), list(range(topk_k)))

    def test_abstopk_tie_breaking_prefers_lower_indices_in_bfloat16(self) -> None:
        # given
        model = AbsTopKSAE(
            n_instances=1,
            d_in=4,
            d_hidden=4,
            topk_k=2,
            resample_latents_threshold=3,
            device="cpu",
        ).to(dtype=torch.bfloat16)
        with torch.no_grad():
            model.W_dec.copy_(torch.eye(4, dtype=torch.bfloat16).unsqueeze(0))
            model.b_enc.zero_()
            model.b_dec.zero_()
        x = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.bfloat16)

        # when
        try:
            output = model(x)
        except (RuntimeError, NotImplementedError) as exception:
            self.skipTest(
                "bfloat16 forward is not supported in this environment: " f"{exception}"
            )
        selected_indices = torch.nonzero(
            output["latents"][0, 0] != 0.0, as_tuple=False
        ).squeeze(-1)

        # then
        self.assertEqual(selected_indices.tolist(), [0, 1])

    def test_topk_and_abstopk_losses_are_reconstruction_only(self) -> None:
        # given
        models = [
            TopKSAE(2, 3, 4, 2, 3, "cpu"),
            AbsTopKSAE(2, 3, 4, 2, 3, "cpu"),
        ]
        x = torch.randn(5, 3)
        sparsity_coeffs = torch.tensor([0.2, 0.4], dtype=torch.float32)

        # when / then
        for model in models:
            output = model(x)
            losses = model.calculate_losses(x, sparsity_coeffs, output)
            self.assertIn("losses", losses)
            self.assertIn("reconstruction_losses", losses)
            self.assertIn("sparsity_losses", losses)
            torch.testing.assert_close(
                losses["losses"], losses["reconstruction_losses"]
            )
            torch.testing.assert_close(
                losses["sparsity_losses"], torch.zeros_like(losses["sparsity_losses"])
            )

    def test_topk_constructor_validates_k_range(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = TopKSAE(1, 3, 4, 0, 3, "cpu")
        with self.assertRaises(ValueError):
            _ = TopKSAE(1, 3, 4, 5, 3, "cpu")

    def test_abstopk_constructor_validates_k_range(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = AbsTopKSAE(1, 3, 4, -1, 3, "cpu")
        with self.assertRaises(ValueError):
            _ = AbsTopKSAE(1, 3, 4, 7, 3, "cpu")

    def test_jump_relu_forward_preserves_bfloat16_latent_dtype(self) -> None:
        # given
        model = JumpReLUSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=3,
            device="cpu",
        ).to(dtype=torch.bfloat16)
        x = torch.tensor([[0.2, -0.1]], dtype=torch.bfloat16)

        # when
        try:
            output = model(x)
        except (RuntimeError, NotImplementedError) as exception:
            self.skipTest(
                "bfloat16 forward is not supported in this environment: " f"{exception}"
            )

        # then
        self.assertEqual(output["latents"].dtype, torch.bfloat16)

    def test_model_imports_exposed_via_package_init(self) -> None:
        # given
        models_module = importlib.import_module("sgnsae.models")

        # then
        self.assertTrue(isinstance(getattr(models_module, "JumpReLUSAE"), type))
        self.assertTrue(isinstance(getattr(models_module, "TopKSAE"), type))
        self.assertTrue(isinstance(getattr(models_module, "AbsTopKSAE"), type))


if __name__ == "__main__":
    unittest.main()
