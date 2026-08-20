import unittest

import torch

from sgnsae.models import GatedSAE


class TestGatedSAEAuxCoeffBackwardCompat(unittest.TestCase):
    def test_calculate_losses_matches_legacy_formula_when_defaults_are_used(
        self,
    ) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=10,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            aux_loss_coeff=1.0,
            dead_zone_reg_coeff=0.0,
            dead_zone_reg_type="l2",
            device="cpu",
        )
        with torch.no_grad():
            model.act_offset_left.data = torch.tensor([[0.2, 0.4]])
            model.act_offset_right.data = torch.tensor([[0.1, 0.3]])
        y_true = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]], dtype=torch.float32)
        output = {
            "x_hat": torch.tensor([[[0.5, 1.5], [1.0, 2.0]]], dtype=torch.float32),
            "pi_gate": torch.tensor([[[0.4, -0.8], [0.9, -0.2]]], dtype=torch.float32),
            "x_hat_frozen": torch.tensor(
                [[[0.0, 0.0], [2.5, 2.5]]], dtype=torch.float32
            ),
        }
        sparsity_coeffs = torch.tensor([0.25], dtype=torch.float32)

        # when
        losses = model.calculate_losses(y_true, sparsity_coeffs, output)

        # then
        reconstruction_loss = (output["x_hat"] - y_true).pow(2).mean(dim=(-1, -2))
        delta_pos = torch.relu(model.act_offset_right).unsqueeze(1)
        delta_neg = torch.relu(model.act_offset_left).unsqueeze(1)
        sparsity_source = torch.relu(output["pi_gate"] - delta_pos) + torch.relu(
            -output["pi_gate"] - delta_neg
        )
        sparsity_loss = sparsity_source.sum(dim=-1).mean(-1)
        aux_loss = (output["x_hat_frozen"] - y_true).pow(2).mean(dim=(-1, -2))
        expected_losses = (
            reconstruction_loss + sparsity_coeffs * sparsity_loss + aux_loss
        )
        torch.testing.assert_close(
            losses["losses"], expected_losses, atol=1e-8, rtol=0.0
        )

    def test_calculate_losses_applies_aux_and_dead_zone_coefficients(self) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=10,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            aux_loss_coeff=0.5,
            dead_zone_reg_coeff=0.2,
            dead_zone_reg_type="l2",
            device="cpu",
        )
        with torch.no_grad():
            model.act_offset.data = torch.tensor([[1.5, 0.5]])
        y_true = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
        output = {
            "x_hat": torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
            "pi_gate": torch.tensor([[[2.0, -1.0]]], dtype=torch.float32),
            "x_hat_frozen": torch.tensor([[[3.0, 4.0]]], dtype=torch.float32),
        }
        sparsity_coeffs = torch.tensor([0.1], dtype=torch.float32)

        # when
        losses = model.calculate_losses(y_true, sparsity_coeffs, output)

        # then
        reconstruction_loss = (output["x_hat"] - y_true).pow(2).mean(dim=(-1, -2))
        delta = torch.relu(model.act_offset).unsqueeze(1)
        sparsity_source = torch.relu(output["pi_gate"] - delta) + torch.relu(
            -output["pi_gate"] - delta
        )
        sparsity_loss = sparsity_source.sum(dim=-1).mean(-1)
        aux_loss = (output["x_hat_frozen"] - y_true).pow(2).mean(dim=(-1, -2))
        dead_zone_regularization = (
            torch.relu(model.act_offset).pow(2) + torch.relu(model.act_offset).pow(2)
        ).mean(dim=-1)
        expected_losses = (
            reconstruction_loss
            + sparsity_coeffs * sparsity_loss
            + 0.5 * aux_loss
            + 0.2 * dead_zone_regularization
        )
        torch.testing.assert_close(
            losses["dead_zone_regularization"],
            dead_zone_regularization,
            atol=1e-8,
            rtol=0.0,
        )
        torch.testing.assert_close(
            losses["losses"], expected_losses, atol=1e-8, rtol=0.0
        )

    def test_calculate_losses_uses_zero_dead_zone_regularization_for_non_sign_aware(
        self,
    ) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=10,
            sign_aware=False,
            aux_loss_coeff=1.0,
            dead_zone_reg_coeff=0.9,
            dead_zone_reg_type="l2",
            device="cpu",
        )
        y_true = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        output = {
            "x_hat": torch.tensor([[[0.5, 2.0]]], dtype=torch.float32),
            "pi_gate": torch.tensor([[[1.0, -1.0]]], dtype=torch.float32),
            "x_hat_frozen": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        }
        sparsity_coeffs = torch.tensor([0.3], dtype=torch.float32)

        # when
        losses = model.calculate_losses(y_true, sparsity_coeffs, output)

        # then
        self.assertAlmostEqual(
            float(losses["dead_zone_regularization"][0].item()), 0.0, places=12
        )

    def test_constructor_rejects_invalid_dead_zone_reg_type(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = GatedSAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=10,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=True,
                aux_loss_coeff=1.0,
                dead_zone_reg_coeff=0.0,
                dead_zone_reg_type="l1",
                device="cpu",
            )

    def test_constructor_rejects_invalid_loss_coeff_values(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = GatedSAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=10,
                sign_aware=False,
                aux_loss_coeff=-0.1,
                dead_zone_reg_coeff=0.0,
                dead_zone_reg_type="l2",
                device="cpu",
            )
        with self.assertRaises(ValueError):
            _ = GatedSAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=10,
                sign_aware=False,
                aux_loss_coeff=1.0,
                dead_zone_reg_coeff=float("inf"),
                dead_zone_reg_type="l2",
                device="cpu",
            )

    def test_constructor_rejects_symmetric_magnitude_without_sign_aware(self) -> None:
        # given / when / then
        with self.assertRaises(ValueError):
            _ = GatedSAE(
                n_instances=1,
                d_in=2,
                d_hidden=2,
                resample_latents_threshold=10,
                sign_aware=False,
                symmetric_magnitude=True,
                aux_loss_coeff=1.0,
                dead_zone_reg_coeff=0.0,
                dead_zone_reg_type="l2",
                device="cpu",
            )

    def test_forward_rejects_invalid_input_dimensions(self) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=10,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            aux_loss_coeff=1.0,
            dead_zone_reg_coeff=0.0,
            dead_zone_reg_type="l2",
            device="cpu",
        )
        invalid_input = torch.zeros((2,), dtype=torch.float32)

        # when / then
        with self.assertRaises(ValueError):
            _ = model.forward(invalid_input)

    def test_forward_covers_non_sign_aware_gate_path(self) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=3,
            resample_latents_threshold=10,
            sign_aware=False,
            aux_loss_coeff=1.0,
            dead_zone_reg_coeff=0.0,
            dead_zone_reg_type="l2",
            device="cpu",
        )
        x = torch.tensor([[1.0, -1.0], [0.5, 2.0]], dtype=torch.float32)

        # when
        output = model.forward(x)

        # then
        self.assertEqual(tuple(output["x_hat"].shape), (1, 2, 2))
        self.assertEqual(tuple(output["pi_gate"].shape), (1, 2, 3))
        self.assertEqual(tuple(output["x_hat_frozen"].shape), (1, 2, 2))
        self.assertEqual(tuple(output["latents"].shape), (1, 2, 3))

    def test_forward_covers_single_offset_and_symmetric_magnitude_paths(self) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=3,
            resample_latents_threshold=10,
            sign_aware=True,
            symmetric_magnitude=True,
            single_activation_offset=True,
            reset_activation_offsets_on_resample=True,
            aux_loss_coeff=1.0,
            dead_zone_reg_coeff=0.0,
            dead_zone_reg_type="l2",
            device="cpu",
        )
        x = torch.tensor([[1.0, -1.0], [0.0, 0.5]], dtype=torch.float32)

        # when
        output = model.forward(x)

        # then
        self.assertEqual(tuple(output["x_hat"].shape), (1, 2, 2))
        self.assertEqual(tuple(output["pi_gate"].shape), (1, 2, 3))
        self.assertEqual(tuple(output["x_hat_frozen"].shape), (1, 2, 2))
        self.assertEqual(tuple(output["latents"].shape), (1, 2, 3))

    def test_calculate_losses_rejects_invalid_dead_zone_reg_type_runtime_value(
        self,
    ) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=10,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            aux_loss_coeff=1.0,
            dead_zone_reg_coeff=0.0,
            dead_zone_reg_type="l2",
            device="cpu",
        )
        model.dead_zone_reg_type = "l1"
        y_true = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
        output = {
            "x_hat": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
            "pi_gate": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
            "x_hat_frozen": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        }
        sparsity_coeffs = torch.tensor([0.0], dtype=torch.float32)

        # when / then
        with self.assertRaises(ValueError):
            _ = model.calculate_losses(y_true, sparsity_coeffs, output)


if __name__ == "__main__":
    unittest.main()
