import unittest
import torch

from sgnsae.models import SAE, GatedSAE


class TestSAEModels(unittest.TestCase):
    def setUp(self):
        self.n_instances = 2
        self.d_in = 8
        self.d_hidden = 16
        self.batch_size = 4
        self.resample_latents_threshold = 100
        self.device = "cpu"

        self.sample_input = torch.randn(self.batch_size, self.d_in)

    def test_sae_act_offset_present_when_sign_aware_true(self):
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        # when
        has_act_offset = hasattr(model, "act_offset")

        # then
        self.assertTrue(has_act_offset)
        self.assertEqual(model.act_offset.shape, (self.n_instances, self.d_hidden))

    def test_sae_act_offset_absent_when_sign_aware_false(self):
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when
        has_act_offset = hasattr(model, "act_offset")

        # then
        self.assertFalse(has_act_offset)

    def test_gated_sae_act_offset_present_when_sign_aware_true(self):
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        # when
        has_act_offset = hasattr(model, "act_offset")

        # then
        self.assertTrue(has_act_offset)
        self.assertEqual(model.act_offset.shape, (self.n_instances, self.d_hidden))

    def test_gated_sae_act_offset_absent_when_sign_aware_false(self):
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when
        has_act_offset = hasattr(model, "act_offset")

        # then
        self.assertFalse(has_act_offset)

    def test_sae_forward_with_sign_aware_true(self):
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        # when
        output = model(self.sample_input)

        # then
        self.assertIn("y_pred", output)
        self.assertIn("latents", output)
        self.assertEqual(
            output["y_pred"].shape, (self.n_instances, self.batch_size, self.d_in)
        )
        self.assertEqual(
            output["latents"].shape, (self.n_instances, self.batch_size, self.d_hidden)
        )

    def test_sae_forward_with_sign_aware_false(self):
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when
        output = model(self.sample_input)

        # then
        self.assertIn("y_pred", output)
        self.assertIn("latents", output)
        self.assertEqual(
            output["y_pred"].shape, (self.n_instances, self.batch_size, self.d_in)
        )
        self.assertEqual(
            output["latents"].shape, (self.n_instances, self.batch_size, self.d_hidden)
        )

    def test_gated_sae_forward_with_sign_aware_true(self):
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        # when
        output = model(self.sample_input)

        # then
        self.assertIn("x_hat", output)
        self.assertIn("latents", output)
        self.assertIn("pi_gate", output)
        self.assertIn("x_hat_frozen", output)
        self.assertEqual(
            output["x_hat"].shape, (self.n_instances, self.batch_size, self.d_in)
        )
        self.assertEqual(
            output["latents"].shape, (self.n_instances, self.batch_size, self.d_hidden)
        )
        self.assertEqual(
            output["pi_gate"].shape, (self.n_instances, self.batch_size, self.d_hidden)
        )
        self.assertEqual(
            output["x_hat_frozen"].shape, (self.n_instances, self.batch_size, self.d_in)
        )

    def test_gated_sae_forward_with_sign_aware_false(self):
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        # when
        output = model(self.sample_input)

        # then
        self.assertIn("x_hat", output)
        self.assertIn("latents", output)
        self.assertIn("pi_gate", output)
        self.assertIn("x_hat_frozen", output)
        self.assertEqual(
            output["x_hat"].shape, (self.n_instances, self.batch_size, self.d_in)
        )
        self.assertEqual(
            output["latents"].shape, (self.n_instances, self.batch_size, self.d_hidden)
        )
        self.assertEqual(
            output["pi_gate"].shape, (self.n_instances, self.batch_size, self.d_hidden)
        )
        self.assertEqual(
            output["x_hat_frozen"].shape, (self.n_instances, self.batch_size, self.d_in)
        )

    def test_sae_latent_activation_differs_based_on_sign_aware(self):
        # given
        torch.manual_seed(42)
        input_data = torch.randn(self.batch_size, self.d_in)

        model_sign_aware = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        model_vanilla = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        with torch.no_grad():
            model_vanilla.b_enc.data = model_sign_aware.b_enc.data.clone()
            model_vanilla.W_dec.data = model_sign_aware.W_dec.data.clone()
            model_vanilla.b_dec.data = model_sign_aware.b_dec.data.clone()

        # when
        output_sign_aware = model_sign_aware(input_data)
        output_vanilla = model_vanilla(input_data)

        # then
        self.assertFalse(
            torch.allclose(output_sign_aware["latents"], output_vanilla["latents"])
        )

    def test_gated_sae_gate_behavior_differs_based_on_sign_aware(self):
        # given
        torch.manual_seed(42)
        input_data = torch.randn(self.batch_size, self.d_in) * 5

        model_sign_aware = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        model_vanilla = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=False,
            device=self.device,
        )

        with torch.no_grad():
            model_vanilla.b_enc.data = model_sign_aware.b_enc.data.clone()
            model_vanilla.W_dec.data = model_sign_aware.W_dec.data.clone()
            model_vanilla.b_dec.data = model_sign_aware.b_dec.data.clone()
            model_vanilla.r_mag.data = model_sign_aware.r_mag.data.clone()
            model_vanilla.b_mag.data = model_sign_aware.b_mag.data.clone()

        # when
        output_sign_aware = model_sign_aware(input_data)
        output_vanilla = model_vanilla(input_data)

        # then
        has_negative_vanilla = (output_vanilla["latents"] < 0).any()
        self.assertFalse(has_negative_vanilla)


    def test_sae_single_activation_offset_creates_correct_parameters(self):
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        # when & then
        self.assertTrue(hasattr(model, "act_offset"))
        self.assertFalse(hasattr(model, "act_offset_left"))
        self.assertFalse(hasattr(model, "act_offset_right"))
        self.assertEqual(model.act_offset.shape, (self.n_instances, self.d_hidden))

    def test_sae_dual_activation_offset_creates_correct_parameters(self):
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device=self.device,
        )

        # when & then
        self.assertFalse(hasattr(model, "act_offset"))
        self.assertTrue(hasattr(model, "act_offset_left"))
        self.assertTrue(hasattr(model, "act_offset_right"))
        self.assertEqual(model.act_offset_left.shape, (self.n_instances, self.d_hidden))
        self.assertEqual(model.act_offset_right.shape, (self.n_instances, self.d_hidden))

    def test_gated_sae_single_activation_offset_creates_correct_parameters(self):
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )

        # when & then
        self.assertTrue(hasattr(model, "act_offset"))
        self.assertFalse(hasattr(model, "act_offset_left"))
        self.assertFalse(hasattr(model, "act_offset_right"))
        self.assertEqual(model.act_offset.shape, (self.n_instances, self.d_hidden))

    def test_gated_sae_dual_activation_offset_creates_correct_parameters(self):
        # given
        model = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device=self.device,
        )

        # when & then
        self.assertFalse(hasattr(model, "act_offset"))
        self.assertTrue(hasattr(model, "act_offset_left"))
        self.assertTrue(hasattr(model, "act_offset_right"))
        self.assertEqual(model.act_offset_left.shape, (self.n_instances, self.d_hidden))
        self.assertEqual(model.act_offset_right.shape, (self.n_instances, self.d_hidden))

    def test_sae_single_vs_dual_offset_produces_different_latents(self):
        # given
        torch.manual_seed(42)
        input_data = torch.randn(self.batch_size, self.d_in) * 2
        
        model_single = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )
        
        model_dual = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device=self.device,
        )
        
        with torch.no_grad():
            model_dual.b_enc.data = model_single.b_enc.data.clone()
            model_dual.W_dec.data = model_single.W_dec.data.clone()
            model_dual.b_dec.data = model_single.b_dec.data.clone()
            
            model_dual.act_offset_left.data.fill_(0.1)
            model_dual.act_offset_right.data.fill_(2.0)
            model_single.act_offset.data.fill_(1.0)

        # when
        output_single = model_single(input_data)
        output_dual = model_dual(input_data)

        # then
        self.assertFalse(torch.allclose(output_single["latents"], output_dual["latents"]))

    def test_gated_sae_single_vs_dual_offset_produces_different_latents(self):
        # given
        torch.manual_seed(42)
        input_data = torch.randn(self.batch_size, self.d_in) * 2
        
        model_single = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device=self.device,
        )
        
        model_dual = GatedSAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device=self.device,
        )
        
        with torch.no_grad():
            model_dual.b_enc.data = model_single.b_enc.data.clone()
            model_dual.W_dec.data = model_single.W_dec.data.clone()
            model_dual.b_dec.data = model_single.b_dec.data.clone()
            model_dual.r_mag.data = model_single.r_mag.data.clone()
            model_dual.b_mag.data = model_single.b_mag.data.clone()
            
            model_dual.act_offset_left.data.fill_(0.1)
            model_dual.act_offset_right.data.fill_(2.0)
            model_single.act_offset.data.fill_(1.0)

        # when
        output_single = model_single(input_data)
        output_dual = model_dual(input_data)

        # then
        self.assertFalse(torch.allclose(output_single["latents"], output_dual["latents"]))

    def test_sae_dual_offset_handles_negative_parameters_correctly(self):
        # given
        model = SAE(
            n_instances=self.n_instances,
            d_in=self.d_in,
            d_hidden=self.d_hidden,
            resample_latents_threshold=self.resample_latents_threshold,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=False,
            device=self.device,
        )
        
        with torch.no_grad():
            model.act_offset_left.data.fill_(-1.0)
            model.act_offset_right.data.fill_(-2.0)
        
        input_data = torch.randn(self.batch_size, self.d_in)

        # when
        output = model(input_data)

        # then
        self.assertEqual(output["latents"].shape, (self.n_instances, self.batch_size, self.d_hidden))
        self.assertEqual(output["y_pred"].shape, (self.n_instances, self.batch_size, self.d_in))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
