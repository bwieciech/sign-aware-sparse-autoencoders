import unittest
import torch

from sgnsae.models.gated_sae import GatedSAE


class TestSymmetricMagnitudeAblation(unittest.TestCase):
    def test_symmetric_magnitude_has_no_r_mag_neg_parameter(self) -> None:
        model = GatedSAE(
            n_instances=1,
            d_in=1,
            d_hidden=1,
            resample_latents_threshold=100,
            sign_aware=True,
            symmetric_magnitude=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )
        names = {name for name, _ in model.named_parameters()}
        self.assertIn("r_mag", names)
        self.assertNotIn("r_mag_neg", names)
        self.assertFalse(hasattr(model, "r_mag_neg"))

    def test_negative_branch_uses_r_mag_when_symmetric_magnitude_true(self) -> None:
        model = GatedSAE(
            n_instances=1,
            d_in=1,
            d_hidden=1,
            resample_latents_threshold=100,
            sign_aware=True,
            symmetric_magnitude=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.tensor([[[1.0]]])
            model.b_dec.data.zero_()
            model.b_enc.data.zero_()
            model.log_alpha_gate.data.zero_()
            model.b_mag.data.zero_()
            model.act_offset.data.zero_()
            model.r_mag.data.fill_(torch.log(torch.tensor(2.0)))  # scale_pos = 2

        x = torch.tensor([[-1.0]])
        out = model(x)
        # With symmetric magnitudes: scale_neg == scale_pos == 2
        # t = -1 => negative regime, a_neg = -ReLU(-(-2 * -1) + 0) = -ReLU(2) = -2
        self.assertAlmostEqual(out["latents"][0, 0, 0].item(), -2.0, places=6)


if __name__ == "__main__":
    unittest.main()
