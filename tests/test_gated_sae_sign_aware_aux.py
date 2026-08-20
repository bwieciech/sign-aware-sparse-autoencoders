import unittest
import torch

from sgnsae.models import GatedSAE


class TestSignAwareGatedAux(unittest.TestCase):
    def test_aux_reconstruction_is_signed_when_sign_aware(self) -> None:
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            resample_latents_threshold=100,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )
        with torch.no_grad():
            model.W_dec.data = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
            model.b_enc.data.zero_()
            model.b_dec.data.zero_()
            model.log_alpha_gate.data.zero_()
            model.r_mag.data.zero_()
            model.r_mag_neg.data.zero_()
            model.b_mag.data.zero_()
            model.act_offset.data.zero_()

        x = torch.tensor([[0.0, -1.0]])
        out = model(x)
        # Aux path should carry negative evidence in the second coord.
        self.assertLess(out["x_hat_frozen"][0, 0, 1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
