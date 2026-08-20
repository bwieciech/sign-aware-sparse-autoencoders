import unittest
import torch

from sgnsae.models import GatedSAE


class TestResampleResetsGatedScalars(unittest.TestCase):
    def test_resample_resets_r_and_b_mag(self) -> None:
        model = GatedSAE(
            n_instances=1,
            d_in=3,
            d_hidden=2,
            resample_latents_threshold=1,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            device="cpu",
        )
        with torch.no_grad():
            model.r_mag.data.fill_(1.0)
            model.r_mag_neg.data.fill_(2.0)
            model.b_mag.data.fill_(3.0)

        inactive = torch.tensor([[2, 0]])  # latent 0 is dead
        mask = model.resample_dead_latents(inactive)
        self.assertTrue(mask[0, 0].item())
        self.assertAlmostEqual(model.r_mag[0, 0].item(), 0.0)
        self.assertAlmostEqual(model.r_mag_neg[0, 0].item(), 0.0)
        self.assertAlmostEqual(model.b_mag[0, 0].item(), 0.0)
