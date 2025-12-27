import unittest

import torch

from sgnsae.models.sae import SAE
from sgnsae.models.gated_sae import GatedSAE


class TestSharedWeightsAndBiJumpReLU(unittest.TestCase):
    def test_no_W_enc_parameter_exists(self) -> None:
        # given
        sae = SAE(
            n_instances=1,
            d_in=4,
            d_hidden=3,
            sign_aware=False,
            resample_latents_threshold=1e9,
            device="cpu",
        )
        gated = GatedSAE(
            n_instances=1,
            d_in=4,
            d_hidden=3,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            resample_latents_threshold=1e9,
            device="cpu",
        )

        # when
        state_dict = gated.state_dict()

        # then
        self.assertFalse(hasattr(sae, "W_enc"))
        self.assertFalse(hasattr(gated, "W_enc"))
        self.assertTrue(all("W_enc" not in k for k in state_dict.keys()))

    def test_sae_encoding_uses_decoder_projection(self) -> None:
        # given
        model = SAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            sign_aware=False,
            resample_latents_threshold=1e9,
            device="cpu",
        )
        model.W_dec.data[0] = torch.eye(2)
        model.b_dec.data.zero_()
        model.b_enc.data.zero_()
        x = torch.tensor([[1.0, -2.0]])

        # when
        output = model(x)
        latents = output["latents"][0]

        # then
        expected = torch.tensor([1.0, 0.0])
        self.assertTrue(torch.allclose(latents[0], expected))

    def test_bijumprelu_option2_negative_branch_bias(self) -> None:
        # given
        model = GatedSAE(
            n_instances=1,
            d_in=2,
            d_hidden=2,
            sign_aware=True,
            reset_activation_offsets_on_resample=True,
            single_activation_offset=True,
            resample_latents_threshold=1e9,
            device="cpu",
        )
        model.W_dec.data[0] = torch.eye(2)
        model.b_dec.data.zero_()
        model.b_enc.data.zero_()
        model.log_alpha_gate.data.zero_()
        model.r_mag.data.zero_()
        model.r_mag_neg.data.zero_()
        model.b_mag.data[0] = torch.tensor([0.0, 0.2])
        model.act_offset.data.zero_()
        x = torch.tensor([[0.0, -3.0]])

        # when
        output = model(x)
        latents = output["latents"][0, 0]

        # then
        self.assertAlmostEqual(latents[1].item(), -3.2, places=6)


if __name__ == "__main__":
    unittest.main()
