import unittest
import torch

from sgnsae.models import GatedSAE


class TestStopGradGateLosses(unittest.TestCase):
    def test_sparsity_loss_does_not_backprop_to_decoder(self) -> None:
        # given
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
            model.b_dec.data.zero_()
            model.b_enc.data.zero_()
            model.log_alpha_gate.data.zero_()
            model.act_offset.data.zero_()

        x = torch.tensor([[1.0, -1.0]])

        # when
        out = model(x)

        # Build sparsity loss directly from pi_gate and offsets.
        pi = out["pi_gate"]
        delta = torch.relu(model.act_offset).unsqueeze(1)
        sparsity_source = torch.relu(pi - delta) + torch.relu(-pi - delta)
        loss = sparsity_source.sum()

        model.zero_grad(set_to_none=True)
        loss.backward()

        # then
        # Because pi_gate uses t.detach(), W_dec should get no gradient from this loss.
        decoder_grad = model.W_dec.grad
        if decoder_grad is None:
            self.assertIsNone(decoder_grad)
        else:
            self.assertAlmostEqual(
                float(decoder_grad.abs().sum().item()), 0.0, places=12
            )


if __name__ == "__main__":
    unittest.main()
