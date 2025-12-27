import unittest
import torch

from sgnsae.experiments.polarity_dial.evaluation import align_decoder_and_features


class DummyDecoder(torch.nn.Module):
    def __init__(self, W_dec: torch.Tensor) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(W_dec)


class TestAlignmentSignInvariant(unittest.TestCase):
    def test_alignment_matches_antipodal_decoder(self) -> None:
        # given
        feature_directions = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        sqrt_099 = float(torch.sqrt(torch.tensor(0.99)))
        W_dec = torch.tensor(
            [[[-1.0, 0.1], [0.0, sqrt_099]]],
            dtype=torch.float32,
        )  # (n_instances=1, d_in=2, d_hidden=2)
        model = DummyDecoder(W_dec)

        # when
        alignment, _ = align_decoder_and_features(model, feature_directions)
        feature_to_latent = {
            p["feature_index"]: p["latent_index"] for p in alignment["matched_pairs"]
        }

        # then
        self.assertEqual(feature_to_latent[0], 0)
        cosine_for_feature0 = next(
            p["cosine_similarity"]
            for p in alignment["matched_pairs"]
            if p["feature_index"] == 0
        )
        self.assertLess(cosine_for_feature0, 0.0)
        self.assertAlmostEqual(abs(cosine_for_feature0), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
