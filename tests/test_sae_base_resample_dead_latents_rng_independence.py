import unittest

import torch

from sgnsae.models import SAE


class TestSAEBaseResampleDeadLatentsRngIndependence(unittest.TestCase):
    def test_instance_0_resample_is_independent_of_other_instances(self) -> None:
        # given
        base_seed = 123
        d_in = 5
        d_hidden = 3
        threshold = 0

        model_single = SAE(
            n_instances=1,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=threshold,
            sign_aware=False,
            device="cpu",
        )
        model_multi = SAE(
            n_instances=2,
            d_in=d_in,
            d_hidden=d_hidden,
            resample_latents_threshold=threshold,
            sign_aware=False,
            device="cpu",
        )

        generator_single = torch.Generator(device="cpu").manual_seed(base_seed)
        generators_multi = [
            torch.Generator(device="cpu").manual_seed(base_seed + 0),
            torch.Generator(device="cpu").manual_seed(base_seed + 1),
        ]

        inactive_first_single = torch.tensor([[1, 0, 0]], dtype=torch.long)
        inactive_second_single = torch.tensor([[0, 1, 0]], dtype=torch.long)

        inactive_first_multi = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.long)
        inactive_between_multi = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.long)
        inactive_second_multi = torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.long)

        # when
        _ = model_single.resample_dead_latents(
            inactive_first_single, generators=[generator_single]
        )
        _ = model_multi.resample_dead_latents(
            inactive_first_multi, generators=generators_multi
        )
        _ = model_multi.resample_dead_latents(
            inactive_between_multi, generators=generators_multi
        )

        _ = model_single.resample_dead_latents(
            inactive_second_single, generators=[generator_single]
        )
        _ = model_multi.resample_dead_latents(
            inactive_second_multi, generators=generators_multi
        )

        # then
        torch.testing.assert_close(
            model_single.W_dec[0, :, 1],
            model_multi.W_dec[0, :, 1],
            rtol=0.0,
            atol=0.0,
        )

    def test_resample_raises_when_generator_count_mismatches_instances(self) -> None:
        # given
        model = SAE(
            n_instances=2,
            d_in=4,
            d_hidden=2,
            resample_latents_threshold=0,
            sign_aware=False,
            device="cpu",
        )
        inactive = torch.tensor([[1, 0], [0, 0]], dtype=torch.long)
        generators = [torch.Generator(device="cpu").manual_seed(0)]

        # when & then
        with self.assertRaises(ValueError):
            _ = model.resample_dead_latents(inactive, generators=generators)
