import unittest

import torch

from sgnsae.experiments.directional_deviation.pooling import pool_exceedances


class TestDirectionalDeviationPooling(unittest.TestCase):
    def test_pool_exceedances_raises_for_non_2d_input(self) -> None:
        # given
        exceedances = torch.zeros(3, dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = pool_exceedances(exceedances=exceedances, pooling="sum")

    def test_pool_exceedances_returns_zeros_for_empty_latent_dimension(self) -> None:
        # given
        exceedances = torch.empty((3, 0), dtype=torch.float32)

        # when
        pooled = pool_exceedances(exceedances=exceedances, pooling="sum")

        # then
        self.assertTrue(torch.allclose(pooled, torch.zeros(3, dtype=torch.float32)))

    def test_pool_exceedances_topk_raises_when_topk_k_missing(self) -> None:
        # given
        exceedances = torch.ones((2, 3), dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = pool_exceedances(exceedances=exceedances, pooling="topk")

    def test_pool_exceedances_topk_raises_when_topk_k_non_positive(self) -> None:
        # given
        exceedances = torch.ones((2, 3), dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = pool_exceedances(exceedances=exceedances, pooling="topk", topk_k=0)

    def test_pool_exceedances_raises_for_unknown_pooling_mode(self) -> None:
        # given
        exceedances = torch.ones((2, 3), dtype=torch.float32)

        # when & then
        with self.assertRaises(ValueError):
            _ = pool_exceedances(
                exceedances=exceedances,
                pooling="unsupported",  # type: ignore[arg-type]
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
