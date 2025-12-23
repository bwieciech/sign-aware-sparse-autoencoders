import math
import unittest

import torch

from sgnsae.experiments.polarity_dial.data import generate_polarity_dial_dataset
from sgnsae.experiments.polarity_dial.config import PolarityDialDataConfig


class TestPolarityDialDataGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"

    def test_generate_dataset_shapes_and_unit_norms(self) -> None:
        # given
        config = PolarityDialDataConfig(
            seed=123,
            num_train=32,
            num_val=16,
            num_test=8,
            input_dim=6,
            num_features=4,
            activation_prob=0.4,
            positive_sign_prob=0.6,
            lognormal_mu=0.0,
            lognormal_sigma=0.5,
            exponential_lambda=1.5,
            noise_std=0.1,
        )

        # when
        data = generate_polarity_dial_dataset(config, self.device)

        # then
        self.assertEqual(data["x_train"].shape, (32, 6))
        self.assertEqual(data["x_val"].shape, (16, 6))
        self.assertEqual(data["x_test"].shape, (8, 6))
        self.assertEqual(data["feature_directions"].shape, (4, 6))
        norms = (
            data["feature_directions"].norm(dim=1)
            - torch.ones(config.num_features, device=self.device)
        ).abs()
        self.assertTrue(bool((norms < 1e-4).all()))

    def test_generate_dataset_reproducible_with_seed(self) -> None:
        # given
        config = PolarityDialDataConfig(
            seed=7,
            num_train=10,
            num_val=5,
            num_test=5,
            input_dim=5,
            num_features=3,
            activation_prob=0.5,
            positive_sign_prob=0.7,
            lognormal_mu=0.0,
            lognormal_sigma=0.5,
            exponential_lambda=1.5,
            noise_std=0.1,
        )

        # when
        data_first = generate_polarity_dial_dataset(config, self.device)
        data_second = generate_polarity_dial_dataset(config, self.device)

        # then
        for key in data_first:
            self.assertTrue(torch.equal(data_first[key], data_second[key]))

    def test_coefficient_sign_distribution_matches_configuration(self) -> None:
        # given
        config = PolarityDialDataConfig(
            seed=42,
            num_train=512,
            num_val=1,
            num_test=1,
            input_dim=4,
            num_features=3,
            activation_prob=0.3,
            positive_sign_prob=0.75,
            lognormal_mu=0.0,
            lognormal_sigma=0.2,
            exponential_lambda=1.5,
            noise_std=0.05,
        )

        # when
        data = generate_polarity_dial_dataset(config, self.device)
        coefficients = data["coefficients_train"].view(-1)
        active_mask = coefficients != 0
        positive_mask = coefficients > 0
        negative_mask = coefficients < 0

        # then
        active_fraction = active_mask.float().mean().item()
        positive_fraction = positive_mask.float().mean().item()
        expected_active = config.activation_prob
        expected_positive = config.activation_prob * config.positive_sign_prob
        self.assertTrue(math.isclose(active_fraction, expected_active, rel_tol=0.2))
        self.assertTrue(math.isclose(positive_fraction, expected_positive, rel_tol=0.2))
        self.assertGreater(coefficients[positive_mask].mean().item(), 0.0)
        self.assertGreater((-coefficients[negative_mask]).mean().item(), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
