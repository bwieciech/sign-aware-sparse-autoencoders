import unittest
from unittest.mock import patch

from sgnsae.experiments.polarity_dial.data import set_random_seed


class TestPolarityDialRandomSeed(unittest.TestCase):
    def test_set_random_seed_seeds_cuda_when_available(self) -> None:
        # given
        seed = 123

        # when
        with (
            patch("sgnsae.experiments.polarity_dial.data.torch.manual_seed"),
            patch(
                "sgnsae.experiments.polarity_dial.data.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "sgnsae.experiments.polarity_dial.data.torch.cuda.manual_seed_all"
            ) as manual_seed_all,
        ):
            set_random_seed(seed)

        # then
        manual_seed_all.assert_called_once_with(seed)

    def test_set_random_seed_does_not_seed_cuda_when_unavailable(self) -> None:
        # given
        seed = 456

        # when
        with (
            patch("sgnsae.experiments.polarity_dial.data.torch.manual_seed"),
            patch(
                "sgnsae.experiments.polarity_dial.data.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "sgnsae.experiments.polarity_dial.data.torch.cuda.manual_seed_all"
            ) as manual_seed_all,
        ):
            set_random_seed(seed)

        # then
        manual_seed_all.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
