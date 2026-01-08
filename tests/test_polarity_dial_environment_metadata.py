import unittest
from unittest.mock import patch

from sgnsae.experiments.polarity_dial import environment


class TestPolarityDialEnvironmentMetadata(unittest.TestCase):
    def test_package_version_returns_none_for_missing_distribution(self) -> None:
        # given
        def raise_package_not_found(_: str) -> str:
            raise environment.PackageNotFoundError

        # when
        with patch.object(environment, "version", side_effect=raise_package_not_found):
            result = environment._package_version("not_a_real_package")

        # then
        self.assertIsNone(result)

    def test_collect_environment_metadata_sets_gpu_name_when_cuda_available(
        self,
    ) -> None:
        # given
        with (
            patch.object(environment.torch.cuda, "is_available", return_value=True),
            patch.object(
                environment.torch.cuda, "get_device_name", return_value="Fake GPU"
            ),
        ):
            # when
            metadata = environment.collect_environment_metadata()

        # then
        self.assertTrue(metadata["torch"]["cuda_available"])
        self.assertEqual(metadata["torch"]["gpu_name"], "Fake GPU")

    def test_collect_environment_metadata_sets_gpu_name_to_none_on_failure(
        self,
    ) -> None:
        # given
        def raise_device_error(_: int) -> str:
            raise RuntimeError("no device")

        with (
            patch.object(environment.torch.cuda, "is_available", return_value=True),
            patch.object(
                environment.torch.cuda,
                "get_device_name",
                side_effect=raise_device_error,
            ),
        ):
            # when
            metadata = environment.collect_environment_metadata()

        # then
        self.assertTrue(metadata["torch"]["cuda_available"])
        self.assertIsNone(metadata["torch"]["gpu_name"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
