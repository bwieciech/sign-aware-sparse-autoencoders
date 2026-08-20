import base64
import io
import unittest

from PIL import Image

from sgnsae.experiments.llm_benchmark.latent_examples import (
    compute_patch_crop_bounds,
    crop_patch_image,
    encode_png_base64,
)


class TestLatentExamplesImageCropping(unittest.TestCase):
    def test_compute_patch_crop_bounds_returns_expected_region(self) -> None:
        # given
        patch_index = 5
        n_patches_w = 4
        patch_size = 16
        image_size = 64

        # when
        bounds = compute_patch_crop_bounds(
            patch_index=patch_index,
            n_patches_w=n_patches_w,
            patch_size=patch_size,
            image_size=image_size,
        )

        # then
        self.assertEqual(bounds, (16, 16, 32, 32))

    def test_crop_patch_image_and_png_base64_roundtrip_keep_patch_dimensions(
        self,
    ) -> None:
        # given
        image = Image.new(mode="RGB", size=(64, 64), color=(10, 20, 30))
        patch_index = 10
        n_patches_w = 4
        patch_size = 16
        image_size = 64

        # when
        patch = crop_patch_image(
            image=image,
            patch_index=patch_index,
            n_patches_w=n_patches_w,
            patch_size=patch_size,
            image_size=image_size,
        )
        encoded = encode_png_base64(patch)
        decoded_bytes = base64.b64decode(encoded.encode("ascii"))
        with Image.open(io.BytesIO(decoded_bytes)) as decoded_patch:
            decoded_size = decoded_patch.size

        # then
        self.assertEqual(patch.size, (16, 16))
        self.assertEqual(decoded_size, (16, 16))
