from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TESTS_DIR = Path(__file__).resolve().parent
CODES_DIR = TESTS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from PIL import Image

from o1 import (
    _fill_projection_holes,
    _project_left_to_synthetic_view,
    create_o1_pipeline_image,
    scale_reference_disparity,
    synthesize_depth_aware_stereo,
)


def _toy_rgb_image(height: int = 4, width: int = 6) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        image[:, x, 0] = x * 30
        image[:, x, 1] = 255 - x * 20
        image[:, x, 2] = 40 + x * 10
    return image


class O1SynthesisTests(unittest.TestCase):
    def test_scale_reference_disparity_keeps_real_range(self) -> None:
        reference = np.array(
            [
                [0.0, 2.0, 4.0, 8.0],
                [12.0, 24.0, 48.0, 96.0],
            ],
            dtype=np.float32,
        )

        scaled = scale_reference_disparity(reference, 8)

        self.assertEqual(scaled.shape, reference.shape)
        self.assertGreater(float(np.max(scaled)), 40.0)
        self.assertLessEqual(float(np.max(scaled)), 96.0)
        self.assertAlmostEqual(float(scaled[0, 0]), 0.0, places=4)
        self.assertGreater(float(scaled[1, 3]), float(scaled[0, 3]))

    def test_projection_overwrites_far_pixels_with_near_pixels(self) -> None:
        image = _toy_rgb_image(height=1, width=6)
        disparity = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]], dtype=np.float32)

        projected, missing = _project_left_to_synthetic_view(image, disparity, shift_pixels=8)

        np.testing.assert_array_equal(projected[0, 0], image[0, 0])
        np.testing.assert_array_equal(projected[0, 1], image[0, 2])
        np.testing.assert_array_equal(projected[0, 2], image[0, 4])
        np.testing.assert_array_equal(projected[0, 3], image[0, 5])
        self.assertTrue(bool(missing[0, 4]))
        self.assertTrue(bool(missing[0, 5]))

    def test_depth_aware_synthesis_returns_non_uniform_result(self) -> None:
        image = _toy_rgb_image(height=6, width=8)
        reference = np.tile(np.array([0.0, 1.0, 2.0, 4.0, 16.0, 16.0, 8.0, 2.0], dtype=np.float32), (6, 1))

        synthetic, disparity = synthesize_depth_aware_stereo(image, reference, 8)

        self.assertEqual(synthetic.shape, image.shape)
        self.assertEqual(disparity.shape, image.shape[:2])
        self.assertFalse(np.array_equal(synthetic, image))
        self.assertGreater(float(np.max(disparity)), 8.0)
        self.assertLessEqual(float(np.max(disparity)), 16.0)

    def test_fill_projection_holes_uses_neighbour_propagation(self) -> None:
        projected = np.array(
            [
                [[10, 20, 30], [0, 0, 0], [70, 80, 90]],
                [[15, 25, 35], [0, 0, 0], [75, 85, 95]],
                [[20, 30, 40], [80, 90, 100], [90, 100, 110]],
            ],
            dtype=np.uint8,
        )
        missing = np.array(
            [
                [False, True, False],
                [False, True, False],
                [False, False, False],
            ],
            dtype=bool,
        )

        filled = _fill_projection_holes(projected, missing)

        self.assertEqual(filled.shape, projected.shape)
        self.assertTrue(np.all(filled[missing] > 0))
        self.assertFalse(np.array_equal(filled[0, 1], projected[0, 1]))
        self.assertFalse(np.array_equal(filled[1, 1], projected[1, 1]))

    def test_create_o1_pipeline_image_writes_expected_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "syn_pipeline.jpg"

            create_o1_pipeline_image(output_path)

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (2240, 1536))
                self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
