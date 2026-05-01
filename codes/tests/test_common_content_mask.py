from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
CODES_DIR = TESTS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from common import content_bbox_from_gray, content_mask_from_gray


class ContentMaskTests(unittest.TestCase):
    def test_trims_only_outer_black_border(self) -> None:
        image = np.zeros((8, 10), dtype=np.uint8)
        image[2:7, 3:9] = 24
        image[4, 5] = 0

        self.assertEqual(content_bbox_from_gray(image), (2, 7, 3, 9))
        mask = content_mask_from_gray(image)

        self.assertFalse(bool(mask[1, 4]))
        self.assertFalse(bool(mask[4, 2]))
        self.assertTrue(bool(mask[4, 5]))
        self.assertEqual(int(mask.sum()), 5 * 6)

    def test_all_black_image_keeps_full_frame(self) -> None:
        image = np.zeros((3, 4), dtype=np.uint8)

        self.assertEqual(content_bbox_from_gray(image), (0, 3, 0, 4))
        self.assertTrue(bool(content_mask_from_gray(image).all()))


if __name__ == "__main__":
    unittest.main()
