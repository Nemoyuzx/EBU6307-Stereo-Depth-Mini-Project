from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
CODES_DIR = TESTS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from o4 import build_o4_content_mask
from o4_dinov2 import _build_nonblack_content_mask


class O4ContentMaskTests(unittest.TestCase):
    def test_o4_mask_removes_black_corners_inside_bbox(self) -> None:
        image = np.zeros((6, 6), dtype=np.uint8)
        image[1:5, 1:5] = 20
        image[1, 1] = 0
        image[1, 4] = 0
        image[4, 1] = 0
        image[4, 4] = 0

        mask = build_o4_content_mask(image)

        self.assertFalse(bool(mask[1, 1]))
        self.assertFalse(bool(mask[1, 4]))
        self.assertFalse(bool(mask[4, 1]))
        self.assertFalse(bool(mask[4, 4]))
        self.assertTrue(bool(mask[2, 2]))
        self.assertEqual(int(mask.sum()), 12)

    def test_dinov2_mask_matches_o4_mask(self) -> None:
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1:6, 1:6] = 25
        image[1, 3] = 0
        image[5, 4] = 0

        self.assertTrue(np.array_equal(build_o4_content_mask(image), _build_nonblack_content_mask(image)))


if __name__ == "__main__":
    unittest.main()
