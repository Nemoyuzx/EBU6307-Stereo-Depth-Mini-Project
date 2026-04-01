from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TESTS_DIR = Path(__file__).resolve().parent
CODES_DIR = TESTS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from o2 import _mutual_ratio_matches, create_sift_detector


def _synthetic_feature_image() -> np.ndarray:
    image = np.zeros((96, 96), dtype=np.uint8)

    image[8:28, 8:28] = 230
    image[13:23, 13:23] = 20

    image[36:64, 12:24] = 180
    image[44:56, 4:32] = 180

    image[20:48, 60:88] = 210
    image[26:42, 66:82] = 40

    image[68:88, 52:84] = 160
    for offset in range(18):
        image[70 + offset, 54 + offset] = 255
        image[70 + offset, 84 - offset] = 255

    return image


def _shift_image(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    shifted = np.zeros_like(image)
    source_y = slice(0, image.shape[0] - shift_y)
    source_x = slice(0, image.shape[1] - shift_x)
    target_y = slice(shift_y, image.shape[0])
    target_x = slice(shift_x, image.shape[1])
    shifted[target_y, target_x] = image[source_y, source_x]
    return shifted


class ManualSiftDetectorTests(unittest.TestCase):
    def test_detect_and_compute_returns_keypoints_and_128d_descriptors(self) -> None:
        detector = create_sift_detector(max_features=64, contrast_threshold=0.04)

        keypoints, descriptors = detector.detectAndCompute(_synthetic_feature_image(), None)

        self.assertGreater(len(keypoints), 0)
        self.assertIsNotNone(descriptors)
        assert descriptors is not None
        self.assertEqual(descriptors.shape[0], len(keypoints))
        self.assertEqual(descriptors.shape[1], 128)
        self.assertLessEqual(len(keypoints), 64)

    def test_shifted_view_produces_mutual_ratio_matches(self) -> None:
        detector = create_sift_detector(max_features=96, contrast_threshold=0.04)
        original = _synthetic_feature_image()
        shifted = _shift_image(original, shift_y=5, shift_x=7)

        left_keypoints, left_descriptors = detector.detectAndCompute(original, None)
        right_keypoints, right_descriptors = detector.detectAndCompute(shifted, None)

        self.assertGreater(len(left_keypoints), 0)
        self.assertGreater(len(right_keypoints), 0)
        self.assertIsNotNone(left_descriptors)
        self.assertIsNotNone(right_descriptors)
        assert left_descriptors is not None
        assert right_descriptors is not None

        raw_matches, ratio_matches, mutual_matches = _mutual_ratio_matches(left_descriptors, right_descriptors, 0.85)

        self.assertGreater(raw_matches, 0)
        self.assertGreater(len(ratio_matches), 0)
        self.assertGreater(len(mutual_matches), 0)


if __name__ == "__main__":
    unittest.main()