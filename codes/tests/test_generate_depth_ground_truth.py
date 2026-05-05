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

from generate_depth_ground_truth import (
    MiddleburyCalibration,
    colorize_depth_preview,
    disparity_to_depth,
    parse_middlebury_calibration,
)


class GenerateDepthGroundTruthTests(unittest.TestCase):
    def test_parse_middlebury_calibration_reads_fx_baseline_and_doffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_path = Path(tmpdir) / "calib.txt"
            calib_path.write_text(
                "\n".join(
                    [
                        "cam0=[1733.74 0 792.27; 0 1733.74 541.89; 0 0 1]",
                        "baseline=536.62",
                        "doffs=12.5",
                        "width=1920",
                        "height=1080",
                    ]
                ),
                encoding="utf-8",
            )

            calibration = parse_middlebury_calibration(calib_path)

        self.assertAlmostEqual(calibration.fx, 1733.74)
        self.assertAlmostEqual(calibration.baseline, 536.62)
        self.assertAlmostEqual(calibration.doffs, 12.5)
        self.assertEqual(calibration.width, 1920)
        self.assertEqual(calibration.height, 1080)

    def test_disparity_to_depth_uses_middlebury_formula(self) -> None:
        disparity = np.array([[10.0, 0.0], [20.0, np.inf]], dtype=np.float32)
        calibration = MiddleburyCalibration(fx=100.0, baseline=50.0, doffs=5.0)

        depth = disparity_to_depth(disparity, calibration)

        self.assertAlmostEqual(float(depth[0, 0]), 100.0 * 50.0 / 15.0, places=4)
        self.assertEqual(float(depth[0, 1]), 0.0)
        self.assertAlmostEqual(float(depth[1, 0]), 100.0 * 50.0 / 25.0, places=4)
        self.assertEqual(float(depth[1, 1]), 0.0)

    def test_colorize_depth_preview_returns_rgb_and_masks_invalid_pixels(self) -> None:
        depth = np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32)
        valid = depth > 0

        preview = colorize_depth_preview(depth, valid)

        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertTrue(np.array_equal(preview[0, 0], np.zeros(3, dtype=np.uint8)))
        self.assertGreater(int(preview[1, 1].sum()), 0)
        self.assertGreater(int(preview[0, 1][0]), int(preview[0, 1][2]))
        self.assertGreater(int(preview[1, 1][2]), int(preview[1, 1][0]))


if __name__ == "__main__":
    unittest.main()
