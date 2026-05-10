from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TESTS_DIR = Path(__file__).resolve().parent
CODES_DIR = TESTS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from config import O3Config
from o3 import (
    add_sift_boundary_support_points,
    filter_sift_seed_outliers,
    inpaint_triangulated_disparity_holes,
    triangulate_sift_seed_disparity,
)


def _test_o3_config() -> O3Config:
    return O3Config(
        pipeline_dir=Path("results/O3a_disparity"),
        disparity_dir=Path("results/O3b_disparity"),
        analysis_dir=Path("results/O3b_disparity"),
        metrics_file=Path("results/O3c_disparity/metrics.csv"),
        max_features=800,
        contrast_threshold=0.04,
        ratio_test=0.8,
        max_draw_matches=80,
        num_disparities=96,
        max_vertical_offset=2.0,
        block_size=5,
        uniqueness_ratio=0,
        speckle_window_size=0,
        speckle_range=8,
        disp12_max_diff=1,
        median_filter_size=3,
        census_window_size=5,
        census_weight=3.0,
        gradient_weight=1.0,
        consistency_threshold=1.0,
        fill_invalid_passes=0,
    )


class O3SiftInterpolationTests(unittest.TestCase):
    def test_filter_sift_seed_outliers_removes_gross_local_mismatch(self) -> None:
        seeds = np.zeros((32, 32), dtype=np.float32)
        seed_mask = np.zeros_like(seeds, dtype=bool)
        for row_index, column_index, disparity in (
            (10, 10, 12.0),
            (11, 12, 12.5),
            (12, 9, 11.5),
            (13, 11, 12.25),
            (12, 13, 70.0),
        ):
            seeds[row_index, column_index] = disparity
            seed_mask[row_index, column_index] = True

        filtered, filtered_mask = filter_sift_seed_outliers(seeds, seed_mask)

        self.assertEqual(int(filtered_mask.sum()), 4)
        self.assertEqual(float(filtered[12, 13]), 0.0)

    def test_triangulated_sift_seed_disparity_forms_continuous_surface(self) -> None:
        seeds = np.zeros((48, 48), dtype=np.float32)
        seed_mask = np.zeros_like(seeds, dtype=bool)
        for row_index, column_index, disparity in (
            (6, 6, 8.0),
            (6, 40, 13.0),
            (40, 6, 16.0),
            (40, 40, 21.0),
        ):
            seeds[row_index, column_index] = disparity
            seed_mask[row_index, column_index] = True
        guide = np.full_like(seeds, 128, dtype=np.uint8)

        disparity = triangulate_sift_seed_disparity(seeds, seed_mask, guide, _test_o3_config())

        valid_values = disparity[disparity > 0]
        self.assertGreater(valid_values.size, 500)
        self.assertGreater(np.unique(np.rint(valid_values)).size, 4)
        self.assertGreater(float(disparity[24, 24]), 12.0)
        self.assertLess(float(disparity[24, 24]), 18.0)

    def test_boundary_support_points_are_derived_from_sift_seed_values(self) -> None:
        points = np.array([[20.0, 20.0], [36.0, 20.0], [20.0, 36.0]], dtype=np.float32)
        values = np.array([7.0, 11.0, 15.0], dtype=np.float32)

        augmented_points, augmented_values = add_sift_boundary_support_points(points, values, (64, 64))

        self.assertGreater(augmented_points.shape[0], points.shape[0])
        self.assertTrue(set(np.unique(augmented_values)).issubset({7.0, 11.0, 15.0}))
        self.assertTrue(np.any((augmented_points[:, 0] == 0.0) | (augmented_points[:, 1] == 0.0)))

    def test_inpaint_triangulated_disparity_holes_fills_from_surface_boundaries(self) -> None:
        disparity = np.full((32, 32), 10.0, dtype=np.float32)
        disparity[:, 16:] = 20.0
        disparity[10:22, 10:22] = 0.0
        guide = np.full_like(disparity, 128, dtype=np.uint8)

        filled = inpaint_triangulated_disparity_holes(disparity, guide, max_distance=24.0, iterations=12)

        self.assertGreater(int((filled > 0).sum()), int((disparity > 0).sum()))
        self.assertGreater(float(filled[16, 16]), 10.0)
        self.assertLess(float(filled[16, 16]), 20.0)


if __name__ == "__main__":
    unittest.main()