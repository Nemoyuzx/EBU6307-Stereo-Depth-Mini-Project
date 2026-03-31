from __future__ import annotations

import csv
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    discover_scenes,
    evaluate_disparity,
    filter_scene_dirs,
    load_gray,
    normalize_for_preview,
    write_png,
    write_scene_text,
)
from .config import O3Config
from .o2 import _mutual_ratio_matches, _select_geometry_aware_matches, create_sift_detector
from .pfm import read_pfm, write_pfm


MetricValue = str | float | int
MetricRow = dict[str, MetricValue]


def read_metrics(metrics_file: Path) -> list[MetricRow]:
    """读取 O3 历史指标。"""
    if not metrics_file.exists():
        return []

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "scene": row.get("scene", ""),
                "valid_disparity_pixels": row.get("valid_disparity_pixels", ""),
                "valid_ground_truth_pixels": row.get("valid_ground_truth_pixels", ""),
                "mae": row.get("mae", ""),
                "rmse": row.get("rmse", ""),
                "bad_1px": row.get("bad_1px", ""),
            }
            for row in reader
            if row.get("scene")
        ]


def write_metrics(metrics_file: Path, rows: list[MetricRow]) -> None:
    """按场景合并并回写 O3 指标。"""
    existing_rows = read_metrics(metrics_file)
    rows_by_scene = {str(row["scene"]): row for row in rows}
    merged_rows: list[MetricRow] = []

    for row in existing_rows:
        scene_name = str(row["scene"])
        replacement = rows_by_scene.pop(scene_name, None)
        merged_rows.append(replacement if replacement is not None else row)

    merged_rows.extend(rows_by_scene.values())

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "valid_disparity_pixels",
                "valid_ground_truth_pixels",
                "mae",
                "rmse",
                "bad_1px",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)


def median_filter_2d(image: Any, kernel_size: int) -> Any:
    """对二维数组做中值滤波，抑制离群视差。"""

    if kernel_size <= 1:
        return image

    radius = kernel_size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    return np.median(windows, axis=(-2, -1)).astype(image.dtype, copy=False)


def box_filter_sum(image: Any, radius: int) -> Any:
    """用积分图思想求窗口求和，加速代价聚合。"""

    if radius <= 0:
        return np.asarray(image, dtype=np.float32)

    padded = np.pad(np.asarray(image, dtype=np.float32), ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    kernel_size = radius * 2 + 1
    return (
        integral[kernel_size:, kernel_size:]
        - integral[:-kernel_size, kernel_size:]
        - integral[kernel_size:, :-kernel_size]
        + integral[:-kernel_size, :-kernel_size]
    )


def bit_count_array(values: Any) -> Any:
    """统计 census 编码异或后的 bit 数量。"""

    array = np.asarray(values)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(array).astype(np.float32)

    expanded = np.unpackbits(array.view(np.uint8), axis=-1)
    return expanded.sum(axis=-1, dtype=np.int16).astype(np.float32)


def compute_horizontal_gradient(image: Any) -> Any:
    """计算水平方向梯度，辅助匹配代价估计。"""

    source = np.asarray(image, dtype=np.float32)
    gradient = np.zeros_like(source, dtype=np.float32)
    if source.shape[1] > 1:
        gradient[:, 1:-1] = 0.5 * (source[:, 2:] - source[:, :-2])
        gradient[:, 0] = source[:, 1] - source[:, 0]
        gradient[:, -1] = source[:, -1] - source[:, -2]
    return gradient


def compute_census_transform(image: Any, window_size: int) -> Any:
    """计算局部 census 描述子，用于增强亮度变化鲁棒性。"""

    source = np.asarray(image, dtype=np.float32)
    height, width = source.shape
    radius = max(0, window_size // 2)
    census = np.zeros((height, width), dtype=np.uint64)
    if radius <= 0 or height <= radius * 2 or width <= radius * 2:
        return census

    center = source[radius : height - radius, radius : width - radius]
    encoded = np.zeros_like(center, dtype=np.uint64)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            if offset_y == 0 and offset_x == 0:
                continue
            neighbor = source[
                radius + offset_y : height - radius + offset_y,
                radius + offset_x : width - radius + offset_x,
            ]
            encoded = (encoded << 1) | (neighbor < center).astype(np.uint64)
    census[radius : height - radius, radius : width - radius] = encoded
    return census


def compute_matching_cost(
    reference_gray: Any,
    target_gray: Any,
    reference_gradient: Any,
    target_gradient: Any,
    reference_census: Any,
    target_census: Any,
    config: O3Config,
    disparity: int,
    target_direction: str = "negative",
) -> Any:

    reference = np.asarray(reference_gray, dtype=np.float32)
    target = np.asarray(target_gray, dtype=np.float32)
    reference_gradient = np.asarray(reference_gradient, dtype=np.float32)
    target_gradient = np.asarray(target_gradient, dtype=np.float32)
    reference_census = np.asarray(reference_census, dtype=np.uint64)
    target_census = np.asarray(target_census, dtype=np.uint64)

    height, width = reference.shape
    shifted_gray = np.zeros_like(target)
    shifted_gradient = np.zeros_like(target_gradient)
    shifted_census = np.zeros_like(target_census)
    if disparity == 0:
        shifted_gray[:] = target
        shifted_gradient[:] = target_gradient
        shifted_census[:] = target_census
    elif target_direction == "positive":
        shifted_gray[:, : width - disparity] = target[:, disparity:]
        shifted_gradient[:, : width - disparity] = target_gradient[:, disparity:]
        shifted_census[:, : width - disparity] = target_census[:, disparity:]
    else:
        shifted_gray[:, disparity:] = target[:, : width - disparity]
        shifted_gradient[:, disparity:] = target_gradient[:, : width - disparity]
        shifted_census[:, disparity:] = target_census[:, : width - disparity]

    support_radius = config.block_size // 2
    intensity_cost = box_filter_sum(np.abs(reference - shifted_gray), support_radius)
    gradient_cost = box_filter_sum(np.abs(reference_gradient - shifted_gradient), support_radius)
    census_cost = box_filter_sum(bit_count_array(reference_census ^ shifted_census), support_radius)
    cost = intensity_cost + (config.gradient_weight * gradient_cost) + (config.census_weight * census_cost)
    if disparity > 0:
        if target_direction == "positive":
            cost[:, width - disparity :] = np.inf
        else:
            cost[:, :disparity] = np.inf
    return cost.astype(np.float32)


def compute_matching_minima(
    reference_gray: Any,
    target_gray: Any,
    reference_gradient: Any,
    target_gradient: Any,
    reference_census: Any,
    target_census: Any,
    config: O3Config,
    target_direction: str = "negative",
) -> tuple[Any, Any, Any]:

    reference = np.asarray(reference_gray, dtype=np.float32)
    target = np.asarray(target_gray, dtype=np.float32)
    reference_gradient = np.asarray(reference_gradient, dtype=np.float32)
    target_gradient = np.asarray(target_gradient, dtype=np.float32)
    reference_census = np.asarray(reference_census, dtype=np.uint64)
    target_census = np.asarray(target_census, dtype=np.uint64)
    height, width = reference.shape
    if width <= 1:
        empty = np.zeros((height, width), dtype=np.float32)
        return empty, empty, empty

    max_disparity = min(max(1, config.num_disparities), width - 1)
    best_cost = np.full((height, width), np.inf, dtype=np.float32)
    second_cost = np.full((height, width), np.inf, dtype=np.float32)
    best_disparity = np.zeros((height, width), dtype=np.float32)

    for disparity in range(max_disparity):
        cost = compute_matching_cost(
            reference,
            target,
            reference_gradient,
            target_gradient,
            reference_census,
            target_census,
            config,
            disparity,
            target_direction=target_direction,
        )
        replace_mask = cost < best_cost
        second_cost = np.where(replace_mask, best_cost, np.minimum(second_cost, cost))
        best_cost = np.where(replace_mask, cost, best_cost)
        best_disparity = np.where(replace_mask, float(disparity), best_disparity)

    return best_disparity.astype(np.float32), best_cost.astype(np.float32), second_cost.astype(np.float32)


def refine_subpixel_disparity(
    reference_gray: Any,
    target_gray: Any,
    reference_gradient: Any,
    target_gradient: Any,
    reference_census: Any,
    target_census: Any,
    integer_disparity: Any,
    config: O3Config,
    target_direction: str = "negative",
) -> Any:

    disparity = np.asarray(integer_disparity, dtype=np.float32)
    height, width = disparity.shape
    max_disparity = min(max(1, config.num_disparities), width - 1)
    lower_cost = np.full((height, width), np.inf, dtype=np.float32)
    center_cost = np.full((height, width), np.inf, dtype=np.float32)
    upper_cost = np.full((height, width), np.inf, dtype=np.float32)
    disparity_index = np.rint(disparity).astype(np.int32)

    for current_disparity in range(max_disparity):
        current_cost = compute_matching_cost(
            reference_gray,
            target_gray,
            reference_gradient,
            target_gradient,
            reference_census,
            target_census,
            config,
            current_disparity,
            target_direction=target_direction,
        )
        center_mask = disparity_index == current_disparity
        lower_mask = disparity_index == (current_disparity + 1)
        upper_mask = disparity_index == (current_disparity - 1)
        center_cost = np.where(center_mask, current_cost, center_cost)
        lower_cost = np.where(lower_mask, current_cost, lower_cost)
        upper_cost = np.where(upper_mask, current_cost, upper_cost)

    refined = disparity.copy()
    valid = (
        (disparity_index > 0)
        & (disparity_index < (max_disparity - 1))
        & np.isfinite(lower_cost)
        & np.isfinite(center_cost)
        & np.isfinite(upper_cost)
    )
    denominator = lower_cost + upper_cost - (2.0 * center_cost)
    stable = valid & (np.abs(denominator) > 1e-3)
    offset = np.zeros_like(refined, dtype=np.float32)
    offset[stable] = (lower_cost[stable] - upper_cost[stable]) / (2.0 * denominator[stable])
    offset = np.clip(offset, -1.0, 1.0)
    refined[stable] = refined[stable] + offset[stable]
    return refined.astype(np.float32)


def left_right_consistency_mask(left_disparity: Any, right_disparity: Any, threshold: float) -> Any:
    """执行左右一致性检查，过滤明显错误的匹配。"""

    left = np.asarray(left_disparity, dtype=np.float32)
    right = np.asarray(right_disparity, dtype=np.float32)
    height, width = left.shape
    columns = np.arange(width, dtype=np.int32)[None, :]
    matched_columns = columns - np.rint(left).astype(np.int32)
    valid = (left > 0) & (matched_columns >= 0) & (matched_columns < width)
    row_indices = np.broadcast_to(np.arange(height, dtype=np.int32)[:, None], left.shape)
    sampled = np.zeros_like(left, dtype=np.float32)
    sampled[valid] = right[row_indices[valid], matched_columns[valid]]
    return valid & (np.abs(left - sampled) <= float(threshold))


def fill_invalid_disparity(disparity: Any, passes: int) -> Any:
    """对无效视差做行列方向的邻域传播填补。"""

    filled = np.asarray(disparity, dtype=np.float32).copy()
    if passes <= 0:
        return filled

    height, width = filled.shape
    for _ in range(passes):
        updated = filled.copy()

        for row_index in range(height):
            row = updated[row_index]
            left_values = np.zeros(width, dtype=np.float32)
            right_values = np.zeros(width, dtype=np.float32)
            has_left = np.zeros(width, dtype=bool)
            has_right = np.zeros(width, dtype=bool)

            current = 0.0
            for column_index in range(width):
                if row[column_index] > 0:
                    current = row[column_index]
                    has_left[column_index] = True
                    left_values[column_index] = current
                elif current > 0:
                    has_left[column_index] = True
                    left_values[column_index] = current

            current = 0.0
            for column_index in range(width - 1, -1, -1):
                if row[column_index] > 0:
                    current = row[column_index]
                    has_right[column_index] = True
                    right_values[column_index] = current
                elif current > 0:
                    has_right[column_index] = True
                    right_values[column_index] = current

            fill_mask = (row <= 0) & has_left & has_right
            row[fill_mask] = np.minimum(left_values[fill_mask], right_values[fill_mask])

        updated = updated.T
        for row_index in range(width):
            row = updated[row_index]
            left_values = np.zeros(height, dtype=np.float32)
            right_values = np.zeros(height, dtype=np.float32)
            has_left = np.zeros(height, dtype=bool)
            has_right = np.zeros(height, dtype=bool)

            current = 0.0
            for column_index in range(height):
                if row[column_index] > 0:
                    current = row[column_index]
                    has_left[column_index] = True
                    left_values[column_index] = current
                elif current > 0:
                    has_left[column_index] = True
                    left_values[column_index] = current

            current = 0.0
            for column_index in range(height - 1, -1, -1):
                if row[column_index] > 0:
                    current = row[column_index]
                    has_right[column_index] = True
                    right_values[column_index] = current
                elif current > 0:
                    has_right[column_index] = True
                    right_values[column_index] = current

            fill_mask = (row <= 0) & has_left & has_right
            row[fill_mask] = np.minimum(left_values[fill_mask], right_values[fill_mask])
        filled = updated.T

    return filled.astype(np.float32)


def filter_speckles(disparity: Any, window_size: int, disparity_range: int) -> Any:
    """移除过小且离散的 speckle 连通域。"""


    filtered = np.asarray(disparity, dtype=np.float32).copy()
    if window_size <= 0:
        return filtered

    height, width = filtered.shape
    visited = np.zeros((height, width), dtype=bool)
    disparity_threshold = max(0.0, float(disparity_range))
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for row_index in range(height):
        for column_index in range(width):
            if visited[row_index, column_index] or filtered[row_index, column_index] <= 0:
                continue

            component: list[tuple[int, int]] = []
            seed_disparity = filtered[row_index, column_index]
            queue = deque([(row_index, column_index)])
            visited[row_index, column_index] = True

            while queue:
                current_row, current_column = queue.popleft()
                component.append((current_row, current_column))
                current_disparity = filtered[current_row, current_column]
                for offset_row, offset_column in neighbors:
                    next_row = current_row + offset_row
                    next_column = current_column + offset_column
                    if (
                        next_row < 0
                        or next_row >= height
                        or next_column < 0
                        or next_column >= width
                        or visited[next_row, next_column]
                    ):
                        continue
                    next_disparity = filtered[next_row, next_column]
                    if next_disparity <= 0:
                        continue
                    if min(abs(next_disparity - current_disparity), abs(next_disparity - seed_disparity)) > disparity_threshold:
                        continue
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))

            if len(component) < window_size:
                for current_row, current_column in component:
                    filtered[current_row, current_column] = 0.0

    return filtered.astype(np.float32)


def compute_numpy_block_disparity(left_gray: Any, right_gray: Any, config: O3Config) -> Any:
    """使用纯 numpy 代价体与后处理得到稠密视差。"""

    left = np.asarray(left_gray, dtype=np.float32)
    right = np.asarray(right_gray, dtype=np.float32)
    left_gradient = compute_horizontal_gradient(left)
    right_gradient = compute_horizontal_gradient(right)
    left_census = compute_census_transform(left, config.census_window_size)
    right_census = compute_census_transform(right, config.census_window_size)
    left_disparity, left_cost, left_second_cost = compute_matching_minima(
        left,
        right,
        left_gradient,
        right_gradient,
        left_census,
        right_census,
        config,
    )
    right_disparity, _, _ = compute_matching_minima(
        right,
        left,
        right_gradient,
        left_gradient,
        right_census,
        left_census,
        config,
        target_direction="positive",
    )

    if config.uniqueness_ratio > 0:
        uniqueness_margin = left_second_cost - left_cost
        uniqueness_floor = np.maximum(1e-3, left_cost * (float(config.uniqueness_ratio) / 100.0))
        left_disparity = np.where(uniqueness_margin >= uniqueness_floor, left_disparity, 0.0)

    left_disparity = refine_subpixel_disparity(
        left,
        right,
        left_gradient,
        right_gradient,
        left_census,
        right_census,
        left_disparity,
        config,
    )
    consistency_limit = max(float(config.consistency_threshold), float(config.disp12_max_diff))
    consistent_mask = left_right_consistency_mask(left_disparity, right_disparity, consistency_limit)
    disparity = np.where(consistent_mask, left_disparity, 0.0).astype(np.float32)

    invalid_margin = max(config.block_size // 2, config.census_window_size // 2)
    if invalid_margin > 0:
        disparity[:invalid_margin, :] = 0.0
        disparity[-invalid_margin:, :] = 0.0
        disparity[:, :invalid_margin] = 0.0
        disparity[:, -invalid_margin:] = 0.0

    disparity = filter_speckles(disparity, config.speckle_window_size, config.speckle_range)
    disparity = fill_invalid_disparity(disparity, config.fill_invalid_passes)
    if config.median_filter_size > 1:
        positive_mask = disparity > 0
        filtered = median_filter_2d(disparity, config.median_filter_size)
        disparity = np.where(positive_mask, filtered, 0.0).astype(np.float32)

    return disparity.astype(np.float32)


def filter_stereo_feature_matches(
    left_keypoints: Any,
    right_keypoints: Any,
    matches: list[Any],
    config: O3Config,
) -> list[Any]:

    valid_matches: list[Any] = []
    max_disparity = float(config.num_disparities)
    vertical_limit = float(config.max_vertical_offset)
    for match in matches:
        left_x, left_y = left_keypoints[match.queryIdx].pt
        right_x, right_y = right_keypoints[match.trainIdx].pt
        disparity = left_x - right_x
        if disparity <= 0.0 or disparity > max_disparity:
            continue
        if abs(left_y - right_y) > vertical_limit:
            continue
        valid_matches.append(match)
    valid_matches.sort(key=lambda match: match.distance)
    return valid_matches


def build_seed_disparity(
    left_keypoints: Any,
    right_keypoints: Any,
    matches: list[Any],
    image_shape: tuple[int, int],
) -> tuple[Any, Any]:

    height, width = image_shape
    seed_disparity = np.zeros((height, width), dtype=np.float32)
    seed_mask = np.zeros((height, width), dtype=bool)
    seed_distance = np.full((height, width), np.inf, dtype=np.float32)

    for match in matches:
        left_x, left_y = left_keypoints[match.queryIdx].pt
        right_x, _ = right_keypoints[match.trainIdx].pt
        column_index = int(np.clip(round(left_x), 0, width - 1))
        row_index = int(np.clip(round(left_y), 0, height - 1))
        if match.distance >= seed_distance[row_index, column_index]:
            continue
        seed_distance[row_index, column_index] = float(match.distance)
        seed_disparity[row_index, column_index] = float(left_x - right_x)
        seed_mask[row_index, column_index] = True

    return seed_disparity, seed_mask


def interpolate_seed_rows(seed_disparity: Any, seed_mask: Any) -> tuple[Any, Any]:
    """先沿行方向把稀疏种子插值成连续视差。"""

    disparity = np.zeros_like(seed_disparity, dtype=np.float32)
    row_mask = np.zeros_like(seed_mask, dtype=bool)
    width = seed_disparity.shape[1]
    columns = np.arange(width, dtype=np.float32)

    for row_index in range(seed_disparity.shape[0]):
        valid_columns = np.flatnonzero(seed_mask[row_index])
        if valid_columns.size == 0:
            continue
        row_values = seed_disparity[row_index, valid_columns]
        unique_columns, unique_indices = np.unique(valid_columns, return_index=True)
        unique_values = row_values[unique_indices]
        if unique_columns.size == 1:
            disparity[row_index, :] = unique_values[0]
        else:
            disparity[row_index, :] = np.interp(columns, unique_columns.astype(np.float32), unique_values).astype(np.float32)
        row_mask[row_index, :] = True

    return disparity, row_mask


def interpolate_seed_columns(row_disparity: Any, row_mask: Any) -> Any:
    """再沿列方向补齐没有种子的行。"""

    disparity = np.asarray(row_disparity, dtype=np.float32).copy()
    valid_rows = np.flatnonzero(np.any(row_mask, axis=1))
    if valid_rows.size == 0:
        return disparity

    if valid_rows.size == 1:
        disparity[:, :] = disparity[valid_rows[0], :]
        return disparity

    first_row = int(valid_rows[0])
    last_row = int(valid_rows[-1])
    disparity[:first_row, :] = disparity[first_row, :]
    disparity[last_row + 1 :, :] = disparity[last_row, :]

    for start_row, end_row in zip(valid_rows[:-1], valid_rows[1:]):
        start_index = int(start_row)
        end_index = int(end_row)
        if end_index - start_index <= 1:
            continue
        span = float(end_index - start_index)
        top = disparity[start_index, :]
        bottom = disparity[end_index, :]
        for row_index in range(start_index + 1, end_index):
            alpha = float(row_index - start_index) / span
            disparity[row_index, :] = ((1.0 - alpha) * top) + (alpha * bottom)

    return disparity.astype(np.float32)


def compute_sift_driven_disparity(left_gray: Any, right_gray: Any, config: O3Config) -> tuple[Any, dict[str, int]]:
    """以 SIFT 稀疏匹配为引导生成简化视差图。"""

    detector = create_sift_detector(config.max_features, config.contrast_threshold)
    left_keypoints, left_descriptors = detector.detectAndCompute(left_gray, None)
    right_keypoints, right_descriptors = detector.detectAndCompute(right_gray, None)

    stats = {
        "left_keypoints": len(left_keypoints),
        "right_keypoints": len(right_keypoints),
        "raw_matches": 0,
        "ratio_matches": 0,
        "mutual_matches": 0,
        "stereo_matches": 0,
    }

    if left_descriptors is None or right_descriptors is None or len(left_descriptors) == 0 or len(right_descriptors) == 0:
        return np.zeros_like(left_gray, dtype=np.float32), stats

    raw_matches, ratio_filtered, mutual_matches = _mutual_ratio_matches(
        left_descriptors,
        right_descriptors,
        float(config.ratio_test),
    )
    stats["raw_matches"] = int(raw_matches)
    stats["ratio_matches"] = len(ratio_filtered)
    stats["mutual_matches"] = len(mutual_matches)

    geometry_aware_matches, _ = _select_geometry_aware_matches(
        left_keypoints,
        right_keypoints,
        ratio_filtered,
        mutual_matches,
    )
    stereo_matches = filter_stereo_feature_matches(left_keypoints, right_keypoints, geometry_aware_matches, config)
    stats["stereo_matches"] = len(stereo_matches)
    if not stereo_matches:
        return np.zeros_like(left_gray, dtype=np.float32), stats

    seed_disparity, seed_mask = build_seed_disparity(
        left_keypoints,
        right_keypoints,
        stereo_matches,
        left_gray.shape,
    )
    row_disparity, row_mask = interpolate_seed_rows(seed_disparity, seed_mask)
    disparity = interpolate_seed_columns(row_disparity, row_mask)
    disparity = np.where(disparity > 0, disparity, 0.0).astype(np.float32)

    invalid_margin = max(config.block_size // 2, 1)
    if invalid_margin > 0:
        disparity[:invalid_margin, :] = 0.0
        disparity[-invalid_margin:, :] = 0.0
        disparity[:, :invalid_margin] = 0.0
        disparity[:, -invalid_margin:] = 0.0

    disparity = filter_speckles(disparity, config.speckle_window_size, config.speckle_range)
    disparity = fill_invalid_disparity(disparity, config.fill_invalid_passes)
    if config.median_filter_size > 1:
        positive_mask = disparity > 0
        filtered = median_filter_2d(disparity, config.median_filter_size)
        disparity = np.where(positive_mask, filtered, 0.0).astype(np.float32)

    return disparity.astype(np.float32), stats


def average_pool_gray(image: Any, factor: int) -> Any:
    """对灰度图做平均池化降采样。"""

    if factor <= 1:
        return image

    height, width = image.shape[:2]
    pooled_height = max(1, height // factor)
    pooled_width = max(1, width // factor)
    trimmed = image[: pooled_height * factor, : pooled_width * factor].astype(np.float32)
    pooled = trimmed.reshape(pooled_height, factor, pooled_width, factor).mean(axis=(1, 3))
    return np.rint(pooled).astype(np.uint8)


def average_pool_2d(image: Any, factor: int) -> Any:
    """对二维浮点图做平均池化降采样。"""

    source = np.asarray(image, dtype=np.float32)
    if factor <= 1:
        return source

    height, width = source.shape[:2]
    pooled_height = max(1, height // factor)
    pooled_width = max(1, width // factor)
    trimmed = source[: pooled_height * factor, : pooled_width * factor]
    return trimmed.reshape(pooled_height, factor, pooled_width, factor).mean(axis=(1, 3)).astype(np.float32)


def validate_results(disparity_dir: Path, analysis_dir: Path, metrics_file: Path, scene_name: str | None = None) -> int:
    """验证 O3 输出结果文件。"""
    print(f"Validating O3 disparity directory: {disparity_dir}")
    print(f"Validating O3 analysis directory: {analysis_dir}")
    print(f"Validating O3 metrics file: {metrics_file}")

    if not metrics_file.exists():
        print(f"O3 metrics file not found: {metrics_file}", file=sys.stderr)
        return 1

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("scene")]

    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
        rows = [row for row in rows if row.get("scene") == scene_name]

    if not rows:
        print("No O3 metric rows found for validation.", file=sys.stderr)
        return 1

    missing_any = False
    for row in rows:
        current_scene = row["scene"]
        disparity_scene_dir = disparity_dir / current_scene
        analysis_scene_dir = analysis_dir / current_scene
        required_paths = (
            disparity_scene_dir / "disp0.pfm",
            disparity_scene_dir / "disp0.png",
            analysis_scene_dir / "error_map.png",
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            missing_any = True
            print(f"[MISSING] {current_scene}: " + ", ".join(missing))
        else:
            print(f"[OK] {current_scene}: required O3 files present")

    if missing_any:
        print("Validation status: issues found")
        return 1

    print("Validation status: all checked O3 scene folders contain the expected files")
    return 0


def run(
    repo_root: Path,
    middlebury_root: Path,
    config: O3Config,
    max_scenes: int | None,
    dry_run: bool,
    scene_name: str | None,
) -> int:

    discovered_scenes = discover_scenes(middlebury_root)
    discovered_count = len(discovered_scenes)
    scenes = filter_scene_dirs(discovered_scenes, scene_name)

    if max_scenes is not None:
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    print(f"Repository root: {repo_root}")
    print(f"Middlebury root: {middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {discovered_count}")
    print(f"O3 disparity output dir: {config.disparity_dir}")
    print(f"O3 analysis output dir: {config.analysis_dir}")
    print(f"O3 metrics file: {config.metrics_file}")
    print(
        "O3 SIFT stereo config: "
        f"max_features={config.max_features}, contrast_threshold={config.contrast_threshold}, "
        f"ratio_test={config.ratio_test}, num_disparities={config.num_disparities}, "
        f"max_vertical_offset={config.max_vertical_offset}, speckle_window_size={config.speckle_window_size}, "
        f"speckle_range={config.speckle_range}, fill_invalid_passes={config.fill_invalid_passes}"
    )
    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
    if scenes:
        print("Scenes to process: " + ", ".join(scene.name for scene in scenes))
    else:
        print("Scenes to process: none")

    if not middlebury_root.exists():
        if dry_run or max_scenes == 0:
            print("Middlebury root does not exist yet. Discovery-only mode completed without processing.")
            return 0
        print(
            f"Middlebury root not found: {middlebury_root}\n"
            "Place dataset scenes there, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    if discovered_count == 0:
        if dry_run or max_scenes == 0:
            print("No valid scenes found. Discovery-only mode completed without processing.")
            return 0
        print(
            f"No scene directories containing im0.png and im1.png were found under {middlebury_root}.\n"
            "Add Middlebury scenes, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    if scene_name is not None and not scenes:
        print(f"No discovered scenes matched --scene-name {scene_name!r} under {middlebury_root}.", file=sys.stderr)
        return 1

    if dry_run or max_scenes == 0:
        print("Dry run requested; no outputs were written.")
        return 0

    config.disparity_dir.mkdir(parents=True, exist_ok=True)
    config.analysis_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, str | int | float]] = []
    for scene_dir in scenes:
        left_gray = load_gray(scene_dir / "im0.png")
        right_gray = load_gray(scene_dir / "im1.png")
        disparity, stats = compute_sift_driven_disparity(left_gray, right_gray, config)
        generator_name = (
            "O3 SIFT-driven stereo disparity from feature matches with row/column interpolation "
            "and lightweight disparity cleanup"
        )
        disparity_mask = disparity > 0

        disparity_scene_dir = config.disparity_dir / scene_dir.name
        analysis_scene_dir = config.analysis_dir / scene_dir.name
        disparity_scene_dir.mkdir(parents=True, exist_ok=True)
        analysis_scene_dir.mkdir(parents=True, exist_ok=True)

        write_pfm(disparity_scene_dir / "disp0.pfm", disparity)
        disparity_preview = normalize_for_preview(disparity, disparity_mask)
        write_png(disparity_scene_dir / "disp0.png", disparity_preview)
        write_scene_text(
            disparity_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                f"generator: {generator_name}",
                f"left_keypoints: {stats['left_keypoints']}",
                f"right_keypoints: {stats['right_keypoints']}",
                f"raw_matches: {stats['raw_matches']}",
                f"ratio_matches: {stats['ratio_matches']}",
                f"mutual_matches: {stats['mutual_matches']}",
                f"stereo_matches: {stats['stereo_matches']}",
                f"max_features: {config.max_features}",
                f"contrast_threshold: {config.contrast_threshold}",
                f"ratio_test: {config.ratio_test}",
                f"num_disparities: {config.num_disparities}",
                f"max_vertical_offset: {config.max_vertical_offset}",
                f"block_size: {config.block_size}",
                f"speckle_window_size: {config.speckle_window_size}",
                f"speckle_range: {config.speckle_range}",
                f"fill_invalid_passes: {config.fill_invalid_passes}",
                f"median_filter_size: {config.median_filter_size}",
                f"valid_disparity_pixels: {int(disparity_mask.sum())}",
            ],
        )

        metrics = {
            "valid_disparity_pixels": int(disparity_mask.sum()),
            "valid_ground_truth_pixels": 0,
            "mae": -1.0,
            "rmse": -1.0,
            "bad_1px": -1.0,
        }
        error_map = np.zeros_like(disparity, dtype=np.float32)
        ground_truth_path = scene_dir / "disp0.pfm"
        if ground_truth_path.exists():
            ground_truth = read_pfm(ground_truth_path)
            if getattr(ground_truth, "ndim", 0) == 3:
                ground_truth = ground_truth[:, :, 0]
            ground_truth = np.asarray(ground_truth, dtype=np.float32)
            metrics = evaluate_disparity(disparity, ground_truth)
            valid_mask = (disparity > 0) & np.isfinite(ground_truth) & (ground_truth > 0)
            error_map[valid_mask] = np.abs(disparity[valid_mask] - ground_truth[valid_mask])

        error_mask = error_map > 0
        error_preview = normalize_for_preview(error_map, error_mask)
        error_color = np.stack([error_preview, error_preview, error_preview], axis=2)
        error_color[~error_mask] = 0
        write_png(analysis_scene_dir / "error_map.png", error_color)
        write_scene_text(
            analysis_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                f"generator: {generator_name}",
                f"ground_truth_available: {'yes' if ground_truth_path.exists() else 'no'}",
                f"mae: {metrics['mae'] if metrics['mae'] >= 0 else 'NA'}",
                f"rmse: {metrics['rmse'] if metrics['rmse'] >= 0 else 'NA'}",
                f"bad_1px: {metrics['bad_1px'] if metrics['bad_1px'] >= 0 else 'NA'}",
            ],
        )

        metric_rows.append(
            {
                "scene": scene_dir.name,
                "valid_disparity_pixels": metrics["valid_disparity_pixels"],
                "valid_ground_truth_pixels": metrics["valid_ground_truth_pixels"],
                "mae": f"{metrics['mae']:.6f}" if metrics["mae"] >= 0 else "NA",
                "rmse": f"{metrics['rmse']:.6f}" if metrics["rmse"] >= 0 else "NA",
                "bad_1px": f"{metrics['bad_1px']:.6f}" if metrics["bad_1px"] >= 0 else "NA",
            }
        )
        mae_text = f"{metrics['mae']:.6f}" if metrics["mae"] >= 0 else "NA"
        rmse_text = f"{metrics['rmse']:.6f}" if metrics["rmse"] >= 0 else "NA"
        bad_text = f"{metrics['bad_1px']:.6f}" if metrics["bad_1px"] >= 0 else "NA"
        print(
            f"Wrote O3 scene: {scene_dir.name} "
            f"(left={stats['left_keypoints']}, right={stats['right_keypoints']}, stereo_matches={stats['stereo_matches']}, "
            f"valid_disparity_pixels={metrics['valid_disparity_pixels']}, mae={mae_text}, rmse={rmse_text}, bad_1px={bad_text})"
        )

    write_metrics(config.metrics_file, metric_rows)
    print(f"Wrote O3 metrics summary: {config.metrics_file}")
    return 0
