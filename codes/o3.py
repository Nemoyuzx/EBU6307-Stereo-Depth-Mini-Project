from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import Delaunay, QhullError, cKDTree

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
    import torch.nn.functional as torch_functional
except Exception:
    torch = None
    torch_functional = None

from common import (
    discover_scenes,
    evaluate_disparity,
    filter_scene_dirs,
    load_gray,
    normalize_for_preview,
    write_png,
    write_scene_text,
)
from config import O3Config, load_config
from o2 import _mutual_ratio_matches, _select_geometry_aware_matches, create_sift_detector
from pfm import read_pfm, write_pfm


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
    merged_rows.sort(key=lambda row: str(row["scene"]))

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


def write_o3_pdf_disparity_csv(metrics_file: Path, rows: list[MetricRow]) -> None:
    """Write the PDF-required O3 disparity-error table with the exact filename."""

    csv_path = metrics_file.parent / "disparity.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: str(row["scene"]))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "valid_disparity_pixels",
                "valid_ground_truth_pixels",
                "disparity_error",
                "rmse",
                "bad_1px",
            ],
        )
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(
                {
                    "scene": row["scene"],
                    "valid_disparity_pixels": row["valid_disparity_pixels"],
                    "valid_ground_truth_pixels": row["valid_ground_truth_pixels"],
                    "disparity_error": row["mae"],
                    "rmse": row["rmse"],
                    "bad_1px": row["bad_1px"],
                }
            )


def write_jpeg(path: Path, image: Any) -> None:
    """Write an RGB or grayscale image as a PDF-required JPG artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=2)
    Image.fromarray(array).convert("RGB").save(path, format="JPEG", quality=92)


def _draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: tuple[int, int, int]) -> None:
    lines = text.split("\n")
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 6
    y = box[1] + max(0, (box[3] - box[1] - total_height) // 2)
    for line, line_width, line_height in zip(lines, line_widths, line_heights):
        x = box[0] + max(0, (box[2] - box[0] - line_width) // 2)
        draw.text((x, y), line, fill=fill)
        y += line_height + 6


def create_o3_pipeline_image(path: Path) -> None:
    """Generate the PDF-required O3 pipeline diagram."""

    canvas = Image.new("RGB", (1500, 520), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1499, 519), outline=(210, 210, 210), width=2)
    draw.text((36, 28), "O3 Stereo Depth Estimation Pipeline using SIFT Features", fill=(20, 20, 20))
    boxes = [
        (40, 130, 235, 300, "Input stereo\nim0.png + im1.png"),
        (290, 130, 485, 300, "Manual SIFT\nkeypoints + descriptors"),
        (540, 130, 735, 300, "Ratio / mutual /\nstereo geometry filtering"),
        (790, 130, 985, 300, "SIFT seed\ndisparity prior"),
        (1040, 130, 1235, 300, "Census-gradient\nSGM cost volume"),
        (1290, 130, 1460, 300, "LR consistency +\nlocal support cleanup"),
    ]
    for left, top, right, bottom, label in boxes:
        draw.rounded_rectangle((left, top, right, bottom), radius=10, fill=(238, 246, 255), outline=(55, 95, 145), width=3)
        _draw_centered_text(draw, (left + 12, top + 12, right - 12, bottom - 12), label, (25, 55, 85))
    for index in range(len(boxes) - 1):
        start_x = boxes[index][2] + 12
        end_x = boxes[index + 1][0] - 12
        y = 215
        draw.line((start_x, y, end_x, y), fill=(45, 80, 125), width=4)
        draw.polygon(((end_x, y), (end_x - 13, y - 8), (end_x - 13, y + 8)), fill=(45, 80, 125))
    draw.text((40, 370), "Outputs:", fill=(20, 20, 20))
    draw.text((40, 400), "results/O3a_disparity/dep_pipeline.jpg", fill=(40, 40, 40))
    draw.text((40, 426), "results/O3b_disparity/example_1.jpg ... example_3.jpg", fill=(40, 40, 40))
    draw.text((40, 452), "results/O3c_disparity/disparity.csv", fill=(40, 40, 40))
    write_jpeg(path, np.asarray(canvas, dtype=np.uint8))


def _resize_for_example(image: Image.Image, target_height: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    width = max(1, int(round(image.width * (float(target_height) / max(1, image.height)))))
    return image.resize((width, target_height), resampling)


def create_o3_example_image(scene_name: str, left_gray: Any, disparity_preview: Any, metrics: dict[str, float | int]) -> Any:
    """Compose a PDF-required example image displaying the O3 disparity map."""

    left_panel = Image.fromarray(np.asarray(left_gray, dtype=np.uint8)).convert("RGB")
    disparity_panel = Image.fromarray(np.asarray(disparity_preview, dtype=np.uint8)).convert("RGB")
    panel_height = 520
    left_panel = _resize_for_example(left_panel, panel_height)
    disparity_panel = _resize_for_example(disparity_panel, panel_height)
    width = left_panel.width + disparity_panel.width + 72
    height = panel_height + 118
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), f"O3 disparity example: {scene_name}", fill=(20, 20, 20))
    draw.text((24, 44), "Left input image", fill=(70, 70, 70))
    disparity_x = left_panel.width + 48
    draw.text((disparity_x, 44), "Predicted disparity map", fill=(70, 70, 70))
    canvas.paste(left_panel, (24, 70))
    canvas.paste(disparity_panel, (disparity_x, 70))
    draw.rectangle((24, 70, 24 + left_panel.width, 70 + panel_height), outline=(180, 180, 180), width=2)
    draw.rectangle((disparity_x, 70, disparity_x + disparity_panel.width, 70 + panel_height), outline=(180, 180, 180), width=2)
    mae = metrics.get("mae", -1.0)
    rmse = metrics.get("rmse", -1.0)
    bad = metrics.get("bad_1px", -1.0)
    metric_text = f"MAE/disparity error: {mae:.6f}    RMSE: {rmse:.6f}    bad_1px: {bad:.6f}" if mae >= 0 else "Ground truth unavailable"
    draw.text((24, height - 32), metric_text, fill=(40, 40, 40))
    return np.asarray(canvas, dtype=np.uint8)


def write_o3_pdf_assets(
    config: O3Config,
    example_payloads: dict[str, dict[str, Any]],
    metric_rows: list[MetricRow],
) -> None:
    """Write the exact O3 files requested by the project PDF."""

    create_o3_pipeline_image(config.disparity_dir / "dep_pipeline.jpg")
    write_o3_pdf_disparity_csv(config.metrics_file, metric_rows)
    preferred_scenes = ("artroom1", "ladder1", "pendulum2")
    selected = [scene for scene in preferred_scenes if scene in example_payloads]
    selected.extend(scene for scene in example_payloads if scene not in selected)
    for index, scene_name in enumerate(selected[:3], start=1):
        payload = example_payloads[scene_name]
        example = create_o3_example_image(scene_name, payload["left_gray"], payload["disparity_preview"], payload["metrics"])
        write_jpeg(config.analysis_dir / f"example_{index}.jpg", example)


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


def left_right_consistency_error(left_disparity: Any, right_disparity: Any) -> Any:
    """返回每个左图像素对应的左右一致性误差，用作置信度而不是硬删除条件。"""

    left = np.asarray(left_disparity, dtype=np.float32)
    right = np.asarray(right_disparity, dtype=np.float32)
    height, width = left.shape
    columns = np.arange(width, dtype=np.int32)[None, :]
    matched_columns = columns - np.rint(left).astype(np.int32)
    valid = (left > 0) & (matched_columns >= 0) & (matched_columns < width)
    row_indices = np.broadcast_to(np.arange(height, dtype=np.int32)[:, None], left.shape)
    sampled = np.zeros_like(left, dtype=np.float32)
    sampled[valid] = right[row_indices[valid], matched_columns[valid]]
    error = np.full_like(left, np.inf, dtype=np.float32)
    error[valid] = np.abs(left[valid] - sampled[valid])
    return error.astype(np.float32)


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


def fill_short_horizontal_disparity_gaps(
    disparity: Any,
    guide_gray: Any,
    max_gap: int,
    max_disparity_delta: float,
    max_intensity_delta: float,
) -> Any:
    """只沿同一扫描线填补短小且两端视差一致的空洞，避免圆形邻域扩散。"""

    result = np.asarray(disparity, dtype=np.float32).copy()
    guide = np.asarray(guide_gray, dtype=np.float32)
    if max_gap <= 0:
        return result

    height, width = result.shape
    for row_index in range(height):
        column_index = 0
        while column_index < width:
            if result[row_index, column_index] > 0:
                column_index += 1
                continue

            gap_start = column_index
            while column_index < width and result[row_index, column_index] <= 0:
                column_index += 1
            gap_end = column_index
            gap_width = gap_end - gap_start
            left_column = gap_start - 1
            right_column = gap_end
            if (
                gap_width > max_gap
                or left_column < 0
                or right_column >= width
                or result[row_index, left_column] <= 0
                or result[row_index, right_column] <= 0
            ):
                continue

            left_disparity = float(result[row_index, left_column])
            right_disparity = float(result[row_index, right_column])
            if abs(left_disparity - right_disparity) > float(max_disparity_delta):
                continue

            interior = guide[row_index, gap_start:gap_end]
            if interior.size == 0:
                continue
            boundary_mean = 0.5 * (float(guide[row_index, left_column]) + float(guide[row_index, right_column]))
            if float(np.max(np.abs(interior - boundary_mean))) > float(max_intensity_delta):
                continue

            result[row_index, gap_start:gap_end] = np.linspace(
                left_disparity,
                right_disparity,
                gap_width + 2,
                dtype=np.float32,
            )[1:-1]

    return result.astype(np.float32)


def fill_short_vertical_disparity_gaps(
    disparity: Any,
    guide_gray: Any,
    max_gap: int,
    max_disparity_delta: float,
    max_intensity_delta: float,
) -> Any:
    """沿列方向复用扫描线补洞，补上水平扫描无法跨过的短竖向断裂。"""

    return fill_short_horizontal_disparity_gaps(
        np.asarray(disparity, dtype=np.float32).T,
        np.asarray(guide_gray, dtype=np.float32).T,
        max_gap=max_gap,
        max_disparity_delta=max_disparity_delta,
        max_intensity_delta=max_intensity_delta,
    ).T.astype(np.float32)


def colorize_disparity_depth_map(disparity: Any, valid_mask: Any) -> Any:
    """把 O3 视差预览映射成 RGB 深度图；黑色表示无效视差。"""

    normalized = normalize_for_preview(disparity, valid_mask).astype(np.float32) / 255.0
    valid = np.asarray(valid_mask, dtype=bool)
    color_stops = np.asarray(
        [
            (0.02, 0.08, 0.35),
            (0.00, 0.35, 0.85),
            (0.00, 0.78, 0.95),
            (0.35, 0.88, 0.35),
            (1.00, 0.85, 0.10),
            (0.95, 0.20, 0.05),
        ],
        dtype=np.float32,
    )
    scaled = np.clip(normalized, 0.0, 1.0) * float(len(color_stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(color_stops) - 1)
    fraction = (scaled - lower.astype(np.float32))[:, :, None]
    rgb = ((1.0 - fraction) * color_stops[lower]) + (fraction * color_stops[upper])
    rgb = np.where(valid[:, :, None], rgb, 0.0)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


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


def joint_weighted_median_filter_disparity(
    disparity: Any,
    guide_gray: Any,
    radius: int,
    sigma_color: float,
    sigma_space: float,
) -> Any:
    """用左图引导的加权中值滤波清理噪声，同时避免把不同物体深度直接平均。"""

    source = np.asarray(disparity, dtype=np.float32)
    guide = np.asarray(guide_gray, dtype=np.float32)
    if radius <= 0:
        return source

    padded_disparity = np.pad(source, ((radius, radius), (radius, radius)), mode="edge")
    padded_guide = np.pad(guide, ((radius, radius), (radius, radius)), mode="edge")
    values: list[Any] = []
    weights: list[Any] = []
    color_scale = max(float(sigma_color), 1e-3)
    space_scale = max(float(sigma_space), 1e-3)

    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            shifted_disparity = padded_disparity[
                radius + offset_y : radius + offset_y + source.shape[0],
                radius + offset_x : radius + offset_x + source.shape[1],
            ]
            shifted_guide = padded_guide[
                radius + offset_y : radius + offset_y + source.shape[0],
                radius + offset_x : radius + offset_x + source.shape[1],
            ]
            valid = np.isfinite(shifted_disparity) & (shifted_disparity > 0)
            spatial_distance = float((offset_y * offset_y) + (offset_x * offset_x))
            spatial_weight = np.exp(-spatial_distance / (2.0 * space_scale * space_scale))
            color_delta = guide - shifted_guide
            color_weight = np.exp(-(color_delta * color_delta) / (2.0 * color_scale * color_scale)).astype(np.float32)
            values.append(shifted_disparity.astype(np.float32, copy=False))
            weights.append((spatial_weight * color_weight * valid.astype(np.float32)).astype(np.float32))

    value_stack = np.stack(values, axis=0)
    weight_stack = np.stack(weights, axis=0)
    order = np.argsort(value_stack, axis=0)
    sorted_values = np.take_along_axis(value_stack, order, axis=0)
    sorted_weights = np.take_along_axis(weight_stack, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    total = cumulative[-1]
    median_index = np.argmax(cumulative >= (0.5 * total[None, :, :]), axis=0)
    filtered = np.take_along_axis(sorted_values, median_index[None, :, :], axis=0)[0]
    return np.where(total > 0, filtered, source).astype(np.float32)


def local_disparity_support_mask(
    disparity: Any,
    candidate_mask: Any,
    radius: int,
    tolerance: float,
    min_count: int,
) -> Any:
    """保留邻域内有足够同类支持的候选视差，剔除孤立噪点。"""

    source = np.asarray(disparity, dtype=np.float32)
    candidates = np.asarray(candidate_mask, dtype=bool)
    if radius <= 0 or min_count <= 1:
        return candidates

    kernel_size = (radius * 2) + 1
    padded_source = np.pad(source, ((radius, radius), (radius, radius)), mode="edge")
    padded_candidates = np.pad(candidates, ((radius, radius), (radius, radius)), mode="constant")
    source_windows = np.lib.stride_tricks.sliding_window_view(padded_source, (kernel_size, kernel_size))
    candidate_windows = np.lib.stride_tricks.sliding_window_view(padded_candidates, (kernel_size, kernel_size))
    close = np.abs(source_windows - source[:, :, None, None]) <= float(tolerance)
    support_count = np.sum(candidate_windows & close, axis=(-2, -1))
    return candidates & (support_count >= int(min_count))


def compute_pixel_matching_cost(
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
    """构建 SGM 使用的像素级匹配代价，避免 block 聚合提前抹平边界。"""

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

    intensity_cost = np.minimum(np.abs(reference - shifted_gray), 32.0)
    gradient_cost = np.minimum(np.abs(reference_gradient - shifted_gradient), 16.0)
    census_cost = bit_count_array(reference_census ^ shifted_census)
    cost = (0.5 * intensity_cost) + (config.gradient_weight * gradient_cost) + (config.census_weight * census_cost)
    if disparity > 0:
        if target_direction == "positive":
            cost[:, width - disparity :] = np.inf
        else:
            cost[:, :disparity] = np.inf
    return cost.astype(np.float32)


def read_scene_disparity_hints(scene_dir: Path) -> dict[str, float]:
    """从 Middlebury calib.txt 读取场景视差范围提示。"""

    hints: dict[str, float] = {}
    calib_path = scene_dir / "calib.txt"
    if not calib_path.exists():
        return hints

    for line in calib_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in {"width", "height", "ndisp", "vmin", "vmax"}:
            continue
        try:
            hints[key] = float(value)
        except ValueError:
            continue
    return hints


def estimate_scene_disparity_bounds(scene_dir: Path, image_shape: tuple[int, int], config: O3Config) -> tuple[int, int]:
    """根据标定文件为高视差场景扩展搜索范围，低视差场景保持默认范围。"""

    base_min = 0
    base_max = min(max(1, int(config.num_disparities)) - 1, image_shape[1] - 1)
    hints = read_scene_disparity_hints(scene_dir)
    if not hints:
        return base_min, base_max

    image_height, image_width = image_shape
    calib_width = hints.get("width", 0.0)
    calib_height = hints.get("height", 0.0)
    scale = 1.0
    if calib_width > 0 and calib_height > 0:
        width_scale = image_width / calib_width
        height_scale = image_height / calib_height
        if abs(width_scale - height_scale) <= 0.05:
            scale = width_scale

    calibrated_max = hints.get("vmax", hints.get("ndisp", 0.0)) * scale
    if calibrated_max <= base_max:
        return base_min, base_max

    calibrated_min = hints.get("vmin", 0.0) * scale
    calibrated_span = max(1.0, calibrated_max - calibrated_min)
    margin = max(8.0, calibrated_span * 0.05)
    search_min = max(0, int(np.floor(calibrated_min - margin)))
    search_max = min(image_width - 1, int(np.ceil(calibrated_max + margin)))
    return search_min, max(search_min, search_max)


def build_matching_cost_volume(
    left_gray: Any,
    right_gray: Any,
    config: O3Config,
    target_direction: str = "negative",
    min_disparity: int = 0,
    max_disparity: int | None = None,
) -> Any:
    """构建 O3 像素级匹配代价体，后续由自实现 SGM 聚合。"""

    left = np.asarray(left_gray, dtype=np.float32)
    right = np.asarray(right_gray, dtype=np.float32)
    left_gradient = compute_horizontal_gradient(left)
    right_gradient = compute_horizontal_gradient(right)
    left_census = compute_census_transform(left, config.census_window_size)
    right_census = compute_census_transform(right, config.census_window_size)
    lower_bound = max(0, int(min_disparity))
    upper_bound = min(
        int(max_disparity) if max_disparity is not None else max(1, config.num_disparities) - 1,
        left.shape[1] - 1,
    )
    upper_bound = max(lower_bound, upper_bound)
    disparities = range(lower_bound, upper_bound + 1)
    volume = np.empty((left.shape[0], left.shape[1], len(disparities)), dtype=np.float32)
    for volume_index, disparity in enumerate(disparities):
        volume[:, :, volume_index] = compute_pixel_matching_cost(
            left,
            right,
            left_gradient,
            right_gradient,
            left_census,
            right_census,
            config,
            disparity,
            target_direction=target_direction,
        )
    return volume


def normalize_cost_volume(cost_volume: Any) -> Any:
    """把代价体转成适合 SGM 聚合的有限相对代价。"""

    source = np.asarray(cost_volume, dtype=np.float32)
    finite = np.isfinite(source)
    if not bool(np.any(finite)):
        return np.zeros_like(source, dtype=np.float32)

    finite_values = source[finite]
    high_cost = float(np.percentile(finite_values, 99.0))
    replacement = high_cost + max(1.0, abs(high_cost) * 0.25)
    normalized = np.where(finite, source, replacement).astype(np.float32)
    normalized -= normalized.min(axis=2, keepdims=True)
    return normalized.astype(np.float32)


def estimate_sgm_penalties(cost_volume: Any) -> tuple[float, float]:
    """根据当前场景代价分布估计 SGM 平滑惩罚。"""

    finite = np.asarray(cost_volume, dtype=np.float32)
    positive = finite[np.isfinite(finite) & (finite > 0)]
    if positive.size == 0:
        return 1.0, 8.0
    scale = float(np.percentile(positive, 60.0))
    p1 = max(1.0, scale * 0.08)
    p2 = max(p1 * 5.0, scale * 0.55)
    return p1, p2


_O3_TORCH_FALLBACK_REPORTED = False


def clear_o3_torch_cache(device: str | None = None) -> None:
    if torch is None or not torch.cuda.is_available():
        return
    if device is not None and not str(device).startswith("cuda"):
        return
    torch.cuda.empty_cache()


def _read_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def select_o3_torch_cost_dtype(height: int, width: int, disparity_count: int, device: str) -> Any:
    if not str(device).startswith("cuda"):
        return torch.float32
    fp32_gib = (int(height) * int(width) * int(disparity_count) * 4) / float(1024**3)
    fp16_threshold_gib = max(0.0, _read_env_float("O3_TORCH_FP16_VOLUME_GIB", 1.0))
    return torch.float16 if fp32_gib >= fp16_threshold_gib else torch.float32


def resolve_o3_torch_device() -> str | None:
    """Return a CUDA device for O3 SGM acceleration when PyTorch is available."""

    if torch is None:
        return None
    if not torch.cuda.is_available():
        return None
    return "cuda"


def compute_horizontal_gradient_torch(image: Any) -> Any:
    source = image.to(dtype=torch.float32)
    gradient = torch.zeros_like(source)
    if source.shape[1] > 1:
        gradient[:, 1:-1] = 0.5 * (source[:, 2:] - source[:, :-2])
        gradient[:, 0] = source[:, 1] - source[:, 0]
        gradient[:, -1] = source[:, -1] - source[:, -2]
    return gradient


def shift_stereo_tensor_torch(target: Any, disparity: int, target_direction: str) -> Any:
    shifted = torch.zeros_like(target)
    width = int(target.shape[1])
    if disparity == 0:
        shifted[:] = target
    elif target_direction == "positive" and disparity < width:
        shifted[:, : width - disparity] = target[:, disparity:]
    elif disparity < width:
        shifted[:, disparity:] = target[:, : width - disparity]
    return shifted


def census_hamming_cost_torch(reference: Any, shifted_target: Any, window_size: int) -> Any:
    radius = max(0, int(window_size) // 2)
    if radius <= 0:
        return torch.zeros_like(reference, dtype=torch.float32)
    kernel_size = (radius * 2) + 1
    height, width = reference.shape
    ref_patches = torch_functional.unfold(reference[None, None, :, :], kernel_size=kernel_size, padding=radius)[0]
    target_patches = torch_functional.unfold(shifted_target[None, None, :, :], kernel_size=kernel_size, padding=radius)[0]
    ref_center = reference.reshape(1, height * width)
    target_center = shifted_target.reshape(1, height * width)
    ref_bits = ref_patches < ref_center
    target_bits = target_patches < target_center
    center_index = (kernel_size * kernel_size) // 2
    ref_bits[center_index, :] = False
    target_bits[center_index, :] = False
    return torch.logical_xor(ref_bits, target_bits).sum(dim=0).reshape(height, width).to(dtype=torch.float32)


def compute_pixel_matching_cost_torch(
    reference_gray: Any,
    target_gray: Any,
    reference_gradient: Any,
    target_gradient: Any,
    config: O3Config,
    disparity: int,
    target_direction: str,
) -> Any:
    height, width = reference_gray.shape
    shifted_gray = shift_stereo_tensor_torch(target_gray, disparity, target_direction)
    shifted_gradient = shift_stereo_tensor_torch(target_gradient, disparity, target_direction)
    intensity_cost = torch.clamp(torch.abs(reference_gray - shifted_gray), max=32.0)
    gradient_cost = torch.clamp(torch.abs(reference_gradient - shifted_gradient), max=16.0)
    census_cost = census_hamming_cost_torch(reference_gray, shifted_gray, config.census_window_size)
    cost = (0.5 * intensity_cost) + (float(config.gradient_weight) * gradient_cost) + (float(config.census_weight) * census_cost)
    if disparity > 0:
        if target_direction == "positive":
            cost[:, width - disparity :] = torch.inf
        else:
            cost[:, :disparity] = torch.inf
    return cost.to(dtype=torch.float32)


def build_matching_cost_volume_torch(
    left_gray: Any,
    right_gray: Any,
    config: O3Config,
    target_direction: str,
    min_disparity: int,
    max_disparity: int | None,
    device: str,
) -> Any:
    left = torch.as_tensor(np.asarray(left_gray, dtype=np.float32), dtype=torch.float32, device=device)
    right = torch.as_tensor(np.asarray(right_gray, dtype=np.float32), dtype=torch.float32, device=device)
    left_gradient = compute_horizontal_gradient_torch(left)
    right_gradient = compute_horizontal_gradient_torch(right)
    lower_bound = max(0, int(min_disparity))
    upper_bound = min(
        int(max_disparity) if max_disparity is not None else max(1, int(config.num_disparities)) - 1,
        int(left.shape[1]) - 1,
    )
    upper_bound = max(lower_bound, upper_bound)
    disparity_count = upper_bound - lower_bound + 1
    cost_dtype = select_o3_torch_cost_dtype(int(left.shape[0]), int(left.shape[1]), disparity_count, device)
    volume = torch.empty((int(left.shape[0]), int(left.shape[1]), disparity_count), dtype=cost_dtype, device=device)
    for volume_index, disparity in enumerate(range(lower_bound, upper_bound + 1)):
        cost = compute_pixel_matching_cost_torch(
            left,
            right,
            left_gradient,
            right_gradient,
            config,
            disparity,
            target_direction,
        )
        volume[:, :, volume_index].copy_(cost.to(dtype=cost_dtype))
        del cost
    return volume


def normalize_cost_volume_torch(cost_volume: Any) -> Any:
    finite_sample = sample_torch_values_for_quantile(cost_volume, finite_only=True)
    if int(finite_sample.numel()) == 0:
        return torch.zeros_like(cost_volume, dtype=torch.float32)
    high_cost = torch.quantile(finite_sample.to(dtype=torch.float32), 0.99)
    replacement = high_cost + torch.clamp(torch.abs(high_cost) * 0.25, min=1.0)
    replacement_value = float(replacement.detach().cpu().item())
    if hasattr(cost_volume, "nan_to_num_"):
        cost_volume.nan_to_num_(nan=replacement_value, posinf=replacement_value, neginf=0.0)
    else:
        cost_volume[~torch.isfinite(cost_volume)] = replacement_value
    cost_volume.sub_(cost_volume.amin(dim=2, keepdim=True))
    return cost_volume


def sample_torch_values_for_quantile(
    values: Any,
    max_samples: int = 1_000_000,
    *,
    finite_only: bool = False,
    positive_only: bool = False,
) -> Any:
    sample = values.reshape(-1)
    count = int(sample.numel())
    if count > max_samples:
        step = max(1, int(np.ceil(count / max_samples)))
        sample = sample[::step]
    if finite_only or positive_only:
        sample_mask = torch.isfinite(sample)
        if positive_only:
            sample_mask = sample_mask & (sample > 0)
        sample = sample[sample_mask]
    return sample


def estimate_sgm_penalties_torch(cost_volume: Any) -> tuple[float, float]:
    positive_sample = sample_torch_values_for_quantile(cost_volume, finite_only=True, positive_only=True)
    if int(positive_sample.numel()) == 0:
        return 1.0, 8.0
    scale = float(torch.quantile(positive_sample.to(dtype=torch.float32), 0.60).detach().cpu().item())
    p1 = max(1.0, scale * 0.08)
    p2 = max(p1 * 5.0, scale * 0.55)
    return p1, p2


def torch_isfinite_positive(value: Any) -> Any:
    return torch.isfinite(value) & (value > 0)


def accumulate_sgm_axis_torch(cost_volume: Any, guide_gray: Any, output: Any, axis: int, reverse: bool, p1: float, p2: float) -> None:
    source = cost_volume
    guide = guide_gray.to(dtype=torch.float32)
    height, width, _ = source.shape
    infinity = torch.tensor(1e9, dtype=torch.float32, device=source.device)
    edge_scale = torch.tensor(8.0, dtype=torch.float32, device=source.device)
    minimum_jump = torch.tensor(max(float(p1) * 2.0, 1.0), dtype=torch.float32, device=source.device)

    if axis == 1:
        first_column = width - 1 if reverse else 0
        previous = source[:, first_column, :].to(dtype=torch.float32)
        output[:, first_column, :].add_(previous)
        columns = range(first_column - 1, -1, -1) if reverse else range(first_column + 1, width)
        for column_index in columns:
            previous_column = column_index + 1 if reverse else column_index - 1
            edge_delta = torch.abs(guide[:, column_index] - guide[:, previous_column])[:, None]
            adaptive_p2 = torch.maximum(minimum_jump, float(p2) / (1.0 + (edge_delta / edge_scale)))
            previous_min = previous.min(dim=1, keepdim=True).values
            from_lower = torch.empty_like(previous)
            from_upper = torch.empty_like(previous)
            from_lower[:, 0] = infinity
            from_lower[:, 1:] = previous[:, :-1] + float(p1)
            from_upper[:, :-1] = previous[:, 1:] + float(p1)
            from_upper[:, -1] = infinity
            jump_cost = previous_min + adaptive_p2
            transition = torch.minimum(torch.minimum(previous, from_lower), torch.minimum(from_upper, jump_cost))
            current = source[:, column_index, :].to(dtype=torch.float32) + transition - previous_min
            output[:, column_index, :].add_(current)
            previous = current
        return

    first_row = height - 1 if reverse else 0
    previous = source[first_row, :, :].to(dtype=torch.float32)
    output[first_row, :, :].add_(previous)
    rows = range(first_row - 1, -1, -1) if reverse else range(first_row + 1, height)
    for row_index in rows:
        previous_row = row_index + 1 if reverse else row_index - 1
        edge_delta = torch.abs(guide[row_index, :] - guide[previous_row, :])[:, None]
        adaptive_p2 = torch.maximum(minimum_jump, float(p2) / (1.0 + (edge_delta / edge_scale)))
        previous_min = previous.min(dim=1, keepdim=True).values
        from_lower = torch.empty_like(previous)
        from_upper = torch.empty_like(previous)
        from_lower[:, 0] = infinity
        from_lower[:, 1:] = previous[:, :-1] + float(p1)
        from_upper[:, :-1] = previous[:, 1:] + float(p1)
        from_upper[:, -1] = infinity
        jump_cost = previous_min + adaptive_p2
        transition = torch.minimum(torch.minimum(previous, from_lower), torch.minimum(from_upper, jump_cost))
        current = source[row_index, :, :].to(dtype=torch.float32) + transition - previous_min
        output[row_index, :, :].add_(current)
        previous = current


def aggregate_sgm_axis_torch(cost_volume: Any, guide_gray: Any, axis: int, reverse: bool, p1: float, p2: float) -> Any:
    aggregated = torch.zeros(cost_volume.shape, dtype=torch.float32, device=cost_volume.device)
    accumulate_sgm_axis_torch(cost_volume, guide_gray, aggregated, axis=axis, reverse=reverse, p1=p1, p2=p2)
    return aggregated


def solve_sgm_from_cost_torch(
    cost_volume: Any,
    reference_gray: Any,
    *,
    min_disparity: int,
    device: str,
    p1: float | None = None,
    p2: float | None = None,
    return_numpy: bool = True,
    destructive: bool = False,
) -> tuple[Any, Any]:
    """对外暴露的 SGM 聚合：接受任意像素/网格级代价体并返回亚像素视差。

    输入 ``cost_volume`` 形状 ``[H, W, D]``（值越小越匹配），
    ``reference_gray`` 形状 ``[H, W]`` 用于边缘自适应惩罚。
    返回 ``(best, margin)``：均为形状 ``[H, W]`` 的 float32。
    """

    if torch is None:
        raise RuntimeError("solve_sgm_from_cost_torch requires PyTorch.")
    with torch.no_grad():
        if isinstance(cost_volume, torch.Tensor):
            cost_tensor = cost_volume.to(device=device)
            if not cost_tensor.is_floating_point():
                cost_tensor = cost_tensor.to(dtype=torch.float32)
            if not destructive and cost_tensor.data_ptr() == cost_volume.data_ptr():
                cost_tensor = cost_tensor.clone()
        else:
            cost_tensor = torch.as_tensor(np.asarray(cost_volume, dtype=np.float32), dtype=torch.float32, device=device)
        cost = normalize_cost_volume_torch(cost_tensor)
        if p1 is None or p2 is None:
            est_p1, est_p2 = estimate_sgm_penalties_torch(cost)
            p1 = float(est_p1) if p1 is None else float(p1)
            p2 = float(est_p2) if p2 is None else float(p2)
        if isinstance(reference_gray, torch.Tensor):
            reference = reference_gray.to(device=device, dtype=torch.float32)
        else:
            reference = torch.as_tensor(np.asarray(reference_gray, dtype=np.float32), dtype=torch.float32, device=device)
        aggregated = cost.to(dtype=torch.float32).clone()
        accumulate_sgm_axis_torch(cost, reference, aggregated, axis=1, reverse=False, p1=p1, p2=p2)
        accumulate_sgm_axis_torch(cost, reference, aggregated, axis=1, reverse=True, p1=p1, p2=p2)
        accumulate_sgm_axis_torch(cost, reference, aggregated, axis=0, reverse=False, p1=p1, p2=p2)
        accumulate_sgm_axis_torch(cost, reference, aggregated, axis=0, reverse=True, p1=p1, p2=p2)

        best_index = torch.argmin(aggregated, dim=2).to(dtype=torch.int64)
        best = (best_index.to(dtype=torch.float32) + float(max(0, int(min_disparity)))).to(dtype=torch.float32)
        if int(aggregated.shape[2]) > 1:
            sorted_cost = torch.topk(aggregated, k=2, dim=2, largest=False).values
            margin = sorted_cost[:, :, 1] - sorted_cost[:, :, 0]
            disparity_count = int(aggregated.shape[2])
            valid_subpixel = (best_index > 0) & (best_index < disparity_count - 1)
            center_cost = torch.gather(aggregated, 2, best_index[:, :, None])[:, :, 0]
            lower_cost = torch.gather(aggregated, 2, torch.clamp(best_index - 1, 0, disparity_count - 1)[:, :, None])[:, :, 0]
            upper_cost = torch.gather(aggregated, 2, torch.clamp(best_index + 1, 0, disparity_count - 1)[:, :, None])[:, :, 0]
            lower_cost = torch.where(valid_subpixel, lower_cost, torch.full_like(lower_cost, torch.inf))
            upper_cost = torch.where(valid_subpixel, upper_cost, torch.full_like(upper_cost, torch.inf))
            denominator = lower_cost - (2.0 * center_cost) + upper_cost
            stable = valid_subpixel & torch.isfinite(lower_cost) & torch.isfinite(center_cost) & torch.isfinite(upper_cost) & (torch.abs(denominator) > 1e-3)
            offset = torch.zeros_like(best)
            offset[stable] = 0.5 * (lower_cost[stable] - upper_cost[stable]) / denominator[stable]
            best = best + torch.clamp(offset, -0.5, 0.5)
        else:
            margin = torch.zeros_like(best)
        if return_numpy:
            return best.detach().cpu().numpy().astype(np.float32), margin.detach().cpu().numpy().astype(np.float32)
        return best.to(dtype=torch.float32), margin.to(dtype=torch.float32)


def solve_sgm_direction_torch(
    reference_gray: Any,
    target_gray: Any,
    config: O3Config,
    target_direction: str,
    min_disparity: int,
    max_disparity: int | None,
    device: str,
) -> tuple[Any, Any]:
    with torch.no_grad():
        raw_cost = build_matching_cost_volume_torch(
            reference_gray,
            target_gray,
            config,
            target_direction=target_direction,
            min_disparity=min_disparity,
            max_disparity=max_disparity,
            device=device,
        )
        clear_o3_torch_cache(device)
        result = solve_sgm_from_cost_torch(
            raw_cost,
            reference_gray,
            min_disparity=min_disparity,
            device=device,
            return_numpy=True,
            destructive=True,
        )
        del raw_cost
        clear_o3_torch_cache(device)
        return result


def aggregate_sgm_axis(cost_volume: Any, guide_gray: Any, axis: int, reverse: bool, p1: float, p2: float) -> Any:
    """沿单一方向执行带图像边缘自适应惩罚的 SGM 动态规划聚合。"""

    aggregated = np.zeros(np.asarray(cost_volume).shape, dtype=np.float32)
    accumulate_sgm_axis(cost_volume, guide_gray, aggregated, axis=axis, reverse=reverse, p1=p1, p2=p2)
    return aggregated


def accumulate_sgm_axis(cost_volume: Any, guide_gray: Any, output: Any, axis: int, reverse: bool, p1: float, p2: float) -> None:
    """把单方向 SGM 聚合结果直接累加到输出体，避免额外完整代价体峰值。"""

    source = np.asarray(cost_volume, dtype=np.float32)
    guide = np.asarray(guide_gray, dtype=np.float32)
    aggregated = np.asarray(output, dtype=np.float32)
    height, width, _ = source.shape
    infinity = np.float32(1e9)
    edge_scale = np.float32(8.0)
    minimum_jump = np.float32(max(float(p1) * 2.0, 1.0))

    if axis == 1:
        first_column = width - 1 if reverse else 0
        previous = source[:, first_column, :].copy()
        aggregated[:, first_column, :] += previous
        columns = range(first_column - 1, -1, -1) if reverse else range(first_column + 1, width)
        for column_index in columns:
            previous_column = column_index + 1 if reverse else column_index - 1
            edge_delta = np.abs(guide[:, column_index] - guide[:, previous_column])[:, None]
            adaptive_p2 = np.maximum(minimum_jump, float(p2) / (1.0 + (edge_delta / edge_scale)))
            previous_min = previous.min(axis=1, keepdims=True)
            from_lower = np.empty_like(previous)
            from_upper = np.empty_like(previous)
            from_lower[:, 0] = infinity
            from_lower[:, 1:] = previous[:, :-1] + p1
            from_upper[:, :-1] = previous[:, 1:] + p1
            from_upper[:, -1] = infinity
            jump_cost = np.broadcast_to(previous_min + adaptive_p2, previous.shape)
            transition = np.minimum.reduce((previous, from_lower, from_upper, jump_cost))
            current = source[:, column_index, :] + transition - previous_min
            aggregated[:, column_index, :] += current
            previous = current
        return

    first_row = height - 1 if reverse else 0
    previous = source[first_row, :, :].copy()
    aggregated[first_row, :, :] += previous
    rows = range(first_row - 1, -1, -1) if reverse else range(first_row + 1, height)
    for row_index in rows:
        previous_row = row_index + 1 if reverse else row_index - 1
        edge_delta = np.abs(guide[row_index, :] - guide[previous_row, :])[:, None]
        adaptive_p2 = np.maximum(minimum_jump, float(p2) / (1.0 + (edge_delta / edge_scale)))
        previous_min = previous.min(axis=1, keepdims=True)
        from_lower = np.empty_like(previous)
        from_upper = np.empty_like(previous)
        from_lower[:, 0] = infinity
        from_lower[:, 1:] = previous[:, :-1] + p1
        from_upper[:, :-1] = previous[:, 1:] + p1
        from_upper[:, -1] = infinity
        jump_cost = np.broadcast_to(previous_min + adaptive_p2, previous.shape)
        transition = np.minimum.reduce((previous, from_lower, from_upper, jump_cost))
        current = source[row_index, :, :] + transition - previous_min
        aggregated[row_index, :, :] += current
        previous = current


def solve_sgm_direction(
    reference_gray: Any,
    target_gray: Any,
    config: O3Config,
    target_direction: str,
    min_disparity: int = 0,
    max_disparity: int | None = None,
) -> tuple[Any, Any]:
    """对一个参考视角执行自实现四方向 SGM，返回视差与聚合置信边际。"""

    global _O3_TORCH_FALLBACK_REPORTED
    torch_device = resolve_o3_torch_device()
    if torch_device is not None:
        try:
            return solve_sgm_direction_torch(
                reference_gray,
                target_gray,
                config,
                target_direction=target_direction,
                min_disparity=min_disparity,
                max_disparity=max_disparity,
                device=torch_device,
            )
        except Exception as exc:
            clear_o3_torch_cache(torch_device)
            if not _O3_TORCH_FALLBACK_REPORTED:
                print(f"O3 CUDA SGM unavailable, falling back to NumPy SGM: {exc}", file=sys.stderr)
                _O3_TORCH_FALLBACK_REPORTED = True

    raw_cost = build_matching_cost_volume(
        reference_gray,
        target_gray,
        config,
        target_direction=target_direction,
        min_disparity=min_disparity,
        max_disparity=max_disparity,
    )
    return solve_sgm_from_cost(raw_cost, reference_gray, min_disparity=min_disparity)


def solve_sgm_from_cost(
    cost_volume: Any,
    reference_gray: Any,
    *,
    min_disparity: int,
    p1: float | None = None,
    p2: float | None = None,
) -> tuple[Any, Any]:
    """对外暴露的 NumPy 版 SGM 聚合：接受任意代价体并返回亚像素视差。"""

    cost = normalize_cost_volume(np.asarray(cost_volume, dtype=np.float32))
    if p1 is None or p2 is None:
        est_p1, est_p2 = estimate_sgm_penalties(cost)
        p1 = float(est_p1) if p1 is None else float(p1)
        p2 = float(est_p2) if p2 is None else float(p2)
    reference = np.asarray(reference_gray, dtype=np.float32)
    aggregated = cost.copy()
    accumulate_sgm_axis(cost, reference, aggregated, axis=1, reverse=False, p1=p1, p2=p2)
    accumulate_sgm_axis(cost, reference, aggregated, axis=1, reverse=True, p1=p1, p2=p2)
    accumulate_sgm_axis(cost, reference, aggregated, axis=0, reverse=False, p1=p1, p2=p2)
    accumulate_sgm_axis(cost, reference, aggregated, axis=0, reverse=True, p1=p1, p2=p2)

    best_index = np.argmin(aggregated, axis=2).astype(np.int32)
    best = (best_index + max(0, int(min_disparity))).astype(np.float32)
    if aggregated.shape[2] > 1:
        sorted_cost = np.partition(aggregated, kth=1, axis=2)
        margin = sorted_cost[:, :, 1] - sorted_cost[:, :, 0]
        _, _, disparity_count = aggregated.shape
        valid_subpixel = (best_index > 0) & (best_index < disparity_count - 1)
        center_cost = np.take_along_axis(aggregated, best_index[:, :, None], axis=2)[:, :, 0]
        lower_cost = np.take_along_axis(aggregated, np.clip(best_index - 1, 0, disparity_count - 1)[:, :, None], axis=2)[:, :, 0]
        upper_cost = np.take_along_axis(aggregated, np.clip(best_index + 1, 0, disparity_count - 1)[:, :, None], axis=2)[:, :, 0]
        lower_cost = np.where(valid_subpixel, lower_cost, np.inf).astype(np.float32)
        upper_cost = np.where(valid_subpixel, upper_cost, np.inf).astype(np.float32)
        denominator = lower_cost - (2.0 * center_cost) + upper_cost
        stable = valid_subpixel & np.isfinite(lower_cost) & np.isfinite(center_cost) & np.isfinite(upper_cost) & (np.abs(denominator) > 1e-3)
        offset = np.zeros_like(best, dtype=np.float32)
        offset[stable] = 0.5 * (lower_cost[stable] - upper_cost[stable]) / denominator[stable]
        best = best + np.clip(offset, -0.5, 0.5).astype(np.float32)
    else:
        margin = np.zeros(best.shape, dtype=np.float32)

    return best.astype(np.float32), margin.astype(np.float32)


def fill_disparity_with_local_weighted_median(
    disparity: Any,
    guide_gray: Any,
    radius: int,
    sigma_color: float,
    sigma_space: float,
    min_count: int,
) -> Any:
    """只用局部有效邻居填补小洞，避免行列全局传播造成大片糊连。"""

    source = np.asarray(disparity, dtype=np.float32)
    guide = np.asarray(guide_gray, dtype=np.float32)
    if radius <= 0:
        return source

    padded_disparity = np.pad(source, ((radius, radius), (radius, radius)), mode="edge")
    padded_guide = np.pad(guide, ((radius, radius), (radius, radius)), mode="edge")
    values: list[Any] = []
    weights: list[Any] = []
    counts = np.zeros_like(source, dtype=np.int16)
    color_scale = max(float(sigma_color), 1e-3)
    space_scale = max(float(sigma_space), 1e-3)

    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            shifted_disparity = padded_disparity[
                radius + offset_y : radius + offset_y + source.shape[0],
                radius + offset_x : radius + offset_x + source.shape[1],
            ]
            shifted_guide = padded_guide[
                radius + offset_y : radius + offset_y + source.shape[0],
                radius + offset_x : radius + offset_x + source.shape[1],
            ]
            valid = np.isfinite(shifted_disparity) & (shifted_disparity > 0)
            counts += valid.astype(np.int16)
            spatial_distance = float((offset_y * offset_y) + (offset_x * offset_x))
            spatial_weight = np.exp(-spatial_distance / (2.0 * space_scale * space_scale))
            color_delta = guide - shifted_guide
            color_weight = np.exp(-(color_delta * color_delta) / (2.0 * color_scale * color_scale)).astype(np.float32)
            values.append(shifted_disparity.astype(np.float32, copy=False))
            weights.append((spatial_weight * color_weight * valid.astype(np.float32)).astype(np.float32))

    value_stack = np.stack(values, axis=0)
    weight_stack = np.stack(weights, axis=0)
    order = np.argsort(value_stack, axis=0)
    sorted_values = np.take_along_axis(value_stack, order, axis=0)
    sorted_weights = np.take_along_axis(weight_stack, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    total = cumulative[-1]
    median_index = np.argmax(cumulative >= (0.5 * total[None, :, :]), axis=0)
    filled = np.take_along_axis(sorted_values, median_index[None, :, :], axis=0)[0]
    fill_mask = (source <= 0) & (counts >= int(min_count)) & (total > 0)
    return np.where(fill_mask, filled, source).astype(np.float32)


def compute_sgm_disparity(
    left_gray: Any,
    right_gray: Any,
    config: O3Config,
    min_disparity: int = 0,
    max_disparity: int | None = None,
    feature_stereo_matches: int | None = None,
    feature_seed_disparity: Any | None = None,
    feature_seed_mask: Any | None = None,
) -> Any:
    """使用自实现左右一致性 SGM 生成稳定且边缘清晰的 O3 视差。"""

    left = np.asarray(left_gray, dtype=np.float32)
    sparse_feature_support = feature_stereo_matches is not None and feature_stereo_matches < 64
    support_tolerance = 1.5 if sparse_feature_support else 2.5
    support_min_count = 8 if sparse_feature_support else 5
    relaxed_consistency_limit = max(2.5 if sparse_feature_support else 3.0, float(config.consistency_threshold), float(config.disp12_max_diff))
    relaxed_support_tolerance = max(3.0, support_tolerance)
    relaxed_support_min_count = 5 if sparse_feature_support else 3
    left_disparity, left_margin = solve_sgm_direction(
        left_gray,
        right_gray,
        config,
        target_direction="negative",
        min_disparity=min_disparity,
        max_disparity=max_disparity,
    )
    right_disparity, _ = solve_sgm_direction(
        right_gray,
        left_gray,
        config,
        target_direction="positive",
        min_disparity=min_disparity,
        max_disparity=max_disparity,
    )

    dense = np.where(np.isfinite(left_disparity) & (left_disparity > 0), left_disparity, 0.0).astype(np.float32)
    seed_prior = np.zeros_like(dense, dtype=np.float32)
    seed_prior_mask = np.zeros_like(dense, dtype=bool)
    seed_near_mask = np.zeros_like(dense, dtype=bool)
    seed_global_mask = np.ones_like(dense, dtype=bool)
    seed_count = 0
    if feature_seed_disparity is not None and feature_seed_mask is not None:
        seed_source = np.asarray(feature_seed_disparity, dtype=np.float32)
        seed_mask = np.asarray(feature_seed_mask, dtype=bool) & np.isfinite(seed_source) & (seed_source > 0)
        if seed_source.shape == dense.shape and bool(np.any(seed_mask)):
            seed_values = seed_source[seed_mask].astype(np.float32)
            seed_count = int(seed_values.size)
            height, width = dense.shape
            seed_spacing = float(np.sqrt((height * width) / max(1, seed_count)))
            local_seed_distance = float(np.clip(seed_spacing * (0.85 if sparse_feature_support else 0.65), 28.0, 96.0))
            distance_result: Any = ndimage.distance_transform_edt(~seed_mask, return_distances=True, return_indices=True)
            seed_distances = np.asarray(distance_result[0], dtype=np.float32)
            seed_indices = np.asarray(distance_result[1], dtype=np.int64)
            seed_prior = seed_source[seed_indices[0], seed_indices[1]].astype(np.float32)
            seed_prior_mask = np.isfinite(seed_prior) & (seed_prior > 0)
            seed_near_mask = seed_prior_mask & (seed_distances <= local_seed_distance)
            if seed_count >= 8:
                low = float(np.percentile(seed_values, 2.0))
                high = float(np.percentile(seed_values, 98.0))
                seed_span = max(1.0, high - low)
                range_margin = max(10.0 if sparse_feature_support else 14.0, seed_span * (0.35 if sparse_feature_support else 0.45))
                seed_global_mask = (dense >= low - range_margin) & (dense <= high + range_margin)
    consistency_limit = max(float(config.consistency_threshold), float(config.disp12_max_diff), 1.5)
    consistency_error = left_right_consistency_error(dense, right_disparity)
    consistent_mask = consistency_error <= consistency_limit
    margin_values = left_margin[np.isfinite(left_margin) & (left_margin > 0)]
    margin_low = float(np.percentile(margin_values, 5.0)) if margin_values.size else 0.0
    prior_tolerance = max(5.0 if sparse_feature_support else 6.5, float(config.disp12_max_diff) * 3.0)
    local_prior_consistent = (~seed_near_mask) | (np.abs(dense - seed_prior) <= prior_tolerance)
    reliable_mask = (dense > 0) & consistent_mask & (left_margin >= margin_low) & local_prior_consistent
    if sparse_feature_support and seed_count >= 8:
        reliable_mask &= seed_global_mask

    texture_x = np.abs(compute_horizontal_gradient(left))
    texture_y = np.abs(compute_horizontal_gradient(left.T).T)
    texture = texture_x + texture_y
    texture_values = texture[np.isfinite(texture)]
    edge_floor = float(np.percentile(texture_values, 90.0)) if texture_values.size else 0.0
    edge_mask = texture >= edge_floor

    support_mask = local_disparity_support_mask(
        dense,
        reliable_mask,
        radius=2,
        tolerance=support_tolerance,
        min_count=support_min_count,
    )
    best = np.where(support_mask, dense, 0.0).astype(np.float32)
    best = filter_speckles(best, 32 if not sparse_feature_support else 48, max(2, config.speckle_range // 3))

    relaxed_prior_tolerance = max(prior_tolerance + 3.0, 8.0)
    relaxed_prior_consistent = (~seed_near_mask) | (np.abs(dense - seed_prior) <= relaxed_prior_tolerance)
    relaxed_reliable_mask = (dense > 0) & (consistency_error <= relaxed_consistency_limit) & (left_margin >= margin_low) & relaxed_prior_consistent
    if sparse_feature_support and seed_count >= 8:
        relaxed_reliable_mask &= seed_global_mask
    relaxed_support_mask = local_disparity_support_mask(
        dense,
        relaxed_reliable_mask,
        radius=2,
        tolerance=relaxed_support_tolerance,
        min_count=relaxed_support_min_count,
    )
    relaxed = np.where(relaxed_support_mask, dense, 0.0).astype(np.float32)
    relaxed = filter_speckles(relaxed, 24 if not sparse_feature_support else 32, max(2, config.speckle_range // 3))
    best = np.where(best > 0, best, relaxed).astype(np.float32)

    if bool(np.any(seed_near_mask)):
        recovery_candidates = (
            (best <= 0)
            & (dense > 0)
            & seed_near_mask
            & (np.abs(dense - seed_prior) <= prior_tolerance)
            & (consistency_error <= relaxed_consistency_limit)
            & (left_margin >= margin_low)
        )
        recovery_mask = local_disparity_support_mask(
            dense,
            recovery_candidates,
            radius=2,
            tolerance=relaxed_support_tolerance,
            min_count=relaxed_support_min_count,
        )
        best = np.where(recovery_mask, dense, best).astype(np.float32)

    if config.median_filter_size > 1:
        positive_mask = best > 0
        filtered = median_filter_2d(best, config.median_filter_size)
        best = np.where(positive_mask & ~edge_mask, filtered, best).astype(np.float32)

    filtered = joint_weighted_median_filter_disparity(
        best,
        left_gray,
        radius=1,
        sigma_color=8.0,
        sigma_space=1.2,
    )
    best = np.where(best > 0, filtered, 0.0).astype(np.float32)

    best = fill_short_horizontal_disparity_gaps(
        best,
        left_gray,
        max_gap=48 if sparse_feature_support else 64,
        max_disparity_delta=3.0 if sparse_feature_support else 4.0,
        max_intensity_delta=24.0 if sparse_feature_support else 28.0,
    )
    best = fill_short_vertical_disparity_gaps(
        best,
        left_gray,
        max_gap=16 if sparse_feature_support else 24,
        max_disparity_delta=3.0 if sparse_feature_support else 4.0,
        max_intensity_delta=24.0 if sparse_feature_support else 28.0,
    )
    if config.median_filter_size > 1:
        positive_mask = best > 0
        filtered = median_filter_2d(best, config.median_filter_size)
        best = np.where(positive_mask, filtered, 0.0).astype(np.float32)

    best = np.where(np.isfinite(best) & (best > 0), best, 0.0).astype(np.float32)
    invalid_margin = max(config.block_size // 2, config.census_window_size // 2)
    if invalid_margin > 0:
        best[:invalid_margin, :] = 0.0
        best[-invalid_margin:, :] = 0.0
        best[:, :invalid_margin] = 0.0
        best[:, -invalid_margin:] = 0.0

    return np.where(best > 0, best, 0.0).astype(np.float32)


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


def propagate_sift_seed_disparity(seed_disparity: Any, seed_mask: Any, guide_gray: Any, config: O3Config) -> Any:
    """把 SIFT 匹配种子限定在局部邻域内传播，避免跨整图行列插值。"""

    seeds = np.asarray(seed_disparity, dtype=np.float32)
    mask = np.asarray(seed_mask, dtype=bool)
    if not bool(np.any(mask)):
        return np.zeros_like(seeds, dtype=np.float32)

    height, width = seeds.shape
    max_distance = float(np.clip(min(height, width) * 0.12, 48.0, 140.0))
    distance_result: Any = ndimage.distance_transform_edt(~mask, return_distances=True, return_indices=True)
    distances = np.asarray(distance_result[0], dtype=np.float32)
    indices = np.asarray(distance_result[1], dtype=np.int64)
    propagated = seeds[indices[0], indices[1]].astype(np.float32)
    disparity = np.where(distances <= max_distance, propagated, 0.0).astype(np.float32)

    disparity = joint_weighted_median_filter_disparity(
        disparity,
        guide_gray,
        radius=2,
        sigma_color=12.0,
        sigma_space=2.0,
    )
    if config.median_filter_size > 1:
        positive_mask = disparity > 0
        filtered = median_filter_2d(disparity, config.median_filter_size)
        disparity = np.where(positive_mask, filtered, 0.0).astype(np.float32)
    return np.where(np.isfinite(disparity) & (disparity > 0), disparity, 0.0).astype(np.float32)


def filter_sift_seed_outliers(seed_disparity: Any, seed_mask: Any) -> tuple[Any, Any]:
    """用局部 SIFT 种子一致性剔除明显离群的错误匹配。"""

    seeds = np.asarray(seed_disparity, dtype=np.float32)
    mask = np.asarray(seed_mask, dtype=bool) & np.isfinite(seeds) & (seeds > 0)
    seed_rows, seed_columns = np.nonzero(mask)
    if seed_rows.size < 5:
        return seeds.copy(), mask.copy()

    seed_points = np.column_stack((seed_columns.astype(np.float32), seed_rows.astype(np.float32)))
    seed_values = seeds[seed_rows, seed_columns].astype(np.float32)
    neighbor_count = min(8, seed_rows.size)
    tree = cKDTree(seed_points)
    _, neighbor_indices = tree.query(seed_points, k=neighbor_count)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]

    neighbor_values = seed_values[neighbor_indices]
    local_median = np.median(neighbor_values, axis=1)
    local_mad = np.median(np.abs(neighbor_values - local_median[:, None]), axis=1)
    robust_sigma = (1.4826 * local_mad) + 1e-3
    keep_threshold = np.maximum(8.0, (4.0 * robust_sigma) + 2.0)
    keep_seed = np.abs(seed_values - local_median) <= keep_threshold

    if int(keep_seed.sum()) < max(3, int(seed_rows.size * 0.35)):
        relaxed_threshold = np.maximum(12.0, (6.0 * robust_sigma) + 2.0)
        keep_seed = np.abs(seed_values - local_median) <= relaxed_threshold

    filtered = np.zeros_like(seeds, dtype=np.float32)
    filtered_mask = np.zeros_like(mask, dtype=bool)
    filtered[seed_rows[keep_seed], seed_columns[keep_seed]] = seed_values[keep_seed]
    filtered_mask[seed_rows[keep_seed], seed_columns[keep_seed]] = True
    return filtered, filtered_mask


def add_sift_boundary_support_points(seed_points: Any, seed_values: Any, image_shape: tuple[int, int]) -> tuple[Any, Any]:
    """用最近真实 SIFT 种子派生边界支撑点，避免三角网只覆盖种子凸包内部。"""

    points = np.asarray(seed_points, dtype=np.float64)
    values = np.asarray(seed_values, dtype=np.float32)
    if values.size == 0:
        return points, values

    height, width = image_shape
    seed_spacing = float(np.sqrt((height * width) / max(1, int(values.size))))
    step = max(48, int(round(seed_spacing * 1.25)))
    border_points: list[tuple[float, float]] = []
    for column_index in range(0, width, step):
        border_points.append((float(column_index), 0.0))
        border_points.append((float(column_index), float(height - 1)))
    for row_index in range(0, height, step):
        border_points.append((0.0, float(row_index)))
        border_points.append((float(width - 1), float(row_index)))
    border_points.extend(
        (
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (0.0, float(height - 1)),
            (float(width - 1), float(height - 1)),
        )
    )

    unique_border_points = np.asarray(sorted(set(border_points)), dtype=np.float64)
    tree = cKDTree(points)
    _, nearest_indices = tree.query(unique_border_points, k=1)
    boundary_values = values[np.asarray(nearest_indices, dtype=np.int64)]
    augmented_points = np.vstack((points, unique_border_points))
    augmented_values = np.concatenate((values, boundary_values.astype(np.float32)))
    return augmented_points, augmented_values


def inpaint_triangulated_disparity_holes(
    disparity: Any,
    guide_gray: Any,
    max_distance: float,
    iterations: int,
) -> Any:
    """对三角网仍未覆盖的小/中型空洞做边缘感知调和补全。"""

    source = np.asarray(disparity, dtype=np.float32)
    guide = np.asarray(guide_gray, dtype=np.float32)
    valid = np.isfinite(source) & (source > 0)
    if not bool(np.any(valid)) or bool(np.all(valid)):
        return np.where(valid, source, 0.0).astype(np.float32)

    distance_result: Any = ndimage.distance_transform_edt(~valid, return_distances=True, return_indices=True)
    distances = np.asarray(distance_result[0], dtype=np.float32)
    indices = np.asarray(distance_result[1], dtype=np.int64)
    fillable = (~valid) & (distances <= float(max_distance))
    if not bool(np.any(fillable)):
        return np.where(valid, source, 0.0).astype(np.float32)

    filled = source.copy()
    nearest_values = source[indices[0], indices[1]]
    filled[fillable] = nearest_values[fillable]
    active = fillable
    available = valid | fillable
    color_scale = 18.0

    for _ in range(max(1, int(iterations))):
        total = np.zeros_like(filled, dtype=np.float32)
        weights = np.zeros_like(filled, dtype=np.float32)

        vertical_weight = np.exp(-((guide[1:, :] - guide[:-1, :]) ** 2) / (2.0 * color_scale * color_scale)).astype(np.float32)
        top_available = available[:-1, :]
        bottom_available = available[1:, :]
        total[1:, :] += vertical_weight * filled[:-1, :] * top_available.astype(np.float32)
        weights[1:, :] += vertical_weight * top_available.astype(np.float32)
        total[:-1, :] += vertical_weight * filled[1:, :] * bottom_available.astype(np.float32)
        weights[:-1, :] += vertical_weight * bottom_available.astype(np.float32)

        horizontal_weight = np.exp(-((guide[:, 1:] - guide[:, :-1]) ** 2) / (2.0 * color_scale * color_scale)).astype(np.float32)
        left_available = available[:, :-1]
        right_available = available[:, 1:]
        total[:, 1:] += horizontal_weight * filled[:, :-1] * left_available.astype(np.float32)
        weights[:, 1:] += horizontal_weight * left_available.astype(np.float32)
        total[:, :-1] += horizontal_weight * filled[:, 1:] * right_available.astype(np.float32)
        weights[:, :-1] += horizontal_weight * right_available.astype(np.float32)

        update_mask = active & (weights > 0)
        estimates = np.divide(total, np.maximum(weights, 1e-6), out=filled.copy(), where=weights > 0)
        filled[update_mask] = (0.55 * filled[update_mask]) + (0.45 * estimates[update_mask])

    result_mask = valid | fillable
    return np.where(result_mask & np.isfinite(filled) & (filled > 0), filled, 0.0).astype(np.float32)


def triangulate_sift_seed_disparity(seed_disparity: Any, seed_mask: Any, guide_gray: Any, config: O3Config) -> Any:
    """用真实 SIFT 匹配种子建立 Delaunay 三角网，只在种子凸包内插值。"""

    filtered_seeds, filtered_mask = filter_sift_seed_outliers(seed_disparity, seed_mask)
    seed_rows, seed_columns = np.nonzero(filtered_mask)
    if seed_rows.size < 3:
        return np.zeros_like(filtered_seeds, dtype=np.float32)

    height, width = filtered_seeds.shape
    seed_points = np.column_stack((seed_columns.astype(np.float64), seed_rows.astype(np.float64)))
    seed_values = filtered_seeds[seed_rows, seed_columns].astype(np.float32)
    try:
        triangulation = Delaunay(seed_points)
    except QhullError:
        return np.zeros_like(filtered_seeds, dtype=np.float32)

    triangle_points = seed_points[triangulation.simplices]
    edge_a = np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 1], axis=1)
    edge_b = np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 2], axis=1)
    edge_c = np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 0], axis=1)
    max_triangle_edge = np.maximum.reduce((edge_a, edge_b, edge_c))
    typical_seed_spacing = float(np.sqrt((height * width) / max(1, seed_rows.size)))
    max_allowed_edge = float(np.clip(typical_seed_spacing * 5.5, 128.0, 520.0))

    disparity = np.zeros((height, width), dtype=np.float32)
    total_pixels = height * width
    chunk_size = 250_000
    grid_columns = np.arange(width, dtype=np.float64)

    for start_index in range(0, total_pixels, chunk_size):
        end_index = min(start_index + chunk_size, total_pixels)
        flat_indices = np.arange(start_index, end_index, dtype=np.int64)
        query_rows = (flat_indices // width).astype(np.float64)
        query_columns = grid_columns[(flat_indices % width).astype(np.int64)]
        query_points = np.column_stack((query_columns, query_rows))
        simplex_indices = triangulation.find_simplex(query_points)
        inside_mask = simplex_indices >= 0
        if not bool(np.any(inside_mask)):
            continue

        valid_flat_indices = flat_indices[inside_mask]
        valid_simplex_indices = simplex_indices[inside_mask]
        compact_triangle_mask = max_triangle_edge[valid_simplex_indices] <= max_allowed_edge
        if not bool(np.any(compact_triangle_mask)):
            continue

        valid_flat_indices = valid_flat_indices[compact_triangle_mask]
        valid_simplex_indices = valid_simplex_indices[compact_triangle_mask]
        valid_points = query_points[inside_mask][compact_triangle_mask]
        transform = triangulation.transform[valid_simplex_indices]
        barycentric_head = np.einsum("ijk,ik->ij", transform[:, :2, :], valid_points - transform[:, 2, :])
        barycentric_weights = np.column_stack(
            (
                barycentric_head[:, 0],
                barycentric_head[:, 1],
                1.0 - barycentric_head.sum(axis=1),
            )
        )
        vertex_values = seed_values[triangulation.simplices[valid_simplex_indices]]
        interpolated_values = np.sum(vertex_values * barycentric_weights.astype(np.float32), axis=1)
        disparity.reshape(-1)[valid_flat_indices] = interpolated_values.astype(np.float32)

    return np.where(np.isfinite(disparity) & (disparity > 0), disparity, 0.0).astype(np.float32)


def compute_sift_driven_disparity(left_gray: Any, right_gray: Any, config: O3Config) -> tuple[Any, dict[str, int]]:
    """以左右图 SIFT 特征描述符匹配为直接深度来源生成视差图。"""

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
    disparity = triangulate_sift_seed_disparity(seed_disparity, seed_mask, left_gray, config)

    invalid_margin = max(config.block_size // 2, 1)
    if invalid_margin > 0:
        disparity[:invalid_margin, :] = 0.0
        disparity[-invalid_margin:, :] = 0.0
        disparity[:, :invalid_margin] = 0.0
        disparity[:, -invalid_margin:] = 0.0

    disparity = filter_speckles(disparity, config.speckle_window_size, config.speckle_range)
    if config.median_filter_size > 1:
        positive_mask = disparity > 0
        filtered = median_filter_2d(disparity, config.median_filter_size)
        disparity = np.where(positive_mask, filtered, 0.0).astype(np.float32)

    return disparity.astype(np.float32), stats


def collect_sift_stereo_stats(left_gray: Any, right_gray: Any, config: O3Config) -> dict[str, Any]:
    """运行手写 SIFT 生成器并统计满足立体几何约束的匹配数量。"""

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
        "seed_count": 0,
    }
    if left_descriptors is None or right_descriptors is None or len(left_descriptors) == 0 or len(right_descriptors) == 0:
        return stats

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
    if stereo_matches:
        seed_disparity, seed_mask = build_seed_disparity(
            left_keypoints,
            right_keypoints,
            stereo_matches,
            left_gray.shape,
        )
        seed_disparity, seed_mask = filter_sift_seed_outliers(seed_disparity, seed_mask)
        stats["seed_disparity"] = seed_disparity
        stats["seed_mask"] = seed_mask
        stats["seed_count"] = int(np.count_nonzero(seed_mask))
    return stats


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
    if scene_name is None:
        pdf_required_paths = (
            disparity_dir / "dep_pipeline.jpg",
            analysis_dir / "example_1.jpg",
            analysis_dir / "example_2.jpg",
            analysis_dir / "example_3.jpg",
            metrics_file.parent / "disparity.csv",
        )
        missing_pdf = [str(path) for path in pdf_required_paths if not path.exists()]
        if missing_pdf:
            missing_any = True
            print("[MISSING] PDF-required O3 files: " + ", ".join(missing_pdf))
        else:
            print("[OK] PDF-required O3 files present")

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
    sgm_backend = "torch_cuda" if resolve_o3_torch_device() is not None else "numpy_cpu"
    print(f"O3 SGM backend: {sgm_backend}")
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
    example_payloads: dict[str, dict[str, Any]] = {}
    for scene_dir in scenes:
        left_gray = load_gray(scene_dir / "im0.png")
        right_gray = load_gray(scene_dir / "im1.png")
        min_disparity, max_disparity = estimate_scene_disparity_bounds(scene_dir, left_gray.shape, config)
        scene_config = replace(config, num_disparities=max(config.num_disparities, max_disparity + 1))
        stats = collect_sift_stereo_stats(left_gray, right_gray, scene_config)
        disparity = compute_sgm_disparity(
            left_gray,
            right_gray,
            scene_config,
            min_disparity=min_disparity,
            max_disparity=max_disparity,
            feature_stereo_matches=stats["stereo_matches"],
            feature_seed_disparity=stats.get("seed_disparity"),
            feature_seed_mask=stats.get("seed_mask"),
        )
        generator_name = "O3 manual-SIFT-guided census-gradient SGM disparity"
        disparity_mask = disparity > 0

        disparity_scene_dir = config.disparity_dir / scene_dir.name
        analysis_scene_dir = config.analysis_dir / scene_dir.name
        disparity_scene_dir.mkdir(parents=True, exist_ok=True)
        analysis_scene_dir.mkdir(parents=True, exist_ok=True)

        write_pfm(disparity_scene_dir / "disp0.pfm", disparity)
        disparity_preview = colorize_disparity_depth_map(disparity, disparity_mask)
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
                f"sift_seed_prior_points: {stats.get('seed_count', 0)}",
                f"max_features: {scene_config.max_features}",
                f"contrast_threshold: {scene_config.contrast_threshold}",
                f"ratio_test: {scene_config.ratio_test}",
                f"base_num_disparities: {config.num_disparities}",
                f"min_disparity: {min_disparity}",
                f"max_disparity: {max_disparity}",
                f"search_disparities: {max_disparity - min_disparity + 1}",
                f"max_vertical_offset: {scene_config.max_vertical_offset}",
                f"block_size: {scene_config.block_size}",
                f"speckle_window_size: {scene_config.speckle_window_size}",
                f"speckle_range: {scene_config.speckle_range}",
                f"fill_invalid_passes: {scene_config.fill_invalid_passes}",
                f"median_filter_size: {scene_config.median_filter_size}",
                "depth_source: manual_sift_guided_dense_sgm_stereo",
                "sgm_disparity_used: yes",
                "postprocess: sift_seed_prior_guided_sgm_left_right_consistency_edge_aware_cleanup",
                "png_preview: rgb_depth_colormap_invalid_black",
                "round_local_gap_fill_used: no",
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
        example_payloads[scene_dir.name] = {
            "left_gray": np.asarray(left_gray, dtype=np.uint8).copy(),
            "disparity_preview": np.asarray(disparity_preview, dtype=np.uint8).copy(),
            "metrics": dict(metrics),
        }

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
            f"disparity_range={min_disparity}-{max_disparity}, valid_disparity_pixels={metrics['valid_disparity_pixels']}, "
            f"mae={mae_text}, rmse={rmse_text}, bad_1px={bad_text})"
        )
        clear_o3_torch_cache()

    write_metrics(config.metrics_file, metric_rows)
    write_o3_pdf_assets(config, example_payloads, metric_rows)
    example_count = min(3, len(example_payloads))
    print(f"Wrote O3 metrics summary: {config.metrics_file}")
    print(f"Wrote O3 PDF disparity table: {config.metrics_file.parent / 'disparity.csv'}")
    print(f"Wrote O3 PDF pipeline image: {config.disparity_dir / 'dep_pipeline.jpg'}")
    if example_count > 0:
        print(f"Wrote O3 PDF examples: {config.analysis_dir / 'example_1.jpg'} ... {config.analysis_dir / f'example_{example_count}.jpg'}")
    return 0


def parse_o3_entry_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "configs" / "dataset_paths.example.yaml"
    parser = argparse.ArgumentParser(description="Run objective O3 from codes/o3.py as a direct entry script.")
    parser.add_argument("--config", type=Path, default=default_config, help=f"Path to YAML config. Default: {default_config}")
    parser.add_argument("--profile", default="local", help="Config profile name. Default: local")
    parser.add_argument("--max-scenes", type=int, default=None, help="Maximum number of scenes to process. Use 0 for discovery-only.")
    parser.add_argument("--scene-name", default=None, help="Run only one scene by exact directory name.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery/config info without writing outputs.")
    parser.add_argument("--validate-results", action="store_true", help="Validate existing outputs without rewriting them.")
    return parser.parse_args(argv)


def run_o3_entry(argv: list[str] | None = None) -> int:
    args = parse_o3_entry_args(argv)
    try:
        project_config = load_config(args.config, args.profile)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.validate_results:
        return validate_results(
            project_config.o3.disparity_dir,
            project_config.o3.analysis_dir,
            project_config.o3.metrics_file,
            args.scene_name,
        )

    return run(
        project_config.repo_root,
        project_config.middlebury_root,
        project_config.o3,
        args.max_scenes,
        args.dry_run,
        args.scene_name,
    )


if __name__ == '__main__':
    raise SystemExit(run_o3_entry())
