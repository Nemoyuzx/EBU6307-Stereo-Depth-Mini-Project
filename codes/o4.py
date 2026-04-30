from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from common import (
    discover_scenes,
    evaluate_disparity,
    filter_scene_dirs,
    load_gray,
    normalize_for_preview,
    write_png,
    write_scene_text,
)
from config import O4Config
from o3 import (
    average_pool_gray,
    box_filter_sum,
    compute_horizontal_gradient,
    fill_disparity_with_local_weighted_median,
    fill_invalid_disparity,
    joint_weighted_median_filter_disparity,
    left_right_consistency_mask,
    median_filter_2d,
)
from o4_torch import (
    average_pool_gray_torch,
    build_token_descriptors_torch,
    build_token_ground_truth_torch,
    collect_o4_training_samples_torch,
    encode_o4_descriptors_torch,
    extract_baseline_patch_tokens_torch,
    fill_invalid_disparity_torch,
    left_right_consistency_mask_torch,
    median_filter_2d_torch,
    predict_token_disparity_torch,
    refine_token_disparity_torch,
    resolve_torch_backend,
    soft_argmax_disparity_torch,
    train_o4_torch_model,
    upsample_token_grid_torch,
)
from pfm import read_pfm, write_pfm


MetricValue = str | float | int
MetricRow = dict[str, MetricValue]


@dataclass
class O4ModelState:
    backend: str
    device: str
    trained: bool
    projection: Any | None = None
    torch_model: Any | None = None


def read_metrics(metrics_file: Path) -> list[MetricRow]:
    """读取 O4 历史指标。"""
    if not metrics_file.exists():
        return []

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "scene": row.get("scene", ""),
                "fold": row.get("fold", ""),
                "token_grid": row.get("token_grid", ""),
                "valid_disparity_pixels": row.get("valid_disparity_pixels", ""),
                "valid_ground_truth_pixels": row.get("valid_ground_truth_pixels", ""),
                "mae": row.get("mae", ""),
                "rmse": row.get("rmse", ""),
                "bad_1px": row.get("bad_1px", ""),
                "mean_confidence": row.get("mean_confidence", ""),
                "estimated_working_set_mb": row.get("estimated_working_set_mb", ""),
            }
            for row in reader
            if row.get("scene")
        ]


def write_metrics(metrics_file: Path, rows: list[MetricRow]) -> None:
    """按场景合并并回写 O4 场景级指标。"""
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
                "fold",
                "token_grid",
                "valid_disparity_pixels",
                "valid_ground_truth_pixels",
                "mae",
                "rmse",
                "bad_1px",
                "mean_confidence",
                "estimated_working_set_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)


def write_fold_metrics(metrics_file: Path, rows: list[MetricRow]) -> None:
    """写出按交叉验证 fold 汇总的统计表。"""
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "scene_count",
                "scenes_with_ground_truth",
                "mean_mae",
                "mean_rmse",
                "mean_bad_1px",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def box_filter_mean(image: Any, kernel_size: int) -> Any:
    """基于 box_filter_sum 计算局部均值。"""

    if kernel_size <= 1:
        return np.asarray(image, dtype=np.float32)
    radius = kernel_size // 2
    area = float((radius * 2 + 1) ** 2)
    return box_filter_sum(image, radius) / area


def build_token_descriptors(image: Any, patch_size: int, context_window_size: int) -> tuple[Any, int, int]:
    """把灰度图编码成 patch token 描述子，供 numpy 后端使用。"""

    source = np.asarray(image, dtype=np.float32) / 255.0
    gradient_x = compute_horizontal_gradient(source)
    gradient_y = compute_horizontal_gradient(source.T).T
    context = box_filter_mean(source, context_window_size)
    channels = [source, gradient_x, gradient_y, context]
    height, width = source.shape
    token_height = height // patch_size
    token_width = width // patch_size
    cropped_height = token_height * patch_size
    cropped_width = token_width * patch_size
    descriptor_parts: list[Any] = []
    for channel in channels:
        cropped = channel[:cropped_height, :cropped_width]
        tokens = cropped.reshape(token_height, patch_size, token_width, patch_size)
        tokens = tokens.transpose(0, 2, 1, 3).reshape(token_height, token_width, patch_size * patch_size)
        descriptor_parts.append(tokens)

    intensity_tokens = descriptor_parts[0]
    descriptor_parts.append(intensity_tokens.mean(axis=2, keepdims=True))
    descriptor_parts.append(intensity_tokens.std(axis=2, keepdims=True))
    descriptors = np.concatenate(descriptor_parts, axis=2).astype(np.float32)
    descriptors -= descriptors.mean(axis=2, keepdims=True)
    descriptors /= np.maximum(descriptors.std(axis=2, keepdims=True), 1e-6)
    return descriptors, cropped_height, cropped_width


def extract_baseline_patch_tokens(image: Any, patch_size: int) -> tuple[Any, int, int]:
    """提取最基础的 patch token 并做归一化。"""

    source = np.asarray(image, dtype=np.float32)
    height, width = source.shape
    token_height = height // patch_size
    token_width = width // patch_size
    cropped_height = token_height * patch_size
    cropped_width = token_width * patch_size
    cropped = source[:cropped_height, :cropped_width]

    tokens = cropped.reshape(token_height, patch_size, token_width, patch_size)
    tokens = tokens.transpose(0, 2, 1, 3).reshape(token_height, token_width, patch_size * patch_size)
    tokens = tokens.astype(np.float32) / 255.0
    tokens -= tokens.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(tokens, axis=2, keepdims=True)
    tokens = np.divide(tokens, np.maximum(norms, 1e-6), out=np.zeros_like(tokens), where=norms > 0)
    return tokens, cropped_height, cropped_width


def project_token_descriptors(descriptors: Any, projection: Any | None) -> Any:
    """把描述子投影到训练后的低维空间。"""

    source = np.asarray(descriptors, dtype=np.float32)
    if projection is None:
        projected = source
    else:
        projected = np.tensordot(source, np.asarray(projection, dtype=np.float32), axes=([2], [0]))
    norms = np.linalg.norm(projected, axis=2, keepdims=True)
    return np.divide(projected, np.maximum(norms, 1e-6), out=np.zeros_like(projected), where=norms > 0)


def predict_token_disparity(
    left_tokens: Any,
    right_tokens: Any,
    min_disparity: int,
    max_disparity: int,
    target_direction: str = "negative",
) -> tuple[Any, Any, Any]:

    token_height, token_width, _ = left_tokens.shape
    best_scores = np.full((token_height, token_width), -np.inf, dtype=np.float32)
    second_scores = np.full((token_height, token_width), -np.inf, dtype=np.float32)
    best_disparities = np.zeros((token_height, token_width), dtype=np.int32)
    all_scores = np.full((max_disparity - min_disparity + 1, token_height, token_width), -np.inf, dtype=np.float32)

    for score_index, disparity in enumerate(range(min_disparity, max_disparity + 1)):
        current_scores = np.full((token_height, token_width), -np.inf, dtype=np.float32)
        if disparity == 0:
            current_scores = np.sum(left_tokens * right_tokens, axis=2, dtype=np.float32)
        elif target_direction == "positive" and disparity < token_width:
            current_scores[:, : token_width - disparity] = np.sum(
                left_tokens[:, : token_width - disparity, :] * right_tokens[:, disparity:, :],
                axis=2,
                dtype=np.float32,
            )
        elif disparity < token_width:
            current_scores[:, disparity:] = np.sum(
                left_tokens[:, disparity:, :] * right_tokens[:, : token_width - disparity, :],
                axis=2,
                dtype=np.float32,
            )
        all_scores[score_index] = current_scores

        replace_mask = current_scores > best_scores
        second_scores = np.where(replace_mask, best_scores, np.maximum(second_scores, current_scores))
        best_scores = np.where(replace_mask, current_scores, best_scores)
        best_disparities = np.where(replace_mask, disparity, best_disparities)

    confidence = np.where(np.isfinite(best_scores), best_scores - second_scores, 0.0).astype(np.float32)
    confidence = np.where(np.isfinite(confidence), confidence, 0.0)
    return best_disparities, confidence, all_scores


def refine_token_disparity(best_disparity: Any, all_scores: Any, min_disparity: int, max_disparity: int) -> Any:
    """对 token 级最佳视差做二次曲线细化。"""

    refined = np.asarray(best_disparity, dtype=np.float32).copy()
    if max_disparity <= min_disparity:
        return refined

    score_index = np.rint(refined).astype(np.int32) - int(min_disparity)
    valid = (score_index > 0) & (score_index < all_scores.shape[0] - 1)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return refined

    center = all_scores[score_index[rows, cols], rows, cols]
    left = all_scores[score_index[rows, cols] - 1, rows, cols]
    right = all_scores[score_index[rows, cols] + 1, rows, cols]
    denominator = left - (2.0 * center) + right
    stable = np.isfinite(center) & np.isfinite(left) & np.isfinite(right) & (np.abs(denominator) > 1e-6)
    offset = np.zeros(rows.shape[0], dtype=np.float32)
    offset[stable] = 0.5 * (left[stable] - right[stable]) / denominator[stable]
    refined[rows, cols] += np.clip(offset, -1.0, 1.0)
    return refined


def soft_argmax_disparity(all_scores: Any, min_disparity: int, max_disparity: int, temperature: float) -> Any:
    """用 soft-argmax 从整条得分曲线回归连续视差。"""

    scores = np.asarray(all_scores, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError("soft_argmax_disparity expects a 3D score volume.")

    stabilized = scores - np.max(scores, axis=0, keepdims=True)
    stabilized = np.where(np.isfinite(stabilized), stabilized, -1e4).astype(np.float32)
    logits = stabilized / max(float(temperature), 1e-3)
    logits -= np.max(logits, axis=0, keepdims=True)
    probability = np.exp(logits).astype(np.float32)
    probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-6)
    disparity_values = np.arange(min_disparity, max_disparity + 1, dtype=np.float32).reshape(-1, 1, 1)
    refined = np.sum(probability * disparity_values, axis=0, dtype=np.float32)
    valid = np.isfinite(scores).any(axis=0)
    return np.where(valid, refined, 0.0).astype(np.float32)


def regress_token_disparity(
    best_disparity: Any,
    all_scores: Any,
    min_disparity: int,
    max_disparity: int,
    regression_mode: str,
    softmax_temperature: float,
) -> Any:
    mode = str(regression_mode or "quadratic").strip().lower() or "quadratic"
    if mode == "soft_argmax":
        return soft_argmax_disparity(all_scores, min_disparity, max_disparity, softmax_temperature)
    return refine_token_disparity(best_disparity, all_scores, min_disparity, max_disparity)


def build_token_ground_truth(disparity: Any, token_span: int, patch_size: int, token_shape: tuple[int, int]) -> Any:
    """把像素级真值视差压缩成 token 级监督标签。"""

    source = np.asarray(disparity, dtype=np.float32)
    token_height, token_width = token_shape
    cropped_height = token_height * token_span
    cropped_width = token_width * token_span
    cropped = source[:cropped_height, :cropped_width]
    if cropped.size == 0:
        return np.full(token_shape, -1, dtype=np.int32)

    reshaped = cropped.reshape(token_height, token_span, token_width, token_span).transpose(0, 2, 1, 3)
    labels = np.full((token_height, token_width), -1, dtype=np.int32)
    for row_index in range(token_height):
        for column_index in range(token_width):
            patch = reshaped[row_index, column_index]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            if valid.size == 0:
                continue
            labels[row_index, column_index] = int(np.rint(np.median(valid) / float(token_span)))
    return labels


def train_o4_numpy_projection(training_descriptors: Any, candidate_descriptors: Any, config: O4Config) -> Any | None:
    """训练一个最小可运行的 numpy 投影模型。"""

    if training_descriptors.size == 0 or candidate_descriptors.size == 0 or config.training_epochs <= 0:
        return None

    feature_dim = int(training_descriptors.shape[1])
    rng = np.random.default_rng(config.random_seed)
    projection = rng.normal(0.0, 0.05, size=(feature_dim, config.model_dim)).astype(np.float32)
    sqrt_dim = float(np.sqrt(config.model_dim))
    batch_size = min(512, training_descriptors.shape[0])

    for _ in range(config.training_epochs):
        order = rng.permutation(training_descriptors.shape[0])
        for start in range(0, order.shape[0], batch_size):
            batch_index = order[start : start + batch_size]
            left_batch = training_descriptors[batch_index]
            candidate_batch = candidate_descriptors[batch_index]

            left_embedding = left_batch @ projection
            candidate_embedding = candidate_batch @ projection
            scores = np.einsum("bd,bkd->bk", left_embedding, candidate_embedding, optimize=True) / sqrt_dim

            scores -= scores.max(axis=1, keepdims=True)
            probability = np.exp(scores)
            probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-6)
            probability[:, 0] -= 1.0
            probability /= float(batch_index.shape[0])

            gradient_left = np.einsum("bk,bkd->bd", probability, candidate_embedding, optimize=True) / sqrt_dim
            gradient_candidate = probability[:, :, None] * left_embedding[:, None, :] / sqrt_dim
            gradient = (left_batch.T @ gradient_left) + (
                candidate_batch.reshape(-1, feature_dim).T @ gradient_candidate.reshape(-1, config.model_dim)
            )
            gradient += config.weight_decay * projection
            projection -= config.training_learning_rate * gradient.astype(np.float32)

    return projection.astype(np.float32)


def train_o4_model(
    training_descriptors: Any,
    candidate_descriptors: Any,
    config: O4Config,
    backend_status: Any,
) -> O4ModelState:
    if training_descriptors.size == 0 or candidate_descriptors.size == 0 or config.training_epochs <= 0:
        return O4ModelState(
            backend=backend_status.selected_backend,
            device=backend_status.device,
            trained=False,
        )

    if backend_status.use_torch:
        torch_model = train_o4_torch_model(
            training_descriptors,
            candidate_descriptors,
            model_dim=config.model_dim,
            hidden_dim=config.encoder_hidden_dim,
            encoder_layers=config.encoder_layers,
            epochs=config.training_epochs,
            learning_rate=config.training_learning_rate,
            weight_decay=config.weight_decay,
            batch_size=config.training_batch_size,
            random_seed=config.random_seed,
            device=backend_status.device,
        )
        if torch_model is not None:
            return O4ModelState(
                backend="torch",
                device=backend_status.device,
                trained=True,
                torch_model=torch_model,
            )

    projection = train_o4_numpy_projection(training_descriptors, candidate_descriptors, config)
    return O4ModelState(
        backend="numpy",
        device="cpu",
        trained=projection is not None,
        projection=projection,
    )


def collect_o4_training_samples(scene_payloads: list[dict[str, Any]], eval_fold: int, config: O4Config) -> tuple[Any, Any]:
    """从非验证 fold 的场景中收集训练样本。"""

    rng = np.random.default_rng(config.random_seed + eval_fold)
    left_samples: list[Any] = []
    candidate_samples: list[Any] = []
    max_samples = max(1, config.max_training_samples)

    for payload in scene_payloads:
        if payload["fold"] == eval_fold or payload["ground_truth_tokens"] is None:
            continue
        labels = payload["ground_truth_tokens"]
        right_descriptors = payload["right_descriptors"]
        min_disparity = payload["min_token_disparity"]
        max_disparity = payload["max_token_disparity"]

        valid_locations = np.argwhere(labels >= min_disparity)
        rng.shuffle(valid_locations)
        for row_index, column_index in valid_locations:
            disparity = int(labels[row_index, column_index])
            if disparity < min_disparity or disparity > max_disparity:
                continue
            match_column = int(column_index) - disparity
            if match_column < 0 or match_column >= right_descriptors.shape[1]:
                continue

            candidate_disparities = [disparity]
            for delta in (1, -1, 2, -2):
                alternative = disparity + delta
                alternative_column = int(column_index) - alternative
                if (
                    alternative != disparity
                    and min_disparity <= alternative <= max_disparity
                    and 0 <= alternative_column < right_descriptors.shape[1]
                ):
                    candidate_disparities.append(alternative)
                if len(candidate_disparities) >= config.negative_samples + 1:
                    break

            while len(candidate_disparities) < config.negative_samples + 1:
                alternative = int(rng.integers(min_disparity, max_disparity + 1))
                alternative_column = int(column_index) - alternative
                if alternative == disparity or alternative_column < 0 or alternative_column >= right_descriptors.shape[1]:
                    continue
                candidate_disparities.append(alternative)

            candidate_vectors = np.stack(
                [right_descriptors[row_index, int(column_index) - value] for value in candidate_disparities],
                axis=0,
            )
            left_samples.append(payload["left_descriptors"][row_index, column_index])
            candidate_samples.append(candidate_vectors)
            if len(left_samples) >= max_samples:
                return np.asarray(left_samples, dtype=np.float32), np.asarray(candidate_samples, dtype=np.float32)

    if not left_samples:
        feature_dim = scene_payloads[0]["left_descriptors"].shape[2] if scene_payloads else 1
        return (
            np.zeros((0, feature_dim), dtype=np.float32),
            np.zeros((0, config.negative_samples + 1, feature_dim), dtype=np.float32),
        )
    return np.asarray(left_samples, dtype=np.float32), np.asarray(candidate_samples, dtype=np.float32)


def upsample_token_grid(token_values: Any, span: int, output_height: int, output_width: int) -> Any:
    """把 token 网格还原回像素分辨率。"""

    upsampled = np.repeat(np.repeat(token_values, span, axis=0), span, axis=1)
    result = np.zeros((output_height, output_width), dtype=upsampled.dtype)
    copy_height = min(output_height, upsampled.shape[0])
    copy_width = min(output_width, upsampled.shape[1])
    result[:copy_height, :copy_width] = upsampled[:copy_height, :copy_width]
    return result


def estimate_o4_working_set_mb(token_height: int, token_width: int, token_dim: int) -> float:
    """粗略估计 O4 推理时的工作集内存。"""
    token_bytes = token_height * token_width * token_dim * 4 * 2
    score_bytes = token_height * token_width * 4 * 4
    total_bytes = token_bytes + score_bytes
    return float(total_bytes / (1024.0 * 1024.0))


def normalize_confidence_weights(confidence: Any) -> Any:
    """把 token 置信度压到 0..1，避免少数极值支配融合。"""

    source = np.asarray(confidence, dtype=np.float32)
    valid = np.isfinite(source) & (source > 0)
    weights = np.zeros_like(source, dtype=np.float32)
    if not bool(np.any(valid)):
        return weights
    scale = float(np.percentile(source[valid], 90.0))
    if scale <= 0:
        return weights
    weights[valid] = np.clip(source[valid] / scale, 0.0, 1.0)
    return weights


def build_o4_display_confidence(confidence: Any, disparity: Any) -> Any:
    """生成与最终 token 视差一致的 confidence 预览。"""

    disparity_source = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity_source) & (disparity_source > 0)
    depth_weight = normalize_confidence_weights(disparity_source)
    display_confidence = np.where(valid, 0.25 + (0.75 * depth_weight), 0.0)
    return display_confidence.astype(np.float32)


def filter_o4_reliable_disparity(disparity: Any, confidence: Any, min_confidence: float, confidence_percentile: float) -> tuple[Any, float, int]:
    """按匹配置信度筛出正式 PFM 使用的可靠 token 预测。"""

    source = np.asarray(disparity, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    valid = np.isfinite(source) & (source > 0) & np.isfinite(conf) & (conf > 0)
    if not bool(np.any(valid)):
        return np.zeros_like(source, dtype=np.float32), float(min_confidence), 0

    positive_confidence = conf[valid]
    percentile_floor = float(np.percentile(positive_confidence, float(confidence_percentile)))
    confidence_floor = max(float(min_confidence), percentile_floor)
    reliable = valid & (conf >= confidence_floor)
    filtered = np.where(reliable, source, 0.0).astype(np.float32)
    return filtered, confidence_floor, int(np.count_nonzero(reliable))


def fill_o4_preview_by_nearest(disparity: Any) -> Any:
    """用最近可靠视差填满展示图，作为仅供 PNG 使用的连续初值。"""

    source = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(source) & (source > 0)
    if not bool(np.any(valid)):
        return np.zeros_like(source, dtype=np.float32)
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = source[tuple(indices)]
    return np.where(np.isfinite(filled) & (filled > 0), filled, 0.0).astype(np.float32)


def guided_smooth_o4_preview(disparity: Any, guide_gray: Any) -> Any:
    """用左图引导滤波去掉最近邻填充的硬边界，让 PNG 预览更像连续深度图。"""

    source = np.asarray(disparity, dtype=np.float32)
    guide = np.asarray(guide_gray, dtype=np.float32)
    guide = (guide - float(guide.min())) / max(float(guide.max() - guide.min()), 1.0)
    guide = guide.astype(np.float32)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        return cv2.ximgproc.guidedFilter(guide=guide, src=source.astype(np.float32), radius=9, eps=1e-3).astype(np.float32)
    return cv2.bilateralFilter(source.astype(np.float32), d=11, sigmaColor=6.0, sigmaSpace=13.0).astype(np.float32)


def build_o4_preview_disparity(disparity: Any, guide_gray: Any) -> Any:
    """为 PNG 预览生成连续视差图，不改变写入 PFM 的可靠像素集合。"""

    preview = np.asarray(disparity, dtype=np.float32).copy()
    if not bool(np.any(np.isfinite(preview) & (preview > 0))):
        return preview

    for radius, sigma_color, sigma_space, min_count in ((2, 10.0, 1.6, 5), (3, 12.0, 2.0, 10)):
        preview = fill_disparity_with_local_weighted_median(
            preview,
            guide_gray,
            radius=radius,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
            min_count=min_count,
        )
    valid = np.isfinite(preview) & (preview > 0)
    valid_values = preview[valid]
    low = float(np.percentile(valid_values, 1.0))
    high = float(np.percentile(valid_values, 99.0))
    if high > low:
        dense_preview = np.clip(fill_o4_preview_by_nearest(preview), low, high)
        smoothed_preview = np.clip(guided_smooth_o4_preview(dense_preview, guide_gray), low, high)
        preview = np.where(valid, (0.85 * preview) + (0.15 * smoothed_preview), smoothed_preview).astype(np.float32)
    filtered = joint_weighted_median_filter_disparity(
        preview,
        guide_gray,
        radius=1,
        sigma_color=8.0,
        sigma_space=1.2,
    )
    return np.where(filtered > 0, filtered, preview).astype(np.float32)


def summarize_o4_folds(rows: list[dict[str, str | int | float]], num_folds: int) -> list[dict[str, str | int | float]]:
    """把场景级指标汇总成 fold 级平均指标。"""
    fold_rows: list[dict[str, str | int | float]] = []
    for fold_index in range(num_folds):
        matching = [row for row in rows if int(row["fold"]) == fold_index]
        valid = [row for row in matching if str(row["mae"]) != "NA"]
        mean_mae = sum(float(row["mae"]) for row in valid) / len(valid) if valid else -1.0
        mean_rmse = sum(float(row["rmse"]) for row in valid) / len(valid) if valid else -1.0
        mean_bad_1px = sum(float(row["bad_1px"]) for row in valid) / len(valid) if valid else -1.0
        fold_rows.append(
            {
                "fold": fold_index,
                "scene_count": len(matching),
                "scenes_with_ground_truth": len(valid),
                "mean_mae": f"{mean_mae:.6f}" if mean_mae >= 0 else "NA",
                "mean_rmse": f"{mean_rmse:.6f}" if mean_rmse >= 0 else "NA",
                "mean_bad_1px": f"{mean_bad_1px:.6f}" if mean_bad_1px >= 0 else "NA",
            }
        )
    return fold_rows


def validate_results(disparity_dir: Path, analysis_dir: Path, metrics_file: Path, scene_name: str | None = None) -> int:
    """验证 O4 输出文件。"""
    print(f"Validating O4 disparity directory: {disparity_dir}")
    print(f"Validating O4 analysis directory: {analysis_dir}")
    print(f"Validating O4 metrics file: {metrics_file}")

    if not metrics_file.exists():
        print(f"O4 metrics file not found: {metrics_file}", file=sys.stderr)
        return 1

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("scene")]

    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
        rows = [row for row in rows if row.get("scene") == scene_name]

    if not rows:
        print("No O4 metric rows found for validation.", file=sys.stderr)
        return 1

    missing_any = False
    for row in rows:
        current_scene = row["scene"]
        disparity_scene_dir = disparity_dir / current_scene
        analysis_scene_dir = analysis_dir / current_scene
        required_paths = (
            disparity_scene_dir / "disp0.pfm",
            disparity_scene_dir / "disp0.png",
            disparity_scene_dir / "disp0_transformer_raw.pfm",
            disparity_scene_dir / "disp0_transformer_raw.png",
            analysis_scene_dir / "confidence.png",
            analysis_scene_dir / "error_map.png",
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            missing_any = True
            print(f"[MISSING] {current_scene}: " + ", ".join(missing))
        else:
            print(f"[OK] {current_scene}: required O4 files present")

    if missing_any:
        print("Validation status: issues found")
        return 1

    print("Validation status: all checked O4 scene folders contain the expected files")
    return 0


def run(
    repo_root: Path,
    middlebury_root: Path,
    config: O4Config,
    max_scenes: int | None,
    dry_run: bool,
    scene_name: str | None,
) -> int:

    discovered_scenes = discover_scenes(middlebury_root)
    discovered_count = len(discovered_scenes)
    scenes = filter_scene_dirs(discovered_scenes, scene_name)
    scene_fold_map = {scene.name: index % config.num_folds for index, scene in enumerate(discovered_scenes)}
    backend_status = resolve_torch_backend(config.backend, config.device, config.prefer_cuda)
    if str(config.execution_mode).strip().lower() == "dinov2_cost_volume":
        from o4_dinov2 import resolve_o4_execution_mode

        execution_status = resolve_o4_execution_mode(
            config.execution_mode,
            backend_status.use_torch,
            config.dinov2_model_name,
            config.dinov2_repo_path,
            config.dinov2_checkpoint_path,
        )
    else:
        execution_status = type("O4ExecutionModeStatus", (), {
            "requested_mode": "baseline",
            "selected_mode": "baseline",
            "descriptor_source": "trainable_stereo_transformer_tokens",
            "available": True,
            "reason": "using the trainable stereo Transformer token encoder baseline",
        })()

    if max_scenes is not None:
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    print(f"Repository root: {repo_root}")
    print(f"Middlebury root: {middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {discovered_count}")
    print(f"O4 disparity output dir: {config.disparity_dir}")
    print(f"O4 analysis output dir: {config.analysis_dir}")
    print(f"O4 metrics file: {config.metrics_file}")
    print(f"O4 fold metrics file: {config.fold_metrics_file}")
    print(
        "O4 model config: "
        f"folds={config.num_folds}, downsample_factor={config.downsample_factor}, "
        f"patch_size={config.patch_size}, backend={config.backend}, resolved_backend={backend_status.selected_backend}, "
        f"device={backend_status.device}, execution_mode={config.execution_mode}, "
        f"descriptor_source={execution_status.descriptor_source}, model_dim={config.model_dim}, "
        f"encoder_hidden_dim={config.encoder_hidden_dim}, "
        f"encoder_layers={config.encoder_layers}, epochs={config.training_epochs}, "
        f"batch_size={config.training_batch_size}, max_disparity={config.max_disparity}, "
        f"min_confidence={config.min_confidence}, token_median_filter_size={config.token_median_filter_size}, "
        f"disparity_regression={config.disparity_regression}"
    )
    print(f"O4 backend status: {backend_status.reason}")
    print(f"O4 execution mode status: {execution_status.reason}")
    if execution_status.selected_mode == "dinov2_cost_volume":
        print(f"O4 DINOv2 model: {config.dinov2_model_name}")
        print(f"O4 DINOv2 repo path: {config.dinov2_repo_path or '(use current environment)'}")
        print(f"O4 DINOv2 checkpoint: {config.dinov2_checkpoint_path}")
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

    if not execution_status.available:
        print(f"O4 execution mode unavailable: {execution_status.reason}", file=sys.stderr)
        return 1

    if execution_status.selected_mode == "dinov2_cost_volume":
        from o4_dinov2 import run_dinov2_objective

        return run_dinov2_objective(repo_root, middlebury_root, config, max_scenes, dry_run=False, scene_name=scene_name)

    config.disparity_dir.mkdir(parents=True, exist_ok=True)
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    use_cuda_o4 = backend_status.use_torch and backend_status.device == "cuda"

    payload_scene_dirs = (
        discovered_scenes
        if execution_status.selected_mode == "baseline" and scene_name is None and max_scenes is None
        else scenes
    )

    scene_payloads: list[dict[str, Any]] = []
    for scene_dir in payload_scene_dirs:
        left_gray = load_gray(scene_dir / "im0.png")
        right_gray = load_gray(scene_dir / "im1.png")
        if use_cuda_o4:
            downsampled_left = average_pool_gray_torch(left_gray, config.downsample_factor, device=backend_status.device)
            downsampled_right = average_pool_gray_torch(right_gray, config.downsample_factor, device=backend_status.device)
            left_descriptors, _, _ = build_token_descriptors_torch(
                downsampled_left,
                config.patch_size,
                config.context_window_size,
                device=backend_status.device,
            )
            right_descriptors, _, _ = build_token_descriptors_torch(
                downsampled_right,
                config.patch_size,
                config.context_window_size,
                device=backend_status.device,
            )
            token_span = config.downsample_factor * config.patch_size
        else:
            downsampled_left = average_pool_gray(left_gray, config.downsample_factor)
            downsampled_right = average_pool_gray(right_gray, config.downsample_factor)
            left_descriptors, _, _ = build_token_descriptors(
                downsampled_left,
                config.patch_size,
                config.context_window_size,
            )
            right_descriptors, _, _ = build_token_descriptors(
                downsampled_right,
                config.patch_size,
                config.context_window_size,
            )
            token_span = config.downsample_factor * config.patch_size
        max_token_disparity = min(
            max(config.min_disparity, config.max_disparity // token_span),
            max(0, left_descriptors.shape[1] - 1),
        )
        min_token_disparity = min(config.min_disparity // token_span, max_token_disparity)

        ground_truth_tokens = None
        ground_truth_path = scene_dir / "disp0.pfm"
        if ground_truth_path.exists():
            ground_truth = read_pfm(ground_truth_path)
            if getattr(ground_truth, "ndim", 0) == 3:
                ground_truth = ground_truth[:, :, 0]
            if use_cuda_o4:
                ground_truth_tokens = build_token_ground_truth_torch(
                    ground_truth,
                    token_span,
                    left_descriptors.shape[:2],
                    device=backend_status.device,
                )
            else:
                ground_truth_tokens = build_token_ground_truth(
                    ground_truth,
                    token_span,
                    config.patch_size,
                    left_descriptors.shape[:2],
                )

        scene_payloads.append(
            {
                "scene_dir": scene_dir,
                "scene_name": scene_dir.name,
                "fold": scene_fold_map[scene_dir.name],
                "left_gray": left_gray,
                "right_gray": right_gray,
                "left_descriptors": left_descriptors,
                "right_descriptors": right_descriptors,
                "execution_mode": execution_status.selected_mode,
                "descriptor_source": execution_status.descriptor_source,
                "token_span": token_span,
                "min_token_disparity": min_token_disparity,
                "max_token_disparity": max_token_disparity,
                "ground_truth_tokens": ground_truth_tokens,
            }
        )

    fold_models: dict[int, O4ModelState] = {}
    fold_training_stats: dict[int, tuple[int, O4ModelState]] = {}
    if execution_status.selected_mode == "baseline":
        for fold_index in range(config.num_folds):
            if use_cuda_o4:
                training_descriptors, candidate_descriptors = collect_o4_training_samples_torch(
                    scene_payloads,
                    fold_index,
                    config,
                    device=backend_status.device,
                )
            else:
                training_descriptors, candidate_descriptors = collect_o4_training_samples(scene_payloads, fold_index, config)
            fold_models[fold_index] = train_o4_model(
                training_descriptors,
                candidate_descriptors,
                config,
                backend_status,
            )
            fold_training_stats[fold_index] = (int(training_descriptors.shape[0]), fold_models[fold_index])

    metric_rows: list[MetricRow] = []
    for scene_dir in scenes:
        payload = next(item for item in scene_payloads if item["scene_name"] == scene_dir.name)
        if execution_status.selected_mode == "baseline":
            model_state = fold_models[payload["fold"]]
        else:
            model_state = O4ModelState(
                backend=backend_status.selected_backend,
                device=backend_status.device,
                trained=False,
            )
        use_cuda_scene = model_state.device == "cuda"

        if execution_status.selected_mode == "baseline" and model_state.torch_model is not None:
            left_tokens = encode_o4_descriptors_torch(
                model_state.torch_model,
                payload["left_descriptors"],
                batch_size=config.inference_batch_size,
                device=model_state.device,
                return_numpy=not use_cuda_scene,
            )
            right_tokens = encode_o4_descriptors_torch(
                model_state.torch_model,
                payload["right_descriptors"],
                batch_size=config.inference_batch_size,
                device=model_state.device,
                return_numpy=not use_cuda_scene,
            )
        elif use_cuda_o4:
            left_tokens = payload["left_descriptors"]
            right_tokens = payload["right_descriptors"]
        else:
            left_tokens = project_token_descriptors(payload["left_descriptors"], model_state.projection)
            right_tokens = project_token_descriptors(payload["right_descriptors"], model_state.projection)

        if left_tokens.shape[:2] != right_tokens.shape[:2]:
            print(f"Token grid mismatch for scene {scene_dir.name}.", file=sys.stderr)
            return 1

        token_span = payload["token_span"]
        min_token_disparity = payload["min_token_disparity"]
        max_token_disparity = payload["max_token_disparity"]
        if execution_status.selected_mode == "baseline":
            training_sample_count, training_state = fold_training_stats[payload["fold"]]
        else:
            training_sample_count = 0
            training_state = model_state
        fallback_used = execution_status.selected_mode == "baseline" and not training_state.trained
        display_source_disparity = None

        if fallback_used:
            fallback_downsample_factor = max(config.downsample_factor, 2)
            fallback_patch_size = max(config.patch_size, 4)
            fallback_token_span = fallback_downsample_factor * fallback_patch_size
            if use_cuda_o4:
                fallback_left = average_pool_gray_torch(payload["left_gray"], fallback_downsample_factor, device=backend_status.device)
                fallback_right = average_pool_gray_torch(payload["right_gray"], fallback_downsample_factor, device=backend_status.device)
                fallback_left_tokens, _, _ = extract_baseline_patch_tokens_torch(
                    fallback_left,
                    fallback_patch_size,
                    device=backend_status.device,
                )
                fallback_right_tokens, _, _ = extract_baseline_patch_tokens_torch(
                    fallback_right,
                    fallback_patch_size,
                    device=backend_status.device,
                )
                fallback_max_token_disparity = min(
                    max(config.min_disparity, config.max_disparity // fallback_token_span),
                    max(0, fallback_left_tokens.shape[1] - 1),
                )
                fallback_min_token_disparity = min(config.min_disparity // fallback_token_span, fallback_max_token_disparity)
                fallback_disparity_tokens, fallback_confidence_tokens, _ = predict_token_disparity_torch(
                    fallback_left_tokens,
                    fallback_right_tokens,
                    fallback_min_token_disparity,
                    fallback_max_token_disparity,
                    device=backend_status.device,
                    return_numpy=False,
                )
                if config.token_median_filter_size > 1:
                    fallback_disparity_tokens = median_filter_2d_torch(
                        fallback_disparity_tokens.float(),
                        config.token_median_filter_size,
                        device=backend_status.device,
                    )
                disparity = upsample_token_grid_torch(
                    fallback_disparity_tokens.float() * float(fallback_token_span),
                    fallback_token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                    device=backend_status.device,
                ).detach().cpu().numpy().astype(np.float32)
                confidence = upsample_token_grid_torch(
                    fallback_confidence_tokens.masked_fill(fallback_confidence_tokens < config.min_confidence, 0.0).float(),
                    fallback_token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                    device=backend_status.device,
                ).detach().cpu().numpy().astype(np.float32)
                display_source_disparity = disparity
            else:
                fallback_left = average_pool_gray(payload["left_gray"], fallback_downsample_factor)
                fallback_right = average_pool_gray(payload["right_gray"], fallback_downsample_factor)
                fallback_left_tokens, _, _ = extract_baseline_patch_tokens(fallback_left, fallback_patch_size)
                fallback_right_tokens, _, _ = extract_baseline_patch_tokens(fallback_right, fallback_patch_size)
                fallback_max_token_disparity = min(
                    max(config.min_disparity, config.max_disparity // fallback_token_span),
                    max(0, fallback_left_tokens.shape[1] - 1),
                )
                fallback_min_token_disparity = min(config.min_disparity // fallback_token_span, fallback_max_token_disparity)
                fallback_disparity_tokens, fallback_confidence_tokens, _ = predict_token_disparity(
                    fallback_left_tokens,
                    fallback_right_tokens,
                    fallback_min_token_disparity,
                    fallback_max_token_disparity,
                )
                if config.token_median_filter_size > 1:
                    fallback_disparity_tokens = median_filter_2d(
                        fallback_disparity_tokens.astype(np.float32),
                        config.token_median_filter_size,
                    ).astype(np.float32)
                disparity = upsample_token_grid(
                    fallback_disparity_tokens.astype(np.float32) * float(fallback_token_span),
                    fallback_token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                confidence = upsample_token_grid(
                    np.where(fallback_confidence_tokens >= config.min_confidence, fallback_confidence_tokens, 0.0).astype(np.float32),
                    fallback_token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                display_source_disparity = disparity
        else:
            if use_cuda_scene:
                token_disparity, token_confidence, token_scores = predict_token_disparity_torch(
                    left_tokens,
                    right_tokens,
                    min_token_disparity,
                    max_token_disparity,
                    device=model_state.device,
                    return_numpy=False,
                )
                right_to_left_disparity, _, _ = predict_token_disparity_torch(
                    right_tokens,
                    left_tokens,
                    min_token_disparity,
                    max_token_disparity,
                    device=model_state.device,
                    target_direction="positive",
                    return_numpy=False,
                )
                refined_token_disparity = refine_token_disparity_torch(
                    token_disparity,
                    token_scores,
                    min_token_disparity,
                    max_token_disparity,
                    device=model_state.device,
                )
                if config.disparity_regression == "soft_argmax":
                    refined_token_disparity = soft_argmax_disparity_torch(
                        token_scores,
                        min_token_disparity,
                        max_token_disparity,
                        config.softmax_temperature,
                        device=model_state.device,
                    )
                consistency_mask = left_right_consistency_mask_torch(
                    refined_token_disparity,
                    right_to_left_disparity,
                    config.consistency_threshold,
                    device=model_state.device,
                )
                confidence_tokens = token_confidence.masked_fill(token_confidence < config.min_confidence, 0.0).float()
                official_token_disparity = refined_token_disparity.masked_fill(~consistency_mask, 0.0).float()
                display_token_disparity = fill_invalid_disparity_torch(
                    official_token_disparity,
                    config.fill_invalid_passes,
                    device=model_state.device,
                )
                if config.token_median_filter_size > 1:
                    token_mask = official_token_disparity > 0
                    filtered_official = median_filter_2d_torch(
                        official_token_disparity,
                        config.token_median_filter_size,
                        device=model_state.device,
                    )
                    official_token_disparity = filtered_official.masked_fill(~token_mask, 0.0).float()
                    display_mask = display_token_disparity > 0
                    filtered_display = median_filter_2d_torch(
                        display_token_disparity,
                        config.token_median_filter_size,
                        device=model_state.device,
                    )
                    display_token_disparity = filtered_display.masked_fill(~display_mask, 0.0).float()
                disparity_tokens = official_token_disparity.float() * float(token_span)
                disparity = upsample_token_grid_torch(
                    disparity_tokens,
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                    device=model_state.device,
                ).detach().cpu().numpy().astype(np.float32)
                display_source_disparity = upsample_token_grid_torch(
                    display_token_disparity.float() * float(token_span),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                    device=model_state.device,
                ).detach().cpu().numpy().astype(np.float32)
                confidence = upsample_token_grid_torch(
                    confidence_tokens,
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                    device=model_state.device,
                ).detach().cpu().numpy().astype(np.float32)
            else:
                if model_state.torch_model is not None:
                    token_disparity, token_confidence, token_scores = predict_token_disparity_torch(
                        left_tokens,
                        right_tokens,
                        min_token_disparity,
                        max_token_disparity,
                        device=model_state.device,
                    )
                    right_to_left_disparity, _, _ = predict_token_disparity_torch(
                        right_tokens,
                        left_tokens,
                        min_token_disparity,
                        max_token_disparity,
                        device=model_state.device,
                        target_direction="positive",
                    )
                else:
                    token_disparity, token_confidence, token_scores = predict_token_disparity(
                        left_tokens,
                        right_tokens,
                        min_token_disparity,
                        max_token_disparity,
                    )
                    right_to_left_disparity, _, _ = predict_token_disparity(
                        right_tokens,
                        left_tokens,
                        min_token_disparity,
                        max_token_disparity,
                        target_direction="positive",
                    )

                refined_token_disparity = regress_token_disparity(
                    token_disparity,
                    token_scores,
                    min_token_disparity,
                    max_token_disparity,
                    config.disparity_regression,
                    config.softmax_temperature,
                )
                consistency_mask = left_right_consistency_mask(
                    refined_token_disparity,
                    right_to_left_disparity.astype(np.float32),
                    config.consistency_threshold,
                )
                confidence_tokens = np.where(token_confidence >= config.min_confidence, token_confidence, 0.0).astype(np.float32)
                official_token_disparity = np.where(consistency_mask, refined_token_disparity, 0.0).astype(np.float32)
                display_token_disparity = fill_invalid_disparity(official_token_disparity, config.fill_invalid_passes)
                if config.token_median_filter_size > 1:
                    token_mask = official_token_disparity > 0
                    filtered_official = median_filter_2d(official_token_disparity, config.token_median_filter_size)
                    official_token_disparity = np.where(token_mask, filtered_official, 0.0).astype(np.float32)
                    display_mask = display_token_disparity > 0
                    filtered_display = median_filter_2d(display_token_disparity, config.token_median_filter_size)
                    display_token_disparity = np.where(display_mask, filtered_display, 0.0).astype(np.float32)
                disparity_tokens = official_token_disparity.astype(np.float32) * float(token_span)
                disparity = upsample_token_grid(
                    disparity_tokens,
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                display_source_disparity = upsample_token_grid(
                    display_token_disparity.astype(np.float32) * float(token_span),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                confidence = upsample_token_grid(
                    confidence_tokens,
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)

        raw_transformer_disparity = np.where(np.isfinite(disparity) & (disparity > 0), disparity, 0.0).astype(np.float32)
        disparity = display_source_disparity if display_source_disparity is not None else raw_transformer_disparity
        disparity = np.where(np.isfinite(disparity) & (disparity > 0), disparity, 0.0).astype(np.float32)
        reliable_confidence_floor = float(config.min_confidence)
        reliable_pixel_count = int(np.count_nonzero(disparity > 0))
        display_disparity = build_o4_preview_disparity(disparity, payload["left_gray"])
        display_confidence = build_o4_display_confidence(confidence, display_disparity)

        disparity_scene_dir = config.disparity_dir / scene_dir.name
        analysis_scene_dir = config.analysis_dir / scene_dir.name
        disparity_scene_dir.mkdir(parents=True, exist_ok=True)
        analysis_scene_dir.mkdir(parents=True, exist_ok=True)

        estimated_working_set_mb = estimate_o4_working_set_mb(
            left_tokens.shape[0],
            left_tokens.shape[1],
            left_tokens.shape[2],
        )
        generator_name = "O4 trainable stereo Transformer token disparity"

        write_pfm(disparity_scene_dir / "disp0.pfm", disparity)
        write_pfm(disparity_scene_dir / "disp0_transformer_raw.pfm", raw_transformer_disparity)
        raw_disparity_preview = normalize_for_preview(raw_transformer_disparity, raw_transformer_disparity > 0)
        write_png(disparity_scene_dir / "disp0_transformer_raw.png", raw_disparity_preview)
        disparity_preview = normalize_for_preview(display_disparity, display_disparity > 0)
        write_png(disparity_scene_dir / "disp0.png", disparity_preview)
        write_scene_text(
            disparity_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                f"generator: {generator_name}",
                f"fold: {scene_fold_map[scene_dir.name]}",
                f"token_grid: {left_tokens.shape[0]}x{left_tokens.shape[1]}",
                f"token_span_pixels: {token_span}",
                f"max_token_disparity: {max_token_disparity}",
                f"requested_backend: {config.backend}",
                f"resolved_backend: {training_state.backend}",
                f"requested_execution_mode: {config.execution_mode}",
                f"resolved_execution_mode: {payload['execution_mode']}",
                f"descriptor_source: {payload['descriptor_source']}",
                "final_disparity_source: O4 transformer token prediction",
                "sgm_detail_fusion_used: no",
                "raw_transformer_disparity: disp0_transformer_raw.pfm",
                "official_pfm_selection: filled_transformer_prediction",
                f"disparity_regression: {config.disparity_regression}",
                f"runtime_device: {training_state.device}",
                f"model_dim: {config.model_dim}",
                f"encoder_hidden_dim: {config.encoder_hidden_dim}",
                f"encoder_layers: {config.encoder_layers}",
                f"training_epochs: {config.training_epochs}",
                f"training_batch_size: {config.training_batch_size}",
                f"training_samples: {training_sample_count}",
                f"trained_projection: {'yes' if training_state.trained else 'no'}",
                f"fallback_coarse_matcher_used: {'yes' if fallback_used else 'no'}",
                f"official_pfm_confidence_floor: {reliable_confidence_floor:.6f}",
                f"official_pfm_reliable_pixels: {reliable_pixel_count}",
                f"token_median_filter_size: {config.token_median_filter_size}",
                f"estimated_working_set_mb: {estimated_working_set_mb:.3f}",
            ],
        )

        metrics = {
            "valid_disparity_pixels": int((disparity > 0).sum()),
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

        confidence_mask = np.isfinite(display_confidence) & (display_confidence > 0)
        confidence_preview = normalize_for_preview(display_confidence, confidence_mask)
        if bool(np.any(confidence_mask)) and int(confidence_preview.max()) == 0:
            confidence_preview[confidence_mask] = 255
        write_png(analysis_scene_dir / "confidence.png", confidence_preview)

        error_mask = error_map > 0
        error_preview = normalize_for_preview(error_map, error_mask)
        error_color = np.stack([error_preview, error_preview, error_preview], axis=2)
        error_color[~error_mask] = 0
        write_png(analysis_scene_dir / "error_map.png", error_color)
        reliable_confidence_mask = (disparity > 0) & np.isfinite(confidence)
        mean_confidence = float(confidence[reliable_confidence_mask].mean()) if bool(np.any(reliable_confidence_mask)) else 0.0
        write_scene_text(
            analysis_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                f"generator: {generator_name}",
                f"mean_confidence: {mean_confidence:.6f}",
                f"resolved_backend: {training_state.backend}",
                f"resolved_execution_mode: {payload['execution_mode']}",
                f"descriptor_source: {payload['descriptor_source']}",
                "final_disparity_source: O4 transformer token prediction",
                "sgm_detail_fusion_used: no",
                "official_pfm_selection: filled_transformer_prediction",
                f"disparity_regression: {config.disparity_regression}",
                f"runtime_device: {training_state.device}",
                f"training_samples: {training_sample_count}",
                f"trained_projection: {'yes' if training_state.trained else 'no'}",
                f"fallback_coarse_matcher_used: {'yes' if fallback_used else 'no'}",
                f"official_pfm_confidence_floor: {reliable_confidence_floor:.6f}",
                f"official_pfm_reliable_pixels: {reliable_pixel_count}",
                f"ground_truth_available: {'yes' if ground_truth_path.exists() else 'no'}",
                f"mae: {metrics['mae'] if metrics['mae'] >= 0 else 'NA'}",
                f"rmse: {metrics['rmse'] if metrics['rmse'] >= 0 else 'NA'}",
                f"bad_1px: {metrics['bad_1px'] if metrics['bad_1px'] >= 0 else 'NA'}",
            ],
        )

        metric_rows.append(
            {
                "scene": scene_dir.name,
                "fold": scene_fold_map[scene_dir.name],
                "token_grid": f"{left_tokens.shape[0]}x{left_tokens.shape[1]}",
                "valid_disparity_pixels": metrics["valid_disparity_pixels"],
                "valid_ground_truth_pixels": metrics["valid_ground_truth_pixels"],
                "mae": f"{metrics['mae']:.6f}" if metrics["mae"] >= 0 else "NA",
                "rmse": f"{metrics['rmse']:.6f}" if metrics["rmse"] >= 0 else "NA",
                "bad_1px": f"{metrics['bad_1px']:.6f}" if metrics["bad_1px"] >= 0 else "NA",
                "mean_confidence": f"{mean_confidence:.6f}",
                "estimated_working_set_mb": f"{estimated_working_set_mb:.6f}",
            }
        )
        mae_text = f"{metrics['mae']:.6f}" if metrics["mae"] >= 0 else "NA"
        rmse_text = f"{metrics['rmse']:.6f}" if metrics["rmse"] >= 0 else "NA"
        bad_text = f"{metrics['bad_1px']:.6f}" if metrics["bad_1px"] >= 0 else "NA"
        print(
            f"Wrote O4 scene: {scene_dir.name} "
            f"(fold={scene_fold_map[scene_dir.name]}, token_grid={left_tokens.shape[0]}x{left_tokens.shape[1]}, "
            f"working_set_mb={estimated_working_set_mb:.3f}, mae={mae_text}, rmse={rmse_text}, bad_1px={bad_text})"
        )

    write_metrics(config.metrics_file, metric_rows)
    write_fold_metrics(config.fold_metrics_file, summarize_o4_folds(metric_rows, config.num_folds))
    print(f"Wrote O4 metrics summary: {config.metrics_file}")
    print(f"Wrote O4 fold summary: {config.fold_metrics_file}")
    return 0


if __name__ == '__main__':
    from entry_utils import run_objective_entry

    raise SystemExit(run_objective_entry('o4', __file__))
