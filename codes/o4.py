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
    content_bbox_from_gray,
    discover_scenes,
    evaluate_disparity,
    filter_scene_dirs,
    load_gray,
    normalize_for_preview,
    rectangular_mask_from_bbox,
    write_png,
    write_scene_text,
)
from config import O3Config, O4Config
from o3 import (
    average_pool_gray,
    box_filter_sum,
    compute_horizontal_gradient,
    colorize_disparity_depth_map,
    estimate_scene_disparity_bounds,
    fill_disparity_with_local_weighted_median,
    fill_short_horizontal_disparity_gaps,
    fill_short_vertical_disparity_gaps,
    filter_speckles,
    joint_weighted_median_filter_disparity,
    left_right_consistency_error,
    left_right_consistency_mask,
    local_disparity_support_mask,
    median_filter_2d,
    solve_sgm_from_cost,
    solve_sgm_from_cost_torch,
)
from o4_torch import (
    average_pool_gray_torch,
    build_stereo_token_transformer,
    build_token_descriptors_torch,
    build_token_ground_truth_torch,
    collect_o4_training_samples_torch,
    encode_o4_descriptors_torch,
    extract_baseline_patch_tokens_torch,
    is_cuda_device_name,
    left_right_consistency_mask_torch,
    median_filter_2d_torch,
    predict_token_disparity_torch,
    refine_token_disparity_torch,
    resolve_torch_backend,
    soft_argmax_disparity_torch,
    train_o4_torch_model,
    upsample_token_grid_torch,
)
from o4_dinov2 import O4ExecutionModeStatus
from o4_PiplineDrawer import write_o4_pipeline_assets
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
    parameter_count: int = 0
    checkpoint_path: Path | None = None


def save_o4_model_checkpoint(path: Path, model_state: O4ModelState, config: O4Config, *, fold: int, sample_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_state.torch_model is not None:
        import torch

        torch.save(
            {
                "state_dict": model_state.torch_model.state_dict(),
                "fold": int(fold),
                "sample_count": int(sample_count),
                "backend": "torch",
                "device": str(model_state.device),
                "input_dim": int(getattr(model_state.torch_model, "input_dim", 0)),
                "model_dim": int(config.model_dim),
                "encoder_hidden_dim": int(config.encoder_hidden_dim),
                "encoder_layers": int(config.encoder_layers),
                "training_epochs": int(config.training_epochs),
                "training_learning_rate": float(config.training_learning_rate),
                "training_batch_size": int(config.training_batch_size),
                "negative_samples": int(config.negative_samples),
                "max_training_samples": int(config.max_training_samples),
                "weight_decay": float(config.weight_decay),
                "parameter_count": int(model_state.parameter_count),
                "execution_mode": str(config.execution_mode),
                "downsample_factor": int(config.downsample_factor),
                "patch_size": int(config.patch_size),
                "max_disparity": int(config.max_disparity),
                "min_disparity": int(config.min_disparity),
                "adapter_type": "StereoTokenTransformer",
            },
            path,
        )
        return
    if model_state.projection is not None:
        np.savez_compressed(
            path,
            projection=np.asarray(model_state.projection, dtype=np.float32),
            fold=np.asarray(int(fold), dtype=np.int32),
            sample_count=np.asarray(int(sample_count), dtype=np.int32),
            model_dim=np.asarray(int(config.model_dim), dtype=np.int32),
            execution_mode=np.asarray(str(config.execution_mode)),
            downsample_factor=np.asarray(int(config.downsample_factor), dtype=np.int32),
            patch_size=np.asarray(int(config.patch_size), dtype=np.int32),
        )


def load_o4_model_checkpoint(path: Path, config: O4Config, *, fold: int, device: str) -> tuple[int, int, O4ModelState] | None:
    if not path.exists():
        return None

    import torch

    checkpoint = torch.load(path, map_location=device)
    expected_values = {
        "fold": int(fold),
        "model_dim": int(config.model_dim),
        "encoder_hidden_dim": int(config.encoder_hidden_dim),
        "encoder_layers": int(config.encoder_layers),
        "negative_samples": int(config.negative_samples),
        "max_training_samples": int(config.max_training_samples),
        "execution_mode": str(config.execution_mode),
        "downsample_factor": int(config.downsample_factor),
        "patch_size": int(config.patch_size),
        "max_disparity": int(config.max_disparity),
        "min_disparity": int(config.min_disparity),
    }
    for key, expected in expected_values.items():
        if checkpoint.get(key) != expected:
            print(
                f"Skipping O4 checkpoint {path}: {key}={checkpoint.get(key)!r} does not match {expected!r}",
                file=sys.stderr,
            )
            return None

    input_dim = int(checkpoint.get("input_dim", 0))
    if input_dim <= 0:
        print(f"Skipping O4 checkpoint {path}: missing input_dim", file=sys.stderr)
        return None

    model = build_stereo_token_transformer(
        input_dim,
        int(config.model_dim),
        int(config.encoder_hidden_dim),
        int(config.encoder_layers),
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model_state = O4ModelState(
        backend="torch",
        device=str(device),
        trained=True,
        torch_model=model,
        parameter_count=int(checkpoint.get("parameter_count", getattr(model, "parameter_count", 0))),
        checkpoint_path=path,
    )
    return int(checkpoint.get("sample_count", 0)), int(checkpoint.get("training_epochs", 0)), model_state


def o4_config_warnings(config: O4Config, execution_status: O4ExecutionModeStatus) -> list[str]:
    if execution_status.selected_mode == "dinov2_cost_volume":
        return []

    warnings: list[str] = []
    recommended = {
        "execution_mode": "baseline_sgm",
        "model_dim": 96,
        "encoder_hidden_dim": 384,
        "encoder_layers": 4,
        "training_epochs": 72,
        "training_batch_size": 512,
        "negative_samples": 12,
        "max_training_samples": 60000,
        "min_confidence": 0.02,
        "token_median_filter_size": 3,
        "fill_invalid_passes": 2,
        "speckle_max_size": 150,
        "speckle_max_diff": 1.0,
    }
    current = {
        "execution_mode": execution_status.selected_mode,
        "model_dim": int(config.model_dim),
        "encoder_hidden_dim": int(config.encoder_hidden_dim),
        "encoder_layers": int(config.encoder_layers),
        "training_epochs": int(config.training_epochs),
        "training_batch_size": int(config.training_batch_size),
        "negative_samples": int(config.negative_samples),
        "max_training_samples": int(config.max_training_samples),
        "min_confidence": float(config.min_confidence),
        "token_median_filter_size": int(config.token_median_filter_size),
        "fill_invalid_passes": int(config.fill_invalid_passes),
        "speckle_max_size": int(config.speckle_max_size),
        "speckle_max_diff": float(config.speckle_max_diff),
    }
    mismatches = [f"{key}={current[key]!r} expected {value!r}" for key, value in recommended.items() if current[key] != value]
    if mismatches:
        warnings.append(
            "O4 config warning: effective config differs from the current baseline_sgm result config; "
            "metrics will not be comparable to results/O4c_transformer."
        )
        warnings.append("O4 config warning: " + ", ".join(mismatches))

    legacy_like = (
        execution_status.selected_mode == "baseline"
        and int(config.max_disparity) <= 96
        and int(config.model_dim) <= 48
        and int(config.training_epochs) <= 24
    )
    if legacy_like:
        warnings.append(
            "O4 config warning: this matches the legacy small baseline seen on the school platform; "
            "upload the current configs/ directory or run with the current baseline_sgm settings."
        )
    return warnings


def estimate_o4_scene_disparity_bounds(
    scene_dir: Path,
    image_shape: tuple[int, int],
    config: O4Config,
    o3_config: O3Config | None,
) -> tuple[int, int]:
    if o3_config is None:
        scene_min = int(config.min_disparity)
        scene_max = int(config.max_disparity)
    else:
        scene_min, scene_max = estimate_scene_disparity_bounds(scene_dir, image_shape, o3_config)
        scene_min = max(int(config.min_disparity), int(scene_min))
        scene_max = int(scene_max)
    image_width = max(1, int(image_shape[1]))
    scene_min = max(0, min(scene_min, image_width - 1))
    scene_max = max(scene_min, min(scene_max, image_width - 1))
    return scene_min, scene_max


def pixel_disparity_bounds_to_token_bounds(
    min_disparity: int,
    max_disparity: int,
    token_span: int,
    token_width: int,
) -> tuple[int, int]:
    span = max(1, int(token_span))
    max_token_limit = max(0, int(token_width) - 1)
    max_token_disparity = min(max_token_limit, max(0, (int(max_disparity) + span - 1) // span))
    min_token_disparity = min(max_token_disparity, max(0, int(min_disparity) // span))
    return min_token_disparity, max_token_disparity


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
                parameter_count=int(getattr(torch_model, "parameter_count", 0)),
            )

    projection = train_o4_numpy_projection(training_descriptors, candidate_descriptors, config)
    return O4ModelState(
        backend="numpy",
        device="cpu",
        trained=projection is not None,
        projection=projection,
        parameter_count=int(projection.size) if projection is not None else 0,
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


def build_o4_content_mask(image: Any, threshold: float = 2.0, min_fraction: float = 0.005) -> Any:
    """Build an O4-only mask that excludes black corners inside the content bbox."""

    source = np.asarray(image)
    if source.ndim == 3:
        source = source.max(axis=2)
    if source.ndim != 2:
        raise ValueError("build_o4_content_mask expects a 2D grayscale image or an RGB image.")

    bbox_mask = rectangular_mask_from_bbox(
        source.shape,
        content_bbox_from_gray(source, threshold=threshold, min_fraction=min_fraction),
    )
    nonblack = np.isfinite(source) & (source.astype(np.float32) > float(threshold))
    if not bool(np.any(nonblack)):
        return np.ones(source.shape, dtype=bool)
    return bbox_mask & nonblack


def build_token_content_mask(content_mask: Any, token_span: int, token_shape: tuple[int, int]) -> Any:
    """Project a pixel-level rectangular content mask to token centers."""

    source = np.asarray(content_mask, dtype=bool)
    token_height, token_width = int(token_shape[0]), int(token_shape[1])
    if source.size == 0 or token_height <= 0 or token_width <= 0:
        return np.zeros((token_height, token_width), dtype=bool)
    rows = np.clip((np.arange(token_height) * int(token_span)) + (int(token_span) // 2), 0, source.shape[0] - 1)
    cols = np.clip((np.arange(token_width) * int(token_span)) + (int(token_span) // 2), 0, source.shape[1] - 1)
    return source[rows[:, None], cols[None, :]].astype(bool)


def build_token_content_mask_torch(content_mask: Any, token_span: int, token_shape: tuple[int, int], *, device: str) -> Any:
    import torch

    return torch.as_tensor(
        build_token_content_mask(content_mask, token_span, token_shape),
        dtype=torch.bool,
        device=device,
    )


def apply_content_mask(disparity: Any, content_mask: Any) -> Any:
    source = np.asarray(disparity, dtype=np.float32)
    mask = np.asarray(content_mask, dtype=bool)
    return np.where(mask & np.isfinite(source) & (source > 0), source, 0.0).astype(np.float32)


def apply_raw_content_mask(disparity: Any, content_mask: Any) -> Any:
    source = np.asarray(disparity, dtype=np.float32)
    mask = np.asarray(content_mask, dtype=bool)
    return np.where(mask & np.isfinite(source), source, 0.0).astype(np.float32)


def to_numpy_float32(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def cleanup_o4_token_disparity_like_o3(
    left_disparity: Any,
    right_disparity: Any,
    guide_gray: Any,
    content_mask: Any,
    margin: Any | None,
    config: O4Config,
) -> Any:
    dense = to_numpy_float32(left_disparity)
    right = to_numpy_float32(right_disparity)
    guide = np.asarray(guide_gray, dtype=np.float32)
    mask = np.asarray(content_mask, dtype=bool)
    dense = np.where(mask & np.isfinite(dense) & (dense > 0), dense, 0.0).astype(np.float32)
    right = np.where(np.isfinite(right) & (right > 0), right, 0.0).astype(np.float32)
    if dense.shape != mask.shape or right.shape != dense.shape or not bool(np.any(dense > 0)):
        return np.zeros(mask.shape, dtype=np.float32)

    if guide.shape != dense.shape:
        guide = cv2.resize(guide.astype(np.float32), (dense.shape[1], dense.shape[0]), interpolation=cv2.INTER_AREA)

    if margin is None:
        margin_source = np.ones_like(dense, dtype=np.float32)
    else:
        margin_source = to_numpy_float32(margin)
        if margin_source.shape != dense.shape:
            margin_source = np.ones_like(dense, dtype=np.float32)
    margin_values = margin_source[mask & np.isfinite(margin_source) & (margin_source > 0)]
    margin_low = float(np.percentile(margin_values, 5.0)) if margin_values.size else 0.0

    consistency_error = left_right_consistency_error(dense, right)
    consistency_limit = max(float(config.consistency_threshold), 1.5)
    relaxed_consistency_limit = max(consistency_limit, 3.0)

    texture_x = np.abs(compute_horizontal_gradient(guide))
    texture_y = np.abs(compute_horizontal_gradient(guide.T).T)
    texture = texture_x + texture_y
    texture_values = texture[np.isfinite(texture)]
    edge_floor = float(np.percentile(texture_values, 90.0)) if texture_values.size else 0.0
    edge_mask = texture >= edge_floor

    reliable_mask = (dense > 0) & mask & (consistency_error <= consistency_limit) & (margin_source >= margin_low)
    support_mask = local_disparity_support_mask(dense, reliable_mask, radius=2, tolerance=2.5, min_count=5)
    best = np.where(support_mask, dense, 0.0).astype(np.float32)
    if config.speckle_max_size > 0:
        best = filter_speckles(best, int(config.speckle_max_size), max(1, int(round(config.speckle_max_diff))))

    relaxed_reliable_mask = (dense > 0) & mask & (consistency_error <= relaxed_consistency_limit) & (margin_source >= margin_low)
    relaxed_support_mask = local_disparity_support_mask(dense, relaxed_reliable_mask, radius=2, tolerance=3.0, min_count=3)
    relaxed = np.where(relaxed_support_mask, dense, 0.0).astype(np.float32)
    if config.speckle_max_size > 0:
        relaxed = filter_speckles(relaxed, max(1, int(round(config.speckle_max_size * 0.75))), max(1, int(round(config.speckle_max_diff))))
    best = np.where(best > 0, best, relaxed).astype(np.float32)

    if config.token_median_filter_size > 1:
        positive_mask = best > 0
        filtered = median_filter_2d(best, config.token_median_filter_size)
        best = np.where(positive_mask & ~edge_mask, filtered, best).astype(np.float32)

    filtered = joint_weighted_median_filter_disparity(best, guide, radius=1, sigma_color=8.0, sigma_space=1.2)
    best = np.where(best > 0, filtered, 0.0).astype(np.float32)
    best = fill_short_horizontal_disparity_gaps(best, guide, max_gap=32, max_disparity_delta=2.0, max_intensity_delta=28.0)
    best = fill_short_vertical_disparity_gaps(best, guide, max_gap=12, max_disparity_delta=2.0, max_intensity_delta=28.0)
    if config.token_median_filter_size > 1:
        positive_mask = best > 0
        filtered = median_filter_2d(best, config.token_median_filter_size)
        best = np.where(positive_mask, filtered, 0.0).astype(np.float32)
    return np.where(mask & np.isfinite(best) & (best > 0), best, 0.0).astype(np.float32)


def token_mask_bbox(mask: Any) -> tuple[int, int, int, int] | None:
    source = np.asarray(mask, dtype=bool)
    locations = np.argwhere(source)
    if locations.size == 0:
        return None
    top = int(locations[:, 0].min())
    bottom = int(locations[:, 0].max() + 1)
    left = int(locations[:, 1].min())
    right = int(locations[:, 1].max() + 1)
    return top, bottom, left, right


def token_mask_bbox_torch(mask: Any) -> tuple[int, int, int, int] | None:
    locations = mask.nonzero(as_tuple=False)
    if int(locations.numel()) == 0:
        return None
    top = int(locations[:, 0].min().detach().cpu().item())
    bottom = int(locations[:, 0].max().detach().cpu().item() + 1)
    left = int(locations[:, 1].min().detach().cpu().item())
    right = int(locations[:, 1].max().detach().cpu().item() + 1)
    return top, bottom, left, right


def stereo_cost_valid_mask_torch(reference_mask: Any, target_mask: Any, min_disparity: int, max_disparity: int, target_direction: str, *, device: str) -> Any:
    import torch

    reference = reference_mask.to(device=device, dtype=torch.bool)
    target = target_mask.to(device=device, dtype=torch.bool)
    height, width = reference.shape
    disparities = list(range(int(min_disparity), int(max_disparity) + 1))
    valid = torch.zeros((height, width, len(disparities)), dtype=torch.bool, device=device)
    for index, disparity in enumerate(disparities):
        if disparity == 0:
            valid[:, :, index] = reference & target
        elif target_direction == "positive" and disparity < width:
            valid[:, : width - disparity, index] = reference[:, : width - disparity] & target[:, disparity:]
        elif disparity < width:
            valid[:, disparity:, index] = reference[:, disparity:] & target[:, : width - disparity]
    return valid


def stereo_cost_valid_mask(reference_mask: Any, target_mask: Any, min_disparity: int, max_disparity: int, target_direction: str) -> Any:
    reference = np.asarray(reference_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    height, width = reference.shape
    disparities = list(range(int(min_disparity), int(max_disparity) + 1))
    valid = np.zeros((height, width, len(disparities)), dtype=bool)
    for index, disparity in enumerate(disparities):
        if disparity == 0:
            valid[:, :, index] = reference & target
        elif target_direction == "positive" and disparity < width:
            valid[:, : width - disparity, index] = reference[:, : width - disparity] & target[:, disparity:]
        elif disparity < width:
            valid[:, disparity:, index] = reference[:, disparity:] & target[:, : width - disparity]
    return valid


def solve_content_masked_sgm_from_cost_torch(cost_volume: Any, guide_gray: Any, content_mask: Any, *, min_disparity: int, device: str) -> tuple[Any, Any]:
    import torch

    mask = content_mask.to(device=device, dtype=torch.bool)
    bbox = token_mask_bbox_torch(mask)
    best = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    margin = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    if bbox is None:
        return best, margin
    top, bottom, left, right = bbox
    crop_best, crop_margin = solve_sgm_from_cost_torch(
        cost_volume[top:bottom, left:right, :],
        guide_gray[top:bottom, left:right],
        min_disparity=int(min_disparity),
        device=device,
        return_numpy=False,
    )
    crop_mask = mask[top:bottom, left:right]
    best[top:bottom, left:right] = crop_best.masked_fill(~crop_mask, 0.0).float()
    margin[top:bottom, left:right] = crop_margin.masked_fill(~crop_mask, 0.0).float()
    return best, margin


def solve_content_masked_sgm_from_cost(cost_volume: Any, guide_gray: Any, content_mask: Any, *, min_disparity: int) -> tuple[Any, Any]:
    mask = np.asarray(content_mask, dtype=bool)
    bbox = token_mask_bbox(mask)
    best = np.zeros(mask.shape, dtype=np.float32)
    margin = np.zeros(mask.shape, dtype=np.float32)
    if bbox is None:
        return best, margin
    top, bottom, left, right = bbox
    crop_best, crop_margin = solve_sgm_from_cost(
        np.asarray(cost_volume, dtype=np.float32)[top:bottom, left:right, :],
        np.asarray(guide_gray, dtype=np.float32)[top:bottom, left:right],
        min_disparity=int(min_disparity),
    )
    crop_mask = mask[top:bottom, left:right]
    best[top:bottom, left:right] = np.where(crop_mask, crop_best, 0.0).astype(np.float32)
    margin[top:bottom, left:right] = np.where(crop_mask, crop_margin, 0.0).astype(np.float32)
    return best, margin


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
    filled = source[tuple(np.asarray(indices, dtype=np.intp))]
    return np.where(np.isfinite(filled) & (filled > 0), filled, 0.0).astype(np.float32)


def filter_token_speckles(disparity: Any, max_speckle_size: int, max_diff: float) -> Any:
    """剔除 token 网格上孤立的小连通块，降低椒盐噪声。"""

    source = np.asarray(disparity, dtype=np.float32).copy()
    if max_speckle_size <= 0:
        return source
    valid = np.isfinite(source) & (source > 0)
    if not bool(np.any(valid)):
        return np.where(valid, source, 0.0).astype(np.float32)

    fixed_scale = 16.0
    fixed_disparity = np.where(valid, np.rint(source * fixed_scale), 0.0).astype(np.int16)
    fixed_max_diff = max(1, int(round(float(max_diff) * fixed_scale)))
    try:
        cv2.filterSpeckles(fixed_disparity, 0, int(max_speckle_size), fixed_max_diff)
        cleaned = fixed_disparity.astype(np.float32) / fixed_scale
    except cv2.error:
        cleaned = source
        cleaned_valid = valid.copy()
        rounded = np.where(valid, np.rint(source).astype(np.int32), -1)
        visited = np.zeros_like(valid, dtype=bool)
        token_height, token_width = source.shape
        for row_index in range(token_height):
            for column_index in range(token_width):
                if not valid[row_index, column_index] or visited[row_index, column_index]:
                    continue
                seed_value = rounded[row_index, column_index]
                stack = [(row_index, column_index)]
                component: list[tuple[int, int]] = []
                while stack:
                    pixel = stack.pop()
                    if visited[pixel]:
                        continue
                    visited[pixel] = True
                    if not valid[pixel] or abs(rounded[pixel] - seed_value) > int(max_diff):
                        continue
                    component.append(pixel)
                    pixel_row, pixel_col = pixel
                    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        neighbor_row = pixel_row + delta_row
                        neighbor_col = pixel_col + delta_col
                        if 0 <= neighbor_row < token_height and 0 <= neighbor_col < token_width:
                            if not visited[neighbor_row, neighbor_col]:
                                stack.append((neighbor_row, neighbor_col))
                if len(component) <= int(max_speckle_size):
                    for pixel in component:
                        cleaned_valid[pixel] = False
        cleaned = np.where(cleaned_valid, source, 0.0).astype(np.float32)
    cleaned = np.where(np.isfinite(cleaned) & (cleaned > 0), cleaned, 0.0).astype(np.float32)
    return cleaned


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
            disparity_scene_dir / "disp0_transformer_raw_filtered.pfm",
            disparity_scene_dir / "disp0_transformer_raw_filtered.png",
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
    o3_config: O3Config | None = None,
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
        mode_label = str(config.execution_mode).strip().lower() or "baseline"
        if mode_label == "baseline_sgm":
            execution_status = O4ExecutionModeStatus(
                requested_mode="baseline_sgm",
                selected_mode="baseline_sgm",
                descriptor_source="trainable_stereo_transformer_tokens",
                available=True,
                reason="ViT token features fed into O3-style SGM aggregation",
            )
        else:
            execution_status = O4ExecutionModeStatus(
                requested_mode="baseline",
                selected_mode="baseline",
                descriptor_source="trainable_stereo_transformer_tokens",
                available=True,
                reason="using the trainable stereo Transformer token encoder baseline",
            )

    if max_scenes is not None:
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    print(f"Repository root: {repo_root}")
    print(f"Middlebury root: {middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {discovered_count}")
    print(f"O4 pipeline output dir: {config.pipeline_dir}")
    print(f"O4 result output dir: {config.disparity_dir}")
    print(f"O4 analysis output dir: {config.analysis_dir}")
    print(f"O4 metrics file: {config.metrics_file}")
    print(f"O4 fold metrics file: {config.fold_metrics_file}")
    max_disparity_label = (
        "scene_adaptive"
        if execution_status.selected_mode in {"baseline", "baseline_sgm"}
        else str(config.max_disparity)
    )
    print(
        "O4 model config: "
        f"folds={config.num_folds}, downsample_factor={config.downsample_factor}, "
        f"patch_size={config.patch_size}, backend={config.backend}, resolved_backend={backend_status.selected_backend}, "
        f"device={backend_status.device}, execution_mode={config.execution_mode}, "
        f"descriptor_source={execution_status.descriptor_source}, dinov2_input_scale={config.dinov2_input_scale}, model_dim={config.model_dim}, "
        f"encoder_hidden_dim={config.encoder_hidden_dim}, "
        f"encoder_layers={config.encoder_layers}, epochs={config.training_epochs}, "
        f"batch_size={config.training_batch_size}, max_disparity={max_disparity_label}, "
        f"min_confidence={config.min_confidence}, token_median_filter_size={config.token_median_filter_size}, "
        f"speckle_max_size={config.speckle_max_size}, fill_invalid_passes={config.fill_invalid_passes}, "
        f"disparity_regression={config.disparity_regression}"
    )
    print(f"O4 backend status: {backend_status.reason}")
    print(f"O4 execution mode status: {execution_status.reason}")
    for warning in o4_config_warnings(config, execution_status):
        print(warning, file=sys.stderr)
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

    config.pipeline_dir.mkdir(parents=True, exist_ok=True)
    write_o4_pipeline_assets(config.pipeline_dir)
    print(f"Wrote O4 PDF pipeline image: {config.pipeline_dir / 'transformer_pipeline.jpg'}")

    if not execution_status.available:
        print(f"O4 execution mode unavailable: {execution_status.reason}", file=sys.stderr)
        return 1

    if execution_status.selected_mode == "dinov2_cost_volume":
        from o4_dinov2 import run_dinov2_objective

        return run_dinov2_objective(repo_root, middlebury_root, config, max_scenes, dry_run=False, scene_name=scene_name)

    trainable_token_modes = {"baseline", "baseline_sgm"}
    config.disparity_dir.mkdir(parents=True, exist_ok=True)
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    use_cuda_o4 = backend_status.use_torch and is_cuda_device_name(backend_status.device)

    payload_scene_dirs = (
        discovered_scenes
        if execution_status.selected_mode in trainable_token_modes and scene_name is None and max_scenes is None
        else scenes
    )

    scene_payloads: list[dict[str, Any]] = []
    for scene_dir in payload_scene_dirs:
        left_gray = load_gray(scene_dir / "im0.png")
        right_gray = load_gray(scene_dir / "im1.png")
        left_content_mask = build_o4_content_mask(left_gray)
        right_content_mask = build_o4_content_mask(right_gray)
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
        scene_min_disparity, scene_max_disparity = estimate_o4_scene_disparity_bounds(
            scene_dir,
            tuple(left_gray.shape),
            config,
            o3_config,
        )
        min_token_disparity, max_token_disparity = pixel_disparity_bounds_to_token_bounds(
            scene_min_disparity,
            scene_max_disparity,
            token_span,
            int(left_descriptors.shape[1]),
        )

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
                "left_content_mask": left_content_mask,
                "right_content_mask": right_content_mask,
                "left_descriptors": left_descriptors,
                "right_descriptors": right_descriptors,
                "execution_mode": execution_status.selected_mode,
                "descriptor_source": execution_status.descriptor_source,
                "token_span": token_span,
                "scene_min_disparity": scene_min_disparity,
                "scene_max_disparity": scene_max_disparity,
                "min_token_disparity": min_token_disparity,
                "max_token_disparity": max_token_disparity,
                "ground_truth_tokens": ground_truth_tokens,
            }
        )

    fold_models: dict[int, O4ModelState] = {}
    fold_training_stats: dict[int, tuple[int, O4ModelState]] = {}
    if execution_status.selected_mode in trainable_token_modes:
        save_checkpoint_dir = config.metrics_file.parent / "o4_models"
        load_checkpoint_dir = config.resume_checkpoint_dir if config.resume_checkpoint_dir is not None else save_checkpoint_dir
        for fold_index in range(config.num_folds):
            checkpoint_path = save_checkpoint_dir / f"o4_model_fold{fold_index}.pt"
            load_checkpoint_path = load_checkpoint_dir / f"o4_model_fold{fold_index}.pt"
            loaded_checkpoint = (
                load_o4_model_checkpoint(load_checkpoint_path, config, fold=fold_index, device=backend_status.device)
                if backend_status.use_torch
                else None
            )
            if loaded_checkpoint is not None:
                sample_count, completed_epochs, model_state = loaded_checkpoint
                additional_epochs = max(0, int(config.training_epochs) - int(completed_epochs))
                if additional_epochs > 0 and model_state.torch_model is not None:
                    if use_cuda_o4:
                        training_descriptors, candidate_descriptors = collect_o4_training_samples_torch(
                            scene_payloads,
                            fold_index,
                            config,
                            device=backend_status.device,
                        )
                    else:
                        training_descriptors, candidate_descriptors = collect_o4_training_samples(scene_payloads, fold_index, config)
                    resumed_model = train_o4_torch_model(
                        training_descriptors,
                        candidate_descriptors,
                        model_dim=config.model_dim,
                        hidden_dim=config.encoder_hidden_dim,
                        encoder_layers=config.encoder_layers,
                        epochs=additional_epochs,
                        learning_rate=config.training_learning_rate,
                        weight_decay=config.weight_decay,
                        batch_size=config.training_batch_size,
                        random_seed=int(config.random_seed) + int(completed_epochs),
                        device=backend_status.device,
                        initial_model=model_state.torch_model,
                        epoch_offset=int(completed_epochs),
                        total_epoch_count=int(config.training_epochs),
                    )
                    if resumed_model is not None:
                        model_state = O4ModelState(
                            backend="torch",
                            device=backend_status.device,
                            trained=True,
                            torch_model=resumed_model,
                            parameter_count=int(getattr(resumed_model, "parameter_count", 0)),
                        )
                        sample_count = int(training_descriptors.shape[0])
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        save_o4_model_checkpoint(checkpoint_path, model_state, config, fold=fold_index, sample_count=sample_count)
                        model_state.checkpoint_path = checkpoint_path
                        print(
                            f"Resumed O4 model fold {fold_index}: {checkpoint_path} "
                            f"(completed_epochs={completed_epochs}, total_epochs={config.training_epochs}, samples={sample_count})",
                            flush=True,
                        )
                fold_models[fold_index] = model_state
                fold_training_stats[fold_index] = (sample_count, model_state)
                if additional_epochs <= 0:
                    print(
                        f"Loaded O4 model fold {fold_index}: {load_checkpoint_path} "
                        f"(epochs={completed_epochs}, samples={sample_count}, parameters={model_state.parameter_count})",
                        flush=True,
                    )
                continue

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
            sample_count = int(training_descriptors.shape[0])
            model_state = fold_models[fold_index]
            if model_state.trained:
                extension = ".pt" if model_state.torch_model is not None else ".npz"
                checkpoint_path = config.metrics_file.parent / "o4_models" / f"o4_model_fold{fold_index}{extension}"
                save_o4_model_checkpoint(checkpoint_path, model_state, config, fold=fold_index, sample_count=sample_count)
                model_state.checkpoint_path = checkpoint_path
                print(
                    f"Saved O4 model fold {fold_index}: {checkpoint_path} "
                    f"(samples={sample_count}, parameters={model_state.parameter_count})",
                    flush=True,
                )
            fold_training_stats[fold_index] = (sample_count, model_state)

    metric_rows: list[MetricRow] = []
    for scene_dir in scenes:
        payload = next(item for item in scene_payloads if item["scene_name"] == scene_dir.name)
        if execution_status.selected_mode in trainable_token_modes:
            model_state = fold_models[payload["fold"]]
        else:
            model_state = O4ModelState(
                backend=backend_status.selected_backend,
                device=backend_status.device,
                trained=False,
            )
        use_cuda_scene = is_cuda_device_name(model_state.device)

        if execution_status.selected_mode in trainable_token_modes and model_state.torch_model is not None:
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
        if execution_status.selected_mode in trainable_token_modes:
            training_sample_count, training_state = fold_training_stats[payload["fold"]]
        else:
            training_sample_count = 0
            training_state = model_state
        fallback_used = execution_status.selected_mode in trainable_token_modes and not training_state.trained
        display_source_disparity = None
        raw_transformer_disparity = None
        filtered_raw_transformer_disparity = None

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
                fallback_min_token_disparity, fallback_max_token_disparity = pixel_disparity_bounds_to_token_bounds(
                    int(payload["scene_min_disparity"]),
                    int(payload["scene_max_disparity"]),
                    fallback_token_span,
                    int(fallback_left_tokens.shape[1]),
                )
                fallback_disparity_tokens, fallback_confidence_tokens, _ = predict_token_disparity_torch(
                    fallback_left_tokens,
                    fallback_right_tokens,
                    fallback_min_token_disparity,
                    fallback_max_token_disparity,
                    device=backend_status.device,
                    return_numpy=False,
                )
                raw_transformer_disparity = upsample_token_grid_torch(
                    fallback_disparity_tokens.float() * float(fallback_token_span),
                    fallback_token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                    device=backend_status.device,
                ).detach().cpu().numpy().astype(np.float32)
                fallback_confidence_mask = fallback_confidence_tokens >= float(config.min_confidence)
                fallback_disparity_tokens = fallback_disparity_tokens.masked_fill(~fallback_confidence_mask, 0.0).float()
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
                filtered_raw_transformer_disparity = disparity
                confidence = upsample_token_grid_torch(
                    fallback_confidence_tokens.float(),
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
                fallback_min_token_disparity, fallback_max_token_disparity = pixel_disparity_bounds_to_token_bounds(
                    int(payload["scene_min_disparity"]),
                    int(payload["scene_max_disparity"]),
                    fallback_token_span,
                    int(fallback_left_tokens.shape[1]),
                )
                fallback_disparity_tokens, fallback_confidence_tokens, _ = predict_token_disparity(
                    fallback_left_tokens,
                    fallback_right_tokens,
                    fallback_min_token_disparity,
                    fallback_max_token_disparity,
                )
                raw_transformer_disparity = upsample_token_grid(
                    fallback_disparity_tokens.astype(np.float32) * float(fallback_token_span),
                    fallback_token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                fallback_confidence_mask = fallback_confidence_tokens >= float(config.min_confidence)
                fallback_disparity_tokens = np.where(fallback_confidence_mask, fallback_disparity_tokens, 0.0).astype(np.float32)
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
                filtered_raw_transformer_disparity = disparity
                confidence = upsample_token_grid(
                    fallback_confidence_tokens.astype(np.float32),
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
                right_to_left_disparity, _, right_to_left_scores = predict_token_disparity_torch(
                    right_tokens,
                    left_tokens,
                    min_token_disparity,
                    max_token_disparity,
                    device=model_state.device,
                    target_direction="positive",
                    return_numpy=False,
                )
                raw_refined_token_disparity = refine_token_disparity_torch(
                    token_disparity,
                    token_scores,
                    min_token_disparity,
                    max_token_disparity,
                    device=model_state.device,
                )
                raw_right_to_left_disparity = refine_token_disparity_torch(
                    right_to_left_disparity,
                    right_to_left_scores,
                    min_token_disparity,
                    max_token_disparity,
                    device=model_state.device,
                )
                if config.disparity_regression == "soft_argmax":
                    raw_refined_token_disparity = soft_argmax_disparity_torch(
                        token_scores,
                        min_token_disparity,
                        max_token_disparity,
                        config.softmax_temperature,
                        device=model_state.device,
                    )
                    raw_right_to_left_disparity = soft_argmax_disparity_torch(
                        right_to_left_scores,
                        min_token_disparity,
                        max_token_disparity,
                        config.softmax_temperature,
                        device=model_state.device,
                    )
                if execution_status.selected_mode == "baseline_sgm":
                    import torch as _torch_sgm

                    # Build per-token cost volumes (lower = better) from ViT cosine scores.
                    left_cost_volume = (-token_scores).permute(1, 2, 0).contiguous().to(dtype=_torch_sgm.float32)
                    right_cost_volume = (-right_to_left_scores).permute(1, 2, 0).contiguous().to(dtype=_torch_sgm.float32)
                    left_token_content_mask = build_token_content_mask_torch(
                        payload["left_content_mask"],
                        token_span,
                        tuple(left_cost_volume.shape[:2]),
                        device=model_state.device,
                    )
                    right_token_content_mask = build_token_content_mask_torch(
                        payload["right_content_mask"],
                        token_span,
                        tuple(right_cost_volume.shape[:2]),
                        device=model_state.device,
                    )
                    left_pair_mask = stereo_cost_valid_mask_torch(
                        left_token_content_mask,
                        right_token_content_mask,
                        min_token_disparity,
                        max_token_disparity,
                        "negative",
                        device=model_state.device,
                    )
                    right_pair_mask = stereo_cost_valid_mask_torch(
                        right_token_content_mask,
                        left_token_content_mask,
                        min_token_disparity,
                        max_token_disparity,
                        "positive",
                        device=model_state.device,
                    )
                    left_cost_volume = left_cost_volume.masked_fill(~left_pair_mask, float("inf"))
                    right_cost_volume = right_cost_volume.masked_fill(~right_pair_mask, float("inf"))

                    # Build token-resolution guides via average pooling so SGM edge penalties
                    # honour structure at the same scale as the cost volume.
                    token_height = int(left_cost_volume.shape[0])
                    token_width = int(left_cost_volume.shape[1])
                    left_guide_tokens = average_pool_gray_torch(
                        payload["left_gray"], token_span, device=model_state.device
                    )[:token_height, :token_width]
                    right_guide_tokens = average_pool_gray_torch(
                        payload["right_gray"], token_span, device=model_state.device
                    )[:token_height, :token_width]

                    sgm_left, sgm_left_margin = solve_content_masked_sgm_from_cost_torch(
                        left_cost_volume,
                        left_guide_tokens,
                        min_disparity=int(min_token_disparity),
                        device=model_state.device,
                        content_mask=left_token_content_mask,
                    )
                    sgm_right, _ = solve_content_masked_sgm_from_cost_torch(
                        right_cost_volume,
                        right_guide_tokens,
                        min_disparity=int(min_token_disparity),
                        device=model_state.device,
                        content_mask=right_token_content_mask,
                    )
                    refined_token_disparity = sgm_left.to(dtype=_torch_sgm.float32)
                    right_to_left_disparity = sgm_right.to(dtype=_torch_sgm.float32)
                else:
                    refined_token_disparity = raw_refined_token_disparity
                    right_to_left_disparity = raw_right_to_left_disparity
                    sgm_left_margin = token_confidence
                    left_token_content_mask = build_token_content_mask_torch(
                        payload["left_content_mask"],
                        token_span,
                        tuple(refined_token_disparity.shape),
                        device=model_state.device,
                    )
                raw_transformer_disparity = upsample_token_grid_torch(
                    raw_refined_token_disparity.float() * float(token_span),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                    device=model_state.device,
                ).detach().cpu().numpy().astype(np.float32)
                raw_consistency_mask = left_right_consistency_mask_torch(
                    raw_refined_token_disparity,
                    raw_right_to_left_disparity,
                    config.consistency_threshold,
                    device=model_state.device,
                ) & left_token_content_mask
                raw_filtered_token_disparity = raw_refined_token_disparity.masked_fill(~raw_consistency_mask, 0.0).float()
                if config.token_median_filter_size > 1:
                    raw_token_mask = raw_filtered_token_disparity > 0
                    raw_filtered = median_filter_2d_torch(
                        raw_filtered_token_disparity,
                        config.token_median_filter_size,
                        device=model_state.device,
                    )
                    raw_filtered_token_disparity = raw_filtered.masked_fill(~raw_token_mask, 0.0).float()
                filtered_raw_transformer_disparity = upsample_token_grid_torch(
                    raw_filtered_token_disparity * float(token_span),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                    device=model_state.device,
                ).detach().cpu().numpy().astype(np.float32)
                cleanup_shape = tuple(to_numpy_float32(refined_token_disparity).shape)
                cleanup_guide_tokens = average_pool_gray(payload["left_gray"], token_span)[: cleanup_shape[0], : cleanup_shape[1]]
                cleanup_content_mask = build_token_content_mask(
                    payload["left_content_mask"],
                    token_span,
                    cleanup_shape,
                )
                official_token_disparity = cleanup_o4_token_disparity_like_o3(
                    refined_token_disparity,
                    right_to_left_disparity,
                    cleanup_guide_tokens,
                    cleanup_content_mask,
                    sgm_left_margin,
                    config,
                )
                disparity = upsample_token_grid(
                    official_token_disparity.astype(np.float32) * float(token_span),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                display_source_disparity = disparity
                confidence = upsample_token_grid(
                    to_numpy_float32(token_confidence),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                raw_transformer_disparity = apply_raw_content_mask(raw_transformer_disparity, payload["left_content_mask"])
                disparity = apply_content_mask(disparity, payload["left_content_mask"])
                display_source_disparity = apply_content_mask(display_source_disparity, payload["left_content_mask"])
                confidence = np.where(np.asarray(payload["left_content_mask"], dtype=bool), confidence, 0.0).astype(np.float32)
            else:
                if model_state.torch_model is not None:
                    token_disparity, token_confidence, token_scores = predict_token_disparity_torch(
                        left_tokens,
                        right_tokens,
                        min_token_disparity,
                        max_token_disparity,
                        device=model_state.device,
                    )
                    right_to_left_disparity, _, right_to_left_scores = predict_token_disparity_torch(
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
                    right_to_left_disparity, _, right_to_left_scores = predict_token_disparity(
                        right_tokens,
                        left_tokens,
                        min_token_disparity,
                        max_token_disparity,
                        target_direction="positive",
                    )

                raw_refined_token_disparity = regress_token_disparity(
                    token_disparity,
                    token_scores,
                    min_token_disparity,
                    max_token_disparity,
                    config.disparity_regression,
                    config.softmax_temperature,
                )
                raw_right_to_left_disparity = regress_token_disparity(
                    right_to_left_disparity,
                    right_to_left_scores,
                    min_token_disparity,
                    max_token_disparity,
                    config.disparity_regression,
                    config.softmax_temperature,
                )
                if execution_status.selected_mode == "baseline_sgm":
                    # NumPy / CPU path: ViT cosine scores → cost volume → O3 SGM aggregation.
                    left_cost_volume_np = np.transpose(-np.asarray(token_scores, dtype=np.float32), (1, 2, 0)).copy()
                    right_cost_volume_np = np.transpose(-np.asarray(right_to_left_scores, dtype=np.float32), (1, 2, 0)).copy()
                    left_token_content_mask_np = build_token_content_mask(
                        payload["left_content_mask"],
                        token_span,
                        tuple(left_cost_volume_np.shape[:2]),
                    )
                    right_token_content_mask_np = build_token_content_mask(
                        payload["right_content_mask"],
                        token_span,
                        tuple(right_cost_volume_np.shape[:2]),
                    )
                    left_pair_mask_np = stereo_cost_valid_mask(
                        left_token_content_mask_np,
                        right_token_content_mask_np,
                        min_token_disparity,
                        max_token_disparity,
                        "negative",
                    )
                    right_pair_mask_np = stereo_cost_valid_mask(
                        right_token_content_mask_np,
                        left_token_content_mask_np,
                        min_token_disparity,
                        max_token_disparity,
                        "positive",
                    )
                    left_cost_volume_np = np.where(left_pair_mask_np, left_cost_volume_np, np.inf).astype(np.float32)
                    right_cost_volume_np = np.where(right_pair_mask_np, right_cost_volume_np, np.inf).astype(np.float32)
                    token_height_np = int(left_cost_volume_np.shape[0])
                    token_width_np = int(left_cost_volume_np.shape[1])
                    left_guide_tokens_np = average_pool_gray(payload["left_gray"], token_span)[:token_height_np, :token_width_np]
                    right_guide_tokens_np = average_pool_gray(payload["right_gray"], token_span)[:token_height_np, :token_width_np]
                    refined_token_disparity, left_margin_np = solve_content_masked_sgm_from_cost(
                        left_cost_volume_np,
                        left_guide_tokens_np,
                        min_disparity=int(min_token_disparity),
                        content_mask=left_token_content_mask_np,
                    )
                    right_to_left_disparity_sgm, _ = solve_content_masked_sgm_from_cost(
                        right_cost_volume_np,
                        right_guide_tokens_np,
                        min_disparity=int(min_token_disparity),
                        content_mask=right_token_content_mask_np,
                    )
                    refined_token_disparity = refined_token_disparity.astype(np.float32)
                    right_to_left_disparity = right_to_left_disparity_sgm.astype(np.float32)
                else:
                    refined_token_disparity = raw_refined_token_disparity
                    right_to_left_disparity = raw_right_to_left_disparity
                    left_margin_np = token_confidence
                    left_token_content_mask_np = build_token_content_mask(
                        payload["left_content_mask"],
                        token_span,
                        tuple(np.asarray(refined_token_disparity).shape),
                    )
                raw_transformer_disparity = upsample_token_grid(
                    raw_refined_token_disparity.astype(np.float32) * float(token_span),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                raw_consistency_mask = left_right_consistency_mask(
                    raw_refined_token_disparity,
                    raw_right_to_left_disparity.astype(np.float32),
                    config.consistency_threshold,
                ) & left_token_content_mask_np
                raw_filtered_token_disparity = np.where(raw_consistency_mask, raw_refined_token_disparity, 0.0).astype(np.float32)
                if config.token_median_filter_size > 1:
                    raw_token_mask = raw_filtered_token_disparity > 0
                    raw_filtered = median_filter_2d(raw_filtered_token_disparity, config.token_median_filter_size)
                    raw_filtered_token_disparity = np.where(raw_token_mask, raw_filtered, 0.0).astype(np.float32)
                filtered_raw_transformer_disparity = upsample_token_grid(
                    raw_filtered_token_disparity.astype(np.float32) * float(token_span),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                cleanup_guide_tokens_np = average_pool_gray(payload["left_gray"], token_span)[: refined_token_disparity.shape[0], : refined_token_disparity.shape[1]]
                official_token_disparity = cleanup_o4_token_disparity_like_o3(
                    refined_token_disparity,
                    right_to_left_disparity.astype(np.float32),
                    cleanup_guide_tokens_np,
                    left_token_content_mask_np,
                    left_margin_np,
                    config,
                )
                disparity = upsample_token_grid(
                    official_token_disparity.astype(np.float32) * float(token_span),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                display_source_disparity = disparity
                confidence = upsample_token_grid(
                    np.asarray(token_confidence, dtype=np.float32),
                    token_span,
                    payload["left_gray"].shape[0],
                    payload["left_gray"].shape[1],
                ).astype(np.float32)
                raw_transformer_disparity = apply_raw_content_mask(raw_transformer_disparity, payload["left_content_mask"])
                disparity = apply_content_mask(disparity, payload["left_content_mask"])
                display_source_disparity = apply_content_mask(display_source_disparity, payload["left_content_mask"])
                confidence = np.where(np.asarray(payload["left_content_mask"], dtype=bool), confidence, 0.0).astype(np.float32)

        if raw_transformer_disparity is None:
            raw_transformer_disparity = np.where(np.isfinite(disparity) & (disparity > 0), disparity, 0.0).astype(np.float32)
        raw_transformer_disparity = apply_raw_content_mask(raw_transformer_disparity, payload["left_content_mask"])
        if filtered_raw_transformer_disparity is None:
            filtered_raw_transformer_disparity = raw_transformer_disparity
        filtered_raw_transformer_disparity = apply_content_mask(
            filtered_raw_transformer_disparity,
            payload["left_content_mask"],
        )
        disparity = display_source_disparity if display_source_disparity is not None else filtered_raw_transformer_disparity
        disparity = apply_content_mask(disparity, payload["left_content_mask"])
        disparity = np.where(np.isfinite(disparity) & (disparity > 0), disparity, 0.0).astype(np.float32)
        reliable_confidence_floor = 0.0
        reliable_pixel_count = int(np.count_nonzero(disparity > 0))
        display_disparity = build_o4_preview_disparity(disparity, payload["left_gray"])
        display_disparity = apply_content_mask(display_disparity, payload["left_content_mask"])
        display_confidence = build_o4_display_confidence(confidence, display_disparity)
        display_confidence = np.where(np.asarray(payload["left_content_mask"], dtype=bool), display_confidence, 0.0).astype(np.float32)

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
        write_pfm(disparity_scene_dir / "disp0_transformer_raw_filtered.pfm", filtered_raw_transformer_disparity)
        raw_preview_mask = np.asarray(payload["left_content_mask"], dtype=bool) & np.isfinite(raw_transformer_disparity)
        raw_disparity_preview = colorize_disparity_depth_map(raw_transformer_disparity, raw_preview_mask)
        write_png(disparity_scene_dir / "disp0_transformer_raw.png", raw_disparity_preview)
        filtered_raw_preview = colorize_disparity_depth_map(
            filtered_raw_transformer_disparity,
            filtered_raw_transformer_disparity > 0,
        )
        write_png(disparity_scene_dir / "disp0_transformer_raw_filtered.png", filtered_raw_preview)
        disparity_preview = colorize_disparity_depth_map(display_disparity, display_disparity > 0)
        write_png(disparity_scene_dir / "disp0.png", disparity_preview)
        disparity_readme_name = "disparity_README.txt" if disparity_scene_dir == analysis_scene_dir else "README.txt"
        write_scene_text(
            disparity_scene_dir / disparity_readme_name,
            [
                f"scene: {scene_dir.name}",
                f"generator: {generator_name}",
                f"fold: {scene_fold_map[scene_dir.name]}",
                f"token_grid: {left_tokens.shape[0]}x{left_tokens.shape[1]}",
                f"token_span_pixels: {token_span}",
                "disparity_range_source: scene_adaptive_from_o3_calib_hints",
                f"scene_min_disparity: {payload['scene_min_disparity']}",
                f"scene_max_disparity: {payload['scene_max_disparity']}",
                f"min_token_disparity: {min_token_disparity}",
                f"max_token_disparity: {max_token_disparity}",
                f"requested_backend: {config.backend}",
                f"resolved_backend: {training_state.backend}",
                f"requested_execution_mode: {config.execution_mode}",
                f"resolved_execution_mode: {payload['execution_mode']}",
                f"descriptor_source: {payload['descriptor_source']}",
                "final_disparity_source: O4 transformer token prediction",
                "sgm_detail_fusion_used: no",
                "raw_transformer_disparity: disp0_transformer_raw.pfm",
                "filtered_raw_transformer_disparity: disp0_transformer_raw_filtered.pfm",
                "filtered_raw_transformer_preview: disp0_transformer_raw_filtered.png",
                "filtered_raw_transformer_filtering: left_right_consistency_plus_token_median_without_fill",
                "official_pfm_selection: o3_style_token_cleanup_from_transformer_sgm_prediction",
                "official_pfm_confidence_filter_applied: no",
                "official_pfm_lr_consistency_filter_applied: relaxed_o3_style_local_support",
                f"disparity_regression: {config.disparity_regression}",
                f"runtime_device: {training_state.device}",
                f"model_dim: {config.model_dim}",
                f"encoder_hidden_dim: {config.encoder_hidden_dim}",
                f"encoder_layers: {config.encoder_layers}",
                f"trainable_parameters: {training_state.parameter_count}",
                f"training_epochs: {config.training_epochs}",
                f"training_batch_size: {config.training_batch_size}",
                f"training_samples: {training_sample_count}",
                f"trained_projection: {'yes' if training_state.trained else 'no'}",
                f"o4_model_checkpoint: {training_state.checkpoint_path if training_state.checkpoint_path is not None else 'none'}",
                f"fallback_coarse_matcher_used: {'yes' if fallback_used else 'no'}",
                "content_mask_applied: yes",
                f"official_pfm_confidence_floor: {reliable_confidence_floor:.6f}",
                f"official_pfm_reliable_pixels: {reliable_pixel_count}",
                f"token_median_filter_size: {config.token_median_filter_size}",
                f"speckle_max_size: {config.speckle_max_size}",
                f"speckle_max_diff: {config.speckle_max_diff:.6f}",
                f"fill_invalid_passes: {config.fill_invalid_passes}",
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
                "official_pfm_selection: o3_style_token_cleanup_from_transformer_sgm_prediction",
                "official_pfm_confidence_filter_applied: no",
                "official_pfm_lr_consistency_filter_applied: relaxed_o3_style_local_support",
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
