from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .common import discover_scenes, ensure_parent, filter_scene_dirs, load_rgb
from .config import O1Config
from .pfm import write_pfm


MetricValue = str | float | int
MetricRow = dict[str, MetricValue]


def synthesize_shift(image: Any, shift_pixels: int) -> Any:
    """把左图沿水平方向平移，合成一个最简右图，用于 O1 的工程化验证。"""

    shifted = np.zeros_like(image)
    if shift_pixels == 0:
        shifted[:] = image
        return shifted

    width = image.shape[1]
    if abs(shift_pixels) >= width:
        return shifted

    if shift_pixels > 0:
        shifted[:, : width - shift_pixels, :] = image[:, shift_pixels:, :]
    else:
        shifted[:, -shift_pixels:, :] = image[:, : width + shift_pixels, :]
    return shifted


def synthesize_disparity(height: int, width: int, shift_pixels: int) -> Any:
    """根据固定平移量构造理想视差图，作为 O1 输出的基准真值。"""

    disparity = np.zeros((height, width), dtype=np.float32)
    if shift_pixels == 0 or abs(shift_pixels) >= width:
        return disparity

    if shift_pixels > 0:
        disparity[:, shift_pixels:] = float(shift_pixels)
    else:
        disparity[:, : width + shift_pixels] = float(-shift_pixels)
    return disparity


def resolve_torch_device(device: str) -> tuple[Any | None, str, bool]:
    """解析 O1 设备选择；当 torch 不可用时回退到 CPU，并返回是否实际用上 CUDA。"""

    normalized = (device or "auto").strip().lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported O1 device: {device}")

    try:
        import torch
    except Exception:
        return None, "cpu(no-torch)", False

    if normalized == "cpu":
        return torch.device("cpu"), "cpu", False
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("O1 device was forced to CUDA, but torch.cuda.is_available() is False.")
        return torch.device("cuda"), f"cuda:{torch.cuda.current_device()}", True

    if torch.cuda.is_available():
        return torch.device("cuda"), f"cuda:{torch.cuda.current_device()}", True
    return torch.device("cpu"), "cpu", False


def _torch_kmeans(features: Any, num_clusters: int, max_iters: int = 25, seed: int = 42) -> Any:
    """使用 torch 在当前 device 上执行简化版 K-Means。"""

    import torch

    if features.ndim != 2:
        raise ValueError(f"Expected features with shape NxD, got {tuple(features.shape)}")

    num_samples = int(features.shape[0])
    if num_samples == 0:
        raise RuntimeError("K-Means received zero samples.")

    actual_clusters = max(1, min(int(num_clusters), num_samples))
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    initial_indices = torch.randperm(num_samples, generator=generator, device=features.device)[:actual_clusters]
    centroids = features[initial_indices].clone()
    assignments = torch.zeros(num_samples, dtype=torch.long, device=features.device)

    for _ in range(max_iters):
        distances = torch.cdist(features, centroids)
        new_assignments = torch.argmin(distances, dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments

        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(assignments, minlength=actual_clusters).to(features.dtype)
        new_centroids.index_add_(0, assignments, features)
        nonzero = counts > 0
        if torch.any(nonzero):
            new_centroids[nonzero] = new_centroids[nonzero] / counts[nonzero].unsqueeze(1)
        if torch.any(~nonzero):
            replacement_indices = torch.randperm(num_samples, generator=generator, device=features.device)[: int((~nonzero).sum().item())]
            new_centroids[~nonzero] = features[replacement_indices]
        centroids = new_centroids

    distances = torch.cdist(features, centroids)
    assignments = torch.argmin(distances, dim=1)
    return assignments


def _torch_resize_gray(gray_image: Any, size: tuple[int, int], *, device: Any, mode: str = "bilinear") -> Any:
    """在 torch device 上缩放单通道图。"""

    import torch
    import torch.nn.functional as F

    width, height = size
    source = torch.from_numpy(np.asarray(gray_image, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    resized = F.interpolate(source, size=(height, width), mode=mode, align_corners=False if mode != "nearest" else None)
    return resized.squeeze(0).squeeze(0).contiguous()


def superpixel_clustering_depth(
    image: Any,
    num_superpixels: int = 400,
    num_clusters: int = 12,
    device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    """使用 SLIC 超像素和 K-Means 做粗粒度区域划分，并生成启发式深度图。

    GPU/CPU 说明：
    - SLIC 超像素依赖 OpenCV ximgproc，目前仍在 CPU 上执行；这是 O1 superpixel_kmeans 主路径里当前最难去掉的 CPU 依赖。
    - 超像素特征聚合、K-Means 聚类、深度图回填、平滑、以及 GPU 路上的深度图缩放改为 torch 张量实现，可在 CUDA 上加速。
    - 若 torch/CUDA 不可用，则自动回退到 CPU torch / NumPy 路径。

    速度说明：
    - 为了让正式 24 场景可在远端稳定跑完，这里只对“超像素深度估计”使用内部工作分辨率；
      原始图像不会被改写，最终 DIBR 仍在原分辨率上执行。
    """

    import cv2

    rgb_image = np.asarray(image, dtype=np.uint8)
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape HxWx3, got {rgb_image.shape}")

    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None or not hasattr(ximgproc, "createSuperpixelSLIC"):
        raise RuntimeError("OpenCV ximgproc is unavailable. Install opencv-contrib-python to use superpixel clustering.")

    height, width = rgb_image.shape[:2]
    max_working_side = 512
    scale = min(1.0, float(max_working_side) / float(max(height, width)))
    if scale < 1.0:
        work_width = max(64, int(round(width * scale)))
        work_height = max(64, int(round(height * scale)))
        work_image = cv2.resize(rgb_image, (work_width, work_height), interpolation=cv2.INTER_AREA)
    else:
        work_image = rgb_image
        work_height, work_width = height, width

    region_size = max(1, int(np.sqrt((work_height * work_width) / max(1, num_superpixels))))
    lab_img = cv2.cvtColor(work_image, cv2.COLOR_RGB2LAB)
    slic = ximgproc.createSuperpixelSLIC(
        lab_img,
        algorithm=ximgproc.MSLIC,
        region_size=region_size,
        ruler=10.0,
    )
    slic.iterate(10)
    slic.enforceLabelConnectivity(10)

    labels = slic.getLabels().astype(np.int64)
    num_labels = int(slic.getNumberOfSuperpixels())
    if num_labels <= 0:
        raise RuntimeError("SLIC failed to produce any superpixels.")

    torch_device, resolved_device, using_cuda = resolve_torch_device(device)
    if torch_device is None:
        # 没有 torch 时，保留原始 CPU 方案作为最终兜底。
        from sklearn.cluster import KMeans

        features: list[list[float]] = []
        centroids_y: list[int] = []
        for label_index in range(num_labels):
            mask = (labels == label_index).astype(np.uint8)
            if np.count_nonzero(mask) == 0:
                features.append([0.0, 0.0, 0.0, 0.0])
                centroids_y.append(0)
                continue
            mean_color = cv2.mean(lab_img, mask=mask)[:3]
            moments = cv2.moments(mask, binaryImage=True)
            centroid_y = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else 0
            normalized_y = (centroid_y / max(1, work_height)) * 100.0
            features.append([float(mean_color[0]), float(mean_color[1]), float(mean_color[2]), float(normalized_y)])
            centroids_y.append(centroid_y)

        actual_clusters = max(1, min(int(num_clusters), num_labels))
        cluster_labels = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10).fit_predict(
            np.asarray(features, dtype=np.float32)
        )
        cluster_depths = np.zeros(actual_clusters, dtype=np.float32)
        for cluster_index in range(actual_clusters):
            y_coords = [centroids_y[i] for i in range(num_labels) if int(cluster_labels[i]) == cluster_index]
            if y_coords:
                cluster_depths[cluster_index] = float((np.mean(y_coords) / max(1, work_height)) * 255.0)
        depth_map = np.zeros((work_height, work_width), dtype=np.uint8)
        for label_index in range(num_labels):
            depth_map[labels == label_index] = int(np.clip(cluster_depths[int(cluster_labels[label_index])], 0.0, 255.0))
        depth_map = cv2.bilateralFilter(depth_map, d=15, sigmaColor=80, sigmaSpace=80)
        if depth_map.shape != (height, width):
            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
        return depth_map, {
            "device": resolved_device,
            "using_cuda": using_cuda,
            "superpixels": num_labels,
            "working_resolution": f"{work_width}x{work_height}",
            "gpu_steps": [],
            "cpu_steps": ["SLIC superpixel segmentation", "feature aggregation", "K-Means clustering", "depth smoothing"],
        }

    import torch
    import torch.nn.functional as F

    labels_t = torch.from_numpy(labels.reshape(-1)).to(device=torch_device, dtype=torch.long)
    lab_t = torch.from_numpy(lab_img.reshape(-1, 3).astype(np.float32)).to(torch_device)

    y_coords = torch.arange(work_height, device=torch_device, dtype=torch.float32).unsqueeze(1).expand(work_height, work_width).reshape(-1)

    counts = torch.bincount(labels_t, minlength=num_labels).clamp_min(1).to(torch.float32)
    color_sums = torch.zeros((num_labels, 3), dtype=torch.float32, device=torch_device)
    color_sums.index_add_(0, labels_t, lab_t)
    mean_colors = color_sums / counts.unsqueeze(1)

    y_sums = torch.zeros(num_labels, dtype=torch.float32, device=torch_device)
    y_sums.index_add_(0, labels_t, y_coords)
    mean_y = y_sums / counts
    normalized_y = (mean_y / max(1, work_height)) * 100.0
    features = torch.cat((mean_colors, normalized_y.unsqueeze(1)), dim=1)

    cluster_labels = _torch_kmeans(features, num_clusters=num_clusters)
    actual_clusters = max(1, min(int(num_clusters), num_labels))

    cluster_depth_sums = torch.zeros(actual_clusters, dtype=torch.float32, device=torch_device)
    cluster_depth_sums.index_add_(0, cluster_labels, mean_y)
    cluster_counts = torch.bincount(cluster_labels, minlength=actual_clusters).clamp_min(1).to(torch.float32)
    cluster_depths = (cluster_depth_sums / cluster_counts) / max(1, work_height) * 255.0

    label_depths = cluster_depths[cluster_labels]
    depth_flat = label_depths[labels_t]
    depth_map_t = depth_flat.reshape(work_height, work_width).clamp(0.0, 255.0)

    # 这里用高斯平滑近似原 bilateral 的温和平滑，保持主体思路不变，同时可在 GPU 上执行。
    kernel_size = 9
    sigma = 2.0
    coords = torch.arange(kernel_size, device=torch_device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, kernel_size, kernel_size)
    depth_map_t = F.conv2d(
        depth_map_t.unsqueeze(0).unsqueeze(0),
        kernel_2d,
        padding=kernel_size // 2,
    ).squeeze(0).squeeze(0)

    if (work_height, work_width) != (height, width):
        depth_map_t = _torch_resize_gray(depth_map_t, (width, height), device=torch_device, mode="bilinear")
    depth_map = depth_map_t.round().clamp(0.0, 255.0).to(torch.uint8).detach().cpu().numpy()
    return depth_map, {
        "device": resolved_device,
        "using_cuda": using_cuda,
        "superpixels": num_labels,
        "working_resolution": f"{work_width}x{work_height}",
        "gpu_steps": [
            "superpixel feature aggregation",
            "K-Means clustering",
            "depth map back-fill",
            "Gaussian depth smoothing",
            "depth upsampling",
        ] if using_cuda else [],
        "cpu_steps": ["SLIC superpixel segmentation", "final tensor->numpy materialization for file output"] + ([] if using_cuda else ["torch feature aggregation", "torch K-Means clustering", "torch depth smoothing", "torch depth upsampling"]),
    }


def generate_stereo_dibr(image: Any, depth_map: Any, baseline_shift: int = 20, device: str = "auto") -> tuple[Any, Any, dict[str, Any]]:
    """根据启发式深度图做简单 DIBR 投影，并修补投影空洞。

    GPU/CPU 说明：
    - 投影散射、空洞检测、邻域扩散修补尽量在 torch 上执行，可落到 CUDA。
    - 若无 torch，则使用原始 CPU NumPy + OpenCV inpaint 兜底。
    - 当前仍不可避免的 CPU 部分主要是最终写盘前的 tensor->numpy 物化；无 torch 时则退回 CPU 投影与 OpenCV inpaint。
    """

    rgb_image = np.asarray(image, dtype=np.uint8)
    depth = np.asarray(depth_map, dtype=np.uint8)
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape HxWx3, got {rgb_image.shape}")
    if depth.shape != rgb_image.shape[:2]:
        raise ValueError(f"Depth map shape {depth.shape} does not match image shape {rgb_image.shape[:2]}")

    height, width, _ = rgb_image.shape
    if baseline_shift == 0:
        info = {"device": "cpu", "using_cuda": False, "gpu_steps": [], "cpu_steps": ["identity copy"]}
        return rgb_image.copy(), rgb_image.copy(), info

    torch_device, resolved_device, using_cuda = resolve_torch_device(device)
    if torch_device is None:
        import cv2

        left_img = np.zeros_like(rgb_image)
        right_img = np.zeros_like(rgb_image)
        left_mask = np.full((height, width), 255, dtype=np.uint8)
        right_mask = np.full((height, width), 255, dtype=np.uint8)

        y_coords, x_coords = np.indices((height, width))
        x_flat = x_coords.reshape(-1)
        y_flat = y_coords.reshape(-1)
        depth_flat = depth.reshape(-1)
        colors_flat = rgb_image.reshape(-1, 3)

        sort_index = np.argsort(depth_flat, kind="stable")
        x_sorted = x_flat[sort_index]
        y_sorted = y_flat[sort_index]
        depth_sorted = depth_flat[sort_index]
        colors_sorted = colors_flat[sort_index]
        disparity = ((depth_sorted.astype(np.float32) / 255.0) * abs(baseline_shift)).astype(np.int32)

        x_left = np.clip(x_sorted + disparity, 0, width - 1)
        x_right = np.clip(x_sorted - disparity, 0, width - 1)

        left_img[y_sorted, x_left] = colors_sorted
        right_img[y_sorted, x_right] = colors_sorted
        left_mask[y_sorted, x_left] = 0
        right_mask[y_sorted, x_right] = 0

        left_inpainted = cv2.inpaint(left_img, left_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        right_inpainted = cv2.inpaint(right_img, right_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return left_inpainted, right_inpainted, {
            "device": resolved_device,
            "using_cuda": using_cuda,
            "gpu_steps": [],
            "cpu_steps": ["DIBR projection", "OpenCV inpaint"],
        }

    import torch
    import torch.nn.functional as F

    rgb_t = torch.from_numpy(rgb_image.astype(np.float32)).to(torch_device)
    depth_t = torch.from_numpy(depth.astype(np.float32)).to(torch_device)
    disparity = torch.round((depth_t / 255.0) * float(abs(baseline_shift))).to(torch.long)

    y_coords = torch.arange(height, device=torch_device, dtype=torch.long).unsqueeze(1).expand(height, width)
    x_coords = torch.arange(width, device=torch_device, dtype=torch.long).unsqueeze(0).expand(height, width)

    def _scatter(target_x: Any) -> tuple[Any, Any]:
        target_x = target_x.clamp(0, width - 1)
        linear_idx = (y_coords * width + target_x).reshape(-1)
        depth_flat = depth_t.reshape(-1)
        color_flat = rgb_t.reshape(-1, 3)
        sort_index = torch.argsort(depth_flat, stable=True)
        linear_idx = linear_idx[sort_index]
        depth_sorted = depth_flat[sort_index]
        color_sorted = color_flat[sort_index]

        canvas = torch.zeros((height * width, 3), dtype=torch.float32, device=torch_device)
        mask = torch.zeros(height * width, dtype=torch.bool, device=torch_device)
        canvas[linear_idx] = color_sorted
        mask[linear_idx] = True
        return canvas.reshape(height, width, 3), mask.reshape(height, width)

    left_img_t, left_mask_t = _scatter(x_coords + disparity)
    right_img_t, right_mask_t = _scatter(x_coords - disparity)

    def _fill_holes(view: Any, valid_mask: Any, passes: int = 6) -> Any:
        filled = view.permute(2, 0, 1).unsqueeze(0).contiguous()
        valid = valid_mask.unsqueeze(0).unsqueeze(0)
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=torch_device,
        ).view(1, 1, 3, 3)
        for _ in range(passes):
            if bool(valid.all()):
                break
            valid_f = valid.to(dtype=torch.float32)
            neighbour_count = F.conv2d(valid_f, kernel, padding=1)
            neighbour_sum = F.conv2d(filled * valid_f, kernel.expand(3, 1, 3, 3), padding=1, groups=3)
            averaged = neighbour_sum / neighbour_count.clamp_min(1.0)
            update_mask = (~valid) & (neighbour_count > 0)
            filled = torch.where(update_mask.expand_as(filled), averaged, filled)
            valid = valid | update_mask
        return filled.squeeze(0).permute(1, 2, 0).contiguous()

    left_filled = _fill_holes(left_img_t, left_mask_t)
    right_filled = _fill_holes(right_img_t, right_mask_t)
    left_img = left_filled.clamp(0.0, 255.0).round().to(torch.uint8).detach().cpu().numpy()
    right_img = right_filled.clamp(0.0, 255.0).round().to(torch.uint8).detach().cpu().numpy()
    return left_img, right_img, {
        "device": resolved_device,
        "using_cuda": using_cuda,
        "gpu_steps": ["DIBR projection scatter", "hole mask construction", "conv-based iterative hole filling"] if using_cuda else [],
        "cpu_steps": ["final tensor->numpy materialization for file output"] if using_cuda else ["torch DIBR projection", "torch hole filling"],
    }


def synthesize_superpixel_stereo(
    image: Any,
    shift_pixels: int,
    num_superpixels: int = 400,
    num_clusters: int = 12,
    device: str = "auto",
) -> tuple[Any, Any, dict[str, Any]]:
    """基于超像素和颜色聚类生成启发式深度，再合成一对虚拟立体图。"""

    if shift_pixels == 0:
        height, width = image.shape[:2]
        return np.asarray(image, dtype=np.uint8).copy(), np.zeros((height, width), dtype=np.float32), {
            "method": "superpixel_kmeans",
            "device": "cpu",
            "using_cuda": False,
            "gpu_steps": [],
            "cpu_steps": ["identity copy"],
            "superpixels": 0,
        }

    depth_map, depth_info = superpixel_clustering_depth(
        image,
        num_superpixels=num_superpixels,
        num_clusters=num_clusters,
        device=device,
    )
    left_view, right_view, dibr_info = generate_stereo_dibr(image, depth_map, baseline_shift=abs(shift_pixels), device=device)
    synthetic = right_view if shift_pixels >= 0 else left_view
    disparity = (depth_map.astype(np.float32) / 255.0) * float(abs(shift_pixels))
    return synthetic, disparity, {
        "method": "superpixel_kmeans",
        "device": depth_info["device"] if depth_info.get("using_cuda") else dibr_info["device"],
        "using_cuda": bool(depth_info.get("using_cuda") or dibr_info.get("using_cuda")),
        "gpu_steps": depth_info.get("gpu_steps", []) + dibr_info.get("gpu_steps", []),
        "cpu_steps": depth_info.get("cpu_steps", []) + dibr_info.get("cpu_steps", []),
        "superpixels": depth_info.get("superpixels", 0),
    }


def compute_ssim(left_image: Any, synthetic_image: Any) -> float:
    """手工实现一个简化版多通道 SSIM，用于衡量原始左图和合成右图的结构相似性。"""

    from scipy.ndimage import gaussian_filter

    left = np.asarray(left_image, dtype=np.float64)
    right = np.asarray(synthetic_image, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("SSIM inputs must have the same shape.")

    if left.ndim == 2:
        left = left[..., None]
        right = right[..., None]

    data_range = 255.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    sigma = 1.5

    channel_scores: list[float] = []
    for channel_index in range(left.shape[2]):
        x = left[..., channel_index]
        y = right[..., channel_index]
        mu_x = gaussian_filter(x, sigma=sigma)
        mu_y = gaussian_filter(y, sigma=sigma)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = gaussian_filter(x * x, sigma=sigma) - mu_x_sq
        sigma_y_sq = gaussian_filter(y * y, sigma=sigma) - mu_y_sq
        sigma_xy = gaussian_filter(x * y, sigma=sigma) - mu_xy

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / np.maximum(denominator, 1e-12)
        channel_scores.append(float(np.mean(ssim_map)))

    return float(np.mean(channel_scores))


def read_metrics(metrics_file: Path) -> list[MetricRow]:
    """读取历史 SSIM 指标，便于增量更新同一个 CSV。"""
    if not metrics_file.exists():
        return []

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "scene": row.get("scene", ""),
                "shift_pixels": row.get("shift_pixels", ""),
                "ssim": row.get("ssim", ""),
                "method": row.get("method", ""),
                "device": row.get("device", ""),
                "used_gpu": row.get("used_gpu", ""),
            }
            for row in reader
            if row.get("scene")
        ]


def copy_if_exists(source: Path, destination: Path) -> bool:
    """若源文件存在则复制到目标位置，并返回是否复制成功。"""
    if not source.exists():
        return False
    ensure_parent(destination)
    shutil.copy2(source, destination)
    return True


def write_metrics(metrics_file: Path, rows: list[MetricRow]) -> None:
    """按 scene+method 合并新旧 SSIM 指标，避免重复追加同一组合。"""
    existing_rows = read_metrics(metrics_file)
    rows_by_key = {(str(row["scene"]), str(row.get("method", ""))): row for row in rows}
    merged_rows: list[MetricRow] = []

    for row in existing_rows:
        key = (str(row["scene"]), str(row.get("method", "")))
        replacement = rows_by_key.pop(key, None)
        merged_rows.append(replacement if replacement is not None else row)

    merged_rows.extend(rows_by_key.values())

    ensure_parent(metrics_file)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "shift_pixels", "ssim", "method", "device", "used_gpu"])
        writer.writeheader()
        writer.writerows(merged_rows)


def build_output_paths(config: O1Config) -> tuple[Path, Path]:
    synthetic_dir = config.synthetic_dir
    metrics_file = config.metrics_file
    tag = (config.output_tag or "").strip()
    if not tag:
        return synthetic_dir, metrics_file

    synthetic_dir = synthetic_dir.with_name(f"{synthetic_dir.name}_{tag}")
    metrics_file = metrics_file.with_name(f"{metrics_file.stem}_{tag}{metrics_file.suffix}")
    return synthetic_dir, metrics_file


def write_scene_metadata(
    scene_output_dir: Path,
    scene_name: str,
    shift_pixels: int,
    calib_copied: bool,
    synthesis_method: str,
    runtime_info: dict[str, Any],
) -> None:
    """为每个合成场景生成 README，说明文件来源与参数。"""
    metadata_path = scene_output_dir / "README.txt"
    gpu_steps = runtime_info.get("gpu_steps", [])
    cpu_steps = runtime_info.get("cpu_steps", [])
    metadata_path.write_text(
        "\n".join(
            [
                f"scene: {scene_name}",
                f"generator: O1 ({synthesis_method})",
                "im0.png: original left image copied from the source scene",
                (
                    "im1.png: synthetic right image created by horizontal shift"
                    if synthesis_method == "shift"
                    else "im1.png: synthetic right image created by superpixel depth estimation + K-Means + DIBR"
                ),
                f"shift_pixels: {shift_pixels}",
                f"device: {runtime_info.get('device', 'cpu')}",
                f"used_gpu: {'yes' if runtime_info.get('using_cuda') else 'no'}",
                f"gpu_steps: {', '.join(gpu_steps) if gpu_steps else 'none'}",
                f"cpu_steps: {', '.join(cpu_steps) if cpu_steps else 'none'}",
                f"calib.txt: {'copied' if calib_copied else 'missing in source scene'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def validate_results(synthetic_dir: Path, scene_name: str | None = None) -> int:
    """检查 O1 输出目录结构是否完整。"""
    required_files = ("im0.png", "im1.png", "disp0.pfm")
    optional_files = ("calib.txt",)

    print(f"Validating synthetic results directory: {synthetic_dir}")
    if not synthetic_dir.exists():
        print(f"Synthetic results directory not found: {synthetic_dir}", file=sys.stderr)
        return 1

    legacy_files = sorted(path for path in synthetic_dir.iterdir() if path.is_file() and not path.name.startswith("."))
    scene_dirs = sorted(path for path in synthetic_dir.iterdir() if path.is_dir())
    scene_dirs = filter_scene_dirs(scene_dirs, scene_name)

    if scene_name is not None:
        print(f"Scene filter: {scene_name}")

    if legacy_files:
        print("Unexpected flat files found directly under the synthetic results root:")
        for path in legacy_files:
            print(f"  - {path.name}")
    else:
        print("Unexpected flat files found directly under the synthetic results root: none")

    if not scene_dirs:
        print("Scene folders found: none")
    else:
        print(f"Scene folders found: {len(scene_dirs)}")

    if scene_name is not None and not scene_dirs:
        print(f"No result scene directories matched --scene-name {scene_name!r}.", file=sys.stderr)
        return 1

    missing_any = False
    for scene_dir in scene_dirs:
        missing = [name for name in required_files if not (scene_dir / name).exists()]
        optional_present = [name for name in optional_files if (scene_dir / name).exists()]
        optional_missing = [name for name in optional_files if not (scene_dir / name).exists()]

        if missing:
            missing_any = True
            print(f"[MISSING] {scene_dir.name}: missing required files: {', '.join(missing)}")
        else:
            print(f"[OK] {scene_dir.name}: required files present")

        if optional_present:
            print(f"  optional present: {', '.join(optional_present)}")
        if optional_missing:
            print(f"  optional missing: {', '.join(optional_missing)}")

    if legacy_files or missing_any:
        print("Validation status: issues found")
        return 1

    print("Validation status: all checked scene folders contain the expected required files")
    return 0


def run(config: O1Config, max_scenes: int | None, dry_run: bool, scene_name: str | None) -> int:
    """执行 O1：发现场景、生成合成右图与视差图、写出指标。"""
    discovered_scenes = discover_scenes(config.middlebury_root)
    discovered_count = len(discovered_scenes)
    scenes = filter_scene_dirs(discovered_scenes, scene_name)

    if max_scenes is not None:
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    output_synthetic_dir, output_metrics_file = build_output_paths(config)

    print(f"Repository root: {config.repo_root}")
    print(f"Middlebury root: {config.middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {discovered_count}")
    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
    print("Scenes to process: " + ", ".join(scene.name for scene in scenes) if scenes else "Scenes to process: none")
    print(f"O1 synthesis method: {config.method}")
    print(f"O1 device preference: {config.device}")
    print(f"O1 synthetic output dir: {output_synthetic_dir}")
    print(f"O1 metrics file: {output_metrics_file}")

    if not config.middlebury_root.exists():
        if dry_run or max_scenes == 0:
            print("Middlebury root does not exist yet. Discovery-only mode completed without processing.")
            return 0
        print(
            f"Middlebury root not found: {config.middlebury_root}\n"
            "Place dataset scenes there, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    if discovered_count == 0:
        if dry_run or max_scenes == 0:
            print("No valid scenes found. Discovery-only mode completed without processing.")
            return 0
        print(
            f"No scene directories containing im0.png and im1.png were found under {config.middlebury_root}.\n"
            "Add Middlebury scenes, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    if scene_name is not None and not scenes:
        print(f"No discovered scenes matched --scene-name {scene_name!r} under {config.middlebury_root}.", file=sys.stderr)
        return 1

    if dry_run or max_scenes == 0:
        print("Dry run requested; no outputs were written.")
        return 0

    if config.method not in {"shift", "superpixel_kmeans"}:
        raise ValueError(f"Unsupported O1 method: {config.method}")

    from PIL import Image

    output_synthetic_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[MetricRow] = []
    for scene_dir in scenes:
        left = load_rgb(scene_dir / "im0.png")
        if config.method == "shift":
            synthetic = synthesize_shift(left, config.shift_pixels)
            disparity = synthesize_disparity(left.shape[0], left.shape[1], config.shift_pixels)
            runtime_info = {
                "method": "shift",
                "device": "cpu",
                "using_cuda": False,
                "gpu_steps": [],
                "cpu_steps": ["horizontal image shift", "constant disparity generation"],
                "superpixels": 0,
            }
        else:
            synthetic, disparity, runtime_info = synthesize_superpixel_stereo(
                left,
                config.shift_pixels,
                num_superpixels=config.superpixel_count,
                num_clusters=config.superpixel_cluster_count,
                device=config.device,
            )

        scene_output_dir = output_synthetic_dir / scene_dir.name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        copy_if_exists(scene_dir / "im0.png", scene_output_dir / "im0.png")
        Image.fromarray(synthetic).save(scene_output_dir / "im1.png")
        write_pfm(scene_output_dir / "disp0.pfm", disparity)

        calib_copied = copy_if_exists(scene_dir / "calib.txt", scene_output_dir / "calib.txt")
        write_scene_metadata(scene_output_dir, scene_dir.name, config.shift_pixels, calib_copied, config.method, runtime_info)
        ssim_value = compute_ssim(left, synthetic)
        metric_rows.append(
            {
                "scene": scene_dir.name,
                "shift_pixels": config.shift_pixels,
                "ssim": f"{ssim_value:.6f}",
                "method": config.method,
                "device": runtime_info.get("device", "cpu"),
                "used_gpu": "yes" if runtime_info.get("using_cuda") else "no",
            }
        )
        print(
            f"Wrote synthetic scene: {scene_output_dir} "
            f"(ssim={ssim_value:.6f}, device={runtime_info.get('device', 'cpu')}, used_gpu={'yes' if runtime_info.get('using_cuda') else 'no'})"
        )

    write_metrics(output_metrics_file, metric_rows)
    print(f"Wrote SSIM summary: {output_metrics_file}")
    return 0
