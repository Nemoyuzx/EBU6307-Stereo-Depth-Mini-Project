from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import interpolate, ndimage
from scipy.ndimage import gaussian_filter

from common import discover_scenes, ensure_parent, filter_scene_dirs, load_rgb
from config import O1Config
from pfm import read_pfm, write_pfm


def scale_reference_disparity(reference_disparity: Any, shift_pixels: int) -> np.ndarray:
    """把参考视差稳健缩放到目标合成位移范围，同时保留相对深度层次。"""

    disparity = np.asarray(reference_disparity, dtype=np.float32)
    if disparity.ndim == 3:
        disparity = disparity[..., 0]
    if disparity.ndim != 2:
        raise ValueError(f"Expected a single-channel disparity map, got shape {disparity.shape}.")

    target_shift = abs(int(shift_pixels))
    if target_shift == 0:
        return np.zeros_like(disparity, dtype=np.float32)

    disparity = np.where(np.isfinite(disparity), disparity, 0.0)
    disparity = np.maximum(disparity, 0.0)
    valid = disparity > 0.0
    if not np.any(valid):
        return np.zeros_like(disparity, dtype=np.float32)

    low = float(np.percentile(disparity[valid], 5.0))
    high = float(np.percentile(disparity[valid], 95.0))
    if high <= low + 1e-6:
        high = float(np.max(disparity[valid]))
    if high <= 1e-6:
        return np.zeros_like(disparity, dtype=np.float32)

    scaled = np.zeros_like(disparity, dtype=np.float32)
    scaled[valid] = np.clip((disparity[valid] - low) / max(high - low, 1e-6), 0.0, 1.0) * float(target_shift)
    scaled = gaussian_filter(scaled, sigma=0.6)
    scaled = np.where(valid, scaled, 0.0)
    return np.clip(scaled, 0.0, float(target_shift)).astype(np.float32)


def _project_left_to_synthetic_view(image: Any, disparity: Any, shift_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """按缩放后的视差把左图前向投影到合成视角，并返回空洞掩码。"""

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape HxWx3, got {rgb.shape}.")

    disparity_map = np.asarray(disparity, dtype=np.float32)
    if disparity_map.shape != rgb.shape[:2]:
        raise ValueError(f"Disparity shape {disparity_map.shape} does not match image shape {rgb.shape[:2]}.")

    disparity_map = np.where(np.isfinite(disparity_map), disparity_map, 0.0)
    disparity_map = np.maximum(disparity_map, 0.0)

    height, width = disparity_map.shape
    direction = 1 if shift_pixels >= 0 else -1
    y_coords, x_coords = np.indices((height, width))
    target_x = np.clip(np.rint(x_coords - direction * disparity_map).astype(np.int32), 0, width - 1)

    # 远处先投影、近处后覆盖，更接近真实遮挡关系。
    order = np.argsort(disparity_map.reshape(-1), kind="stable")
    projected = np.zeros_like(rgb)
    occupied = np.zeros((height, width), dtype=bool)
    projected[y_coords.reshape(-1)[order], target_x.reshape(-1)[order]] = rgb.reshape(-1, 3)[order]
    occupied[y_coords.reshape(-1)[order], target_x.reshape(-1)[order]] = True
    return projected, ~occupied


def _fill_projection_holes(projected: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """修补重投影留下的空洞。"""

    if not np.any(missing_mask):
        return projected

    filled = projected.astype(np.float32)
    valid = (~missing_mask).astype(np.float32)
    # 用 4 邻域做迭代扩散，让已有像素逐步向空洞内部传播，避免依赖 OpenCV inpaint。
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    for _ in range(16):
        if bool(np.all(valid > 0.5)):
            break
        neighbour_count = ndimage.convolve(valid, kernel, mode="nearest")
        update_mask = (valid < 0.5) & (neighbour_count > 0.0)
        if not np.any(update_mask):
            break
        for channel_index in range(filled.shape[2]):
            neighbour_sum = ndimage.convolve(filled[..., channel_index] * valid, kernel, mode="nearest")
            averaged = neighbour_sum / np.maximum(neighbour_count, 1.0)
            filled[..., channel_index] = np.where(update_mask, averaged, filled[..., channel_index])
        valid = np.where(update_mask, 1.0, valid)

    # 若仍有孤立空洞，再对剩余空洞做基于周围有效像素的线性插值。
    unresolved = valid < 0.5
    if np.any(unresolved):
        sample_mask = ~unresolved
        sample_y, sample_x = np.nonzero(sample_mask)
        query_y, query_x = np.nonzero(unresolved)

        if sample_y.size > 0:
            sample_points = np.column_stack((sample_y, sample_x))
            query_points = np.column_stack((query_y, query_x))
            for channel_index in range(filled.shape[2]):
                sample_values = filled[..., channel_index][sample_mask]
                interpolated = interpolate.griddata(
                    sample_points,
                    sample_values,
                    query_points,
                    method="linear",
                    fill_value=np.nan,
                )
                if np.isnan(interpolated).any():
                    nearest = interpolate.griddata(
                        sample_points,
                        sample_values,
                        query_points[np.isnan(interpolated)],
                        method="nearest",
                    )
                    interpolated[np.isnan(interpolated)] = nearest
                filled[..., channel_index][unresolved] = interpolated

    return np.clip(filled, 0.0, 255.0).round().astype(np.uint8)


def synthesize_depth_aware_stereo(image: Any, reference_disparity: Any, shift_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """用参考视差驱动重投影，生成更接近真实双目的合成视图。"""

    scaled_disparity = scale_reference_disparity(reference_disparity, shift_pixels)
    projected, missing_mask = _project_left_to_synthetic_view(image, scaled_disparity, shift_pixels)
    synthetic = _fill_projection_holes(projected, missing_mask)
    return synthetic, scaled_disparity


def compute_ssim(left_image: Any, synthetic_image: Any) -> float:
    """手工实现一个简化版多通道 SSIM，用于衡量原始左图和合成右图的结构相似性。"""

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
        # 先计算局部均值，再据此构造方差和协方差项。
        mu_x = gaussian_filter(x, sigma=sigma)
        mu_y = gaussian_filter(y, sigma=sigma)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = gaussian_filter(x * x, sigma=sigma) - mu_x_sq
        sigma_y_sq = gaussian_filter(y * y, sigma=sigma) - mu_y_sq
        sigma_xy = gaussian_filter(x * y, sigma=sigma) - mu_xy

        # 这里保留 SSIM 的亮度项和结构项，只是按通道分别求平均。
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / np.maximum(denominator, 1e-12)
        channel_scores.append(float(np.mean(ssim_map)))

    return float(np.mean(channel_scores))


def read_metrics(metrics_file: Path) -> list[dict[str, str]]:
    """读取历史 SSIM 指标，便于增量更新同一个 CSV。"""
    if not metrics_file.exists():
        return []

    # 只回读 O1 自己关心的列，避免历史文件里夹带其他字段影响写回。
    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {"scene": row.get("scene", ""), "shift_pixels": row.get("shift_pixels", ""), "ssim": row.get("ssim", "")}
            for row in reader
            if row.get("scene")
        ]


def copy_if_exists(source: Path, destination: Path) -> bool:
    """若源文件存在则复制到目标位置，并返回是否复制成功。"""
    if not source.exists():
        return False
    # 复制前先补目录，让调用方只关心“要不要复制”。
    ensure_parent(destination)
    shutil.copy2(source, destination)
    return True


def write_metrics(metrics_file: Path, rows: list[dict[str, str | float | int]]) -> None:
    """按 scene 合并新旧 SSIM 指标，避免重复追加同一场景。"""
    existing_rows = read_metrics(metrics_file)
    rows_by_scene = {str(row["scene"]): row for row in rows}
    merged_rows: list[dict[str, str | float | int]] = []

    # 优先保留旧文件顺序，这样增量重跑时 diff 会更稳定。
    for row in existing_rows:
        scene_name = row["scene"]
        replacement = rows_by_scene.pop(scene_name, None)
        merged_rows.append(replacement if replacement is not None else dict(row))

    # 新出现的场景追加到末尾。
    merged_rows.extend(rows_by_scene.values())

    ensure_parent(metrics_file)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "shift_pixels", "ssim"])
        writer.writeheader()
        writer.writerows(merged_rows)


def write_scene_metadata(scene_output_dir: Path, scene_name: str, shift_pixels: int, calib_copied: bool) -> None:
    """为每个合成场景生成 README，说明文件来源与参数。"""
    metadata_path = scene_output_dir / "README.txt"
    # README 主要用于回溯这个合成场景是怎么生成出来的。
    metadata_path.write_text(
        "\n".join(
            [
                f"scene: {scene_name}",
                "generator: depth-aware O1 synthesis",
                "im0.png: original left image copied from the source scene",
                "im1.png: synthetic right image created by disparity-guided reprojection of im0.png",
                f"target_max_shift_pixels: {abs(shift_pixels)}",
                "disp0.pfm: rescaled left-view disparity derived from the source disp0.pfm",
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
        # O1 的验证目标是目录结构和关键文件是否齐全，而不是数值精度评测。
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
    """执行 O1：用参考视差重投影生成合成右图与对应视差图、并写出指标。"""
    discovered_scenes = discover_scenes(config.middlebury_root)
    discovered_count = len(discovered_scenes)
    scenes = filter_scene_dirs(discovered_scenes, scene_name)

    if max_scenes is not None:
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    print(f"Repository root: {config.repo_root}")
    print(f"Middlebury root: {config.middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {discovered_count}")
    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
    print("Scenes to process: " + ", ".join(scene.name for scene in scenes) if scenes else "Scenes to process: none")

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

    config.synthetic_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, str | float | int]] = []
    failed_scenes: list[str] = []
    for scene_dir in scenes:
        # O1 直接复用场景自带的参考视差，把左图重投影成新的合成右图。
        left = load_rgb(scene_dir / "im0.png")
        disparity_path = scene_dir / "disp0.pfm"
        if not disparity_path.exists():
            failed_scenes.append(scene_dir.name)
            print(
                f"Scene {scene_dir.name} is missing disp0.pfm, which is required for depth-aware O1 synthesis.",
                file=sys.stderr,
            )
            continue

        source_disparity = read_pfm(disparity_path)
        synthetic, disparity = synthesize_depth_aware_stereo(left, source_disparity, config.shift_pixels)
        ssim_score = compute_ssim(left, synthetic)
        scene_output_dir = config.synthetic_dir / scene_dir.name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        # 输出布局尽量贴近原始场景结构，方便后续 objective 直接复用。
        copy_if_exists(scene_dir / "im0.png", scene_output_dir / "im0.png")
        Image.fromarray(synthetic).save(scene_output_dir / "im1.png")
        write_pfm(scene_output_dir / "disp0.pfm", disparity)

        calib_copied = copy_if_exists(scene_dir / "calib.txt", scene_output_dir / "calib.txt")
        write_scene_metadata(scene_output_dir, scene_dir.name, config.shift_pixels, calib_copied)
        metric_rows.append({"scene": scene_dir.name, "shift_pixels": config.shift_pixels, "ssim": f"{ssim_score:.6f}"})
        print(
            f"Wrote synthetic scene: {scene_output_dir} "
            f"(calib.txt copied: {'yes' if calib_copied else 'no'}, SSIM: {ssim_score:.6f})"
        )

    if failed_scenes:
        print("Skipped scenes without disp0.pfm: " + ", ".join(failed_scenes), file=sys.stderr)

    if not metric_rows:
        return 1 if failed_scenes else 0

    write_metrics(config.metrics_file, metric_rows)
    print(f"Wrote SSIM summary: {config.metrics_file}")
    return 1 if failed_scenes else 0


if __name__ == '__main__':
    from entry_utils import run_objective_entry

    raise SystemExit(run_objective_entry('o1', __file__))
