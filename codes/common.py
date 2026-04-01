from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


OFFICIAL_MIDDLEBURY_2021_SCENES = {
    "artroom1",
    "artroom2",
    "bandsaw1",
    "bandsaw2",
    "chess1",
    "chess2",
    "chess3",
    "curule1",
    "curule2",
    "curule3",
    "ladder1",
    "ladder2",
    "octogons1",
    "octogons2",
    "pendulum1",
    "pendulum2",
    "podium1",
    "skates1",
    "skates2",
    "skiboots1",
    "skiboots2",
    "skiboots3",
    "traproom1",
    "traproom2",
}
EXCLUDED_DATASET_DIR_NAMES = {"data", "o4_tiny_scene"}


def discover_scenes(middlebury_root: Path) -> list[Path]:
    """扫描 Middlebury 根目录，只返回项目真正要处理的标准场景目录。"""
    if not middlebury_root.exists():
        return []

    scenes: list[Path] = []
    for candidate in sorted(path for path in middlebury_root.iterdir() if path.is_dir()):
        if candidate.name in EXCLUDED_DATASET_DIR_NAMES:
            continue
        if candidate.name not in OFFICIAL_MIDDLEBURY_2021_SCENES:
            continue
        if (candidate / "im0.png").exists() and (candidate / "im1.png").exists():
            scenes.append(candidate)
    return scenes


def filter_scene_dirs(scene_dirs: list[Path], scene_name: str | None) -> list[Path]:
    """按场景名过滤目录列表；scene_name 为空时直接返回原列表。"""
    if scene_name is None:
        return scene_dirs
    return [path for path in scene_dirs if path.name == scene_name]


def load_rgb(path: Path) -> Any:
    """读取 RGB 图像。优先走 Pillow；若环境缺少 Pillow，则退回 macOS 的 sips 转换流程。"""

    return np.asarray(Image.open(path).convert("RGB"))


def load_gray(path: Path) -> Any:
    """读取灰度图像，供只需要单通道亮度信息的目标模块复用。"""

    return np.asarray(Image.open(path).convert("L"))


def read_image_with_sips(path: Path) -> Any:
    """使用 macOS sips 把原图转成 BMP，再走纯 Python BMP 解析，作为无 Pillow/OpenCV 时的兜底方案。"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        bmp_path = Path(tmp_dir) / f"{path.stem}.bmp"
        subprocess.run(
            ["sips", "-s", "format", "bmp", str(path), "--out", str(bmp_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return read_bmp(bmp_path)


def write_png(path: Path, image: Any) -> None:
    """写出 PNG 预览图。优先用 Pillow；若缺失则先写 BMP，再借助 sips 转成 PNG。"""

    ensure_parent(path)
    image = np.asarray(image, dtype=np.uint8)

    Image.fromarray(image).save(path, format="PNG")


def read_bmp(path: Path) -> Any:
    """读取 24-bit BMP 文件并转成 RGB numpy 数组。"""

    data = path.read_bytes()
    if data[:2] != b"BM":
        raise ValueError(f"Unsupported BMP header: {path}")

    pixel_offset = int.from_bytes(data[10:14], "little")
    dib_size = int.from_bytes(data[14:18], "little")
    if dib_size < 40:
        raise ValueError(f"Unsupported BMP DIB header size in {path}: {dib_size}")

    width = int.from_bytes(data[18:22], "little", signed=True)
    height = int.from_bytes(data[22:26], "little", signed=True)
    planes = int.from_bytes(data[26:28], "little")
    bits_per_pixel = int.from_bytes(data[28:30], "little")
    compression = int.from_bytes(data[30:34], "little")

    if planes != 1 or bits_per_pixel != 24 or compression != 0:
        raise ValueError(f"Unsupported BMP encoding in {path}: {bits_per_pixel}-bit compression={compression}")

    row_stride = ((abs(width) * 3 + 3) // 4) * 4
    rows = []
    for row_index in range(abs(height)):
        start = pixel_offset + row_index * row_stride
        end = start + abs(width) * 3
        row = np.frombuffer(data[start:end], dtype=np.uint8).reshape(abs(width), 3)
        rows.append(row[:, ::-1])

    image = np.stack(rows, axis=0)
    if height > 0:
        image = image[::-1]
    return image


def write_bmp(path: Path, image: Any) -> None:
    """把灰度图或 RGB 图编码为最简单的 24-bit BMP 文件。"""

    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape for BMP export: {image.shape}")

    height, width, _ = image.shape
    row_stride = ((width * 3 + 3) // 4) * 4
    pixel_array_size = row_stride * height
    file_size = 14 + 40 + pixel_array_size

    header = bytearray()
    header.extend(b"BM")
    header.extend(file_size.to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((54).to_bytes(4, "little"))
    header.extend((40).to_bytes(4, "little"))
    header.extend(width.to_bytes(4, "little", signed=True))
    header.extend(height.to_bytes(4, "little", signed=True))
    header.extend((1).to_bytes(2, "little"))
    header.extend((24).to_bytes(2, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend(pixel_array_size.to_bytes(4, "little"))
    header.extend((2835).to_bytes(4, "little", signed=True))
    header.extend((2835).to_bytes(4, "little", signed=True))
    header.extend((0).to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))

    rows: list[bytes] = []
    padding = b"\x00" * (row_stride - width * 3)
    for row in image[::-1]:
        rows.append(row[:, ::-1].tobytes() + padding)

    ensure_parent(path)
    path.write_bytes(bytes(header) + b"".join(rows))


def ensure_parent(path: Path) -> None:
    """确保目标路径的父目录存在，避免写文件时报目录不存在错误。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_scene_text(path: Path, lines: list[str]) -> None:
    """把说明性文本统一写成 UTF-8 文本文件，并自动补结尾换行。"""
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_for_preview(disparity: Any, mask: Any) -> Any:
    """把任意浮点视差/误差图缩放到 0~255，方便生成可视化预览。"""

    preview = np.zeros(disparity.shape, dtype=np.uint8)
    finite_mask = np.asarray(mask, dtype=bool) & np.isfinite(disparity)
    if bool(np.any(finite_mask)):
        values = np.asarray(disparity[finite_mask], dtype=np.float32)
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum > minimum:
            normalized = (values - minimum) * (255.0 / (maximum - minimum))
        else:
            normalized = np.zeros_like(values, dtype=np.float32)
        preview[finite_mask] = np.clip(np.ravel(normalized), 0.0, 255.0).astype(np.uint8)
    return preview


def evaluate_disparity(predicted: Any, ground_truth: Any) -> dict[str, float | int]:
    """按有效像素区域评估预测视差与真值视差的误差指标。"""

    predicted_valid = predicted > 0
    ground_truth_valid = np.isfinite(ground_truth) & (ground_truth > 0)
    valid_mask = predicted_valid & ground_truth_valid

    if not bool(np.any(valid_mask)):
        return {
            "valid_disparity_pixels": int(predicted_valid.sum()),
            "valid_ground_truth_pixels": int(ground_truth_valid.sum()),
            "mae": -1.0,
            "rmse": -1.0,
            "bad_1px": -1.0,
        }

    error = np.abs(predicted[valid_mask] - ground_truth[valid_mask])
    return {
        "valid_disparity_pixels": int(predicted_valid.sum()),
        "valid_ground_truth_pixels": int(ground_truth_valid.sum()),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "bad_1px": float((error > 1.0).mean()),
    }
