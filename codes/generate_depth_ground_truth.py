from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common import discover_scenes, filter_scene_dirs, normalize_for_preview, write_png, write_scene_text
from pfm import read_pfm, write_pfm


@dataclass(frozen=True)
class MiddleburyCalibration:
    fx: float
    baseline: float
    doffs: float
    width: int | None = None
    height: int | None = None


def _parse_calibration_value(raw: str) -> float:
    return float(raw.strip())


def _parse_camera_matrix_first_value(raw: str) -> float:
    cleaned = raw.strip()
    if not cleaned.startswith("[") or not cleaned.endswith("]"):
        raise ValueError(f"Unsupported camera matrix format: {raw!r}")
    first_row = cleaned[1:-1].split(";")[0].strip()
    first_value = first_row.split()[0]
    return float(first_value)


def parse_middlebury_calibration(path: Path) -> MiddleburyCalibration:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        values[key.strip()] = raw_value.strip()

    if "cam0" not in values:
        raise ValueError(f"Calibration file is missing cam0: {path}")
    if "baseline" not in values:
        raise ValueError(f"Calibration file is missing baseline: {path}")

    return MiddleburyCalibration(
        fx=_parse_camera_matrix_first_value(values["cam0"]),
        baseline=_parse_calibration_value(values["baseline"]),
        doffs=_parse_calibration_value(values.get("doffs", "0")),
        width=int(float(values["width"])) if "width" in values else None,
        height=int(float(values["height"])) if "height" in values else None,
    )


def disparity_to_depth(disparity: Any, calibration: MiddleburyCalibration) -> Any:
    source = np.asarray(disparity, dtype=np.float32)
    if source.ndim == 3:
        source = source[:, :, 0]
    if source.ndim != 2:
        raise ValueError("disparity_to_depth expects a 2D disparity map or a PFM grayscale array.")

    denominator = source + float(calibration.doffs)
    valid = np.isfinite(source) & (source > 0) & np.isfinite(denominator) & (denominator > 0)
    depth = np.zeros(source.shape, dtype=np.float32)
    if bool(np.any(valid)):
        depth[valid] = (float(calibration.fx) * float(calibration.baseline)) / denominator[valid]
    return depth


def colorize_depth_preview(depth: Any, valid_mask: Any) -> Any:
    valid = np.asarray(valid_mask, dtype=bool)
    normalized = normalize_for_preview(depth, valid).astype(np.float32) / 255.0
    # Depth increases with distance, but O4 disparity previews use warm colors for nearer pixels.
    # Invert the normalized depth so near stays red and far stays blue.
    normalized = np.where(valid, 1.0 - normalized, 0.0).astype(np.float32)
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


def generate_scene_depth_ground_truth(scene_dir: Path, output_dir: Path) -> dict[str, str | int | float]:
    calibration = parse_middlebury_calibration(scene_dir / "calib.txt")
    disparity = read_pfm(scene_dir / "disp0.pfm")
    depth = disparity_to_depth(disparity, calibration)
    valid = np.isfinite(depth) & (depth > 0)

    scene_output_dir = output_dir / scene_dir.name
    write_pfm(scene_output_dir / "depth0.pfm", depth)
    write_png(scene_output_dir / "depth0.png", colorize_depth_preview(depth, valid))
    write_scene_text(
        scene_output_dir / "metadata.txt",
        [
            f"scene={scene_dir.name}",
            "source_disparity=disp0.pfm",
            "source_calibration=calib.txt",
            "formula=depth = fx * baseline / (disparity + doffs)",
            "png_preview=rgb_depth_colormap_near_red_far_blue_invalid_black",
            f"fx={calibration.fx}",
            f"baseline={calibration.baseline}",
            f"doffs={calibration.doffs}",
            "depth_unit=same_as_baseline",
        ],
    )

    if bool(np.any(valid)):
        minimum = float(depth[valid].min())
        maximum = float(depth[valid].max())
        mean = float(depth[valid].mean())
    else:
        minimum = -1.0
        maximum = -1.0
        mean = -1.0

    return {
        "scene": scene_dir.name,
        "fx": float(calibration.fx),
        "baseline": float(calibration.baseline),
        "doffs": float(calibration.doffs),
        "valid_depth_pixels": int(valid.sum()),
        "min_depth": minimum,
        "max_depth": maximum,
        "mean_depth": mean,
    }


def write_summary(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "fx",
                "baseline",
                "doffs",
                "valid_depth_pixels",
                "min_depth",
                "max_depth",
                "mean_depth",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate depth ground truth from Middlebury disp0.pfm and calib.txt.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory containing Middlebury scene folders.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/O4_depth_ground_truth"),
        help="Output directory for generated depth truth. Default: results/O4_depth_ground_truth",
    )
    parser.add_argument("--scene-name", default=None, help="Generate depth truth only for this scene name.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit generated scenes. Use 0 to report discovery only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dirs = filter_scene_dirs(discover_scenes(args.dataset_root), args.scene_name)
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[: max(0, int(args.max_scenes))]

    print(f"Discovered {len(scene_dirs)} scene(s) under {args.dataset_root}")
    if args.max_scenes == 0:
        return 0

    rows: list[dict[str, str | int | float]] = []
    for scene_dir in scene_dirs:
        if not (scene_dir / "disp0.pfm").exists() or not (scene_dir / "calib.txt").exists():
            print(f"Skipping {scene_dir.name}: missing disp0.pfm or calib.txt")
            continue
        rows.append(generate_scene_depth_ground_truth(scene_dir, args.output_dir))
        print(f"Generated depth ground truth for {scene_dir.name}")

    write_summary(args.output_dir / "summary.csv", rows)
    print(f"Wrote summary to {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
