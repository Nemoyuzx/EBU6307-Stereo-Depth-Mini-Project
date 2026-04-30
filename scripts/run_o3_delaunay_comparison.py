from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
CODES_DIR = REPO_ROOT / "codes"
codes_dir_text = str(CODES_DIR)
while codes_dir_text in sys.path:
    sys.path.remove(codes_dir_text)
sys.path.insert(0, codes_dir_text)

from common import discover_scenes, evaluate_disparity, filter_scene_dirs, load_gray, normalize_for_preview, write_png, write_scene_text
from config import O3Config, load_config
from o3 import colorize_disparity_depth_map, compute_sift_driven_disparity, estimate_scene_disparity_bounds
from pfm import read_pfm, write_pfm


MetricRow = dict[str, str | int | float]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = REPO_ROOT / "configs" / "dataset_paths.example.yaml"
    parser = argparse.ArgumentParser(description="Run O3 SIFT+Delaunay triangulation comparison against the current O3 output.")
    parser.add_argument("--config", type=Path, default=default_config, help=f"Path to YAML config. Default: {default_config}")
    parser.add_argument("--profile", default="local", help="Config profile name. Default: local")
    parser.add_argument("--max-scenes", type=int, default=None, help="Maximum number of scenes to process. Use 0 for discovery-only.")
    parser.add_argument("--scene-name", default=None, help="Run only one scene by exact directory name.")
    parser.add_argument("--current-o3-dir", type=Path, default=None, help="Directory containing the current O3 scene disp0.pfm files.")
    parser.add_argument("--output-disparity-dir", type=Path, default=REPO_ROOT / "results" / "O3_delaunay_disparity")
    parser.add_argument("--output-analysis-dir", type=Path, default=REPO_ROOT / "results" / "O3_delaunay_analysis")
    parser.add_argument("--output-comparison-dir", type=Path, default=REPO_ROOT / "results" / "O3_delaunay_comparison")
    return parser.parse_args(argv)


def format_metric(value: float | int) -> str:
    numeric = float(value)
    return f"{numeric:.6f}" if numeric >= 0 else "NA"


def compare_disparities(delaunay: Any, current: Any, ground_truth: Any | None) -> dict[str, float | int]:
    delaunay_source = np.asarray(delaunay, dtype=np.float32)
    current_source = np.asarray(current, dtype=np.float32)
    delaunay_valid = np.isfinite(delaunay_source) & (delaunay_source > 0)
    current_valid = np.isfinite(current_source) & (current_source > 0)
    common_valid = delaunay_valid & current_valid

    result: dict[str, float | int] = {
        "delaunay_valid_pixels": int(delaunay_valid.sum()),
        "current_valid_pixels": int(current_valid.sum()),
        "common_valid_pixels": int(common_valid.sum()),
        "delaunay_only_pixels": int((delaunay_valid & ~current_valid).sum()),
        "current_only_pixels": int((current_valid & ~delaunay_valid).sum()),
        "mean_abs_diff_to_current": -1.0,
        "rmse_diff_to_current": -1.0,
        "bad_1px_diff_to_current": -1.0,
        "delaunay_better_gt_pixels": 0,
        "current_better_gt_pixels": 0,
    }
    if bool(np.any(common_valid)):
        diff = np.abs(delaunay_source[common_valid] - current_source[common_valid])
        result["mean_abs_diff_to_current"] = float(diff.mean())
        result["rmse_diff_to_current"] = float(np.sqrt((diff * diff).mean()))
        result["bad_1px_diff_to_current"] = float((diff > 1.0).mean())

    if ground_truth is not None:
        truth = np.asarray(ground_truth, dtype=np.float32)
        gt_valid = np.isfinite(truth) & (truth > 0)
        compare_mask = common_valid & gt_valid
        if bool(np.any(compare_mask)):
            delaunay_error = np.abs(delaunay_source[compare_mask] - truth[compare_mask])
            current_error = np.abs(current_source[compare_mask] - truth[compare_mask])
            result["delaunay_better_gt_pixels"] = int((delaunay_error < current_error).sum())
            result["current_better_gt_pixels"] = int((current_error < delaunay_error).sum())
    return result


def build_difference_preview(delaunay: Any, current: Any) -> tuple[Any, Any]:
    delaunay_source = np.asarray(delaunay, dtype=np.float32)
    current_source = np.asarray(current, dtype=np.float32)
    common_valid = (delaunay_source > 0) & np.isfinite(delaunay_source) & (current_source > 0) & np.isfinite(current_source)
    diff = np.zeros_like(delaunay_source, dtype=np.float32)
    diff[common_valid] = np.abs(delaunay_source[common_valid] - current_source[common_valid])
    preview = normalize_for_preview(diff, diff > 0)
    color = np.stack([preview, np.zeros_like(preview), 255 - preview], axis=2)
    color[~common_valid] = 0
    return diff, color.astype(np.uint8)


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    width = max(1, int(round(image.width * (float(target_height) / max(1, image.height)))))
    return image.resize((width, target_height), resampling)


def compose_side_by_side(left_gray: Any, current_preview: Any, delaunay_preview: Any, difference_preview: Any, scene_name: str) -> Any:
    panels = [
        ("Left image", Image.fromarray(np.asarray(left_gray, dtype=np.uint8)).convert("RGB")),
        ("Current O3", Image.fromarray(np.asarray(current_preview, dtype=np.uint8)).convert("RGB")),
        ("Delaunay", Image.fromarray(np.asarray(delaunay_preview, dtype=np.uint8)).convert("RGB")),
        ("Abs diff", Image.fromarray(np.asarray(difference_preview, dtype=np.uint8)).convert("RGB")),
    ]
    panel_height = 360
    resized = [(label, resize_to_height(image, panel_height)) for label, image in panels]
    gap = 18
    title_height = 58
    label_height = 32
    width = sum(image.width for _, image in resized) + (len(resized) + 1) * gap
    height = title_height + panel_height + label_height + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 18), f"O3 current vs SIFT+Delaunay triangulation: {scene_name}", fill=(20, 20, 20))
    x = gap
    y = title_height
    for label, image in resized:
        draw.text((x, y - 24), label, fill=(60, 60, 60))
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + image.width, y + image.height), outline=(180, 180, 180), width=2)
        x += image.width + gap
    return np.asarray(canvas, dtype=np.uint8)


def write_metrics(path: Path, rows: list[MetricRow]) -> None:
    fieldnames = [
        "scene",
        "left_keypoints",
        "right_keypoints",
        "stereo_matches",
        "min_disparity",
        "max_disparity",
        "delaunay_valid_pixels",
        "current_valid_pixels",
        "common_valid_pixels",
        "delaunay_only_pixels",
        "current_only_pixels",
        "delaunay_mae",
        "delaunay_rmse",
        "delaunay_bad_1px",
        "current_mae",
        "current_rmse",
        "current_bad_1px",
        "mean_abs_diff_to_current",
        "rmse_diff_to_current",
        "bad_1px_diff_to_current",
        "delaunay_better_gt_pixels",
        "current_better_gt_pixels",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary(path: Path, rows: list[MetricRow]) -> None:
    numeric_fields = [
        "delaunay_valid_pixels",
        "current_valid_pixels",
        "common_valid_pixels",
        "delaunay_mae",
        "delaunay_rmse",
        "delaunay_bad_1px",
        "current_mae",
        "current_rmse",
        "current_bad_1px",
        "mean_abs_diff_to_current",
        "rmse_diff_to_current",
        "bad_1px_diff_to_current",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean"])
        writer.writeheader()
        writer.writerow({"metric": "scene_count", "mean": len(rows)})
        for field in numeric_fields:
            values: list[float] = []
            for row in rows:
                try:
                    value = float(row[field])
                except (KeyError, TypeError, ValueError):
                    continue
                if value >= 0:
                    values.append(value)
            mean_value = sum(values) / len(values) if values else -1.0
            writer.writerow({"metric": field, "mean": format_metric(mean_value)})


def run_scene(scene_dir: Path, config: O3Config, current_o3_dir: Path, disparity_dir: Path, analysis_dir: Path) -> MetricRow:
    left_gray = load_gray(scene_dir / "im0.png")
    right_gray = load_gray(scene_dir / "im1.png")
    min_disparity, max_disparity = estimate_scene_disparity_bounds(scene_dir, left_gray.shape, config)
    scene_config = replace(config, num_disparities=max(config.num_disparities, max_disparity + 1))
    delaunay_disparity, stats = compute_sift_driven_disparity(left_gray, right_gray, scene_config)
    delaunay_mask = np.isfinite(delaunay_disparity) & (delaunay_disparity > 0)
    current_path = current_o3_dir / scene_dir.name / "disp0.pfm"
    current_disparity = read_pfm(current_path) if current_path.exists() else np.zeros_like(delaunay_disparity, dtype=np.float32)
    if getattr(current_disparity, "ndim", 0) == 3:
        current_disparity = current_disparity[:, :, 0]
    current_disparity = np.asarray(current_disparity, dtype=np.float32)

    ground_truth = None
    ground_truth_path = scene_dir / "disp0.pfm"
    if ground_truth_path.exists():
        ground_truth = read_pfm(ground_truth_path)
        if getattr(ground_truth, "ndim", 0) == 3:
            ground_truth = ground_truth[:, :, 0]
        ground_truth = np.asarray(ground_truth, dtype=np.float32)

    delaunay_metrics = evaluate_disparity(delaunay_disparity, ground_truth) if ground_truth is not None else {
        "valid_disparity_pixels": int(delaunay_mask.sum()),
        "valid_ground_truth_pixels": 0,
        "mae": -1.0,
        "rmse": -1.0,
        "bad_1px": -1.0,
    }
    current_metrics = evaluate_disparity(current_disparity, ground_truth) if ground_truth is not None else {
        "valid_disparity_pixels": int((current_disparity > 0).sum()),
        "valid_ground_truth_pixels": 0,
        "mae": -1.0,
        "rmse": -1.0,
        "bad_1px": -1.0,
    }
    difference_metrics = compare_disparities(delaunay_disparity, current_disparity, ground_truth)
    difference_map, difference_preview = build_difference_preview(delaunay_disparity, current_disparity)

    scene_disparity_dir = disparity_dir / scene_dir.name
    scene_analysis_dir = analysis_dir / scene_dir.name
    scene_disparity_dir.mkdir(parents=True, exist_ok=True)
    scene_analysis_dir.mkdir(parents=True, exist_ok=True)

    delaunay_preview = colorize_disparity_depth_map(delaunay_disparity, delaunay_mask)
    current_preview = colorize_disparity_depth_map(current_disparity, current_disparity > 0)
    write_pfm(scene_disparity_dir / "disp0.pfm", delaunay_disparity)
    write_png(scene_disparity_dir / "disp0.png", delaunay_preview)
    write_png(scene_analysis_dir / "current_o3.png", current_preview)
    write_png(scene_analysis_dir / "difference_to_current.png", difference_preview)
    write_png(
        scene_analysis_dir / "side_by_side.png",
        compose_side_by_side(left_gray, current_preview, delaunay_preview, difference_preview, scene_dir.name),
    )

    error_map = np.zeros_like(delaunay_disparity, dtype=np.float32)
    if ground_truth is not None:
        valid_error = delaunay_mask & np.isfinite(ground_truth) & (ground_truth > 0)
        error_map[valid_error] = np.abs(delaunay_disparity[valid_error] - ground_truth[valid_error])
    error_preview = normalize_for_preview(error_map, error_map > 0)
    error_color = np.stack([error_preview, error_preview, error_preview], axis=2)
    error_color[error_map <= 0] = 0
    write_png(scene_analysis_dir / "error_map.png", error_color)
    write_scene_text(
        scene_disparity_dir / "README.txt",
        [
            f"scene: {scene_dir.name}",
            "generator: O3 SIFT seed Delaunay triangulation interpolation",
            "current_o3_reference: SIFT+SGM output in results/O3a_disparity",
            f"left_keypoints: {stats['left_keypoints']}",
            f"right_keypoints: {stats['right_keypoints']}",
            f"raw_matches: {stats['raw_matches']}",
            f"ratio_matches: {stats['ratio_matches']}",
            f"mutual_matches: {stats['mutual_matches']}",
            f"stereo_matches: {stats['stereo_matches']}",
            f"min_disparity: {min_disparity}",
            f"max_disparity: {max_disparity}",
            f"delaunay_valid_pixels: {delaunay_metrics['valid_disparity_pixels']}",
            f"current_valid_pixels: {current_metrics['valid_disparity_pixels']}",
            f"mean_abs_diff_to_current: {format_metric(difference_metrics['mean_abs_diff_to_current'])}",
        ],
    )

    return {
        "scene": scene_dir.name,
        "left_keypoints": stats["left_keypoints"],
        "right_keypoints": stats["right_keypoints"],
        "stereo_matches": stats["stereo_matches"],
        "min_disparity": min_disparity,
        "max_disparity": max_disparity,
        "delaunay_valid_pixels": delaunay_metrics["valid_disparity_pixels"],
        "current_valid_pixels": current_metrics["valid_disparity_pixels"],
        "common_valid_pixels": difference_metrics["common_valid_pixels"],
        "delaunay_only_pixels": difference_metrics["delaunay_only_pixels"],
        "current_only_pixels": difference_metrics["current_only_pixels"],
        "delaunay_mae": format_metric(delaunay_metrics["mae"]),
        "delaunay_rmse": format_metric(delaunay_metrics["rmse"]),
        "delaunay_bad_1px": format_metric(delaunay_metrics["bad_1px"]),
        "current_mae": format_metric(current_metrics["mae"]),
        "current_rmse": format_metric(current_metrics["rmse"]),
        "current_bad_1px": format_metric(current_metrics["bad_1px"]),
        "mean_abs_diff_to_current": format_metric(difference_metrics["mean_abs_diff_to_current"]),
        "rmse_diff_to_current": format_metric(difference_metrics["rmse_diff_to_current"]),
        "bad_1px_diff_to_current": format_metric(difference_metrics["bad_1px_diff_to_current"]),
        "delaunay_better_gt_pixels": difference_metrics["delaunay_better_gt_pixels"],
        "current_better_gt_pixels": difference_metrics["current_better_gt_pixels"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        app_config = load_config(args.config, args.profile)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    current_o3_dir = args.current_o3_dir or app_config.o3.disparity_dir
    scenes = filter_scene_dirs(discover_scenes(app_config.middlebury_root), args.scene_name)
    if args.max_scenes is not None:
        if args.max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[: args.max_scenes]

    print(f"Repository root: {app_config.repo_root}")
    print(f"Middlebury root: {app_config.middlebury_root}")
    print(f"Current O3 disparity dir: {current_o3_dir}")
    print(f"Delaunay disparity dir: {args.output_disparity_dir}")
    print(f"Delaunay analysis dir: {args.output_analysis_dir}")
    print(f"Comparison dir: {args.output_comparison_dir}")
    print("Scenes to process: " + (", ".join(scene.name for scene in scenes) if scenes else "none"))

    if args.max_scenes == 0:
        print("Discovery-only mode completed without processing.")
        return 0
    if not scenes:
        print("No scenes matched the requested filter.", file=sys.stderr)
        return 1

    args.output_disparity_dir.mkdir(parents=True, exist_ok=True)
    args.output_analysis_dir.mkdir(parents=True, exist_ok=True)
    args.output_comparison_dir.mkdir(parents=True, exist_ok=True)

    rows: list[MetricRow] = []
    for scene_dir in scenes:
        row = run_scene(scene_dir, app_config.o3, current_o3_dir, args.output_disparity_dir, args.output_analysis_dir)
        rows.append(row)
        print(
            f"Wrote Delaunay comparison: {scene_dir.name} "
            f"(delaunay_mae={row['delaunay_mae']}, current_mae={row['current_mae']}, "
            f"mean_abs_diff={row['mean_abs_diff_to_current']})"
        )

    metrics_file = args.output_comparison_dir / "metrics.csv"
    summary_file = args.output_comparison_dir / "summary.csv"
    write_metrics(metrics_file, rows)
    write_summary(summary_file, rows)
    print(f"Wrote Delaunay comparison metrics: {metrics_file}")
    print(f"Wrote Delaunay comparison summary: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
