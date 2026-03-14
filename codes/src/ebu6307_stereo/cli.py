from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pfm import write_pfm


@dataclass(frozen=True)
class O1Config:
    repo_root: Path
    middlebury_root: Path
    synthetic_dir: Path
    metrics_file: Path
    shift_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal O1 stereo baseline: discover Middlebury scenes, synthesize a shifted image, and report SSIM."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--profile",
        default="local",
        help="Config profile to use when the YAML contains profile blocks such as local/remote. Default: local.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of discovered scenes to process. Use 0 to report discovery only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report config and discovered scenes without writing outputs.",
    )
    return parser.parse_args()


def load_config(config_path: Path, profile: str) -> O1Config:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the config. Install the project environment first.") from exc

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    profile_block = raw_config.get(profile) if isinstance(raw_config.get(profile), dict) else None
    result_block = raw_config.get("results", {}) if isinstance(raw_config.get("results"), dict) else {}
    o1_block = raw_config.get("o1", {}) if isinstance(raw_config.get("o1"), dict) else {}
    repo_root_value = raw_config.get("repo_root") or (profile_block or {}).get("repo_root")
    repo_root = resolve_path(config_path.resolve().parents[1], repo_root_value) if repo_root_value else config_path.resolve().parents[1]

    middlebury_value = (
        raw_config.get("middlebury_root")
        or (profile_block or {}).get("middlebury_root")
    )
    if not middlebury_value:
        raise ValueError(
            f"Config {config_path} does not define middlebury_root at the top level or under profile '{profile}'."
        )

    synthetic_value = (
        result_block.get("o1_synthetic_dir")
        or result_block.get("o1b_synthetic_dir")
        or "results/O1b_synthetic_data"
    )
    metrics_value = (
        result_block.get("o1_metrics_file")
        or result_block.get("o1c_metrics_file")
        or "results/O1c_synthetic_data/SSIM.csv"
    )
    shift_pixels = int(o1_block.get("shift_pixels", 8))

    return O1Config(
        repo_root=repo_root,
        middlebury_root=resolve_path(repo_root, middlebury_value),
        synthetic_dir=resolve_path(repo_root, synthetic_value),
        metrics_file=resolve_path(repo_root, metrics_value),
        shift_pixels=shift_pixels,
    )


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def discover_scenes(middlebury_root: Path) -> list[Path]:
    if not middlebury_root.exists():
        return []

    scenes: list[Path] = []
    for candidate in sorted(path for path in middlebury_root.iterdir() if path.is_dir()):
        if (candidate / "im0.png").exists() and (candidate / "im1.png").exists():
            scenes.append(candidate)
    return scenes


def load_rgb(path: Path) -> Any:
    from PIL import Image
    import numpy as np

    return np.asarray(Image.open(path).convert("RGB"))


def synthesize_shift(image: Any, shift_pixels: int) -> Any:
    import numpy as np

    # Shift the reference image horizontally to form a crude synthetic right view.
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
    import numpy as np

    disparity = np.zeros((height, width), dtype=np.float32)
    if shift_pixels == 0 or abs(shift_pixels) >= width:
        return disparity

    if shift_pixels > 0:
        disparity[:, shift_pixels:] = float(shift_pixels)
    else:
        disparity[:, : width + shift_pixels] = float(-shift_pixels)
    return disparity


def compute_ssim(left_image: Any, synthetic_image: Any) -> float:
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            left_image,
            synthetic_image,
            channel_axis=2,
            data_range=255,
        )
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    ensure_parent(destination)
    shutil.copy2(source, destination)
    return True


def write_metrics(metrics_file: Path, rows: list[dict[str, str | float | int]]) -> None:
    ensure_parent(metrics_file)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "shift_pixels", "ssim"])
        writer.writeheader()
        writer.writerows(rows)


def write_scene_metadata(scene_output_dir: Path, scene_name: str, shift_pixels: int, calib_copied: bool) -> None:
    metadata_path = scene_output_dir / "README.txt"
    metadata_path.write_text(
        "\n".join(
            [
                f"scene: {scene_name}",
                "generator: minimal O1 baseline",
                "im0.png: original left image copied from the source scene",
                "im1.png: synthetic right image created by horizontal shift",
                f"shift_pixels: {shift_pixels}",
                f"calib.txt: {'copied' if calib_copied else 'missing in source scene'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_o1(config: O1Config, max_scenes: int | None, dry_run: bool) -> int:
    scenes = discover_scenes(config.middlebury_root)
    discovered_count = len(scenes)

    if max_scenes is not None:
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    print(f"Repository root: {config.repo_root}")
    print(f"Middlebury root: {config.middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {discovered_count}")
    if scenes:
        print("Scenes to process: " + ", ".join(scene.name for scene in scenes))
    else:
        print("Scenes to process: none")

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

    if dry_run or max_scenes == 0:
        print("Dry run requested; no outputs were written.")
        return 0

    config.synthetic_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, str | float | int]] = []
    for scene_dir in scenes:
        from PIL import Image

        left = load_rgb(scene_dir / "im0.png")
        synthetic = synthesize_shift(left, config.shift_pixels)
        disparity = synthesize_disparity(left.shape[0], left.shape[1], config.shift_pixels)
        scene_output_dir = config.synthetic_dir / scene_dir.name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        left_output_path = scene_output_dir / "im0.png"
        right_output_path = scene_output_dir / "im1.png"
        disparity_output_path = scene_output_dir / "disp0.pfm"

        copy_if_exists(scene_dir / "im0.png", left_output_path)
        Image.fromarray(synthetic).save(right_output_path)
        write_pfm(disparity_output_path, disparity)

        calib_copied = copy_if_exists(scene_dir / "calib.txt", scene_output_dir / "calib.txt")
        write_scene_metadata(scene_output_dir, scene_dir.name, config.shift_pixels, calib_copied)
        metric_rows.append(
            {
                "scene": scene_dir.name,
                "shift_pixels": config.shift_pixels,
                "ssim": f"{compute_ssim(left, synthetic):.6f}",
            }
        )
        print(f"Wrote synthetic scene: {scene_output_dir} (calib.txt copied: {'yes' if calib_copied else 'no'})")

    write_metrics(config.metrics_file, metric_rows)
    print(f"Wrote SSIM summary: {config.metrics_file}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config, args.profile)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    return run_o1(config, args.max_scenes, args.dry_run)
