from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_parent(dst)
    shutil.copy2(src, dst)
    return True


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    ensure_parent(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_o2_repeatability(metrics_csv: Path, pdf_named_csv: Path) -> None:
    rows = load_rows(metrics_csv)
    if not rows:
        raise FileNotFoundError(f"Missing O2 metrics source: {metrics_csv}")
    exported = []
    for row in rows:
        exported.append(
            {
                "scene": row.get("scene", ""),
                "transform_family": row.get("transform_family", ""),
                "left_keypoints": row.get("left_keypoints", ""),
                "right_keypoints": row.get("right_keypoints", ""),
                "raw_knn_matches": row.get("raw_knn_matches", ""),
                "ratio_test_matches": row.get("ratio_test_matches", ""),
                "repeatable_matches": row.get("repeatable_matches", row.get("ratio_test_matches", "")),
                "repeatability": row.get("repeatability", row.get("repeatability_proxy", "")),
            }
        )
    write_rows(
        pdf_named_csv,
        exported,
        [
            "scene",
            "transform_family",
            "left_keypoints",
            "right_keypoints",
            "raw_knn_matches",
            "ratio_test_matches",
            "repeatable_matches",
            "repeatability",
        ],
    )


def _fit_images_horizontally(images: list[Image.Image], target_height: int, gap: int = 16) -> Image.Image:
    resized = []
    for image in images:
        scale = target_height / image.height
        resized.append(image.resize((int(image.width * scale), target_height), Image.Resampling.LANCZOS))
    total_width = sum(img.width for img in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_width, target_height), (255, 255, 255))
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width + gap
    return canvas


def annotate(image: Image.Image, title: str, lines: list[str]) -> Image.Image:
    font = ImageFont.load_default()
    padding = 24
    text_gap = 8
    draw_probe = ImageDraw.Draw(image)
    text_lines = [title] + lines
    text_heights = []
    max_text_width = 0
    for line in text_lines:
        bbox = draw_probe.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_text_width = max(max_text_width, w)
        text_heights.append(h)
    text_block_height = sum(text_heights) + text_gap * (len(text_lines) - 1)
    width = max(image.width, max_text_width + padding * 2)
    height = image.height + text_block_height + padding * 2
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(image, ((width - image.width) // 2, padding + text_block_height + padding))
    draw = ImageDraw.Draw(canvas)
    y = padding
    for idx, line in enumerate(text_lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text(((width - w) // 2, y), line, fill=(0, 0, 0), font=font)
        y += h + text_gap
    return canvas


def export_o2_examples(metrics_csv: Path, count: int = 3) -> list[Path]:
    rows = load_rows(metrics_csv)
    if not rows:
        raise FileNotFoundError(f"Missing O2 metrics source: {metrics_csv}")

    preferred_families = ["rotation", "affine", "intensity"]
    selected: list[dict[str, str]] = []
    for family in preferred_families:
        family_rows = [row for row in rows if row.get("transform_family") == family]
        if not family_rows:
            continue
        selected.append(max(family_rows, key=lambda r: float(r.get("repeatability", r.get("repeatability_proxy", "0")) or 0.0)))

    if len(selected) < count:
        leftovers = [row for row in rows if row not in selected]
        leftovers.sort(key=lambda r: float(r.get("repeatability", r.get("repeatability_proxy", "0")) or 0.0), reverse=True)
        selected.extend(leftovers[: count - len(selected)])

    outputs: list[Path] = []
    for index, row in enumerate(selected[:count], start=1):
        scene = row["scene"]
        family = row.get("transform_family", "unknown")
        kp_dir = RESULTS / "O2a_sift" / scene
        match_dir = RESULTS / "O2b_sift" / scene
        images = [
            Image.open(kp_dir / "im0_keypoints.png").convert("RGB"),
            Image.open(kp_dir / "im1_keypoints.png").convert("RGB"),
            Image.open(match_dir / "sift_matches.png").convert("RGB"),
        ]
        strip = _fit_images_horizontally(images, target_height=520)
        card = annotate(
            strip,
            title=f"O2 example {index}: {scene} ({family})",
            lines=[
                "Transformation protocol: SIFT on original image first, then detect again after a randomly generated transformation.",
                f"Dominant transform family: {family}",
                f"Repeatable keypoints: {row.get('repeatable_matches', row.get('ratio_test_matches', ''))} / min({row.get('left_keypoints', '')}, {row.get('right_keypoints', '')}) = {row.get('repeatability', row.get('repeatability_proxy', ''))}",
            ],
        )
        out_path = RESULTS / "O2b_sift" / f"example_{index}.jpg"
        ensure_parent(out_path)
        card.save(out_path, quality=95)
        outputs.append(out_path)
    return outputs


def main() -> int:
    o1_metrics = RESULTS / "O1c_synthetic_data" / "SSIM.csv"
    o2_metrics = RESULTS / "O2c_sift" / "metrics.csv"
    o2_pdf_metrics = RESULTS / "O2c_sift" / "Reapitability.csv"

    if not o1_metrics.exists():
        raise FileNotFoundError(f"Missing O1 metrics source: {o1_metrics}")
    if not o2_metrics.exists():
        raise FileNotFoundError(f"Missing O2 metrics source: {o2_metrics}")

    export_o2_repeatability(o2_metrics, o2_pdf_metrics)
    examples = export_o2_examples(o2_metrics, count=3)

    print(f"Verified O1 metrics source: {o1_metrics}")
    print(f"Exported O2 PDF-named metrics: {o2_pdf_metrics}")
    for example in examples:
        print(f"Exported O2 example: {example}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
