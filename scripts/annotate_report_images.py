from __future__ import annotations

from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "latex" / "generated" / "report_annotations"

ANNOTATION_SPECS: list[tuple[str, str]] = [
    ("EBU6307_ZIXI_YU_231223210/results/O1b_synthetic_data/artroom2/im1.png", "1"),
    ("EBU6307_ZIXI_YU_231223210/results/O1b_synthetic_data/chess3/im1.png", "2"),
    ("EBU6307_ZIXI_YU_231223210/results/O1b_synthetic_data/ladder1/im1.png", "3"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/artroom2/im0_keypoints.png", "1a"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/artroom2/im1_keypoints.png", "1b"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/artroom2/sift_matches.png", "1c"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/chess3/im0_keypoints.png", "2a"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/chess3/im1_keypoints.png", "2b"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/chess3/sift_matches.png", "2c"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/ladder1/im0_keypoints.png", "3a"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/ladder1/im1_keypoints.png", "3b"),
    ("EBU6307_ZIXI_YU_231223210/results/O2b_sift/ladder1/sift_matches.png", "3c"),
    ("EBU6307_ZIXI_YU_231223210/results/O3b_disparity/artroom2/disp0.png", "1"),
    ("EBU6307_ZIXI_YU_231223210/results/O3b_disparity/chess3/disp0.png", "2"),
    ("EBU6307_ZIXI_YU_231223210/results/O3b_disparity/ladder1/disp0.png", "3"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/artroom2/disp0_transformer_raw.png", "1a"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/artroom2/disp0_transformer_raw_filtered.png", "1b"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/artroom2/disp0.png", "1c"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/chess3/disp0_transformer_raw.png", "2a"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/chess3/disp0_transformer_raw_filtered.png", "2b"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/chess3/disp0.png", "2c"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/ladder1/disp0_transformer_raw.png", "3a"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/ladder1/disp0_transformer_raw_filtered.png", "3b"),
    ("EBU6307_ZIXI_YU_231223210/results/O4b_transformer/ladder1/disp0.png", "3c"),
]


def annotate_image(source_path: Path, output_path: Path, label: str) -> None:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {source_path}")

    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(width, height) / 900.0)
    thickness = max(2, round(font_scale * 2.0))
    outline_thickness = thickness + 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    x_offset = max(12, round(width * 0.03))
    y_offset = max(text_height + baseline + 8, round(height * 0.08))
    x_offset = min(x_offset, max(12, width - text_width - 12))
    y_offset = min(y_offset, max(text_height + baseline + 8, height - 12))

    cv2.putText(
        image,
        label,
        (x_offset, y_offset),
        font,
        font_scale,
        (0, 0, 0),
        outline_thickness,
        cv2.LINE_8,
    )

    cv2.putText(
        image,
        label,
        (x_offset, y_offset),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Unable to write image: {output_path}")


def main() -> None:
    expected_outputs = {
        OUTPUT_ROOT / relative_path
        for relative_path, _ in ANNOTATION_SPECS
    }

    for existing_path in OUTPUT_ROOT.rglob("*"):
        if existing_path.is_file() and existing_path not in expected_outputs:
            existing_path.unlink()

    for relative_path, label in ANNOTATION_SPECS:
        source_path = REPO_ROOT / relative_path
        output_path = OUTPUT_ROOT / relative_path
        annotate_image(source_path, output_path, label)


if __name__ == "__main__":
    main()