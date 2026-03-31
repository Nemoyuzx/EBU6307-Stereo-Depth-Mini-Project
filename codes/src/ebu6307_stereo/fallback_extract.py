from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    """解析提取临时左右视频帧所需的命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Extract one temporary left/right frame pair from stereo videos into a "
            "scene-like folder for fallback engineering validation."
        )
    )
    parser.add_argument("--left-video", type=Path, required=True, help="Path to the left stereo video.")
    parser.add_argument("--right-video", type=Path, required=True, help="Path to the right stereo video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive a temporary scene-like folder.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Zero-based frame index to extract from both videos. Default: 0.",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help="Optional output folder name. Default: derived from the video filenames.",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Print the planned extraction target without writing files.",
    )
    return parser.parse_args()


def require_existing_file(path: Path, label: str) -> None:
    """确保输入视频文件存在。"""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def derive_scene_name(left_video: Path, right_video: Path) -> str:
    """根据左右视频文件名生成默认场景目录名。"""
    return f"tmp_fallback_{left_video.stem}_{right_video.stem}"


def read_frame(video_path: Path, frame_index: int):
    """从指定视频中读取某一帧。"""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError(
                f"Could not seek to frame {frame_index} in video: {video_path}"
            )

        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Could not read frame {frame_index} from video: {video_path}"
            )
        return frame
    finally:
        capture.release()


def write_readme(
    readme_path: Path,
    left_video: Path,
    right_video: Path,
    frame_index: int,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> None:
    """写出临时场景 README，记录来源并强调其不是最终实验数据。"""
    readme_path.write_text(
        "\n".join(
            [
                "Temporary fallback engineering stereo input",
                "",
                "Status: NOT final assignment data and NOT a Middlebury-format source scene.",
                "Purpose: temporary O1 engineering validation only.",
                "",
                f"left_video: {left_video}",
                f"right_video: {right_video}",
                f"frame_index: {frame_index}",
                f"im0.png_shape_bgr: {left_shape}",
                f"im1.png_shape_bgr: {right_shape}",
                "",
                "Files:",
                "- im0.png: frame extracted from the left stereo video",
                "- im1.png: frame extracted from the right stereo video",
                "- README.txt: provenance and temporary-use label",
                "",
                "Do not mix this folder with final assignment datasets, reported metrics, or final results.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """执行临时左右帧提取流程。"""
    args = parse_args()

    if args.frame_index < 0:
        raise ValueError("--frame-index must be >= 0.")

    require_existing_file(args.left_video, "Left video")
    require_existing_file(args.right_video, "Right video")

    scene_name = args.scene_name or derive_scene_name(args.left_video, args.right_video)
    scene_dir = args.output_dir / scene_name

    print(f"left_video={args.left_video}")
    print(f"right_video={args.right_video}")
    print(f"frame_index={args.frame_index}")
    print(f"scene_dir={scene_dir}")

    if args.describe_only:
        print("describe_only=true")
        return 0


    left_frame = read_frame(args.left_video, args.frame_index)
    right_frame = read_frame(args.right_video, args.frame_index)

    scene_dir.mkdir(parents=True, exist_ok=True)
    left_output = scene_dir / "im0.png"
    right_output = scene_dir / "im1.png"

    if not cv2.imwrite(str(left_output), left_frame):
        raise RuntimeError(f"Could not write image: {left_output}")
    if not cv2.imwrite(str(right_output), right_frame):
        raise RuntimeError(f"Could not write image: {right_output}")

    write_readme(
        readme_path=scene_dir / "README.txt",
        left_video=args.left_video,
        right_video=args.right_video,
        frame_index=args.frame_index,
        left_shape=tuple(left_frame.shape),
        right_shape=tuple(right_frame.shape),
    )

    print("wrote:")
    print(f"  {left_output}")
    print(f"  {right_output}")
    print(f"  {scene_dir / 'README.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
