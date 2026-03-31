from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .o4_dinov2 import resolve_dinov2_checkpoint_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数，统一入口参数命名。"""
    parser = argparse.ArgumentParser(
        description="Minimal stereo assignment CLI with stable O1/O2/O3 behavior and a minimal runnable O4 baseline."
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
        "--objective",
        choices=("o1", "o2", "o3", "o4"),
        default="o1",
        help="Assignment objective to run. Default: o1.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of discovered scenes to process. Use 0 to report discovery only.",
    )
    parser.add_argument(
        "--scene-name",
        help="Process or validate only the scene directory with this exact name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report config and discovered scenes without writing outputs.",
    )
    parser.add_argument(
        "--validate-results",
        action="store_true",
        help="Validate the configured output directory for the selected objective without modifying any files.",
    )
    parser.add_argument(
        "--o1-method",
        choices=("shift", "superpixel_kmeans"),
        default=None,
        help="Override the O1 synthesis method.",
    )
    parser.add_argument(
        "--o1-superpixel-count",
        type=int,
        default=None,
        help="Override the O1 superpixel count when --o1-method superpixel_kmeans is used.",
    )
    parser.add_argument(
        "--o1-superpixel-cluster-count",
        type=int,
        default=None,
        help="Override the O1 K-Means cluster count when --o1-method superpixel_kmeans is used.",
    )
    parser.add_argument(
        "--o1-device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="Override the O1 execution device preference.",
    )
    parser.add_argument(
        "--o1-output-tag",
        default=None,
        help="Append a tag to the configured O1 output folder / SSIM CSV filename to avoid overwrite.",
    )
    parser.add_argument(
        "--o4-execution-mode",
        choices=("baseline", "dinov2_cost_volume"),
        default=None,
        help="Override the O4 execution mode. Only used with --objective o4.",
    )
    parser.add_argument(
        "--o4-dinov2-model",
        default=None,
        help="Override the O4 DINOv2 model selector, for example facebook/dinov2-base or dinov2_vitb14_reg.",
    )
    parser.add_argument(
        "--o4-dinov2-checkpoint",
        type=Path,
        default=None,
        help="Override the explicit local O4 DINOv2 checkpoint path. The weights are loaded from this file, and model code must come from a local dinov2 Python install or checkout.",
    )
    parser.add_argument(
        "--o4-dinov2-repo",
        type=Path,
        default=None,
        help="Override the local path added to PYTHONPATH before importing dinov2.hub.backbones. Point this at a local DINOv2 checkout root or package parent directory.",
    )
    parser.add_argument(
        "--o4-regression-mode",
        choices=("quadratic", "soft_argmax"),
        default=None,
        help="Override O4 disparity regression mode. Only used with --objective o4.",
    )
    return parser.parse_args()


def main() -> int:
    """加载配置并根据 objective 分发到 O1/O2/O3/O4。"""
    args = parse_args()
    try:
        config = load_config(args.config, args.profile)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.objective == "o1" and (
        args.o1_method is not None
        or args.o1_superpixel_count is not None
        or args.o1_superpixel_cluster_count is not None
        or args.o1_device is not None
        or args.o1_output_tag is not None
    ):
        config = replace(
            config,
            o1=replace(
                config.o1,
                method=args.o1_method or config.o1.method,
                superpixel_count=args.o1_superpixel_count if args.o1_superpixel_count is not None else config.o1.superpixel_count,
                superpixel_cluster_count=(
                    args.o1_superpixel_cluster_count
                    if args.o1_superpixel_cluster_count is not None
                    else config.o1.superpixel_cluster_count
                ),
                device=args.o1_device or config.o1.device,
                output_tag=args.o1_output_tag if args.o1_output_tag is not None else config.o1.output_tag,
            ),
        )

    if args.objective == "o4" and (
        args.o4_execution_mode is not None
        or args.o4_dinov2_model is not None
        or args.o4_dinov2_checkpoint is not None
        or args.o4_dinov2_repo is not None
        or args.o4_regression_mode is not None
    ):
        dinov2_model_name = args.o4_dinov2_model or config.o4.dinov2_model_name
        dinov2_checkpoint_path = (
            args.o4_dinov2_checkpoint
            if args.o4_dinov2_checkpoint is not None
            else (
                resolve_dinov2_checkpoint_path(dinov2_model_name, None)
                if args.o4_dinov2_model is not None
                else config.o4.dinov2_checkpoint_path
            )
        )
        config = replace(
            config,
            o4=replace(
                config.o4,
                execution_mode=args.o4_execution_mode or config.o4.execution_mode,
                dinov2_model_name=dinov2_model_name,
                dinov2_repo_path=args.o4_dinov2_repo if args.o4_dinov2_repo is not None else config.o4.dinov2_repo_path,
                dinov2_checkpoint_path=dinov2_checkpoint_path,
                disparity_regression=args.o4_regression_mode or config.o4.disparity_regression,
            ),
        )

    if args.validate_results:
        if args.objective == "o2":
            from . import o2

            return o2.validate_results(
                config.o2.keypoints_dir,
                config.o2.matches_dir,
                config.o2.metrics_file,
                args.scene_name,
            )
        if args.objective == "o3":
            from . import o3

            return o3.validate_results(
                config.o3.disparity_dir,
                config.o3.analysis_dir,
                config.o3.metrics_file,
                args.scene_name,
            )
        if args.objective == "o4":
            from . import o4

            return o4.validate_results(
                config.o4.disparity_dir,
                config.o4.analysis_dir,
                config.o4.metrics_file,
                args.scene_name,
            )
        from . import o1

        return o1.validate_results(config.o1.synthetic_dir, args.scene_name)

    if args.objective == "o2":
        from . import o2

        return o2.run(
            config.repo_root,
            config.middlebury_root,
            config.o2,
            args.max_scenes,
            args.dry_run,
            args.scene_name,
        )

    if args.objective == "o3":
        from . import o3

        return o3.run(
            config.repo_root,
            config.middlebury_root,
            config.o3,
            args.max_scenes,
            args.dry_run,
            args.scene_name,
        )

    if args.objective == "o4":
        from . import o4

        return o4.run(
            config.repo_root,
            config.middlebury_root,
            config.o4,
            args.max_scenes,
            args.dry_run,
            args.scene_name,
        )

    from . import o1

    return o1.run(config.o1, args.max_scenes, args.dry_run, args.scene_name)


if __name__ == "__main__":
    raise SystemExit(main())
