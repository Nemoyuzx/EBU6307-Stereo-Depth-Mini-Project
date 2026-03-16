from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from .common import discover_scenes, filter_scene_dirs, load_bgr, load_gray, write_scene_text
from .config import O2Config


def read_metrics(metrics_file: Path) -> list[dict[str, str]]:
    if not metrics_file.exists():
        return []

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "scene": row.get("scene", ""),
                "left_keypoints": row.get("left_keypoints", ""),
                "right_keypoints": row.get("right_keypoints", ""),
                "raw_knn_matches": row.get("raw_knn_matches", ""),
                "ratio_test_matches": row.get("ratio_test_matches", ""),
                "repeatability_proxy": row.get("repeatability_proxy", ""),
            }
            for row in reader
            if row.get("scene")
        ]


def write_metrics(metrics_file: Path, rows: list[dict[str, str | float | int]]) -> None:
    existing_rows = read_metrics(metrics_file)
    rows_by_scene = {str(row["scene"]): row for row in rows}
    merged_rows: list[dict[str, str | float | int]] = []

    for row in existing_rows:
        scene_name = row["scene"]
        replacement = rows_by_scene.pop(scene_name, None)
        merged_rows.append(replacement if replacement is not None else row)

    merged_rows.extend(rows_by_scene.values())

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "left_keypoints",
                "right_keypoints",
                "raw_knn_matches",
                "ratio_test_matches",
                "repeatability_proxy",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)


def create_sift_detector(max_features: int, contrast_threshold: float) -> Any:
    import cv2

    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is unavailable in this environment. Install an OpenCV build with SIFT support.")
    return cv2.SIFT_create(nfeatures=max_features, contrastThreshold=contrast_threshold)


def draw_keypoints(image_bgr: Any, keypoints: Any) -> Any:
    import cv2

    return cv2.drawKeypoints(image_bgr, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def _ratio_filtered_matches(knn_matches: Any, ratio_test: float) -> list[Any]:
    filtered: list[Any] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_test * second.distance:
            filtered.append(first)
    return filtered


def _mutual_ratio_matches(left_descriptors: Any, right_descriptors: Any, ratio_test: float) -> tuple[int, list[Any], list[Any]]:
    import cv2

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward_knn = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    backward_knn = matcher.knnMatch(right_descriptors, left_descriptors, k=2)
    forward_ratio = _ratio_filtered_matches(forward_knn, ratio_test)
    backward_ratio = _ratio_filtered_matches(backward_knn, ratio_test)
    backward_pairs = {(match.trainIdx, match.queryIdx) for match in backward_ratio}
    mutual = [match for match in forward_ratio if (match.queryIdx, match.trainIdx) in backward_pairs]
    mutual.sort(key=lambda match: match.distance)
    return len(forward_knn), forward_ratio, mutual


def _geometric_inlier_matches(
    left_keypoints: Any,
    right_keypoints: Any,
    matches: list[Any],
) -> tuple[list[Any], int]:
    import cv2
    import numpy as np

    if len(matches) < 8:
        return matches, 0

    left_points = np.float32([left_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    right_points = np.float32([right_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    fundamental, inlier_mask = cv2.findFundamentalMat(
        left_points,
        right_points,
        cv2.FM_RANSAC,
        2.5,
        0.99,
    )
    if fundamental is None or inlier_mask is None:
        return matches, 0

    mask = np.asarray(inlier_mask, dtype=np.uint8).reshape(-1) > 0
    inliers = [match for match, keep in zip(matches, mask) if keep]
    if len(inliers) < 8:
        return matches, 0
    return inliers, int(mask.sum())


def _select_geometry_aware_matches(
    left_keypoints: Any,
    right_keypoints: Any,
    ratio_filtered: list[Any],
    mutual_matches: list[Any],
) -> tuple[list[Any], int]:
    candidate_matches = ratio_filtered
    mutual_retention = (len(mutual_matches) / len(ratio_filtered)) if ratio_filtered else 0.0
    if len(mutual_matches) >= 24 and mutual_retention >= 0.75:
        candidate_matches = mutual_matches

    if len(candidate_matches) < 24:
        return candidate_matches, 0

    geometric_matches, ransac_inliers = _geometric_inlier_matches(left_keypoints, right_keypoints, candidate_matches)
    if len(geometric_matches) >= max(24, int(len(candidate_matches) * 0.5)):
        return geometric_matches, ransac_inliers
    return candidate_matches, ransac_inliers


def validate_results(keypoints_dir: Path, matches_dir: Path, metrics_file: Path, scene_name: str | None = None) -> int:
    print(f"Validating O2 keypoint directory: {keypoints_dir}")
    print(f"Validating O2 match directory: {matches_dir}")
    print(f"Validating O2 metrics file: {metrics_file}")

    if not metrics_file.exists():
        print(f"O2 metrics file not found: {metrics_file}", file=sys.stderr)
        return 1

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("scene")]

    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
        rows = [row for row in rows if row.get("scene") == scene_name]

    if not rows:
        print("No O2 metric rows found for validation.", file=sys.stderr)
        return 1

    missing_any = False
    for row in rows:
        current_scene = row["scene"]
        keypoint_scene_dir = keypoints_dir / current_scene
        match_scene_dir = matches_dir / current_scene
        required_paths = (
            keypoint_scene_dir / "im0_keypoints.png",
            keypoint_scene_dir / "im1_keypoints.png",
            match_scene_dir / "sift_matches.png",
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            missing_any = True
            print(f"[MISSING] {current_scene}: " + ", ".join(missing))
        else:
            print(f"[OK] {current_scene}: required O2 files present")

    if missing_any:
        print("Validation status: issues found")
        return 1

    print("Validation status: all checked O2 scene folders contain the expected files")
    return 0


def run(
    repo_root: Path,
    middlebury_root: Path,
    config: O2Config,
    max_scenes: int | None,
    dry_run: bool,
    scene_name: str | None,
) -> int:
    discovered_scenes = discover_scenes(middlebury_root)
    discovered_count = len(discovered_scenes)
    scenes = filter_scene_dirs(discovered_scenes, scene_name)

    if max_scenes is not None:
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    print(f"Repository root: {repo_root}")
    print(f"Middlebury root: {middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {discovered_count}")
    print(f"O2 keypoint output dir: {config.keypoints_dir}")
    print(f"O2 match output dir: {config.matches_dir}")
    print(f"O2 metrics file: {config.metrics_file}")
    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
    print("Scenes to process: " + ", ".join(scene.name for scene in scenes) if scenes else "Scenes to process: none")

    if not middlebury_root.exists():
        if dry_run or max_scenes == 0:
            print("Middlebury root does not exist yet. Discovery-only mode completed without processing.")
            return 0
        print(
            f"Middlebury root not found: {middlebury_root}\n"
            "Place dataset scenes there, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    if discovered_count == 0:
        if dry_run or max_scenes == 0:
            print("No valid scenes found. Discovery-only mode completed without processing.")
            return 0
        print(
            f"No scene directories containing im0.png and im1.png were found under {middlebury_root}.\n"
            "Add Middlebury scenes, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    if scene_name is not None and not scenes:
        print(f"No discovered scenes matched --scene-name {scene_name!r} under {middlebury_root}.", file=sys.stderr)
        return 1

    if dry_run or max_scenes == 0:
        print("Dry run requested; no outputs were written.")
        return 0

    import cv2

    detector = create_sift_detector(config.max_features, config.contrast_threshold)
    config.keypoints_dir.mkdir(parents=True, exist_ok=True)
    config.matches_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, str | int | float]] = []
    for scene_dir in scenes:
        left_gray = load_gray(scene_dir / "im0.png")
        right_gray = load_gray(scene_dir / "im1.png")
        left_bgr = load_bgr(scene_dir / "im0.png")
        right_bgr = load_bgr(scene_dir / "im1.png")

        left_keypoints, left_descriptors = detector.detectAndCompute(left_gray, None)
        right_keypoints, right_descriptors = detector.detectAndCompute(right_gray, None)

        raw_matches = 0
        ratio_matches = 0
        mutual_matches = []
        good_matches = []
        ransac_inliers = 0
        effective_ratio_test = float(config.ratio_test)
        if left_descriptors is not None and right_descriptors is not None and len(left_descriptors) > 0 and len(right_descriptors) > 0:
            raw_matches, ratio_filtered, mutual_matches = _mutual_ratio_matches(
                left_descriptors,
                right_descriptors,
                effective_ratio_test,
            )
            ratio_matches = len(ratio_filtered)
            good_matches, ransac_inliers = _select_geometry_aware_matches(
                left_keypoints,
                right_keypoints,
                ratio_filtered,
                mutual_matches,
            )

        left_count = len(left_keypoints)
        right_count = len(right_keypoints)
        repeatability = float(len(good_matches) / min(left_count, right_count)) if min(left_count, right_count) > 0 else 0.0

        keypoint_scene_dir = config.keypoints_dir / scene_dir.name
        match_scene_dir = config.matches_dir / scene_dir.name
        keypoint_scene_dir.mkdir(parents=True, exist_ok=True)
        match_scene_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(keypoint_scene_dir / "im0_keypoints.png"), draw_keypoints(left_bgr, left_keypoints))
        cv2.imwrite(str(keypoint_scene_dir / "im1_keypoints.png"), draw_keypoints(right_bgr, right_keypoints))
        write_scene_text(
            keypoint_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                "generator: O2 SIFT baseline with relaxed ratio, conditional mutual matching, and F-matrix RANSAC",
                f"im0_keypoints: {left_count}",
                f"im1_keypoints: {right_count}",
                f"max_features: {config.max_features}",
                f"contrast_threshold: {config.contrast_threshold}",
            ],
        )

        draw_count = min(len(good_matches), config.max_draw_matches)
        match_image = cv2.drawMatches(
            left_bgr,
            left_keypoints,
            right_bgr,
            right_keypoints,
            good_matches[:draw_count],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(match_scene_dir / "sift_matches.png"), match_image)
        write_scene_text(
            match_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                "generator: O2 SIFT baseline with relaxed ratio, conditional mutual matching, and F-matrix RANSAC",
                f"raw_knn_matches: {raw_matches}",
                f"ratio_filtered_matches: {ratio_matches}",
                f"mutual_matches: {len(mutual_matches)}",
                f"fundamental_inliers: {ransac_inliers if ransac_inliers > 0 else len(good_matches)}",
                f"ratio_test_matches: {len(good_matches)}",
                f"ratio_test: {effective_ratio_test}",
                f"drawn_matches: {draw_count}",
                "repeatability_proxy: ratio_test_matches / min(left_keypoints, right_keypoints)",
            ],
        )

        metric_rows.append(
            {
                "scene": scene_dir.name,
                "left_keypoints": left_count,
                "right_keypoints": right_count,
                "raw_knn_matches": raw_matches,
                "ratio_test_matches": len(good_matches),
                "repeatability_proxy": f"{repeatability:.6f}",
            }
        )
        print(
            f"Wrote O2 scene: {scene_dir.name} "
            f"(left={left_count}, right={right_count}, ratio={ratio_matches}, mutual={len(mutual_matches)}, "
            f"good_matches={len(good_matches)}, repeatability_proxy={repeatability:.6f})"
        )

    write_metrics(config.metrics_file, metric_rows)
    print(f"Wrote O2 metrics summary: {config.metrics_file}")
    return 0
