from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .common import discover_scenes, filter_scene_dirs, load_bgr, load_gray, write_scene_text
from .config import O2Config

METRIC_FIELDNAMES = [
    "scene",
    "transform_family",
    "random_seed",
    "left_keypoints",
    "right_keypoints",
    "raw_knn_matches",
    "ratio_test_matches",
    "repeatable_matches",
    "repeatability",
    "repeatability_proxy",
    "homography_threshold_px",
    "transform_params_json",
]

DISTINCT_TRANSFORM_FAMILIES = (
    "rotation",
    "affine",
    "intensity",
)


def read_metrics(metrics_file: Path) -> list[dict[str, str]]:
    if not metrics_file.exists():
        return []

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            if not row.get("scene"):
                continue
            rows.append({key: row.get(key, "") for key in METRIC_FIELDNAMES})
        return rows


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
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
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


def _geometric_inlier_matches(left_keypoints: Any, right_keypoints: Any, matches: list[Any]) -> tuple[list[Any], int]:
    import cv2
    import numpy as np

    if len(matches) < 8:
        return matches, 0

    left_points = np.float32([left_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    right_points = np.float32([right_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    fundamental, inlier_mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC, 2.5, 0.99)
    if fundamental is None or inlier_mask is None:
        return matches, 0

    mask = np.asarray(inlier_mask, dtype=np.uint8).reshape(-1) > 0
    inliers = [match for match, keep in zip(matches, mask) if keep]
    if len(inliers) < 8:
        return matches, 0
    return inliers, int(mask.sum())


def _select_geometry_aware_matches(left_keypoints: Any, right_keypoints: Any, ratio_filtered: list[Any], mutual_matches: list[Any]) -> tuple[list[Any], int]:
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


def _project_point(homography: Any, x: float, y: float) -> tuple[float, float] | None:
    import numpy as np

    point = np.asarray([x, y, 1.0], dtype=np.float64)
    projected = homography @ point
    if abs(float(projected[2])) < 1e-8:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _rng_for_scene(scene_name: str) -> tuple[int, Any]:
    import numpy as np

    seed = int(hashlib.sha256(scene_name.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    return seed, np.random.default_rng(seed)


def _transform_family(scene_name: str, scene_index: int) -> str:
    _, rng = _rng_for_scene(f"family::{scene_index}::{scene_name}")
    return str(rng.choice(DISTINCT_TRANSFORM_FAMILIES))


def _rotation_matrix(width: int, height: int, angle_deg: float, scale: float) -> Any:
    import cv2
    import numpy as np

    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle_deg, scale)
    homography = np.eye(3, dtype=np.float32)
    homography[:2, :] = affine
    return homography


def _affine_homography(width: int, height: int, rng: Any) -> tuple[Any, dict[str, float]]:
    import numpy as np

    angle_deg = float(rng.uniform(-16.0, 16.0))
    scale = float(rng.uniform(0.88, 1.12))
    shear_x = float(rng.uniform(-0.08, 0.08))
    shear_y = float(rng.uniform(-0.05, 0.05))
    tx = float(rng.uniform(-0.06, 0.06) * width)
    ty = float(rng.uniform(-0.06, 0.06) * height)

    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scaling = np.array(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    shear = np.array(
        [
            [1.0, shear_x, 0.0],
            [shear_y, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    center_to_origin = np.array(
        [
            [1.0, 0.0, -width / 2.0],
            [0.0, 1.0, -height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    origin_to_center = np.array(
        [
            [1.0, 0.0, width / 2.0 + tx],
            [0.0, 1.0, height / 2.0 + ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    homography = origin_to_center @ shear @ scaling @ rotation @ center_to_origin
    return homography, {
        "angle_deg": angle_deg,
        "scale": scale,
        "shear_x": shear_x,
        "shear_y": shear_y,
        "tx_px": tx,
        "ty_px": ty,
    }


def _apply_intensity_variation(image_bgr: Any, rng: Any) -> tuple[Any, dict[str, float]]:
    import cv2
    import numpy as np

    alpha = float(rng.uniform(0.7, 1.35))
    beta = float(rng.uniform(-32.0, 32.0))
    gamma = float(rng.uniform(0.75, 1.35))
    noise_sigma = float(rng.uniform(0.0, 8.0))
    blur_kernel = int(rng.choice([0, 3, 5]))

    working = image_bgr.astype(np.float32) / 255.0
    working = np.clip(working * alpha + beta / 255.0, 0.0, 1.0)
    working = np.power(working, gamma)
    if noise_sigma > 0.0:
        working += rng.normal(0.0, noise_sigma / 255.0, size=working.shape)
    working = np.clip(working, 0.0, 1.0)
    transformed = np.rint(working * 255.0).astype(np.uint8)
    if blur_kernel > 0:
        transformed = cv2.GaussianBlur(transformed, (blur_kernel, blur_kernel), 0)
    return transformed, {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "noise_sigma": noise_sigma,
        "blur_kernel": blur_kernel,
    }


def generate_transformed_view(image_bgr: Any, scene_name: str, scene_index: int) -> tuple[Any, Any, str, int, dict[str, float | int | str]]:
    import cv2
    import numpy as np

    height, width = image_bgr.shape[:2]
    seed, rng = _rng_for_scene(f"transform::{scene_index}::{scene_name}")
    family = _transform_family(scene_name, scene_index)

    if family == "rotation":
        angle_deg = float(rng.uniform(-28.0, 28.0))
        scale = float(rng.uniform(0.82, 1.18))
        tx = float(rng.uniform(-0.03, 0.03) * width)
        ty = float(rng.uniform(-0.03, 0.03) * height)
        homography = _rotation_matrix(width, height, angle_deg, scale)
        homography[0, 2] += tx
        homography[1, 2] += ty
        transformed = cv2.warpPerspective(
            image_bgr,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        params = {
            "family": family,
            "angle_deg": angle_deg,
            "scale": scale,
            "tx_px": tx,
            "ty_px": ty,
        }
        return transformed, homography, family, seed, params

    if family == "affine":
        homography, params = _affine_homography(width, height, rng)
        transformed = cv2.warpPerspective(
            image_bgr,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        params["family"] = family
        return transformed, homography, family, seed, params

    transformed, params = _apply_intensity_variation(image_bgr, rng)
    homography = np.eye(3, dtype=np.float32)
    params["family"] = family
    return transformed, homography, family, seed, params


def _repeatable_matches(
    original_keypoints: Any,
    transformed_keypoints: Any,
    ratio_matches: list[Any],
    homography: Any,
    threshold_px: float,
) -> list[Any]:
    repeatable: list[Any] = []
    for match in ratio_matches:
        source = original_keypoints[match.queryIdx].pt
        target = transformed_keypoints[match.trainIdx].pt
        projected = _project_point(homography, float(source[0]), float(source[1]))
        if projected is None:
            continue
        dx = projected[0] - float(target[0])
        dy = projected[1] - float(target[1])
        if (dx * dx + dy * dy) ** 0.5 <= threshold_px:
            repeatable.append(match)
    repeatable.sort(key=lambda item: item.distance)
    return repeatable


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
    threshold_px = 4.0

    for scene_index, scene_dir in enumerate(scenes):
        original_bgr = load_bgr(scene_dir / "im0.png")
        original_gray = load_gray(scene_dir / "im0.png")
        transformed_bgr, homography, transform_family, random_seed, transform_params = generate_transformed_view(
            original_bgr,
            scene_dir.name,
            scene_index,
        )
        transformed_gray = cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2GRAY)

        original_keypoints, original_descriptors = detector.detectAndCompute(original_gray, None)
        transformed_keypoints, transformed_descriptors = detector.detectAndCompute(transformed_gray, None)

        raw_matches = 0
        ratio_matches = []
        repeatable_matches = []
        if (
            original_descriptors is not None
            and transformed_descriptors is not None
            and len(original_descriptors) > 0
            and len(transformed_descriptors) > 0
        ):
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            knn_matches = matcher.knnMatch(original_descriptors, transformed_descriptors, k=2)
            raw_matches = len(knn_matches)
            ratio_matches = _ratio_filtered_matches(knn_matches, float(config.ratio_test))
            repeatable_matches = _repeatable_matches(
                original_keypoints,
                transformed_keypoints,
                ratio_matches,
                homography,
                threshold_px,
            )

        original_count = len(original_keypoints)
        transformed_count = len(transformed_keypoints)
        repeatability = (
            float(len(repeatable_matches) / min(original_count, transformed_count))
            if min(original_count, transformed_count) > 0
            else 0.0
        )

        keypoint_scene_dir = config.keypoints_dir / scene_dir.name
        match_scene_dir = config.matches_dir / scene_dir.name
        keypoint_scene_dir.mkdir(parents=True, exist_ok=True)
        match_scene_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(keypoint_scene_dir / "im0_keypoints.png"), draw_keypoints(original_bgr, original_keypoints))
        cv2.imwrite(str(keypoint_scene_dir / "im1_keypoints.png"), draw_keypoints(transformed_bgr, transformed_keypoints))
        write_scene_text(
            keypoint_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                "generator: O2 SIFT repeatability baseline on original image plus random transformation",
                "evaluation_definition: detect SIFT on original image before transformation, detect again after a randomly generated transformation, then keep only descriptor matches consistent with the known random transform",
                f"transform_family: {transform_family}",
                f"random_seed: {random_seed}",
                f"transform_params_json: {json.dumps(transform_params, sort_keys=True)}",
                f"im0_keypoints: {original_count}",
                f"im1_keypoints: {transformed_count}",
                f"max_features: {config.max_features}",
                f"contrast_threshold: {config.contrast_threshold}",
            ],
        )

        draw_count = min(len(repeatable_matches), config.max_draw_matches)
        match_image = cv2.drawMatches(
            original_bgr,
            original_keypoints,
            transformed_bgr,
            transformed_keypoints,
            repeatable_matches[:draw_count],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(match_scene_dir / "sift_matches.png"), match_image)
        write_scene_text(
            match_scene_dir / "README.txt",
            [
                f"scene: {scene_dir.name}",
                "generator: O2 SIFT repeatability baseline on original image plus random transformation",
                f"transform_family: {transform_family}",
                f"random_seed: {random_seed}",
                f"transform_params_json: {json.dumps(transform_params, sort_keys=True)}",
                f"raw_knn_matches: {raw_matches}",
                f"ratio_filtered_matches: {len(ratio_matches)}",
                f"repeatable_matches: {len(repeatable_matches)}",
                f"homography_threshold_px: {threshold_px}",
                f"ratio_test: {config.ratio_test}",
                f"drawn_matches: {draw_count}",
                "repeatability: repeatable_matches / min(original_keypoints, transformed_keypoints)",
            ],
        )

        metric_rows.append(
            {
                "scene": scene_dir.name,
                "transform_family": transform_family,
                "random_seed": random_seed,
                "left_keypoints": original_count,
                "right_keypoints": transformed_count,
                "raw_knn_matches": raw_matches,
                "ratio_test_matches": len(ratio_matches),
                "repeatable_matches": len(repeatable_matches),
                "repeatability": f"{repeatability:.6f}",
                "repeatability_proxy": f"{repeatability:.6f}",
                "homography_threshold_px": f"{threshold_px:.2f}",
                "transform_params_json": json.dumps(transform_params, sort_keys=True),
            }
        )
        print(
            f"Wrote O2 scene: {scene_dir.name} "
            f"(family={transform_family}, seed={random_seed}, original={original_count}, transformed={transformed_count}, "
            f"ratio={len(ratio_matches)}, repeatable={len(repeatable_matches)}, repeatability={repeatability:.6f})"
        )

    write_metrics(config.metrics_file, metric_rows)
    print(f"Wrote O2 metrics summary: {config.metrics_file}")
    return 0
