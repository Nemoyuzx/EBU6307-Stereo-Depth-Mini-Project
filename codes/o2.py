from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from entry_utils import run_objective_entry
from common import discover_scenes, filter_scene_dirs, load_bgr, load_gray, write_scene_text
from config import O2Config

# 这一组列名既决定 CSV 的写出顺序，也决定旧指标文件在回读时按什么字段补齐。
# 这里故意把“场景身份”“随机变换信息”“匹配统计”“最终 repeatability 指标”放在一起，
# 这样后面无论是人工看 CSV，还是脚本按列解析，都比较稳定。
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

MetricValue = str | float | int
MetricRow = dict[str, MetricValue]
TransformParams = dict[str, MetricValue]


def read_metrics(metrics_file: Path) -> list[MetricRow]:
    """读取 O2 历史指标。"""
    # 第一次运行时指标文件还不存在，这里直接返回空列表，让上层按“无历史结果”处理即可。
    if not metrics_file.exists():
        return []

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[MetricRow] = []
        for row in reader:
            # scene 是后面做按场景合并的主键；没有 scene 的行基本可以视为脏数据。
            if not row.get("scene"):
                continue
            # 只保留当前 O2 关心的字段，避免旧文件里混入无关列时影响后续写回格式。
            rows.append({key: row.get(key, "") for key in METRIC_FIELDNAMES})
        return rows


def write_metrics(metrics_file: Path, rows: list[MetricRow]) -> None:
    """按场景合并并回写 O2 指标。"""
    # O2 是按场景重跑的：同一个 scene 再跑一次时，希望新结果覆盖旧结果，
    # 没有重跑的 scene 则保持原样，所以这里不是简单追加，而是做一次按 scene 的 merge。
    existing_rows = read_metrics(metrics_file)
    rows_by_scene = {str(row["scene"]): row for row in rows}
    merged_rows: list[MetricRow] = []

    for row in existing_rows:
        scene_name = str(row["scene"])
        replacement = rows_by_scene.pop(scene_name, None)
        merged_rows.append(replacement if replacement is not None else row)

    # 走到这里，rows_by_scene 里剩下的是“旧文件里没有、这次新产生”的场景。
    merged_rows.extend(rows_by_scene.values())

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged_rows)


def create_sift_detector(max_features: int, contrast_threshold: float) -> Any:
    """创建 SIFT 检测器，并在环境不支持 SIFT 时给出明确报错。"""

    sift_create = cast(Any, getattr(cv2, "SIFT_create", None))
    if sift_create is None:
        raise RuntimeError("OpenCV SIFT is unavailable in this environment. Install an OpenCV build with SIFT support.")
    # contrastThreshold 对应 SIFT 第 2 步里的低对比度点剔除强度；
    # nfeatures 则限制最终保留下来的关键点数量，便于控制工程侧开销。
    return sift_create(nfeatures=max_features, contrastThreshold=contrast_threshold)


def draw_keypoints(image_bgr: Any, keypoints: Any) -> Any:
    """把关键点以可视化形式画到图像上，便于结果检查。"""

    draw_keypoints_fn = cast(Any, cv2.drawKeypoints)
    # 这里画出的圆圈大小和朝向，来自 SIFT 第 2、3 步之后得到的关键点位置、尺度和主方向。
    return draw_keypoints_fn(image_bgr, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def _ratio_filtered_matches(knn_matches: Any, ratio_test: float) -> list[Any]:
    """对 KNN 匹配应用 Lowe ratio test。"""
    filtered: list[Any] = []
    for pair in knn_matches:
        # 正常情况下 knnMatch(k=2) 会返回两个候选；极少数边界情况不足两个时，
        # 没法做 ratio test，直接跳过最稳妥。
        if len(pair) < 2:
            continue
        first, second = pair
        # 最近邻必须显著优于次近邻，才认为这个描述符匹配“足够明确”。
        if first.distance < ratio_test * second.distance:
            filtered.append(first)
    return filtered


def _mutual_ratio_matches(left_descriptors: Any, right_descriptors: Any, ratio_test: float) -> tuple[int, list[Any], list[Any]]:
    """先做双向 KNN+ratio test，再取互为匹配的描述子对，减少偶然误匹配。"""

    # SIFT 描述符默认用 L2 距离；这里显式指定，避免读代码时还要回忆 OpenCV 默认行为。
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward_knn = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    backward_knn = matcher.knnMatch(right_descriptors, left_descriptors, k=2)
    forward_ratio = _ratio_filtered_matches(forward_knn, ratio_test)
    backward_ratio = _ratio_filtered_matches(backward_knn, ratio_test)
    # 反向保留下来的匹配先编码成索引对，后面判断“是否互为最近邻”会更直接。
    backward_pairs = {(match.trainIdx, match.queryIdx) for match in backward_ratio}
    mutual = [match for match in forward_ratio if (match.queryIdx, match.trainIdx) in backward_pairs]
    # 为了让可视化和调试输出更稳定，这里按距离从小到大排序。
    mutual.sort(key=lambda match: match.distance)
    return len(forward_knn), forward_ratio, mutual


def _geometric_inlier_matches(left_keypoints: Any, right_keypoints: Any, matches: list[Any]) -> tuple[list[Any], int]:
    """使用基础矩阵 RANSAC 剔除几何上明显不合理的匹配。"""

    # 基础矩阵最少需要 8 对点；样本太少时强行做 RANSAC 只会让结果更不稳定。
    if len(matches) < 8:
        return matches, 0

    # OpenCV 这里要求输入形状是 N x 1 x 2，所以要先把 KeyPoint.pt 摘出来再 reshape。
    left_points = np.asarray([left_keypoints[match.queryIdx].pt for match in matches], dtype=np.float32).reshape(-1, 1, 2)
    right_points = np.asarray([right_keypoints[match.trainIdx].pt for match in matches], dtype=np.float32).reshape(-1, 1, 2)
    fundamental, inlier_mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC, 2.5, 0.99)
    # 估计失败时，宁可保留原匹配，也不要伪造一个“清洗后”的结果。
    if fundamental is None or inlier_mask is None:
        return matches, 0

    mask = np.asarray(inlier_mask, dtype=np.uint8).reshape(-1) > 0
    inliers = [match for match, keep in zip(matches, mask) if keep]
    # 即便 RANSAC 返回了 mask，如果最终内点少于 8 个，也说明几何约束不够可信。
    if len(inliers) < 8:
        return matches, 0
    return inliers, int(mask.sum())


def _select_geometry_aware_matches(left_keypoints: Any, right_keypoints: Any, ratio_filtered: list[Any], mutual_matches: list[Any]) -> tuple[list[Any], int]:
    """结合 mutual matching 和 RANSAC 结果选择更稳健的匹配集合。"""
    # 默认从单向 ratio test 结果开始，因为这是最宽松、召回率最高的候选集。
    candidate_matches = ratio_filtered
    mutual_retention = (len(mutual_matches) / len(ratio_filtered)) if ratio_filtered else 0.0
    # 当 mutual matching 保留了足够多的点，而且保留比例也够高时，
    # 说明双向一致性没有把样本削得太狠，这时优先用 mutual 结果更稳。
    if len(mutual_matches) >= 24 and mutual_retention >= 0.75:
        candidate_matches = mutual_matches

    # 候选集太小时不做 RANSAC，因为几何模型会被少量偶然点严重影响。
    if len(candidate_matches) < 24:
        return candidate_matches, 0

    geometric_matches, ransac_inliers = _geometric_inlier_matches(left_keypoints, right_keypoints, candidate_matches)
    # 只有当 RANSAC 之后还剩下“数量够多、比例也不离谱”的内点时，才相信几何筛选真的起了正作用。
    if len(geometric_matches) >= max(24, int(len(candidate_matches) * 0.5)):
        return geometric_matches, ransac_inliers
    return candidate_matches, ransac_inliers


def _project_point(homography: Any, x: float, y: float) -> tuple[float, float] | None:
    """把图像点通过单应矩阵投影到变换后坐标系；若齐次坐标退化则返回 None。"""

    # 这里显式构造成齐次坐标，便于直接和 3x3 homography 做矩阵乘法。
    point = np.asarray([x, y, 1.0], dtype=np.float64)
    projected = homography @ point
    # 分母接近 0 说明投影退化了；继续相除只会放大数值误差。
    if abs(float(projected[2])) < 1e-8:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _rng_for_scene(scene_name: str) -> tuple[int, Any]:
    """为每个场景稳定地产生随机种子，保证多次运行结果可复现。"""

    # 不直接用 Python 内建 hash，是因为它默认带随机扰动；这里改用 sha256 保证跨进程稳定。
    seed = int(hashlib.sha256(scene_name.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    return seed, np.random.default_rng(seed)


def _transform_family(scene_name: str, scene_index: int) -> str:
    """稳定地为场景挑选一种变换族。"""
    # family 和 transform 本身分开播种，是为了避免“改了某一类变换的参数采样逻辑后，
    # 连场景落到哪一类 family 都跟着漂移”。
    _, rng = _rng_for_scene(f"family::{scene_index}::{scene_name}")
    return str(rng.choice(DISTINCT_TRANSFORM_FAMILIES))


def _rotation_matrix(width: int, height: int, angle_deg: float, scale: float) -> Any:
    """构造绕图像中心旋转/缩放的 3x3 单应矩阵。"""

    # OpenCV 返回的是 2x3 仿射矩阵，这里手动扩成 3x3，后面做点投影时会统一很多。
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle_deg, scale)
    homography = np.eye(3, dtype=np.float32)
    homography[:2, :] = affine
    return homography


def _affine_homography(width: int, height: int, rng: Any) -> tuple[Any, dict[str, float]]:
    """随机生成仿射风格的单应矩阵及其参数说明。"""

    # 这里把角度、尺度、剪切、平移拆开采样，目的是让 README / metrics 里能把变换说清楚，
    # 后面如果某一类变化导致 repeatability 明显下降，也容易定位是哪个因素更敏感。
    angle_deg = float(rng.uniform(-16.0, 16.0))
    scale = float(rng.uniform(0.88, 1.12))
    shear_x = float(rng.uniform(-0.08, 0.08))
    shear_y = float(rng.uniform(-0.05, 0.05))
    tx = float(rng.uniform(-0.06, 0.06) * width)
    ty = float(rng.uniform(-0.06, 0.06) * height)

    angle = np.deg2rad(angle_deg)
    # 下面这几块矩阵是按“先把中心挪到原点，再做 rotation/scale/shear，
    # 最后挪回图像中心并叠加平移”的思路拼起来的。
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
    """只改变亮度/对比度/噪声/模糊，不改变几何结构，用来测试 SIFT 对光照变化的鲁棒性。"""

    # alpha/beta/gamma 分别覆盖线性对比度、整体亮度和非线性亮度响应；
    # 再叠一点噪声和模糊，能更接近真实采集条件里的曝光/成像扰动。
    alpha = float(rng.uniform(0.7, 1.35))
    beta = float(rng.uniform(-32.0, 32.0))
    gamma = float(rng.uniform(0.75, 1.35))
    noise_sigma = float(rng.uniform(0.0, 8.0))
    blur_kernel = int(rng.choice([0, 3, 5]))

    # 先转到 [0, 1] 浮点域处理，再回写成 uint8，这样组合多个强度操作时更可控。
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


def generate_transformed_view(image_bgr: Any, scene_name: str, scene_index: int) -> tuple[Any, Any, str, int, TransformParams]:
    """根据场景稳定随机地产生一种变换后的右图，并返回对应的真实几何/光照参数。"""

    height, width = image_bgr.shape[:2]
    seed, rng = _rng_for_scene(f"transform::{scene_index}::{scene_name}")
    family = _transform_family(scene_name, scene_index)

    if family == "rotation":
        # rotation 家族只做“绕中心旋转/缩放 + 小范围平移”，属于比较直观的相机姿态扰动。
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
        params: TransformParams = {
            "family": family,
            "angle_deg": angle_deg,
            "scale": scale,
            "tx_px": tx,
            "ty_px": ty,
        }
        return transformed, homography, family, seed, params

    if family == "affine":
        # affine 家族比单纯旋转更激进一些，额外引入 shear，能测试关键点在形变下的稳定性。
        homography, affine_params = _affine_homography(width, height, rng)
        transformed = cv2.warpPerspective(
            image_bgr,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        params: TransformParams = {
            "family": family,
            "angle_deg": affine_params["angle_deg"],
            "scale": affine_params["scale"],
            "shear_x": affine_params["shear_x"],
            "shear_y": affine_params["shear_y"],
            "tx_px": affine_params["tx_px"],
            "ty_px": affine_params["ty_px"],
        }
        return transformed, homography, family, seed, params

    # intensity 家族不改几何，所以真实 homography 就是单位阵；
    # 这样后面 repeatability 下降时，基本可以直接归因到描述符对光照扰动的抗性不够。
    transformed, intensity_params = _apply_intensity_variation(image_bgr, rng)
    homography = np.eye(3, dtype=np.float32)
    params: TransformParams = {
        "family": family,
        "alpha": intensity_params["alpha"],
        "beta": intensity_params["beta"],
        "gamma": intensity_params["gamma"],
        "noise_sigma": intensity_params["noise_sigma"],
        "blur_kernel": intensity_params["blur_kernel"],
    }
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
        # queryIdx 指向原图关键点，trainIdx 指向变换后图像里的候选对应点。
        source = original_keypoints[match.queryIdx].pt
        target = transformed_keypoints[match.trainIdx].pt
        projected = _project_point(homography, float(source[0]), float(source[1]))
        if projected is None:
            continue
        dx = projected[0] - float(target[0])
        dy = projected[1] - float(target[1])
        # 这里的判据很直接：描述符匹配到的 target 点，如果离真实投影位置足够近，
        # 就把它记成“可重复匹配”。
        if (dx * dx + dy * dy) ** 0.5 <= threshold_px:
            repeatable.append(match)
    # 同样按距离排序，保证图像可视化时优先画出更可靠的匹配。
    repeatable.sort(key=lambda item: item.distance)
    return repeatable


def validate_results(keypoints_dir: Path, matches_dir: Path, metrics_file: Path, scene_name: str | None = None) -> int:
    """验证 O2 输出文件是否存在。"""
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

    # 校验逻辑以 metrics.csv 为入口，因为 metrics 代表“哪些 scene 被认为已经成功跑过”。
    if not rows:
        print("No O2 metric rows found for validation.", file=sys.stderr)
        return 1

    missing_any = False
    for row in rows:
        current_scene = row["scene"]
        keypoint_scene_dir = keypoints_dir / current_scene
        match_scene_dir = matches_dir / current_scene
        # O2 最核心的产物就是两张关键点图和一张匹配图；README 丢了虽然不理想，
        # 但通常不影响后续查结果，所以这里先把“最低限度必须存在的文件”收紧到这三项。
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
    # “总共发现多少 scene”“这次实际会处理多少 scene”
    discovered_scenes = discover_scenes(middlebury_root)
    discovered_count = len(discovered_scenes)
    scenes = filter_scene_dirs(discovered_scenes, scene_name)

    if max_scenes is not None:
        # 负数没有实际意义，直接按参数错误返回，比悄悄当作 0 或 None 更容易排查。
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    # 关键路径和输出位置都打印 读哪里、写哪里、筛了哪些 scene
    print(f"Repository root: {repo_root}")
    print(f"Middlebury root: {middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {discovered_count}")
    print(f"O2 keypoint output dir: {config.keypoints_dir}")
    print(f"O2 match output dir: {config.matches_dir}")
    print(f"O2 metrics file: {config.metrics_file}")
    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
    print("Scenes to process: " + ", ".join(scene.name for scene in scenes) if scenes else "Scenes to process: none")

    #判定数据集指定目录是否存在
    if not middlebury_root.exists():
        # dry-run / discovery-only 模式下，跳过检查
        if dry_run or max_scenes == 0:
            print("Middlebury root does not exist yet. Discovery-only mode completed without processing.")
            return 0
        print(
            f"Middlebury root not found: {middlebury_root}\n"
            "Place dataset scenes there, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    # 目录存在但没有任何合法 scene 时
    if discovered_count == 0:
        # dry-run / discovery-only 模式下，跳过检查
        if dry_run or max_scenes == 0:
            print("No valid scenes found. Discovery-only mode completed without processing.")
            return 0
        print(
            f"No scene directories containing im0.png and im1.png were found under {middlebury_root}.\n"
            "Add Middlebury scenes, or rerun with --dry-run or --max-scenes 0.",
            file=sys.stderr,
        )
        return 1

    # 指定的 scene_name 不存在
    if scene_name is not None and not scenes:
        print(f"No discovered scenes matched --scene-name {scene_name!r} under {middlebury_root}.", file=sys.stderr)
        return 1

    # 到这里说明发现逻辑都正常；如果只是 dry-run，就不再创建目录也不做任何图像计算。
    if dry_run or max_scenes == 0:
        print("Dry run requested; no outputs were written.")
        return 0

    # detector 在整个 O2 过程中复用即可，没有必要为每个 scene 单独创建。
    detector = create_sift_detector(config.max_features, config.contrast_threshold)
    # 输出目录在主循环之前统一建好，避免后面每个 scene 都做一次重复判定。
    config.keypoints_dir.mkdir(parents=True, exist_ok=True)
    config.matches_dir.mkdir(parents=True, exist_ok=True)

    # metric_rows 先在内存里累计，最后统一写 CSV，避免处理中途留下半截文件。
    metric_rows: list[dict[str, str | int | float]] = []
    # 4 像素阈值不是 SIFT 算法本身的参数，而是这里定义“几何上算 repeatable”的工程判据。
    threshold_px = 4.0

    for scene_index, scene_dir in enumerate(scenes):
        # O2 的左图固定取原始 im0；右图不是直接用 Middlebury 的 im1，
        # 而是由左图生成一个带可控扰动的 transformed view，用来做 repeatability 评估。
        original_bgr = load_bgr(scene_dir / "im0.png")
        original_gray = load_gray(scene_dir / "im0.png")
        transformed_bgr, homography, transform_family, random_seed, transform_params = generate_transformed_view(
            original_bgr,
            scene_dir.name,
            scene_index,
        )
        # SIFT 检测和描述通常在灰度图上做，这里显式转灰度，避免 OpenCV 内部隐式转换带来歧义。
        transformed_gray = cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2GRAY)

        # 在整张 original_gray 上执行一次完整 SIFT；第二个参数传 None，表示不额外提供 mask，
        # 所以 OpenCV 会直接在整幅图像内完成检测和描述。
        # 返回的 original_keypoints 是“已经走完 SIFT 前 3 步”的结果：
        # 1) 先在 original_gray 的尺度空间 / DoG 金字塔里找极值候选点；
        # 2) 再结合 create_sift_detector(...) 里设定的 contrastThreshold 做低对比度剔除，
        #    同时过滤边缘响应并完成亚像素级精定位；
        # 3) 然后给每个保留下来的关键点分配主方向，因此 keypoint 的 pt / size / angle
        #    在这里都已经是可直接使用的稳定属性。
        # 返回的 original_descriptors 则对应 SIFT 第 4 步：围绕每个 original_keypoints 统计局部梯度，
        # 为每个关键点生成一个 128 维描述符，供后面的 BFMatcher 做匹配。
        original_keypoints, original_descriptors = detector.detectAndCompute(original_gray, None)
        # 对变换后的 transformed_gray 重复同一套 detect + compute 流程，得到另一组
        # transformed_keypoints / transformed_descriptors；下面的 repeatability 评估
        # 就是专门比较这两次 detectAndCompute(...) 的输出是否还能对应上。
        transformed_keypoints, transformed_descriptors = detector.detectAndCompute(transformed_gray, None)

        # 先把匹配统计初始化成“没有匹配”的状态，这样即便某个 scene 一个描述符都没提到，
        # 后面生成 README / CSV 时也还是能写出完整记录。
        raw_matches = 0
        ratio_matches = []
        repeatable_matches = []
        if (
            original_descriptors is not None
            and transformed_descriptors is not None
            and len(original_descriptors) > 0
            and len(transformed_descriptors) > 0
        ):
            # 这里开始不再属于 SIFT 本体，而是对第 4 步产出的描述符做项目级评估。
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            # 先对 128 维描述符做 KNN 最近邻匹配，建立候选对应关系。
            knn_matches = matcher.knnMatch(original_descriptors, transformed_descriptors, k=2)
            raw_matches = len(knn_matches)
            # Lowe ratio test 用来剔除“最近邻和次近邻太接近”的歧义匹配。
            ratio_matches = _ratio_filtered_matches(knn_matches, float(config.ratio_test))
            # 再结合已知随机变换的真实几何关系，筛出真正可重复的关键点对应。
            repeatable_matches = _repeatable_matches(
                original_keypoints,
                transformed_keypoints,
                ratio_matches,
                homography,
                threshold_px,
            )

        original_count = len(original_keypoints)
        transformed_count = len(transformed_keypoints)
        # repeatability 采用 min(left, right) 归一化，是为了避免某一边关键点特别多时
        # 把分母拉得过大；这里更关心“较小那一侧里有多少点还能稳定找回来”。
        repeatability = (
            float(len(repeatable_matches) / min(original_count, transformed_count))
            if min(original_count, transformed_count) > 0
            else 0.0
        )

        # O2 把关键点可视化和匹配可视化分到两个目录，是为了让“检测质量”和“匹配质量”
        # 可以分开检查，不至于所有产物都堆在一个场景文件夹里。
        keypoint_scene_dir = config.keypoints_dir / scene_dir.name
        match_scene_dir = config.matches_dir / scene_dir.name
        keypoint_scene_dir.mkdir(parents=True, exist_ok=True)
        match_scene_dir.mkdir(parents=True, exist_ok=True)

        # 这两张图主要用于人工检查：关键点是否覆盖了有纹理的区域、变换后是否仍有足够响应。
        cv2.imwrite(str(keypoint_scene_dir / "im0_keypoints.png"), draw_keypoints(original_bgr, original_keypoints))
        cv2.imwrite(str(keypoint_scene_dir / "im1_keypoints.png"), draw_keypoints(transformed_bgr, transformed_keypoints))
        # README 则把“这一 scene 是怎么变出来的、参数是多少、关键点数量是多少”落成文本，
        # 后面回看单个场景时不必再去翻 metrics.csv。
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

        # drawMatches 图太密会完全看不清，所以这里只画最可靠的前若干条 repeatable matches。
        draw_count = min(len(repeatable_matches), config.max_draw_matches)
        draw_matches_fn = cast(Any, cv2.drawMatches)
        match_image = draw_matches_fn(
            original_bgr,
            original_keypoints,
            transformed_bgr,
            transformed_keypoints,
            repeatable_matches[:draw_count],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(match_scene_dir / "sift_matches.png"), match_image)
        # 匹配 README 更偏向评估视角：这次总共匹配了多少、ratio 之后剩多少、几何上真正 repeatable 的有多少。
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

        # CSV 里的字段尽量保持原子化，方便后面直接 pandas 读进来做统计或画图。
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
        # 每个 scene 处理完就打印一行摘要，跑长任务时不用等到最后也能知道有没有明显异常。
        print(
            f"Wrote O2 scene: {scene_dir.name} "
            f"(family={transform_family}, seed={random_seed}, original={original_count}, transformed={transformed_count}, "
            f"ratio={len(ratio_matches)}, repeatable={len(repeatable_matches)}, repeatability={repeatability:.6f})"
        )

    # 整批 scene 全部处理完后，再统一写 metrics summary，保证文件内容总是对应一轮完整运行。
    write_metrics(config.metrics_file, metric_rows)
    print(f"Wrote O2 metrics summary: {config.metrics_file}")
    return 0


if __name__ == '__main__':
    # 统一走 run_objective_entry，让 O2 的命令行行为和 O1/O3/O4 保持一致。
    raise SystemExit(run_objective_entry('o2', __file__))
