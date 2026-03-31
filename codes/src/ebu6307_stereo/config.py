from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class O1Config:
    repo_root: Path
    middlebury_root: Path
    synthetic_dir: Path
    metrics_file: Path
    shift_pixels: int


@dataclass(frozen=True)
class O2Config:
    keypoints_dir: Path
    matches_dir: Path
    metrics_file: Path
    max_features: int
    contrast_threshold: float
    ratio_test: float
    max_draw_matches: int


@dataclass(frozen=True)
class O3Config:
    disparity_dir: Path
    analysis_dir: Path
    metrics_file: Path
    max_features: int
    contrast_threshold: float
    ratio_test: float
    max_draw_matches: int
    num_disparities: int
    max_vertical_offset: float
    block_size: int
    uniqueness_ratio: int
    speckle_window_size: int
    speckle_range: int
    disp12_max_diff: int
    median_filter_size: int
    census_window_size: int
    census_weight: float
    gradient_weight: float
    consistency_threshold: float
    fill_invalid_passes: int


@dataclass(frozen=True)
class O4Config:
    disparity_dir: Path
    analysis_dir: Path
    metrics_file: Path
    fold_metrics_file: Path
    num_folds: int
    downsample_factor: int
    patch_size: int
    max_disparity: int
    min_disparity: int
    min_confidence: float
    token_median_filter_size: int
    backend: str
    device: str
    prefer_cuda: bool
    execution_mode: str
    dinov2_model_name: str
    dinov2_repo_path: Path | None
    dinov2_checkpoint_path: Path
    model_dim: int
    encoder_hidden_dim: int
    encoder_layers: int
    disparity_regression: str
    softmax_temperature: float
    training_epochs: int
    training_learning_rate: float
    training_batch_size: int
    inference_batch_size: int
    negative_samples: int
    max_training_samples: int
    weight_decay: float
    context_window_size: int
    fine_detail_weight: float
    consistency_threshold: float
    fill_invalid_passes: int
    random_seed: int


@dataclass(frozen=True)
class AppConfig:
    repo_root: Path
    middlebury_root: Path
    o1: O1Config
    o2: O2Config
    o3: O3Config
    o4: O4Config


def load_config(config_path: Path, profile: str) -> AppConfig:
    """读取配置文件并补齐默认值，最终组装成各目标阶段统一使用的 AppConfig。"""

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = parse_simple_yaml(handle.read())

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    profile_block = raw_config.get(profile) if isinstance(raw_config.get(profile), dict) else None
    result_block = raw_config.get("results", {}) if isinstance(raw_config.get("results"), dict) else {}
    o1_block = raw_config.get("o1", {}) if isinstance(raw_config.get("o1"), dict) else {}
    repo_root_value = raw_config.get("repo_root") or (profile_block or {}).get("repo_root")
    repo_root = resolve_path(config_path.resolve().parents[1], repo_root_value) if repo_root_value else config_path.resolve().parents[1]

    middlebury_value = raw_config.get("middlebury_root") or (profile_block or {}).get("middlebury_root")
    if not middlebury_value:
        raise ValueError(
            f"Config {config_path} does not define middlebury_root at the top level or under profile '{profile}'."
        )

    synthetic_value = result_block.get("o1_synthetic_dir") or result_block.get("o1b_synthetic_dir") or "results/O1b_synthetic_data"
    metrics_value = result_block.get("o1_metrics_file") or result_block.get("o1c_metrics_file") or "results/O1c_synthetic_data/SSIM.csv"
    shift_pixels = int(o1_block.get("shift_pixels", 8))

    keypoints_value = result_block.get("o2a_sift_dir") or "results/O2a_sift"
    matches_value = result_block.get("o2b_sift_dir") or "results/O2b_sift"
    o2_metrics_value = result_block.get("o2c_metrics_file") or "results/O2c_sift/metrics.csv"

    o2_block = raw_config.get("o2", {}) if isinstance(raw_config.get("o2"), dict) else {}
    max_features = int(o2_block.get("max_features", 800))
    contrast_threshold = float(o2_block.get("contrast_threshold", 0.04))
    ratio_test = float(o2_block.get("ratio_test", 0.8))
    max_draw_matches = int(o2_block.get("max_draw_matches", 80))

    disparity_value = result_block.get("o3a_disparity_dir") or "results/O3a_disparity"
    analysis_value = result_block.get("o3b_disparity_dir") or "results/O3b_disparity"
    o3_metrics_value = result_block.get("o3c_metrics_file") or "results/O3c_disparity/metrics.csv"

    o3_block = raw_config.get("o3", {}) if isinstance(raw_config.get("o3"), dict) else {}
    o3_max_features = int(o3_block.get("max_features", max_features))
    o3_contrast_threshold = float(o3_block.get("contrast_threshold", contrast_threshold))
    o3_ratio_test = float(o3_block.get("ratio_test", ratio_test))
    o3_max_draw_matches = int(o3_block.get("max_draw_matches", max_draw_matches))
    num_disparities = int(o3_block.get("num_disparities", 64))
    if num_disparities <= 0:
        num_disparities = 16
    if num_disparities % 16 != 0:
        num_disparities = ((num_disparities // 16) + 1) * 16
    max_vertical_offset = max(0.0, float(o3_block.get("max_vertical_offset", 2.0)))

    block_size = int(o3_block.get("block_size", 15))
    if block_size < 5:
        block_size = 5
    if block_size % 2 == 0:
        block_size += 1

    uniqueness_ratio = int(o3_block.get("uniqueness_ratio", 10))
    speckle_window_size = int(o3_block.get("speckle_window_size", 100))
    speckle_range = int(o3_block.get("speckle_range", 8))
    disp12_max_diff = int(o3_block.get("disp12_max_diff", 1))
    median_filter_size = max(1, int(o3_block.get("median_filter_size", 3)))
    if median_filter_size % 2 == 0:
        median_filter_size += 1
    census_window_size = max(3, int(o3_block.get("census_window_size", 5)))
    if census_window_size % 2 == 0:
        census_window_size += 1
    census_window_size = min(census_window_size, 9)
    census_weight = max(0.0, float(o3_block.get("census_weight", 3.0)))
    gradient_weight = max(0.0, float(o3_block.get("gradient_weight", 1.0)))
    consistency_threshold = max(0.0, float(o3_block.get("consistency_threshold", 1.0)))
    fill_invalid_passes = max(0, int(o3_block.get("fill_invalid_passes", 2)))

    o4_disparity_value = result_block.get("o4a_transformer_dir") or "results/O4a_transformer"
    o4_analysis_value = result_block.get("o4b_transformer_dir") or "results/O4b_transformer"
    o4_metrics_value = result_block.get("o4c_metrics_file") or "results/O4c_transformer/metrics.csv"
    o4_fold_metrics_value = result_block.get("o4c_fold_metrics_file") or "results/O4c_transformer/fold_summary.csv"

    o4_block = raw_config.get("o4", {}) if isinstance(raw_config.get("o4"), dict) else {}
    num_folds = max(1, int(o4_block.get("num_folds", 5)))
    downsample_factor = max(1, int(o4_block.get("downsample_factor", 2)))
    patch_size = max(1, int(o4_block.get("patch_size", 4)))
    max_disparity = max(1, int(o4_block.get("max_disparity", 64)))
    min_disparity = max(0, int(o4_block.get("min_disparity", 0)))
    min_confidence = max(0.0, float(o4_block.get("min_confidence", 0.02)))
    token_median_filter_size = max(1, int(o4_block.get("token_median_filter_size", 3)))
    if token_median_filter_size % 2 == 0:
        token_median_filter_size += 1
    backend = str(o4_block.get("backend", "auto")).strip().lower() or "auto"
    if backend not in {"auto", "torch", "numpy"}:
        backend = "auto"
    device = str(o4_block.get("device", "auto")).strip().lower() or "auto"
    if device not in {"auto", "cuda", "cpu"}:
        device = "auto"
    prefer_cuda = bool(o4_block.get("prefer_cuda", True))
    execution_mode = str(o4_block.get("execution_mode", "baseline")).strip().lower() or "baseline"
    if execution_mode not in {"baseline", "dinov2_cost_volume"}:
        raise ValueError(
            "o4.execution_mode must be one of: baseline, dinov2_cost_volume"
        )
    dinov2_model_name = str(o4_block.get("dinov2_model_name", "facebook/dinov2-base")).strip() or "facebook/dinov2-base"
    dinov2_repo_value = o4_block.get("dinov2_repo_path")
    dinov2_checkpoint_value = o4_block.get("dinov2_checkpoint_path")
    from .o4_dinov2 import resolve_dinov2_checkpoint_path
    model_dim = max(4, int(o4_block.get("model_dim", 24)))
    encoder_hidden_dim = max(model_dim, int(o4_block.get("encoder_hidden_dim", max(64, model_dim * 2))))
    encoder_layers = max(1, int(o4_block.get("encoder_layers", 2)))
    disparity_regression = str(o4_block.get("disparity_regression", "quadratic")).strip().lower() or "quadratic"
    if disparity_regression not in {"quadratic", "soft_argmax"}:
        raise ValueError("o4.disparity_regression must be one of: quadratic, soft_argmax")
    softmax_temperature = max(1e-3, float(o4_block.get("softmax_temperature", 1.0)))
    training_epochs = max(0, int(o4_block.get("training_epochs", 24)))
    training_learning_rate = max(1e-5, float(o4_block.get("training_learning_rate", 1e-3)))
    training_batch_size = max(32, int(o4_block.get("training_batch_size", 1024)))
    inference_batch_size = max(32, int(o4_block.get("inference_batch_size", 4096)))
    negative_samples = max(1, int(o4_block.get("negative_samples", 6)))
    max_training_samples = max(128, int(o4_block.get("max_training_samples", 12000)))
    weight_decay = max(0.0, float(o4_block.get("weight_decay", 1e-4)))
    context_window_size = max(1, int(o4_block.get("context_window_size", 3)))
    if context_window_size % 2 == 0:
        context_window_size += 1
    fine_detail_weight = max(0.0, float(o4_block.get("fine_detail_weight", 0.35)))
    consistency_threshold = max(0.0, float(o4_block.get("consistency_threshold", 1.0)))
    fill_invalid_passes = max(0, int(o4_block.get("fill_invalid_passes", 2)))
    random_seed = int(o4_block.get("random_seed", 0))

    middlebury_root = resolve_path(repo_root, middlebury_value)
    return AppConfig(
        repo_root=repo_root,
        middlebury_root=middlebury_root,
        o1=O1Config(
            repo_root=repo_root,
            middlebury_root=middlebury_root,
            synthetic_dir=resolve_path(repo_root, synthetic_value),
            metrics_file=resolve_path(repo_root, metrics_value),
            shift_pixels=shift_pixels,
        ),
        o2=O2Config(
            keypoints_dir=resolve_path(repo_root, keypoints_value),
            matches_dir=resolve_path(repo_root, matches_value),
            metrics_file=resolve_path(repo_root, o2_metrics_value),
            max_features=max_features,
            contrast_threshold=contrast_threshold,
            ratio_test=ratio_test,
            max_draw_matches=max_draw_matches,
        ),
        o3=O3Config(
            disparity_dir=resolve_path(repo_root, disparity_value),
            analysis_dir=resolve_path(repo_root, analysis_value),
            metrics_file=resolve_path(repo_root, o3_metrics_value),
            max_features=o3_max_features,
            contrast_threshold=o3_contrast_threshold,
            ratio_test=o3_ratio_test,
            max_draw_matches=o3_max_draw_matches,
            num_disparities=num_disparities,
            max_vertical_offset=max_vertical_offset,
            block_size=block_size,
            uniqueness_ratio=uniqueness_ratio,
            speckle_window_size=speckle_window_size,
            speckle_range=speckle_range,
            disp12_max_diff=disp12_max_diff,
            median_filter_size=median_filter_size,
            census_window_size=census_window_size,
            census_weight=census_weight,
            gradient_weight=gradient_weight,
            consistency_threshold=consistency_threshold,
            fill_invalid_passes=fill_invalid_passes,
        ),
        o4=O4Config(
            disparity_dir=resolve_path(repo_root, o4_disparity_value),
            analysis_dir=resolve_path(repo_root, o4_analysis_value),
            metrics_file=resolve_path(repo_root, o4_metrics_value),
            fold_metrics_file=resolve_path(repo_root, o4_fold_metrics_value),
            num_folds=num_folds,
            downsample_factor=downsample_factor,
            patch_size=patch_size,
            max_disparity=max_disparity,
            min_disparity=min_disparity,
            min_confidence=min_confidence,
            token_median_filter_size=token_median_filter_size,
            backend=backend,
            device=device,
            prefer_cuda=prefer_cuda,
            execution_mode=execution_mode,
            dinov2_model_name=dinov2_model_name,
            dinov2_repo_path=(
                resolve_path(repo_root, str(dinov2_repo_value))
                if str(dinov2_repo_value or "").strip()
                else None
            ),
            dinov2_checkpoint_path=resolve_path(
                repo_root,
                str(resolve_dinov2_checkpoint_path(dinov2_model_name, dinov2_checkpoint_value)),
            ),
            model_dim=model_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_layers=encoder_layers,
            disparity_regression=disparity_regression,
            softmax_temperature=softmax_temperature,
            training_epochs=training_epochs,
            training_learning_rate=training_learning_rate,
            training_batch_size=training_batch_size,
            inference_batch_size=inference_batch_size,
            negative_samples=negative_samples,
            max_training_samples=max_training_samples,
            weight_decay=weight_decay,
            context_window_size=context_window_size,
            fine_detail_weight=fine_detail_weight,
            consistency_threshold=consistency_threshold,
            fill_invalid_passes=fill_invalid_passes,
            random_seed=random_seed,
        ),
    )


def resolve_path(repo_root: Path, value: str) -> Path:
    """把相对路径解析到仓库根目录下，绝对路径则保持不变。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """解析本项目约定的极简 YAML 子集，只支持缩进映射结构。"""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ValueError("Fallback YAML parser only supports key: value mappings.")

        key, value = line.strip().split(":", 1)
        while indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        value = value.strip()
        if not value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        current[key] = parse_simple_yaml_scalar(value)

    return root


def parse_simple_yaml_scalar(value: str) -> Any:
    """把 YAML 标量文本转成 bool/int/float/str。"""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
