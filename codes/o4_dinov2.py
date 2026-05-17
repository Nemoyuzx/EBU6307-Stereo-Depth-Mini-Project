from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np

from common import (
    copy_if_exists,
    content_bbox_from_gray,
    discover_scenes,
    evaluate_disparity,
    filter_scene_dirs,
    load_gray,
    normalize_for_preview,
    rectangular_mask_from_bbox,
    write_png,
    write_scene_text,
)
from o3 import colorize_disparity_depth_map, solve_sgm_from_cost_torch
from o4_torch import (
    build_token_ground_truth_torch,
    cuda_device_index,
    encode_o4_descriptors_torch,
    is_cuda_device_name,
    train_o4_torch_model,
)
from pfm import read_pfm, write_pfm


_MODEL_CACHE: dict[tuple[str, str, str, str], tuple[Any, int]] = {}
_ACTIVE_DINOV2_IMPORT_ROOT: str | None = None
_DEFAULT_CHECKPOINT_ROOT = Path("/limx_embop/tos/users/Nemo/self-work/models")
_DINO_V2_IMAGE_MEAN = (0.485, 0.456, 0.406)
_DINO_V2_IMAGE_STD = (0.229, 0.224, 0.225)
_DINO_V2_TILE_TOKEN_LIMIT = 4096
_DINO_V2_TILE_OVERLAP_TOKENS = 8


@dataclass(frozen=True)
class O4ExecutionModeStatus:
    requested_mode: str
    selected_mode: str
    descriptor_source: str
    available: bool
    reason: str


@dataclass(frozen=True)
class Dinov2ModelSpec:
    builder_name: str
    checkpoint_filename: str
    selectors: tuple[str, ...]


@dataclass(frozen=True)
class Dinov2StereoPrediction:
    disparity: Any
    raw_disparity: Any
    confidence: Any
    token_height: int
    token_width: int
    descriptor_dim: int
    token_span: int
    model_patch_size: int
    min_token_disparity: int
    max_token_disparity: int
    estimated_working_set_mb: float


_DINO_V2_VARIANTS: tuple[Dinov2ModelSpec, ...] = (
    Dinov2ModelSpec(
        builder_name="dinov2_vits14_reg",
        checkpoint_filename="dinov2_vits14_reg4_pretrain.pth",
        selectors=("facebook/dinov2-small", "dinov2_vits14_reg"),
    ),
    Dinov2ModelSpec(
        builder_name="dinov2_vitb14_reg",
        checkpoint_filename="dinov2_vitb14_reg4_pretrain.pth",
        selectors=("facebook/dinov2-base", "dinov2_vitb14_reg"),
    ),
)
_DINO_V2_MODEL_SPECS: dict[str, Dinov2ModelSpec] = {
    selector: spec for spec in _DINO_V2_VARIANTS for selector in spec.selectors
}
_DINO_V2_CHECKPOINT_FILENAMES = {spec.checkpoint_filename for spec in _DINO_V2_VARIANTS}


def copy_o4_source_images(scene_dir: Path, *output_dirs: Path) -> None:
    """把源场景的原始左右图复制到 O4 结果目录，便于直接查看或提交。"""

    unique_output_dirs: list[Path] = []
    for output_dir in output_dirs:
        if output_dir not in unique_output_dirs:
            unique_output_dirs.append(output_dir)
    for output_dir in unique_output_dirs:
        for image_name in ("im0.png", "im1.png"):
            copy_if_exists(scene_dir / image_name, output_dir / image_name)


def resolve_dinov2_checkpoint_path(model_name: str, checkpoint_path: str | Path | None) -> Path:
    raw_path = str(checkpoint_path).strip() if checkpoint_path is not None else ""
    if raw_path:
        return Path(raw_path).expanduser()

    spec = _resolve_dinov2_model_spec(model_name)
    return _DEFAULT_CHECKPOINT_ROOT / spec.checkpoint_filename


def resolve_o4_execution_mode(
    requested_mode: str,
    use_torch: bool,
    dinov2_model_name: str,
    dinov2_repo_path: str | Path | None,
    dinov2_checkpoint_path: str | Path | None,
) -> O4ExecutionModeStatus:
    mode = str(requested_mode or "baseline").strip().lower() or "baseline"
    if mode == "baseline":
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="baseline",
            descriptor_source="handcrafted_patch_tokens",
            available=True,
            reason="using the existing trainable token projection baseline",
        )

    if mode != "dinov2_cost_volume":
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode=mode,
            descriptor_source="unknown",
            available=False,
            reason=f"unsupported O4 execution mode: {requested_mode!r}",
        )

    if not use_torch:
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_direct_model_patch_tokens",
            available=False,
            reason="dinov2 direct model prediction requires the torch backend",
        )

    if not str(dinov2_model_name).strip():
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_direct_model_patch_tokens",
            available=False,
            reason="dinov2 direct model prediction requires o4.dinov2_model_name",
        )

    try:
        spec = _resolve_dinov2_model_spec(dinov2_model_name)
    except ValueError as exc:
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_direct_model_patch_tokens",
            available=False,
            reason=str(exc),
        )

    resolved_checkpoint = resolve_dinov2_checkpoint_path(dinov2_model_name, dinov2_checkpoint_path)
    if not resolved_checkpoint.is_file():
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_direct_model_patch_tokens",
            available=False,
            reason=(
                "DINOv2 direct model prediction requires o4.dinov2_checkpoint_path to point at a local checkpoint file; "
                f"missing: {resolved_checkpoint}. Expected filename for {spec.builder_name}: {spec.checkpoint_filename}"
            ),
        )
    try:
        resolved_repo_path = resolve_dinov2_repo_path(dinov2_repo_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_direct_model_patch_tokens",
            available=False,
            reason=str(exc),
        )

    return O4ExecutionModeStatus(
        requested_mode=mode,
        selected_mode="dinov2_cost_volume",
        descriptor_source="dinov2_direct_model_patch_tokens",
        available=True,
        reason=(
            "using pretrained DINOv2 model forward pass to predict stereo disparity from patch tokens "
            f"(model={spec.builder_name}, repo={resolved_repo_path or 'environment'}, checkpoint={resolved_checkpoint})"
        ),
    )


def _require_torch() -> Any:
    import torch

    return torch


def _resolve_dinov2_model_spec(model_name: str) -> Dinov2ModelSpec:
    normalized = str(model_name).strip()
    if normalized in _DINO_V2_MODEL_SPECS:
        return _DINO_V2_MODEL_SPECS[normalized]
    raise ValueError(
        "Unsupported O4 DINOv2 model selector "
        f"{model_name!r}. Expected one of: {', '.join(sorted(_DINO_V2_MODEL_SPECS))}"
    )


def resolve_dinov2_repo_path(repo_path: str | Path | None) -> Path | None:
    raw_path = str(repo_path).strip() if repo_path is not None else ""
    if not raw_path:
        return None

    resolved = Path(raw_path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            "Configured O4 DINOv2 repo path was not found: "
            f"{resolved}. Update o4.dinov2_repo_path or --o4-dinov2-repo to a local DINOv2 checkout root."
        )
    if not resolved.is_dir():
        raise NotADirectoryError(
            "Configured O4 DINOv2 repo path is not a directory: "
            f"{resolved}. Point o4.dinov2_repo_path or --o4-dinov2-repo at a local DINOv2 checkout root."
        )

    package_dir = resolved / "dinov2" / "hub"
    if not (
        (package_dir / "backbones.py").is_file()
        or (package_dir / "backbones" / "__init__.py").is_file()
    ):
        raise ValueError(
            "Configured O4 DINOv2 repo path does not expose dinov2.hub.backbones: "
            f"{resolved}. Expected dinov2/hub/backbones.py under that directory."
        )
    return resolved


def _prepare_downsampled_gray(image: Any, downsample_factor: int, device: str) -> Any:
    torch = _require_torch()
    import torch.nn.functional as F

    if torch.is_tensor(image):
        source = image.to(device=device, dtype=torch.float32)
    else:
        source = torch.as_tensor(image.copy() if hasattr(image, "copy") else image, dtype=torch.float32, device=device)

    if source.ndim != 2:
        raise ValueError("DINOv2 extraction expects a single-channel grayscale image.")

    factor = max(1, int(downsample_factor))
    if factor <= 1:
        return source / 255.0

    height, width = source.shape
    pooled_height = max(1, height // factor)
    pooled_width = max(1, width // factor)
    trimmed = source[: pooled_height * factor, : pooled_width * factor]
    if trimmed.numel() == 0:
        raise ValueError("Image is too small for the requested downsample factor.")
    pooled = F.avg_pool2d(trimmed.unsqueeze(0).unsqueeze(0), kernel_size=factor, stride=factor)
    return pooled.squeeze(0).squeeze(0) / 255.0


def _resolve_patch_size(model: Any) -> int:
    patch_size = getattr(model, "patch_size", None)
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    if isinstance(patch_size, int) and patch_size > 0:
        return patch_size

    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None:
        embed_patch_size = getattr(patch_embed, "patch_size", None)
        if isinstance(embed_patch_size, tuple):
            embed_patch_size = embed_patch_size[0]
        if isinstance(embed_patch_size, int) and embed_patch_size > 0:
            return embed_patch_size
        proj = getattr(patch_embed, "proj", None)
        kernel_size = getattr(proj, "kernel_size", None)
        if isinstance(kernel_size, tuple) and kernel_size and isinstance(kernel_size[0], int):
            return int(kernel_size[0])

    raise ValueError("Unable to infer the DINOv2 patch size from the loaded model.")


def _extract_checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return _normalize_checkpoint_state_dict(nested)
        if payload:
            return _normalize_checkpoint_state_dict(payload)
    raise ValueError("Unsupported DINOv2 checkpoint format.")


def _normalize_checkpoint_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = str(key)
        for prefix in ("module.", "backbone."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized[normalized_key] = value
    return normalized


def _prepare_local_dinov2_import(repo_path: str | Path | None) -> Path | None:
    global _ACTIVE_DINOV2_IMPORT_ROOT

    resolved_repo_path = resolve_dinov2_repo_path(repo_path)
    repo_marker = str(resolved_repo_path) if resolved_repo_path is not None else None
    if repo_marker != _ACTIVE_DINOV2_IMPORT_ROOT:
        for module_name in ("dinov2.hub.backbones", "dinov2.hub", "dinov2"):
            sys.modules.pop(module_name, None)
        _ACTIVE_DINOV2_IMPORT_ROOT = repo_marker
    if resolved_repo_path is not None:
        repo_entry = str(resolved_repo_path)
        if repo_entry in sys.path:
            sys.path.remove(repo_entry)
        sys.path.insert(0, repo_entry)
    importlib.invalidate_caches()
    return resolved_repo_path


def _load_local_dinov2_builder(model_name: str, repo_path: str | Path | None) -> Any:
    resolved_repo_path = _prepare_local_dinov2_import(repo_path)
    try:
        dinov2_backbones = importlib.import_module("dinov2.hub.backbones")
    except ImportError as exc:
        raise ImportError(
            "The O4 DINOv2 path requires a local DINOv2 Python implementation on PYTHONPATH "
            f"or in the current environment. repo_path={resolved_repo_path or 'environment'}. "
            "Install or expose the local facebookresearch/dinov2 package instead of relying on "
            "torch.hub remote loading."
        ) from exc

    builder = getattr(dinov2_backbones, model_name, None)
    if builder is None or not callable(builder):
        raise ValueError(
            "The local DINOv2 implementation does not expose the requested model builder "
            f"{model_name!r} in dinov2.hub.backbones. Expected one of: "
            f"{', '.join(spec.builder_name for spec in _DINO_V2_VARIANTS)}"
        )
    return builder


def _load_dinov2_model(
    model_name: str,
    checkpoint_path: str | Path,
    device: str,
    repo_path: str | Path | None = None,
) -> tuple[Any, int]:
    checkpoint = resolve_dinov2_checkpoint_path(model_name, checkpoint_path)
    resolved_repo_path = resolve_dinov2_repo_path(repo_path)
    spec = _resolve_dinov2_model_spec(model_name)
    cache_key = (spec.builder_name, str(checkpoint), str(device), str(resolved_repo_path or ""))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Configured O4 DINOv2 checkpoint was not found: "
            f"{checkpoint}. Update o4.dinov2_checkpoint_path to an existing local file for "
            f"{spec.builder_name} (expected filename: {spec.checkpoint_filename})."
        )

    checkpoint_filename = checkpoint.name
    if (
        checkpoint_filename in _DINO_V2_CHECKPOINT_FILENAMES
        and checkpoint_filename != spec.checkpoint_filename
    ):
        raise ValueError(
            "Configured O4 DINOv2 checkpoint filename does not match the selected model variant: "
            f"selector={model_name!r}, builder={spec.builder_name!r}, "
            f"expected_checkpoint={spec.checkpoint_filename!r}, got_checkpoint={checkpoint_filename!r}"
        )

    torch = _require_torch()
    model_builder = _load_local_dinov2_builder(spec.builder_name, resolved_repo_path)
    model = model_builder(pretrained=False)
    state_dict = _extract_checkpoint_state_dict(torch.load(checkpoint, map_location="cpu"))
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", ()))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", ()))
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Failed to load the configured O4 DINOv2 checkpoint cleanly. "
            f"selector={model_name!r}, builder={spec.builder_name!r}, checkpoint={checkpoint}. "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    model = model.to(device)
    model.eval()
    patch_size = _resolve_patch_size(model)
    _MODEL_CACHE[cache_key] = (model, patch_size)
    return model, patch_size


def _prepare_dinov2_pixel_values(image: Any) -> Any:
    torch = _require_torch()

    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image tensor, got rank {image.ndim}")

    rgb = image.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    mean = torch.tensor(_DINO_V2_IMAGE_MEAN, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(_DINO_V2_IMAGE_STD, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _extract_patch_tokens(model: Any, pixel_values: Any, expected_tokens: int) -> Any:
    with _require_torch().inference_mode():
        if hasattr(model, "forward_features"):
            outputs = model.forward_features(pixel_values)
        else:
            outputs = model(pixel_values)

    if isinstance(outputs, dict):
        for key in ("x_norm_patchtokens", "x_prenorm", "last_hidden_state"):
            tokens = outputs.get(key)
            if tokens is not None:
                break
        else:
            tokens = None
    else:
        tokens = getattr(outputs, "last_hidden_state", outputs)

    if tokens is None:
        raise ValueError("DINOv2 model returned no patch tokens")

    if getattr(tokens, "ndim", 0) != 3:
        raise ValueError(f"Unexpected DINOv2 token tensor rank: {getattr(tokens, 'ndim', 'unknown')}")

    token_count = int(tokens.shape[1])
    if token_count == expected_tokens:
        return tokens
    if token_count > expected_tokens:
        return tokens[:, token_count - expected_tokens :, :]
    raise ValueError(f"unexpected DINOv2 token count {token_count} for grid size {expected_tokens}")


def extract_dinov2_patch_tokens(
    image: Any,
    *,
    downsample_factor: int,
    input_scale: int,
    model_name: str,
    repo_path: str | Path | None,
    checkpoint_path: str | Path,
    device: str,
    return_numpy: bool,
) -> tuple[Any, int, int, int]:
    torch = _require_torch()
    import torch.nn.functional as F

    model, patch_size = _load_dinov2_model(model_name, checkpoint_path, device, repo_path=repo_path)
    source = _prepare_downsampled_gray(image, downsample_factor, device)
    scale = max(1, int(input_scale))
    if scale > 1:
        source = F.interpolate(
            source.unsqueeze(0).unsqueeze(0),
            scale_factor=float(scale),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    height, width = source.shape
    token_height = height // patch_size
    token_width = width // patch_size
    if token_height <= 0 or token_width <= 0:
        raise ValueError(
            f"downsampled image size {height}x{width} is smaller than the DINOv2 patch size {patch_size}"
        )

    cropped_height = token_height * patch_size
    cropped_width = token_width * patch_size
    cropped = source[:cropped_height, :cropped_width].clamp(0.0, 1.0)
    expected_tokens = token_height * token_width

    if expected_tokens <= _DINO_V2_TILE_TOKEN_LIMIT:
        pixel_values = _prepare_dinov2_pixel_values(cropped).to(device=device)
        tokens = _extract_patch_tokens(model, pixel_values, expected_tokens)
        descriptors = tokens.reshape(token_height, token_width, -1).contiguous().to(dtype=pixel_values.dtype)
    else:
        tile_height, tile_width = _select_dinov2_tile_shape(token_height, token_width)
        row_starts = _tile_token_starts(token_height, tile_height, _DINO_V2_TILE_OVERLAP_TOKENS)
        col_starts = _tile_token_starts(token_width, tile_width, _DINO_V2_TILE_OVERLAP_TOKENS)
        descriptor_sum = None
        descriptor_count = torch.zeros((token_height, token_width, 1), dtype=torch.float32, device=device)
        for row_start in row_starts:
            row_end = row_start + tile_height
            for col_start in col_starts:
                col_end = col_start + tile_width
                tile = cropped[
                    row_start * patch_size : row_end * patch_size,
                    col_start * patch_size : col_end * patch_size,
                ]
                pixel_values = _prepare_dinov2_pixel_values(tile).to(device=device)
                tile_tokens = _extract_patch_tokens(model, pixel_values, tile_height * tile_width)
                tile_descriptors = tile_tokens.reshape(tile_height, tile_width, -1).contiguous().to(dtype=torch.float32)
                tile_descriptors = F.normalize(tile_descriptors, dim=2, eps=1e-6)
                if descriptor_sum is None:
                    descriptor_sum = torch.zeros((token_height, token_width, tile_descriptors.shape[2]), dtype=torch.float32, device=device)
                descriptor_sum[row_start:row_end, col_start:col_end, :] += tile_descriptors
                descriptor_count[row_start:row_end, col_start:col_end, :] += 1.0
        if descriptor_sum is None:
            raise ValueError("DINOv2 tiled patch-token extraction produced no tiles.")
        descriptors = descriptor_sum / torch.clamp(descriptor_count, min=1.0)

    descriptors = F.normalize(descriptors, dim=2, eps=1e-6).to(dtype=_require_torch().float32)
    if return_numpy:
        return descriptors.detach().cpu().numpy().astype("float32"), cropped_height, cropped_width, patch_size
    return descriptors, cropped_height, cropped_width, patch_size


def _select_dinov2_tile_shape(token_height: int, token_width: int) -> tuple[int, int]:
    token_limit = max(1, int(_DINO_V2_TILE_TOKEN_LIMIT))
    height = max(1, int(token_height))
    width = max(1, int(token_width))
    if height * width <= token_limit:
        return height, width
    side = max(1, int(token_limit ** 0.5))
    aspect = width / float(height)
    tile_height = max(1, min(height, int(round(side / max(aspect ** 0.5, 1e-6)))))
    tile_width = max(1, min(width, token_limit // tile_height))
    while tile_height * tile_width > token_limit:
        if tile_width >= tile_height and tile_width > 1:
            tile_width -= 1
        elif tile_height > 1:
            tile_height -= 1
        else:
            break
    return tile_height, tile_width


def _tile_token_starts(total_tokens: int, tile_tokens: int, overlap_tokens: int) -> list[int]:
    total = int(total_tokens)
    tile = max(1, min(int(tile_tokens), total))
    if total <= tile:
        return [0]
    overlap = max(0, min(int(overlap_tokens), tile - 1))
    stride = max(1, tile - overlap)
    starts = list(range(0, total - tile + 1, stride))
    last = total - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _as_float_tensor(value: Any, device: str) -> Any:
    torch = _require_torch()
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _predict_token_disparity_torch(
    left_tokens: Any,
    right_tokens: Any,
    min_disparity: int,
    max_disparity: int,
    *,
    device: str,
    target_direction: str = "negative",
    reference_valid_mask: Any | None = None,
    target_valid_mask: Any | None = None,
) -> tuple[Any, Any, Any]:
    torch = _require_torch()

    left = _as_float_tensor(left_tokens, device)
    right = _as_float_tensor(right_tokens, device)
    token_height, token_width, _ = left.shape
    reference_valid = None if reference_valid_mask is None else reference_valid_mask.to(device=device, dtype=torch.bool)
    target_valid = None if target_valid_mask is None else target_valid_mask.to(device=device, dtype=torch.bool)
    best_scores = torch.full((token_height, token_width), -torch.inf, dtype=torch.float32, device=device)
    second_scores = torch.full((token_height, token_width), -torch.inf, dtype=torch.float32, device=device)
    best_disparities = torch.zeros((token_height, token_width), dtype=torch.int32, device=device)
    all_scores = torch.full(
        (int(max_disparity) - int(min_disparity) + 1, token_height, token_width),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )

    for score_index, disparity in enumerate(range(int(min_disparity), int(max_disparity) + 1)):
        current_scores = torch.full((token_height, token_width), -torch.inf, dtype=torch.float32, device=device)
        if disparity == 0:
            current_scores = (left * right).sum(dim=2)
            if reference_valid is not None and target_valid is not None:
                current_scores = current_scores.masked_fill(~(reference_valid & target_valid), -torch.inf)
        elif target_direction == "positive" and disparity < token_width:
            shifted_scores = (
                left[:, : token_width - disparity, :] * right[:, disparity:, :]
            ).sum(dim=2)
            if reference_valid is not None and target_valid is not None:
                shifted_valid = reference_valid[:, : token_width - disparity] & target_valid[:, disparity:]
                shifted_scores = shifted_scores.masked_fill(~shifted_valid, -torch.inf)
            current_scores[:, : token_width - disparity] = shifted_scores
        elif disparity < token_width:
            shifted_scores = (
                left[:, disparity:, :] * right[:, : token_width - disparity, :]
            ).sum(dim=2)
            if reference_valid is not None and target_valid is not None:
                shifted_valid = reference_valid[:, disparity:] & target_valid[:, : token_width - disparity]
                shifted_scores = shifted_scores.masked_fill(~shifted_valid, -torch.inf)
            current_scores[:, disparity:] = shifted_scores

        all_scores[score_index] = current_scores
        replace_mask = current_scores > best_scores
        second_scores = torch.where(replace_mask, best_scores, torch.maximum(second_scores, current_scores))
        best_scores = torch.where(replace_mask, current_scores, best_scores)
        best_disparities = torch.where(replace_mask, torch.full_like(best_disparities, disparity), best_disparities)

    confidence = torch.where(torch.isfinite(best_scores), best_scores - second_scores, torch.zeros_like(best_scores))
    return best_disparities, confidence, all_scores


def _refine_token_disparity_torch(best_disparity: Any, all_scores: Any, min_disparity: int, max_disparity: int, *, device: str) -> Any:
    torch = _require_torch()
    refined = _as_float_tensor(best_disparity, device).clone()
    scores = _as_float_tensor(all_scores, device)
    if int(max_disparity) <= int(min_disparity):
        return refined

    score_index = torch.round(refined).to(dtype=torch.long) - int(min_disparity)
    valid = (score_index > 0) & (score_index < scores.shape[0] - 1)
    rows, cols = torch.nonzero(valid, as_tuple=True)
    if rows.numel() == 0:
        return refined

    center_index = score_index[rows, cols]
    center = scores[center_index, rows, cols]
    left = scores[center_index - 1, rows, cols]
    right = scores[center_index + 1, rows, cols]
    denominator = left - (2.0 * center) + right
    stable = torch.isfinite(center) & torch.isfinite(left) & torch.isfinite(right) & (torch.abs(denominator) > 1e-6)
    offset = torch.zeros_like(center)
    offset[stable] = 0.5 * (left[stable] - right[stable]) / denominator[stable]
    refined[rows, cols] += torch.clamp(offset, -1.0, 1.0)
    return refined


def _soft_argmax_disparity_torch(all_scores: Any, min_disparity: int, max_disparity: int, temperature: float, *, device: str) -> Any:
    torch = _require_torch()
    scores = _as_float_tensor(all_scores, device)
    stabilized = scores - torch.max(scores, dim=0, keepdim=True).values
    stabilized = torch.where(torch.isfinite(stabilized), stabilized, torch.full_like(stabilized, -1e4))
    logits = stabilized / max(float(temperature), 1e-3)
    probability = torch.softmax(logits, dim=0)
    values = torch.arange(int(min_disparity), int(max_disparity) + 1, dtype=torch.float32, device=device).view(-1, 1, 1)
    refined = (probability * values).sum(dim=0)
    valid = torch.isfinite(scores).any(dim=0)
    return torch.where(valid, refined, torch.zeros_like(refined)).float()


def _left_right_consistency_mask_torch(left_disparity: Any, right_disparity: Any, threshold: float, *, device: str) -> Any:
    torch = _require_torch()
    left = _as_float_tensor(left_disparity, device)
    right = _as_float_tensor(right_disparity, device)
    rows, cols = torch.meshgrid(
        torch.arange(left.shape[0], device=device),
        torch.arange(left.shape[1], device=device),
        indexing="ij",
    )
    partner_cols = torch.round(cols.float() - left).long()
    valid = (left > 0) & (partner_cols >= 0) & (partner_cols < left.shape[1])
    sampled = torch.zeros_like(left)
    sampled[valid] = right[rows[valid], partner_cols[valid]]
    return valid & (torch.abs(left - sampled) <= float(threshold))


def _fill_invalid_disparity_torch(disparity: Any, passes: int, *, device: str) -> Any:
    torch = _require_torch()
    import torch.nn.functional as F

    source = _as_float_tensor(disparity, device)
    filled = source.clone()
    for _ in range(max(0, int(passes))):
        valid = filled > 0
        if bool(torch_bool_all(valid)):
            break
        padded_values = F.pad(filled.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        padded_valid = F.pad(valid.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        value_windows = F.unfold(padded_values, kernel_size=(3, 3)).reshape(1, 9, filled.shape[0], filled.shape[1])
        valid_windows = F.unfold(padded_valid, kernel_size=(3, 3)).reshape(1, 9, filled.shape[0], filled.shape[1])
        counts = valid_windows.sum(dim=1).squeeze(0)
        sums = (value_windows * valid_windows).sum(dim=1).squeeze(0)
        estimates = sums / torch.clamp(counts, min=1.0)
        filled = torch.where((~valid) & (counts > 0), estimates, filled)
    return filled.float()


def torch_bool_all(mask: Any) -> bool:
    return bool(mask.detach().all().item())


def _median_filter_2d_torch(image: Any, kernel_size: int, *, device: str) -> Any:
    import torch.nn.functional as F

    source = _as_float_tensor(image, device)
    if int(kernel_size) <= 1:
        return source
    radius = int(kernel_size) // 2
    padded = F.pad(source.unsqueeze(0).unsqueeze(0), (radius, radius, radius, radius), mode="replicate")
    windows = F.unfold(padded, kernel_size=(int(kernel_size), int(kernel_size)))
    return windows.median(dim=1).values.reshape(source.shape[0], source.shape[1]).to(dtype=source.dtype)


def _upsample_token_grid_torch(token_values: Any, span: int, output_height: int, output_width: int, *, device: str) -> Any:
    source = _as_float_tensor(token_values, device)
    upsampled = source.repeat_interleave(int(span), dim=0).repeat_interleave(int(span), dim=1)
    result = source.new_zeros((int(output_height), int(output_width)))
    copy_height = min(int(output_height), int(upsampled.shape[0]))
    copy_width = min(int(output_width), int(upsampled.shape[1]))
    result[:copy_height, :copy_width] = upsampled[:copy_height, :copy_width]
    return result.float()


def _build_token_content_mask_torch(content_mask: Any, token_span: int, token_shape: tuple[int, int], *, device: str) -> Any:
    torch = _require_torch()
    import numpy as np

    source = np.asarray(content_mask, dtype=bool)
    token_height, token_width = int(token_shape[0]), int(token_shape[1])
    if source.size == 0 or token_height <= 0 or token_width <= 0:
        return torch.zeros((token_height, token_width), dtype=torch.bool, device=device)
    rows = np.clip((np.arange(token_height) * int(token_span)) + (int(token_span) // 2), 0, source.shape[0] - 1)
    cols = np.clip((np.arange(token_width) * int(token_span)) + (int(token_span) // 2), 0, source.shape[1] - 1)
    return torch.as_tensor(source[rows[:, None], cols[None, :]], dtype=torch.bool, device=device)


def _stereo_cost_valid_mask_torch(reference_mask: Any, target_mask: Any, min_disparity: int, max_disparity: int, target_direction: str, *, device: str) -> Any:
    torch = _require_torch()

    reference = reference_mask.to(device=device, dtype=torch.bool)
    target = target_mask.to(device=device, dtype=torch.bool)
    height, width = reference.shape
    disparities = list(range(int(min_disparity), int(max_disparity) + 1))
    valid = torch.zeros((height, width, len(disparities)), dtype=torch.bool, device=device)
    for index, disparity in enumerate(disparities):
        if disparity == 0:
            valid[:, :, index] = reference & target
        elif target_direction == "positive" and disparity < width:
            valid[:, : width - disparity, index] = reference[:, : width - disparity] & target[:, disparity:]
        elif disparity < width:
            valid[:, disparity:, index] = reference[:, disparity:] & target[:, : width - disparity]
    return valid


def _build_nonblack_content_mask(image: Any, threshold: float = 2.0, min_fraction: float = 0.005) -> Any:
    source = np.asarray(image)
    if source.ndim == 3:
        source = source.max(axis=2)
    if source.ndim != 2:
        raise ValueError("_build_nonblack_content_mask expects a 2D grayscale image or an RGB image.")

    bbox_mask = rectangular_mask_from_bbox(
        source.shape,
        content_bbox_from_gray(source, threshold=threshold, min_fraction=min_fraction),
    )
    nonblack = np.isfinite(source) & (source.astype(np.float32) > float(threshold))
    if not bool(np.any(nonblack)):
        return np.ones(source.shape, dtype=bool)
    return bbox_mask & nonblack


def _token_mask_bbox_torch(mask: Any) -> tuple[int, int, int, int] | None:
    locations = mask.nonzero(as_tuple=False)
    if int(locations.numel()) == 0:
        return None
    top = int(locations[:, 0].min().detach().cpu().item())
    bottom = int(locations[:, 0].max().detach().cpu().item() + 1)
    left = int(locations[:, 1].min().detach().cpu().item())
    right = int(locations[:, 1].max().detach().cpu().item() + 1)
    return top, bottom, left, right


def _solve_content_masked_sgm_from_cost_torch(cost_volume: Any, guide_gray: Any, content_mask: Any, *, min_disparity: int, device: str) -> tuple[Any, Any]:
    torch = _require_torch()

    mask = content_mask.to(device=device, dtype=torch.bool)
    bbox = _token_mask_bbox_torch(mask)
    best = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    margin = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    if bbox is None:
        return best, margin
    top, bottom, left, right = bbox
    crop_best, crop_margin = solve_sgm_from_cost_torch(
        cost_volume[top:bottom, left:right, :],
        guide_gray[top:bottom, left:right],
        min_disparity=int(min_disparity),
        device=device,
        return_numpy=False,
    )
    crop_mask = mask[top:bottom, left:right]
    best[top:bottom, left:right] = crop_best.masked_fill(~crop_mask, 0.0).float()
    margin[top:bottom, left:right] = crop_margin.masked_fill(~crop_mask, 0.0).float()
    return best, margin


def _build_token_guide_torch(image: Any, downsample_factor: int, input_scale: int, patch_size: int, token_shape: tuple[int, int], *, device: str) -> Any:
    import torch.nn.functional as F

    source = _prepare_downsampled_gray(image, downsample_factor, device) * 255.0
    scale = max(1, int(input_scale))
    if scale > 1:
        source = F.interpolate(
            source.unsqueeze(0).unsqueeze(0),
            scale_factor=float(scale),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    token_height, token_width = int(token_shape[0]), int(token_shape[1])
    cropped = source[: token_height * int(patch_size), : token_width * int(patch_size)].contiguous()
    if cropped.numel() == 0:
        return source.new_zeros((token_height, token_width), dtype=source.dtype)
    guide = cropped.reshape(token_height, int(patch_size), token_width, int(patch_size))
    return guide.permute(0, 2, 1, 3).mean(dim=(2, 3)).to(dtype=source.dtype)


def _apply_content_mask(disparity: Any, content_mask: Any) -> Any:
    import numpy as np

    source = np.asarray(disparity, dtype=np.float32)
    mask = np.asarray(content_mask, dtype=bool)
    return np.where(mask & np.isfinite(source) & (source > 0), source, 0.0).astype(np.float32)


def _sample_dinov2_training_pairs(
    left_tokens: Any,
    right_tokens: Any,
    labels: Any,
    *,
    min_disparity: int,
    max_disparity: int,
    sample_limit: int,
    negative_samples: int,
    generator: Any,
    device: str,
) -> tuple[Any, Any]:
    torch = _require_torch()

    limit = max(0, int(sample_limit))
    negatives_count = max(1, int(negative_samples))
    feature_dim = int(left_tokens.shape[2])
    if limit <= 0:
        return (
            torch.zeros((0, feature_dim), dtype=torch.float32, device=device),
            torch.zeros((0, negatives_count + 1, feature_dim), dtype=torch.float32, device=device),
        )

    valid_locations = torch.nonzero((labels >= int(min_disparity)) & (labels <= int(max_disparity)), as_tuple=False)
    if int(valid_locations.numel()) == 0:
        return (
            torch.zeros((0, feature_dim), dtype=torch.float32, device=device),
            torch.zeros((0, negatives_count + 1, feature_dim), dtype=torch.float32, device=device),
        )
    order = torch.randperm(valid_locations.shape[0], generator=generator, device=device)
    valid_locations = valid_locations.index_select(0, order)

    rows = valid_locations[:, 0].to(dtype=torch.int64)
    cols = valid_locations[:, 1].to(dtype=torch.int64)
    disparities = labels[rows, cols].to(dtype=torch.int64)
    match_columns = cols - disparities
    sample_mask = (
        (match_columns >= 0)
        & (match_columns < int(right_tokens.shape[1]))
        & (disparities >= int(min_disparity))
        & (disparities <= int(max_disparity))
    )
    rows = rows[sample_mask][:limit]
    cols = cols[sample_mask][:limit]
    disparities = disparities[sample_mask][:limit]
    sample_count = int(rows.shape[0])
    if sample_count == 0:
        return (
            torch.zeros((0, feature_dim), dtype=torch.float32, device=device),
            torch.zeros((0, negatives_count + 1, feature_dim), dtype=torch.float32, device=device),
        )

    hard_count = min(negatives_count, 6)
    hard_offsets = torch.tensor((1, -1, 2, -2, 3, -3), dtype=torch.int64, device=device)[:hard_count]
    hard_negatives = disparities[:, None] + hard_offsets[None, :]
    random_count = negatives_count - hard_count
    if random_count > 0:
        random_negatives = torch.randint(
            int(min_disparity),
            int(max_disparity) + 1,
            (sample_count, random_count),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        negatives = torch.cat([hard_negatives, random_negatives], dim=1)
    else:
        negatives = hard_negatives

    valid_negative = (
        (negatives != disparities[:, None])
        & ((cols[:, None] - negatives) >= 0)
        & ((cols[:, None] - negatives) < int(right_tokens.shape[1]))
    )
    retry_count = 0
    while not bool(valid_negative.all()) and retry_count < 32:
        replacement = torch.randint(
            int(min_disparity),
            int(max_disparity) + 1,
            negatives.shape,
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        negatives = torch.where(valid_negative, negatives, replacement)
        valid_negative = (
            (negatives != disparities[:, None])
            & ((cols[:, None] - negatives) >= 0)
            & ((cols[:, None] - negatives) < int(right_tokens.shape[1]))
        )
        retry_count += 1
    if not bool(valid_negative.all()):
        negatives = torch.clamp(negatives, min=int(min_disparity), max=int(max_disparity))
        safe_columns = torch.clamp(cols[:, None] - negatives, min=0, max=int(right_tokens.shape[1]) - 1)
    else:
        safe_columns = cols[:, None] - negatives

    candidate_columns = torch.cat([(cols - disparities)[:, None], safe_columns], dim=1)
    left_samples = left_tokens[rows, cols].to(dtype=torch.float32)
    candidate_samples = right_tokens[rows[:, None], candidate_columns].to(dtype=torch.float32)
    return left_samples, candidate_samples


def _collect_dinov2_training_samples_by_fold(
    scene_dirs: list[Path],
    scene_fold_map: dict[str, int],
    config: Any,
    *,
    device: str,
) -> dict[int, tuple[Any, Any, int]]:
    torch = _require_torch()

    fold_count = max(1, int(config.num_folds))
    max_samples = max(1, int(config.max_training_samples))
    training_counts = {
        fold: max(1, sum(1 for scene_dir in scene_dirs if scene_fold_map[scene_dir.name] != fold))
        for fold in range(fold_count)
    }
    per_fold_scene_limit = {
        fold: max(1, int(np.ceil(max_samples / float(training_counts[fold]))))
        for fold in range(fold_count)
    }
    remaining = {fold: max_samples for fold in range(fold_count)}
    left_batches: dict[int, list[Any]] = {fold: [] for fold in range(fold_count)}
    candidate_batches: dict[int, list[Any]] = {fold: [] for fold in range(fold_count)}
    generators = {fold: torch.Generator(device=device).manual_seed(int(config.random_seed) + fold) for fold in range(fold_count)}

    for scene_index, scene_dir in enumerate(scene_dirs, start=1):
        ground_truth_path = scene_dir / "disp0.pfm"
        if not ground_truth_path.exists():
            continue
        left_gray = load_gray(scene_dir / "im0.png")
        right_gray = load_gray(scene_dir / "im1.png")
        left_tokens, _, _, model_patch_size = extract_dinov2_patch_tokens(
            left_gray,
            downsample_factor=config.downsample_factor,
            input_scale=config.dinov2_input_scale,
            model_name=config.dinov2_model_name,
            repo_path=config.dinov2_repo_path,
            checkpoint_path=config.dinov2_checkpoint_path,
            device=device,
            return_numpy=False,
        )
        right_tokens, _, _, _ = extract_dinov2_patch_tokens(
            right_gray,
            downsample_factor=config.downsample_factor,
            input_scale=config.dinov2_input_scale,
            model_name=config.dinov2_model_name,
            repo_path=config.dinov2_repo_path,
            checkpoint_path=config.dinov2_checkpoint_path,
            device=device,
            return_numpy=False,
        )
        token_span = max(1, int(round((float(config.downsample_factor) * float(model_patch_size)) / float(max(1, int(config.dinov2_input_scale))))))
        max_token_disparity = min(
            max(int(config.min_disparity), int(config.max_disparity) // token_span),
            max(0, int(left_tokens.shape[1]) - 1),
        )
        min_token_disparity = min(int(config.min_disparity) // token_span, max_token_disparity)
        ground_truth = read_pfm(ground_truth_path)
        if getattr(ground_truth, "ndim", 0) == 3:
            ground_truth = ground_truth[:, :, 0]
        labels = build_token_ground_truth_torch(
            ground_truth,
            token_span,
            tuple(left_tokens.shape[:2]),
            device=device,
        )
        scene_fold = int(scene_fold_map[scene_dir.name])
        for fold in range(fold_count):
            if fold == scene_fold or remaining[fold] <= 0:
                continue
            sample_limit = min(remaining[fold], per_fold_scene_limit[fold])
            left_samples, candidate_samples = _sample_dinov2_training_pairs(
                left_tokens,
                right_tokens,
                labels,
                min_disparity=min_token_disparity,
                max_disparity=max_token_disparity,
                sample_limit=sample_limit,
                negative_samples=config.negative_samples,
                generator=generators[fold],
                device=device,
            )
            if int(left_samples.shape[0]) == 0:
                continue
            left_batches[fold].append(left_samples.detach().cpu())
            candidate_batches[fold].append(candidate_samples.detach().cpu())
            remaining[fold] -= int(left_samples.shape[0])
        print(
            f"Collected DINOv2 adapter samples from {scene_dir.name} ({scene_index}/{len(scene_dirs)}); "
            + ", ".join(f"fold{fold}={max_samples - remaining[fold]}" for fold in range(fold_count)),
            flush=True,
        )
        del left_tokens, right_tokens, labels
        if is_cuda_device_name(device):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    result: dict[int, tuple[Any, Any, int]] = {}
    for fold in range(fold_count):
        if left_batches[fold]:
            left_tensor = torch.cat(left_batches[fold], dim=0)
            candidate_tensor = torch.cat(candidate_batches[fold], dim=0)
            result[fold] = (left_tensor, candidate_tensor, int(left_tensor.shape[0]))
        else:
            feature_dim = 768
            result[fold] = (
                torch.zeros((0, feature_dim), dtype=torch.float32),
                torch.zeros((0, int(config.negative_samples) + 1, feature_dim), dtype=torch.float32),
                0,
            )
    return result


def _save_dinov2_adapter_checkpoint(path: Path, model: Any, config: Any, *, fold: int, sample_count: int) -> None:
    torch = _require_torch()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "fold": int(fold),
            "sample_count": int(sample_count),
            "input_dim": int(getattr(model, "input_dim", 0)),
            "model_dim": int(config.model_dim),
            "encoder_hidden_dim": int(config.encoder_hidden_dim),
            "encoder_layers": int(config.encoder_layers),
            "training_epochs": int(config.training_epochs),
            "training_learning_rate": float(config.training_learning_rate),
            "negative_samples": int(config.negative_samples),
            "max_training_samples": int(config.max_training_samples),
            "parameter_count": int(getattr(model, "parameter_count", 0)),
            "dinov2_model_name": str(config.dinov2_model_name),
            "dinov2_checkpoint_path": str(config.dinov2_checkpoint_path),
            "dinov2_input_scale": int(config.dinov2_input_scale),
            "downsample_factor": int(config.downsample_factor),
            "adapter_type": "StereoTokenTransformer",
        },
        path,
    )


def _upsample_token_grid_bilinear_torch(token_values: Any, output_height: int, output_width: int, *, device: str) -> Any:
    import torch.nn.functional as F

    torch = _require_torch()
    source = _as_float_tensor(token_values, device)
    valid = (source > 0).float()
    values = F.interpolate(
        (source * valid).unsqueeze(0).unsqueeze(0),
        size=(int(output_height), int(output_width)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    weights = F.interpolate(
        valid.unsqueeze(0).unsqueeze(0),
        size=(int(output_height), int(output_width)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    return torch.where(weights > 1e-6, values / torch.clamp(weights, min=1e-6), torch.zeros_like(values)).float()


def predict_dinov2_stereo_disparity(
    left_gray: Any,
    right_gray: Any,
    *,
    downsample_factor: int,
    max_disparity: int,
    min_disparity: int,
    min_confidence: float,
    input_scale: int,
    token_median_filter_size: int,
    consistency_threshold: float,
    fill_invalid_passes: int,
    disparity_regression: str,
    softmax_temperature: float,
    model_name: str,
    repo_path: str | Path | None,
    checkpoint_path: str | Path,
    device: str,
    adapter_model: Any | None = None,
    adapter_batch_size: int = 4096,
) -> Dinov2StereoPrediction:
    torch = _require_torch()

    left_tokens, _, _, model_patch_size = extract_dinov2_patch_tokens(
        left_gray,
        downsample_factor=downsample_factor,
        input_scale=input_scale,
        model_name=model_name,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        device=device,
        return_numpy=False,
    )
    right_tokens, _, _, _ = extract_dinov2_patch_tokens(
        right_gray,
        downsample_factor=downsample_factor,
        input_scale=input_scale,
        model_name=model_name,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        device=device,
        return_numpy=False,
    )
    if left_tokens.shape[:2] != right_tokens.shape[:2]:
        raise ValueError(f"DINOv2 token grid mismatch: left={tuple(left_tokens.shape[:2])}, right={tuple(right_tokens.shape[:2])}")

    if adapter_model is not None:
        adapter_model.eval()
        left_tokens = encode_o4_descriptors_torch(
            adapter_model,
            left_tokens,
            batch_size=int(adapter_batch_size),
            device=device,
            return_numpy=False,
        )
        right_tokens = encode_o4_descriptors_torch(
            adapter_model,
            right_tokens,
            batch_size=int(adapter_batch_size),
            device=device,
            return_numpy=False,
        )

    token_height, token_width, descriptor_dim = left_tokens.shape
    token_span = max(1, int(round((float(downsample_factor) * float(model_patch_size)) / float(max(1, int(input_scale))))))
    max_token_disparity = min(
        max(int(min_disparity), int(max_disparity) // token_span),
        max(0, int(token_width) - 1),
    )
    min_token_disparity = min(int(min_disparity) // token_span, max_token_disparity)
    left_content_mask = _build_nonblack_content_mask(left_gray)
    right_content_mask = _build_nonblack_content_mask(right_gray)
    left_token_content_mask = _build_token_content_mask_torch(
        left_content_mask,
        token_span,
        tuple(left_tokens.shape[:2]),
        device=device,
    )
    right_token_content_mask = _build_token_content_mask_torch(
        right_content_mask,
        token_span,
        tuple(right_tokens.shape[:2]),
        device=device,
    )

    token_disparity, token_confidence, token_scores = _predict_token_disparity_torch(
        left_tokens,
        right_tokens,
        min_token_disparity,
        max_token_disparity,
        device=device,
        reference_valid_mask=left_token_content_mask,
        target_valid_mask=right_token_content_mask,
    )
    right_to_left_disparity, _, right_to_left_scores = _predict_token_disparity_torch(
        right_tokens,
        left_tokens,
        min_token_disparity,
        max_token_disparity,
        device=device,
        target_direction="positive",
        reference_valid_mask=right_token_content_mask,
        target_valid_mask=left_token_content_mask,
    )

    if str(disparity_regression).strip().lower() == "soft_argmax":
        refined_token_disparity = _soft_argmax_disparity_torch(
            token_scores,
            min_token_disparity,
            max_token_disparity,
            softmax_temperature,
            device=device,
        )
        right_to_left_disparity = _soft_argmax_disparity_torch(
            right_to_left_scores,
            min_token_disparity,
            max_token_disparity,
            softmax_temperature,
            device=device,
        )
    else:
        refined_token_disparity = _refine_token_disparity_torch(
            token_disparity,
            token_scores,
            min_token_disparity,
            max_token_disparity,
            device=device,
        )
        right_to_left_disparity = _refine_token_disparity_torch(
            right_to_left_disparity,
            right_to_left_scores,
            min_token_disparity,
            max_token_disparity,
            device=device,
        )
    left_cost_volume = (-token_scores).permute(1, 2, 0).contiguous().to(dtype=torch.float32)
    right_cost_volume = (-right_to_left_scores).permute(1, 2, 0).contiguous().to(dtype=torch.float32)
    left_pair_mask = _stereo_cost_valid_mask_torch(
        left_token_content_mask,
        right_token_content_mask,
        min_token_disparity,
        max_token_disparity,
        "negative",
        device=device,
    )
    right_pair_mask = _stereo_cost_valid_mask_torch(
        right_token_content_mask,
        left_token_content_mask,
        min_token_disparity,
        max_token_disparity,
        "positive",
        device=device,
    )
    left_cost_volume = left_cost_volume.masked_fill(~left_pair_mask, float("inf"))
    right_cost_volume = right_cost_volume.masked_fill(~right_pair_mask, float("inf"))
    left_guide_tokens = _build_token_guide_torch(
        left_gray,
        downsample_factor,
        input_scale,
        model_patch_size,
        tuple(left_tokens.shape[:2]),
        device=device,
    )
    right_guide_tokens = _build_token_guide_torch(
        right_gray,
        downsample_factor,
        input_scale,
        model_patch_size,
        tuple(right_tokens.shape[:2]),
        device=device,
    )
    refined_token_disparity, sgm_confidence = _solve_content_masked_sgm_from_cost_torch(
        left_cost_volume,
        left_guide_tokens,
        left_token_content_mask,
        min_disparity=int(min_token_disparity),
        device=device,
    )
    right_to_left_disparity, _ = _solve_content_masked_sgm_from_cost_torch(
        right_cost_volume,
        right_guide_tokens,
        right_token_content_mask,
        min_disparity=int(min_token_disparity),
        device=device,
    )
    token_confidence = sgm_confidence
    consistency_mask = _left_right_consistency_mask_torch(
        refined_token_disparity,
        right_to_left_disparity,
        consistency_threshold,
        device=device,
    )
    confidence_mask = token_confidence >= float(min_confidence)
    valid_token_mask = consistency_mask & confidence_mask & left_token_content_mask
    confidence_tokens = token_confidence.masked_fill(~valid_token_mask, 0.0).float()
    official_token_disparity = refined_token_disparity.masked_fill(~valid_token_mask, 0.0).float()
    display_token_disparity = _fill_invalid_disparity_torch(official_token_disparity, fill_invalid_passes, device=device)
    display_token_disparity = display_token_disparity.masked_fill(~left_token_content_mask, 0.0).float()
    if int(token_median_filter_size) > 1:
        token_mask = official_token_disparity > 0
        filtered_official = _median_filter_2d_torch(official_token_disparity, token_median_filter_size, device=device)
        official_token_disparity = filtered_official.masked_fill(~token_mask, 0.0).float()
        display_mask = display_token_disparity > 0
        filtered_display = _median_filter_2d_torch(display_token_disparity, token_median_filter_size, device=device)
        display_token_disparity = filtered_display.masked_fill(~display_mask, 0.0).float()
    official_token_disparity = official_token_disparity.masked_fill(~left_token_content_mask, 0.0).float()
    display_token_disparity = display_token_disparity.masked_fill(~left_token_content_mask, 0.0).float()

    output_height, output_width = int(left_gray.shape[0]), int(left_gray.shape[1])
    token_disparity_pixels = official_token_disparity.float() * float(token_span)
    display_token_disparity_pixels = display_token_disparity.float() * float(token_span)
    raw_disparity = _upsample_token_grid_torch(
        token_disparity_pixels,
        token_span,
        output_height,
        output_width,
        device=device,
    )
    disparity = _upsample_token_grid_bilinear_torch(
        display_token_disparity_pixels,
        output_height,
        output_width,
        device=device,
    )
    confidence = _upsample_token_grid_bilinear_torch(
        confidence_tokens,
        output_height,
        output_width,
        device=device,
    )
    raw_disparity = torch.where(torch.isfinite(raw_disparity) & (raw_disparity > 0), raw_disparity, torch.zeros_like(raw_disparity)).float()
    disparity = torch.where(torch.isfinite(disparity) & (disparity > 0), disparity, torch.zeros_like(disparity)).float()
    confidence = torch.where(torch.isfinite(confidence) & (confidence > 0), confidence, torch.zeros_like(confidence)).float()
    pixel_content_mask = torch.as_tensor(left_content_mask, dtype=torch.bool, device=device)
    raw_disparity = raw_disparity.masked_fill(~pixel_content_mask, 0.0).float()
    disparity = disparity.masked_fill(~pixel_content_mask, 0.0).float()
    confidence = confidence.masked_fill(~pixel_content_mask, 0.0).float()

    score_count = max_token_disparity - min_token_disparity + 1
    estimated_bytes = (
        (2 * int(token_height) * int(token_width) * int(descriptor_dim))
        + (score_count * int(token_height) * int(token_width))
        + (4 * int(token_height) * int(token_width))
    ) * 4
    return Dinov2StereoPrediction(
        disparity=disparity.detach().cpu().numpy().astype("float32"),
        raw_disparity=raw_disparity.detach().cpu().numpy().astype("float32"),
        confidence=confidence.detach().cpu().numpy().astype("float32"),
        token_height=int(token_height),
        token_width=int(token_width),
        descriptor_dim=int(descriptor_dim),
        token_span=int(token_span),
        model_patch_size=int(model_patch_size),
        min_token_disparity=int(min_token_disparity),
        max_token_disparity=int(max_token_disparity),
        estimated_working_set_mb=float(estimated_bytes / (1024.0 * 1024.0)),
    )


def _metric_text(value: float) -> str:
    return f"{value:.6f}" if float(value) >= 0 else "NA"


def _write_metrics(metrics_file: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scene",
        "fold",
        "token_grid",
        "valid_disparity_pixels",
        "valid_ground_truth_pixels",
        "mae",
        "rmse",
        "bad_1px",
        "mean_confidence",
        "estimated_working_set_mb",
    ]
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_fold_metrics(fold_metrics_file: Path, rows: list[dict[str, Any]], num_folds: int) -> None:
    fieldnames = ["fold", "scene_count", "mean_mae", "mean_rmse", "mean_bad_1px"]
    fold_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    by_fold: dict[int, list[dict[str, Any]]] = {index: [] for index in range(max(1, int(num_folds)))}
    for row in rows:
        by_fold.setdefault(int(row["fold"]), []).append(row)

    with fold_metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fold, fold_rows in sorted(by_fold.items()):
            numeric = []
            for row in fold_rows:
                try:
                    numeric.append((float(row["mae"]), float(row["rmse"]), float(row["bad_1px"])))
                except (TypeError, ValueError):
                    pass
            if numeric:
                mean_mae = sum(item[0] for item in numeric) / len(numeric)
                mean_rmse = sum(item[1] for item in numeric) / len(numeric)
                mean_bad = sum(item[2] for item in numeric) / len(numeric)
                writer.writerow(
                    {
                        "fold": fold,
                        "scene_count": len(fold_rows),
                        "mean_mae": f"{mean_mae:.6f}",
                        "mean_rmse": f"{mean_rmse:.6f}",
                        "mean_bad_1px": f"{mean_bad:.6f}",
                    }
                )
            else:
                writer.writerow({"fold": fold, "scene_count": len(fold_rows), "mean_mae": "NA", "mean_rmse": "NA", "mean_bad_1px": "NA"})


def refine_dinov2_output_disparity(disparity: Any, confidence: Any, guide_gray: Any, content_mask: Any | None = None) -> Any:
    import cv2
    import numpy as np
    from scipy import ndimage

    source = np.asarray(disparity, dtype=np.float32)
    guide = np.asarray(guide_gray, dtype=np.float32)
    region = np.ones(source.shape, dtype=bool) if content_mask is None else np.asarray(content_mask, dtype=bool)
    valid = np.isfinite(source) & (source > 0) & region
    if not bool(np.any(valid)):
        return np.zeros_like(source, dtype=np.float32)

    valid_values = source[valid]
    low = float(np.percentile(valid_values, 0.5))
    high = float(np.percentile(valid_values, 99.5))
    if not high > low:
        low = float(valid_values.min())
        high = float(valid_values.max())
    clipped = np.where(valid, np.clip(source, low, high), 0.0).astype(np.float32)

    nearest_indices = np.asarray(ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True), dtype=np.int64)
    dense = clipped[nearest_indices[0], nearest_indices[1]].astype(np.float32)
    dense = np.where(np.isfinite(dense) & (dense > 0), dense, 0.0).astype(np.float32)
    dense = np.where(region, dense, 0.0).astype(np.float32)

    guide_range = max(float(guide.max() - guide.min()), 1.0)
    guide_unit = ((guide - float(guide.min())) / guide_range).astype(np.float32)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        smoothed = cv2.ximgproc.guidedFilter(guide=guide_unit, src=dense, radius=11, eps=2e-3).astype(np.float32)
    else:
        smoothed = cv2.bilateralFilter(dense, d=13, sigmaColor=8.0, sigmaSpace=15.0).astype(np.float32)

    conf = np.asarray(confidence, dtype=np.float32)
    conf_valid = np.isfinite(conf) & (conf > 0)
    if bool(np.any(conf_valid)):
        scale = max(float(np.percentile(conf[conf_valid], 95.0)), 1e-6)
        keep_weight = np.clip(conf / scale, 0.0, 1.0).astype(np.float32)
    else:
        keep_weight = np.zeros_like(source, dtype=np.float32)
    keep_weight = np.where(valid & region, keep_weight, 0.0).astype(np.float32)
    refined = (keep_weight * clipped) + ((1.0 - keep_weight) * smoothed)
    refined = np.where(region & np.isfinite(refined) & (refined > 0), refined, 0.0).astype(np.float32)
    return refined


def filter_dinov2_reliable_disparity(disparity: Any, confidence: Any, min_confidence: float, confidence_percentile: float) -> tuple[Any, float, int]:
    import numpy as np

    source = np.asarray(disparity, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    valid = np.isfinite(source) & (source > 0) & np.isfinite(conf) & (conf > 0)
    if not bool(np.any(valid)):
        return np.zeros_like(source, dtype=np.float32), float(min_confidence), 0

    positive_confidence = conf[valid]
    percentile_floor = float(np.percentile(positive_confidence, float(confidence_percentile)))
    confidence_floor = max(float(min_confidence), percentile_floor)
    reliable = valid & (conf >= confidence_floor)
    return np.where(reliable, source, 0.0).astype(np.float32), confidence_floor, int(np.count_nonzero(reliable))


def _resolve_run_device(config: Any) -> tuple[str, str]:
    torch = _require_torch()

    def select_cuda(device_name: str, reason: str) -> tuple[str, str]:
        requested_cuda_index = cuda_device_index(device_name)
        memory_device = 0 if requested_cuda_index is None else requested_cuda_index
        try:
            torch.cuda.set_per_process_memory_fraction(0.19, memory_device)
        except Exception:
            pass
        return device_name, reason

    requested = str(getattr(config, "device", "auto")).strip().lower() or "auto"
    prefer_cuda = bool(getattr(config, "prefer_cuda", True))
    if is_cuda_device_name(requested) and torch.cuda.is_available():
        requested_cuda_index = cuda_device_index(requested)
        visible_cuda_count = int(torch.cuda.device_count())
        if requested_cuda_index is not None and requested_cuda_index >= visible_cuda_count:
            return "cpu", f"{requested} requested but only {visible_cuda_count} CUDA device(s) are visible; using CPU"
        return select_cuda(requested, f"{requested} requested and available; memory fraction capped at 0.19")
    if is_cuda_device_name(requested):
        return "cpu", f"{requested} requested but CUDA is unavailable; using CPU"
    if requested == "cpu":
        return "cpu", "CPU requested"
    if prefer_cuda and torch.cuda.is_available():
        return select_cuda("cuda", "torch CUDA is available; memory fraction capped at 0.19")
    return "cpu", "using CPU because CUDA is unavailable or not preferred"


def validate_dinov2_results(disparity_dir: Path, analysis_dir: Path, metrics_file: Path, scene_name: str | None = None) -> int:
    print(f"Validating O4 DINOv2 disparity directory: {disparity_dir}")
    print(f"Validating O4 DINOv2 analysis directory: {analysis_dir}")
    print(f"Validating O4 DINOv2 metrics file: {metrics_file}")
    if not metrics_file.exists():
        print(f"Missing metrics file: {metrics_file}", file=sys.stderr)
        return 1

    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("scene")]
    if scene_name is not None:
        rows = [row for row in rows if row.get("scene") == scene_name]
    if not rows:
        print("No O4 DINOv2 metric rows found for validation.", file=sys.stderr)
        return 1

    missing_any = False
    for row in rows:
        current_scene = row["scene"]
        required_paths = (
            disparity_dir / current_scene / "im0.png",
            disparity_dir / current_scene / "im1.png",
            disparity_dir / current_scene / "disp0.pfm",
            disparity_dir / current_scene / "disp0.png",
            disparity_dir / current_scene / "disp0_transformer_raw.pfm",
            disparity_dir / current_scene / "disp0_transformer_raw.png",
            analysis_dir / current_scene / "im0.png",
            analysis_dir / current_scene / "im1.png",
            analysis_dir / current_scene / "confidence.png",
            analysis_dir / current_scene / "error_map.png",
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            missing_any = True
            print(f"[MISSING] {current_scene}: " + ", ".join(missing))
        else:
            print(f"[OK] {current_scene}: required O4 DINOv2 files present")
    if missing_any:
        print("Validation status: issues found")
        return 1
    print("Validation status: all checked O4 DINOv2 scene folders contain the expected files")
    return 0


def run_dinov2_objective(
    repo_root: Path,
    middlebury_root: Path,
    config: Any,
    max_scenes: int | None,
    dry_run: bool,
    scene_name: str | None,
) -> int:
    discovered_scenes = discover_scenes(middlebury_root)
    scenes = filter_scene_dirs(discovered_scenes, scene_name)
    if max_scenes is not None:
        if max_scenes < 0:
            print("--max-scenes must be zero or greater.", file=sys.stderr)
            return 2
        scenes = scenes[:max_scenes]

    try:
        device, device_reason = _resolve_run_device(config)
    except Exception as exc:
        print(f"O4 DINOv2 device resolution failed: {exc}", file=sys.stderr)
        return 1
    status = resolve_o4_execution_mode(
        "dinov2_cost_volume",
        use_torch=True,
        dinov2_model_name=config.dinov2_model_name,
        dinov2_repo_path=config.dinov2_repo_path,
        dinov2_checkpoint_path=config.dinov2_checkpoint_path,
    )

    print(f"Repository root: {repo_root}")
    print(f"Middlebury root: {middlebury_root}")
    print(f"Discovered scenes with im0.png/im1.png: {len(discovered_scenes)}")
    print(f"O4 DINOv2 disparity output dir: {config.disparity_dir}")
    print(f"O4 DINOv2 analysis output dir: {config.analysis_dir}")
    print(f"O4 DINOv2 metrics file: {config.metrics_file}")
    print(f"O4 DINOv2 fold metrics file: {config.fold_metrics_file}")
    print(
        "O4 DINOv2 model config: "
        f"downsample_factor={config.downsample_factor}, max_disparity={config.max_disparity}, "
        f"dinov2_input_scale={config.dinov2_input_scale}, device={device}, descriptor_source={status.descriptor_source}, "
        f"model={config.dinov2_model_name}, checkpoint={config.dinov2_checkpoint_path}, "
        f"disparity_regression={config.disparity_regression}"
    )
    print(f"O4 DINOv2 device status: {device_reason}")
    print(f"O4 DINOv2 execution status: {status.reason}")
    if scene_name is not None:
        print(f"Scene filter: {scene_name}")
    print("Scenes to process: " + (", ".join(scene.name for scene in scenes) if scenes else "none"))

    if dry_run or max_scenes == 0:
        print("Dry run requested; no outputs were written.")
        return 0
    if not middlebury_root.exists():
        print(f"Middlebury root not found: {middlebury_root}", file=sys.stderr)
        return 1
    if not scenes:
        print(f"No scene directories containing im0.png and im1.png were found under {middlebury_root}.", file=sys.stderr)
        return 1
    if not status.available:
        print(f"O4 DINOv2 execution mode unavailable: {status.reason}", file=sys.stderr)
        return 1

    config.disparity_dir.mkdir(parents=True, exist_ok=True)
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    scene_fold_map = {scene.name: index % config.num_folds for index, scene in enumerate(discovered_scenes)}
    adapter_models: dict[int, Any] = {}
    adapter_paths: dict[int, Path] = {}
    adapter_sample_counts: dict[int, int] = {}
    if int(config.training_epochs) > 0:
        adapter_dir = config.metrics_file.parent / "dinov2_adapters"
        print(
            "Training DINOv2 stereo adapters: "
            f"folds={config.num_folds}, epochs={config.training_epochs}, "
            f"samples_per_fold<={config.max_training_samples}, save_dir={adapter_dir}",
            flush=True,
        )
        samples_by_fold = _collect_dinov2_training_samples_by_fold(discovered_scenes, scene_fold_map, config, device=device)
        for fold in range(int(config.num_folds)):
            training_descriptors, candidate_descriptors, sample_count = samples_by_fold[fold]
            adapter_sample_counts[fold] = int(sample_count)
            if sample_count <= 0:
                print(f"Skipping DINOv2 adapter fold {fold}: no training samples", flush=True)
                continue
            adapter = train_o4_torch_model(
                training_descriptors,
                candidate_descriptors,
                model_dim=config.model_dim,
                hidden_dim=config.encoder_hidden_dim,
                encoder_layers=config.encoder_layers,
                epochs=config.training_epochs,
                learning_rate=config.training_learning_rate,
                weight_decay=config.weight_decay,
                batch_size=config.training_batch_size,
                random_seed=int(config.random_seed) + fold,
                device=device,
            )
            if adapter is None:
                print(f"Skipping DINOv2 adapter fold {fold}: training returned no model", flush=True)
                continue
            adapter.eval()
            adapter_models[fold] = adapter
            adapter_path = adapter_dir / f"dinov2_adapter_fold{fold}.pt"
            _save_dinov2_adapter_checkpoint(adapter_path, adapter, config, fold=fold, sample_count=sample_count)
            adapter_paths[fold] = adapter_path
            print(
                f"Saved DINOv2 adapter fold {fold}: {adapter_path} "
                f"(samples={sample_count}, parameters={int(getattr(adapter, 'parameter_count', 0))})",
                flush=True,
            )
    metric_rows: list[dict[str, Any]] = []
    generator_name = "O4 DINOv2 trained stereo-adapter patch-token disparity" if adapter_models else "O4 DINOv2 direct model patch-token stereo disparity"
    for scene_dir in scenes:
        left_gray = load_gray(scene_dir / "im0.png")
        right_gray = load_gray(scene_dir / "im1.png")
        left_content_mask = _build_nonblack_content_mask(left_gray)
        fold = scene_fold_map[scene_dir.name]
        adapter_model = adapter_models.get(fold)
        prediction = predict_dinov2_stereo_disparity(
            left_gray,
            right_gray,
            downsample_factor=config.downsample_factor,
            max_disparity=config.max_disparity,
            min_disparity=config.min_disparity,
            min_confidence=config.min_confidence,
            input_scale=config.dinov2_input_scale,
            token_median_filter_size=config.token_median_filter_size,
            consistency_threshold=config.consistency_threshold,
            fill_invalid_passes=config.fill_invalid_passes,
            disparity_regression=config.disparity_regression,
            softmax_temperature=config.softmax_temperature,
            model_name=config.dinov2_model_name,
            repo_path=config.dinov2_repo_path,
            checkpoint_path=config.dinov2_checkpoint_path,
            device=device,
            adapter_model=adapter_model,
            adapter_batch_size=config.inference_batch_size,
        )
        display_disparity = refine_dinov2_output_disparity(prediction.disparity, prediction.confidence, left_gray, left_content_mask)
        disparity, reliable_confidence_floor, reliable_pixel_count = filter_dinov2_reliable_disparity(
            display_disparity,
            prediction.confidence,
            config.min_confidence,
            55.0,
        )
        disparity = _apply_content_mask(disparity, left_content_mask)
        raw_disparity = _apply_content_mask(prediction.raw_disparity, left_content_mask)
        confidence = np.where(np.asarray(left_content_mask, dtype=bool), prediction.confidence, 0.0).astype(np.float32)
        disparity_scene_dir = config.disparity_dir / scene_dir.name
        analysis_scene_dir = config.analysis_dir / scene_dir.name
        disparity_scene_dir.mkdir(parents=True, exist_ok=True)
        analysis_scene_dir.mkdir(parents=True, exist_ok=True)
        copy_o4_source_images(scene_dir, disparity_scene_dir, analysis_scene_dir)

        write_pfm(disparity_scene_dir / "disp0.pfm", disparity)
        write_pfm(disparity_scene_dir / "disp0_transformer_raw.pfm", raw_disparity)
        disparity_preview = colorize_disparity_depth_map(display_disparity, display_disparity > 0)
        raw_disparity_preview = colorize_disparity_depth_map(raw_disparity, raw_disparity > 0)
        write_png(disparity_scene_dir / "disp0.png", disparity_preview)
        write_png(disparity_scene_dir / "disp0_transformer_raw.png", raw_disparity_preview)

        confidence_mask = confidence > 0
        confidence_preview = normalize_for_preview(confidence, confidence_mask)
        if bool(confidence_mask.any()) and int(confidence_preview.max()) == 0:
            confidence_preview[confidence_mask] = 255
        write_png(analysis_scene_dir / "confidence.png", confidence_preview)

        metrics = {"valid_disparity_pixels": int((disparity > 0).sum()), "valid_ground_truth_pixels": 0, "mae": -1.0, "rmse": -1.0, "bad_1px": -1.0}
        error_map = disparity * 0.0
        ground_truth_path = scene_dir / "disp0.pfm"
        if ground_truth_path.exists():
            ground_truth = read_pfm(ground_truth_path)
            if getattr(ground_truth, "ndim", 0) == 3:
                ground_truth = ground_truth[:, :, 0]
            metrics = evaluate_disparity(disparity, ground_truth)
            valid_mask = (disparity > 0) & (ground_truth > 0)
            error_map[valid_mask] = abs(disparity[valid_mask] - ground_truth[valid_mask])
        error_preview = normalize_for_preview(error_map, error_map > 0)
        error_color = error_preview[:, :, None].repeat(3, axis=2)
        error_color[error_map <= 0] = 0
        write_png(analysis_scene_dir / "error_map.png", error_color)

        reliable_confidence_mask = (disparity > 0) & (confidence > 0)
        mean_confidence = float(confidence[reliable_confidence_mask].mean()) if bool(reliable_confidence_mask.any()) else 0.0
        readme_lines = [
            f"scene: {scene_dir.name}",
            f"generator: {generator_name}",
            f"fold: {fold}",
            f"token_grid: {prediction.token_height}x{prediction.token_width}",
            f"descriptor_dim: {prediction.descriptor_dim}",
            f"token_span_pixels: {prediction.token_span}",
            f"dinov2_input_scale: {config.dinov2_input_scale}",
            f"model_patch_size: {prediction.model_patch_size}",
            f"min_token_disparity: {prediction.min_token_disparity}",
            f"max_token_disparity: {prediction.max_token_disparity}",
            f"resolved_execution_mode: dinov2_cost_volume",
            f"descriptor_source: {status.descriptor_source}",
            "im0.png: original left image copied from the source scene",
            "im1.png: original right image copied from the source scene",
            "final_disparity_source: DINOv2 patch-token cost volume with O3-style SGM aggregation",
            f"training_used: {'yes' if adapter_model is not None else 'no'}",
            f"dinov2_adapter_checkpoint: {adapter_paths.get(fold, 'none')}",
            f"dinov2_adapter_training_samples: {adapter_sample_counts.get(fold, 0)}",
            f"dinov2_adapter_parameters: {int(getattr(adapter_model, 'parameter_count', 0)) if adapter_model is not None else 0}",
            "sgm_detail_fusion_used: yes",
            "raw_transformer_disparity: disp0_transformer_raw.pfm",
            "final_upsampling: weighted_bilinear_from_dinov2_sgm_patch_tokens",
            "final_postprocess: confidence_weighted_guided_smoothing_from_dinov2_prediction",
            f"official_pfm_confidence_floor: {reliable_confidence_floor:.6f}",
            f"official_pfm_reliable_pixels: {reliable_pixel_count}",
            f"disparity_regression: {config.disparity_regression}",
            f"runtime_device: {device}",
            f"estimated_working_set_mb: {prediction.estimated_working_set_mb:.3f}",
        ]
        write_scene_text(disparity_scene_dir / "README.txt", readme_lines)
        write_scene_text(
            analysis_scene_dir / "README.txt",
            readme_lines
            + [
                f"mean_confidence: {mean_confidence:.6f}",
                f"ground_truth_available: {'yes' if ground_truth_path.exists() else 'no'}",
                f"mae: {metrics['mae'] if metrics['mae'] >= 0 else 'NA'}",
                f"rmse: {metrics['rmse'] if metrics['rmse'] >= 0 else 'NA'}",
                f"bad_1px: {metrics['bad_1px'] if metrics['bad_1px'] >= 0 else 'NA'}",
            ],
        )

        metric_rows.append(
            {
                "scene": scene_dir.name,
                "fold": fold,
                "token_grid": f"{prediction.token_height}x{prediction.token_width}",
                "valid_disparity_pixels": metrics["valid_disparity_pixels"],
                "valid_ground_truth_pixels": metrics["valid_ground_truth_pixels"],
                "mae": _metric_text(metrics["mae"]),
                "rmse": _metric_text(metrics["rmse"]),
                "bad_1px": _metric_text(metrics["bad_1px"]),
                "mean_confidence": f"{mean_confidence:.6f}",
                "estimated_working_set_mb": f"{prediction.estimated_working_set_mb:.6f}",
            }
        )
        print(
            f"Wrote O4 DINOv2 scene: {scene_dir.name} "
            f"(fold={fold}, token_grid={prediction.token_height}x{prediction.token_width}, "
            f"mae={_metric_text(metrics['mae'])}, rmse={_metric_text(metrics['rmse'])}, bad_1px={_metric_text(metrics['bad_1px'])})"
        )

    _write_metrics(config.metrics_file, metric_rows)
    _write_fold_metrics(config.fold_metrics_file, metric_rows, config.num_folds)
    print(f"Wrote O4 DINOv2 metrics summary: {config.metrics_file}")
    print(f"Wrote O4 DINOv2 fold summary: {config.fold_metrics_file}")
    return 0


def _repo_root_from(module_file: str) -> Path:
    return Path(module_file).resolve().parents[1]


def _build_parser(module_file: str) -> argparse.ArgumentParser:
    repo_root = _repo_root_from(module_file)
    default_config = repo_root / "configs" / "dataset_paths.remote_dinov2.example.yaml"
    parser = argparse.ArgumentParser(description="Run O4 DINOv2 direct model stereo prediction.")
    parser.add_argument("--config", type=Path, default=default_config, help=f"Path to YAML config. Default: {default_config}")
    parser.add_argument("--profile", default="remote", help="Config profile name. Default: remote")
    parser.add_argument("--max-scenes", type=int, default=None, help="Maximum number of scenes to process. Use 0 for discovery-only.")
    parser.add_argument("--scene-name", default=None, help="Run only one scene by exact directory name.")
    parser.add_argument("--dry-run", action="store_true", help="Print discovery/config info without writing outputs.")
    parser.add_argument("--validate-results", action="store_true", help="Validate existing DINOv2 outputs without rewriting them.")
    parser.add_argument("--o4-dinov2-model", default=None, help="Override DINOv2 model selector.")
    parser.add_argument("--o4-dinov2-checkpoint", type=Path, default=None, help="Override local DINOv2 checkpoint path.")
    parser.add_argument("--o4-dinov2-repo", type=Path, default=None, help="Override local DINOv2 repo path.")
    parser.add_argument("--o4-regression-mode", choices=("quadratic", "soft_argmax"), default=None, help="Override disparity regression mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from dataclasses import replace

    from config import load_config

    parser = _build_parser(__file__)
    args = parser.parse_args(argv)
    repo_root = _repo_root_from(__file__)
    try:
        loaded = load_config(args.config, args.profile)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    dinov2_model_name = args.o4_dinov2_model or loaded.o4.dinov2_model_name
    dinov2_checkpoint_path = args.o4_dinov2_checkpoint or loaded.o4.dinov2_checkpoint_path
    config = replace(
        loaded.o4,
        execution_mode="dinov2_cost_volume",
        dinov2_model_name=dinov2_model_name,
        dinov2_checkpoint_path=dinov2_checkpoint_path,
        dinov2_repo_path=args.o4_dinov2_repo if args.o4_dinov2_repo is not None else loaded.o4.dinov2_repo_path,
        disparity_regression=args.o4_regression_mode or loaded.o4.disparity_regression,
    )
    if args.validate_results:
        return validate_dinov2_results(config.disparity_dir, config.analysis_dir, config.metrics_file, args.scene_name)
    return run_dinov2_objective(repo_root, loaded.middlebury_root, config, args.max_scenes, args.dry_run, args.scene_name)


if __name__ == "__main__":
    raise SystemExit(main())
