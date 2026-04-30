from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any


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


def extract_dinov2_descriptors(*args: Any, **kwargs: Any) -> tuple[Any, int, int, int]:
    return extract_dinov2_patch_tokens(*args, **kwargs)


def _select_dinov2_tile_shape(token_height: int, token_width: int) -> tuple[int, int]:
    token_limit = max(1, int(_DINO_V2_TILE_TOKEN_LIMIT))
    if int(token_height) * int(token_width) <= token_limit:
        return int(token_height), int(token_width)
    if token_height <= token_width and token_height <= token_limit:
        tile_height = int(token_height)
        tile_width = max(1, min(int(token_width), token_limit // tile_height))
        return tile_height, tile_width
    if token_width <= token_limit:
        tile_width = int(token_width)
        tile_height = max(1, min(int(token_height), token_limit // tile_width))
        return tile_height, tile_width
    side = max(1, int(token_limit ** 0.5))
    tile_height = min(int(token_height), side)
    tile_width = max(1, min(int(token_width), token_limit // tile_height))
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
) -> tuple[Any, Any, Any]:
    torch = _require_torch()

    left = _as_float_tensor(left_tokens, device)
    right = _as_float_tensor(right_tokens, device)
    token_height, token_width, _ = left.shape
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
        elif target_direction == "positive" and disparity < token_width:
            current_scores[:, : token_width - disparity] = (
                left[:, : token_width - disparity, :] * right[:, disparity:, :]
            ).sum(dim=2)
        elif disparity < token_width:
            current_scores[:, disparity:] = (
                left[:, disparity:, :] * right[:, : token_width - disparity, :]
            ).sum(dim=2)

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
    token_median_filter_size: int,
    consistency_threshold: float,
    fill_invalid_passes: int,
    disparity_regression: str,
    softmax_temperature: float,
    model_name: str,
    repo_path: str | Path | None,
    checkpoint_path: str | Path,
    device: str,
) -> Dinov2StereoPrediction:
    torch = _require_torch()

    left_tokens, _, _, model_patch_size = extract_dinov2_patch_tokens(
        left_gray,
        downsample_factor=downsample_factor,
        model_name=model_name,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        device=device,
        return_numpy=False,
    )
    right_tokens, _, _, _ = extract_dinov2_patch_tokens(
        right_gray,
        downsample_factor=downsample_factor,
        model_name=model_name,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        device=device,
        return_numpy=False,
    )
    if left_tokens.shape[:2] != right_tokens.shape[:2]:
        raise ValueError(f"DINOv2 token grid mismatch: left={tuple(left_tokens.shape[:2])}, right={tuple(right_tokens.shape[:2])}")

    token_height, token_width, descriptor_dim = left_tokens.shape
    token_span = max(1, int(downsample_factor) * int(model_patch_size))
    max_token_disparity = min(
        max(int(min_disparity), int(max_disparity) // token_span),
        max(0, int(token_width) - 1),
    )
    min_token_disparity = min(int(min_disparity) // token_span, max_token_disparity)

    token_disparity, token_confidence, token_scores = _predict_token_disparity_torch(
        left_tokens,
        right_tokens,
        min_token_disparity,
        max_token_disparity,
        device=device,
    )
    right_to_left_disparity, _, _ = _predict_token_disparity_torch(
        right_tokens,
        left_tokens,
        min_token_disparity,
        max_token_disparity,
        device=device,
        target_direction="positive",
    )

    if str(disparity_regression).strip().lower() == "soft_argmax":
        refined_token_disparity = _soft_argmax_disparity_torch(
            token_scores,
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
    consistency_mask = _left_right_consistency_mask_torch(
        refined_token_disparity,
        right_to_left_disparity,
        consistency_threshold,
        device=device,
    )
    confidence_tokens = token_confidence.masked_fill(token_confidence < float(min_confidence), 0.0).float()
    official_token_disparity = refined_token_disparity.masked_fill(~consistency_mask, 0.0).float()
    display_token_disparity = _fill_invalid_disparity_torch(official_token_disparity, fill_invalid_passes, device=device)
    if int(token_median_filter_size) > 1:
        token_mask = official_token_disparity > 0
        filtered_official = _median_filter_2d_torch(official_token_disparity, token_median_filter_size, device=device)
        official_token_disparity = filtered_official.masked_fill(~token_mask, 0.0).float()
        display_mask = display_token_disparity > 0
        filtered_display = _median_filter_2d_torch(display_token_disparity, token_median_filter_size, device=device)
        display_token_disparity = filtered_display.masked_fill(~display_mask, 0.0).float()

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


def refine_dinov2_output_disparity(disparity: Any, confidence: Any, guide_gray: Any) -> Any:
    import cv2
    import numpy as np
    from scipy import ndimage

    source = np.asarray(disparity, dtype=np.float32)
    guide = np.asarray(guide_gray, dtype=np.float32)
    valid = np.isfinite(source) & (source > 0)
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
    keep_weight = np.where(valid, keep_weight, 0.0).astype(np.float32)
    refined = (keep_weight * clipped) + ((1.0 - keep_weight) * smoothed)
    refined = np.where(np.isfinite(refined) & (refined > 0), refined, 0.0).astype(np.float32)
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

    def select_cuda(reason: str) -> tuple[str, str]:
        try:
            torch.cuda.set_per_process_memory_fraction(0.19, 0)
        except Exception:
            pass
        return "cuda", reason

    requested = str(getattr(config, "device", "auto")).strip().lower() or "auto"
    prefer_cuda = bool(getattr(config, "prefer_cuda", True))
    if requested == "cuda" and torch.cuda.is_available():
        return select_cuda("torch CUDA requested and available; memory fraction capped at 0.19")
    if requested == "cuda":
        return "cpu", "torch CUDA requested but unavailable; using CPU"
    if requested == "cpu":
        return "cpu", "CPU requested"
    if prefer_cuda and torch.cuda.is_available():
        return select_cuda("torch CUDA is available; memory fraction capped at 0.19")
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
            disparity_dir / current_scene / "disp0.pfm",
            disparity_dir / current_scene / "disp0.png",
            disparity_dir / current_scene / "disp0_transformer_raw.pfm",
            disparity_dir / current_scene / "disp0_transformer_raw.png",
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
    from common import discover_scenes, evaluate_disparity, filter_scene_dirs, load_gray, normalize_for_preview, write_png, write_scene_text
    from pfm import read_pfm, write_pfm

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
        f"device={device}, descriptor_source={status.descriptor_source}, "
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
    metric_rows: list[dict[str, Any]] = []
    generator_name = "O4 DINOv2 direct model patch-token stereo disparity"
    for scene_dir in scenes:
        left_gray = load_gray(scene_dir / "im0.png")
        right_gray = load_gray(scene_dir / "im1.png")
        prediction = predict_dinov2_stereo_disparity(
            left_gray,
            right_gray,
            downsample_factor=config.downsample_factor,
            max_disparity=config.max_disparity,
            min_disparity=config.min_disparity,
            min_confidence=config.min_confidence,
            token_median_filter_size=config.token_median_filter_size,
            consistency_threshold=config.consistency_threshold,
            fill_invalid_passes=config.fill_invalid_passes,
            disparity_regression=config.disparity_regression,
            softmax_temperature=config.softmax_temperature,
            model_name=config.dinov2_model_name,
            repo_path=config.dinov2_repo_path,
            checkpoint_path=config.dinov2_checkpoint_path,
            device=device,
        )
        display_disparity = refine_dinov2_output_disparity(prediction.disparity, prediction.confidence, left_gray)
        disparity, reliable_confidence_floor, reliable_pixel_count = filter_dinov2_reliable_disparity(
            display_disparity,
            prediction.confidence,
            config.min_confidence,
            55.0,
        )
        raw_disparity = prediction.raw_disparity
        confidence = prediction.confidence
        disparity_scene_dir = config.disparity_dir / scene_dir.name
        analysis_scene_dir = config.analysis_dir / scene_dir.name
        disparity_scene_dir.mkdir(parents=True, exist_ok=True)
        analysis_scene_dir.mkdir(parents=True, exist_ok=True)

        write_pfm(disparity_scene_dir / "disp0.pfm", disparity)
        write_pfm(disparity_scene_dir / "disp0_transformer_raw.pfm", raw_disparity)
        disparity_preview = normalize_for_preview(display_disparity, display_disparity > 0)
        raw_disparity_preview = normalize_for_preview(raw_disparity, raw_disparity > 0)
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

        fold = scene_fold_map[scene_dir.name]
        reliable_confidence_mask = (disparity > 0) & (confidence > 0)
        mean_confidence = float(confidence[reliable_confidence_mask].mean()) if bool(reliable_confidence_mask.any()) else 0.0
        readme_lines = [
            f"scene: {scene_dir.name}",
            f"generator: {generator_name}",
            f"fold: {fold}",
            f"token_grid: {prediction.token_height}x{prediction.token_width}",
            f"descriptor_dim: {prediction.descriptor_dim}",
            f"token_span_pixels: {prediction.token_span}",
            f"model_patch_size: {prediction.model_patch_size}",
            f"min_token_disparity: {prediction.min_token_disparity}",
            f"max_token_disparity: {prediction.max_token_disparity}",
            f"resolved_execution_mode: dinov2_cost_volume",
            f"descriptor_source: {status.descriptor_source}",
            "final_disparity_source: DINOv2 model forward patch-token prediction",
            "training_used: no",
            "sgm_detail_fusion_used: no",
            "raw_transformer_disparity: disp0_transformer_raw.pfm",
            "final_upsampling: weighted_bilinear_from_dinov2_patch_tokens",
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
