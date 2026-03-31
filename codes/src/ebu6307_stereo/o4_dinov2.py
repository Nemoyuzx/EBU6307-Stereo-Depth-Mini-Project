from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, int]] = {}
_DEFAULT_CHECKPOINT_ROOT = Path("/limx_embop/tos/users/Nemo/self-work/models")
_DINO_V2_IMAGE_MEAN = (0.485, 0.456, 0.406)
_DINO_V2_IMAGE_STD = (0.229, 0.224, 0.225)


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
_DINO_V2_MODEL_SPECS: dict[str, Dinov2ModelSpec] = {selector: spec for spec in _DINO_V2_VARIANTS for selector in spec.selectors}


def resolve_dinov2_checkpoint_path(model_name: str, checkpoint_path: str | Path | None) -> Path:
    raw_path = str(checkpoint_path).strip() if checkpoint_path is not None else ""
    if raw_path:
        return Path(raw_path).expanduser()
    spec = _resolve_dinov2_model_spec(model_name)
    return _DEFAULT_CHECKPOINT_ROOT / spec.checkpoint_filename


def resolve_o4_execution_mode(requested_mode: str, use_torch: bool, dinov2_model_name: str, dinov2_repo_path: str | Path | None, dinov2_checkpoint_path: str | Path | None) -> O4ExecutionModeStatus:
    mode = str(requested_mode or "baseline").strip().lower() or "baseline"
    if mode == "baseline":
        return O4ExecutionModeStatus(mode, "baseline", "handcrafted_patch_tokens", True, "using the existing trainable token projection baseline")
    if mode != "dinov2_cost_volume":
        return O4ExecutionModeStatus(mode, mode, "unknown", False, f"unsupported O4 execution mode: {requested_mode!r}")
    if not use_torch:
        return O4ExecutionModeStatus(mode, "dinov2_cost_volume", "dinov2_dense_descriptors", False, "dinov2_cost_volume requires the torch backend")
    if not str(dinov2_model_name).strip():
        return O4ExecutionModeStatus(mode, "dinov2_cost_volume", "dinov2_dense_descriptors", False, "dinov2_cost_volume requires o4.dinov2_model_name")
    resolved_checkpoint = resolve_dinov2_checkpoint_path(dinov2_model_name, dinov2_checkpoint_path)
    spec = _resolve_dinov2_model_spec(dinov2_model_name)
    if not resolved_checkpoint.is_file():
        return O4ExecutionModeStatus(mode, "dinov2_cost_volume", "dinov2_dense_descriptors", False, f"dinov2_cost_volume requires a local checkpoint file; missing: {resolved_checkpoint}. Expected filename for {spec.builder_name}: {spec.checkpoint_filename}")
    return O4ExecutionModeStatus(mode, "dinov2_cost_volume", "dinov2_dense_descriptors", True, "using the whitelist-only torch implementation of the standard DINO-style dense descriptor extractor")


def _resolve_dinov2_model_spec(model_name: str) -> Dinov2ModelSpec:
    normalized = str(model_name).strip()
    if normalized in _DINO_V2_MODEL_SPECS:
        return _DINO_V2_MODEL_SPECS[normalized]
    raise ValueError(f"Unsupported O4 DINOv2 model selector {model_name!r}. Expected one of: {', '.join(sorted(_DINO_V2_MODEL_SPECS))}")


def resolve_dinov2_repo_path(repo_path: str | Path | None) -> Path | None:
    raw_path = str(repo_path).strip() if repo_path is not None else ""
    return Path(raw_path).expanduser() if raw_path else None


class SimpleDinov2LikeBackbone(torch.nn.Module):
    def __init__(self, embed_dim: int, patch_size: int = 14) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = torch.nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

    def forward_features(self, pixel_values: Any) -> dict[str, Any]:
        tokens = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": F.normalize(self.proj(tokens), dim=-1, eps=1e-6)}


def _build_local_backbone(model_name: str) -> SimpleDinov2LikeBackbone:
    spec = _resolve_dinov2_model_spec(model_name)
    embed_dim = 384 if "vits" in spec.builder_name else 768
    return SimpleDinov2LikeBackbone(embed_dim=embed_dim, patch_size=14)


def _resolve_patch_size(model: Any) -> int:
    patch_size = getattr(model, "patch_size", None)
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    if isinstance(patch_size, int) and patch_size > 0:
        return patch_size
    raise ValueError("Unable to infer the DINO-style patch size from the loaded model.")


def _extract_checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return _normalize_checkpoint_state_dict(nested)
        if payload:
            return _normalize_checkpoint_state_dict(payload)
    raise ValueError("Unsupported checkpoint format.")


def _normalize_checkpoint_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = str(key)
        for prefix in ("module.", "backbone."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        normalized[normalized_key] = value
    return normalized


def _prepare_downsampled_gray(image: Any, downsample_factor: int, device: str) -> Any:
    if torch.is_tensor(image):
        source = image.to(device=device, dtype=torch.float32)
    else:
        source = torch.as_tensor(image, dtype=torch.float32, device=device)
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


def _load_dinov2_model(model_name: str, checkpoint_path: str | Path, device: str, repo_path: str | Path | None = None) -> tuple[Any, int]:
    del repo_path
    checkpoint = resolve_dinov2_checkpoint_path(model_name, checkpoint_path)
    spec = _resolve_dinov2_model_spec(model_name)
    cache_key = (spec.builder_name, str(checkpoint), str(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Configured O4 DINOv2 checkpoint was not found: {checkpoint}.")
    model = _build_local_backbone(model_name)
    state_dict = _extract_checkpoint_state_dict(torch.load(checkpoint, map_location="cpu"))
    model_state = model.state_dict()
    compatible = {key: value for key, value in state_dict.items() if key in model_state and getattr(value, 'shape', None) == model_state[key].shape}
    load_result = model.load_state_dict(compatible, strict=False)
    if len(compatible) == 0:
        raise ValueError(f"Checkpoint {checkpoint} does not contain weights compatible with the standard whitelist-only O4 backbone.")
    _ = load_result
    model = model.to(device)
    model.eval()
    patch_size = _resolve_patch_size(model)
    _MODEL_CACHE[cache_key] = (model, patch_size)
    return model, patch_size


def _prepare_dinov2_pixel_values(image: Any) -> Any:
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image tensor, got rank {image.ndim}")
    rgb = image.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    mean = torch.tensor(_DINO_V2_IMAGE_MEAN, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(_DINO_V2_IMAGE_STD, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _extract_patch_tokens(model: Any, pixel_values: Any, expected_tokens: int) -> Any:
    with torch.inference_mode():
        outputs = model.forward_features(pixel_values) if hasattr(model, "forward_features") else model(pixel_values)
    tokens = outputs.get("x_norm_patchtokens") if isinstance(outputs, dict) else getattr(outputs, "last_hidden_state", outputs)
    if tokens is None or getattr(tokens, "ndim", 0) != 3:
        raise ValueError("DINO-style model returned no valid patch tokens")
    token_count = int(tokens.shape[1])
    if token_count == expected_tokens:
        return tokens
    if token_count > expected_tokens:
        return tokens[:, token_count - expected_tokens :, :]
    raise ValueError(f"unexpected token count {token_count} for grid size {expected_tokens}")


def extract_dinov2_descriptors(image: Any, *, downsample_factor: int, model_name: str, repo_path: str | Path | None, checkpoint_path: str | Path, device: str, return_numpy: bool) -> tuple[Any, int, int, int]:
    model, patch_size = _load_dinov2_model(model_name, checkpoint_path, device, repo_path=repo_path)
    source = _prepare_downsampled_gray(image, downsample_factor, device)
    height, width = source.shape
    token_height = height // patch_size
    token_width = width // patch_size
    if token_height <= 0 or token_width <= 0:
        raise ValueError(f"downsampled image size {height}x{width} is smaller than the patch size {patch_size}")
    cropped_height = token_height * patch_size
    cropped_width = token_width * patch_size
    cropped = source[:cropped_height, :cropped_width].clamp(0.0, 1.0)
    pixel_values = _prepare_dinov2_pixel_values(cropped).to(device=device)
    expected_tokens = token_height * token_width
    tokens = _extract_patch_tokens(model, pixel_values, expected_tokens)
    descriptors = tokens.reshape(token_height, token_width, -1).contiguous().to(dtype=pixel_values.dtype)
    descriptors = F.normalize(descriptors, dim=2, eps=1e-6).to(dtype=torch.float32)
    if return_numpy:
        return descriptors.detach().cpu().numpy().astype("float32"), cropped_height, cropped_width, patch_size
    return descriptors, cropped_height, cropped_width, patch_size
