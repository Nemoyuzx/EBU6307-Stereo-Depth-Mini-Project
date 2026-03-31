from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_MODEL_CACHE: dict[tuple[str, str, str, str], tuple[Any, int]] = {}
_HF_ENDPOINT = "https://hf-mirror.com"


@dataclass(frozen=True)
class O4ExecutionModeStatus:
    requested_mode: str
    selected_mode: str
    descriptor_source: str
    available: bool
    reason: str


def resolve_o4_execution_mode(
    requested_mode: str,
    use_torch: bool,
    dinov3_repo_path: Path | None,
    dinov3_model_name: str,
    dinov3_weights: str,
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

    if mode != "dinov3_cost_volume":
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
            selected_mode="dinov3_cost_volume",
            descriptor_source="dinov3_dense_descriptors",
            available=False,
            reason="dinov3_cost_volume requires the torch backend",
        )

    repo_path = dinov3_repo_path.expanduser() if dinov3_repo_path is not None else None
    if repo_path is None:
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov3_cost_volume",
            descriptor_source="dinov3_dense_descriptors",
            available=False,
            reason="dinov3_cost_volume requires o4.dinov3_repo_path to point at a local facebookresearch/dinov3 checkout",
        )
    if not repo_path.is_dir():
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov3_cost_volume",
            descriptor_source="dinov3_dense_descriptors",
            available=False,
            reason=f"dinov3_repo_path does not exist: {repo_path}",
        )
    if not (repo_path / "hubconf.py").is_file():
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov3_cost_volume",
            descriptor_source="dinov3_dense_descriptors",
            available=False,
            reason=f"dinov3_repo_path is missing hubconf.py: {repo_path}",
        )
    if not str(dinov3_model_name).strip():
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov3_cost_volume",
            descriptor_source="dinov3_dense_descriptors",
            available=False,
            reason="dinov3_cost_volume requires o4.dinov3_model_name",
        )
    if not str(dinov3_weights).strip():
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov3_cost_volume",
            descriptor_source="dinov3_dense_descriptors",
            available=False,
            reason="dinov3_cost_volume requires o4.dinov3_weights so weight loading stays explicit",
        )

    return O4ExecutionModeStatus(
        requested_mode=mode,
        selected_mode="dinov3_cost_volume",
        descriptor_source="dinov3_dense_descriptors",
        available=True,
        reason=(
            "using official DINOv3 dense patch descriptors via local torch.hub loading "
            f"({dinov3_model_name}, weights={dinov3_weights})"
        ),
    )


def _require_torch() -> Any:
    """按需导入 torch。这里保留函数内导入，是因为 torch 属于重量级可选依赖；仅在真正走 torch/DINO 路径时才需要它。"""
    import torch

    return torch


def _prepare_downsampled_gray(image: Any, downsample_factor: int, device: str) -> Any:
    torch = _require_torch()
    # 这里保留局部导入：F 只在 torch 路径中使用，避免模块导入阶段强依赖 torch.nn。
    import torch.nn.functional as F

    if torch.is_tensor(image):
        source = image.to(device=device, dtype=torch.float32)
    else:
        source = torch.as_tensor(image, dtype=torch.float32, device=device)

    if source.ndim != 2:
        raise ValueError("DINOv3 extraction expects a single-channel grayscale image.")

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

    raise ValueError("Unable to infer the DINOv3 patch size from the loaded model.")


def _prepend_to_syspath(path: Path) -> tuple[str, bool]:
    resolved = str(path.resolve())
    already_present = resolved in sys.path
    if not already_present:
        sys.path.insert(0, resolved)
    return resolved, already_present


def _strip_selector_prefix(raw: str) -> str:
    prefix = "Weights."
    if raw.startswith(prefix):
        return raw[len(prefix) :].strip()
    return raw


def _load_dinov3_weights_enum(repo_path: Path) -> Any | None:
    backbones_path = repo_path / "dinov3" / "hub" / "backbones.py"
    if not backbones_path.is_file():
        return None

    added_path, already_present = _prepend_to_syspath(repo_path)
    try:
        module = importlib.import_module("dinov3.hub.backbones")
        module_file = Path(getattr(module, "__file__", "")).resolve()
        if module_file != backbones_path.resolve():
            for module_name in ("dinov3.hub.backbones", "dinov3.hub", "dinov3"):
                sys.modules.pop(module_name, None)
            module = importlib.import_module("dinov3.hub.backbones")
            module_file = Path(getattr(module, "__file__", "")).resolve()
            if module_file != backbones_path.resolve():
                return None
    except Exception:
        return None
    finally:
        if not already_present:
            try:
                sys.path.remove(added_path)
            except ValueError:
                pass

    return getattr(module, "Weights", None)


def _is_remote_weight_location(raw: str) -> bool:
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        return True
    return bool(parsed.scheme and parsed.netloc)


def _coerce_weight_arg(repo_path: Path, weights: str) -> Any:
    raw = str(weights).strip()
    if not raw:
        raise ValueError("DINOv3 weights must be configured explicitly.")

    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate
    if _is_remote_weight_location(raw):
        return raw

    selector = _strip_selector_prefix(raw)
    weights_enum = _load_dinov3_weights_enum(repo_path)
    if weights_enum is not None:
        try:
            return weights_enum[selector]
        except Exception:
            pass

    return raw


def _load_dinov3_model(repo_path: Path, model_name: str, weights: str, device: str) -> tuple[Any, int]:
    cache_key = (str(repo_path.resolve()), str(model_name), str(weights), str(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    torch = _require_torch()
    os.environ.setdefault("HF_ENDPOINT", _HF_ENDPOINT)
    model = torch.hub.load(
        str(repo_path),
        str(model_name).strip(),
        source="local",
        weights=_coerce_weight_arg(repo_path, weights),
    )
    model = model.to(device)
    model.eval()
    patch_size = _resolve_patch_size(model)
    _MODEL_CACHE[cache_key] = (model, patch_size)
    return model, patch_size


def _extract_patch_tokens(model: Any, pixel_values: Any, expected_tokens: int) -> Any:
    outputs: Any
    with _require_torch().inference_mode():
        if hasattr(model, "forward_features"):
            outputs = model.forward_features(pixel_values)
        else:
            outputs = model(pixel_values)

    if isinstance(outputs, dict):
        for key in ("x_norm_patchtokens", "patchtokens", "x_patchtokens", "last_hidden_state"):
            value = outputs.get(key)
            if hasattr(value, "shape"):
                tokens = value
                break
        else:
            raise ValueError(f"Unsupported DINOv3 forward_features output keys: {sorted(outputs)}")
    else:
        tokens = outputs

    if getattr(tokens, "ndim", 0) != 3:
        raise ValueError(f"Unexpected DINOv3 token tensor rank: {getattr(tokens, 'ndim', 'unknown')}")

    token_count = int(tokens.shape[1])
    if token_count == expected_tokens:
        return tokens
    if token_count > expected_tokens:
        return tokens[:, token_count - expected_tokens :, :]
    raise ValueError(f"unexpected DINOv3 token count {token_count} for grid size {expected_tokens}")


def extract_dinov3_descriptors(
    image: Any,
    *,
    downsample_factor: int,
    repo_path: Path,
    model_name: str,
    weights: str,
    device: str,
    return_numpy: bool,
) -> tuple[Any, int, int, int]:
    torch = _require_torch()
    # 这里保留局部导入：F 只在 torch 路径中使用，避免模块导入阶段强依赖 torch.nn。
    import torch.nn.functional as F

    model, patch_size = _load_dinov3_model(repo_path, model_name, weights, device)
    source = _prepare_downsampled_gray(image, downsample_factor, device)
    height, width = source.shape
    token_height = height // patch_size
    token_width = width // patch_size
    if token_height <= 0 or token_width <= 0:
        raise ValueError(
            f"downsampled image size {height}x{width} is smaller than the DINOv3 patch size {patch_size}"
        )

    cropped_height = token_height * patch_size
    cropped_width = token_width * patch_size
    cropped = source[:cropped_height, :cropped_width]
    rgb = cropped.unsqueeze(0).repeat(1, 3, 1, 1)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_values = (rgb - mean) / std

    expected_tokens = token_height * token_width
    tokens = _extract_patch_tokens(model, pixel_values, expected_tokens)
    descriptors = tokens.reshape(token_height, token_width, -1).contiguous().to(dtype=torch.float32)
    descriptors = F.normalize(descriptors, dim=2, eps=1e-6)
    if return_numpy:
        return descriptors.detach().cpu().numpy().astype("float32"), cropped_height, cropped_width, patch_size
    return descriptors, cropped_height, cropped_width, patch_size
