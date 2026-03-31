from __future__ import annotations

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
            descriptor_source="dinov2_dense_descriptors",
            available=False,
            reason="dinov2_cost_volume requires the torch backend",
        )

    if not str(dinov2_model_name).strip():
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_dense_descriptors",
            available=False,
            reason="dinov2_cost_volume requires o4.dinov2_model_name",
        )

    try:
        spec = _resolve_dinov2_model_spec(dinov2_model_name)
    except ValueError as exc:
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_dense_descriptors",
            available=False,
            reason=str(exc),
        )

    resolved_checkpoint = resolve_dinov2_checkpoint_path(dinov2_model_name, dinov2_checkpoint_path)
    if not resolved_checkpoint.is_file():
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_dense_descriptors",
            available=False,
            reason=(
                "dinov2_cost_volume requires o4.dinov2_checkpoint_path to point at a local checkpoint file; "
                f"missing: {resolved_checkpoint}. Expected filename for {spec.builder_name}: {spec.checkpoint_filename}"
            ),
        )
    try:
        resolved_repo_path = resolve_dinov2_repo_path(dinov2_repo_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        return O4ExecutionModeStatus(
            requested_mode=mode,
            selected_mode="dinov2_cost_volume",
            descriptor_source="dinov2_dense_descriptors",
            available=False,
            reason=str(exc),
        )

    return O4ExecutionModeStatus(
        requested_mode=mode,
        selected_mode="dinov2_cost_volume",
        descriptor_source="dinov2_dense_descriptors",
        available=True,
        reason=(
            "using pretrained DINOv2 dense patch descriptors from a local checkpoint "
            f"(model={spec.builder_name}, repo={resolved_repo_path or 'environment'}, checkpoint={resolved_checkpoint})"
        ),
    )


def _require_torch() -> Any:
    """按需导入 torch。这里保留函数内导入，是因为 torch 属于重量级可选依赖；仅在真正走 torch/DINO 路径时才需要它。"""
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
    # 这里保留局部导入：F 只在 torch 路径中使用，避免模块导入阶段强依赖 torch.nn。
    import torch.nn.functional as F

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


def extract_dinov2_descriptors(
    image: Any,
    *,
    downsample_factor: int,
    model_name: str,
    repo_path: str | Path | None,
    checkpoint_path: str | Path,
    device: str,
    return_numpy: bool,
) -> tuple[Any, int, int, int]:
    # 这里保留局部导入：F 只在 torch 路径中使用，避免模块导入阶段强依赖 torch.nn。
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
    pixel_values = _prepare_dinov2_pixel_values(cropped).to(device=device)

    expected_tokens = token_height * token_width
    tokens = _extract_patch_tokens(model, pixel_values, expected_tokens)
    descriptors = tokens.reshape(token_height, token_width, -1).contiguous().to(dtype=pixel_values.dtype)
    descriptors = F.normalize(descriptors, dim=2, eps=1e-6).to(dtype=_require_torch().float32)
    if return_numpy:
        return descriptors.detach().cpu().numpy().astype("float32"), cropped_height, cropped_width, patch_size
    return descriptors, cropped_height, cropped_width, patch_size
