from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _as_float_tensor(values: Any, device: str) -> Any:
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.array(values, copy=True), dtype=torch.float32, device=device)


def _as_int_tensor(values: Any, device: str, *, dtype: Any) -> Any:
    if torch.is_tensor(values):
        return values.to(device=device, dtype=dtype)
    return torch.as_tensor(np.array(values, copy=True), dtype=dtype, device=device)


def _box_filter_sum_torch(image: Any, radius: int) -> Any:
    source = image if image.ndim == 4 else image.unsqueeze(0).unsqueeze(0)
    if radius <= 0:
        return source.squeeze(0).squeeze(0)
    kernel_size = radius * 2 + 1
    padded = F.pad(source, (radius, radius, radius, radius), mode="replicate")
    summed = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1) * float(kernel_size * kernel_size)
    return summed.squeeze(0).squeeze(0)


def _compute_horizontal_gradient_torch(image: Any) -> Any:
    
    source = image if image.ndim == 2 else image.squeeze(0).squeeze(0)
    gradient = torch.zeros_like(source, dtype=torch.float32)
    if source.shape[1] > 1:
        gradient[:, 1:-1] = 0.5 * (source[:, 2:] - source[:, :-2])
        gradient[:, 0] = source[:, 1] - source[:, 0]
        gradient[:, -1] = source[:, -1] - source[:, -2]
    return gradient


def _shift_2d_tensor(image: Any, disparity: int, target_direction: str, fill_value: float = 0.0) -> Any:
    
    height, width = image.shape
    shifted = torch.full_like(image, float(fill_value))
    if disparity == 0:
        shifted.copy_(image)
        return shifted
    if disparity >= width:
        return shifted
    if target_direction == "positive":
        shifted[:, : width - disparity] = image[:, disparity:]
    else:
        shifted[:, disparity:] = image[:, : width - disparity]
    return shifted


def _shift_3d_tensor(image: Any, disparity: int, target_direction: str, fill_value: float = 0.0) -> Any:
    
    channels, height, width = image.shape
    shifted = torch.full((channels, height, width), float(fill_value), dtype=image.dtype, device=image.device)
    if disparity == 0:
        shifted.copy_(image)
        return shifted
    if disparity >= width:
        return shifted
    if target_direction == "positive":
        shifted[:, :, : width - disparity] = image[:, :, disparity:]
    else:
        shifted[:, :, disparity:] = image[:, :, : width - disparity]
    return shifted


def average_pool_gray_torch(image: Any, factor: int, *, device: str) -> Any:
    source = _as_float_tensor(image, device)
    if factor <= 1:
        return torch.round(source).to(dtype=torch.uint8)
    height, width = source.shape[:2]
    pooled_height = max(1, height // factor)
    pooled_width = max(1, width // factor)
    trimmed = source[: pooled_height * factor, : pooled_width * factor]
    pooled = F.avg_pool2d(trimmed.unsqueeze(0).unsqueeze(0), kernel_size=factor, stride=factor)
    return torch.round(pooled.squeeze(0).squeeze(0)).to(dtype=torch.uint8)


def average_pool_2d_torch(image: Any, factor: int, *, device: str) -> Any:
    source = _as_float_tensor(image, device)
    if factor <= 1:
        return source
    height, width = source.shape[:2]
    pooled_height = max(1, height // factor)
    pooled_width = max(1, width // factor)
    trimmed = source[: pooled_height * factor, : pooled_width * factor]
    pooled = F.avg_pool2d(trimmed.unsqueeze(0).unsqueeze(0), kernel_size=factor, stride=factor)
    return pooled.squeeze(0).squeeze(0)


def median_filter_2d_torch(image: Any, kernel_size: int, *, device: str) -> Any:
    source = _as_float_tensor(image, device)
    if kernel_size <= 1:
        return source
    radius = kernel_size // 2
    padded = F.pad(source.unsqueeze(0).unsqueeze(0), (radius, radius, radius, radius), mode="replicate")
    windows = F.unfold(padded, kernel_size=(kernel_size, kernel_size))
    median = windows.median(dim=1).values.reshape(source.shape[0], source.shape[1])
    return median.to(dtype=source.dtype)


def upsample_token_grid_torch(token_values: Any, span: int, output_height: int, output_width: int, *, device: str) -> Any:
    source = _as_float_tensor(token_values, device)
    upsampled = source.repeat_interleave(int(span), dim=0).repeat_interleave(int(span), dim=1)
    result = source.new_zeros((output_height, output_width))
    copy_height = min(output_height, int(upsampled.shape[0]))
    copy_width = min(output_width, int(upsampled.shape[1]))
    result[:copy_height, :copy_width] = upsampled[:copy_height, :copy_width]
    return result


def build_token_descriptors_torch(image: Any, patch_size: int, context_window_size: int, *, device: str) -> tuple[Any, int, int]:
    
    source = _as_float_tensor(image, device) / 255.0
    gradient_x = _compute_horizontal_gradient_torch(source)
    gradient_y = _compute_horizontal_gradient_torch(source.transpose(0, 1)).transpose(0, 1)
    area = float(max(1, context_window_size * context_window_size))
    context = _box_filter_sum_torch(source, context_window_size // 2) / area
    channels = [source, gradient_x, gradient_y, context]
    height, width = source.shape
    token_height = height // patch_size
    token_width = width // patch_size
    cropped_height = token_height * patch_size
    cropped_width = token_width * patch_size
    descriptor_parts: list[Any] = []
    for channel in channels:
        cropped = channel[:cropped_height, :cropped_width].contiguous()
        tokens = cropped.reshape(token_height, patch_size, token_width, patch_size)
        tokens = tokens.permute(0, 2, 1, 3).reshape(token_height, token_width, patch_size * patch_size)
        descriptor_parts.append(tokens)

    intensity_tokens = descriptor_parts[0]
    descriptor_parts.append(intensity_tokens.mean(dim=2, keepdim=True))
    descriptor_parts.append(intensity_tokens.std(dim=2, keepdim=True, unbiased=False))
    descriptors = torch.cat(descriptor_parts, dim=2).to(dtype=torch.float32)
    descriptors = descriptors - descriptors.mean(dim=2, keepdim=True)
    descriptors = descriptors / torch.clamp(descriptors.std(dim=2, keepdim=True, unbiased=False), min=1e-6)
    return descriptors, cropped_height, cropped_width


def build_token_ground_truth_torch(disparity: Any, token_span: int, token_shape: tuple[int, int], *, device: str) -> Any:
    
    source = _as_float_tensor(disparity, device)
    token_height, token_width = token_shape
    cropped_height = token_height * token_span
    cropped_width = token_width * token_span
    cropped = source[:cropped_height, :cropped_width]
    if cropped.numel() == 0:
        return torch.full(token_shape, -1, dtype=torch.int32, device=device)

    patches = cropped.reshape(token_height, token_span, token_width, token_span).permute(0, 2, 1, 3)
    patches = patches.reshape(token_height, token_width, token_span * token_span)
    valid_mask = torch.isfinite(patches) & (patches > 0)
    masked = torch.where(valid_mask, patches, torch.full_like(patches, float("nan")))
    median = torch.nanmedian(masked, dim=2).values
    labels = torch.full((token_height, token_width), -1, dtype=torch.int32, device=device)
    has_valid = valid_mask.any(dim=2) & torch.isfinite(median)
    labels[has_valid] = torch.round(median[has_valid] / float(token_span)).to(dtype=torch.int32)
    return labels


def collect_o4_training_samples_torch(scene_payloads: list[dict[str, Any]], eval_fold: int, config: Any, *, device: str) -> tuple[Any, Any]:
    
    max_samples = max(1, int(config.max_training_samples))
    negative_samples = max(1, int(config.negative_samples))
    left_batches: list[Any] = []
    candidate_batches: list[Any] = []
    total = 0

    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.random_seed) + int(eval_fold))

    for payload in scene_payloads:
        if payload["fold"] == eval_fold or payload["ground_truth_tokens"] is None:
            continue

        labels = _as_int_tensor(payload["ground_truth_tokens"], device, dtype=torch.int32)
        right_descriptors = _as_float_tensor(payload["right_descriptors"], device)
        left_descriptors = _as_float_tensor(payload["left_descriptors"], device)
        min_disparity = int(payload["min_token_disparity"])
        max_disparity = int(payload["max_token_disparity"])

        valid_locations = torch.nonzero(labels >= min_disparity, as_tuple=False)
        if valid_locations.numel() == 0:
            continue

        order = torch.randperm(valid_locations.shape[0], generator=generator, device=device)
        valid_locations = valid_locations.index_select(0, order)
        remaining = max_samples - total
        if remaining <= 0:
            break
        selected = valid_locations[:remaining]
        rows = selected[:, 0]
        cols = selected[:, 1]
        disparities = labels[rows, cols].to(dtype=torch.int64)
        match_columns = cols.to(dtype=torch.int64) - disparities
        sample_mask = (
            (disparities >= min_disparity)
            & (disparities <= max_disparity)
            & (match_columns >= 0)
            & (match_columns < right_descriptors.shape[1])
        )
        if not bool(sample_mask.any()):
            continue

        rows = rows[sample_mask].to(dtype=torch.int64)
        cols = cols[sample_mask].to(dtype=torch.int64)
        disparities = disparities[sample_mask]
        sample_count = int(rows.shape[0])
        if sample_count == 0:
            continue

        negatives = torch.randint(
            min_disparity,
            max_disparity + 1,
            (sample_count, negative_samples),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        valid_negative = (
            (negatives != disparities[:, None])
            & ((cols[:, None] - negatives) >= 0)
            & ((cols[:, None] - negatives) < right_descriptors.shape[1])
        )
        while not bool(valid_negative.all()):
            replacement = torch.randint(
                min_disparity,
                max_disparity + 1,
                negatives.shape,
                generator=generator,
                device=device,
                dtype=torch.int64,
            )
            negatives = torch.where(valid_negative, negatives, replacement)
            valid_negative = (
                (negatives != disparities[:, None])
                & ((cols[:, None] - negatives) >= 0)
                & ((cols[:, None] - negatives) < right_descriptors.shape[1])
            )

        candidate_disparities = torch.cat([disparities[:, None], negatives], dim=1)
        candidate_columns = cols[:, None] - candidate_disparities
        left_batches.append(left_descriptors[rows, cols])
        candidate_batches.append(right_descriptors[rows[:, None], candidate_columns])
        total += sample_count
        if total >= max_samples:
            break

    if not left_batches:
        feature_dim = int(scene_payloads[0]["left_descriptors"].shape[2]) if scene_payloads else 1
        return (
            torch.zeros((0, feature_dim), dtype=torch.float32, device=device),
            torch.zeros((0, negative_samples + 1, feature_dim), dtype=torch.float32, device=device),
        )
    return torch.cat(left_batches, dim=0), torch.cat(candidate_batches, dim=0)


def extract_baseline_patch_tokens_torch(image: Any, patch_size: int, *, device: str) -> tuple[Any, int, int]:
    
    source = _as_float_tensor(image, device)
    height, width = source.shape
    token_height = height // patch_size
    token_width = width // patch_size
    cropped_height = token_height * patch_size
    cropped_width = token_width * patch_size
    cropped = source[:cropped_height, :cropped_width].contiguous()
    tokens = cropped.reshape(token_height, patch_size, token_width, patch_size)
    tokens = tokens.permute(0, 2, 1, 3).reshape(token_height, token_width, patch_size * patch_size)
    tokens = tokens.to(dtype=torch.float32) / 255.0
    tokens = tokens - tokens.mean(dim=2, keepdim=True)
    norms = torch.linalg.norm(tokens, dim=2, keepdim=True)
    return torch.where(norms > 0, tokens / torch.clamp(norms, min=1e-6), torch.zeros_like(tokens)), cropped_height, cropped_width


def refine_token_disparity_torch(best_disparity: Any, all_scores: Any, min_disparity: int, max_disparity: int, *, device: str) -> Any:
    
    refined = _as_float_tensor(best_disparity, device).clone()
    score_volume = _as_float_tensor(all_scores, device)
    if max_disparity <= min_disparity:
        return refined
    score_index = torch.round(refined).to(dtype=torch.int64) - int(min_disparity)
    valid = (score_index > 0) & (score_index < score_volume.shape[0] - 1)
    rows, cols = torch.nonzero(valid, as_tuple=True)
    if rows.numel() == 0:
        return refined
    current_index = score_index[rows, cols]
    center = score_volume[current_index, rows, cols]
    left = score_volume[current_index - 1, rows, cols]
    right = score_volume[current_index + 1, rows, cols]
    denominator = left - (2.0 * center) + right
    stable = torch.isfinite(center) & torch.isfinite(left) & torch.isfinite(right) & (torch.abs(denominator) > 1e-6)
    offset = torch.zeros_like(center)
    offset[stable] = 0.5 * (left[stable] - right[stable]) / denominator[stable]
    refined[rows, cols] = refined[rows, cols] + torch.clamp(offset, -1.0, 1.0)
    return refined


def soft_argmax_disparity_torch(all_scores: Any, min_disparity: int, max_disparity: int, temperature: float, *, device: str) -> Any:
    
    scores = _as_float_tensor(all_scores, device)
    if scores.ndim != 3:
        raise ValueError("soft_argmax_disparity_torch expects a 3D score volume.")

    stabilized = scores - scores.amax(dim=0, keepdim=True)
    invalid = ~torch.isfinite(stabilized)
    stabilized = stabilized.masked_fill(invalid, -1e4)
    logits = stabilized / max(float(temperature), 1e-3)
    probabilities = torch.softmax(logits, dim=0)
    valid = torch.isfinite(scores).any(dim=0)
    disparity_values = torch.arange(
        int(min_disparity),
        int(max_disparity) + 1,
        device=device,
        dtype=torch.float32,
    ).view(-1, 1, 1)
    refined = (probabilities * disparity_values).sum(dim=0)
    return torch.where(valid, refined, torch.zeros_like(refined))


def left_right_consistency_mask_torch(left_disparity: Any, right_disparity: Any, threshold: float, *, device: str) -> Any:
    
    left = _as_float_tensor(left_disparity, device)
    right = _as_float_tensor(right_disparity, device)
    height, width = left.shape
    columns = torch.arange(width, device=device, dtype=torch.int64).unsqueeze(0).expand(height, width)
    matched_columns = columns - torch.round(left).to(dtype=torch.int64)
    valid = (left > 0) & (matched_columns >= 0) & (matched_columns < width)
    sampled = torch.zeros_like(left)
    row_indices = torch.arange(height, device=device, dtype=torch.int64).unsqueeze(1).expand(height, width)
    sampled[valid] = right[row_indices[valid], matched_columns[valid]]
    return valid & (torch.abs(left - sampled) <= float(threshold))


def _propagate_last_valid(values: Any) -> tuple[Any, Any]:
    
    mask = values > 0
    indices = torch.arange(values.shape[1], device=values.device, dtype=torch.int64).unsqueeze(0).expand_as(values)
    base = torch.zeros_like(indices)
    last_index = torch.where(mask, indices, base).cummax(dim=1).values
    has_value = mask.cumsum(dim=1) > 0
    gathered = values.gather(1, last_index)
    return gathered, has_value


def fill_invalid_disparity_torch(disparity: Any, passes: int, *, device: str) -> Any:
    
    filled = _as_float_tensor(disparity, device).clone()
    if passes <= 0:
        return filled
    for _ in range(int(passes)):
        left_values, has_left = _propagate_last_valid(filled)
        right_values_rev, has_right_rev = _propagate_last_valid(torch.flip(filled, dims=[1]))
        right_values = torch.flip(right_values_rev, dims=[1])
        has_right = torch.flip(has_right_rev, dims=[1])
        fill_mask = (filled <= 0) & has_left & has_right
        filled = torch.where(fill_mask, torch.minimum(left_values, right_values), filled)

        transposed = filled.transpose(0, 1)
        top_values, has_top = _propagate_last_valid(transposed)
        bottom_values_rev, has_bottom_rev = _propagate_last_valid(torch.flip(transposed, dims=[1]))
        bottom_values = torch.flip(bottom_values_rev, dims=[1])
        has_bottom = torch.flip(has_bottom_rev, dims=[1])
        fill_mask = (transposed <= 0) & has_top & has_bottom
        transposed = torch.where(fill_mask, torch.minimum(top_values, bottom_values), transposed)
        filled = transposed.transpose(0, 1)
    return filled


def compute_block_disparity_torch(left_gray: Any, right_gray: Any, config: Any, *, device: str) -> Any:
    
    left = _as_float_tensor(left_gray, device)
    right = _as_float_tensor(right_gray, device)
    left_gradient = _compute_horizontal_gradient_torch(left)
    right_gradient = _compute_horizontal_gradient_torch(right)

    radius = max(0, int(config.census_window_size) // 2)
    census_offsets = [
        (offset_y, offset_x)
        for offset_y in range(-radius, radius + 1)
        for offset_x in range(-radius, radius + 1)
        if not (offset_y == 0 and offset_x == 0)
    ]
    if radius > 0 and left.shape[0] > radius * 2 and left.shape[1] > radius * 2:
        center_left = left[radius : left.shape[0] - radius, radius : left.shape[1] - radius]
        center_right = right[radius : right.shape[0] - radius, radius : right.shape[1] - radius]
        left_census = torch.stack([
            (left[radius + dy : left.shape[0] - radius + dy, radius + dx : left.shape[1] - radius + dx] < center_left).to(torch.float32)
            for dy, dx in census_offsets
        ], dim=0)
        right_census = torch.stack([
            (right[radius + dy : right.shape[0] - radius + dy, radius + dx : right.shape[1] - radius + dx] < center_right).to(torch.float32)
            for dy, dx in census_offsets
        ], dim=0)
        if radius > 0:
            left_census = F.pad(left_census, (radius, radius, radius, radius))
            right_census = F.pad(right_census, (radius, radius, radius, radius))
    else:
        left_census = left.new_zeros((1, left.shape[0], left.shape[1]))
        right_census = right.new_zeros((1, right.shape[0], right.shape[1]))

    height, width = left.shape
    if width <= 1:
        return left.new_zeros((height, width))

    max_disparity = min(max(1, int(config.num_disparities)), width - 1)
    support_radius = int(config.block_size) // 2

    def compute_matching(reference: Any, target: Any, reference_gradient: Any, target_gradient: Any, reference_census: Any, target_census: Any, target_direction: str) -> tuple[Any, Any, Any]:
        best_cost = torch.full((height, width), torch.inf, dtype=torch.float32, device=device)
        second_cost = torch.full((height, width), torch.inf, dtype=torch.float32, device=device)
        best_disparity = torch.zeros((height, width), dtype=torch.float32, device=device)

        for disparity in range(max_disparity):
            shifted_gray = _shift_2d_tensor(target, disparity, target_direction)
            shifted_gradient = _shift_2d_tensor(target_gradient, disparity, target_direction)
            shifted_census = _shift_3d_tensor(target_census, disparity, target_direction)

            intensity_cost = _box_filter_sum_torch(torch.abs(reference - shifted_gray), support_radius)
            gradient_cost = _box_filter_sum_torch(torch.abs(reference_gradient - shifted_gradient), support_radius)
            census_cost = _box_filter_sum_torch((reference_census != shifted_census).sum(dim=0, dtype=torch.float32), support_radius)
            cost = intensity_cost + (float(config.gradient_weight) * gradient_cost) + (float(config.census_weight) * census_cost)
            if disparity > 0:
                if target_direction == "positive":
                    cost[:, width - disparity :] = torch.inf
                else:
                    cost[:, :disparity] = torch.inf

            replace_mask = cost < best_cost
            second_cost = torch.where(replace_mask, best_cost, torch.minimum(second_cost, cost))
            best_cost = torch.where(replace_mask, cost, best_cost)
            best_disparity = torch.where(replace_mask, torch.full_like(best_disparity, float(disparity)), best_disparity)

        return best_disparity, best_cost, second_cost

    left_disparity, left_cost, left_second_cost = compute_matching(
        left,
        right,
        left_gradient,
        right_gradient,
        left_census,
        right_census,
        "negative",
    )
    right_disparity, _, _ = compute_matching(
        right,
        left,
        right_gradient,
        left_gradient,
        right_census,
        left_census,
        "positive",
    )

    if int(config.uniqueness_ratio) > 0:
        uniqueness_margin = left_second_cost - left_cost
        uniqueness_floor = torch.clamp(left_cost * (float(config.uniqueness_ratio) / 100.0), min=1e-3)
        left_disparity = torch.where(uniqueness_margin >= uniqueness_floor, left_disparity, torch.zeros_like(left_disparity))

    consistent_mask = left_right_consistency_mask_torch(left_disparity, right_disparity, float(config.consistency_threshold), device=device)
    disparity = torch.where(consistent_mask, left_disparity, torch.zeros_like(left_disparity))

    invalid_margin = max(int(config.block_size) // 2, int(config.census_window_size) // 2)
    if invalid_margin > 0:
        disparity[:invalid_margin, :] = 0.0
        disparity[-invalid_margin:, :] = 0.0
        disparity[:, :invalid_margin] = 0.0
        disparity[:, -invalid_margin:] = 0.0

    disparity = fill_invalid_disparity_torch(disparity, int(config.fill_invalid_passes), device=device)
    if int(config.median_filter_size) > 1:
        positive_mask = disparity > 0
        filtered = median_filter_2d_torch(disparity, int(config.median_filter_size), device=device)
        disparity = torch.where(positive_mask, filtered, torch.zeros_like(disparity))

    return disparity.to(dtype=torch.float32)


@dataclass(frozen=True)
class TorchBackendStatus:
    selected_backend: str
    use_torch: bool
    device: str
    reason: str


def resolve_torch_backend(
    requested_backend: str,
    requested_device: str,
    prefer_cuda: bool,
) -> TorchBackendStatus:
    backend_name = str(requested_backend or "auto").strip().lower()
    device_name = str(requested_device or "auto").strip().lower()

    if backend_name == "numpy":
        return TorchBackendStatus(
            selected_backend="numpy",
            use_torch=False,
            device="cpu",
            reason="configured to use the numpy fallback backend",
        )

    try:
        _ = torch
    except Exception as exc:
        fallback_reason = f"torch import failed: {exc}"
        if backend_name == "torch":
            fallback_reason = f"{fallback_reason}; falling back to numpy"
        return TorchBackendStatus(
            selected_backend="numpy",
            use_torch=False,
            device="cpu",
            reason=fallback_reason,
        )

    cuda_available = bool(torch.cuda.is_available())
    if device_name == "cuda":
        if cuda_available:
            return TorchBackendStatus(
                selected_backend="torch",
                use_torch=True,
                device="cuda",
                reason="configured for the torch backend on CUDA",
            )
        return TorchBackendStatus(
            selected_backend="numpy" if backend_name == "auto" else "torch",
            use_torch=backend_name == "torch",
            device="cpu",
            reason="CUDA requested but torch.cuda.is_available() is false",
        )

    if device_name == "cpu":
        return TorchBackendStatus(
            selected_backend="torch",
            use_torch=True,
            device="cpu",
            reason="configured for the torch backend on CPU",
        )

    if prefer_cuda and cuda_available:
        return TorchBackendStatus(
            selected_backend="torch",
            use_torch=True,
            device="cuda",
            reason="torch is installed and CUDA is available",
        )

    return TorchBackendStatus(
        selected_backend="torch",
        use_torch=True,
        device="cpu",
        reason="torch is installed; using CPU because CUDA is unavailable or not preferred",
    )


def train_o4_torch_model(
    training_descriptors: Any,
    candidate_descriptors: Any,
    *,
    model_dim: int,
    hidden_dim: int,
    encoder_layers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    random_seed: int,
    device: str,
) -> Any | None:
    if training_descriptors.size == 0 or candidate_descriptors.size == 0 or epochs <= 0:
        return None

    feature_dim = int(training_descriptors.shape[1])
    if feature_dim <= 0:
        return None

    torch.manual_seed(int(random_seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(random_seed))

    class StereoTokenTransformer(torch.nn.Module):
        def __init__(self, input_dim: int, embedding_dim: int, inner_dim: int, depth: int) -> None:
            super().__init__()
            self.sequence_length = min(4, max(1, int(input_dim)))
            head_count = 4 if embedding_dim % 4 == 0 else 2 if embedding_dim % 2 == 0 else 1
            self.input_projection = torch.nn.Linear(input_dim, embedding_dim * self.sequence_length)
            self.position_embedding = torch.nn.Parameter(torch.zeros(1, self.sequence_length, embedding_dim))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=head_count,
                dim_feedforward=max(int(inner_dim), int(embedding_dim) * 2),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(depth)))
            self.output_norm = torch.nn.LayerNorm(embedding_dim)

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            sequence = self.input_projection(inputs).reshape(inputs.shape[0], self.sequence_length, -1)
            encoded = self.encoder(sequence + self.position_embedding).mean(dim=1)
            encoded = self.output_norm(encoded)
            return F.normalize(encoded, dim=-1, eps=1e-6)

        def forward(self, left: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
            left_embedding = self.encode(left)
            candidate_embedding = self.encode(candidates.reshape(-1, candidates.shape[-1])).reshape(
                candidates.shape[0],
                candidates.shape[1],
                -1,
            )
            return torch.einsum("bd,bkd->bk", left_embedding, candidate_embedding)

    model = StereoTokenTransformer(
        input_dim=feature_dim,
        embedding_dim=int(model_dim),
        inner_dim=max(int(hidden_dim), int(model_dim)),
        depth=max(1, int(encoder_layers)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    left_tensor = _as_float_tensor(training_descriptors, device)
    candidate_tensor = _as_float_tensor(candidate_descriptors, device)
    target = torch.zeros((min(batch_size, left_tensor.shape[0]),), dtype=torch.long, device=device)

    for _ in range(int(epochs)):
        order = torch.randperm(left_tensor.shape[0], device=device)
        for start in range(0, order.shape[0], int(batch_size)):
            batch_index = order[start : start + int(batch_size)]
            current_left = left_tensor.index_select(0, batch_index)
            current_candidates = candidate_tensor.index_select(0, batch_index)
            current_target = target[: batch_index.shape[0]]

            optimizer.zero_grad(set_to_none=True)
            logits = model(current_left, current_candidates)
            loss = F.cross_entropy(logits, current_target)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def encode_o4_descriptors_torch(
    model: Any,
    descriptors: Any,
    *,
    batch_size: int,
    device: str,
    return_numpy: bool = True,
) -> Any:
    source = _as_float_tensor(descriptors, device)
    if source.numel() == 0:
        return source

    token_height, token_width, feature_dim = source.shape
    flattened = source.reshape(-1, feature_dim)
    outputs: list[Any] = []

    with torch.inference_mode():
        for start in range(0, flattened.shape[0], int(batch_size)):
            batch = flattened[start : start + int(batch_size)]
            encoded = model.encode(batch)
            outputs.append(encoded)

    encoded = torch.cat(outputs, dim=0).reshape(token_height, token_width, -1).to(dtype=torch.float32)
    if return_numpy:
        return encoded.detach().cpu().numpy().astype("float32")
    return encoded


def predict_token_disparity_torch(
    left_tokens: Any,
    right_tokens: Any,
    min_disparity: int,
    max_disparity: int,
    *,
    device: str,
    target_direction: str = "negative",
    return_numpy: bool = True,
) -> tuple[Any, Any, Any]:
    left = _as_float_tensor(left_tokens, device)
    right = _as_float_tensor(right_tokens, device)
    token_height, token_width, _ = left.shape

    best_scores = torch.full((token_height, token_width), -torch.inf, dtype=torch.float32, device=device)
    second_scores = torch.full((token_height, token_width), -torch.inf, dtype=torch.float32, device=device)
    best_disparities = torch.zeros((token_height, token_width), dtype=torch.int32, device=device)
    all_scores = torch.full(
        (max_disparity - min_disparity + 1, token_height, token_width),
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
        best_disparities = torch.where(
            replace_mask,
            torch.full_like(best_disparities, int(disparity)),
            best_disparities,
        )

    confidence = torch.where(torch.isfinite(best_scores), best_scores - second_scores, torch.zeros_like(best_scores))
    if return_numpy:
        return (
            best_disparities.detach().cpu().numpy().astype("int32"),
            confidence.detach().cpu().numpy().astype("float32"),
            all_scores.detach().cpu().numpy().astype("float32"),
        )
    return best_disparities, confidence, all_scores
