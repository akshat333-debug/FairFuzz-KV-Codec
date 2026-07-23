from enum import Enum
from typing import List, Optional, Sequence, Tuple

import torch

# Standard KV tensor shape convention used throughout this codebase:
# [layers, batch, heads, seq_len, head_dim] (see cache_capture/hf_capture.py).
HEADS_DIM = 2
SEQ_DIM = 3
CHANNEL_DIM = 4


class Granularity(str, Enum):
    PER_TENSOR = "per_tensor"
    PER_HEAD = "per_head"
    PER_CHANNEL = "per_channel"
    GROUPWISE = "groupwise"


class ClipMethod(str, Enum):
    MINMAX = "minmax"  # true min/max, no clipping
    PERCENTILE = "percentile"
    MSE_OPTIMAL = "mse_optimal"


def _keep_dim_for(granularity: Granularity) -> Optional[int]:
    if granularity == Granularity.PER_TENSOR:
        return None
    if granularity == Granularity.PER_HEAD:
        return HEADS_DIM
    if granularity == Granularity.PER_CHANNEL:
        return CHANNEL_DIM
    raise ValueError(f"{granularity} has no single keep-dim - handled separately")


def _flatten_keep_dim(tensor: torch.Tensor, keep_dim: Optional[int]) -> torch.Tensor:
    """Rearrange so `keep_dim` (if any) becomes dim 0 and everything else is
    flattened into dim 1 - lets a single torch.quantile/amin/amax(dim=1) call
    do the reduction regardless of how many dims are being collapsed."""
    if keep_dim is None:
        return tensor.reshape(1, -1)
    perm = [keep_dim] + [d for d in range(tensor.dim()) if d != keep_dim]
    return tensor.permute(perm).contiguous().reshape(tensor.size(keep_dim), -1)


def _grouped_keep_dim_view(tensor: torch.Tensor, group_size: Optional[int]) -> torch.Tensor:
    """Split head_dim into (num_groups, group_size) and move num_groups to
    dim 0, flattening EVERYTHING else (layers/batch/heads/seq AND
    group_size) into dim 1 - so groupwise sits properly between per_tensor
    (1 value) and per_channel (head_dim values): num_groups values total,
    each shared across every layer/batch/head/seq position, exactly like
    per_channel shares one scale per channel across all of those."""
    channel_size = tensor.size(CHANNEL_DIM)
    if not group_size or channel_size % group_size != 0:
        raise ValueError(
            f"groupwise quantization needs head_dim ({channel_size}) divisible by group_size ({group_size})"
        )
    num_groups = channel_size // group_size
    grouped = tensor.reshape(*tensor.shape[:-1], num_groups, group_size)
    keep_dim = grouped.dim() - 2  # the newly-created num_groups axis
    return _flatten_keep_dim(grouped, keep_dim)


def compute_min_max(
    tensor: torch.Tensor, granularity: Granularity, group_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """True min/max at the requested granularity. Returned tensors are
    COMPACT (one value per group/head/channel/tensor, not pre-broadcast to
    the full tensor shape) - broadcast_min_max expands them only when doing
    elementwise quantize/dequantize math, so stored scale/zp tensors stay
    small (matters for the "all overhead counted, nothing wasted" gate)."""
    if granularity == Granularity.GROUPWISE:
        flat = _grouped_keep_dim_view(tensor, group_size)
        return flat.amin(dim=1), flat.amax(dim=1)

    keep_dim = _keep_dim_for(granularity)
    flat = _flatten_keep_dim(tensor, keep_dim)
    return flat.amin(dim=1), flat.amax(dim=1)


def compute_percentile_range(
    tensor: torch.Tensor, granularity: Granularity, percentile: float, group_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip to the [percentile, 100-percentile] range instead of true
    min/max - bounds a small number of extreme outliers from blowing up the
    scale for everyone else. percentile is in (0, 50)."""
    if not 0 < percentile < 50:
        raise ValueError(f"percentile must be in (0, 50), got {percentile}")
    lo_q, hi_q = percentile / 100.0, 1.0 - percentile / 100.0

    if granularity == Granularity.GROUPWISE:
        flat = _grouped_keep_dim_view(tensor, group_size)
        return flat.quantile(lo_q, dim=1), flat.quantile(hi_q, dim=1)

    keep_dim = _keep_dim_for(granularity)
    flat = _flatten_keep_dim(tensor, keep_dim)
    return flat.quantile(lo_q, dim=1), flat.quantile(hi_q, dim=1)


def broadcast_min_max(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    tensor_shape: Sequence[int],
    granularity: Granularity,
    group_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand a compact (min_val, max_val) pair back to a shape broadcastable
    against the full tensor, for the actual quantize/dequantize arithmetic."""
    if granularity == Granularity.GROUPWISE:
        num_groups = min_val.shape[0]
        channel_size = tensor_shape[-1]
        gsize = channel_size // num_groups

        def _expand(v: torch.Tensor) -> torch.Tensor:
            per_channel = v.unsqueeze(-1).expand(num_groups, gsize).reshape(channel_size)
            shape = [1] * (len(tensor_shape) - 1) + [channel_size]
            return per_channel.reshape(shape)

        return _expand(min_val), _expand(max_val)

    keep_dim = _keep_dim_for(granularity)
    shape = [1] * len(tensor_shape)
    if keep_dim is not None:
        shape[keep_dim] = tensor_shape[keep_dim]
    return min_val.reshape(shape), max_val.reshape(shape)


def compute_mse_optimal_range(
    tensor: torch.Tensor,
    granularity: Granularity,
    num_bits: int,
    symmetric: bool,
    group_size: Optional[int] = None,
    num_candidates: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Search a small set of candidate clip fractions of the true min/max
    range and keep whichever minimizes quantization MSE, per group/head/
    channel/tensor independently. A coarse grid search, not a continuous
    optimizer - simple, auditable, and cheap enough to run per-tensor.
    ponytail: 20 candidates evenly spaced in [0.5, 1.0] of the true range;
    finer search only helps if this proves too coarse in practice."""
    from fairfuzzkv_codec.quantization.uniform import dequantize_uniform, quantize_uniform

    true_min, true_max = compute_min_max(tensor, granularity, group_size)
    fractions = torch.linspace(0.5, 1.0, num_candidates)

    best_min, best_max = true_min.clone(), true_max.clone()
    best_mse = None

    for frac in fractions:
        cand_min = true_min * frac.item()
        cand_max = true_max * frac.item()
        b_min, b_max = broadcast_min_max(cand_min, cand_max, tensor.shape, granularity, group_size)

        if symmetric:
            abs_max = torch.max(b_min.abs(), b_max.abs()).clamp(min=1e-5)
            scale = abs_max / ((2 ** (num_bits - 1)) - 1)
            zero_point = torch.zeros_like(scale)
        else:
            span = torch.max(b_max - b_min, torch.full_like(b_max, 1e-5))
            qmin, qmax = 0, (2**num_bits) - 1
            scale = span / (qmax - qmin)
            zero_point = torch.clamp(qmin - torch.round(b_min / scale), qmin, qmax)

        q = quantize_uniform(tensor, num_bits, scale, zero_point, symmetric)
        deq = dequantize_uniform(q, scale, zero_point, dtype=tensor.dtype)
        mse_per_group = (deq - tensor).pow(2)

        if granularity == Granularity.GROUPWISE:
            mse_reduced = _grouped_keep_dim_view(mse_per_group, group_size).mean(dim=1)
        else:
            keep_dim = _keep_dim_for(granularity)
            mse_reduced = _flatten_keep_dim(mse_per_group, keep_dim).mean(dim=1)

        if best_mse is None:
            best_mse = mse_reduced
        else:
            improved = mse_reduced < best_mse
            best_min = torch.where(improved, cand_min, best_min)
            best_max = torch.where(improved, cand_max, best_max)
            best_mse = torch.where(improved, mse_reduced, best_mse)

    return best_min, best_max


def select_range(
    tensor: torch.Tensor,
    granularity: Granularity,
    method: ClipMethod,
    num_bits: int = 8,
    symmetric: bool = True,
    group_size: Optional[int] = None,
    percentile: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single entry point: pick the clipping strategy, return a compact
    (min_val, max_val) pair at the requested granularity."""
    if method == ClipMethod.MINMAX:
        return compute_min_max(tensor, granularity, group_size)
    if method == ClipMethod.PERCENTILE:
        return compute_percentile_range(tensor, granularity, percentile, group_size)
    if method == ClipMethod.MSE_OPTIMAL:
        return compute_mse_optimal_range(tensor, granularity, num_bits, symmetric, group_size)
    raise ValueError(f"unknown clip method: {method}")


def aggregate_calibration_range(
    calibration_tensors: List[torch.Tensor], granularity: Granularity, group_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine several calibration samples (same shape except seq_len) into
    one reusable range: concatenate along the sequence dimension - each
    calibration sample contributes additional token positions to the
    population the per-head/per-channel/groupwise range is estimated over."""
    if not calibration_tensors:
        raise ValueError("calibration_tensors must be non-empty")
    combined = torch.cat(calibration_tensors, dim=SEQ_DIM)
    return compute_min_max(combined, granularity, group_size)
