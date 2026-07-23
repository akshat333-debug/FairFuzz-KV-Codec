import torch
from typing import Tuple, Optional

def compute_scales_and_zeropoints(
    tensor: torch.Tensor, 
    num_bits: int, 
    symmetric: bool = True, 
    dim: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scales and zero-points for uniform quantization.
    dim: If None, per-tensor. If int, per-channel along that dim.
    """
    qmin = -(2 ** (num_bits - 1)) if symmetric else 0
    qmax = (2 ** (num_bits - 1)) - 1 if symmetric else (2 ** num_bits) - 1

    if dim is None:
        # Per-tensor
        min_val = tensor.min()
        max_val = tensor.max()
    else:
        # Per-channel
        dims_to_reduce = tuple(i for i in range(tensor.dim()) if i != dim)
        min_val = tensor.amin(dim=dims_to_reduce, keepdim=True)
        max_val = tensor.amax(dim=dims_to_reduce, keepdim=True)

    if symmetric:
        abs_max = torch.max(min_val.abs(), max_val.abs())
        abs_max = torch.clamp(abs_max, min=1e-5)
        scale = abs_max / qmax
        zero_point = torch.zeros_like(scale)
    else:
        max_val = torch.max(max_val, min_val + 1e-5)
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)

    return scale, zero_point

def quantize_uniform(
    tensor: torch.Tensor, 
    num_bits: int, 
    scale: torch.Tensor, 
    zero_point: torch.Tensor, 
    symmetric: bool = True
) -> torch.Tensor:
    """
    Quantize tensor to integer using given scale/zp.
    Strict saturation handling via torch.clamp.
    """
    qmin = -(2 ** (num_bits - 1)) if symmetric else 0
    qmax = (2 ** (num_bits - 1)) - 1 if symmetric else (2 ** num_bits) - 1
    
    q_tensor = torch.round(tensor / scale) + zero_point
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    
    # Cast to int8 if fits
    if num_bits <= 8:
        q_tensor = q_tensor.to(torch.int8)
    elif num_bits <= 16:
        q_tensor = q_tensor.to(torch.int16)
    
    return q_tensor

def dequantize_uniform(
    q_tensor: torch.Tensor, 
    scale: torch.Tensor, 
    zero_point: torch.Tensor, 
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize back to float.
    """
    q_tensor_float = q_tensor.to(torch.float32)
    deq = (q_tensor_float - zero_point) * scale
    return deq.to(dtype)
