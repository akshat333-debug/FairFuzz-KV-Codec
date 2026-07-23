from typing import Any, Dict

import torch
from pydantic import BaseModel


class SaturationReport(BaseModel):
    num_elements: int
    num_saturated_low: int
    num_saturated_high: int
    saturation_rate: float  # (low + high) / num_elements

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def compute_saturation(tensor: torch.Tensor, qmin: int, qmax: int, scale: torch.Tensor, zero_point: torch.Tensor) -> SaturationReport:
    """How many values would be clamped (saturated) at the quantization
    boundary before rounding - i.e. how much the chosen clip range is
    actually cutting off, not just theoretical headroom."""
    raw = tensor / scale + zero_point
    num_low = int((raw < qmin).sum().item())
    num_high = int((raw > qmax).sum().item())
    total = tensor.numel()
    return SaturationReport(
        num_elements=total,
        num_saturated_low=num_low,
        num_saturated_high=num_high,
        saturation_rate=(num_low + num_high) / total if total > 0 else 0.0,
    )
