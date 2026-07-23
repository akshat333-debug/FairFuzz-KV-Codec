from typing import Dict

from pydantic import BaseModel, Field

BITWIDTH_MAP_SCHEMA_VERSION = 1


class BitWidthMap(BaseModel):
    """Compact per-layer/per-head bit-width configuration for K and V
    independently (mixed K/V precision). Only deviations from the default
    are stored - a model with 32 uniform layers costs a few bytes of JSON,
    not one entry per layer/head. Override keys are "{layer}" (whole layer)
    or "{layer}.{head}" (single head), head overrides taking precedence."""

    schema_version: int = BITWIDTH_MAP_SCHEMA_VERSION
    default_k_bits: int
    default_v_bits: int
    k_overrides: Dict[str, int] = Field(default_factory=dict)
    v_overrides: Dict[str, int] = Field(default_factory=dict)

    def bits_for(self, tensor_name: str, layer: int, head: int) -> int:
        if tensor_name not in ("k", "v"):
            raise ValueError(f"tensor_name must be 'k' or 'v', got {tensor_name!r}")
        overrides = self.k_overrides if tensor_name == "k" else self.v_overrides
        default = self.default_k_bits if tensor_name == "k" else self.default_v_bits

        head_key = f"{layer}.{head}"
        if head_key in overrides:
            return overrides[head_key]
        layer_key = str(layer)
        if layer_key in overrides:
            return overrides[layer_key]
        return default

    def bits_for_layer(self, tensor_name: str, layer: int) -> int:
        """Layer-level lookup only, ignoring any head-level override. This is
        what ScalarQuantCodec consults on its fast LAYER path (used when the
        map has no head overrides for this tensor). When head overrides ARE
        present the codec switches to per-(layer, head) cell grouping and
        consults bits_for(tensor, layer, head) instead."""
        if tensor_name not in ("k", "v"):
            raise ValueError(f"tensor_name must be 'k' or 'v', got {tensor_name!r}")
        overrides = self.k_overrides if tensor_name == "k" else self.v_overrides
        default = self.default_k_bits if tensor_name == "k" else self.default_v_bits
        return overrides.get(str(layer), default)

    def set_layer_bits(self, tensor_name: str, layer: int, bits: int) -> None:
        overrides = self.k_overrides if tensor_name == "k" else self.v_overrides
        overrides[str(layer)] = bits

    def set_head_bits(self, tensor_name: str, layer: int, head: int, bits: int) -> None:
        overrides = self.k_overrides if tensor_name == "k" else self.v_overrides
        overrides[f"{layer}.{head}"] = bits
