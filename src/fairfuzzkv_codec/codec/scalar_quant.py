from typing import Any, Dict, List, Optional, Tuple

import torch

from fairfuzzkv_codec.codec.base import BaseCodec
from fairfuzzkv_codec.codec.binary_serializer import BinarySerializer
from fairfuzzkv_codec.quantization.bitwidth_map import BitWidthMap
from fairfuzzkv_codec.quantization.diagnostics import compute_saturation
from fairfuzzkv_codec.quantization.packing import pack_int4, pack_int8, unpack_int4, unpack_int8
from fairfuzzkv_codec.quantization.scales import ClipMethod, Granularity, broadcast_min_max, select_range

# KV tensor shape convention: [layers, batch, heads, seq_len, head_dim].
LAYER_DIM = 0
HEAD_DIM = 2


def _scale_zp_from_range(
    min_val: torch.Tensor, max_val: torch.Tensor, num_bits: int, symmetric: bool
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if symmetric:
        abs_max = torch.max(min_val.abs(), max_val.abs()).clamp(min=1e-5)
        qmax = (2 ** (num_bits - 1)) - 1
        qmin = -(2 ** (num_bits - 1))
        scale = abs_max / qmax
        zero_point = torch.zeros_like(scale)
    else:
        span = torch.max(max_val - min_val, torch.full_like(max_val, 1e-5))
        qmin, qmax = 0, (2**num_bits) - 1
        scale = span / (qmax - qmin)
        zero_point = torch.clamp(qmin - torch.round(min_val / scale), qmin, qmax)
    return scale, zero_point, qmin, qmax


class ScalarQuantCodec(BaseCodec):
    """Strong scalar quantization codec: symmetric/asymmetric INT8/INT4,
    per-tensor/per-head/per-channel/groupwise scale granularity, percentile
    or MSE-optimal clipping, genuine INT4 nibble packing, and mixed K/V /
    per-layer bit-width configuration via a shared BitWidthMap.

    One instance handles ONE of K or V (tensor_name="k" or "v") - construct
    two instances sharing the same BitWidthMap for mixed K/V precision,
    matching how the rest of this codebase already calls encode_prefill(K)
    and encode_prefill(V) as separate calls (see benchmarks/fragkv_minpairs/
    runner.py). Mixed precision groups by LAYER when the BitWidthMap has no
    head-level overrides for this tensor (fast path, byte-identical to the
    original layer-only implementation); when head overrides ARE present it
    switches to a per-(layer, head) cell grouping so individual heads can
    carry distinct bit-widths. Each cell is quantized as a single-head slice,
    so PER_HEAD and PER_TENSOR granularity coincide within a cell (one scale
    per head, which is exactly what per-head means for one head)."""

    def __init__(
        self,
        config_hash: str,
        tensor_name: str,
        granularity: Granularity = Granularity.PER_CHANNEL,
        clip_method: ClipMethod = ClipMethod.MINMAX,
        symmetric: bool = True,
        group_size: Optional[int] = None,
        percentile: float = 0.5,
        bitwidth_map: Optional[BitWidthMap] = None,
        default_bits: int = 8,
    ):
        if tensor_name not in ("k", "v"):
            raise ValueError(f"tensor_name must be 'k' or 'v', got {tensor_name!r}")
        self.config_hash = config_hash
        self.tensor_name = tensor_name
        self.granularity = granularity
        self.clip_method = clip_method
        self.symmetric = symmetric
        self.group_size = group_size
        self.percentile = percentile
        self.bitwidth_map = bitwidth_map or BitWidthMap(default_k_bits=default_bits, default_v_bits=default_bits)

    def _group_layers_by_bits(self, num_layers: int) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for layer in range(num_layers):
            bits = self.bitwidth_map.bits_for_layer(self.tensor_name, layer)
            groups.setdefault(bits, []).append(layer)
        return groups

    def _has_head_overrides(self) -> bool:
        overrides = (
            self.bitwidth_map.k_overrides if self.tensor_name == "k" else self.bitwidth_map.v_overrides
        )
        return any("." in key for key in overrides)

    def _group_cells_by_bits(
        self, num_layers: int, num_heads: int
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Partition every (layer, head) cell by its bit-width, consulting the
        full bits_for(layer, head) lookup so head-level overrides take effect."""
        groups: Dict[int, List[Tuple[int, int]]] = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                bits = self.bitwidth_map.bits_for(self.tensor_name, layer, head)
                groups.setdefault(bits, []).append((layer, head))
        return groups

    def _encode_group(
        self,
        key: str,
        slice_tensor: torch.Tensor,
        bits: int,
        tensors: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
        saturation_reports: Dict[str, Any],
    ) -> None:
        """Quantize + serialize one bit-width group. Identical for the layer
        and cell paths - they differ only in how slice_tensor is gathered and
        how positions are recorded/scattered back on decode."""
        min_val, max_val = select_range(
            slice_tensor, self.granularity, self.clip_method,
            num_bits=bits, symmetric=self.symmetric,
            group_size=self.group_size, percentile=self.percentile,
        )
        # Compute scale/zero_point at the COMPACT (min_val, max_val)
        # resolution - one value per group/head/channel/tensor - and only
        # broadcast to full element resolution for the arithmetic below.
        # Storing the broadcast version would silently inflate groupwise
        # storage to one scale per element instead of one per group.
        scale, zero_point, qmin, qmax = _scale_zp_from_range(min_val, max_val, bits, self.symmetric)
        b_scale, b_zp = broadcast_min_max(scale, zero_point, slice_tensor.shape, self.granularity, self.group_size)

        saturation = compute_saturation(slice_tensor, qmin, qmax, b_scale, b_zp)
        saturation_reports[key] = saturation.to_dict()

        # Symmetric ranges are signed (fit int8); asymmetric ranges are
        # unsigned (8-bit asymmetric spans 0..255, which overflows a
        # signed int8 container) - the intermediate dtype must match or
        # values >=128 silently wrap negative.
        q_dtype = torch.int8 if self.symmetric else torch.uint8
        q = torch.round(slice_tensor / b_scale + b_zp).clamp(qmin, qmax).to(q_dtype)

        if bits == 4:
            packed = pack_int4(q, signed=self.symmetric)
            tensors[f"{key}_data"] = packed
            metadata[f"{key}_data_dtype"] = "uint8"
            metadata[f"{key}_data_shape"] = [packed.numel()]
            metadata[f"{key}_data_logical_bits_per_element"] = 4.0 * q.numel() / packed.numel()
        else:
            packed = pack_int8(q)
            tensors[f"{key}_data"] = packed
            metadata[f"{key}_data_dtype"] = "int8" if self.symmetric else "uint8"
            metadata[f"{key}_data_shape"] = [packed.numel()]
            metadata[f"{key}_data_logical_bits_per_element"] = float(bits)

        tensors[f"{key}_scale"] = scale.to(torch.float32).contiguous()
        metadata[f"{key}_scale_dtype"] = "float32"
        metadata[f"{key}_scale_shape"] = list(tensors[f"{key}_scale"].shape)

        tensors[f"{key}_zp"] = zero_point.to(torch.float32).contiguous()
        metadata[f"{key}_zp_dtype"] = "float32"
        metadata[f"{key}_zp_shape"] = list(tensors[f"{key}_zp"].shape)

        metadata[f"{key}_bits"] = bits
        metadata[f"{key}_original_shape"] = list(slice_tensor.shape)
        metadata[f"{key}_num_elements"] = q.numel()

    def encode_prefill(self, kv_cache: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        if torch.isnan(kv_cache).any():
            raise ValueError("ScalarQuantCodec cannot quantize a tensor containing NaN values")

        num_layers = kv_cache.size(LAYER_DIM)
        head_mode = self._has_head_overrides()

        tensors: Dict[str, torch.Tensor] = {}
        metadata: Dict[str, Any] = {
            "tensor_name": self.tensor_name,
            "granularity": self.granularity.value,
            "clip_method": self.clip_method.value,
            "symmetric": self.symmetric,
            "group_size": self.group_size,
            "full_shape": list(kv_cache.shape),
            "group_mode": "cell" if head_mode else "layer",
            "group_keys": [],
        }
        saturation_reports: Dict[str, Any] = {}

        if not head_mode:
            for bits, layer_indices in self._group_layers_by_bits(num_layers).items():
                key = f"grp{bits}b_{layer_indices[0]}"
                metadata["group_keys"].append(key)
                layer_idx_tensor = torch.tensor(layer_indices, dtype=torch.long)
                slice_tensor = kv_cache.index_select(LAYER_DIM, layer_idx_tensor)
                self._encode_group(key, slice_tensor, bits, tensors, metadata, saturation_reports)
                metadata[f"{key}_layer_indices"] = layer_indices
        else:
            num_heads = kv_cache.size(HEAD_DIM)
            for bits, cells in self._group_cells_by_bits(num_layers, num_heads).items():
                l0, h0 = cells[0]
                key = f"grp{bits}b_{l0}h{h0}"
                metadata["group_keys"].append(key)
                # Gather each (layer, head) cell as a single-head slice
                # [1, batch, 1, seq, head_dim] and stack on the layer axis, so
                # the 5D shape (and every granularity path) stays valid.
                slices = [
                    kv_cache[lyr : lyr + 1, :, hd : hd + 1, :, :] for (lyr, hd) in cells
                ]
                slice_tensor = torch.cat(slices, dim=LAYER_DIM)
                self._encode_group(key, slice_tensor, bits, tensors, metadata, saturation_reports)
                metadata[f"{key}_cells"] = [[lyr, hd] for (lyr, hd) in cells]

        metadata["saturation"] = saturation_reports

        byte_stream, accountant = BinarySerializer.serialize(self.config_hash, tensors, metadata)
        metadata["accountant_report"] = accountant.report()
        return byte_stream, metadata

    def decode(
        self, byte_stream: bytes, metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        _config_hash, meta, tensors = BinarySerializer.deserialize(byte_stream)
        full_shape = tuple(meta["full_shape"])
        output = torch.zeros(full_shape, dtype=dtype, device=device)

        symmetric = meta["symmetric"]
        granularity = Granularity(meta["granularity"])
        group_size = meta["group_size"]
        group_mode = meta.get("group_mode", "layer")

        for key in meta["group_keys"]:
            bits = meta[f"{key}_bits"]
            original_shape = tuple(meta[f"{key}_original_shape"])
            num_elements = meta[f"{key}_num_elements"]

            packed = tensors[f"{key}_data"]
            if bits == 4:
                q = unpack_int4(packed, num_elements, signed=symmetric)
            else:
                q = unpack_int8(packed)
            q = q.reshape(original_shape).to(device)

            scale = tensors[f"{key}_scale"].to(device)
            zero_point = tensors[f"{key}_zp"].to(device)
            b_scale, b_zp = broadcast_min_max(scale, zero_point, original_shape, granularity, group_size)

            recon = (q.to(torch.float32) - b_zp) * b_scale
            recon = recon.to(dtype)

            if group_mode == "cell":
                # recon is [num_cells, batch, 1, seq, head_dim]; scatter each
                # cell back to its original (layer, head) position.
                for i, (layer, head) in enumerate(meta[f"{key}_cells"]):
                    output[layer, :, head, :, :] = recon[i, :, 0, :, :]
            else:
                layer_idx_tensor = torch.tensor(meta[f"{key}_layer_indices"], dtype=torch.long, device=device)
                output.index_copy_(LAYER_DIM, layer_idx_tensor, recon)

        return output

    def encode_decode_step(self, new_token_kv: torch.Tensor, current_state: Any) -> Tuple[bytes, torch.Tensor, Any]:
        return b"", new_token_kv, current_state
