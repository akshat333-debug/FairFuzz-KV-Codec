import torch
from typing import Tuple, Dict, Any
from fairfuzzkv_codec.codec.base import BaseCodec
from fairfuzzkv_codec.codec.binary_serializer import BinarySerializer
from fairfuzzkv_codec.quantization.uniform import compute_scales_and_zeropoints, quantize_uniform, dequantize_uniform
from fairfuzzkv_codec.pruning.topk import compute_topk_mask, apply_topk_pruning

class FullKVFP16Codec(BaseCodec):
    def __init__(self, config_hash: str):
        self.config_hash = config_hash

    def encode_prefill(self, kv_cache: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        tensor_fp16 = kv_cache.to(torch.float16)
        tensors = {"kv": tensor_fp16}
        metadata: Dict[str, Any] = {
            "kv_shape": list(tensor_fp16.shape),
            "kv_dtype": "float16",
        }
        
        byte_stream, accountant = BinarySerializer.serialize(self.config_hash, tensors, metadata)
        metadata["accountant_report"] = accountant.report()
        return byte_stream, metadata

    def decode(self, byte_stream: bytes, metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
        config_hash, meta, tensors = BinarySerializer.deserialize(byte_stream)
        return tensors["kv"].to(device=device, dtype=dtype)

    def encode_decode_step(self, new_token_kv: torch.Tensor, current_state: Any) -> Tuple[bytes, torch.Tensor, Any]:
        return b"", new_token_kv, current_state


class UniformQuantCodec(BaseCodec):
    def __init__(self, config_hash: str, num_bits: int, symmetric: bool = True, per_channel: bool = True):
        self.config_hash = config_hash
        self.num_bits = num_bits
        self.symmetric = symmetric
        # If per_channel, we assume channel is the last dimension (head_dim)
        self.dim = -1 if per_channel else None

    def encode_prefill(self, kv_cache: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        scale, zp = compute_scales_and_zeropoints(kv_cache, self.num_bits, self.symmetric, self.dim)
        q_tensor = quantize_uniform(kv_cache, self.num_bits, scale, zp, self.symmetric)
        
        tensors = {
            "kv": q_tensor,
            "scale": scale.to(torch.float32),
            "zp": zp.to(torch.float32)
        }
        
        metadata: Dict[str, Any] = {
            "kv_shape": list(q_tensor.shape),
            "kv_dtype": "int8",
            "kv_logical_bits_per_element": self.num_bits,
            "scale_shape": list(scale.shape),
            "scale_dtype": "float32",
            "zp_shape": list(zp.shape),
            "zp_dtype": "float32",
            "symmetric": self.symmetric,
            "num_bits": self.num_bits
        }
        
        byte_stream, accountant = BinarySerializer.serialize(self.config_hash, tensors, metadata)
        metadata["accountant_report"] = accountant.report()
        return byte_stream, metadata

    def decode(self, byte_stream: bytes, metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
        config_hash, meta, tensors = BinarySerializer.deserialize(byte_stream)
        
        q_tensor = tensors["kv"].to(device)
        scale = tensors["scale"].to(device)
        zp = tensors["zp"].to(device)
        
        deq_tensor = dequantize_uniform(q_tensor, scale, zp, dtype=dtype)
        return deq_tensor

    def encode_decode_step(self, new_token_kv: torch.Tensor, current_state: Any) -> Tuple[bytes, torch.Tensor, Any]:
        return b"", new_token_kv, current_state


class TopKCodec(BaseCodec):
    def __init__(self, config_hash: str, retention_ratio: float):
        self.config_hash = config_hash
        self.retention_ratio = retention_ratio

    def encode_prefill(self, kv_cache: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        mask = compute_topk_mask(kv_cache, self.retention_ratio)
        
        # For simplicity in this baseline, we store the full tensor but only account 
        # for the retained tokens logically. A true sparse tensor format would pack this.
        # This is a baseline, so we zero out and store full, but report true logical overhead.
        pruned_tensor = apply_topk_pruning(kv_cache, mask).to(torch.float16)
        
        tensors = {"kv": pruned_tensor}
        metadata: Dict[str, Any] = {
            "kv_shape": list(pruned_tensor.shape),
            "kv_dtype": "float16",
            "retention_ratio": self.retention_ratio,
            "kv_logical_bits_per_element": 16.0 * self.retention_ratio # Logical overhead is proportional to retention
        }
        
        byte_stream, accountant = BinarySerializer.serialize(self.config_hash, tensors, metadata)
        # This baseline stores pruned tokens densely (zeroed) rather than packing
        # a real sparse mask/index list, so no mask bytes are actually written to
        # byte_stream. Track the theoretical logical cost (1 bit per token for the
        # mask) without inflating serialized_bytes beyond what was really written.
        num_tokens = kv_cache.size(0) * kv_cache.size(1) * kv_cache.size(2) * kv_cache.size(3)
        accountant.add_logical_only_component("mask_overhead", num_tokens)

        metadata["accountant_report"] = accountant.report()
        return byte_stream, metadata

    def decode(self, byte_stream: bytes, metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
        config_hash, meta, tensors = BinarySerializer.deserialize(byte_stream)
        return tensors["kv"].to(device=device, dtype=dtype)

    def encode_decode_step(self, new_token_kv: torch.Tensor, current_state: Any) -> Tuple[bytes, torch.Tensor, Any]:
        return b"", new_token_kv, current_state
