import torch
from typing import Tuple, Dict, Any

from fairfuzzkv_codec.codec.base import BaseCodec
from fairfuzzkv_codec.codec.binary_serializer import BinarySerializer
from fairfuzzkv_codec.pruning.topk import apply_topk_pruning


class ExplicitMaskCodec(BaseCodec):
    """Same storage/accounting convention as `baselines.TopKCodec` (dense
    zeroed storage, logical retention-ratio accounting, no packed sparse
    index list), but the retention mask is supplied by the CALLER instead of
    being computed internally via L2-norm top-k. Lets Gate 4 (Prompt 14)
    compare pruning masks chosen by different repair-priority scorers while
    keeping the exact same byte-accounting rules TopKCodec already uses -
    so results are directly comparable to the project's existing baselines."""

    def __init__(self, config_hash: str, keep_mask: torch.Tensor):
        self.config_hash = config_hash
        self.keep_mask = keep_mask  # bool, 1D over the sequence axis (position-level, shared across layers/heads/batch)

    def _broadcast_mask(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """`apply_topk_pruning` expects a mask shaped like `compute_topk_mask`'s
        output: broadcastable over [layers, batch, heads, seq, head_dim]. A
        plain 1D [seq] mask reshapes to [1,1,1,seq,1]."""
        seq_len = self.keep_mask.shape[-1]
        return self.keep_mask.reshape([1] * (kv_cache.dim() - 2) + [seq_len, 1])

    def encode_prefill(self, kv_cache: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        mask = self._broadcast_mask(kv_cache)
        pruned_tensor = apply_topk_pruning(kv_cache, mask).to(torch.float16)
        retention_ratio = float(self.keep_mask.float().mean().item())

        tensors = {"kv": pruned_tensor}
        metadata: Dict[str, Any] = {
            "kv_shape": list(pruned_tensor.shape),
            "kv_dtype": "float16",
            "retention_ratio": retention_ratio,
            "kv_logical_bits_per_element": 16.0 * retention_ratio,
        }

        byte_stream, accountant = BinarySerializer.serialize(self.config_hash, tensors, metadata)
        num_tokens = kv_cache.numel() // kv_cache.size(-1)
        accountant.add_logical_only_component("mask_overhead", num_tokens)
        metadata["accountant_report"] = accountant.report()
        return byte_stream, metadata

    def decode(self, byte_stream: bytes, metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
        config_hash, meta, tensors = BinarySerializer.deserialize(byte_stream)
        return tensors["kv"].to(device=device, dtype=dtype)

    def encode_decode_step(self, new_token_kv: torch.Tensor, current_state: Any) -> Tuple[bytes, torch.Tensor, Any]:
        return b"", new_token_kv, current_state
