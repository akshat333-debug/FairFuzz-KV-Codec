import torch

from fairfuzzkv_codec.codec.base import BaseCodec
from fairfuzzkv_codec.codec.baselines import UniformQuantCodec, TopKCodec

class MatchedBitEvaluator:
    """
    Evaluator that strictly enforces a matched-total-bit budget for compression comparisons.
    Given a target bits/element (CodecBudget.total_bits_per_element), it tunes the codec
    parameters until the serialized size falls within tolerance. Refuses comparisons if
    unable to match.
    """
    def __init__(self, target_bits_per_token: float, tolerance: float = 0.05):
        self.target_bits = target_bits_per_token
        self.tolerance = tolerance

    def _eval_size(self, codec: BaseCodec, kv_cache: torch.Tensor) -> float:
        """Returns the actual logical bits per scalar element based on byte accountant report."""
        byte_stream, meta = codec.encode_prefill(kv_cache)
        accountant_report = meta["accountant_report"]

        # total logical bits / total scalar elements (layers * batch * heads * seq * head_dim)
        actual_bits_per_element = accountant_report["logical_bits"] / kv_cache.numel()
        return actual_bits_per_element

    def tune_topk(self, kv_cache: torch.Tensor, config_hash: str) -> TopKCodec:
        """Binary search for retention ratio."""
        low, high = 0.01, 1.0
        best_codec = None
        best_diff = float('inf')
        
        # 10 iterations of binary search is usually enough for 0.05 tolerance
        for _ in range(10):
            mid = (low + high) / 2
            codec = TopKCodec(config_hash, retention_ratio=mid)
            actual_bpt = self._eval_size(codec, kv_cache)
            
            diff = abs(actual_bpt - self.target_bits)
            if diff < best_diff:
                best_diff = diff
                best_codec = codec
                
            if actual_bpt > self.target_bits:
                high = mid
            else:
                low = mid
                
        # Final check
        if best_codec is None:
            raise ValueError(f"Failed to tune TopK to {self.target_bits} bpt")
            
        final_bpt = self._eval_size(best_codec, kv_cache)
        if abs(final_bpt - self.target_bits) / self.target_bits > self.tolerance:
            raise ValueError(f"Matched-bit constraint failed: target={self.target_bits}, actual={final_bpt}")
            
        return best_codec

    def check_quant_codec(self, kv_cache: torch.Tensor, config_hash: str, bits: int) -> UniformQuantCodec:
        """For uniform quantization, bits are discrete. Check if it matches."""
        codec = UniformQuantCodec(config_hash, num_bits=bits)
        actual_bpt = self._eval_size(codec, kv_cache)
        
        if abs(actual_bpt - self.target_bits) / self.target_bits > self.tolerance:
            raise ValueError(f"Matched-bit constraint failed: target={self.target_bits}, actual={actual_bpt} (for {bits}-bit quant)")
        return codec
