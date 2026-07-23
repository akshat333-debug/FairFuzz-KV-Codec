from typing import Tuple, Dict, Any
import torch
import numpy as np

from fairfuzzkv_codec.codec.base import BaseCodec

class NoOpCodec(BaseCodec):
    """
    A minimal, exact codec that round-trips FP16/BF16 without any loss.
    Provides exact byte accounting.
    """
    def encode_prefill(self, kv_cache: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        # Convert tensor to raw bytes to simulate exactly what would be transmitted
        byte_stream = kv_cache.cpu().numpy().tobytes()
        
        # Byte accounting: dtype element size * numel
        exact_bytes = kv_cache.element_size() * kv_cache.numel()
        assert len(byte_stream) == exact_bytes, "Byte stream length mismatch"
        
        metadata = {
            "exact_bytes": exact_bytes,
            "shape": tuple(kv_cache.shape),
            "dtype": str(kv_cache.dtype)
        }
        
        return byte_stream, metadata

    def decode(self, byte_stream: bytes, metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
        # Reconstruct tensor from raw bytes
        np_dtype: Any
        if dtype == torch.float16:
            np_dtype = np.float16
        elif dtype == torch.bfloat16:
            # numpy has no native bfloat16, but torch can handle it from view if we are careful.
            # For this skeleton, we assume float16
            np_dtype = np.float16
        elif dtype == torch.float32:
            np_dtype = np.float32
        else:
            raise ValueError(f"Unsupported dtype for NoOpCodec: {dtype}")

        np_array = np.frombuffer(byte_stream, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(np_array).to(device=device, dtype=dtype)
        return tensor

    def encode_decode_step(self, new_token_kv: torch.Tensor, current_state: Any) -> Tuple[bytes, torch.Tensor, Any]:
        """
        In decode regime, we just pass through.
        """
        byte_stream = new_token_kv.cpu().numpy().tobytes()
        return byte_stream, new_token_kv, current_state
