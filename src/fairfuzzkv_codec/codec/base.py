import abc
from typing import Tuple, Dict, Any
import torch

class BaseCodec(abc.ABC):
    @abc.abstractmethod
    def encode_prefill(self, kv_cache: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """
        Encode the entire prefill context.
        Returns the compressed byte stream and allocation/metadata maps.
        """
        pass

    @abc.abstractmethod
    def decode(self, byte_stream: bytes, metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
        """
        Reconstruct the KV-cache from bytes.
        """
        pass

    @abc.abstractmethod
    def encode_decode_step(self, new_token_kv: torch.Tensor, current_state: Any) -> Tuple[bytes, torch.Tensor, Any]:
        """
        Autoregressive decode step. Compress the new token and return the updated state/bytes.
        """
        pass
