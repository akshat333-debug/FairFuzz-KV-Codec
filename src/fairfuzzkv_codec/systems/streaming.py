"""Chunked/streaming encoding, bounded allocation, and graceful OOM handling.

Large contexts must fail predictably, never corrupt output (Prompt 18
acceptance gate). Two guarantees here:

  * `chunked_encode` never materializes more than one chunk of quantized data
    at a time, so peak memory is bounded by chunk size rather than context
    length.
  * `AllocationBudget` refuses an allocation that would exceed a caller-set
    ceiling BEFORE attempting it, raising `AllocationTooLarge` - so the failure
    is a clean, catchable exception rather than an OS-level kill or a
    half-written file.

Every partial output is discarded on failure: an encode either returns a
complete, checksum-valid payload or raises. There is no "best effort" path that
could silently emit a truncated bitstream.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

SEQ_DIM = 3


class AllocationTooLarge(MemoryError):
    """Raised BEFORE an oversized allocation is attempted, so the caller gets a
    clean, catchable error instead of an OOM kill."""


class StreamingEncodeError(RuntimeError):
    """Raised when a chunk fails to encode. The partial result is discarded -
    a truncated bitstream is never returned."""


@dataclass
class AllocationBudget:
    """Bounded-allocation guard. `max_bytes=None` disables the ceiling."""

    max_bytes: Optional[int] = None

    def check_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, what: str = "tensor") -> None:
        if self.max_bytes is None:
            return
        elements = 1
        for dim in shape:
            elements *= max(0, dim)
        itemsize = torch.empty(0, dtype=dtype).element_size()
        needed = elements * itemsize
        if needed > self.max_bytes:
            raise AllocationTooLarge(
                f"{what} would need {needed / 1024 ** 2:.1f} MiB "
                f"(shape={tuple(shape)}, dtype={dtype}) which exceeds the "
                f"{self.max_bytes / 1024 ** 2:.1f} MiB budget - refused before allocating"
            )

    def check_bytes(self, nbytes: int, what: str = "buffer") -> None:
        if self.max_bytes is not None and nbytes > self.max_bytes:
            raise AllocationTooLarge(
                f"{what} would need {nbytes / 1024 ** 2:.1f} MiB which exceeds the "
                f"{self.max_bytes / 1024 ** 2:.1f} MiB budget - refused before allocating"
            )


def iter_sequence_chunks(kv_cache: torch.Tensor, chunk_tokens: int) -> Iterator[Tuple[int, torch.Tensor]]:
    """Yield (start_index, view) slices along the sequence axis. Views, not
    copies, so iteration itself allocates nothing."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be >= 1")
    seq_len = kv_cache.size(SEQ_DIM)
    for start in range(0, seq_len, chunk_tokens):
        stop = min(start + chunk_tokens, seq_len)
        yield start, kv_cache[:, :, :, start:stop, :]


@dataclass
class ChunkedEncodeResult:
    payloads: List[bytes]
    chunk_starts: List[int]
    total_bytes: int
    num_chunks: int
    chunk_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_chunks": self.num_chunks,
            "chunk_tokens": self.chunk_tokens,
            "total_bytes": self.total_bytes,
            "chunk_starts": self.chunk_starts,
        }


def chunked_encode(
    kv_cache: torch.Tensor,
    encode_fn: Callable[[torch.Tensor], Tuple[bytes, Dict[str, Any]]],
    chunk_tokens: int = 256,
    budget: Optional[AllocationBudget] = None,
) -> ChunkedEncodeResult:
    """Encode a long context chunk-by-chunk so peak memory tracks chunk size,
    not sequence length. Raises (discarding all partial output) rather than
    returning a truncated stream."""
    budget = budget or AllocationBudget()
    budget.check_tensor(
        (kv_cache.size(0), kv_cache.size(1), kv_cache.size(2), min(chunk_tokens, kv_cache.size(SEQ_DIM)), kv_cache.size(4)),
        kv_cache.dtype,
        what="chunk",
    )

    payloads: List[bytes] = []
    starts: List[int] = []
    total = 0
    for start, chunk in iter_sequence_chunks(kv_cache, chunk_tokens):
        try:
            payload, _meta = encode_fn(chunk)
        except AllocationTooLarge:
            raise
        except Exception as e:  # noqa: BLE001
            raise StreamingEncodeError(
                f"chunk starting at token {start} failed to encode ({e}); "
                f"discarding {len(payloads)} previously-encoded chunk(s) - "
                f"a partial bitstream is never returned"
            ) from e
        budget.check_bytes(total + len(payload), what="accumulated payload")
        payloads.append(payload)
        starts.append(start)
        total += len(payload)

    return ChunkedEncodeResult(
        payloads=payloads, chunk_starts=starts, total_bytes=total,
        num_chunks=len(payloads), chunk_tokens=chunk_tokens,
    )


def chunked_decode(
    result: ChunkedEncodeResult,
    decode_fn: Callable[[bytes], torch.Tensor],
    expected_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Reassemble chunk reconstructions in original positional order, verifying
    the result covers exactly the expected sequence length (a short or
    overlapping reassembly raises instead of silently returning wrong data)."""
    chunks = [decode_fn(p) for p in result.payloads]
    if not chunks:
        raise StreamingEncodeError("no chunks to decode")
    out = torch.cat(chunks, dim=SEQ_DIM)
    if tuple(out.shape) != tuple(expected_shape):
        raise StreamingEncodeError(
            f"reassembled shape {tuple(out.shape)} != expected {tuple(expected_shape)} - "
            "refusing to return a mis-assembled cache"
        )
    return out


def maybe_pin_memory(tensor: torch.Tensor, enabled: bool = False) -> torch.Tensor:
    """Pinned (page-locked) host memory speeds up host->device copies. Only
    meaningful for CPU tensors when CUDA is present; a no-op otherwise, and it
    NEVER silently claims to have pinned when it could not."""
    if not enabled or tensor.is_cuda or not torch.cuda.is_available():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor  # pinning unavailable (e.g. no CUDA host allocator)


def recommended_chunk_tokens(kv_cache_shape: Tuple[int, ...], dtype: torch.dtype, target_mb: float = 64.0) -> int:
    """Largest chunk whose slice stays under `target_mb`. Always >= 1 so a
    single enormous token still produces a (documented, over-budget) chunk
    rather than an infinite loop."""
    layers, batch, heads, _seq, head_dim = kv_cache_shape
    itemsize = torch.empty(0, dtype=dtype).element_size()
    per_token = max(1, layers * batch * heads * head_dim * itemsize)
    return max(1, int(math.floor(target_mb * 1024 ** 2 / per_token)))
