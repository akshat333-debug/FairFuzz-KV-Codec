"""Linde-Buzo-Gray (LBG) vector quantization core.

Vector formation, deterministic LBG/k-means codebook training, nearest-codeword
encoding, and diagnostics. This is the CPU reference implementation and is
authoritative - an optional FAISS acceleration path lives behind
`nearest_codeword(..., use_faiss=True)` but never changes results, only speed.

Shape convention everywhere: KV tensors are [layers, batch, heads, seq, head_dim].
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import torch

CHANNEL_DIM = 4
SEQ_DIM = 3


class VectorPolicy(str, Enum):
    HEAD_BLOCK = "head_block"  # contiguous head-dim blocks of length vector_dim
    CROSS_TOKEN = "cross_token"  # head-dim block spanning token_span consecutive tokens


@dataclass
class LBGDiagnostics:
    codebook_size: int
    vector_dim: int
    num_vectors: int
    used_codewords: int
    dead_codewords: int
    utilization: float
    final_mse: float
    iterations: int
    converged: bool
    train_seconds: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "codebook_size": self.codebook_size,
            "vector_dim": self.vector_dim,
            "num_vectors": self.num_vectors,
            "used_codewords": self.used_codewords,
            "dead_codewords": self.dead_codewords,
            "utilization": self.utilization,
            "final_mse": self.final_mse,
            "iterations": self.iterations,
            "converged": self.converged,
            "train_seconds": self.train_seconds,
        }


@dataclass
class VectorLayout:
    """Everything needed to invert form_vectors back to the original tensor."""

    policy: VectorPolicy
    original_shape: Tuple[int, ...]
    vector_dim: int
    token_span: int = 1
    extra: Dict[str, int] = field(default_factory=dict)


def form_vectors(
    tensor: torch.Tensor,
    vector_dim: int,
    policy: VectorPolicy = VectorPolicy.HEAD_BLOCK,
    token_span: int = 1,
) -> Tuple[torch.Tensor, VectorLayout]:
    """Turn a 5D KV tensor into a 2D [num_vectors, dim] matrix.

    HEAD_BLOCK: split head_dim into contiguous blocks of `vector_dim`; each
    block is one vector. dim == vector_dim.

    CROSS_TOKEN: same head-dim block, but concatenated across `token_span`
    consecutive token positions, so a vector captures short-range temporal
    structure. dim == vector_dim * token_span.
    """
    if tensor.dim() != 5:
        raise ValueError(f"expected 5D [L,B,H,S,D] tensor, got shape {tuple(tensor.shape)}")
    L, B, H, S, D = tensor.shape
    if D % vector_dim != 0:
        raise ValueError(f"head_dim ({D}) must be divisible by vector_dim ({vector_dim})")

    if policy == VectorPolicy.HEAD_BLOCK:
        # [L,B,H,S, D//vd, vd] -> [-1, vd]
        vectors = tensor.reshape(L, B, H, S, D // vector_dim, vector_dim).reshape(-1, vector_dim)
        return vectors.contiguous(), VectorLayout(policy, (L, B, H, S, D), vector_dim, 1)

    if policy == VectorPolicy.CROSS_TOKEN:
        if token_span < 1:
            raise ValueError("token_span must be >= 1")
        if S % token_span != 0:
            raise ValueError(f"seq_len ({S}) must be divisible by token_span ({token_span})")
        # [L,B,H, S//ts, ts, D//vd, vd] -> group (ts, vd) into one vector of ts*vd.
        nblk = D // vector_dim
        t = tensor.reshape(L, B, H, S // token_span, token_span, nblk, vector_dim)
        # move channel-block axis next to the leading dims, keep (token_span, vector_dim) contiguous
        t = t.permute(0, 1, 2, 3, 5, 4, 6).contiguous()  # [L,B,H,S//ts,nblk,ts,vd]
        vectors = t.reshape(-1, token_span * vector_dim)
        return vectors.contiguous(), VectorLayout(
            policy, (L, B, H, S, D), vector_dim, token_span, {"nblk": nblk}
        )

    raise ValueError(f"unknown vector policy: {policy}")


def unform_vectors(vectors: torch.Tensor, layout: VectorLayout) -> torch.Tensor:
    """Exact inverse of form_vectors - restores original [L,B,H,S,D] shape/order."""
    L, B, H, S, D = layout.original_shape
    vd = layout.vector_dim
    if layout.policy == VectorPolicy.HEAD_BLOCK:
        return vectors.reshape(L, B, H, S, D // vd, vd).reshape(L, B, H, S, D)
    if layout.policy == VectorPolicy.CROSS_TOKEN:
        ts = layout.token_span
        nblk = D // vd
        t = vectors.reshape(L, B, H, S // ts, nblk, ts, vd)
        t = t.permute(0, 1, 2, 3, 5, 4, 6).contiguous()  # back to [L,B,H,S//ts,ts,nblk,vd]
        return t.reshape(L, B, H, S, D)
    raise ValueError(f"unknown vector policy: {layout.policy}")


def nearest_codeword(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    chunk_size: int = 4096,
    use_faiss: bool = False,
) -> torch.Tensor:
    """Assign each vector to its nearest codeword (squared-L2). Distances are
    computed in row-chunks so the N x K distance matrix never materializes in
    full for large caches. FAISS is used only if explicitly requested AND
    importable; the CPU torch path is authoritative and always available."""
    if use_faiss:
        idx = _faiss_nearest(vectors, codebook)
        if idx is not None:
            return idx
    n = vectors.size(0)
    out = torch.empty(n, dtype=torch.long, device=vectors.device)
    cb_sq = (codebook * codebook).sum(dim=1)  # [K]
    for start in range(0, n, chunk_size):
        chunk = vectors[start : start + chunk_size]
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 is constant per row.
        dist = cb_sq.unsqueeze(0) - 2.0 * (chunk @ codebook.t())
        out[start : start + chunk_size] = dist.argmin(dim=1)
    return out


def _faiss_nearest(vectors: torch.Tensor, codebook: torch.Tensor) -> Optional[torch.Tensor]:
    try:
        import faiss  # type: ignore
    except ImportError:
        return None
    import numpy as np

    index = faiss.IndexFlatL2(codebook.size(1))
    index.add(codebook.detach().cpu().numpy().astype(np.float32))
    _dist, idx = index.search(vectors.detach().cpu().numpy().astype(np.float32), 1)
    return torch.from_numpy(idx[:, 0].astype(np.int64)).to(vectors.device)


def _lloyd_step(
    vectors: torch.Tensor, codebook: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One assignment + centroid-update pass. Returns (new_codebook, assignments)."""
    idx = nearest_codeword(vectors, codebook, chunk_size=chunk_size)
    k, d = codebook.shape
    sums = torch.zeros(k, d, dtype=vectors.dtype, device=vectors.device)
    sums.index_add_(0, idx, vectors)
    counts = torch.bincount(idx, minlength=k).to(vectors.dtype)
    nonempty = counts > 0
    new_cb = codebook.clone()
    new_cb[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    return new_cb, idx


def _recover_empty_clusters(
    codebook: torch.Tensor, vectors: torch.Tensor, idx: torch.Tensor, eps: float
) -> torch.Tensor:
    """Reseed any empty codeword by splitting the most-populated one - keeps the
    codebook fully utilized instead of leaving dead entries mid-training."""
    k = codebook.size(0)
    counts = torch.bincount(idx, minlength=k)
    empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if empty.numel() == 0:
        return codebook
    cb = codebook.clone()
    for e in empty.tolist():
        biggest = int(torch.argmax(counts).item())
        cb[e] = cb[biggest] * (1.0 + eps)
        cb[biggest] = cb[biggest] * (1.0 - eps)
        # split the biggest in half so we don't keep reseeding from the same one
        counts[biggest] = counts[biggest] // 2
        counts[e] = counts[biggest]
    return cb


def train_lbg(
    vectors: torch.Tensor,
    codebook_size: int,
    seed: int = 42,
    max_iters: int = 50,
    tol: float = 1e-4,
    split_eps: float = 0.01,
    minibatch: Optional[int] = None,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, LBGDiagnostics]:
    """Deterministic LBG codebook training.

    Starts from the global centroid and repeatedly SPLITS every centroid
    (c -> c(1+eps), c(1-eps)), running Lloyd iterations after each split until
    the codebook reaches `codebook_size` (which must be a power of two).
    Deterministic under fixed `seed`; `minibatch` subsamples vectors per Lloyd
    step for large caches (sampling is seeded, so still deterministic).
    """
    import time

    t0 = time.perf_counter()
    if codebook_size < 1 or (codebook_size & (codebook_size - 1)) != 0:
        raise ValueError(f"codebook_size must be a power of two, got {codebook_size}")
    if vectors.dim() != 2:
        raise ValueError("vectors must be 2D [num_vectors, dim]")
    if vectors.size(0) < codebook_size:
        raise ValueError(
            f"need at least codebook_size ({codebook_size}) vectors, got {vectors.size(0)}"
        )

    vectors = vectors.to(torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def _draw(v: torch.Tensor) -> torch.Tensor:
        if minibatch is None or minibatch >= v.size(0):
            return v
        perm = torch.randperm(v.size(0), generator=generator)[:minibatch]
        return v[perm]

    codebook = vectors.mean(dim=0, keepdim=True)  # [1, d]
    iters = 0
    converged = False
    while codebook.size(0) < codebook_size:
        # split every centroid
        codebook = torch.cat([codebook * (1.0 + split_eps), codebook * (1.0 - split_eps)], dim=0)
        prev_mse = None
        for _ in range(max_iters):
            batch = _draw(vectors)
            codebook, idx = _lloyd_step(batch, codebook, chunk_size)
            codebook = _recover_empty_clusters(codebook, batch, idx, split_eps)
            iters += 1
            recon = codebook[idx]
            mse = (recon - batch).pow(2).mean().item()
            if prev_mse is not None and prev_mse > 0 and abs(prev_mse - mse) / prev_mse < tol:
                converged = True
                break
            prev_mse = mse

    # final full-data assignment for honest diagnostics
    final_idx = nearest_codeword(vectors, codebook, chunk_size=chunk_size)
    final_recon = codebook[final_idx]
    final_mse = (final_recon - vectors).pow(2).mean().item()
    used = int(torch.unique(final_idx).numel())
    diag = LBGDiagnostics(
        codebook_size=codebook_size,
        vector_dim=vectors.size(1),
        num_vectors=vectors.size(0),
        used_codewords=used,
        dead_codewords=codebook_size - used,
        utilization=used / codebook_size,
        final_mse=final_mse,
        iterations=iters,
        converged=converged,
        train_seconds=time.perf_counter() - t0,
    )
    return codebook.contiguous(), diag


def index_bits(codebook_size: int) -> int:
    """Logical bits per index = ceil(log2(codebook_size))."""
    return max(1, (codebook_size - 1).bit_length())
