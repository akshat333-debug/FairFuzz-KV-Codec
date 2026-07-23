"""LBG vector-quantization codec.

Trains (or reuses calibration-fit) LBG codebooks and encodes a KV tensor as
codebook + nearest-codeword indices, with global / per-layer / per-head
codebook scopes. Codebook overhead is serialized and fully counted by the
ByteAccountant, so matched-total-bit comparisons against the scalar codecs are
honest (a small corpus where the codebook cost dominates will legitimately look
worse - that case is reported, not hidden).
"""

import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from fairfuzzkv_codec.codec.base import BaseCodec
from fairfuzzkv_codec.codec.binary_serializer import BinarySerializer
from fairfuzzkv_codec.quantization.vector_quant import (
    LBGDiagnostics,
    VectorPolicy,
    form_vectors,
    index_bits,
    nearest_codeword,
    train_lbg,
    unform_vectors,
)

LAYER_DIM = 0
HEAD_DIM = 2


class CodebookScope(str, Enum):
    GLOBAL = "global"
    PER_LAYER = "per_layer"
    PER_HEAD = "per_head"


class LBGVectorQuantCodec(BaseCodec):
    """One instance handles ONE of K or V (tensor_name="k"/"v"), mirroring
    ScalarQuantCodec so the two are directly swappable in benchmarks/runners."""

    def __init__(
        self,
        config_hash: str,
        tensor_name: str,
        vector_dim: int = 8,
        codebook_size: int = 256,
        policy: VectorPolicy = VectorPolicy.HEAD_BLOCK,
        token_span: int = 1,
        scope: CodebookScope = CodebookScope.GLOBAL,
        seed: int = 42,
        minibatch: Optional[int] = None,
        use_faiss: bool = False,
        max_iters: int = 50,
    ):
        if tensor_name not in ("k", "v"):
            raise ValueError(f"tensor_name must be 'k' or 'v', got {tensor_name!r}")
        if codebook_size > 256:
            # uint8 index container; larger sizes would need a wider index dtype
            # and honest multi-byte accounting - out of scope for this codec.
            raise ValueError("codebook_size must be <= 256 (uint8 index storage)")
        self.config_hash = config_hash
        self.tensor_name = tensor_name
        self.vector_dim = vector_dim
        self.codebook_size = codebook_size
        self.policy = policy
        self.token_span = token_span
        self.scope = scope
        self.seed = seed
        self.minibatch = minibatch
        self.use_faiss = use_faiss
        self.max_iters = max_iters
        # scope_key -> trained codebook, populated by fit() or lazily at encode.
        self._codebooks: Dict[str, torch.Tensor] = {}
        self.last_diagnostics: Dict[str, LBGDiagnostics] = {}

    # ---- scope partitioning -------------------------------------------------

    def _scope_slices(self, tensor: torch.Tensor) -> List[Tuple[str, torch.Tensor, Dict[str, Any]]]:
        """Return (scope_key, sub_tensor_5d, placement_meta) for each codebook
        scope. placement_meta records how to scatter the reconstruction back."""
        if self.scope == CodebookScope.GLOBAL:
            return [("g", tensor, {"kind": "global"})]
        if self.scope == CodebookScope.PER_LAYER:
            out = []
            for layer in range(tensor.size(LAYER_DIM)):
                sub = tensor[layer : layer + 1]
                out.append((f"L{layer}", sub, {"kind": "layer", "layer": layer}))
            return out
        if self.scope == CodebookScope.PER_HEAD:
            out = []
            for head in range(tensor.size(HEAD_DIM)):
                sub = tensor[:, :, head : head + 1]
                out.append((f"H{head}", sub, {"kind": "head", "head": head}))
            return out
        raise ValueError(f"unknown scope: {self.scope}")

    # ---- calibration fit (leakage-safe) -------------------------------------

    def fit(self, calibration_tensor: torch.Tensor) -> Dict[str, LBGDiagnostics]:
        """Train codebooks on calibration data ONLY. Encoding different data
        afterwards keeps the codebook independent of the encoded tensor - the
        standard guard against fitting-and-scoring on the same sample."""
        if torch.isnan(calibration_tensor).any():
            raise ValueError("LBG calibration tensor contains NaN")
        self._codebooks.clear()
        diags: Dict[str, LBGDiagnostics] = {}
        for key, sub, _meta in self._scope_slices(calibration_tensor):
            vectors, _layout = form_vectors(sub, self.vector_dim, self.policy, self.token_span)
            cb, diag = train_lbg(
                vectors, self.codebook_size, seed=self.seed,
                minibatch=self.minibatch, max_iters=self.max_iters,
            )
            self._codebooks[key] = cb
            diags[key] = diag
        self.last_diagnostics = diags
        return diags

    # ---- encode / decode ----------------------------------------------------

    def encode_prefill(self, kv_cache: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        if torch.isnan(kv_cache).any():
            raise ValueError("LBGVectorQuantCodec cannot quantize a tensor containing NaN")

        tensors: Dict[str, torch.Tensor] = {}
        metadata: Dict[str, Any] = {
            "codec": "lbg_vq",
            "tensor_name": self.tensor_name,
            "policy": self.policy.value,
            "vector_dim": self.vector_dim,
            "token_span": self.token_span,
            "codebook_size": self.codebook_size,
            "scope": self.scope.value,
            "full_shape": list(kv_cache.shape),
            "scope_keys": [],
        }
        diags: Dict[str, LBGDiagnostics] = {}

        for key, sub, place in self._scope_slices(kv_cache):
            vectors, layout = form_vectors(sub, self.vector_dim, self.policy, self.token_span)

            cb = self._codebooks.get(key)
            if cb is None:
                # No calibration fit provided: self-calibrate on this tensor and
                # embed the codebook in the bitstream (still counted as overhead).
                cb, diag = train_lbg(
                    vectors, self.codebook_size, seed=self.seed,
                    minibatch=self.minibatch, max_iters=self.max_iters,
                )
                diags[key] = diag

            idx = nearest_codeword(vectors, cb, use_faiss=self.use_faiss).to(torch.uint8)

            metadata["scope_keys"].append(key)
            metadata[f"{key}_place"] = place
            metadata[f"{key}_sub_shape"] = list(sub.shape)
            metadata[f"{key}_num_vectors"] = int(vectors.size(0))
            metadata[f"{key}_cb_version"] = _codebook_version(cb)

            # indices: one uint8 byte each on disk, but only index_bits logical.
            tensors[f"{key}_idx"] = idx.contiguous()
            metadata[f"{key}_idx_dtype"] = "uint8"
            metadata[f"{key}_idx_shape"] = [idx.numel()]
            metadata[f"{key}_idx_logical_bits_per_element"] = float(index_bits(self.codebook_size))

            # codebook: real float32 overhead, fully counted.
            tensors[f"{key}_cb"] = cb.to(torch.float32).contiguous()
            metadata[f"{key}_cb_dtype"] = "float32"
            metadata[f"{key}_cb_shape"] = list(cb.shape)

        byte_stream, accountant = BinarySerializer.serialize(self.config_hash, tensors, metadata)
        # Diagnostics (which include wall-clock train_seconds) are attached to
        # the RETURNED metadata only, never into the serialized bytes - keeps
        # the bitstream deterministic under a fixed seed. Decode never needs them.
        if diags:
            self.last_diagnostics = diags
            metadata["diagnostics"] = {k: d.to_dict() for k, d in diags.items()}
        metadata["accountant_report"] = accountant.report()
        return byte_stream, metadata

    def decode(
        self, byte_stream: bytes, metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        _config_hash, meta, tensors = BinarySerializer.deserialize(byte_stream)
        full_shape = tuple(meta["full_shape"])
        output = torch.zeros(full_shape, dtype=dtype, device=device)

        policy = VectorPolicy(meta["policy"])
        vector_dim = meta["vector_dim"]
        token_span = meta["token_span"]

        for key in meta["scope_keys"]:
            sub_shape = tuple(meta[f"{key}_sub_shape"])
            idx = tensors[f"{key}_idx"].to(torch.long).to(device)
            cb = tensors[f"{key}_cb"].to(device).to(torch.float32)
            recon_vectors = cb[idx]

            from fairfuzzkv_codec.quantization.vector_quant import VectorLayout

            L, B, H, S, D = sub_shape
            layout = VectorLayout(policy, sub_shape, vector_dim, token_span, {"nblk": D // vector_dim})
            recon = unform_vectors(recon_vectors, layout).to(dtype)

            place = meta[f"{key}_place"]
            if place["kind"] == "global":
                output.copy_(recon)
            elif place["kind"] == "layer":
                output[place["layer"] : place["layer"] + 1] = recon
            elif place["kind"] == "head":
                output[:, :, place["head"] : place["head"] + 1] = recon
            else:
                raise ValueError(f"unknown placement kind: {place['kind']}")

        return output

    def encode_decode_step(self, new_token_kv: torch.Tensor, current_state: Any) -> Tuple[bytes, torch.Tensor, Any]:
        return b"", new_token_kv, current_state


def _codebook_version(codebook: torch.Tensor) -> str:
    """Short content hash of a codebook, for bitstream integrity / provenance."""
    return hashlib.sha256(codebook.to(torch.float32).contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
