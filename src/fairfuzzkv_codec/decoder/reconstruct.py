"""High-level encode-to-container / decode-from-container bridge.

Wraps a scalar or LBG codec payload in an FFK1 container (geometry header,
tokenizer hash, checksums, section directory) and reconstructs the cache tensor
from it, dispatching to the right codec by the recorded codec name. Both scalar
and LBG payloads round-trip through exactly this path.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from fairfuzzkv_codec.codec.base import BaseCodec
from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
from fairfuzzkv_codec.codec.vector_quant import LBGVectorQuantCodec
from fairfuzzkv_codec.metadata_coding.container import (
    SECTION_CODEC_PAYLOAD,
    SECTION_RETENTION,
    Container,
    Section,
    pack,
    unpack,
)

SCALAR_CODEC_NAME = "scalar_quant"
LBG_CODEC_NAME = "lbg_vq"


def _codec_name(codec: BaseCodec) -> str:
    if isinstance(codec, LBGVectorQuantCodec):
        return LBG_CODEC_NAME
    if isinstance(codec, ScalarQuantCodec):
        return SCALAR_CODEC_NAME
    raise ValueError(f"unsupported codec type for container: {type(codec).__name__}")


def encode_to_container(
    codec: BaseCodec,
    kv_cache: torch.Tensor,
    tokenizer_hash: str = "",
    retention_payload: Optional[bytes] = None,
) -> bytes:
    """Encode a KV tensor with `codec` and wrap it in an FFK1 container. The
    geometry header records cache shape, tokenizer hash, and codec name so the
    decoder is fully self-describing."""
    payload, meta = codec.encode_prefill(kv_cache)
    shape = list(kv_cache.shape)
    geometry: Dict[str, Any] = {
        "format": "FFK1",
        "codec_name": _codec_name(codec),
        "geometry": {
            "layers": shape[0], "batch": shape[1], "heads": shape[2],
            "seq": shape[3], "head_dim": shape[4],
        },
        "tokenizer_hash": tokenizer_hash,
        "tensor_name": meta.get("tensor_name", ""),
        "logical_bits": meta["accountant_report"]["logical_bits"],
    }
    sections = [Section(SECTION_CODEC_PAYLOAD, payload)]
    if retention_payload is not None:
        sections.append(Section(SECTION_RETENTION, retention_payload))
    return pack(geometry, sections)


@dataclass
class CompletenessReport:
    codec_name: str
    expected_shape: Tuple[int, ...]
    actual_shape: Tuple[int, ...]
    shape_ok: bool
    has_codec_payload: bool
    has_retention: bool
    num_sections: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "codec_name": self.codec_name,
            "expected_shape": list(self.expected_shape),
            "actual_shape": list(self.actual_shape),
            "shape_ok": self.shape_ok,
            "has_codec_payload": self.has_codec_payload,
            "has_retention": self.has_retention,
            "num_sections": self.num_sections,
        }


def _build_decoder(codec_name: str) -> BaseCodec:
    # decode() reads everything it needs from the payload metadata, so the
    # instance config here is irrelevant - construct a minimal valid instance.
    if codec_name == LBG_CODEC_NAME:
        return LBGVectorQuantCodec("", tensor_name="k")
    if codec_name == SCALAR_CODEC_NAME:
        return ScalarQuantCodec("", tensor_name="k")
    raise ValueError(f"unknown codec name in container: {codec_name}")


def decode_from_container(
    data: bytes, dtype: torch.dtype = torch.float32, device: str = "cpu"
) -> Tuple[torch.Tensor, CompletenessReport]:
    """Parse + validate the container, reconstruct the cache tensor, and return
    it alongside a completeness diagnostic."""
    container: Container = unpack(data)
    geo = container.geometry
    codec_name = str(geo["codec_name"])
    g = geo["geometry"]  # type: ignore[index]
    expected_shape = (g["layers"], g["batch"], g["heads"], g["seq"], g["head_dim"])  # type: ignore[index]

    payload = container.get(SECTION_CODEC_PAYLOAD)
    codec = _build_decoder(codec_name)
    tensor = codec.decode(payload, {}, expected_shape, dtype, device)

    report = CompletenessReport(
        codec_name=codec_name,
        expected_shape=expected_shape,
        actual_shape=tuple(tensor.shape),
        shape_ok=tuple(tensor.shape) == expected_shape,
        has_codec_payload=container.has(SECTION_CODEC_PAYLOAD),
        has_retention=container.has(SECTION_RETENTION),
        num_sections=len(container.sections),
    )
    return tensor, report
