import pytest
import torch

from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
from fairfuzzkv_codec.codec.vector_quant import LBGVectorQuantCodec
from fairfuzzkv_codec.decoder import decode_from_container, encode_to_container
from fairfuzzkv_codec.metadata_coding.container import CorruptContainerError
from fairfuzzkv_codec.metadata_coding.golomb_rice import decode_retention, encode_retention


def _sample_kv(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(3, 1, 2, 12, 16, generator=g)


def test_scalar_payload_round_trips_through_container():
    K = _sample_kv()
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    data = encode_to_container(codec, K, tokenizer_hash="abc123")
    recon, report = decode_from_container(data)
    assert report.codec_name == "scalar_quant"
    assert report.shape_ok
    assert recon.shape == K.shape
    # 8-bit scalar is near-lossless on this data.
    assert (recon - K).pow(2).mean().item() < 1e-2


def test_lbg_payload_round_trips_through_container():
    K = _sample_kv()
    codec = LBGVectorQuantCodec("h", tensor_name="k", vector_dim=8, codebook_size=64)
    data = encode_to_container(codec, K, tokenizer_hash="abc123")
    recon, report = decode_from_container(data)
    assert report.codec_name == "lbg_vq"
    assert report.shape_ok
    assert recon.shape == K.shape
    assert torch.isfinite(recon).all()


def test_container_encoding_deterministic():
    K = _sample_kv()
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    d1 = encode_to_container(codec, K, tokenizer_hash="t")
    d2 = encode_to_container(codec, K, tokenizer_hash="t")
    assert d1 == d2


def test_file_size_matches_byte_accountant():
    K = _sample_kv()
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    payload, meta = codec.encode_prefill(K)
    # the codec's own accountant must equal the real payload length exactly.
    assert meta["accountant_report"]["serialized_bytes"] == len(payload)


def test_optional_retention_section_present_and_decodes():
    K = _sample_kv()
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    positions = [0, 3, 7, 11]
    ret = encode_retention(positions, universe=12)
    data = encode_to_container(codec, K, retention_payload=ret)
    recon, report = decode_from_container(data)
    assert report.has_retention
    from fairfuzzkv_codec.metadata_coding.container import SECTION_RETENTION, unpack

    got, uni = decode_retention(unpack(data).get(SECTION_RETENTION))
    assert got == positions and uni == 12


def test_corrupted_container_rejected_at_decode():
    K = _sample_kv()
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    data = bytearray(encode_to_container(codec, K))
    data[-6] ^= 0xFF  # corrupt payload before trailer
    with pytest.raises(CorruptContainerError):
        decode_from_container(bytes(data))
