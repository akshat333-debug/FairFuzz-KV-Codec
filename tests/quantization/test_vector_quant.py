import torch

from fairfuzzkv_codec.codec.vector_quant import CodebookScope, LBGVectorQuantCodec
from fairfuzzkv_codec.quantization.vector_quant import (
    VectorPolicy,
    form_vectors,
    index_bits,
    nearest_codeword,
    train_lbg,
    unform_vectors,
)


def _sample_kv(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # [layers, batch, heads, seq, head_dim]
    return torch.randn(3, 1, 2, 12, 16, generator=g)


# ---- vector formation ------------------------------------------------------

def test_head_block_formation_round_trips_exactly():
    K = _sample_kv()
    for vd in (4, 8, 16):
        vectors, layout = form_vectors(K, vd, VectorPolicy.HEAD_BLOCK)
        assert vectors.shape[1] == vd
        restored = unform_vectors(vectors, layout)
        assert torch.equal(restored, K)


def test_cross_token_formation_round_trips_exactly():
    K = _sample_kv()
    vectors, layout = form_vectors(K, 8, VectorPolicy.CROSS_TOKEN, token_span=3)
    assert vectors.shape[1] == 8 * 3
    restored = unform_vectors(vectors, layout)
    assert torch.equal(restored, K)


# ---- LBG training ----------------------------------------------------------

def test_lbg_training_is_deterministic_under_fixed_seed():
    K = _sample_kv()
    vectors, _ = form_vectors(K, 8, VectorPolicy.HEAD_BLOCK)
    cb1, d1 = train_lbg(vectors, 16, seed=123)
    cb2, d2 = train_lbg(vectors, 16, seed=123)
    assert torch.equal(cb1, cb2)
    assert d1.final_mse == d2.final_mse


def test_lbg_rejects_non_power_of_two():
    K = _sample_kv()
    vectors, _ = form_vectors(K, 8, VectorPolicy.HEAD_BLOCK)
    try:
        train_lbg(vectors, 100, seed=1)
        assert False, "expected ValueError for non-power-of-two codebook_size"
    except ValueError:
        pass


def test_larger_codebook_lowers_distortion():
    K = _sample_kv()
    vectors, _ = form_vectors(K, 8, VectorPolicy.HEAD_BLOCK)
    _, d16 = train_lbg(vectors, 16, seed=7)
    _, d64 = train_lbg(vectors, 64, seed=7)
    assert d64.final_mse < d16.final_mse
    assert d16.utilization > 0.0


def test_minibatch_training_is_deterministic():
    K = _sample_kv()
    vectors, _ = form_vectors(K, 8, VectorPolicy.HEAD_BLOCK)
    cb1, _ = train_lbg(vectors, 16, seed=5, minibatch=64)
    cb2, _ = train_lbg(vectors, 16, seed=5, minibatch=64)
    assert torch.equal(cb1, cb2)


def test_index_bits():
    assert index_bits(16) == 4
    assert index_bits(64) == 6
    assert index_bits(256) == 8


# ---- codec end-to-end ------------------------------------------------------

def test_codec_round_trip_shape_and_order_preserved():
    K = _sample_kv()
    codec = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=64)
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")
    assert recon.shape == K.shape
    assert torch.isfinite(recon).all()
    # VQ is lossy but should track the signal, not scramble positions.
    assert (recon - K).pow(2).mean().item() < K.pow(2).mean().item()


def test_codec_encoding_is_deterministic():
    K = _sample_kv()
    c1 = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=64, seed=9)
    c2 = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=64, seed=9)
    s1, _ = c1.encode_prefill(K)
    s2, _ = c2.encode_prefill(K)
    assert s1 == s2


def test_codebook_overhead_is_serialized_and_counted():
    K = _sample_kv()
    codec = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=64)
    _stream, meta = codec.encode_prefill(K)
    report = meta["accountant_report"]
    # codebook float32 bytes must appear in the serialized components.
    cb_components = [k for k in report["components"] if k.endswith("_cb") or "_cb" in k]
    assert any(report["components"][k] > 0 for k in cb_components)
    assert report["serialized_bytes"] > 0


def test_larger_codebook_size_uses_more_index_bits_logically():
    K = _sample_kv()
    _s16, m16 = LBGVectorQuantCodec("h", "k", vector_dim=8, codebook_size=16).encode_prefill(K)
    _s64, m64 = LBGVectorQuantCodec("h", "k", vector_dim=8, codebook_size=64).encode_prefill(K)
    assert m16["g_idx_logical_bits_per_element"] == 4.0
    assert m64["g_idx_logical_bits_per_element"] == 6.0


def test_per_layer_scope_trains_one_codebook_per_layer():
    K = _sample_kv()
    codec = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=16, scope=CodebookScope.PER_LAYER)
    stream, meta = codec.encode_prefill(K)
    assert meta["scope_keys"] == ["L0", "L1", "L2"]
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")
    assert recon.shape == K.shape


def test_per_head_scope_trains_one_codebook_per_head():
    K = _sample_kv()
    codec = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=16, scope=CodebookScope.PER_HEAD)
    stream, meta = codec.encode_prefill(K)
    assert meta["scope_keys"] == ["H0", "H1"]
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")
    assert recon.shape == K.shape


def test_calibration_fit_prevents_encoding_from_retraining():
    calib = _sample_kv(seed=1)
    data = _sample_kv(seed=2)
    codec = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=16)
    codec.fit(calib)
    version_before = codec.last_diagnostics["g"].final_mse
    _stream, meta = codec.encode_prefill(data)
    # No self-calibration diagnostics were emitted (codebook came from fit()).
    assert "diagnostics" not in meta
    assert version_before is not None


def test_nan_rejected():
    K = _sample_kv()
    K[0, 0, 0, 0, 0] = float("nan")
    codec = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=16)
    try:
        codec.encode_prefill(K)
        assert False, "expected ValueError on NaN"
    except ValueError:
        pass


def test_codebook_size_capped_at_256():
    try:
        LBGVectorQuantCodec("hash", "k", codebook_size=512)
        assert False, "expected ValueError for codebook_size > 256"
    except ValueError:
        pass


def test_diagnostics_report_utilization_and_dead_codes():
    K = _sample_kv()
    codec = LBGVectorQuantCodec("hash", "k", vector_dim=8, codebook_size=64)
    codec.encode_prefill(K)
    diag = codec.last_diagnostics["g"]
    assert diag.used_codewords + diag.dead_codewords == 64
    assert 0.0 <= diag.utilization <= 1.0


def test_faiss_flag_matches_cpu_reference():
    # use_faiss=True silently falls back to the CPU path when FAISS is absent;
    # either way results must equal the authoritative CPU reference.
    K = _sample_kv()
    vectors, _ = form_vectors(K, 8, VectorPolicy.HEAD_BLOCK)
    cb, _ = train_lbg(vectors, 16, seed=3)
    idx_cpu = nearest_codeword(vectors, cb, use_faiss=False)
    idx_faiss = nearest_codeword(vectors, cb, use_faiss=True)
    assert torch.equal(idx_cpu, idx_faiss)
