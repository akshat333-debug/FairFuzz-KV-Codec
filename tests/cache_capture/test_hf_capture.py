import torch
import pytest

from fairfuzzkv_codec.cache_capture.hf_capture import HFCapture
from fairfuzzkv_codec.core.config import LayerHeadSelection
from fairfuzzkv_codec.codec.baselines import FullKVFP16Codec, UniformQuantCodec
from fairfuzzkv_codec.benchmarks.attention_harness import AttentionVerificationHarness

# Tiny real Qwen2 model (2 layers, 2 KV heads, head_dim=2) so this stays fast on CPU.
TINY_MODEL = "yujiepan/qwen2-tiny-random"


@pytest.fixture(scope="module")
def capture():
    return HFCapture(TINY_MODEL, device="cpu", dtype=torch.float32)


def test_capture_prefill_kv_shapes(capture):
    selection = LayerHeadSelection()
    K, V = capture.capture_prefill_kv("hello world", selection)

    # [layers, batch, heads, seq_len, head_dim]
    assert K.dim() == 5
    assert K.shape == V.shape
    assert K.shape[0] == capture.model.config.num_hidden_layers
    assert K.shape[2] == capture.model.config.num_key_value_heads


def test_capture_layer_head_selection(capture):
    selection = LayerHeadSelection(layers=[0], heads=[0])
    K, V = capture.capture_prefill_kv("hello world", selection)

    assert K.shape[0] == 1  # one selected layer
    assert K.shape[2] == 1  # one selected head


def test_real_kv_reconstruction_via_attention_harness(capture):
    """Gate 2 (Prompt 2): real captured KV must round-trip through a codec and
    reproduce attention output within tolerance, not just synthetic fixtures."""
    selection = LayerHeadSelection()
    K, V = capture.capture_prefill_kv("hello world", selection)

    harness = AttentionVerificationHarness(head_dim=K.size(-1))
    q_dummy = torch.randn_like(K)

    # FullKV FP16 codec must reconstruct the real captured cache exactly (~0 MSE
    # up to fp16 rounding).
    codec = FullKVFP16Codec(config_hash="test_hash")
    k_bytes, k_meta = codec.encode_prefill(K)
    K_recon = codec.decode(k_bytes, k_meta, tuple(k_meta["kv_shape"]), K.dtype, "cpu")
    v_bytes, v_meta = codec.encode_prefill(V)
    V_recon = codec.decode(v_bytes, v_meta, tuple(v_meta["kv_shape"]), V.dtype, "cpu")

    mse_fp16 = harness.verify_reconstruction(q_dummy, K, V, K_recon, V_recon)
    assert mse_fp16 < 1e-3

    # INT8 quantization must reconstruct with small, finite distortion (not NaN/inf).
    q_codec = UniformQuantCodec(config_hash="test_hash", num_bits=8)
    q_bytes, q_meta = q_codec.encode_prefill(K)
    K_recon_int8 = q_codec.decode(q_bytes, q_meta, tuple(q_meta["kv_shape"]), K.dtype, "cpu")

    mse_int8 = harness.verify_reconstruction(q_dummy, K, V, K_recon_int8, V)
    assert mse_int8 == mse_int8  # not NaN
    assert mse_int8 < 1.0
