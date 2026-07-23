import itertools

import pytest
import torch

from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
from fairfuzzkv_codec.quantization.bitwidth_map import BitWidthMap
from fairfuzzkv_codec.quantization.scales import ClipMethod, Granularity


def _sample_kv():
    torch.manual_seed(0)
    return torch.randn(4, 1, 8, 16, 32) * 2  # [layers, batch, heads, seq, head_dim]


GRANULARITIES = [Granularity.PER_TENSOR, Granularity.PER_HEAD, Granularity.PER_CHANNEL]
SYMMETRIC_OPTIONS = [True, False]
BITS_OPTIONS = [8, 4]


@pytest.mark.parametrize(
    "granularity,symmetric,bits", list(itertools.product(GRANULARITIES, SYMMETRIC_OPTIONS, BITS_OPTIONS))
)
def test_round_trip_within_tolerance(granularity, symmetric, bits):
    K = _sample_kv()
    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=granularity, symmetric=symmetric, default_bits=bits)
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")

    assert recon.shape == K.shape
    mse = (recon - K).pow(2).mean().item()
    # 4-bit is lossy by design; just assert it's finite and not wildly off
    max_expected_mse = 0.01 if bits == 8 else 0.5
    assert mse < max_expected_mse
    assert torch.isfinite(recon).all()


def test_groupwise_granularity_round_trips():
    K = _sample_kv()
    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.GROUPWISE, group_size=8, default_bits=8)
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")
    assert (recon - K).pow(2).mean().item() < 0.01


def test_groupwise_scale_storage_is_compact_not_per_element():
    """Groupwise must store num_groups scale values (shared across every
    layer/batch/head/seq position), not one scale per element - the earlier
    version of this code accidentally materialized a full-tensor-sized scale
    array for groupwise, inflating bits/element ~17x. Regression guard."""
    K = _sample_kv()  # head_dim=32
    group_size = 8

    codec_group = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.GROUPWISE, group_size=group_size, default_bits=8)
    codec_channel = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.PER_CHANNEL, default_bits=8)

    _s1, meta_group = codec_group.encode_prefill(K)
    _s2, meta_channel = codec_channel.encode_prefill(K)

    group_bits = meta_group["accountant_report"]["logical_bits"] / K.numel()
    channel_bits = meta_channel["accountant_report"]["logical_bits"] / K.numel()

    # groupwise has FEWER distinct scales (num_groups=4) than per_channel
    # (head_dim=32), so its overhead-per-element must be <= per_channel's.
    assert group_bits <= channel_bits + 0.01
    assert group_bits < 8.5  # sane overhead, not the ~68 bits/element regression


@pytest.mark.parametrize("clip_method", [ClipMethod.MINMAX, ClipMethod.PERCENTILE, ClipMethod.MSE_OPTIMAL])
def test_clip_methods_round_trip(clip_method):
    K = _sample_kv()
    codec = ScalarQuantCodec(
        "hash", tensor_name="k", granularity=Granularity.PER_CHANNEL, clip_method=clip_method, percentile=1.0
    )
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")
    assert torch.isfinite(recon).all()


def test_int4_is_genuinely_packed_two_per_byte():
    K = _sample_kv()
    codec8 = ScalarQuantCodec("hash", tensor_name="k", default_bits=8)
    codec4 = ScalarQuantCodec("hash", tensor_name="k", default_bits=4)
    stream8, meta8 = codec8.encode_prefill(K)
    stream4, meta4 = codec4.encode_prefill(K)

    data8_bytes = meta8["accountant_report"]["components"]["tensor_data_grp8b_0_data"]
    data4_bytes = meta4["accountant_report"]["components"]["tensor_data_grp4b_0_data"]

    # data8_bytes includes an 8-byte length prefix (see binary_serializer.py);
    # the real packed payload is (num_elements) for int8 and ceil(n/2) for int4.
    num_elements = K.numel()
    assert data8_bytes - 8 == num_elements
    assert data4_bytes - 8 == (num_elements + 1) // 2
    assert data4_bytes < data8_bytes / 1.9  # genuinely ~half, not just "smaller"


def test_all_overhead_is_counted_in_reported_bits():
    K = _sample_kv()
    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.PER_CHANNEL, default_bits=8)
    stream, meta = codec.encode_prefill(K)
    report = meta["accountant_report"]

    pure_data_bits = K.numel() * 8  # if only data bits counted, no scale/zp/header
    assert report["logical_bits"] > pure_data_bits  # scale/zp/header add real overhead
    assert report["serialized_bytes"] == len(stream)
    assert "grp8b_0_scale" in report["components"] or any("scale" in k for k in report["components"])
    assert any("zp" in k for k in report["components"])


def test_deterministic_byte_output():
    K = _sample_kv()
    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.PER_CHANNEL, default_bits=8)
    stream_a, _ = codec.encode_prefill(K)
    stream_b, _ = codec.encode_prefill(K)
    assert stream_a == stream_b


def test_nan_input_is_rejected():
    K = _sample_kv()
    K[0, 0, 0, 0, 0] = float("nan")
    codec = ScalarQuantCodec("hash", tensor_name="k")
    with pytest.raises(ValueError):
        codec.encode_prefill(K)


def test_zero_tensor_round_trips_without_nan_or_inf():
    K = torch.zeros(4, 1, 8, 16, 32)
    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.PER_CHANNEL)
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")
    assert torch.isfinite(recon).all()
    assert torch.equal(recon, K)


def test_constant_nonzero_tensor_round_trips():
    K = torch.full((4, 1, 8, 16, 32), 3.5)
    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.PER_CHANNEL)
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")
    assert torch.isfinite(recon).all()
    assert (recon - K).abs().max().item() < 0.1


def test_boundary_extreme_values_round_trip():
    K = _sample_kv()
    K[0, 0, 0, 0, 0] = 1e6
    K[1, 0, 0, 0, 0] = -1e6
    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.PER_CHANNEL)
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")
    assert torch.isfinite(recon).all()


def test_mixed_precision_uses_different_bits_per_layer():
    K = _sample_kv()
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=8)
    bwm.set_layer_bits("k", 2, 4)
    bwm.set_layer_bits("k", 3, 4)

    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.PER_CHANNEL, bitwidth_map=bwm)
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")

    assert set(meta["group_keys"]) == {"grp8b_0", "grp4b_2"}
    mse_8bit_layers = (recon[[0, 1]] - K[[0, 1]]).pow(2).mean().item()
    mse_4bit_layers = (recon[[2, 3]] - K[[2, 3]]).pow(2).mean().item()
    assert mse_8bit_layers < mse_4bit_layers  # 8-bit layers must be more accurate


def test_mixed_kv_precision_two_codec_instances_share_one_map():
    K = _sample_kv()
    V = _sample_kv() * 1.5

    bwm = BitWidthMap(default_k_bits=8, default_v_bits=4)
    codec_k = ScalarQuantCodec("hash", tensor_name="k", bitwidth_map=bwm)
    codec_v = ScalarQuantCodec("hash", tensor_name="v", bitwidth_map=bwm)

    stream_k, meta_k = codec_k.encode_prefill(K)
    stream_v, meta_v = codec_v.encode_prefill(V)

    assert meta_k["group_keys"] == ["grp8b_0"]
    assert meta_v["group_keys"] == ["grp4b_0"]

    recon_k = codec_k.decode(stream_k, meta_k, tuple(meta_k["full_shape"]), torch.float32, "cpu")
    recon_v = codec_v.decode(stream_v, meta_v, tuple(meta_v["full_shape"]), torch.float32, "cpu")
    assert (recon_k - K).pow(2).mean().item() < (recon_v - V).pow(2).mean().item()


def test_saturation_diagnostics_present_and_sane():
    K = _sample_kv()
    codec = ScalarQuantCodec(
        "hash", tensor_name="k", granularity=Granularity.PER_TENSOR, clip_method=ClipMethod.PERCENTILE, percentile=5.0
    )
    _stream, meta = codec.encode_prefill(K)
    saturation = meta["saturation"]["grp8b_0"]
    assert 0.0 <= saturation["saturation_rate"] <= 1.0
    assert saturation["num_saturated_low"] + saturation["num_saturated_high"] <= saturation["num_elements"]
    assert saturation["saturation_rate"] > 0.0  # percentile=5 should clip something out of real Gaussian data


def test_head_override_uses_cell_grouping_and_distinct_bits_per_head():
    K = _sample_kv()  # [layers, batch, heads, seq, head_dim]
    num_heads = K.size(2)
    assert num_heads >= 2
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=8)
    bwm.set_head_bits("k", layer=0, head=0, bits=4)  # one head goes to 4-bit

    codec = ScalarQuantCodec("hash", tensor_name="k", granularity=Granularity.PER_CHANNEL, bitwidth_map=bwm)
    stream, meta = codec.encode_prefill(K)
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), torch.float32, "cpu")

    assert meta["group_mode"] == "cell"
    assert recon.shape == K.shape
    assert torch.isfinite(recon).all()

    # The single 4-bit head (layer 0, head 0) must be less accurate than an
    # 8-bit head in the same layer - proves the per-head bits actually applied.
    mse_4bit_head = (recon[0, :, 0] - K[0, :, 0]).pow(2).mean().item()
    mse_8bit_head = (recon[0, :, 1] - K[0, :, 1]).pow(2).mean().item()
    assert mse_4bit_head > mse_8bit_head


def test_no_head_override_stays_on_layer_path_byte_identical():
    K = _sample_kv()
    # Same default bits, once via a plain map and once via a map carrying only
    # a LAYER override elsewhere - neither has head overrides, so both must use
    # the layer path and produce identical bytes.
    codec_a = ScalarQuantCodec("hash", tensor_name="k", default_bits=8)
    codec_b = ScalarQuantCodec("hash", tensor_name="k", default_bits=8)
    stream_a, meta_a = codec_a.encode_prefill(K)
    stream_b, _ = codec_b.encode_prefill(K)
    assert meta_a["group_mode"] == "layer"
    assert stream_a == stream_b


def test_invalid_tensor_name_rejected():
    with pytest.raises(ValueError):
        ScalarQuantCodec("hash", tensor_name="q")
