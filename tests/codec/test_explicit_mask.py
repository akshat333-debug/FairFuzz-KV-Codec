import torch

from fairfuzzkv_codec.codec.explicit_mask import ExplicitMaskCodec


def _kv(seq=8):
    return torch.randn(1, 1, 1, seq, 4)  # [layers, batch, heads, seq, head_dim]


def test_evicted_positions_are_zeroed_and_kept_positions_survive():
    kv = _kv()
    keep = torch.tensor([True, True, False, False, True, False, False, True])
    codec = ExplicitMaskCodec("test", keep)
    byte_stream, meta = codec.encode_prefill(kv)
    recon = codec.decode(byte_stream, meta, tuple(meta["kv_shape"]), torch.float32, "cpu")

    assert torch.allclose(recon[..., ~keep, :], torch.zeros_like(recon[..., ~keep, :]))
    assert torch.allclose(recon[..., keep, :], kv[..., keep, :].to(torch.float16).to(torch.float32), atol=1e-3)


def test_retention_ratio_matches_actual_mask_fraction():
    kv = _kv(seq=10)
    keep = torch.tensor([True] * 3 + [False] * 7)
    codec = ExplicitMaskCodec("test", keep)
    _, meta = codec.encode_prefill(kv)
    assert abs(meta["retention_ratio"] - 0.3) < 1e-6
    assert abs(meta["kv_logical_bits_per_element"] - 4.8) < 1e-5  # 16 * 0.3


def test_identical_masks_give_identical_logical_bits_matched_bit_guarantee():
    kv1, kv2 = _kv(), _kv()
    keep = torch.tensor([True, False, True, False, True, False, True, False])
    _, meta1 = ExplicitMaskCodec("a", keep).encode_prefill(kv1)
    _, meta2 = ExplicitMaskCodec("b", keep).encode_prefill(kv2)
    assert meta1["accountant_report"]["logical_bits"] == meta2["accountant_report"]["logical_bits"]


def _kv_multilayer(layers=3, seq=8):
    return torch.randn(layers, 1, 1, seq, 4)


def test_2d_per_layer_mask_applies_a_different_retention_per_layer():
    kv = _kv_multilayer(layers=3, seq=8)
    keep = torch.zeros(3, 8, dtype=torch.bool)
    keep[0, :6] = True  # layer 0 keeps 6
    keep[1, :4] = True  # layer 1 keeps 4
    keep[2, :2] = True  # layer 2 keeps 2
    codec = ExplicitMaskCodec("test", keep)
    byte_stream, meta = codec.encode_prefill(kv)
    recon = codec.decode(byte_stream, meta, tuple(meta["kv_shape"]), torch.float32, "cpu")

    for layer in range(3):
        assert torch.allclose(recon[layer, ..., ~keep[layer], :], torch.zeros_like(recon[layer, ..., ~keep[layer], :]))
        assert torch.allclose(
            recon[layer, ..., keep[layer], :], kv[layer, ..., keep[layer], :].to(torch.float16).to(torch.float32), atol=1e-3,
        )
    assert abs(meta["retention_ratio"] - (6 + 4 + 2) / 24) < 1e-6


def test_2d_mask_wrong_layer_count_raises():
    kv = _kv_multilayer(layers=3, seq=8)
    keep = torch.zeros(2, 8, dtype=torch.bool)  # wrong layer count
    codec = ExplicitMaskCodec("test", keep)
    try:
        codec.encode_prefill(kv)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_3d_mask_rejected():
    kv = _kv_multilayer(layers=3, seq=8)
    keep = torch.zeros(3, 1, 8, dtype=torch.bool)
    codec = ExplicitMaskCodec("test", keep)
    try:
        codec.encode_prefill(kv)
        assert False, "expected ValueError"
    except ValueError:
        pass
