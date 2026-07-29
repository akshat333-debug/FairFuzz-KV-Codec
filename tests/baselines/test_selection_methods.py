import torch

from fairfuzzkv_codec.baselines.selection_methods import h2o_mask, pyramidkv_mask, snapkv_mask


def _synthetic_attn(layers=3, heads=2, seq=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(layers, 1, heads, seq, seq, generator=g)
    return torch.softmax(logits, dim=-1)


def test_h2o_mask_shape_and_retention():
    attn = _synthetic_attn(seq=20)
    mask = h2o_mask(attn, heavy_ratio=0.3, recent_ratio=0.2)
    assert mask.shape == (20,)
    assert mask.dtype == torch.bool
    assert mask.any()  # never evicts everything


def test_h2o_always_keeps_the_most_recent_position():
    attn = _synthetic_attn(seq=20)
    mask = h2o_mask(attn, heavy_ratio=0.1, recent_ratio=0.2)
    assert mask[-1].item() is True  # last position is always in the recency window


def test_h2o_union_is_at_least_as_large_as_either_component():
    attn = _synthetic_attn(seq=20)
    small_union = h2o_mask(attn, heavy_ratio=0.1, recent_ratio=0.1)
    big_union = h2o_mask(attn, heavy_ratio=0.5, recent_ratio=0.1)
    assert big_union.sum() >= small_union.sum()


def test_snapkv_mask_shape_and_retention():
    attn = _synthetic_attn(seq=20)
    mask = snapkv_mask(attn, keep_ratio=0.4, observation_window=4, pooling_kernel=3)
    assert mask.shape == (20,)
    kept = int(mask.sum().item())
    assert kept >= int(20 * 0.4) - 2  # top-k selection is approximate after pooling ties, allow small slack


def test_snapkv_higher_keep_ratio_keeps_more():
    attn = _synthetic_attn(seq=20)
    low = snapkv_mask(attn, keep_ratio=0.2, observation_window=4)
    high = snapkv_mask(attn, keep_ratio=0.6, observation_window=4)
    assert high.sum() > low.sum()


def test_snapkv_observation_window_larger_than_seq_does_not_crash():
    attn = _synthetic_attn(seq=8)
    mask = snapkv_mask(attn, keep_ratio=0.5, observation_window=1000)
    assert mask.shape == (8,)


def test_pyramidkv_mask_is_2d_per_layer():
    attn = _synthetic_attn(layers=4, seq=20)
    mask = pyramidkv_mask(attn, total_keep_ratio=0.4, pyramid_ratio=2.0)
    assert mask.shape == (4, 20)
    assert mask.dtype == torch.bool


def test_pyramidkv_early_layers_keep_more_than_late_layers():
    attn = _synthetic_attn(layers=6, seq=40, seed=1)
    mask = pyramidkv_mask(attn, total_keep_ratio=0.3, pyramid_ratio=3.0)
    per_layer_counts = mask.sum(dim=1)
    assert per_layer_counts[0] > per_layer_counts[-1]


def test_pyramidkv_average_retention_hits_target():
    attn = _synthetic_attn(layers=8, seq=50, seed=2)
    target = 0.3
    mask = pyramidkv_mask(attn, total_keep_ratio=target, pyramid_ratio=2.0)
    actual = mask.float().mean().item()
    assert abs(actual - target) < 0.05


def test_pyramidkv_single_layer_uses_flat_multiplier():
    attn = _synthetic_attn(layers=1, seq=20)
    mask = pyramidkv_mask(attn, total_keep_ratio=0.5)
    assert abs(mask.float().mean().item() - 0.5) < 0.1
