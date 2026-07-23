import torch

from fairfuzzkv_codec.quantization.scales import (
    ClipMethod,
    Granularity,
    aggregate_calibration_range,
    broadcast_min_max,
    compute_min_max,
    compute_mse_optimal_range,
    compute_percentile_range,
    select_range,
)


def _sample_tensor():
    torch.manual_seed(0)
    return torch.randn(2, 1, 4, 10, 8) * 3


def test_per_tensor_shape_is_scalar():
    t = _sample_tensor()
    mn, mx = compute_min_max(t, Granularity.PER_TENSOR)
    assert mn.shape == (1,)
    assert mx.shape == (1,)


def test_per_head_shape_matches_head_count():
    t = _sample_tensor()
    mn, mx = compute_min_max(t, Granularity.PER_HEAD)
    assert mn.shape == (4,)


def test_per_channel_shape_matches_head_dim():
    t = _sample_tensor()
    mn, mx = compute_min_max(t, Granularity.PER_CHANNEL)
    assert mn.shape == (8,)


def test_groupwise_shape_matches_num_groups():
    """Groupwise sits between per_tensor (1 value) and per_channel (head_dim
    values): num_groups compact values total, shared globally across every
    layer/batch/head/seq position - exactly like per_channel shares one
    scale per channel across all of those, just with channels grouped."""
    t = _sample_tensor()
    mn, mx = compute_min_max(t, Granularity.GROUPWISE, group_size=4)
    assert mn.shape == (2,)  # head_dim=8 / group_size=4 = 2 groups
    assert mx.shape == (2,)


def test_groupwise_rejects_non_divisible_group_size():
    t = _sample_tensor()  # head_dim=8
    try:
        compute_min_max(t, Granularity.GROUPWISE, group_size=3)
        assert False, "expected ValueError for non-divisible group_size"
    except ValueError:
        pass


def test_percentile_clipping_bounds_outliers():
    t = _sample_tensor()
    t[0, 0, 0, 0, 0] = 1000.0
    _true_min, true_max = compute_min_max(t, Granularity.PER_TENSOR)
    _clip_min, clip_max = compute_percentile_range(t, Granularity.PER_TENSOR, percentile=1.0)
    assert clip_max.item() < true_max.item()
    assert clip_max.item() < 50.0  # well below the injected 1000.0 outlier


def test_percentile_requires_valid_range():
    t = _sample_tensor()
    for bad in (0, 50, -1, 60):
        try:
            compute_percentile_range(t, Granularity.PER_TENSOR, percentile=bad)
            assert False, f"expected ValueError for percentile={bad}"
        except ValueError:
            pass


def test_mse_optimal_never_worse_than_minmax():
    """The candidate grid always includes fraction=1.0 (true min/max), so the
    search should never pick something with strictly higher MSE."""
    from fairfuzzkv_codec.quantization.uniform import dequantize_uniform, quantize_uniform

    t = _sample_tensor()
    t[0, 0, 0, 0, 0] = 50.0  # a mild outlier, not extreme enough to force clipping to be useless

    minmax_min, minmax_max = compute_min_max(t, Granularity.PER_CHANNEL)
    opt_min, opt_max = compute_mse_optimal_range(t, Granularity.PER_CHANNEL, num_bits=8, symmetric=True)

    def _mse_for(mn, mx):
        b_min, b_max = broadcast_min_max(mn, mx, t.shape, Granularity.PER_CHANNEL)
        abs_max = torch.max(b_min.abs(), b_max.abs()).clamp(min=1e-5)
        scale = abs_max / 127
        zp = torch.zeros_like(scale)
        q = quantize_uniform(t, 8, scale, zp, symmetric=True)
        deq = dequantize_uniform(q, scale, zp, dtype=t.dtype)
        return (deq - t).pow(2).mean().item()

    assert _mse_for(opt_min, opt_max) <= _mse_for(minmax_min, minmax_max) + 1e-9


def test_zero_range_tensor_does_not_produce_nan_or_inf():
    t = torch.zeros(2, 1, 4, 10, 8)
    mn, mx = compute_min_max(t, Granularity.PER_CHANNEL)
    b_min, b_max = broadcast_min_max(mn, mx, t.shape, Granularity.PER_CHANNEL)
    abs_max = torch.max(b_min.abs(), b_max.abs()).clamp(min=1e-5)
    scale = abs_max / 127
    assert torch.isfinite(scale).all()
    assert not torch.isnan(scale).any()


def test_select_range_dispatches_correctly():
    t = _sample_tensor()
    for method in (ClipMethod.MINMAX, ClipMethod.PERCENTILE, ClipMethod.MSE_OPTIMAL):
        mn, mx = select_range(t, Granularity.PER_CHANNEL, method, num_bits=8, symmetric=True, percentile=1.0)
        assert mn.shape == (8,)


def test_calibration_aggregation_is_deterministic():
    samples = [torch.randn(2, 1, 4, 5, 8) for _ in range(3)]
    mn1, mx1 = aggregate_calibration_range(samples, Granularity.PER_HEAD)
    mn2, mx2 = aggregate_calibration_range(samples, Granularity.PER_HEAD)
    assert torch.equal(mn1, mn2)
    assert torch.equal(mx1, mx2)


def test_calibration_requires_nonempty_list():
    try:
        aggregate_calibration_range([], Granularity.PER_TENSOR)
        assert False, "expected ValueError for empty calibration list"
    except ValueError:
        pass
