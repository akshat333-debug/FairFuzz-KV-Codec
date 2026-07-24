import numpy as np
import torch

from fairfuzzkv_codec.allocation.calibration import (
    calibrate_layers_mixed,
    calibrate_layers_scalar,
    encode_with_allocation,
    make_split,
)
from fairfuzzkv_codec.allocation.allocator import solve_exact
from fairfuzzkv_codec.allocation.curves import DistortionCurve, fit_exponential, marginal_decay


def _sample_kv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(3, 1, 2, 30, 16, generator=g)


# ---- curves ----------------------------------------------------------------

def test_exponential_fit_recovers_clean_curve():
    bits = [2, 4, 6, 8]
    alpha, beta = 5.0, 0.5
    d = [alpha * np.exp(-beta * b) for b in bits]
    fit = fit_exponential(bits, d)
    assert fit is not None and fit.supported
    assert abs(fit.beta - beta) < 1e-6
    assert fit.r2 > 0.999


def test_curve_falls_back_to_monotone_when_fit_poor():
    # non-exponential, noisy-but-monotone points -> exponential not "supported",
    # monotone interpolation used instead.
    bits = [2, 4, 6, 8]
    d = [10.0, 9.9, 2.0, 1.9]
    curve = DistortionCurve(bits, d)
    # prediction between measured points must stay within the bracket.
    mid = curve.predict(5)
    assert 2.0 <= mid <= 9.9


def test_monotone_enforced_against_measurement_inversions():
    curve = DistortionCurve([2, 4, 6], [3.0, 5.0, 1.0])  # 4-bit inverted upward
    assert curve.distortions[1] <= curve.distortions[0]  # clamped


def test_marginal_decay_is_nonnegative_for_monotone_curve():
    curve = DistortionCurve([2, 4, 8], [8.0, 4.0, 1.0])
    for _b, slope in marginal_decay(curve):
        assert slope >= 0


# ---- calibration + real-encoder drive --------------------------------------

def test_split_is_disjoint():
    K = _sample_kv()
    s = make_split(K)
    total = s.train.size(3) + s.val.size(3) + s.test.size(3)
    assert total == K.size(3)
    assert s.train.size(3) > 0 and s.val.size(3) > 0 and s.test.size(3) > 0


def test_calibration_produces_cohorts_with_bit_options():
    K = _sample_kv()
    cohorts = calibrate_layers_scalar(K, bit_choices=[4, 8])
    assert len(cohorts) == 3
    for c in cohorts:
        labels = {o.label for o in c.options}
        assert labels == {"int4", "int8"}
        int4 = next(o for o in c.options if o.label == "int4")
        int8 = next(o for o in c.options if o.label == "int8")
        # 8-bit costs more bits and distorts less than 4-bit (on real data).
        assert int8.total_bits > int4.total_bits
        assert int8.distortion <= int4.distortion


def test_mixed_calibration_includes_scalar_and_lbg_options():
    K = _sample_kv()
    cohorts = calibrate_layers_mixed(K, scalar_bits=[4, 8], lbg_configs=[(8, 16)])
    assert len(cohorts) == 3
    for c in cohorts:
        labels = {o.label for o in c.options}
        assert "int4" in labels and "int8" in labels
        assert any(lbl.startswith("lbg_") for lbl in labels)  # LBG option present
        for o in c.options:
            assert o.total_bits > 0  # codebook overhead counted, nothing free


def test_allocation_drives_the_real_encoder_within_budget():
    K = _sample_kv()
    cohorts = calibrate_layers_scalar(K, bit_choices=[4, 8])
    # budget between all-4bit and all-8bit -> mixed allocation.
    lo = sum(min(o.total_bits for o in c.options) for c in cohorts)
    hi = sum(max(o.total_bits for o in c.options) for c in cohorts)
    budget = (lo + hi) // 2
    alloc = solve_exact(cohorts, budget)
    assert alloc.feasible

    stream, meta = encode_with_allocation(K, alloc, tensor_name="k")
    # the real serialized encoding must respect the allocator's bit budget.
    assert meta["accountant_report"]["serialized_bytes"] * 8 <= budget
    # and it must actually use mixed per-layer bits (group_mode layer, >1 group)
    # whenever the allocation isn't uniform.
    chosen = {o.label for o in alloc.choice.values()}
    if len(chosen) > 1:
        assert len(meta["group_keys"]) > 1
