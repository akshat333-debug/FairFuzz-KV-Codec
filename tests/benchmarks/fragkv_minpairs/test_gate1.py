"""Pre-registration test: proves the Gate 1 decision function (gate1.py) is
frozen and behaves correctly on synthetic EffectSizeResult fixtures, BEFORE
it is ever pointed at real FragKV-MinPairs study predictions. This file must
exist and pass prior to running the real study - that is what "pre-registered"
means operationally here."""

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.gate1 import (
    CONTROL_CONFOUND_THRESHOLD,
    PRACTICAL_EFFECT_THRESHOLD,
    WEAK_EFFECT_THRESHOLD,
    Gate1Decision,
    decide_gate1,
)
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.stats_utils import EffectSizeResult


def _effect(codec_name, effect_size, monotonic=True, n=150):
    return EffectSizeResult(
        codec_name=codec_name,
        low_n_g=1,
        high_n_g=8,
        n_paired_groups=n,
        accuracy_by_n_g={1: 0.9, 2: 0.85, 4: 0.8, 8: 0.9 - effect_size},
        effect_size=effect_size,
        ci_low=effect_size - 0.05,
        ci_high=effect_size + 0.05,
        p_value=0.01,
        monotonic_non_increasing=monotonic,
        power_note="",
    )


def test_pass_when_meaningful_effect_and_no_control_confound():
    control = _effect("FullKV", effect_size=0.0)
    lossy = [_effect("UniformQuant8", effect_size=0.15)]
    report = decide_gate1(control, lossy)
    assert report.decision == Gate1Decision.PASS
    assert not report.control_confound


def test_weak_pass_when_effect_between_thresholds():
    control = _effect("FullKV", effect_size=0.0)
    lossy = [_effect("UniformQuant8", effect_size=(PRACTICAL_EFFECT_THRESHOLD + WEAK_EFFECT_THRESHOLD) / 2)]
    report = decide_gate1(control, lossy)
    assert report.decision == Gate1Decision.WEAK_PASS


def test_weak_pass_when_meaningful_but_control_confounded():
    control = _effect("FullKV", effect_size=CONTROL_CONFOUND_THRESHOLD + 0.05)
    lossy = [_effect("UniformQuant8", effect_size=0.20)]
    report = decide_gate1(control, lossy)
    assert report.decision == Gate1Decision.WEAK_PASS
    assert report.control_confound


def test_fail_when_effect_negligible():
    control = _effect("FullKV", effect_size=0.0)
    lossy = [_effect("UniformQuant8", effect_size=0.01)]
    report = decide_gate1(control, lossy)
    assert report.decision == Gate1Decision.FAIL


def test_fail_when_effect_meaningful_but_not_monotonic():
    """Directional consistency is required, not just endpoint effect size -
    a big n_g=1 vs n_g=8 gap with a non-monotonic middle doesn't qualify."""
    control = _effect("FullKV", effect_size=0.0)
    lossy = [_effect("UniformQuant8", effect_size=0.20, monotonic=False)]
    report = decide_gate1(control, lossy)
    assert report.decision == Gate1Decision.FAIL


def test_fail_when_no_paired_groups():
    control = _effect("FullKV", effect_size=0.0, n=0)
    lossy = [_effect("UniformQuant8", effect_size=0.20, n=0)]
    report = decide_gate1(control, lossy)
    assert report.decision == Gate1Decision.FAIL


def test_pass_if_any_lossy_codec_qualifies_even_if_another_does_not():
    control = _effect("FullKV", effect_size=0.0)
    lossy = [
        _effect("UniformQuant8", effect_size=0.01),
        _effect("TopK", effect_size=0.25),
    ]
    report = decide_gate1(control, lossy)
    assert report.decision == Gate1Decision.PASS


def test_decision_is_pure_deterministic_function_of_inputs():
    control = _effect("FullKV", effect_size=0.0)
    lossy = [_effect("UniformQuant8", effect_size=0.15)]
    r1 = decide_gate1(control, lossy)
    r2 = decide_gate1(control, lossy)
    assert r1.decision == r2.decision
    assert r1.reasoning == r2.reasoning
