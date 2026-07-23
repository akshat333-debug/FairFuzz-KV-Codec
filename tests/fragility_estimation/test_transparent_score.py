import pytest

from fairfuzzkv_codec.fragility_estimation.leakage import LeakageError
from fairfuzzkv_codec.fragility_estimation.schema import FeatureVector
from fairfuzzkv_codec.fragility_estimation.transparent_score import transparent_monotone_score


def _make_fv(**overrides):
    base = dict(
        unit_char_span=(0, 5),
        num_subtokens=1.0,
        chars_per_token=5.0,
        bytes_per_token=5.0,
        continuation_ratio=0.0,
        script_transitions=0.0,
        normalization_sensitivity=0.0,
        rare_token_indicator=0.0,
        boundary_mismatch=0.0,
        token_cost_inflation=1.0,
    )
    base.update(overrides)
    return FeatureVector(**base)


def test_score_is_in_unit_interval():
    fv = _make_fv()
    result = transparent_monotone_score(fv)
    assert 0.0 <= result.score <= 1.0


def test_contributions_sum_to_score():
    fv = _make_fv(num_subtokens=5, continuation_ratio=0.4, script_transitions=2, token_cost_inflation=2.5)
    result = transparent_monotone_score(fv)
    assert abs(sum(result.feature_contributions.values()) - result.score) < 1e-9


def test_score_is_monotone_in_boundary_mismatch():
    low = transparent_monotone_score(_make_fv(boundary_mismatch=0.0))
    high = transparent_monotone_score(_make_fv(boundary_mismatch=1.0))
    assert high.score > low.score


def test_score_is_monotone_in_token_cost_inflation():
    low = transparent_monotone_score(_make_fv(token_cost_inflation=1.0))
    high = transparent_monotone_score(_make_fv(token_cost_inflation=3.0))
    assert high.score > low.score


def test_score_is_monotone_in_num_subtokens():
    low = transparent_monotone_score(_make_fv(num_subtokens=1))
    high = transparent_monotone_score(_make_fv(num_subtokens=8))
    assert high.score > low.score


def test_deterministic_rerun_same_score():
    fv = _make_fv(num_subtokens=3, continuation_ratio=0.3, rare_token_indicator=0.2)
    s1 = transparent_monotone_score(fv).score
    s2 = transparent_monotone_score(fv).score
    assert s1 == s2


def test_rejects_leaked_feature_via_direct_dict_call():
    from fairfuzzkv_codec.fragility_estimation.leakage import validate_no_leakage

    with pytest.raises(LeakageError):
        validate_no_leakage({"num_subtokens": 1.0, "language": "hi"})

    with pytest.raises(LeakageError):
        validate_no_leakage({"num_subtokens": 1.0, "task_accuracy": 0.9})
