import torch

from fairfuzzkv_codec.repair_scoring.fuzzy import (
    DEFAULT_RULES, INPUT_LEVELS, OUTPUT_LEVELS, Rule, fuzzy_priority_scores,
    fuzzy_repair_priority, infer, triangular,
)
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs


def test_triangular_membership_known_points():
    # low = (0,0,1): a plateau of 1 at x=0, linear down to 0 at x=1.
    assert triangular(0.0, 0.0, 0.0, 1.0) == 1.0
    assert triangular(1.0, 0.0, 0.0, 1.0) == 0.0
    assert abs(triangular(0.3, 0.0, 0.0, 1.0) - 0.7) < 1e-9
    # high = (0,1,1): mirror image.
    assert triangular(1.0, 0.0, 1.0, 1.0) == 1.0
    assert triangular(0.0, 0.0, 1.0, 1.0) == 0.0
    # proper triangle, peak at b.
    assert triangular(0.5, 0.0, 0.5, 1.0) == 1.0
    assert triangular(0.0, 0.0, 0.5, 1.0) == 0.0
    assert triangular(1.0, 0.0, 0.5, 1.0) == 0.0
    # outside support -> 0.
    assert triangular(-1.0, 0.0, 0.5, 1.0) == 0.0
    assert triangular(2.0, 0.0, 0.5, 1.0) == 0.0


def _all_high_row():
    return {"fragility": 1.0, "evidence_importance": 1.0, "completion_cost": 1.0, "staleness": 0.0}


def _all_low_row():
    return {"fragility": 0.0, "evidence_importance": 0.0, "completion_cost": 0.0, "staleness": 0.0}


def test_all_high_inputs_fire_high_priority_rule_and_score_higher_than_all_low():
    high_priority, high_trace = infer(_all_high_row())
    low_priority, low_trace = infer(_all_low_row())
    assert high_priority > low_priority
    fired = {t.rule_name for t in high_trace if t.firing_strength > 0}
    assert "R1_all_high" in fired
    fired_low = {t.rule_name for t in low_trace if t.firing_strength > 0}
    assert "R8_all_low" in fired_low


def test_rule_trace_covers_every_rule_even_when_not_fired():
    priority, trace = infer(_all_high_row())
    assert {t.rule_name for t in trace} == {r.name for r in DEFAULT_RULES}
    # R8 (all-low) should not fire when everything is high.
    r8 = next(t for t in trace if t.rule_name == "R8_all_low")
    assert r8.firing_strength == 0.0


def test_staleness_boost_increases_priority_monotonically():
    base = _all_low_row()
    low_stale, _ = infer({**base, "staleness": 0.0})
    high_stale, _ = infer({**base, "staleness": 1.0})
    assert high_stale >= low_stale


def test_uncertainty_variable_absent_never_fires_its_rule():
    row = _all_low_row()  # no "uncertainty" key at all
    _, trace = infer(row)
    r10 = next(t for t in trace if t.rule_name == "R10_uncertain_boost")
    assert r10.firing_strength == 0.0


def test_priority_is_bounded_and_deterministic():
    row = {"fragility": 0.4, "evidence_importance": 0.7, "completion_cost": 0.2, "staleness": 0.6}
    p1, _ = infer(row)
    p2, _ = infer(row)
    assert p1 == p2
    assert 0.0 <= p1 <= 1.0


def test_no_rule_fires_returns_neutral_default_not_crash():
    priority, trace = infer({"fragility": 0.5}, rules=[Rule("only_stale", {"staleness": "high"}, "high")])
    assert priority == 0.5
    assert trace[0].firing_strength == 0.0


def test_batched_scores_match_per_row_infer():
    n = 6
    g = torch.Generator().manual_seed(7)
    inputs = ScorerInputs(
        fragility=torch.rand(n, generator=g), evidence_importance=torch.rand(n, generator=g),
        completion_cost=torch.rand(n, generator=g), staleness=torch.rand(n, generator=g),
    )
    batched = fuzzy_priority_scores(inputs)
    results = fuzzy_repair_priority(inputs)
    assert len(results) == n
    for i, r in enumerate(results):
        assert abs(batched[i].item() - r.priority) < 1e-6  # float32 tensor vs python-float centroid


def test_monotone_in_all_core_inputs_holding_others_fixed():
    # matches the existing repair_score contract (Prompt 9): higher input -> higher (or equal) priority.
    base = {"fragility": 0.5, "evidence_importance": 0.5, "completion_cost": 0.5, "staleness": 0.5}
    baseline, _ = infer(base)
    for field in ("fragility", "evidence_importance", "completion_cost", "staleness"):
        bumped, _ = infer({**base, field: 0.9})
        assert bumped >= baseline - 1e-9, f"{field} bump decreased priority"


def test_default_levels_partition_output_universe():
    assert set(OUTPUT_LEVELS.keys()) == {"low", "medium", "high"}
    assert set(INPUT_LEVELS.keys()) == {"low", "high"}
