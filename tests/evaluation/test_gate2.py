"""Gate 2 frozen-logic tests on SYNTHETIC fixtures - written before any real
study is run, so the PASS/WEAK_PASS/FAIL logic cannot be tuned to a result."""

from fairfuzzkv_codec.evaluation.disparity import compare_systems, compute_disparity
from fairfuzzkv_codec.evaluation.gate2 import (
    Gate2Decision,
    RunComparison,
    decide_gate2,
    paired_bootstrap_worst_benefit,
    run_comparison_from_records,
)
from fairfuzzkv_codec.evaluation.isolation import (
    PredictionRecord,
    cohort_counts,
    degradation_per_cohort,
    full_correct_ids,
    isolate,
)


def _rec(eid, cohort, system, correct):
    return PredictionRecord(example_id=eid, cohort=cohort, system=system, correct=correct)


# ---- isolation -------------------------------------------------------------

def test_isolation_keeps_only_full_correct_examples():
    records = [
        _rec("e1", "A", "full", True),
        _rec("e1", "A", "minimax", False),
        _rec("e2", "A", "full", False),  # full wrong -> e2 dropped
        _rec("e2", "A", "minimax", True),
    ]
    iso = isolate(records)
    ids = {r.example_id for r in iso}
    assert ids == {"e1"}


def test_cohort_counts_visible():
    records = [
        _rec("e1", "A", "minimax", True),
        _rec("e2", "A", "minimax", False),
        _rec("e3", "B", "minimax", True),
    ]
    counts = cohort_counts(records, "minimax")
    assert counts == {"A": 2, "B": 1}


def test_degradation_is_fraction_of_full_correct_gone_wrong():
    records = [
        _rec("e1", "A", "full", True), _rec("e1", "A", "minimax", False),
        _rec("e2", "A", "full", True), _rec("e2", "A", "minimax", True),
        _rec("e3", "B", "full", True), _rec("e3", "B", "minimax", True),
    ]
    fc = full_correct_ids(records)
    delta = degradation_per_cohort(records, "minimax", fc)
    assert delta["A"] == 0.5  # 1 of 2 broke
    assert delta["B"] == 0.0


# ---- disparity metrics -----------------------------------------------------

def test_disparity_metrics_bounded_and_consistent():
    rep = compute_disparity({"A": 0.4, "B": 0.1, "C": 0.1})
    assert rep.worst_group_drop == 0.4
    assert rep.best_group_drop == 0.1
    assert abs(rep.absolute_range - 0.3) < 1e-9
    assert rep.cddb == rep.std  # headline == bounded std
    assert 0.0 <= rep.cddb <= 1.0
    assert abs(rep.mean_degradation - 0.2) < 1e-9


def test_guarded_ratio_does_not_explode_at_zero():
    rep = compute_disparity({"A": 0.5, "B": 0.0})
    assert rep.guarded_ratio < 1e6  # floored, not infinite


def test_compare_systems_signs():
    agg = compute_disparity({"A": 0.4, "B": 0.1})  # worst 0.4
    mm = compute_disparity({"A": 0.2, "B": 0.2})   # worst 0.2, more equal
    cmp = compare_systems(agg, mm)
    assert cmp.fairness_benefit_worst == 0.2  # minimax lowered the worst by 0.2
    assert cmp.disparity_reduction_cddb > 0   # minimax more equal
    # aggregate mean 0.25 vs minimax mean 0.20 -> minimax cheaper here (cost<0)
    assert cmp.aggregate_quality_cost < 0


# ---- decision logic --------------------------------------------------------

def _run(benefit, cost, matched=True, seed=0, budget=100):
    return RunComparison(budget, seed, benefit, benefit, cost, matched)


def test_pass_requires_practical_benefit_consistency_ci_and_cost():
    runs = [_run(0.08, 0.02, seed=s) for s in range(5)]
    rep = decide_gate2(runs, ci_low=0.03, ci_high=0.12)
    assert rep.decision == Gate2Decision.PASS


def test_weak_pass_on_small_benefit():
    runs = [_run(0.03, 0.02, seed=s) for s in range(5)]
    rep = decide_gate2(runs, ci_low=0.01, ci_high=0.06)
    assert rep.decision == Gate2Decision.WEAK_PASS


def test_weak_pass_when_ci_includes_zero_despite_practical_point():
    runs = [_run(0.06, 0.02, seed=s) for s in range(5)]
    rep = decide_gate2(runs, ci_low=-0.01, ci_high=0.13)  # CI straddles 0
    assert rep.decision == Gate2Decision.WEAK_PASS


def test_fail_when_no_benefit():
    runs = [_run(-0.01, 0.02, seed=s) for s in range(5)]
    rep = decide_gate2(runs, ci_low=-0.05, ci_high=0.03)
    assert rep.decision == Gate2Decision.FAIL
    assert "NEGATIVE evidence" in rep.reasoning


def test_fail_when_matched_bits_violated():
    runs = [_run(0.10, 0.01, matched=False, seed=s) for s in range(5)]
    rep = decide_gate2(runs, ci_low=0.05, ci_high=0.15)
    assert rep.decision == Gate2Decision.FAIL
    assert "matched-bit" in rep.reasoning


def test_fail_when_quality_cost_too_high():
    runs = [_run(0.08, 0.20, seed=s) for s in range(5)]  # huge average cost
    rep = decide_gate2(runs, ci_low=0.03, ci_high=0.12)
    assert rep.decision != Gate2Decision.PASS


def test_decision_is_deterministic():
    runs = [_run(0.08, 0.02, seed=s) for s in range(5)]
    a = decide_gate2(runs, 0.03, 0.12)
    b = decide_gate2(runs, 0.03, 0.12)
    assert a.decision == b.decision and a.reasoning == b.reasoning


# ---- bootstrap + end to end on synthetic records ---------------------------

def test_bootstrap_and_run_comparison_end_to_end():
    records = []
    # cohort A: minimax fixes 2 of 3 that aggregate broke; cohort B both fine.
    for i in range(3):
        records.append(_rec(f"a{i}", "A", "full", True))
        records.append(_rec(f"a{i}", "A", "aggregate", i == 0))       # aggregate breaks a1,a2
        records.append(_rec(f"a{i}", "A", "minimax", i != 2))          # minimax breaks only a2
    for i in range(3):
        records.append(_rec(f"b{i}", "B", "full", True))
        records.append(_rec(f"b{i}", "B", "aggregate", True))
        records.append(_rec(f"b{i}", "B", "minimax", True))

    iso = isolate(records)
    cmp = run_comparison_from_records(iso, budget=100, seed=0, matched_bits_ok=True)
    assert cmp.fairness_benefit_worst > 0  # minimax helped cohort A's worst degradation

    ids = sorted({r.example_id for r in iso})
    point, lo, hi = paired_bootstrap_worst_benefit(iso, ids, n_boot=200, seed=1)
    assert lo <= point <= hi
