from fairfuzzkv_codec.evaluation.gate4 import (
    Gate4Decision, RunComparison, SystemMetrics, decide_gate4, paired_bootstrap_fuzzy_vs_best_simple,
)


def _metrics(system: str, accuracy: float, worst: float, cddb: float = 0.1, accept_rate: float = 0.5, latency: float = 1e-4) -> SystemMetrics:
    return SystemMetrics(
        system=system, task_accuracy=accuracy, worst_cohort_degradation=worst, cddb=cddb,
        mean_kv_mse=0.05, repair_accept_rate=accept_rate, mean_latency_seconds_per_candidate=latency,
    )


def _run(budget: float, seed: int, fuzzy_acc: float, norepair_acc: float, fuzzy_worst: float, norepair_worst: float,
          simple_acc: float = 0.5, matched_bits_ok: bool = True) -> RunComparison:
    return RunComparison(
        budget_retention_ratio=budget, seed=seed, matched_bits_ok=matched_bits_ok,
        metrics={
            "no_repair": _metrics("no_repair", norepair_acc, norepair_worst),
            "fuzzy": _metrics("fuzzy", fuzzy_acc, fuzzy_worst),
            "monotone": _metrics("monotone", simple_acc, 0.3),
            "knapsack": _metrics("knapsack", simple_acc, 0.3),
            "logistic": _metrics("logistic", simple_acc, 0.3),
        },
    )


def test_clear_pass_when_fuzzy_beats_norepair_and_simple_consistently():
    runs = [
        _run(0.3, 42, fuzzy_acc=0.70, norepair_acc=0.50, fuzzy_worst=0.10, norepair_worst=0.30, simple_acc=0.55),
        _run(0.5, 42, fuzzy_acc=0.75, norepair_acc=0.55, fuzzy_worst=0.05, norepair_worst=0.25, simple_acc=0.60),
        _run(0.3, 7, fuzzy_acc=0.68, norepair_acc=0.48, fuzzy_worst=0.12, norepair_worst=0.32, simple_acc=0.52),
        _run(0.5, 7, fuzzy_acc=0.72, norepair_acc=0.52, fuzzy_worst=0.08, norepair_worst=0.28, simple_acc=0.58),
    ]
    report = decide_gate4(runs, ci_low_vs_best_simple=0.05, ci_high_vs_best_simple=0.20)
    assert report.decision == Gate4Decision.PASS
    assert not report.failure_notes


def test_fail_when_no_benefit_over_norepair():
    runs = [
        _run(0.3, 42, fuzzy_acc=0.50, norepair_acc=0.50, fuzzy_worst=0.30, norepair_worst=0.30),
        _run(0.5, 42, fuzzy_acc=0.52, norepair_acc=0.53, fuzzy_worst=0.29, norepair_worst=0.28),
        _run(0.3, 7, fuzzy_acc=0.49, norepair_acc=0.50, fuzzy_worst=0.31, norepair_worst=0.30),
        _run(0.5, 7, fuzzy_acc=0.51, norepair_acc=0.50, fuzzy_worst=0.30, norepair_worst=0.29),
    ]
    report = decide_gate4(runs, ci_low_vs_best_simple=-0.01, ci_high_vs_best_simple=0.02)
    assert report.decision == Gate4Decision.FAIL


def test_fail_when_dominated_by_best_simple_competitor_even_if_beats_norepair():
    # fuzzy beats no_repair handily, but a simple competitor beats fuzzy by even more.
    runs = [
        _run(0.3, 42, fuzzy_acc=0.65, norepair_acc=0.50, fuzzy_worst=0.10, norepair_worst=0.30, simple_acc=0.80),
        _run(0.5, 42, fuzzy_acc=0.68, norepair_acc=0.52, fuzzy_worst=0.09, norepair_worst=0.28, simple_acc=0.82),
        _run(0.3, 7, fuzzy_acc=0.64, norepair_acc=0.49, fuzzy_worst=0.11, norepair_worst=0.31, simple_acc=0.79),
        _run(0.5, 7, fuzzy_acc=0.66, norepair_acc=0.50, fuzzy_worst=0.10, norepair_worst=0.29, simple_acc=0.81),
    ]
    report = decide_gate4(runs, ci_low_vs_best_simple=-0.20, ci_high_vs_best_simple=-0.10)
    assert report.decision == Gate4Decision.FAIL
    assert any("dominated" in note for note in report.failure_notes)


def test_fail_when_fuzzy_overprotects_and_increases_worst_cohort_degradation():
    runs = [
        _run(0.3, 42, fuzzy_acc=0.60, norepair_acc=0.50, fuzzy_worst=0.40, norepair_worst=0.30, simple_acc=0.55),
        _run(0.5, 42, fuzzy_acc=0.62, norepair_acc=0.52, fuzzy_worst=0.38, norepair_worst=0.28, simple_acc=0.57),
        _run(0.3, 7, fuzzy_acc=0.58, norepair_acc=0.49, fuzzy_worst=0.41, norepair_worst=0.31, simple_acc=0.53),
        _run(0.5, 7, fuzzy_acc=0.61, norepair_acc=0.51, fuzzy_worst=0.39, norepair_worst=0.29, simple_acc=0.56),
    ]
    report = decide_gate4(runs, ci_low_vs_best_simple=0.02, ci_high_vs_best_simple=0.10)
    assert report.decision == Gate4Decision.FAIL
    assert any("overprotecting" in note or "INCREASED" in note for note in report.failure_notes)


def test_weak_pass_for_modest_inconsistent_benefit():
    runs = [
        _run(0.3, 42, fuzzy_acc=0.56, norepair_acc=0.50, fuzzy_worst=0.24, norepair_worst=0.30, simple_acc=0.53),
        _run(0.5, 42, fuzzy_acc=0.53, norepair_acc=0.52, fuzzy_worst=0.27, norepair_worst=0.28, simple_acc=0.54),
        _run(0.3, 7, fuzzy_acc=0.55, norepair_acc=0.49, fuzzy_worst=0.25, norepair_worst=0.31, simple_acc=0.51),
        _run(0.5, 7, fuzzy_acc=0.50, norepair_acc=0.51, fuzzy_worst=0.29, norepair_worst=0.29, simple_acc=0.52),
    ]
    report = decide_gate4(runs, ci_low_vs_best_simple=0.0, ci_high_vs_best_simple=0.06)
    assert report.decision in (Gate4Decision.WEAK_PASS, Gate4Decision.FAIL)  # modest/inconsistent, not a clean PASS


def test_matched_bits_violation_forces_fail_regardless_of_metrics():
    runs = [
        _run(0.3, 42, fuzzy_acc=0.90, norepair_acc=0.50, fuzzy_worst=0.01, norepair_worst=0.40, matched_bits_ok=False),
    ]
    report = decide_gate4(runs, ci_low_vs_best_simple=0.2, ci_high_vs_best_simple=0.3)
    assert report.decision == Gate4Decision.FAIL
    assert "matched-bit" in report.failure_notes[0]


def test_no_runs_is_fail_not_a_crash():
    report = decide_gate4([], ci_low_vs_best_simple=0.0, ci_high_vs_best_simple=0.0)
    assert report.decision == Gate4Decision.FAIL


def test_run_comparison_derived_quantities():
    run = _run(0.3, 42, fuzzy_acc=0.7, norepair_acc=0.5, fuzzy_worst=0.1, norepair_worst=0.3, simple_acc=0.6)
    assert abs(run.fuzzy_vs_norepair_accuracy_gain() - 0.2) < 1e-9
    assert abs(run.fuzzy_vs_norepair_worst_cohort_gain() - 0.2) < 1e-9
    assert abs(run.best_simple_accuracy() - 0.6) < 1e-9
    assert abs(run.fuzzy_vs_best_simple_accuracy_gain() - 0.1) < 1e-9


def test_paired_bootstrap_fuzzy_vs_best_simple_positive_when_fuzzy_always_right():
    outcomes = {
        f"ex{i}": {"fuzzy": True, "monotone": False, "knapsack": False, "logistic": False}
        for i in range(20)
    }
    point, lo, hi = paired_bootstrap_fuzzy_vs_best_simple(outcomes, n_boot=200, seed=1)
    assert point == 1.0
    assert lo > 0.5
    assert lo <= hi


def test_paired_bootstrap_fuzzy_vs_best_simple_zero_when_identical():
    outcomes = {
        f"ex{i}": {"fuzzy": i % 2 == 0, "monotone": i % 2 == 0, "knapsack": i % 2 == 0, "logistic": i % 2 == 0}
        for i in range(20)
    }
    point, lo, hi = paired_bootstrap_fuzzy_vs_best_simple(outcomes, n_boot=200, seed=1)
    assert point == 0.0
    assert lo <= 0.0 <= hi


def test_decision_is_deterministic_pure_function():
    runs = [_run(0.3, 42, fuzzy_acc=0.7, norepair_acc=0.5, fuzzy_worst=0.1, norepair_worst=0.3)]
    r1 = decide_gate4(runs, 0.0, 0.1)
    r2 = decide_gate4(runs, 0.0, 0.1)
    assert r1.decision == r2.decision
    assert r1.reasoning == r2.reasoning


def test_gate4_is_reproducible_from_raw_predictions(tmp_path):
    """Prompt 14 acceptance gate: 'Gate 4 result is reproducible from raw
    runs.' The decision must be recomputable from the per-example predictions
    file alone - no model access, no re-running the study."""
    import json

    from fairfuzzkv_codec.evaluation.gate4 import compute_gate4_from_predictions

    rows = []
    # 2 budgets x 1 seed; fuzzy strictly worse than no_repair -> FAIL
    for budget in (0.3, 0.5):
        for i in range(4):
            eid = f"g{i}"
            rows.append({
                "example_id": eid, "n_g": 1, "budget_retention_ratio": budget, "seed": 42,
                "system": "full", "correct": True, "bits_per_element": 16.0,
                "kv_mse": 0.0, "repair_accepted": 0, "repair_attempted": 0,
            })
            for system, correct in (
                ("no_repair", True), ("fuzzy", False),
                ("monotone", True), ("knapsack", True), ("logistic", True),
            ):
                rows.append({
                    "example_id": eid, "n_g": 1, "budget_retention_ratio": budget, "seed": 42,
                    "system": system, "correct": correct, "bits_per_element": 4.0,
                    "kv_mse": 1.0, "repair_accepted": 0, "repair_attempted": 1,
                })

    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    report = compute_gate4_from_predictions(str(path), n_boot=100, seed=0)
    assert report.decision == Gate4Decision.FAIL
    assert report.mean_accuracy_gain < 0  # fuzzy lost to no_repair on every run
    assert len(report.runs) == 2  # both budgets recovered from the raw file
    # and it is deterministic: same file + same seed -> same decision
    again = compute_gate4_from_predictions(str(path), n_boot=100, seed=0)
    assert again.decision == report.decision
    assert again.mean_accuracy_gain == report.mean_accuracy_gain
