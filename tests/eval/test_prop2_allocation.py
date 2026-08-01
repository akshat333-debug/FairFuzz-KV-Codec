"""Proposition 2 (Allocation Optimality).

Stated precisely:

    P2. The aggregate allocator returns a bit assignment that minimizes total
        distortion subject to the complete serialized bit budget; the scalable
        approximation stays within a bounded optimality gap of that optimum;
        and no returned allocation ever exceeds the budget. The minimax
        allocator likewise attains the optimal worst-cohort distortion, which
        is monotone non-increasing in budget.

P2 concerns SOLVER correctness - that the optimizer solves the problem it
claims to solve. It is deliberately NOT a claim that the allocation improves
downstream fairness: Gate 2 tested that and returned FAIL (RISK_REGISTER R-09).
"""

import random

import pytest

from fairfuzzkv_codec.allocation.allocator import (
    BitOption,
    Cohort,
    optimality_gap,
    solve_exact,
    solve_greedy,
)
from fairfuzzkv_codec.allocation.minimax import solve_minimax_exact, solve_minimax_waterfill


def _cohort(cid, opts):
    return Cohort(cohort_id=cid, options=[BitOption(label=lbl, total_bits=b, distortion=d) for lbl, b, d in opts])


def _brute_force_best(cohorts, budget):
    """Independent reference: enumerate every combination. Exponential, so only
    used on tiny instances - but it depends on NO code under test, which is the
    point of an optimality proof."""
    import itertools

    best = None
    for combo in itertools.product(*[c.allowed_options() for c in cohorts]):
        total_bits = sum(o.total_bits for o in combo)
        if total_bits > budget:
            continue
        total_d = sum(o.distortion for o in combo)
        if best is None or total_d < best:
            best = total_d
    return best


def _random_instance(rng, n_cohorts):
    cohorts = []
    for i in range(n_cohorts):
        bits = sorted(rng.sample(range(1, 9), rng.randint(2, 3)))
        opts = [(f"o{j}", b, round(6.0 / b + rng.random(), 3)) for j, b in enumerate(bits)]
        cohorts.append(_cohort(f"C{i}", opts))
    return cohorts


def test_p2_exact_solver_matches_independent_brute_force():
    """The core optimality claim, checked against exhaustive enumeration."""
    rng = random.Random(0)
    for _ in range(120):
        cohorts = _random_instance(rng, rng.randint(1, 4))
        floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
        budget = floor + rng.randint(0, 8)

        result = solve_exact(cohorts, budget)
        reference = _brute_force_best(cohorts, budget)

        if reference is None:
            assert not result.feasible
        else:
            assert result.feasible
            assert result.total_distortion == pytest.approx(reference, abs=1e-9)


def test_p2_no_allocation_ever_exceeds_the_budget():
    rng = random.Random(1)
    for _ in range(120):
        cohorts = _random_instance(rng, rng.randint(1, 5))
        floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
        budget = floor + rng.randint(0, 10)
        for solver in (solve_exact, solve_greedy):
            r = solver(cohorts, budget)
            if r.feasible:
                assert r.total_bits <= budget


def test_p2_greedy_stays_within_a_bounded_optimality_gap():
    rng = random.Random(2)
    worst_gap = 0.0
    for _ in range(120):
        cohorts = _random_instance(rng, rng.randint(1, 4))
        floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
        budget = floor + rng.randint(0, 8)
        exact, approx = solve_exact(cohorts, budget), solve_greedy(cohorts, budget)
        if exact.feasible and approx.feasible:
            # the approximation can never BEAT the optimum
            assert approx.total_distortion >= exact.total_distortion - 1e-9
            worst_gap = max(worst_gap, optimality_gap(exact, approx))
    assert worst_gap < 0.5, f"greedy gap exceeded the documented bound: {worst_gap}"


def test_p2_total_distortion_is_monotone_non_increasing_in_budget():
    """More budget must never force a worse optimum."""
    rng = random.Random(3)
    cohorts = _random_instance(rng, 4)
    floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
    previous = None
    for extra in range(0, 16):
        r = solve_exact(cohorts, floor + extra)
        if not r.feasible:
            continue
        if previous is not None:
            assert r.total_distortion <= previous + 1e-9
        previous = r.total_distortion


def test_p2_minimax_attains_optimal_worst_cohort_distortion():
    """The minimax analogue: the exact solver's worst-cohort value must be the
    best achievable, so no feasible assignment beats it."""
    import itertools

    rng = random.Random(4)
    for _ in range(60):
        cohorts = _random_instance(rng, rng.randint(1, 3))
        floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
        budget = floor + rng.randint(0, 6)

        result = solve_minimax_exact(cohorts, budget)
        if not result.feasible:
            continue

        best_worst = None
        for combo in itertools.product(*[c.allowed_options() for c in cohorts]):
            if sum(o.total_bits for o in combo) > budget:
                continue
            worst = max(o.distortion for o in combo)
            if best_worst is None or worst < best_worst:
                best_worst = worst
        assert best_worst is not None
        assert result.worst_distortion == pytest.approx(best_worst, abs=1e-9)


def test_p2_minimax_worst_case_is_monotone_non_increasing_in_budget():
    rng = random.Random(5)
    cohorts = _random_instance(rng, 4)
    floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
    previous = None
    for extra in range(0, 16):
        r = solve_minimax_exact(cohorts, floor + extra)
        if not r.feasible:
            continue
        if previous is not None:
            assert r.worst_distortion <= previous + 1e-9
        previous = r.worst_distortion


def test_p2_minimax_production_solver_never_beats_the_exact_optimum():
    rng = random.Random(6)
    for _ in range(60):
        cohorts = _random_instance(rng, rng.randint(1, 4))
        floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
        budget = floor + rng.randint(0, 6)
        exact = solve_minimax_exact(cohorts, budget)
        production = solve_minimax_waterfill(cohorts, budget)
        if exact.feasible and production.feasible:
            assert production.worst_distortion >= exact.worst_distortion - 1e-9
            assert production.allocation.total_bits <= budget


def test_p2_does_not_imply_the_gate2_fairness_claim():
    """Solver optimality is not a fairness result. Guard the ledger wording."""
    from pathlib import Path

    ledger = Path("CLAIMS_LEDGER.md")
    if not ledger.exists():
        pytest.skip("CLAIMS_LEDGER.md not reachable from this working directory")
    c21 = next(
        (line for line in ledger.read_text(encoding="utf-8").splitlines() if line.startswith("| C-21 (Gate 2)")),
        None,
    )
    assert c21 is not None
    assert "FAIL" in c21
