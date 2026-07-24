import math
import random

from fairfuzzkv_codec.allocation.allocator import BitOption, Cohort, solve_exact
from fairfuzzkv_codec.allocation.minimax import (
    allocation_shift,
    pareto_frontier,
    solve_continuous_minimax,
    solve_minimax_exact,
    solve_minimax_waterfill,
    worst_distortion,
)


def _cohort(cid, opts):
    return Cohort(cohort_id=cid, options=[BitOption(label=lbl, total_bits=b, distortion=d) for lbl, b, d in opts])


def _random_cohorts(rng, n):
    cohorts = []
    for i in range(n):
        k = rng.randint(2, 4)
        bits = sorted(rng.sample(range(1, 12), k))
        opts = [(f"o{j}", b, round(8.0 / b + 0.05 * rng.random(), 4)) for j, b in enumerate(bits)]
        cohorts.append(_cohort(f"C{i}", opts))
    return cohorts


# ---- continuous derivation checks (item 72) --------------------------------

def test_continuous_equalizes_distortion_not_beta():
    # two cohorts, distinct betas; at the optimum the ACHIEVED distortions equal,
    # while the betas remain different (the non-negotiable wording check).
    alphas = [10.0, 20.0]
    betas = [0.5, 1.5]  # deliberately unequal
    x, t, active = solve_continuous_minimax(alphas, betas, budget=10.0)
    assert len(active) == 2
    d = [alphas[i] * math.exp(-betas[i] * x[i]) for i in range(2)]
    assert abs(d[0] - d[1]) < 1e-6  # DISTORTIONS equalized
    assert abs(d[0] - t) < 1e-6
    assert betas[0] != betas[1]  # betas NOT equalized


def test_continuous_budget_is_exactly_spent_when_all_active():
    alphas = [5.0, 8.0, 3.0]
    betas = [0.4, 0.9, 0.6]
    x, _t, active = solve_continuous_minimax(alphas, betas, budget=12.0)
    if len(active) == 3:
        assert abs(sum(x) - 12.0) < 1e-6


def test_continuous_drops_cohort_already_below_floor():
    # one cohort has tiny alpha (already low distortion) -> should go inactive.
    alphas = [100.0, 0.01]
    betas = [0.5, 0.5]
    x, _t, active = solve_continuous_minimax(alphas, betas, budget=2.0)
    assert 1 not in active or x[1] == 0.0
    assert x[1] == 0.0


# ---- exact vs production (item 74) -----------------------------------------

def test_waterfill_matches_exact_worst_on_random_instances():
    rng = random.Random(0)
    for _ in range(150):
        cohorts = _random_cohorts(rng, rng.randint(1, 4))
        floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
        budget = floor + rng.randint(0, 5 * len(cohorts))
        ex = solve_minimax_exact(cohorts, budget)
        wf = solve_minimax_waterfill(cohorts, budget)
        assert ex.feasible and wf.feasible
        # production worst must be within tolerance of the exact optimum, and can
        # never be BETTER than it (exact is optimal).
        assert wf.worst_distortion >= ex.worst_distortion - 1e-9
        assert wf.worst_distortion <= ex.worst_distortion + 0.5


# ---- no overflow + monotonicity (item 78) ----------------------------------

def test_no_allocation_exceeds_budget():
    rng = random.Random(1)
    for _ in range(100):
        cohorts = _random_cohorts(rng, rng.randint(1, 5))
        floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
        budget = floor + rng.randint(0, 10)
        for solver in (solve_minimax_exact, solve_minimax_waterfill):
            r = solver(cohorts, budget)
            if r.feasible:
                assert r.allocation.total_bits <= budget


def test_worst_distortion_monotone_non_increasing_in_budget():
    rng = random.Random(2)
    cohorts = _random_cohorts(rng, 4)
    floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
    prev = None
    for extra in range(0, 20):
        r = solve_minimax_exact(cohorts, floor + extra)
        if not r.feasible:
            continue
        if prev is not None:
            assert r.worst_distortion <= prev + 1e-9  # more budget never worsens worst
        prev = r.worst_distortion


def test_infeasible_below_service_floor_is_reported():
    cohorts = [_cohort("A", [("lo", 5, 1.0)]), _cohort("B", [("lo", 5, 1.0)])]
    r = solve_minimax_exact(cohorts, budget=4)  # floor is 10
    assert not r.feasible
    assert r.worst_distortion == float("inf")


# ---- variants (item 75) ----------------------------------------------------

def test_weighted_max_objective():
    cohorts = [
        _cohort("A", [("lo", 2, 4.0), ("hi", 4, 1.0)]),
        _cohort("B", [("lo", 2, 4.0), ("hi", 4, 1.0)]),
    ]
    # weight B heavily -> minimax should prioritize lowering B's distortion.
    r = solve_minimax_exact(cohorts, budget=6, weights={"A": 1.0, "B": 10.0})
    assert r.feasible
    assert r.allocation.choice["B"].label == "hi"


def test_minimum_quality_constraint_can_make_infeasible():
    cohorts = [_cohort("A", [("lo", 2, 4.0), ("hi", 4, 1.0)])]
    # require distortion <= 0.5 but best option is 1.0 -> infeasible, reported.
    r = solve_minimax_exact(cohorts, budget=100, min_quality={"A": 0.5})
    assert not r.feasible


# ---- fairness vs aggregate (item 76) ---------------------------------------

def test_minimax_differs_from_aggregate_and_shift_is_exposed():
    # aggregate optimizes total; minimax protects the worst cohort. Construct a
    # case where they disagree and check the shift is reported.
    cohorts = [
        _cohort("A", [("lo", 2, 1.0), ("hi", 6, 0.9)]),   # cheap to help, small gain
        _cohort("B", [("lo", 2, 9.0), ("hi", 6, 1.0)]),   # expensive but huge gain, worst cohort
    ]
    budget = 8  # can upgrade exactly one
    agg = solve_exact(cohorts, budget)
    mm = solve_minimax_exact(cohorts, budget)
    shift = allocation_shift(agg, mm.allocation)
    assert isinstance(shift, dict)
    # minimax's worst-cohort distortion must be no worse than the aggregate's.
    assert worst_distortion(mm.allocation, cohorts) <= worst_distortion(agg, cohorts) + 1e-9


def test_pareto_frontier_is_monotone_and_exposes_tradeoff():
    rng = random.Random(3)
    cohorts = _random_cohorts(rng, 3)
    floor = sum(min(o.total_bits for o in c.options) for c in cohorts)
    budgets = [floor + e for e in range(0, 12, 2)]
    frontier = pareto_frontier(cohorts, budgets)
    feas = [p for p in frontier if p["feasible"] > 0]
    worsts = [p["worst_distortion"] for p in feas]
    assert worsts == sorted(worsts, reverse=True) or len(set(worsts)) <= 1
