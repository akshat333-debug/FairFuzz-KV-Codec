import random

from fairfuzzkv_codec.allocation.allocator import (
    BitOption,
    Cohort,
    optimality_gap,
    solve_exact,
    solve_greedy,
)


def _cohort(cid, opts):
    return Cohort(cohort_id=cid, options=[BitOption(label=lbl, total_bits=b, distortion=d) for lbl, b, d in opts])


def test_exact_picks_min_distortion_within_budget():
    cohorts = [
        _cohort("A", [("lo", 2, 10.0), ("hi", 4, 2.0)]),
        _cohort("B", [("lo", 2, 8.0), ("hi", 4, 1.0)]),
    ]
    alloc = solve_exact(cohorts, budget=8)  # can afford both hi
    assert alloc.feasible
    assert alloc.choice["A"].label == "hi" and alloc.choice["B"].label == "hi"
    assert alloc.total_distortion == 3.0
    assert alloc.total_bits <= 8


def test_exact_respects_tight_budget():
    cohorts = [
        _cohort("A", [("lo", 2, 10.0), ("hi", 4, 2.0)]),
        _cohort("B", [("lo", 2, 8.0), ("hi", 4, 1.0)]),
    ]
    # budget 6: can upgrade only one cohort (+2 bits). Upgrading A saves 10->2=8;
    # upgrading B saves 8->1=7. Exact must pick the bigger reduction: A.
    alloc = solve_exact(cohorts, budget=6)
    assert alloc.feasible
    assert alloc.choice["A"].label == "hi" and alloc.choice["B"].label == "lo"
    assert alloc.total_distortion == 10.0  # 2 (A hi) + 8 (B lo)


def test_infeasible_budget_reported_not_fudged():
    cohorts = [_cohort("A", [("lo", 4, 1.0)]), _cohort("B", [("lo", 4, 1.0)])]
    alloc = solve_exact(cohorts, budget=4)  # floor is 8
    assert not alloc.feasible
    assert alloc.total_distortion == float("inf")


def test_min_service_level_forbids_cheap_options():
    cohorts = [_cohort("A", [("lo", 2, 10.0), ("hi", 4, 2.0)])]
    cohorts[0].min_service_label = "hi"  # forbid the cheap 2-bit option
    alloc = solve_exact(cohorts, budget=4)
    assert alloc.choice["A"].label == "hi"
    # and if budget can't even meet the service floor -> infeasible
    tight = solve_exact(cohorts, budget=2)
    assert not tight.feasible


def test_empty_cohort_list_is_feasible_trivial():
    assert solve_exact([], budget=10).feasible
    assert solve_greedy([], budget=10).feasible


def test_greedy_matches_exact_on_random_small_instances():
    rng = random.Random(0)
    max_gap = 0.0
    for _ in range(200):
        n = rng.randint(1, 4)
        cohorts = []
        for i in range(n):
            k = rng.randint(1, 3)
            opts = []
            bits = sorted(rng.sample(range(1, 10), k))
            for j, b in enumerate(bits):
                # distortion decreasing in bits (realistic)
                opts.append((f"o{j}", b, round(10.0 / b + rng.random(), 3)))
            cohorts.append(_cohort(f"C{i}", opts))
        budget = rng.randint(n, 6 * n)
        ex = solve_exact(cohorts, budget)
        gr = solve_greedy(cohorts, budget)
        assert ex.feasible == gr.feasible
        if ex.feasible:
            assert gr.total_bits <= budget
            gap = optimality_gap(ex, gr)
            max_gap = max(max_gap, gap)
    # greedy is a heuristic; on these instances it must stay within a small gap.
    assert max_gap < 0.5, f"greedy optimality gap too large: {max_gap}"


def test_greedy_never_exceeds_budget():
    cohorts = [
        _cohort("A", [("lo", 1, 5.0), ("mid", 3, 2.0), ("hi", 9, 0.5)]),
        _cohort("B", [("lo", 1, 4.0), ("hi", 5, 0.5)]),
    ]
    alloc = solve_greedy(cohorts, budget=7)
    assert alloc.feasible and alloc.total_bits <= 7
