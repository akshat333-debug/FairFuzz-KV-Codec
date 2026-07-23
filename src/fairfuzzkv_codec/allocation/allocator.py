"""Aggregate rate-distortion allocator (the Gate-2 control condition).

Chooses one quantizer option per cohort to minimize total distortion
    minimize   sum_l  D_l(b_l)
    subject to sum_l  cost_l(b_l)  <=  B
where cost includes per-element bits AND fixed serialized overhead (LBG codebook,
metadata). This is a multiple-choice knapsack:
  * `solve_exact`  - dynamic programming reference (optimal, small instances).
  * `solve_greedy` - marginal-gain water-filling (scalable, large instances).
`optimality_gap` compares the two so the approximation is validated, not assumed.

This allocator is the aggregate-optimal baseline the fairness method must beat -
it is implemented strongly and fairly, with no cohort favoritism.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

INFEASIBLE = float("inf")


@dataclass(frozen=True)
class BitOption:
    """One quantizer choice for a cohort. `total_bits` is the FULL serialized
    cost of encoding this cohort with this option (per-element bits * n_elements
    + fixed overhead like an LBG codebook), so nothing is uncounted."""

    label: str
    total_bits: int
    distortion: float


@dataclass
class Cohort:
    cohort_id: str
    options: List[BitOption]
    min_service_label: Optional[str] = None  # if set, options below it are disallowed

    def allowed_options(self) -> List[BitOption]:
        if self.min_service_label is None:
            return list(self.options)
        # a "min service level" forbids options CHEAPER (fewer bits) than the named one.
        floor = next(o for o in self.options if o.label == self.min_service_label)
        return [o for o in self.options if o.total_bits >= floor.total_bits]


@dataclass
class Allocation:
    choice: Dict[str, BitOption] = field(default_factory=dict)
    total_bits: int = 0
    total_distortion: float = 0.0
    feasible: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "choice": {c: o.label for c, o in self.choice.items()},
            "total_bits": self.total_bits,
            "total_distortion": self.total_distortion,
            "feasible": self.feasible,
        }


def _min_cost(cohort: Cohort) -> int:
    opts = cohort.allowed_options()
    if not opts:
        raise ValueError(f"cohort {cohort.cohort_id} has no allowed options")
    return min(o.total_bits for o in opts)


def solve_exact(cohorts: List[Cohort], budget: int) -> Allocation:
    """Optimal multiple-choice knapsack by DP over integer bit-budget. Empty
    cohorts (no options) are rejected upstream; a budget below the sum of
    minimum service levels is reported infeasible rather than fudged."""
    if not cohorts:
        return Allocation(feasible=True)
    floor = sum(_min_cost(c) for c in cohorts)
    if floor > budget:
        return Allocation(feasible=False, total_bits=floor, total_distortion=INFEASIBLE)

    # dp[b] = (min total distortion using cohorts so far with exactly-<= b bits,
    #          chosen options). Iterate cohorts, carrying best distortion per budget.
    NEG = None
    dp: List[Optional[float]] = [None] * (budget + 1)
    back: List[Optional[Dict[str, BitOption]]] = [None] * (budget + 1)
    dp[0] = 0.0
    back[0] = {}

    for cohort in cohorts:
        ndp: List[Optional[float]] = [None] * (budget + 1)
        nback: List[Optional[Dict[str, BitOption]]] = [None] * (budget + 1)
        for b in range(budget + 1):
            if dp[b] is None:
                continue
            for opt in cohort.allowed_options():
                nb = b + opt.total_bits
                if nb > budget:
                    continue
                cand = dp[b] + opt.distortion  # type: ignore[operator]
                if ndp[nb] is None or cand < ndp[nb]:  # type: ignore[operator]
                    ndp[nb] = cand
                    chosen = dict(back[b])  # type: ignore[arg-type]
                    chosen[cohort.cohort_id] = opt
                    nback[nb] = chosen
        dp, back = ndp, nback
        _ = NEG

    best_b, best_d = None, None
    for b in range(budget + 1):
        if dp[b] is not None and (best_d is None or dp[b] < best_d):  # type: ignore[operator]
            best_b, best_d = b, dp[b]
    if best_b is None:
        return Allocation(feasible=False, total_distortion=INFEASIBLE)
    choice = back[best_b] or {}
    return Allocation(choice=choice, total_bits=best_b, total_distortion=best_d or 0.0, feasible=True)


def solve_greedy(cohorts: List[Cohort], budget: int) -> Allocation:
    """Water-filling: start every cohort at its cheapest allowed option, then
    repeatedly apply the upgrade with the best distortion-reduction-per-extra-bit
    until no affordable upgrade remains. O(cohorts * options * upgrades)."""
    if not cohorts:
        return Allocation(feasible=True)
    floor = sum(_min_cost(c) for c in cohorts)
    if floor > budget:
        return Allocation(feasible=False, total_bits=floor, total_distortion=INFEASIBLE)

    # initialize at cheapest allowed option per cohort
    choice: Dict[str, BitOption] = {}
    for c in cohorts:
        opts = c.allowed_options()
        choice[c.cohort_id] = min(opts, key=lambda o: o.total_bits)
    used = sum(o.total_bits for o in choice.values())

    by_id = {c.cohort_id: c for c in cohorts}
    while True:
        best_gain = 0.0
        best_move = None  # (cohort_id, option)
        for cid, cur in choice.items():
            for opt in by_id[cid].allowed_options():
                extra = opt.total_bits - cur.total_bits
                if extra <= 0:
                    continue
                if used + extra > budget:
                    continue
                gain = (cur.distortion - opt.distortion) / extra
                if gain > best_gain:
                    best_gain = gain
                    best_move = (cid, opt)
        if best_move is None:
            break
        cid, opt = best_move
        used += opt.total_bits - choice[cid].total_bits
        choice[cid] = opt

    total_d = sum(o.distortion for o in choice.values())
    return Allocation(choice=choice, total_bits=used, total_distortion=total_d, feasible=True)


def optimality_gap(exact: Allocation, approx: Allocation) -> float:
    """Relative distortion gap (approx - exact) / exact. 0 == approx matched the
    optimum. Both must be feasible."""
    if not exact.feasible or not approx.feasible:
        return INFEASIBLE
    if exact.total_distortion <= 0:
        return abs(approx.total_distortion - exact.total_distortion)
    return (approx.total_distortion - exact.total_distortion) / exact.total_distortion
