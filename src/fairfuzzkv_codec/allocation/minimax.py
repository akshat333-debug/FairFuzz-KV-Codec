"""Fairness-constrained minimax allocation.

Minimize the WORST-cohort distortion at a fixed complete bit budget - the
central fairness claim. The continuous derivation and its KKT conditions are in
`ALLOCATION_MATH.md`; the key correct statement is that the optimum **equalizes
the achieved distortion `D_l` across active cohorts** (NOT beta - beta_l are
fixed curve parameters, not decision variables).

Two solvers:
  * `solve_minimax_exact`     - epigraph binary search over discrete options,
                                optimal for small instances (reference).
  * `solve_minimax_waterfill` - continuous water-filling then discrete
                                projection, the scalable production path.

Objectives: max distortion (default, frozen), weighted max, and optional hard
minimum-quality constraints. All costs are full serialized bits (incl. codebook
fixed cost), so no allocation can overflow the encoded-bit budget.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from fairfuzzkv_codec.allocation.allocator import Allocation, BitOption, Cohort

INFEASIBLE = float("inf")
_EPS = 1e-9


@dataclass
class MinimaxResult:
    allocation: Allocation
    worst_distortion: float
    average_distortion: float
    equalized_target: float  # t: the common worst-case distortion floor
    feasible: bool
    active_cohorts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "allocation": self.allocation.to_dict(),
            "worst_distortion": self.worst_distortion,
            "average_distortion": self.average_distortion,
            "equalized_target": self.equalized_target,
            "feasible": self.feasible,
            "active_cohorts": self.active_cohorts,
        }


def _weight(weights: Optional[Dict[str, float]], cohort_id: str) -> float:
    return 1.0 if weights is None else weights.get(cohort_id, 1.0)


def _eligible_options(
    cohort: Cohort, min_quality: Optional[Dict[str, float]]
) -> List[BitOption]:
    opts = cohort.allowed_options()
    if min_quality and cohort.cohort_id in min_quality:
        q = min_quality[cohort.cohort_id]
        opts = [o for o in opts if o.distortion <= q + _EPS]
    return opts


# ---- exact reference (epigraph binary search over discrete options) --------

def solve_minimax_exact(
    cohorts: List[Cohort],
    budget: int,
    weights: Optional[Dict[str, float]] = None,
    min_quality: Optional[Dict[str, float]] = None,
) -> MinimaxResult:
    """Optimal discrete minimax. For a candidate worst-case value t (swept over
    the sorted distinct weighted distortions), every cohort picks its CHEAPEST
    option with weighted distortion <= t; the smallest t whose total cost fits
    the budget is the optimum."""
    if not cohorts:
        return MinimaxResult(Allocation(feasible=True), 0.0, 0.0, 0.0, True, [])

    per_cohort: Dict[str, List[BitOption]] = {}
    for c in cohorts:
        opts = _eligible_options(c, min_quality)
        if not opts:  # minimum-quality unmeetable for this cohort
            return _infeasible(cohorts)
        per_cohort[c.cohort_id] = opts

    candidates = sorted({
        _weight(weights, c.cohort_id) * o.distortion
        for c in cohorts for o in per_cohort[c.cohort_id]
    })

    for t in candidates:
        choice: Dict[str, BitOption] = {}
        total = 0
        ok = True
        for c in cohorts:
            feas = [o for o in per_cohort[c.cohort_id]
                    if _weight(weights, c.cohort_id) * o.distortion <= t + _EPS]
            if not feas:
                ok = False
                break
            best = min(feas, key=lambda o: o.total_bits)
            choice[c.cohort_id] = best
            total += best.total_bits
        if ok and total <= budget:
            return _result(cohorts, choice, total, weights)
    return _infeasible(cohorts)


# ---- continuous water-filling ----------------------------------------------

def solve_continuous_minimax(
    alphas: Sequence[float], betas: Sequence[float], budget: float
) -> Tuple[List[float], float, List[int]]:
    """Continuous solution in TOTAL-BITS space (see ALLOCATION_MATH.md).
    Returns (x per cohort, equalized distortion t, active indices)."""
    L = len(alphas)
    active = list(range(L))
    x = [0.0] * L
    Lam = 0.0
    while active:
        s_inv = sum(1.0 / betas[i] for i in active)
        s_lna = sum(math.log(alphas[i]) / betas[i] for i in active)
        Lam = (s_lna - budget) / s_inv
        x = [((math.log(alphas[i]) - Lam) / betas[i]) if i in active else 0.0 for i in range(L)]
        neg = [i for i in active if x[i] < 0]
        if not neg:
            break
        worst = min(neg, key=lambda i: x[i])
        active.remove(worst)
        x[worst] = 0.0
    t = math.exp(Lam) if active else max(alphas)
    return x, t, active


def _fit_alpha_beta(options: List[BitOption]) -> Tuple[float, float]:
    """Fit D = alpha*exp(-beta*x) over (total_bits, distortion) option points via
    log-linear regression. Falls back to a 2-point fit; guards non-positive
    distortion with a tiny floor so the log is defined."""
    from fairfuzzkv_codec.allocation.curves import fit_exponential

    bits = [float(o.total_bits) for o in options]
    dist = [max(o.distortion, _EPS) for o in options]
    fit = fit_exponential(bits, dist)
    if fit is not None and fit.beta > 0:
        return fit.alpha, fit.beta
    # degenerate fallback: assume a mild decay so the cohort still responds to bits
    lo, hi = min(options, key=lambda o: o.total_bits), max(options, key=lambda o: o.total_bits)
    span = max(hi.total_bits - lo.total_bits, 1)
    beta = max((math.log(max(lo.distortion, _EPS)) - math.log(max(hi.distortion, _EPS))) / span, _EPS)
    alpha = max(lo.distortion, _EPS) * math.exp(beta * lo.total_bits)
    return alpha, beta


def solve_minimax_waterfill(
    cohorts: List[Cohort],
    budget: int,
    weights: Optional[Dict[str, float]] = None,
    min_quality: Optional[Dict[str, float]] = None,
) -> MinimaxResult:
    """Production solver: continuous water-filling, then project each cohort to
    its cheapest discrete option meeting the equalized floor, repairing any
    budget overflow. Never returns an allocation over budget."""
    if not cohorts:
        return MinimaxResult(Allocation(feasible=True), 0.0, 0.0, 0.0, True, [])

    per_cohort: Dict[str, List[BitOption]] = {}
    alphas, betas = [], []
    for c in cohorts:
        opts = _eligible_options(c, min_quality)
        if not opts:
            return _infeasible(cohorts)
        per_cohort[c.cohort_id] = opts
        a, b = _fit_alpha_beta(opts)
        alphas.append(a)
        betas.append(b)

    _x, t, _active = solve_continuous_minimax(alphas, betas, float(budget))

    # Discrete water-filling: start every cohort at its cheapest eligible option,
    # then repeatedly pour the remaining budget into the CURRENT WORST cohort by
    # upgrading it to the affordable option that most reduces its (weighted)
    # distortion. Pouring into the worst is exactly the minimax move, and it never
    # exceeds the budget because every upgrade is affordability-checked.
    choice: Dict[str, BitOption] = {
        c.cohort_id: min(per_cohort[c.cohort_id], key=lambda o: o.total_bits) for c in cohorts
    }
    total = sum(o.total_bits for o in choice.values())
    if total > budget:
        return _infeasible(cohorts)  # cannot even meet the floor

    # Spend budget ONLY on the current worst cohort, and take the cheapest step
    # that lowers its distortion (conserving bits for later worst cohorts). If the
    # worst cohort cannot be improved affordably, upgrading any OTHER cohort can
    # only lower a non-max value - it never reduces the worst - so we stop.
    while True:
        worst_c = max(cohorts, key=lambda c: _weight(weights, c.cohort_id) * choice[c.cohort_id].distortion)
        cur = choice[worst_c.cohort_id]
        affordable = [
            o for o in per_cohort[worst_c.cohort_id]
            if o.distortion < cur.distortion - _EPS
            and total + (o.total_bits - cur.total_bits) <= budget
        ]
        if not affordable:
            break
        step = min(affordable, key=lambda o: o.total_bits)  # cheapest distortion-reducing step
        total += step.total_bits - cur.total_bits
        choice[worst_c.cohort_id] = step

    return _result(cohorts, choice, total, weights, equalized_target=t)


# ---- helpers ---------------------------------------------------------------

def _result(
    cohorts: List[Cohort],
    choice: Dict[str, BitOption],
    total: int,
    weights: Optional[Dict[str, float]],
    equalized_target: Optional[float] = None,
) -> MinimaxResult:
    weighted = [_weight(weights, c.cohort_id) * choice[c.cohort_id].distortion for c in cohorts]
    raw = [choice[c.cohort_id].distortion for c in cohorts]
    worst = max(weighted)
    avg = sum(raw) / len(raw)
    active = [c.cohort_id for c in cohorts if choice[c.cohort_id].total_bits > min(
        o.total_bits for o in c.allowed_options()) - 1]
    alloc = Allocation(
        choice=choice, total_bits=total,
        total_distortion=sum(raw), feasible=True,
    )
    return MinimaxResult(
        allocation=alloc, worst_distortion=worst, average_distortion=avg,
        equalized_target=equalized_target if equalized_target is not None else worst,
        feasible=True, active_cohorts=active,
    )


def _infeasible(cohorts: List[Cohort]) -> MinimaxResult:
    return MinimaxResult(
        Allocation(feasible=False, total_distortion=INFEASIBLE),
        worst_distortion=INFEASIBLE, average_distortion=INFEASIBLE,
        equalized_target=INFEASIBLE, feasible=False, active_cohorts=[],
    )


# ---- comparison metrics (item 76) ------------------------------------------

def worst_distortion(alloc: Allocation, cohorts: List[Cohort], weights: Optional[Dict[str, float]] = None) -> float:
    if not alloc.feasible:
        return INFEASIBLE
    return max(_weight(weights, c.cohort_id) * alloc.choice[c.cohort_id].distortion for c in cohorts)


def pareto_frontier(
    cohorts: List[Cohort], budgets: Sequence[int], weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, float]]:
    """Sweep budgets and report (budget, worst, average) for the minimax
    solution - the curve that exposes the cost of fairness rather than hiding it."""
    out: List[Dict[str, float]] = []
    for b in budgets:
        r = solve_minimax_exact(cohorts, b, weights=weights)
        out.append({
            "budget": float(b),
            "worst_distortion": r.worst_distortion,
            "average_distortion": r.average_distortion,
            "feasible": float(r.feasible),
        })
    return out


def allocation_shift(aggregate: Allocation, minimax: Allocation) -> Dict[str, Tuple[str, str]]:
    """Per-cohort (aggregate_choice_label -> minimax_choice_label) where they
    differ - shows how fairness reallocates bits away from the aggregate optimum."""
    shifts: Dict[str, Tuple[str, str]] = {}
    for cid, agg_opt in aggregate.choice.items():
        mm_opt = minimax.choice.get(cid)
        if mm_opt is not None and mm_opt.label != agg_opt.label:
            shifts[cid] = (agg_opt.label, mm_opt.label)
    return shifts
