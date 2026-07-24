"""Gate 2 decision: does minimax allocation meaningfully reduce cross-cohort
compression-degradation disparity vs aggregate-optimal, at matched bits?

PRE-REGISTERED thresholds - fixed HERE, committed, and tested against synthetic
fixtures BEFORE this module is pointed at any real study. Per Prompt 12's
non-negotiable: this decides whether the Q1 fairness thesis survives, so the
PASS/WEAK_PASS/FAIL logic must not be rewritten after seeing results. If Gate 2
FAILs, the codec is preserved and minimax is reported as NEGATIVE evidence -
never fabricated into a fairness claim.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from fairfuzzkv_codec.evaluation.disparity import compare_systems, compute_disparity
from fairfuzzkv_codec.evaluation.isolation import (
    PredictionRecord,
    degradation_per_cohort,
    full_correct_ids,
)

# =============================================================================
# PRE-REGISTERED Gate 2 thresholds (frozen). Do not edit after a real run.
# =============================================================================

# Minimax must cut the WORST-cohort degradation by at least this (absolute) to
# count as a practically meaningful fairness benefit - p-values alone insufficient.
WORST_IMPROVE_PRACTICAL = 0.05

# Below practical but at/above this: a real-but-small directional signal.
WORST_IMPROVE_WEAK = 0.02

# Minimax's average (aggregate) degradation may exceed the aggregate control's by
# at most this - beyond it, the fairness benefit is not worth the quality cost.
MAX_AGGREGATE_COST = 0.05

# Fraction of (budget, seed) runs whose worst-cohort benefit must be positive for
# the effect to be "directionally consistent".
DIRECTIONAL_CONSISTENCY = 0.8


class Gate2Decision(str, Enum):
    PASS = "PASS"
    WEAK_PASS = "WEAK_PASS"
    FAIL = "FAIL"


@dataclass
class RunComparison:
    budget: int
    seed: int
    fairness_benefit_worst: float
    disparity_reduction_cddb: float
    aggregate_quality_cost: float
    matched_bits_ok: bool


@dataclass
class Gate2Report:
    decision: Gate2Decision
    reasoning: str
    mean_fairness_benefit_worst: float
    mean_aggregate_quality_cost: float
    directional_consistency: float
    ci_low: float
    ci_high: float
    runs: List[RunComparison] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "mean_fairness_benefit_worst": self.mean_fairness_benefit_worst,
            "mean_aggregate_quality_cost": self.mean_aggregate_quality_cost,
            "directional_consistency": self.directional_consistency,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "runs": [r.__dict__ for r in self.runs],
        }


def paired_bootstrap_worst_benefit(
    records: List[PredictionRecord],
    example_ids: List[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Paired bootstrap over examples of the worst-cohort fairness benefit
    (aggregate.worst - minimax.worst). Resamples example ids WITH replacement,
    recomputes both systems' per-cohort degradation, and returns
    (point_estimate, ci_low, ci_high) at 95%."""
    rng = random.Random(seed)
    ids = list(example_ids)

    def _benefit(sample_ids: List[str]) -> float:
        subset = [r for r in records if r.example_id in set(sample_ids)]
        fc = full_correct_ids(subset)
        agg = compute_disparity(degradation_per_cohort(subset, "aggregate", fc))
        mm = compute_disparity(degradation_per_cohort(subset, "minimax", fc))
        return agg.worst_group_drop - mm.worst_group_drop

    point = _benefit(ids)
    stats = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        stats.append(_benefit(sample))
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[int(0.975 * len(stats)) - 1] if len(stats) > 1 else point
    return point, lo, hi


def decide_gate2(runs: List[RunComparison], ci_low: float, ci_high: float) -> Gate2Report:
    """Apply the frozen thresholds to the per-run comparisons + bootstrap CI."""
    if not runs:
        return Gate2Report(Gate2Decision.FAIL, "no runs", 0.0, 0.0, 0.0, ci_low, ci_high)

    mean_benefit = sum(r.fairness_benefit_worst for r in runs) / len(runs)
    mean_cost = sum(r.aggregate_quality_cost for r in runs) / len(runs)
    consistency = sum(1 for r in runs if r.fairness_benefit_worst > 0) / len(runs)
    all_matched = all(r.matched_bits_ok for r in runs)

    if not all_matched:
        return Gate2Report(
            Gate2Decision.FAIL,
            "matched-bit tolerance violated in at least one run - comparison invalid",
            mean_benefit, mean_cost, consistency, ci_low, ci_high, runs,
        )

    cost_ok = mean_cost <= MAX_AGGREGATE_COST
    consistent = consistency >= DIRECTIONAL_CONSISTENCY
    ci_excludes_zero = ci_low > 0

    if (
        mean_benefit >= WORST_IMPROVE_PRACTICAL
        and consistent
        and ci_excludes_zero
        and cost_ok
    ):
        decision = Gate2Decision.PASS
        reason = (
            f"minimax cut worst-cohort degradation by {mean_benefit:.3f} "
            f"(>= {WORST_IMPROVE_PRACTICAL}), directionally consistent "
            f"({consistency:.0%}), 95% CI [{ci_low:.3f}, {ci_high:.3f}] excludes 0, "
            f"average-quality cost {mean_cost:.3f} within budget ({MAX_AGGREGATE_COST})."
        )
    elif mean_benefit >= WORST_IMPROVE_WEAK and consistency >= 0.5 and cost_ok:
        decision = Gate2Decision.WEAK_PASS
        reason = (
            f"minimax shows a real but modest worst-cohort benefit "
            f"({mean_benefit:.3f}, between {WORST_IMPROVE_WEAK} and "
            f"{WORST_IMPROVE_PRACTICAL}) or a CI that includes 0 "
            f"(CI [{ci_low:.3f}, {ci_high:.3f}]); consistency {consistency:.0%}, "
            f"cost {mean_cost:.3f}. Below the practical fairness bar."
        )
    else:
        decision = Gate2Decision.FAIL
        reason = (
            f"minimax did not meaningfully reduce worst-cohort degradation "
            f"(benefit {mean_benefit:.3f}, consistency {consistency:.0%}, "
            f"cost {mean_cost:.3f}). Codec is preserved; minimax is reported as "
            f"NEGATIVE evidence for the fairness thesis, not a validated claim."
        )
    return Gate2Report(decision, reason, mean_benefit, mean_cost, consistency, ci_low, ci_high, runs)


def run_comparison_from_records(
    records: List[PredictionRecord], budget: int, seed: int, matched_bits_ok: bool
) -> RunComparison:
    """Compute one (budget, seed) aggregate-vs-minimax comparison on the
    intersection-full-correct subset of a set of prediction records."""
    fc = full_correct_ids(records)
    agg = compute_disparity(degradation_per_cohort(records, "aggregate", fc))
    mm = compute_disparity(degradation_per_cohort(records, "minimax", fc))
    cmp = compare_systems(agg, mm)
    return RunComparison(
        budget=budget, seed=seed,
        fairness_benefit_worst=cmp.fairness_benefit_worst,
        disparity_reduction_cddb=cmp.disparity_reduction_cddb,
        aggregate_quality_cost=cmp.aggregate_quality_cost,
        matched_bits_ok=matched_bits_ok,
    )
