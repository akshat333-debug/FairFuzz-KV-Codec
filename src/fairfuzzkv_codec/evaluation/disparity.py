"""Robust cross-cohort degradation-disparity metrics for Gate 2.

Compression-induced degradation for cohort l at budget B:

    Delta_{l,B} = acc_full(l) - acc_system(l)

computed on the intersection-full-correct subset (see isolation.py), where
acc_full(l) == 1.0 by construction, so Delta_{l,B} is exactly the fraction of
base-model-correct examples in cohort l that the compressed system got wrong.

Ratio disparity metrics (max/min) explode when a denominator is near zero, so
the headline CDDB is defined as the (bounded) standard deviation of the
per-cohort degradations, and we additionally report absolute range, worst-group
drop, mean, and a risk-coverage view - never a bare ratio as the sole metric.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DisparityReport:
    delta_per_cohort: Dict[str, float]
    worst_group_drop: float  # max_l Delta_l  (the cohort hurt most)
    best_group_drop: float  # min_l Delta_l
    absolute_range: float  # max - min  (bounded in [0,1])
    std: float  # population std of Delta_l
    mean_degradation: float  # mean_l Delta_l  (aggregate quality cost proxy)
    cddb: float  # headline bounded disparity == std (never explodes near zero)
    guarded_ratio: float  # max/min with a floor - reported but flagged unstable
    risk_coverage: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "delta_per_cohort": self.delta_per_cohort,
            "worst_group_drop": self.worst_group_drop,
            "best_group_drop": self.best_group_drop,
            "absolute_range": self.absolute_range,
            "std": self.std,
            "mean_degradation": self.mean_degradation,
            "cddb": self.cddb,
            "guarded_ratio": self.guarded_ratio,
            "risk_coverage": self.risk_coverage,
        }


def _population_std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def compute_disparity(delta_per_cohort: Dict[str, float], ratio_floor: float = 1e-3) -> DisparityReport:
    """Build the full robust disparity view from per-cohort degradations."""
    if not delta_per_cohort:
        return DisparityReport({}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, [])
    deltas = list(delta_per_cohort.values())
    worst = max(deltas)
    best = min(deltas)
    mean = sum(deltas) / len(deltas)
    std = _population_std(deltas)

    # risk-coverage: sort cohorts best->worst degradation, report cumulative mean
    # degradation as coverage grows (the "if we only served the safest fraction
    # of cohorts, what disparity remains" view).
    ordered = sorted(deltas)
    risk_coverage: List[Tuple[float, float]] = []
    running = 0.0
    for i, d in enumerate(ordered, start=1):
        running += d
        risk_coverage.append((i / len(ordered), running / i))

    return DisparityReport(
        delta_per_cohort=dict(delta_per_cohort),
        worst_group_drop=worst,
        best_group_drop=best,
        absolute_range=worst - best,
        std=std,
        mean_degradation=mean,
        cddb=std,  # bounded headline disparity
        guarded_ratio=worst / max(best, ratio_floor),
        risk_coverage=risk_coverage,
    )


@dataclass
class SystemComparison:
    """aggregate (control) vs minimax at one matched budget."""

    fairness_benefit_worst: float  # aggregate.worst - minimax.worst  (>0 = minimax helps worst cohort)
    disparity_reduction_cddb: float  # aggregate.cddb - minimax.cddb   (>0 = minimax reduces disparity)
    aggregate_quality_cost: float  # minimax.mean - aggregate.mean     (>0 = minimax costs average)

    def to_dict(self) -> Dict[str, float]:
        return {
            "fairness_benefit_worst": self.fairness_benefit_worst,
            "disparity_reduction_cddb": self.disparity_reduction_cddb,
            "aggregate_quality_cost": self.aggregate_quality_cost,
        }


def compare_systems(aggregate: DisparityReport, minimax: DisparityReport) -> SystemComparison:
    return SystemComparison(
        fairness_benefit_worst=aggregate.worst_group_drop - minimax.worst_group_drop,
        disparity_reduction_cddb=aggregate.cddb - minimax.cddb,
        aggregate_quality_cost=minimax.mean_degradation - aggregate.mean_degradation,
    )
