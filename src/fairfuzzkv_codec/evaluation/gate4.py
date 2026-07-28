"""Gate 4 decision: does the Module 3 fuzzy repair-priority scorer beat
no-repair AND the simpler competitors, consistently, at matched bits?

PRE-REGISTERED thresholds - fixed HERE, committed, and tested against
synthetic fixtures (`tests/evaluation/test_gate4.py`) BEFORE this module is
pointed at any real study, per Prompt 14's non-negotiable instruction: "do
not postpone the naming decision" and "require consistent benefit, not a
single favorable metric." If Gate 4 FAILs, the codec (quantization/pruning/
allocation/format) is preserved; only the fuzzy-scoring claim and the
project name/claims (see `core/naming.py`) change - never fabricated into a
validated benefit.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

# =============================================================================
# PRE-REGISTERED Gate 4 thresholds (frozen). Do not edit after a real run.
# Magnitudes matched to Gate 1 (accuracy gain) and Gate 2 (worst-cohort gain)
# for methodological consistency across the project's gates.
# =============================================================================

PRACTICAL_ACCURACY_GAIN = 0.10
WEAK_ACCURACY_GAIN = 0.03

PRACTICAL_WORST_COHORT_GAIN = 0.05
WEAK_WORST_COHORT_GAIN = 0.02

# Fraction of (budget, seed) runs whose fuzzy-vs-no_repair benefit must be
# positive for the effect to be "directionally consistent".
DIRECTIONAL_CONSISTENCY = 0.8

# fuzzy must not be dominated by the best simple competitor: the paired
# bootstrap CI lower bound on (fuzzy_accuracy - best_simple_accuracy) must
# stay above this (small negative slack absorbs pilot-scale noise; it is
# NOT a free pass - a CI comfortably below zero still fails this check).
SIMPLE_NOT_DOMINATED_CI_FLOOR = -0.05


class Gate4Decision(str, Enum):
    PASS = "PASS"
    WEAK_PASS = "WEAK_PASS"
    FAIL = "FAIL"


@dataclass
class SystemMetrics:
    system: str
    task_accuracy: float
    worst_cohort_degradation: float
    cddb: float
    mean_kv_mse: float
    repair_accept_rate: float  # accepted swaps / attempted swaps; 0.0 for no_repair
    mean_latency_seconds_per_candidate: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "system": self.system, "task_accuracy": self.task_accuracy,
            "worst_cohort_degradation": self.worst_cohort_degradation, "cddb": self.cddb,
            "mean_kv_mse": self.mean_kv_mse, "repair_accept_rate": self.repair_accept_rate,
            "mean_latency_seconds_per_candidate": self.mean_latency_seconds_per_candidate,
        }


@dataclass
class RunComparison:
    budget_retention_ratio: float
    seed: int
    matched_bits_ok: bool
    metrics: Dict[str, SystemMetrics]  # system name -> metrics, for this run

    def fuzzy_vs_norepair_accuracy_gain(self) -> float:
        return self.metrics["fuzzy"].task_accuracy - self.metrics["no_repair"].task_accuracy

    def fuzzy_vs_norepair_worst_cohort_gain(self) -> float:
        return self.metrics["no_repair"].worst_cohort_degradation - self.metrics["fuzzy"].worst_cohort_degradation

    def best_simple_accuracy(self) -> float:
        return max(self.metrics[s].task_accuracy for s in ("monotone", "knapsack", "logistic"))

    def fuzzy_vs_best_simple_accuracy_gain(self) -> float:
        return self.metrics["fuzzy"].task_accuracy - self.best_simple_accuracy()

    def to_dict(self) -> Dict[str, object]:
        return {
            "budget_retention_ratio": self.budget_retention_ratio, "seed": self.seed,
            "matched_bits_ok": self.matched_bits_ok,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "fuzzy_vs_norepair_accuracy_gain": self.fuzzy_vs_norepair_accuracy_gain(),
            "fuzzy_vs_norepair_worst_cohort_gain": self.fuzzy_vs_norepair_worst_cohort_gain(),
            "fuzzy_vs_best_simple_accuracy_gain": self.fuzzy_vs_best_simple_accuracy_gain(),
        }


@dataclass
class Gate4Report:
    decision: Gate4Decision
    reasoning: str
    mean_accuracy_gain: float
    mean_worst_cohort_gain: float
    accuracy_directional_consistency: float
    worst_cohort_directional_consistency: float
    ci_low_vs_best_simple: float
    ci_high_vs_best_simple: float
    failure_notes: List[str] = field(default_factory=list)
    runs: List[RunComparison] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision": self.decision.value, "reasoning": self.reasoning,
            "mean_accuracy_gain": self.mean_accuracy_gain,
            "mean_worst_cohort_gain": self.mean_worst_cohort_gain,
            "accuracy_directional_consistency": self.accuracy_directional_consistency,
            "worst_cohort_directional_consistency": self.worst_cohort_directional_consistency,
            "ci_low_vs_best_simple": self.ci_low_vs_best_simple,
            "ci_high_vs_best_simple": self.ci_high_vs_best_simple,
            "failure_notes": self.failure_notes,
            "runs": [r.to_dict() for r in self.runs],
        }


def paired_bootstrap_fuzzy_vs_best_simple(
    example_outcomes: Dict[str, Dict[str, bool]], n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float, float]:
    """Paired bootstrap over pooled examples (every (group, n_g, budget, seed)
    unit, keyed uniquely) of fuzzy_accuracy - best_simple_accuracy, where
    best_simple is recomputed per resample as the max of monotone/knapsack/
    logistic accuracy on that resample. Returns (point, ci_low, ci_high) at 95%."""
    rng = random.Random(seed)
    ids = list(example_outcomes.keys())

    def _benefit(sample_ids: List[str]) -> float:
        fuzzy_correct = sum(1 for i in sample_ids if example_outcomes[i]["fuzzy"])
        fuzzy_acc = fuzzy_correct / len(sample_ids)
        simple_accs = []
        for system in ("monotone", "knapsack", "logistic"):
            correct = sum(1 for i in sample_ids if example_outcomes[i][system])
            simple_accs.append(correct / len(sample_ids))
        return fuzzy_acc - max(simple_accs)

    point = _benefit(ids)
    stats = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        stats.append(_benefit(sample))
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[int(0.975 * len(stats)) - 1] if len(stats) > 1 else point
    return point, lo, hi


def decide_gate4(runs: List[RunComparison], ci_low_vs_best_simple: float, ci_high_vs_best_simple: float) -> Gate4Report:
    """Pure function of its inputs. Requires CONSISTENT benefit across THREE
    independent checks - accuracy vs no_repair, worst-cohort vs no_repair,
    and non-domination vs the best simple competitor - not a single
    favorable metric (Prompt 14 non-negotiable)."""
    if not runs:
        return Gate4Report(Gate4Decision.FAIL, "no runs", 0.0, 0.0, 0.0, 0.0, ci_low_vs_best_simple, ci_high_vs_best_simple, ["no runs available"])

    failure_notes: List[str] = []
    if not all(r.matched_bits_ok for r in runs):
        failure_notes.append("matched-bit tolerance violated in at least one run - comparison invalid")
        return Gate4Report(
            Gate4Decision.FAIL, "matched-bit tolerance violated - see failure_notes",
            0.0, 0.0, 0.0, 0.0, ci_low_vs_best_simple, ci_high_vs_best_simple, failure_notes, runs,
        )

    accuracy_gains = [r.fuzzy_vs_norepair_accuracy_gain() for r in runs]
    worst_gains = [r.fuzzy_vs_norepair_worst_cohort_gain() for r in runs]
    mean_accuracy_gain = sum(accuracy_gains) / len(accuracy_gains)
    mean_worst_gain = sum(worst_gains) / len(worst_gains)
    accuracy_consistency = sum(1 for g in accuracy_gains if g > 0) / len(accuracy_gains)
    worst_consistency = sum(1 for g in worst_gains if g > 0) / len(worst_gains)

    not_dominated = ci_low_vs_best_simple >= SIMPLE_NOT_DOMINATED_CI_FLOOR

    if not not_dominated:
        failure_notes.append(
            f"fuzzy is dominated by the best simple competitor: 95% CI on "
            f"(fuzzy - best_simple) accuracy is [{ci_low_vs_best_simple:.3f}, "
            f"{ci_high_vs_best_simple:.3f}], below the {SIMPLE_NOT_DOMINATED_CI_FLOOR} floor"
        )
    if mean_worst_gain < 0:
        failure_notes.append(
            f"fuzzy INCREASED worst-cohort degradation on average ({mean_worst_gain:.3f}) - "
            "overprotecting some cohorts at the expense of others (Prompt 14 item 98 failure mode)"
        )

    accuracy_practical = mean_accuracy_gain >= PRACTICAL_ACCURACY_GAIN and accuracy_consistency >= DIRECTIONAL_CONSISTENCY
    worst_practical = mean_worst_gain >= PRACTICAL_WORST_COHORT_GAIN and worst_consistency >= DIRECTIONAL_CONSISTENCY

    accuracy_weak = mean_accuracy_gain >= WEAK_ACCURACY_GAIN and accuracy_consistency >= 0.5
    worst_weak = mean_worst_gain >= WEAK_WORST_COHORT_GAIN and worst_consistency >= 0.5

    if accuracy_practical and worst_practical and not_dominated:
        decision = Gate4Decision.PASS
        reasoning = (
            f"fuzzy beats no_repair on task accuracy (+{mean_accuracy_gain:.3f}, "
            f"{accuracy_consistency:.0%} consistent) AND worst-cohort degradation "
            f"(+{mean_worst_gain:.3f}, {worst_consistency:.0%} consistent), both above "
            f"practical thresholds, and is not dominated by the best simple competitor "
            f"(CI [{ci_low_vs_best_simple:.3f}, {ci_high_vs_best_simple:.3f}])."
        )
    elif accuracy_weak and worst_weak and not_dominated:
        decision = Gate4Decision.WEAK_PASS
        reasoning = (
            f"fuzzy shows a real but modest benefit over no_repair (accuracy "
            f"+{mean_accuracy_gain:.3f}, worst-cohort +{mean_worst_gain:.3f}) and is not "
            f"dominated by simple competitors, but falls below the practical/consistency bar."
        )
    else:
        decision = Gate4Decision.FAIL
        reasoning = (
            f"fuzzy did not show a consistent, practically meaningful benefit over "
            f"no_repair (accuracy gain {mean_accuracy_gain:.3f}, {accuracy_consistency:.0%} "
            f"consistent; worst-cohort gain {mean_worst_gain:.3f}, {worst_consistency:.0%} "
            f"consistent) and/or was dominated by the best simple competitor "
            f"(CI [{ci_low_vs_best_simple:.3f}, {ci_high_vs_best_simple:.3f}]). Codec is "
            f"preserved; fuzzy scoring is reported as NEGATIVE evidence, not a validated claim."
        )
        if not failure_notes:
            failure_notes.append("below practical/consistency thresholds on at least one required check")

    return Gate4Report(
        decision=decision, reasoning=reasoning,
        mean_accuracy_gain=mean_accuracy_gain, mean_worst_cohort_gain=mean_worst_gain,
        accuracy_directional_consistency=accuracy_consistency,
        worst_cohort_directional_consistency=worst_consistency,
        ci_low_vs_best_simple=ci_low_vs_best_simple, ci_high_vs_best_simple=ci_high_vs_best_simple,
        failure_notes=failure_notes, runs=runs,
    )
