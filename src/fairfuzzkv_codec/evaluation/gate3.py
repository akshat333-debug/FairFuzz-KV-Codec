"""Gate 3 decision: do the Gate 1 (fragmentation) and Gate 2 (fairness)
findings reproduce across two materially different tokenizer/model
families?

PRE-REGISTERED decision logic - fixed HERE, committed, and tested against
synthetic fixtures (`tests/evaluation/test_gate3.py`) BEFORE this module is
pointed at any real cross-model study, per Prompt 17's non-negotiable: "a
single-model success is a course result, not a general research claim."

Reproducibility is judged on DECISION CATEGORY (does the qualitative
finding - PASS/WEAK_PASS vs FAIL - agree across families), never on pooled
significance alone (the acceptance gate's own wording) - two families each
individually WEAK_PASS is a reproduced (if modest) finding; one PASS and one
FAIL is not, even if pooling both families' raw examples together would
yield a "significant" combined p-value.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Sequence, Tuple


class Gate3Decision(str, Enum):
    PASS = "PASS"
    WEAK_PASS = "WEAK_PASS"
    FAIL = "FAIL"


# Any decision in this set counts as "the gate found a signal" for
# reproducibility-category comparison; FAIL is "no signal". WEAK_PASS is
# grouped with PASS deliberately - Gate 1's own real Qwen result was
# WEAK_PASS, so requiring a full PASS to "count" would make reproducing
# Gate 1's own historical finding definitionally impossible.
_SIGNAL_DECISIONS = {"PASS", "WEAK_PASS"}


def _category(decision_value: str) -> str:
    return "SIGNAL" if decision_value in _SIGNAL_DECISIONS else "NO_SIGNAL"


@dataclass
class FamilyGateResult:
    """One model/tokenizer family's Gate 1 + Gate 2 decisions, for Gate 3 to
    compare against another family's."""

    model_name: str
    tokenizer_family: str  # e.g. "byte-level BPE" / "SentencePiece"
    gate1_decision: str  # "PASS" | "WEAK_PASS" | "FAIL"
    gate2_decision: str
    gate1_effect_size: float
    gate2_worst_cohort_benefit: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name, "tokenizer_family": self.tokenizer_family,
            "gate1_decision": self.gate1_decision, "gate2_decision": self.gate2_decision,
            "gate1_effect_size": self.gate1_effect_size, "gate2_worst_cohort_benefit": self.gate2_worst_cohort_benefit,
        }


@dataclass
class Gate3Report:
    decision: Gate3Decision
    reasoning: str
    gate1_reproduces: bool
    gate2_reproduces: bool
    cohort_transfer_verdict: str  # "universal" | "model_specific" | "not_analyzed"
    claim_scope_statement: str
    families: List[FamilyGateResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision": self.decision.value, "reasoning": self.reasoning,
            "gate1_reproduces": self.gate1_reproduces, "gate2_reproduces": self.gate2_reproduces,
            "cohort_transfer_verdict": self.cohort_transfer_verdict,
            "claim_scope_statement": self.claim_scope_statement,
            "families": [f.to_dict() for f in self.families],
        }


def _claim_scope_statement(decision: Gate3Decision, cohort_verdict: str) -> str:
    if decision == Gate3Decision.PASS:
        base = (
            "The qualitative Gate 1/Gate 2 pattern reproduces across both tested families. "
            "This still does NOT establish a universal claim beyond these two families/scales - "
            "see each family's own Gate 1/Gate 2 report for the underlying finding's own strength."
        )
    elif decision == Gate3Decision.WEAK_PASS:
        base = (
            "One of Gate 1 or Gate 2 reproduced in category across families, the other did not. "
            "Treat the reproducing gate's finding as provisionally cross-family, and the "
            "non-reproducing one as family-specific until re-tested."
        )
    else:
        base = (
            "Neither Gate 1 nor Gate 2's decision category reproduced across families. "
            "Per the Prompt 17 non-negotiable, do NOT report either finding as a general claim - "
            "narrow scope to the specific model/tokenizer family it was measured on."
        )
    if cohort_verdict == "model_specific":
        base += (
            " Additionally: fragility cohort risk-band assignment does NOT transfer between these "
            "tokenizer families (cross-tokenizer stability below the universal-agreement threshold) - "
            "do not claim a universal risk threshold; cohorts require per-tokenizer recalibration."
        )
    elif cohort_verdict == "universal":
        base += (
            " Cohort risk-band assignment DOES transfer between these two families on the compared "
            "corpus (agreement rate at/above the universal threshold) - still only evidence for these "
            "two families, not a claim of universality beyond them."
        )
    return base


def decide_gate3(family_a: FamilyGateResult, family_b: FamilyGateResult, cohort_transfer_verdict: str = "not_analyzed") -> Gate3Report:
    """Pure function of its inputs - no I/O, no randomness."""
    gate1_reproduces = _category(family_a.gate1_decision) == _category(family_b.gate1_decision)
    gate2_reproduces = _category(family_a.gate2_decision) == _category(family_b.gate2_decision)

    if gate1_reproduces and gate2_reproduces:
        decision = Gate3Decision.PASS
        reasoning = (
            f"Both Gate 1 ({family_a.gate1_decision} vs {family_b.gate1_decision}) and Gate 2 "
            f"({family_a.gate2_decision} vs {family_b.gate2_decision}) land in the same signal/no-signal "
            f"category across {family_a.model_name} and {family_b.model_name}."
        )
    elif gate1_reproduces or gate2_reproduces:
        decision = Gate3Decision.WEAK_PASS
        which = "Gate 1" if gate1_reproduces else "Gate 2"
        other = "Gate 2" if gate1_reproduces else "Gate 1"
        reasoning = f"{which} reproduces in category across families; {other} does not."
    else:
        decision = Gate3Decision.FAIL
        reasoning = (
            f"Neither gate reproduces in category: Gate 1 ({family_a.gate1_decision} vs "
            f"{family_b.gate1_decision}), Gate 2 ({family_a.gate2_decision} vs {family_b.gate2_decision})."
        )

    return Gate3Report(
        decision=decision, reasoning=reasoning, gate1_reproduces=gate1_reproduces, gate2_reproduces=gate2_reproduces,
        cohort_transfer_verdict=cohort_transfer_verdict,
        claim_scope_statement=_claim_scope_statement(decision, cohort_transfer_verdict),
        families=[family_a, family_b],
    )


def hierarchical_bootstrap(
    family_examples: Mapping[str, Sequence[float]], n_boot: int = 2000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Two-level (hierarchical/stratified) bootstrap across model families
    AND examples (Prompt 17 item 118): each replicate resamples WHICH
    families contribute (with replacement), then independently resamples
    each drawn family's own examples (with replacement) before averaging -
    so both levels of variability (across families, and across examples
    within a family) are reflected in the CI, not just pooled example-level
    variance. Returns (point estimate, ci_low, ci_high) at 95%, computed
    over the per-family means (unweighted by family sample size, so an
    over-represented family cannot dominate)."""
    if not family_examples:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    names = list(family_examples.keys())

    def _family_mean(name: str, values: Sequence[float]) -> float:
        return sum(values) / len(values)

    point = sum(_family_mean(n, family_examples[n]) for n in names) / len(names)

    stats: List[float] = []
    for _ in range(n_boot):
        drawn_family_names = [rng.choice(names) for _ in names]
        replicate_means = []
        for name in drawn_family_names:
            values = family_examples[name]
            resampled = [rng.choice(values) for _ in values]
            replicate_means.append(sum(resampled) / len(resampled))
        stats.append(sum(replicate_means) / len(replicate_means))

    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[int(0.975 * len(stats)) - 1] if len(stats) > 1 else point
    return point, lo, hi
