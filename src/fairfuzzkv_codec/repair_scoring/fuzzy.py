"""Mamdani fuzzy inference for repair-priority scoring.

A real, inspectable fuzzy system - triangular membership functions, an
explicit rule base, min/max Mamdani aggregation, and centroid
defuzzification - not a neural network renamed "fuzzy". Every rule and
breakpoint below is fixed and documented (like `transparent_score.py`'s
audit-baseline weights), not learned.

Inputs are already normalized to [0, 1] (see `inputs.normalize_inputs`) and
each carries two membership levels, "low"/"high", as linear ramps (a
degenerate triangle - still a valid, standard fuzzy-set shape):

    low(x)  = triangular(x, 0, 0, 1)   ->  1 - x
    high(x) = triangular(x, 0, 1, 1)   ->  x

The output "priority" variable uses a proper three-level triangular
partition (low/medium/high) so centroid defuzzification is meaningful.

Rule base: the three primary drivers (fragility, evidence_importance,
completion_cost) form a full 2x2x2 = 8-rule core table (every combination of
low/high fires exactly one consequent). staleness and uncertainty are
modulating rules that only ever push priority up when HIGH, at half weight -
this keeps the rule count small enough to read in one sitting (10 rules)
while remaining monotone non-decreasing in every input, matching Prompt 9's
`repair_score` contract ("higher = more worth reintroducing").
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs

Breakpoints = Tuple[float, float, float]

INPUT_LEVELS: Dict[str, Breakpoints] = {
    "low": (0.0, 0.0, 1.0),
    "high": (0.0, 1.0, 1.0),
}

OUTPUT_LEVELS: Dict[str, Breakpoints] = {
    "low": (0.0, 0.0, 0.5),
    "medium": (0.0, 0.5, 1.0),
    "high": (0.5, 1.0, 1.0),
}

OUTPUT_UNIVERSE_SIZE = 101


@dataclass(frozen=True)
class Rule:
    name: str
    antecedents: Dict[str, str]  # var_name -> level_name (must exist in the levels dict used to fuzzify that var)
    consequent: str  # level name in OUTPUT_LEVELS
    weight: float = 1.0


DEFAULT_RULES: List[Rule] = [
    Rule("R1_all_high", {"fragility": "high", "evidence_importance": "high", "completion_cost": "high"}, "high"),
    Rule("R2_frag_evidence_high", {"fragility": "high", "evidence_importance": "high", "completion_cost": "low"}, "high"),
    Rule("R3_frag_cost_high", {"fragility": "high", "evidence_importance": "low", "completion_cost": "high"}, "medium"),
    Rule("R4_frag_high_only", {"fragility": "high", "evidence_importance": "low", "completion_cost": "low"}, "medium"),
    Rule("R5_evidence_cost_high", {"fragility": "low", "evidence_importance": "high", "completion_cost": "high"}, "medium"),
    Rule("R6_evidence_high_only", {"fragility": "low", "evidence_importance": "high", "completion_cost": "low"}, "medium"),
    Rule("R7_cost_high_only", {"fragility": "low", "evidence_importance": "low", "completion_cost": "high"}, "low"),
    Rule("R8_all_low", {"fragility": "low", "evidence_importance": "low", "completion_cost": "low"}, "low"),
    Rule("R9_stale_boost", {"staleness": "high"}, "high", weight=0.5),
    Rule("R10_uncertain_boost", {"uncertainty": "high"}, "medium", weight=0.5),
]


def triangular(x: float, a: float, b: float, c: float) -> float:
    """Standard triangular membership; handles degenerate a==b or b==c ramps."""
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def fuzzify(value: float, levels: Dict[str, Breakpoints]) -> Dict[str, float]:
    return {name: triangular(value, *bp) for name, bp in levels.items()}


@dataclass
class RuleTrace:
    rule_name: str
    firing_strength: float
    consequent_level: str

    def to_dict(self) -> Dict[str, object]:
        return {"rule_name": self.rule_name, "firing_strength": self.firing_strength, "consequent_level": self.consequent_level}


@dataclass
class FuzzyResult:
    priority: float
    rule_trace: List[RuleTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"priority": self.priority, "rule_trace": [t.to_dict() for t in self.rule_trace]}

    def fired_rules(self) -> List[RuleTrace]:
        return [t for t in self.rule_trace if t.firing_strength > 0.0]


def _rule_firing_strength(rule: Rule, fuzzified: Dict[str, Dict[str, float]]) -> float:
    """AND = min over the rule's named antecedents. A rule referencing a
    variable that wasn't supplied (e.g. R10 when uncertainty is None) never
    fires, rather than raising or guessing."""
    degrees = []
    for var, level in rule.antecedents.items():
        if var not in fuzzified:
            return 0.0
        degrees.append(fuzzified[var].get(level, 0.0))
    if not degrees:
        return 0.0
    return min(degrees) * rule.weight


def infer(
    inputs_row: Dict[str, float],
    rules: Optional[List[Rule]] = None,
    input_levels: Optional[Dict[str, Breakpoints]] = None,
    output_levels: Optional[Dict[str, Breakpoints]] = None,
) -> Tuple[float, List[RuleTrace]]:
    """Single-candidate Mamdani inference: fuzzify -> rule firing (min) ->
    aggregate output fuzzy set (max of clipped consequents) -> centroid
    defuzzify. Returns (priority in [0,1], per-rule trace)."""
    rules = rules if rules is not None else DEFAULT_RULES
    input_levels = input_levels if input_levels is not None else INPUT_LEVELS
    output_levels = output_levels if output_levels is not None else OUTPUT_LEVELS

    fuzzified = {var: fuzzify(val, input_levels) for var, val in inputs_row.items()}

    universe = [i / (OUTPUT_UNIVERSE_SIZE - 1) for i in range(OUTPUT_UNIVERSE_SIZE)]
    agg = [0.0] * OUTPUT_UNIVERSE_SIZE
    trace: List[RuleTrace] = []

    for rule in rules:
        strength = _rule_firing_strength(rule, fuzzified)
        trace.append(RuleTrace(rule.name, strength, rule.consequent))
        if strength <= 0.0:
            continue
        bp = output_levels[rule.consequent]
        for i, u in enumerate(universe):
            clipped = min(triangular(u, *bp), strength)
            if clipped > agg[i]:
                agg[i] = clipped

    total_mass = sum(agg)
    if total_mass <= 1e-9:
        # no rule fired at all - report a neutral default rather than a
        # division-by-zero or a fabricated confident score.
        return 0.5, trace

    centroid = sum(u * a for u, a in zip(universe, agg)) / total_mass
    return centroid, trace


def fuzzy_repair_priority(
    inputs: ScorerInputs,
    rules: Optional[List[Rule]] = None,
    input_levels: Optional[Dict[str, Breakpoints]] = None,
    output_levels: Optional[Dict[str, Breakpoints]] = None,
) -> List[FuzzyResult]:
    """Per-candidate fuzzy inference, one FuzzyResult (priority + inspectable
    rule trace) per row of `inputs`."""
    d = inputs.as_dict()
    n = inputs.fragility.shape[0]
    results: List[FuzzyResult] = []
    for i in range(n):
        row = {name: float(values[i].item()) for name, values in d.items()}
        priority, trace = infer(row, rules, input_levels, output_levels)
        results.append(FuzzyResult(priority=priority, rule_trace=trace))
    return results


def fuzzy_priority_scores(
    inputs: ScorerInputs,
    rules: Optional[List[Rule]] = None,
    input_levels: Optional[Dict[str, Breakpoints]] = None,
    output_levels: Optional[Dict[str, Breakpoints]] = None,
) -> torch.Tensor:
    """Priorities only, as a tensor - the shape every competitor scorer
    returns, so fuzzy and non-fuzzy scorers are directly comparable."""
    results = fuzzy_repair_priority(inputs, rules, input_levels, output_levels)
    return torch.tensor([r.priority for r in results], dtype=torch.float32)
