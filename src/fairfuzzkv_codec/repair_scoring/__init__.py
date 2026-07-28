"""Module 3 - Fuzzy Repair-Priority Scorer and Simpler Competitor Suite.

Scores repair CANDIDATES (evicted-or-kept surface groups, one score per
candidate) for how worth-reintroducing they are during Prompt 9's
budget-neutral repair swap (`fairfuzzkv_codec.pruning.repair.RepairContract`).
This module is an alternate SCORER only - it does not change the repair mass
condition, the codec, or the local bound validator; if it is removed the
codec is byte-identical (Prompt 13 non-negotiable instruction).

Four interchangeable scorers, selectable via `ablation.ScorerConfig`:
- `fuzzy`: a documented, inspectable Mamdani fuzzy-inference system (real
  triangular membership functions + rule base + centroid defuzzification -
  not a neural network renamed "fuzzy").
- `monotone`: a plain weighted sum (generalizes Prompt 9's `repair_score`
  with staleness/uncertainty).
- `logistic`: a sigmoid-transformed weighted sum. NOT a fitted classifier -
  no ground-truth repair-outcome labels exist to fit/validate against
  without fabricating a result.
- `knapsack`: classic greedy value/cost ratio (completion_cost as the cost).

All four consume the SAME `ScorerInputs` (fragility, evidence_importance,
completion_cost, staleness, optional uncertainty), normalized with
train-only statistics (`inputs.fit_input_normalizers`), so comparisons are
apples-to-apples under identical candidates and budgets.
"""

from fairfuzzkv_codec.repair_scoring.ablation import ScorerConfig, ScorerType, run_ablation, score_candidates
from fairfuzzkv_codec.repair_scoring.competitors import knapsack_value_cost_ratio, logistic_score, monotone_weighted_score
from fairfuzzkv_codec.repair_scoring.fuzzy import DEFAULT_RULES, FuzzyResult, Rule, fuzzy_priority_scores, fuzzy_repair_priority
from fairfuzzkv_codec.repair_scoring.inputs import NormalizationStats, ScorerInputs, fit_input_normalizers, normalize_inputs
from fairfuzzkv_codec.repair_scoring.integration import propose_repair_swap
from fairfuzzkv_codec.repair_scoring.sensitivity import ComplexityReport, measure_complexity, sensitivity_to_breakpoints, sensitivity_to_rules

__all__ = [
    "ScorerConfig", "ScorerType", "run_ablation", "score_candidates",
    "knapsack_value_cost_ratio", "logistic_score", "monotone_weighted_score",
    "DEFAULT_RULES", "FuzzyResult", "Rule", "fuzzy_priority_scores", "fuzzy_repair_priority",
    "NormalizationStats", "ScorerInputs", "fit_input_normalizers", "normalize_inputs",
    "propose_repair_swap",
    "ComplexityReport", "measure_complexity", "sensitivity_to_breakpoints", "sensitivity_to_rules",
]
