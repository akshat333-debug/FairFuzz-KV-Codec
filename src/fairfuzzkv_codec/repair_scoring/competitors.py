"""Simpler (non-fuzzy) repair-priority competitors.

Three scorers, all consuming the same normalized `ScorerInputs` as the fuzzy
engine and all monotone non-decreasing in every field:

- `monotone_weighted_score`: plain weighted sum. Generalizes Prompt 9's
  `pruning.repair.repair_score` to the full 4-5 field input set on the
  shared [0, 1] normalized scale (repair_score itself is left unchanged -
  it is the frozen Prompt 9 deliverable).
- `logistic_score`: a sigmoid-transformed weighted sum - a smooth
  alternative to the hard linear score. This is NOT a fitted logistic
  regression classifier: no ground-truth repair-outcome labels exist yet to
  fit or validate against, and inventing one would fabricate a result (the
  same discipline `fragility_estimation.calibrated_model` applies to its
  proxy label - see PENDING.md for why a fitted model is out of scope here).
- `knapsack_value_cost_ratio`: classic greedy knapsack heuristic,
  value / cost, with `completion_cost` as the cost denominator.

A small calibrated tree/MLP competitor (Prompt 13 item 88, "if justified")
is deliberately NOT implemented for the same reason as `logistic_score`
above: there is no real labeled repair-outcome dataset to validate it
against, and the acceptance gate only requires three non-fuzzy competitors.
"""

from typing import Dict, Optional

import torch

from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs


def _resolve_weights(available: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if weights is None:
        return {name: 1.0 / len(available) for name in available}
    missing = [name for name in weights if name not in available]
    if missing:
        raise ValueError(f"weights reference unknown field(s): {missing}")
    return weights


def monotone_weighted_score(inputs: ScorerInputs, weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    d = inputs.as_dict()
    weights = _resolve_weights(d, weights)
    total = torch.zeros_like(d["fragility"])
    for name, w in weights.items():
        total = total + w * d[name]
    return total


def logistic_score(
    inputs: ScorerInputs,
    weights: Optional[Dict[str, float]] = None,
    bias: float = 0.0,
    steepness: float = 6.0,
) -> torch.Tensor:
    """sigmoid(steepness * (linear - 0.5) + bias), where linear is the same
    weighted sum as `monotone_weighted_score`. `steepness` is a fixed,
    documented constant (not fit) controlling how sharply the score
    saturates toward 0/1 away from the midpoint."""
    linear = monotone_weighted_score(inputs, weights)
    return torch.sigmoid(steepness * (linear - 0.5) + bias)


def knapsack_value_cost_ratio(
    inputs: ScorerInputs,
    value_weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    d = inputs.as_dict()
    value_fields = {name: values for name, values in d.items() if name != "completion_cost"}
    value_weights = _resolve_weights(value_fields, value_weights)
    value = torch.zeros_like(d["fragility"])
    for name, w in value_weights.items():
        value = value + w * value_fields[name]
    cost = d["completion_cost"].clamp(min=eps)
    return value / cost
