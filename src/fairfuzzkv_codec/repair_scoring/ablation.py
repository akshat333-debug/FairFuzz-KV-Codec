"""Ablation registry: select and run any repair-priority scorer through one
config-driven entry point, so every scorer is evaluated reproducibly on
identical candidate groups and budgets (Prompt 13 item 89/92).

No scorer sees anything beyond the shared `ScorerInputs` - all structural
signals (fragility, attention mass, cost, staleness, uncertainty), never a
task label or compression outcome, so no scorer gets privileged access
another lacks.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch

from fairfuzzkv_codec.repair_scoring.competitors import knapsack_value_cost_ratio, logistic_score, monotone_weighted_score
from fairfuzzkv_codec.repair_scoring.fuzzy import DEFAULT_RULES, Breakpoints, INPUT_LEVELS, OUTPUT_LEVELS, Rule, fuzzy_priority_scores
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs


class ScorerType(str, Enum):
    FUZZY = "fuzzy"
    MONOTONE = "monotone"
    LOGISTIC = "logistic"
    KNAPSACK = "knapsack"


@dataclass
class ScorerConfig:
    scorer_type: ScorerType
    weights: Optional[Dict[str, float]] = None
    bias: float = 0.0
    steepness: float = 6.0
    rules: Optional[List[Rule]] = None
    input_levels: Optional[Dict[str, Breakpoints]] = None
    output_levels: Optional[Dict[str, Breakpoints]] = None


def score_candidates(inputs: ScorerInputs, config: ScorerConfig) -> torch.Tensor:
    if config.scorer_type == ScorerType.FUZZY:
        return fuzzy_priority_scores(
            inputs,
            rules=config.rules or DEFAULT_RULES,
            input_levels=config.input_levels or INPUT_LEVELS,
            output_levels=config.output_levels or OUTPUT_LEVELS,
        )
    if config.scorer_type == ScorerType.MONOTONE:
        return monotone_weighted_score(inputs, config.weights)
    if config.scorer_type == ScorerType.LOGISTIC:
        return logistic_score(inputs, config.weights, config.bias, config.steepness)
    if config.scorer_type == ScorerType.KNAPSACK:
        return knapsack_value_cost_ratio(inputs, config.weights)
    raise ValueError(f"unknown scorer_type: {config.scorer_type}")


def run_ablation(inputs: ScorerInputs, configs: Optional[List[ScorerConfig]] = None) -> Dict[str, torch.Tensor]:
    """Score the SAME `inputs` with every configured scorer. Default configs
    cover all four scorer types at their default (equal-weight) settings."""
    configs = configs if configs is not None else [ScorerConfig(t) for t in ScorerType]
    return {config.scorer_type.value: score_candidates(inputs, config) for config in configs}
