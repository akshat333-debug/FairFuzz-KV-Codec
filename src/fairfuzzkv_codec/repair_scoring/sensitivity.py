"""Sensitivity analysis over fuzzy breakpoints/rules, and scorer complexity
(parameter count, measured latency) for the ablation comparison.
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from fairfuzzkv_codec.repair_scoring.fuzzy import DEFAULT_RULES, INPUT_LEVELS, OUTPUT_LEVELS, Breakpoints, Rule, fuzzy_priority_scores
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs


def sensitivity_to_breakpoints(inputs: ScorerInputs, perturb_delta: float = 0.1) -> Dict[str, float]:
    """For each output-level breakpoint (low.a/b/c, medium.a/b/c, high.a/b/c),
    nudge it by `perturb_delta` (clamped to [0,1]) and measure the mean
    absolute priority shift across all candidates vs the unperturbed rule
    base - a per-breakpoint importance measure."""
    baseline = fuzzy_priority_scores(inputs)
    results: Dict[str, float] = {}
    for level_name, bp in OUTPUT_LEVELS.items():
        for i, param_name in enumerate(("a", "b", "c")):
            perturbed_bp = list(bp)
            perturbed_bp[i] = min(1.0, max(0.0, perturbed_bp[i] + perturb_delta))
            perturbed_levels: Dict[str, Breakpoints] = dict(OUTPUT_LEVELS)
            perturbed_levels[level_name] = (perturbed_bp[0], perturbed_bp[1], perturbed_bp[2])
            perturbed = fuzzy_priority_scores(inputs, output_levels=perturbed_levels)
            results[f"{level_name}.{param_name}"] = float((perturbed - baseline).abs().mean().item())
    return results


def sensitivity_to_rules(inputs: ScorerInputs, rules: Optional[List[Rule]] = None) -> Dict[str, float]:
    """Leave-one-rule-out: drop each rule, measure mean absolute priority
    shift vs the full rule base - a rule's measured contribution."""
    rules = rules if rules is not None else DEFAULT_RULES
    baseline = fuzzy_priority_scores(inputs, rules=rules)
    results: Dict[str, float] = {}
    for i, rule in enumerate(rules):
        reduced = rules[:i] + rules[i + 1 :]
        perturbed = fuzzy_priority_scores(inputs, rules=reduced)
        results[rule.name] = float((perturbed - baseline).abs().mean().item())
    return results


@dataclass
class ComplexityReport:
    scorer_name: str
    latency_seconds_per_candidate: float  # median over `repeats`
    num_parameters: int
    latency_min_seconds_per_candidate: float = 0.0
    latency_max_seconds_per_candidate: float = 0.0
    warm_up_runs: int = 0
    repeats: int = 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "scorer_name": self.scorer_name,
            "latency_seconds_per_candidate": self.latency_seconds_per_candidate,
            "latency_min_seconds_per_candidate": self.latency_min_seconds_per_candidate,
            "latency_max_seconds_per_candidate": self.latency_max_seconds_per_candidate,
            "warm_up_runs": self.warm_up_runs,
            "repeats": self.repeats,
            "num_parameters": self.num_parameters,
        }


def measure_complexity(
    scorer_name: str,
    score_fn: Callable[[ScorerInputs], torch.Tensor],
    inputs: ScorerInputs,
    num_parameters: int,
    warm_up: int = 2,
    repeats: int = 5,
) -> ComplexityReport:
    """Measured (not estimated) per-candidate latency. Uses warm-up runs plus
    repeated timed samples and reports the MEDIAN with min/max, matching the
    discipline `evaluation/profiler.py` already applies elsewhere - a single
    un-warmed call is dominated by first-call overhead and is not a usable
    number to compare scorers with."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    n = max(1, inputs.fragility.shape[0])

    for _ in range(max(0, warm_up)):
        score_fn(inputs)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        score_fn(inputs)
        samples.append((time.perf_counter() - start) / n)
    samples.sort()

    return ComplexityReport(
        scorer_name=scorer_name,
        latency_seconds_per_candidate=samples[len(samples) // 2],
        num_parameters=num_parameters,
        latency_min_seconds_per_candidate=samples[0],
        latency_max_seconds_per_candidate=samples[-1],
        warm_up_runs=max(0, warm_up),
        repeats=repeats,
    )


def fuzzy_num_parameters(rules: Optional[List[Rule]] = None) -> int:
    rules = rules if rules is not None else DEFAULT_RULES
    breakpoint_params = sum(len(bp) for bp in INPUT_LEVELS.values()) + sum(len(bp) for bp in OUTPUT_LEVELS.values())
    return len(rules) + breakpoint_params
