import torch

from fairfuzzkv_codec.repair_scoring.competitors import monotone_weighted_score
from fairfuzzkv_codec.repair_scoring.fuzzy import fuzzy_priority_scores
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs
from fairfuzzkv_codec.repair_scoring.sensitivity import (
    fuzzy_num_parameters, measure_complexity, sensitivity_to_breakpoints, sensitivity_to_rules,
)


def _inputs(n=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ScorerInputs(
        fragility=torch.rand(n, generator=g), evidence_importance=torch.rand(n, generator=g),
        completion_cost=torch.rand(n, generator=g), staleness=torch.rand(n, generator=g),
    )


def test_sensitivity_to_breakpoints_covers_all_output_params_and_is_nonnegative():
    result = sensitivity_to_breakpoints(_inputs())
    assert set(result.keys()) == {f"{lvl}.{p}" for lvl in ("low", "medium", "high") for p in ("a", "b", "c")}
    assert all(v >= 0.0 for v in result.values())
    assert any(v > 0.0 for v in result.values())  # at least one breakpoint matters


def test_sensitivity_to_rules_flags_core_rules_as_more_important_than_nothing():
    result = sensitivity_to_rules(_inputs(seed=5))
    assert len(result) == 10  # DEFAULT_RULES count
    assert all(v >= 0.0 for v in result.values())
    assert any(v > 0.0 for v in result.values())


def test_measure_complexity_reports_positive_latency_and_correct_param_count():
    inputs = _inputs()
    report = measure_complexity("fuzzy", fuzzy_priority_scores, inputs, fuzzy_num_parameters())
    assert report.scorer_name == "fuzzy"
    assert report.latency_seconds_per_candidate >= 0.0
    assert report.num_parameters == fuzzy_num_parameters()


def test_fuzzy_has_more_parameters_than_monotone_weighted_score():
    inputs = _inputs()
    fuzzy_report = measure_complexity("fuzzy", fuzzy_priority_scores, inputs, fuzzy_num_parameters())
    monotone_report = measure_complexity("monotone", monotone_weighted_score, inputs, num_parameters=4)
    assert fuzzy_report.num_parameters > monotone_report.num_parameters
