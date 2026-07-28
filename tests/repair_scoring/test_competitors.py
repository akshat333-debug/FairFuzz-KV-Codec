import torch

from fairfuzzkv_codec.repair_scoring.competitors import knapsack_value_cost_ratio, logistic_score, monotone_weighted_score
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs


def _inputs():
    return ScorerInputs(
        fragility=torch.tensor([0.0, 1.0, 0.5]),
        evidence_importance=torch.tensor([0.0, 1.0, 0.5]),
        completion_cost=torch.tensor([0.0, 1.0, 0.5]),
        staleness=torch.tensor([0.0, 1.0, 0.5]),
    )


def test_monotone_weighted_score_higher_inputs_score_higher():
    s = monotone_weighted_score(_inputs())
    assert s[1] > s[2] > s[0]


def test_monotone_weighted_score_default_weights_sum_to_one_effect():
    inp = ScorerInputs(
        fragility=torch.tensor([1.0]), evidence_importance=torch.tensor([1.0]),
        completion_cost=torch.tensor([1.0]), staleness=torch.tensor([1.0]),
    )
    s = monotone_weighted_score(inp)
    assert abs(s.item() - 1.0) < 1e-6  # all fields=1, equal weights summing to 1 -> score=1


def test_monotone_weighted_score_rejects_unknown_weight_field():
    try:
        monotone_weighted_score(_inputs(), weights={"nonexistent_field": 1.0})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_logistic_score_monotone_and_bounded():
    s = logistic_score(_inputs())
    assert s[1] > s[2] > s[0]
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_logistic_score_midpoint_is_half():
    inp = ScorerInputs(
        fragility=torch.tensor([0.5]), evidence_importance=torch.tensor([0.5]),
        completion_cost=torch.tensor([0.5]), staleness=torch.tensor([0.5]),
    )
    s = logistic_score(inp)
    assert abs(s.item() - 0.5) < 1e-6


def test_knapsack_ratio_prefers_low_cost_high_value_candidate():
    inp = ScorerInputs(
        fragility=torch.tensor([1.0, 1.0]), evidence_importance=torch.tensor([1.0, 1.0]),
        completion_cost=torch.tensor([0.1, 0.9]), staleness=torch.tensor([1.0, 1.0]),
    )
    ratio = knapsack_value_cost_ratio(inp)
    assert ratio[0] > ratio[1]  # same value, lower cost -> higher ratio


def test_knapsack_ratio_never_divides_by_exact_zero():
    inp = ScorerInputs(
        fragility=torch.tensor([1.0]), evidence_importance=torch.tensor([1.0]),
        completion_cost=torch.tensor([0.0]), staleness=torch.tensor([1.0]),
    )
    ratio = knapsack_value_cost_ratio(inp)
    assert torch.isfinite(ratio).all()


def test_all_three_competitors_are_deterministic():
    inp = _inputs()
    for fn in (monotone_weighted_score, logistic_score, knapsack_value_cost_ratio):
        a, b = fn(inp), fn(inp)
        assert torch.equal(a, b)
