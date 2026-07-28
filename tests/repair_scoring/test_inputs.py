import torch

from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs, fit_input_normalizers, normalize_inputs


def _inputs(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ScorerInputs(
        fragility=torch.rand(n, generator=g) * 10,
        evidence_importance=torch.rand(n, generator=g) * 10,
        completion_cost=torch.rand(n, generator=g) * 10,
        staleness=torch.rand(n, generator=g) * 10,
    )


def test_mismatched_length_raises():
    try:
        ScorerInputs(
            fragility=torch.zeros(3), evidence_importance=torch.zeros(3),
            completion_cost=torch.zeros(3), staleness=torch.zeros(2),
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_uncertainty_optional_field_names():
    inp = _inputs()
    assert "uncertainty" not in inp.field_names()
    inp2 = ScorerInputs(**inp.as_dict(), uncertainty=torch.rand(5))
    assert "uncertainty" in inp2.field_names()


def test_train_only_normalization_maps_train_range_to_unit_interval():
    train = _inputs(seed=1)
    stats = fit_input_normalizers(train)
    normalized_train = normalize_inputs(train, stats)
    for name, values in normalized_train.as_dict().items():
        assert values.min().item() >= 0.0
        assert values.max().item() <= 1.0
    # the train min/max literally hit 0/1
    assert torch.isclose(normalized_train.fragility.min(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(normalized_train.fragility.max(), torch.tensor(1.0), atol=1e-5)


def test_eval_values_outside_train_range_are_clamped_not_refit():
    train = _inputs(seed=2)
    stats = fit_input_normalizers(train)
    eval_inputs = ScorerInputs(
        fragility=torch.tensor([train.fragility.max().item() + 100.0, train.fragility.min().item() - 100.0]),
        evidence_importance=torch.zeros(2), completion_cost=torch.zeros(2), staleness=torch.zeros(2),
    )
    normalized_eval = normalize_inputs(eval_inputs, stats)
    assert normalized_eval.fragility[0].item() == 1.0  # clamped high, not > 1
    assert normalized_eval.fragility[1].item() == 0.0  # clamped low, not < 0


def test_degenerate_constant_train_range_falls_back_to_neutral_constant():
    train = ScorerInputs(
        fragility=torch.full((4,), 3.0), evidence_importance=torch.zeros(4),
        completion_cost=torch.zeros(4), staleness=torch.zeros(4),
    )
    stats = fit_input_normalizers(train)
    normalized = normalize_inputs(train, stats)
    assert torch.allclose(normalized.fragility, torch.full((4,), 0.5))
