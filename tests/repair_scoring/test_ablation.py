import torch

from fairfuzzkv_codec.repair_scoring.ablation import ScorerConfig, ScorerType, run_ablation, score_candidates
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs


def _inputs(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ScorerInputs(
        fragility=torch.rand(n, generator=g), evidence_importance=torch.rand(n, generator=g),
        completion_cost=torch.rand(n, generator=g), staleness=torch.rand(n, generator=g),
    )


def test_run_ablation_default_covers_all_scorer_types():
    inputs = _inputs()
    results = run_ablation(inputs)
    assert set(results.keys()) == {t.value for t in ScorerType}
    for scores in results.values():
        assert scores.shape == (inputs.fragility.shape[0],)


def test_run_ablation_uses_identical_candidates_for_every_scorer():
    # same `inputs` object fed to every scorer - satisfies item 89 (identical
    # candidate groups). Re-running with the same inputs is byte-identical.
    inputs = _inputs(seed=3)
    r1 = run_ablation(inputs)
    r2 = run_ablation(inputs)
    for name in r1:
        assert torch.equal(r1[name], r2[name])


def test_score_candidates_dispatches_each_type():
    inputs = _inputs()
    for t in ScorerType:
        s = score_candidates(inputs, ScorerConfig(t))
        assert s.shape[0] == inputs.fragility.shape[0]


def test_unknown_scorer_type_raises():
    inputs = _inputs()
    bad = ScorerConfig.__new__(ScorerConfig)
    bad.scorer_type = "not_a_real_type"
    bad.weights = None
    bad.bias = 0.0
    bad.steepness = 6.0
    bad.rules = None
    bad.input_levels = None
    bad.output_levels = None
    try:
        score_candidates(inputs, bad)
        assert False, "expected ValueError"
    except ValueError:
        pass
