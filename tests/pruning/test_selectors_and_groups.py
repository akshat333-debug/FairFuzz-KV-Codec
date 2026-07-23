import torch

from fairfuzzkv_codec.pruning.group_aware import (
    GroupAgg,
    aggregate_group_scores,
    group_aware_mask,
    retained_positions_from_mask,
)
from fairfuzzkv_codec.pruning.selectors import (
    RecencySelector,
    TopAttentionMassSelector,
    TopKTokenScoreSelector,
    attention_mass,
    token_l2_scores,
)


def _scores(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 1, 2, 8, generator=g)  # [L,B,H,S]


def test_recency_keeps_last_k_regardless_of_score():
    scores = _scores()
    mask = RecencySelector().select(scores, keep=3)
    assert mask[..., -3:].all()
    assert not mask[..., :-3].any()


def test_topk_keeps_highest_scoring():
    scores = torch.zeros(1, 1, 1, 5)
    scores[0, 0, 0] = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3])
    mask = TopKTokenScoreSelector().select(scores, keep=2)
    assert mask[0, 0, 0].tolist() == [False, True, False, True, False]


def test_selector_budget_never_exceeds_seq_or_underflows():
    scores = _scores()
    for keep in (0, 1, 100):
        mask = TopKTokenScoreSelector().select(scores, keep)
        kept = mask.sum(dim=3)
        assert (kept >= 1).all() and (kept <= scores.size(3)).all()


def test_attention_mass_reduces_over_queries():
    attn = torch.rand(1, 1, 1, 4, 6)  # [L,B,H,q,k]
    mass = attention_mass(attn)
    assert mass.shape == (1, 1, 1, 6)
    assert torch.allclose(mass[0, 0, 0], attn[0, 0, 0].sum(dim=0))


def test_top_attention_mass_selector():
    attn = torch.zeros(1, 1, 1, 2, 4)
    attn[0, 0, 0, :, 2] = 1.0  # key 2 gets all mass
    mass = attention_mass(attn)
    mask = TopAttentionMassSelector().select(mass, keep=1)
    assert mask[0, 0, 0, 2].item() is True


def test_group_aggregation_rules():
    scores = torch.zeros(1, 1, 1, 4)
    scores[0, 0, 0] = torch.tensor([1.0, 3.0, 2.0, 2.0])
    gids = torch.tensor([0, 0, 1, 1])
    gmax = aggregate_group_scores(scores, gids, GroupAgg.MAX)
    gsum = aggregate_group_scores(scores, gids, GroupAgg.SUM)
    gnorm = aggregate_group_scores(scores, gids, GroupAgg.NORMALIZED)
    assert gmax[0].item() == 3.0 and gmax[1].item() == 2.0
    assert gsum[0].item() == 4.0 and gsum[1].item() == 4.0
    assert gnorm[0].item() == 2.0 and gnorm[1].item() == 2.0


def test_group_aware_keeps_whole_groups_coherently():
    # group 1 has the highest normalized score -> kept entirely; a group is
    # never partially retained (surface coherence).
    scores = torch.zeros(1, 1, 2, 6)
    scores[..., 2:4] = 5.0  # group 1 positions
    gids = torch.tensor([0, 0, 1, 1, 2, 2])
    mask = group_aware_mask(scores, gids, keep_positions=2, rule=GroupAgg.NORMALIZED)
    kept = retained_positions_from_mask(mask)
    # whichever groups are kept, each is kept fully (pairs), never split.
    for g_positions in ([0, 1], [2, 3], [4, 5]):
        inside = [p for p in g_positions if p in kept]
        assert inside in ([], g_positions), f"group split: {inside}"
    assert 2 in kept and 3 in kept  # highest-score group retained


def test_group_aware_reconstructs_valid_positions():
    scores = token_l2_scores(torch.randn(2, 1, 2, 6, 4))
    gids = torch.tensor([0, 0, 1, 1, 2, 2])
    mask = group_aware_mask(scores, gids, keep_positions=4)
    positions = retained_positions_from_mask(mask)
    assert all(0 <= p < 6 for p in positions)
    assert positions == sorted(positions)
