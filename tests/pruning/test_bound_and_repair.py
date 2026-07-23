import torch

from fairfuzzkv_codec.pruning.bound import validate_local_bound
from fairfuzzkv_codec.pruning.repair import RepairContract, repair_score


def _qkv(sq=3, sk=6, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(sq, d, generator=g),
        torch.randn(sk, d, generator=g),
        torch.randn(sk, d, generator=g),
    )


# ---- local bound -----------------------------------------------------------

def test_bound_holds_on_random_eviction():
    q, k, v = _qkv()
    evicted = torch.tensor([True, False, True, False, False, False])
    r = validate_local_bound(q, k, v, evicted)
    assert r.all_hold()
    assert (r.slack[~r.assumption_failed] >= 0).all()


def test_zero_eviction_gives_zero_error():
    q, k, v = _qkv()
    evicted = torch.zeros(6, dtype=torch.bool)
    r = validate_local_bound(q, k, v, evicted)
    assert torch.allclose(r.observed_error, torch.zeros_like(r.observed_error), atol=1e-5)
    assert r.all_hold()


def test_empty_kept_set_reported_as_assumption_failure_not_pass():
    q, k, v = _qkv()
    evicted = torch.ones(6, dtype=torch.bool)  # evict everything -> renorm undefined
    r = validate_local_bound(q, k, v, evicted)
    assert r.assumption_failed.all()
    assert r.notes  # explicitly reported, not silently "passed"


def test_adversarial_concentrated_attention():
    # mass concentrated on two keys (~0.5 each); evict one -> large p_E ~0.5 but
    # the kept twin keeps renormalization valid, so the bound must hold.
    d = 4
    q = torch.ones(2, d)
    k = torch.zeros(6, d)
    k[0] = 10.0
    k[1] = 10.0  # keys 0 and 1 split the mass, everything else ~0
    v = torch.randn(6, d)
    evicted = torch.tensor([True, False, False, False, False, False])
    r = validate_local_bound(q, k, v, evicted)
    assert 0.3 < r.p_E.max().item() < 0.7  # genuinely concentrated, not total
    assert not r.assumption_failed.any()
    assert r.all_hold()


def test_evicting_the_sole_dominant_key_is_an_assumption_failure():
    # if ONE key holds essentially all mass and it is evicted, the kept set has
    # ~zero mass -> renormalization undefined -> reported, not a fake pass.
    d = 4
    q = torch.ones(2, d)
    k = torch.zeros(6, d)
    k[0] = 10.0
    v = torch.randn(6, d)
    r = validate_local_bound(q, k, v, torch.tensor([True, False, False, False, False, False]))
    assert r.assumption_failed.all() and r.notes


def test_adversarial_large_value_norms():
    q, k, _ = _qkv()
    v = torch.randn(6, 4) * 1000.0  # huge M
    evicted = torch.tensor([True, False, True, False, False, False])
    r = validate_local_bound(q, k, v, evicted)
    assert r.M > 100.0
    assert r.all_hold()


def test_adversarial_zero_attention_group_is_safe_to_evict():
    d = 4
    q = torch.ones(2, d)
    k = torch.zeros(6, d)
    k[5] = -50.0  # key 5 gets ~zero attention
    v = torch.randn(6, d)
    evicted = torch.tensor([False, False, False, False, False, True])
    r = validate_local_bound(q, k, v, evicted)
    assert r.p_E.max().item() < 1e-3  # negligible evicted mass
    assert torch.allclose(r.observed_error, torch.zeros_like(r.observed_error), atol=1e-3)


def test_adversarial_duplicated_subtokens():
    # duplicated keys/values: evicting one duplicate leaves its twin, small error.
    d = 4
    q, _, _ = _qkv()
    base_k = torch.randn(3, d)
    base_v = torch.randn(3, d)
    k = torch.cat([base_k, base_k], dim=0)  # duplicated
    v = torch.cat([base_v, base_v], dim=0)
    evicted = torch.tensor([True, False, False, False, False, False])
    r = validate_local_bound(q, k, v, evicted)
    assert r.all_hold()


# ---- repair contract -------------------------------------------------------

def test_repair_score_monotone_in_components():
    frag = torch.tensor([0.0, 1.0])
    ev = torch.tensor([0.0, 1.0])
    cc = torch.tensor([0.0, 1.0])
    s = repair_score(frag, ev, cc)
    assert s[1] > s[0]


def test_repair_accepts_beneficial_swap():
    # reintroduce a high-mass key, evict a low-mass one => p_E cannot worsen.
    d = 4
    q = torch.ones(2, d)
    k = torch.zeros(6, d)
    k[0] = 10.0  # high attention
    k[5] = -50.0  # ~zero attention
    v = torch.randn(6, d)
    evicted0 = torch.tensor([True, False, False, False, False, False])  # evicting the important key
    contract = RepairContract(delta=0.0)
    new_evicted = contract.evaluate_swap(q, k, v, evicted0, reintroduce=[0], evict=[5], layer=1, head=2)
    assert contract.accepted_actions()
    assert new_evicted[0].item() is False and new_evicted[5].item() is True


def test_repair_rejects_swap_that_worsens_mass_beyond_delta():
    d = 4
    q = torch.ones(2, d)
    k = torch.zeros(6, d)
    k[0] = 10.0  # important
    k[5] = -50.0  # negligible
    v = torch.randn(6, d)
    evicted0 = torch.tensor([False, False, False, False, False, True])  # evicting the negligible key
    contract = RepairContract(delta=0.01)
    # propose evicting the important key instead -> should blow past delta -> rejected.
    new_evicted = contract.evaluate_swap(q, k, v, evicted0, reintroduce=[5], evict=[0])
    assert contract.rejected_actions()
    assert torch.equal(new_evicted, evicted0)  # unchanged on reject


def test_repair_never_violates_mass_condition_on_accepts():
    q, k, v = _qkv(sk=8)
    contract = RepairContract(delta=0.02)
    rng = torch.Generator().manual_seed(3)
    for _ in range(20):
        evicted0 = torch.rand(8, generator=rng) < 0.4
        idx = torch.randperm(8, generator=rng)
        contract.evaluate_swap(q, k, v, evicted0, reintroduce=[int(idx[0])], evict=[int(idx[1])])
    for a in contract.accepted_actions():
        assert a.p_E_repair_max <= a.p_E0_max + a.delta + 1e-6


def test_baseline_and_repaired_compared_at_equal_total_bits():
    # repair is budget-neutral: an accepted swap reintroduces exactly as many
    # positions as it evicts, so the retained count (== total bits) is preserved.
    q, k, v = _qkv(sk=8)
    evicted0 = torch.tensor([True, True, False, False, False, False, False, False])
    baseline_kept = int((~evicted0).sum().item())
    contract = RepairContract(delta=1.0)  # permissive, force an accept
    new_evicted = contract.evaluate_swap(q, k, v, evicted0, reintroduce=[0], evict=[7])
    repaired_kept = int((~new_evicted).sum().item())
    assert repaired_kept == baseline_kept  # equal bits


def test_repair_log_inspectable_per_layer_head():
    q, k, v = _qkv()
    contract = RepairContract(delta=0.05)
    evicted0 = torch.tensor([True, False, False, False, False, False])
    contract.evaluate_swap(q, k, v, evicted0, reintroduce=[0], evict=[1], layer=3, head=7)
    entries = contract.actions_for(layer=3, head=7)
    assert len(entries) == 1
    d = entries[0].to_dict()
    assert d["layer"] == 3 and d["head"] == 7
    assert "p_E0_max" in d and "p_E_repair_max" in d and "min_slack" in d
