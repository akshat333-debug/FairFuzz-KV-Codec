import torch

from fairfuzzkv_codec.pruning.repair import RepairContract
from fairfuzzkv_codec.repair_scoring.ablation import ScorerConfig, ScorerType, score_candidates
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs
from fairfuzzkv_codec.repair_scoring.integration import propose_repair_swap


def test_propose_repair_swap_is_budget_neutral():
    priority = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    evicted0 = torch.tensor([True, True, False, False, False, False])
    reintroduce, evict = propose_repair_swap(priority, evicted0, n=1)
    assert len(reintroduce) == len(evict) == 1
    assert reintroduce[0] == 0  # evicted position with highest priority (0.9)
    assert evict[0] == 3  # kept position with lowest priority (0.2)


def test_propose_repair_swap_clamps_n_to_available_positions():
    priority = torch.rand(4)
    evicted0 = torch.tensor([True, False, False, False])  # only 1 evicted, 3 kept
    reintroduce, evict = propose_repair_swap(priority, evicted0, n=5)
    assert len(reintroduce) == len(evict) == 1


def test_propose_repair_swap_mismatched_lengths_raises():
    try:
        propose_repair_swap(torch.zeros(3), torch.zeros(4, dtype=torch.bool), n=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def _qkv(sq=2, sk=6, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.ones(sq, d),
        torch.randn(sk, d, generator=g),
        torch.randn(sk, d, generator=g),
    )


def test_every_scorer_type_can_drive_a_real_repair_contract_swap():
    # end-to-end: Module 3 score -> propose_repair_swap -> UNCHANGED Prompt 9
    # RepairContract, for every scorer type, at the SAME budget (n=1).
    q, k, v = _qkv()
    inputs = ScorerInputs(
        fragility=torch.tensor([0.9, 0.1, 0.5, 0.2, 0.6, 0.3]),
        evidence_importance=torch.tensor([0.8, 0.2, 0.4, 0.3, 0.5, 0.1]),
        completion_cost=torch.tensor([0.3, 0.7, 0.5, 0.4, 0.6, 0.2]),
        staleness=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    )
    evicted0 = torch.tensor([True, True, False, False, False, False])

    for scorer_type in ScorerType:
        priority = score_candidates(inputs, ScorerConfig(scorer_type))
        reintroduce, evict = propose_repair_swap(priority, evicted0, n=1)
        contract = RepairContract(delta=1.0)  # permissive - we're testing wiring, not acceptance
        new_evicted = contract.evaluate_swap(q, k, v, evicted0, reintroduce=reintroduce, evict=evict)
        # budget-neutral regardless of accept/reject: total kept count preserved on accept,
        # unchanged on reject - either way the mass condition in the log holds.
        for action in contract.log:
            assert action.p_E_repair_max <= action.p_E0_max + action.delta + 1e-6
        assert int((~new_evicted).sum().item()) == int((~evicted0).sum().item())
