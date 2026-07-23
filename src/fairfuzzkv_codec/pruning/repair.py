"""Attention-mass repair contract.

Repair reintroduces high-value evicted groups (scored by fragility, evidence
value, and completion cost) while keeping the bit budget fixed - so it evicts an
equal number of low-value kept positions in exchange (budget-neutral swap). The
contract it enforces is deliberately LOCAL and modest:

    for each audited head/query:   p_E^repair  <=  p_E^0 + delta

i.e. repair must not make the evicted attention mass meaningfully WORSE than the
baseline it started from. Every accepted and rejected swap is logged with its
bound terms so the decision is fully auditable per layer/head/query. This is not
a claim that repair improves end-to-end accuracy (see CLAIMS_LEDGER C-18/C-19).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class RepairAction:
    layer: int
    head: int
    accepted: bool
    reason: str
    reintroduced_positions: List[int]
    evicted_positions: List[int]
    p_E0_max: float  # baseline evicted mass, worst query
    p_E_repair_max: float  # post-swap evicted mass, worst query
    delta: float
    M: float
    max_observed_error: float
    min_slack: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer": self.layer, "head": self.head, "accepted": self.accepted,
            "reason": self.reason, "reintroduced_positions": self.reintroduced_positions,
            "evicted_positions": self.evicted_positions, "p_E0_max": self.p_E0_max,
            "p_E_repair_max": self.p_E_repair_max, "delta": self.delta, "M": self.M,
            "max_observed_error": self.max_observed_error, "min_slack": self.min_slack,
        }


def repair_score(
    fragility: torch.Tensor,
    evidence_value: torch.Tensor,
    completion_cost: torch.Tensor,
    w_fragility: float = 1.0,
    w_evidence: float = 1.0,
    w_completion: float = 1.0,
) -> torch.Tensor:
    """Higher = more worth reintroducing. Completion cost is a penalty (a group
    that is expensive to complete without its KV is more valuable to keep)."""
    return w_fragility * fragility + w_evidence * evidence_value + w_completion * completion_cost


class RepairContract:
    """Evaluates budget-neutral repair swaps against the local mass condition
    and accumulates an auditable per-head log."""

    def __init__(self, delta: float = 0.02):
        if delta < 0:
            raise ValueError("delta must be >= 0")
        self.delta = delta
        self.log: List[RepairAction] = []

    def evaluate_swap(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        evicted0: torch.Tensor,
        reintroduce: Sequence[int],
        evict: Sequence[int],
        layer: int = 0,
        head: int = 0,
    ) -> torch.Tensor:
        """Propose evicted0 - reintroduce + evict (equal sizes = budget-neutral).
        Accept iff worst-query p_E^repair <= worst-query p_E^0 + delta. Returns
        the resulting evicted mask (unchanged if rejected). Always logs."""
        from fairfuzzkv_codec.pruning.bound import validate_local_bound

        reintroduce = list(reintroduce)
        evict = list(evict)
        if len(reintroduce) != len(evict):
            raise ValueError("budget-neutral swap requires equal reintroduce/evict counts")

        evicted1 = evicted0.clone()
        evicted1[reintroduce] = False  # bring these back
        evicted1[evict] = True  # give up these instead

        d = q.size(1)
        a = torch.softmax((q @ k.t()) / (d ** 0.5), dim=1)
        p_E0 = a[:, evicted0].sum(dim=1) if evicted0.any() else torch.zeros(q.size(0))
        p_E1 = a[:, evicted1].sum(dim=1) if evicted1.any() else torch.zeros(q.size(0))
        p_E0_max = float(p_E0.max().item())
        p_E1_max = float(p_E1.max().item())

        accepted = p_E1_max <= p_E0_max + self.delta
        result = validate_local_bound(q, k, v, evicted1)

        self.log.append(RepairAction(
            layer=layer, head=head, accepted=accepted,
            reason="within delta" if accepted else "would exceed p_E0 + delta",
            reintroduced_positions=reintroduce, evicted_positions=evict,
            p_E0_max=p_E0_max, p_E_repair_max=p_E1_max, delta=self.delta,
            M=result.M, max_observed_error=float(result.observed_error.max().item()),
            min_slack=float(result.slack.min().item()),
        ))
        return evicted1 if accepted else evicted0

    def accepted_actions(self) -> List[RepairAction]:
        return [a for a in self.log if a.accepted]

    def rejected_actions(self) -> List[RepairAction]:
        return [a for a in self.log if not a.accepted]

    def actions_for(self, layer: Optional[int] = None, head: Optional[int] = None) -> List[RepairAction]:
        return [
            a for a in self.log
            if (layer is None or a.layer == layer) and (head is None or a.head == head)
        ]
