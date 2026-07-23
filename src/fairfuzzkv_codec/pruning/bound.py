"""Local per-head attention-mass bound validator.

For a single attention head, if a set E of key/value positions is evicted and the
kept weights are renormalized, the output error obeys

    || O - O_hat ||_2  <=  2 * M * p_E

where p_E is the attention probability mass on E (per query) and M = max_i ||v_i||.
This is a LOCAL, per-head, single-layer statement - NOT an end-to-end generation
guarantee (see CLAIMS_LEDGER C-18/C-19). The validator REPORTS when its assumptions
fail (e.g. an empty kept set makes renormalization undefined) instead of
manufacturing a passing bound.
"""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class BoundResult:
    p_E: torch.Tensor  # [q] evicted mass per query
    M: float  # max value-vector L2 norm
    observed_error: torch.Tensor  # [q] ||O - O_hat||_2
    bound: torch.Tensor  # [q] 2 * M * p_E
    slack: torch.Tensor  # [q] bound - observed_error
    holds: torch.Tensor  # [q] bool, observed <= bound (+eps), where assumptions hold
    assumption_failed: torch.Tensor  # [q] bool, kept set empty for this query
    notes: List[str] = field(default_factory=list)

    def all_hold(self) -> bool:
        """True only if every query with valid assumptions satisfies the bound."""
        valid = ~self.assumption_failed
        return bool((self.holds[valid]).all().item()) if valid.any() else False


def validate_local_bound(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    evicted: torch.Tensor,
    eps: float = 1e-5,
) -> BoundResult:
    """q:[Sq,d] k:[Sk,d] v:[Sk,d] evicted:[Sk] bool. Single head."""
    if q.dim() != 2 or k.dim() != 2 or v.dim() != 2:
        raise ValueError("q, k, v must be 2D [seq, dim] for a single head")
    if evicted.dtype != torch.bool or evicted.shape[0] != k.shape[0]:
        raise ValueError("evicted must be a bool mask over key positions")

    d = q.size(1)
    logits = (q @ k.t()) / (d ** 0.5)  # [Sq, Sk]
    a = torch.softmax(logits, dim=1)  # full attention weights
    out = a @ v  # attention output O, [Sq, d]

    kept = ~evicted
    p_E = a[:, evicted].sum(dim=1) if evicted.any() else torch.zeros(q.size(0))

    # renormalize over kept set; empty kept set => assumption failure for that query
    kept_mass = a[:, kept].sum(dim=1) if kept.any() else torch.zeros(q.size(0))
    assumption_failed = kept_mass <= eps  # nothing left to renormalize onto

    a_hat = torch.zeros_like(a)
    if kept.any():
        safe = kept_mass.clamp(min=eps)
        a_hat[:, kept] = a[:, kept] / safe.unsqueeze(1)
    out_hat = a_hat @ v  # O_hat

    M = float(torch.norm(v, p=2, dim=1).max().item()) if v.numel() else 0.0
    observed = torch.norm(out - out_hat, p=2, dim=1)
    bound = 2.0 * M * p_E
    slack = bound - observed
    holds = observed <= bound + eps

    notes: List[str] = []
    if assumption_failed.any():
        notes.append(
            f"{int(assumption_failed.sum().item())} query(ies) had an empty kept set "
            f"(renormalization undefined) - reported as assumption failure, not a bound pass"
        )
    return BoundResult(
        p_E=p_E, M=M, observed_error=observed, bound=bound,
        slack=slack, holds=holds, assumption_failed=assumption_failed, notes=notes,
    )
