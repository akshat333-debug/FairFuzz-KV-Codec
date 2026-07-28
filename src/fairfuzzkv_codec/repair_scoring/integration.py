"""Wires a repair-priority scorer's output into Prompt 9's UNCHANGED repair
mass constraint and codec pipeline (`pruning.repair.RepairContract`).

This module proposes WHICH budget-neutral swap to attempt; `RepairContract`
still decides whether to accept it against `p_E^repair <= p_E^0 + delta`.
Module 3 never bypasses or loosens that contract.
"""

from typing import List, Tuple

import torch


def propose_repair_swap(priority: torch.Tensor, evicted0: torch.Tensor, n: int) -> Tuple[List[int], List[int]]:
    """reintroduce = the `n` currently-evicted positions with the highest
    priority; evict = the `n` currently-kept positions with the lowest
    priority. Equal-length lists -> a budget-neutral swap, exactly the
    contract `RepairContract.evaluate_swap` requires. `n` is silently
    clamped to what's available on either side (never raises for an
    over-large request)."""
    if priority.shape[0] != evicted0.shape[0]:
        raise ValueError(f"priority length {priority.shape[0]} != evicted0 length {evicted0.shape[0]}")

    evicted_positions = torch.nonzero(evicted0, as_tuple=False).flatten()
    kept_positions = torch.nonzero(~evicted0, as_tuple=False).flatten()
    n = min(n, int(evicted_positions.numel()), int(kept_positions.numel()))
    if n == 0:
        return [], []

    top_evicted = evicted_positions[torch.topk(priority[evicted_positions], n).indices]
    bottom_kept = kept_positions[torch.topk(-priority[kept_positions], n).indices]
    return top_evicted.tolist(), bottom_kept.tolist()
