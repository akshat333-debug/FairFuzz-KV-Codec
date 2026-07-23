"""Group-aware pruning: aggregate token-level scores to surface groups and
retain/evict WHOLE groups coherently.

A surface group (from the Module 1 GroupMapper) is a contiguous span of token
positions that must be kept or dropped together - splitting a Devanagari
conjunct or an emoji ZWJ sequence across a pruning boundary is exactly what this
project exists to avoid. group_ids is [seq] (same across layers/heads/batch, as
surface structure is a property of the token stream, not the head).
"""

from enum import Enum
from typing import Dict, List

import torch

SEQ_AXIS = 3


class GroupAgg(str, Enum):
    MAX = "max"
    SUM = "sum"
    NORMALIZED = "normalized"  # sum / group_size (mean), so long groups don't win by length alone


def _group_positions(group_ids: torch.Tensor) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for pos, gid in enumerate(group_ids.tolist()):
        groups.setdefault(int(gid), []).append(pos)
    return groups


def aggregate_group_scores(
    scores: torch.Tensor, group_ids: torch.Tensor, rule: GroupAgg
) -> Dict[int, torch.Tensor]:
    """Reduce per-position scores [L,B,H,S] to one score per group, per
    [L,B,H]. Returns {group_id: score_tensor[L,B,H]}."""
    groups = _group_positions(group_ids)
    out: Dict[int, torch.Tensor] = {}
    for gid, positions in groups.items():
        idx = torch.tensor(positions, dtype=torch.long)
        sub = scores.index_select(SEQ_AXIS, idx)  # [L,B,H,len]
        if rule == GroupAgg.MAX:
            out[gid] = sub.amax(dim=SEQ_AXIS)
        elif rule == GroupAgg.SUM:
            out[gid] = sub.sum(dim=SEQ_AXIS)
        elif rule == GroupAgg.NORMALIZED:
            out[gid] = sub.mean(dim=SEQ_AXIS)
        else:
            raise ValueError(f"unknown aggregation rule: {rule}")
    return out


def group_aware_mask(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    keep_positions: int,
    rule: GroupAgg = GroupAgg.NORMALIZED,
) -> torch.Tensor:
    """Select whole groups by aggregated score until the per-head position
    budget is (approximately) filled, then expand to a coherent position mask
    [L,B,H,S]. A group is never partially kept.

    Groups are ranked by a single [L,B,H]-reduced score (mean over heads/layers)
    so the SAME groups are kept across every head - preserving surface coherence
    globally rather than letting different heads split a group differently.
    """
    L, B, H, S = scores.shape
    group_scores = aggregate_group_scores(scores, group_ids, rule)
    positions_of = _group_positions(group_ids)

    # one scalar rank per group (mean across L,B,H) -> global coherent ordering.
    ranked = sorted(
        group_scores.keys(),
        key=lambda gid: float(group_scores[gid].mean().item()),
        reverse=True,
    )

    mask = torch.zeros(L, B, H, S, dtype=torch.bool)
    kept = 0
    budget = max(1, min(S, keep_positions))
    for gid in ranked:
        pos = positions_of[gid]
        if kept >= budget:
            break
        mask[..., pos] = True
        kept += len(pos)
    return mask


def retained_positions_from_mask(mask: torch.Tensor) -> List[int]:
    """Sequence positions kept by at least one (layer, batch, head) - the set a
    group-aware reconstruction must be able to address. Coherent group-aware
    masks keep a position across all heads or none, but this tolerates either."""
    collapsed = mask.any(dim=(0, 1, 2))  # [S]
    return torch.nonzero(collapsed, as_tuple=False).flatten().tolist()
