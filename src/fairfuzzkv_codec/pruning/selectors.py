"""Baseline KV pruning selectors.

Every selector maps a per-key-position importance signal to a boolean keep-mask
of shape [layers, batch, heads, seq]. Selectors are deliberately INDEPENDENT of
quantization - they consume scores/attention and emit masks; a codec applies the
mask separately (see pruning/topk.py::apply_topk_pruning). This keeps pruning and
quantization composable without either depending on the other's internals.

Shape convention: KV tensors are [layers, batch, heads, seq, head_dim]; scores
and masks drop the head_dim -> [layers, batch, heads, seq].
"""

from abc import ABC, abstractmethod

import torch

SEQ_AXIS = 3


def _budget_k(seq_len: int, keep: int) -> int:
    return max(1, min(seq_len, keep))


def token_l2_scores(kv_cache: torch.Tensor) -> torch.Tensor:
    """Per-position L2 magnitude over head_dim -> [layers, batch, heads, seq]."""
    return torch.norm(kv_cache, p=2, dim=-1)


def attention_mass(attn_weights: torch.Tensor) -> torch.Tensor:
    """Reduce query-by-key attention [L,B,H,Sq,Sk] to per-key mass [L,B,H,Sk]
    by summing over the query axis (total probability each key received)."""
    if attn_weights.dim() != 5:
        raise ValueError("attn_weights must be [layers, batch, heads, q, k]")
    return attn_weights.sum(dim=3)


class Selector(ABC):
    """keep = number of positions to retain per (layer, batch, head)."""

    @abstractmethod
    def select(self, scores: torch.Tensor, keep: int) -> torch.Tensor: ...


class RecencySelector(Selector):
    """Keep the most recent `keep` positions - ignores scores entirely (the
    canonical position-only baseline)."""

    def select(self, scores: torch.Tensor, keep: int) -> torch.Tensor:
        L, B, H, S = scores.shape
        k = _budget_k(S, keep)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[..., S - k :] = True
        return mask


class TopKTokenScoreSelector(Selector):
    """Keep the `keep` highest-scoring positions per head (score is caller-
    supplied: L2 norm, saliency, etc.)."""

    def select(self, scores: torch.Tensor, keep: int) -> torch.Tensor:
        S = scores.size(SEQ_AXIS)
        k = _budget_k(S, keep)
        _, idx = torch.topk(scores, k, dim=SEQ_AXIS)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(SEQ_AXIS, idx, True)
        return mask


class TopAttentionMassSelector(Selector):
    """Keep the `keep` positions that received the most attention mass. Scores
    ARE the per-key attention mass (from attention_mass())."""

    def select(self, scores: torch.Tensor, keep: int) -> torch.Tensor:
        return TopKTokenScoreSelector().select(scores, keep)
