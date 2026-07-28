"""Repair-priority candidate inputs and train-only normalization.

Every scorer in this module consumes the same five named signals per
candidate group: tokenizer fragility, evidence/attention importance,
completion cost, staleness, and optional uncertainty. All are defined so
that HIGHER = more worth reintroducing (matches Prompt 9's
`repair.repair_score` convention).

Normalization is fit on a TRAIN split only (`fit_input_normalizers`) and
applied unchanged to eval candidates (`normalize_inputs`) - the same
train/apply split discipline as `quantization/calibration.py`'s calibration
range, so eval-set statistics never leak into the scorer's [0, 1] scale.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class NormalizationStats:
    name: str
    min_value: float
    max_value: float

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        span = self.max_value - self.min_value
        if span <= 1e-9:
            # degenerate (constant) training range - documented neutral fallback,
            # not a divide-by-zero crash or a fabricated spread.
            return torch.full_like(values, 0.5)
        return ((values - self.min_value) / span).clamp(0.0, 1.0)


def fit_normalization_stats(name: str, train_values: torch.Tensor) -> NormalizationStats:
    return NormalizationStats(name=name, min_value=float(train_values.min().item()), max_value=float(train_values.max().item()))


@dataclass
class ScorerInputs:
    fragility: torch.Tensor
    evidence_importance: torch.Tensor
    completion_cost: torch.Tensor
    staleness: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        n = self.fragility.shape[0]
        for name in ("evidence_importance", "completion_cost", "staleness"):
            t = getattr(self, name)
            if t.shape[0] != n:
                raise ValueError(f"{name} length {t.shape[0]} != fragility length {n}")
        if self.uncertainty is not None and self.uncertainty.shape[0] != n:
            raise ValueError(f"uncertainty length {self.uncertainty.shape[0]} != fragility length {n}")

    def field_names(self) -> List[str]:
        names = ["fragility", "evidence_importance", "completion_cost", "staleness"]
        if self.uncertainty is not None:
            names.append("uncertainty")
        return names

    def as_dict(self) -> Dict[str, torch.Tensor]:
        d = {
            "fragility": self.fragility,
            "evidence_importance": self.evidence_importance,
            "completion_cost": self.completion_cost,
            "staleness": self.staleness,
        }
        if self.uncertainty is not None:
            d["uncertainty"] = self.uncertainty
        return d


def fit_input_normalizers(train_inputs: ScorerInputs) -> Dict[str, NormalizationStats]:
    return {name: fit_normalization_stats(name, values) for name, values in train_inputs.as_dict().items()}


def normalize_inputs(inputs: ScorerInputs, stats: Dict[str, NormalizationStats]) -> ScorerInputs:
    """Apply already-fit (train-only) stats to `inputs`. Fields absent from
    `stats` pass through unchanged rather than silently being dropped."""
    d = inputs.as_dict()
    normalized = {name: (stats[name].apply(values) if name in stats else values) for name, values in d.items()}
    return ScorerInputs(
        fragility=normalized["fragility"],
        evidence_importance=normalized["evidence_importance"],
        completion_cost=normalized["completion_cost"],
        staleness=normalized["staleness"],
        uncertainty=normalized.get("uncertainty"),
    )
