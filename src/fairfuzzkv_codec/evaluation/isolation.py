"""Intersection-full-correct isolation subset for Gate 2.

To isolate COMPRESSION effects from base-model inequality, we keep only the
examples the lossless FullKV system answers correctly. Degradation is then
measured purely as "compression broke an example the base model got right",
never conflated with the base model simply being worse on some cohort.

A `PredictionRecord` is one (example, system) outcome. Systems compared are
"full" (lossless control), "aggregate", and "minimax".
"""

from dataclasses import dataclass
from typing import Dict, List, Set

FULL_SYSTEM = "full"


@dataclass(frozen=True)
class PredictionRecord:
    example_id: str
    cohort: str
    system: str
    correct: bool
    bits_per_element: float = 0.0


def full_correct_ids(records: List[PredictionRecord]) -> Set[str]:
    """Example ids the lossless FullKV control answered correctly."""
    return {r.example_id for r in records if r.system == FULL_SYSTEM and r.correct}


def isolate(records: List[PredictionRecord]) -> List[PredictionRecord]:
    """Keep only records for examples that FullKV got right - the intersection-
    full-correct subset. Reproducible: pure function of the input records."""
    keep = full_correct_ids(records)
    return [r for r in records if r.example_id in keep]


def cohort_counts(records: List[PredictionRecord], system: str) -> Dict[str, int]:
    """Visible per-cohort sample sizes for one system (post-isolation counts if
    called on isolated records) - so under-powered cohorts are never hidden."""
    counts: Dict[str, int] = {}
    for r in records:
        if r.system == system:
            counts[r.cohort] = counts.get(r.cohort, 0) + 1
    return counts


def degradation_per_cohort(
    records: List[PredictionRecord], system: str, full_correct: Set[str]
) -> Dict[str, float]:
    """Delta_{l,B} for one system: on the full-correct subset, the fraction of a
    cohort's examples the system got WRONG (== 1 - accuracy, since full-correct
    accuracy is 1.0 by construction)."""
    wrong: Dict[str, int] = {}
    total: Dict[str, int] = {}
    for r in records:
        if r.system != system or r.example_id not in full_correct:
            continue
        total[r.cohort] = total.get(r.cohort, 0) + 1
        if not r.correct:
            wrong[r.cohort] = wrong.get(r.cohort, 0) + 1
    return {c: wrong.get(c, 0) / n for c, n in total.items() if n > 0}
