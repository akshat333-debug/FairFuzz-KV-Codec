import json
from pathlib import Path
from typing import List, Union

from fairfuzzkv_codec.fragility_estimation.schema import CohortBand, CohortDefinition

_DEFAULT_BAND_LABELS = ["low", "medium", "high", "critical"]


def build_cohort_definition(
    scores: List[float],
    tokenizer_name: str,
    corpus_id: str,
    num_bands: int = 4,
    min_band_samples: int = 3,
) -> CohortDefinition:
    """Quantile-based risk bands with a deterministic minimum-sample-size
    merge rule. Reruns on the same score list always produce the same bands
    (no randomness anywhere in this path)."""
    if not scores:
        raise ValueError("cannot build a cohort definition from an empty score list")

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    edges = [sorted_scores[int(round(q * (n - 1)))] for q in (i / num_bands for i in range(num_bands + 1))]

    labels = _DEFAULT_BAND_LABELS[:num_bands] if num_bands <= len(_DEFAULT_BAND_LABELS) else [
        f"band_{i}" for i in range(num_bands)
    ]

    bands: List[CohortBand] = []
    for i in range(num_bands):
        lower, upper = edges[i], edges[i + 1]
        is_last = i == num_bands - 1
        count = sum(1 for s in scores if (lower <= s <= upper if is_last else lower <= s < upper))
        bands.append(CohortBand(label=labels[i], lower=lower, upper=upper, count=count))

    bands = _merge_small_bands(bands, min_band_samples)

    return CohortDefinition(
        tokenizer_name=tokenizer_name,
        corpus_id=corpus_id,
        min_band_samples=min_band_samples,
        bands=bands,
    )


def _merge_small_bands(bands: List[CohortBand], min_samples: int) -> List[CohortBand]:
    """Deterministic tie rule: a below-minimum band always merges into its
    higher-risk neighbor first (next index); only the topmost band, with no
    higher neighbor, merges backward into the previous (lower-risk) band."""
    bands = list(bands)
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, b in enumerate(bands):
            if b.count < min_samples:
                if i + 1 < len(bands):
                    nxt = bands[i + 1]
                    merged = CohortBand(
                        label=f"{b.label}+{nxt.label}", lower=b.lower, upper=nxt.upper, count=b.count + nxt.count
                    )
                    bands = bands[:i] + [merged] + bands[i + 2 :]
                else:
                    prev = bands[i - 1]
                    merged = CohortBand(
                        label=f"{prev.label}+{b.label}", lower=prev.lower, upper=b.upper, count=prev.count + b.count
                    )
                    bands = bands[: i - 1] + [merged]
                changed = True
                break
    return bands


def assign_cohort_index(score: float, definition: CohortDefinition) -> int:
    """Deterministic tie rule: a score exactly at a shared boundary goes to
    the lower-risk band (upper bound exclusive), except the topmost band's
    upper bound, which is inclusive so the maximum score is always covered.
    Returns the band's ordinal position (0 = lowest risk), which is what
    cross-tokenizer stability compares - band labels/boundaries are fit
    independently per tokenizer and aren't directly comparable, but ordinal
    position (lowest risk .. highest risk) is."""
    for i, band in enumerate(definition.bands):
        is_last = i == len(definition.bands) - 1
        if band.lower <= score <= band.upper if is_last else band.lower <= score < band.upper:
            return i
    return 0 if score < definition.bands[0].lower else len(definition.bands) - 1


def assign_cohort(score: float, definition: CohortDefinition) -> str:
    return definition.bands[assign_cohort_index(score, definition)].label


def write_cohort_manifest(definition: CohortDefinition, path: Union[str, Path]) -> None:
    Path(path).write_text(definition.model_dump_json(indent=2))


def load_cohort_manifest(path: Union[str, Path]) -> CohortDefinition:
    return CohortDefinition.model_validate(json.loads(Path(path).read_text()))
