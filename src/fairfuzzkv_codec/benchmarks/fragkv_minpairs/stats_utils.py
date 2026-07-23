import random
from typing import Dict, List, Optional

from pydantic import BaseModel

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import FRAGMENTATION_LEVELS


class PredictionRecord(BaseModel):
    """One raw model prediction - the atomic unit the whole Gate 1 report
    must be reproducible from, per the acceptance gate. Nothing in gate1.py
    or stats_utils.py should ever need anything else."""

    group_id: str
    n_g: int
    codec_name: str
    target_bits_per_element: float
    actual_bits_per_element: float
    generated_text: str
    parsed_value: Optional[int] = None
    correct: bool
    kv_reconstruction_mse: Optional[float] = None


class EffectSizeResult(BaseModel):
    codec_name: str
    low_n_g: int
    high_n_g: int
    n_paired_groups: int
    accuracy_by_n_g: Dict[int, float]
    effect_size: float  # mean(correct_at_low - correct_at_high); positive = fragmentation hurts
    ci_low: float
    ci_high: float
    p_value: float
    monotonic_non_increasing: bool
    power_note: str


def _index_by_group(records: List[PredictionRecord], codec_name: str) -> Dict[str, Dict[int, PredictionRecord]]:
    by_group: Dict[str, Dict[int, PredictionRecord]] = {}
    for r in records:
        if r.codec_name != codec_name:
            continue
        by_group.setdefault(r.group_id, {})[r.n_g] = r
    return by_group


def _accuracy_by_n_g(by_group: Dict[str, Dict[int, PredictionRecord]]) -> Dict[int, float]:
    result = {}
    for n_g in FRAGMENTATION_LEVELS:
        values = [g[n_g].correct for g in by_group.values() if n_g in g]
        result[n_g] = (sum(values) / len(values)) if values else float("nan")
    return result


def _is_monotonic_non_increasing(accuracy_by_n_g: Dict[int, float], tolerance: float = 0.02) -> bool:
    values = [accuracy_by_n_g[n_g] for n_g in FRAGMENTATION_LEVELS if accuracy_by_n_g[n_g] == accuracy_by_n_g[n_g]]
    for a, b in zip(values, values[1:]):
        if b > a + tolerance:
            return False
    return True


def compute_effect_size(
    records: List[PredictionRecord],
    codec_name: str,
    low_n_g: int = 1,
    high_n_g: int = 8,
    n_bootstrap: int = 2000,
    n_permutations: int = 5000,
    seed: int = 12345,
) -> EffectSizeResult:
    """Paired effect of fragmentation level, matched by group_id. Positive
    effect_size means the low-n_g variant succeeded more often than the
    high-n_g variant under this codec - i.e. fragmentation hurt task success.
    CI via group-level bootstrap, p-value via sign-flip permutation test
    (the appropriate null for a paired design: under no effect, which of the
    two matched variants "wins" for a given group is exchangeable)."""
    by_group = _index_by_group(records, codec_name)
    paired_diffs = [
        float(g[low_n_g].correct) - float(g[high_n_g].correct)
        for g in by_group.values()
        if low_n_g in g and high_n_g in g
    ]
    n = len(paired_diffs)
    accuracy_by_n_g = _accuracy_by_n_g(by_group)

    if n == 0:
        return EffectSizeResult(
            codec_name=codec_name, low_n_g=low_n_g, high_n_g=high_n_g, n_paired_groups=0,
            accuracy_by_n_g=accuracy_by_n_g, effect_size=0.0, ci_low=0.0, ci_high=0.0,
            p_value=1.0, monotonic_non_increasing=False,
            power_note="No paired groups available - cannot estimate an effect.",
        )

    rng = random.Random(seed)
    observed = sum(paired_diffs) / n

    boot_means = []
    for _ in range(n_bootstrap):
        sample = [paired_diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    ci_low = boot_means[int(0.025 * n_bootstrap)]
    ci_high = boot_means[min(int(0.975 * n_bootstrap), n_bootstrap - 1)]

    perm_means = []
    for _ in range(n_permutations):
        flipped = [d if rng.random() < 0.5 else -d for d in paired_diffs]
        perm_means.append(sum(flipped) / n)
    p_value = sum(1 for m in perm_means if abs(m) >= abs(observed)) / n_permutations

    power_note = (
        f"n={n} paired groups. Bootstrap/permutation intervals widen fast below n=100; "
        "treat this as a pilot-scale estimate, not a well-powered study, until n is scaled up."
        if n < 100
        else f"n={n} paired groups - adequately powered for a >=10-point effect at conventional alpha."
    )

    return EffectSizeResult(
        codec_name=codec_name,
        low_n_g=low_n_g,
        high_n_g=high_n_g,
        n_paired_groups=n,
        accuracy_by_n_g=accuracy_by_n_g,
        effect_size=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        monotonic_non_increasing=_is_monotonic_non_increasing(accuracy_by_n_g),
        power_note=power_note,
    )
