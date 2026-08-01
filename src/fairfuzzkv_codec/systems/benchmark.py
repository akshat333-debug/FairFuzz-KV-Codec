"""Latency/memory measurement with warm-up, synchronization, and percentiles.

`evaluation/profiler.py`'s `TelemetryTracker` (Prompt 2) is preserved and still
used for coarse peak-memory tracking. This module adds what Prompt 18 item 123
requires and that one does not have: p50/p95, bootstrap confidence intervals,
explicit CUDA synchronization, and a `measured` flag so an ESTIMATED number can
never be mistaken for a measured one.

Nothing here infers a speedup. `speedup_vs` is computed only from two sets of
real samples; there is no analytical/projected path.
"""

import gc
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def _sync(device: str) -> None:
    """CUDA kernels are async; without this the timer measures launch time, not
    execution time. No-op on CPU."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class LatencyStats:
    label: str
    n_samples: int
    warmup_runs: int
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float
    stdev_seconds: float
    ci95_low: float
    ci95_high: float
    measured: bool = True  # always True here; the field exists so callers that
    # record an ESTIMATE must set it False explicitly and be visibly different.
    raw_samples: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _percentile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolated percentile; q in [0,1]."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _bootstrap_ci(samples: List[float], n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap CI of the MEAN. Returns (nan, nan) for <2 samples rather
    than a fake zero-width interval."""
    if len(samples) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        resample = [rng.choice(samples) for _ in samples]
        means.append(sum(resample) / len(resample))
    means.sort()
    return means[int(0.025 * n_boot)], means[min(int(0.975 * n_boot), n_boot - 1)]


def measure_latency(
    label: str,
    fn: Callable[[], Any],
    warmup: int = 3,
    repeats: int = 20,
    device: str = "cpu",
    seed: int = 0,
) -> LatencyStats:
    """Warm-up, then `repeats` synchronized timed runs. Reports p50/p95 and a
    bootstrap CI on the mean - a single timing is not a measurement."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    for _ in range(max(0, warmup)):
        fn()
    _sync(device)

    samples: List[float] = []
    for _ in range(repeats):
        gc.collect()  # keep a stray collection from landing inside a timed run
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - start)

    ordered = sorted(samples)
    lo, hi = _bootstrap_ci(samples, seed=seed)
    return LatencyStats(
        label=label,
        n_samples=len(samples),
        warmup_runs=max(0, warmup),
        mean_seconds=statistics.fmean(samples),
        p50_seconds=_percentile(ordered, 0.50),
        p95_seconds=_percentile(ordered, 0.95),
        min_seconds=ordered[0],
        max_seconds=ordered[-1],
        stdev_seconds=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        ci95_low=lo,
        ci95_high=hi,
        raw_samples=samples,
    )


@dataclass
class MemoryReading:
    peak_cpu_mb: Optional[float]
    peak_gpu_mb: Optional[float]
    measured: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def measure_peak_memory(fn: Callable[[], Any], device: str = "cpu") -> MemoryReading:
    """Peak memory across one call. CPU peak uses tracemalloc (Python-level
    allocations only - it does NOT see torch's C++ allocator, which is stated
    here rather than silently over-claimed). GPU peak uses torch's own counter,
    which IS authoritative for device memory."""
    peak_gpu: Optional[float] = None
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    import tracemalloc

    tracemalloc.start()
    try:
        fn()
        _sync(device)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    if device.startswith("cuda") and torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return MemoryReading(
        peak_cpu_mb=peak / (1024 ** 2),
        peak_gpu_mb=peak_gpu,
        measured=True,
    )


def speedup(baseline: LatencyStats, candidate: LatencyStats) -> Dict[str, Any]:
    """Ratio of MEASURED medians, with an explicit overlap check. If the two
    confidence intervals overlap, the speedup is reported as NOT significant -
    a ratio computed from noise is not a speedup claim."""
    ratio = baseline.p50_seconds / candidate.p50_seconds if candidate.p50_seconds > 0 else float("inf")
    overlapping = not (baseline.ci95_low > candidate.ci95_high or candidate.ci95_low > baseline.ci95_high)
    return {
        "baseline": baseline.label,
        "candidate": candidate.label,
        "p50_speedup": ratio,
        "confidence_intervals_overlap": overlapping,
        "significant": not overlapping,
        "interpretation": (
            "CIs overlap - this ratio is not a demonstrated speedup"
            if overlapping else "CIs disjoint - measured difference"
        ),
        "measured": True,
    }
