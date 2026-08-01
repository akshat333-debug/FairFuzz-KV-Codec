"""Codebook / calibration caching with an explicit benchmark-contamination guard.

Caching a trained LBG codebook or a calibration range is a pure speed win - IF
the cache key fully determines the artifact. If it does not, a cached entry
fitted on one split can silently leak into the evaluation of another, which is
exactly the contamination this project's train/test discipline exists to
prevent (see `allocation/calibration.py`).

So the key here is content-addressed: it includes a hash of the DATA the
artifact was fitted on plus every config field that affects the fit. A lookup
with a different split therefore misses rather than returning a foreign
artifact. `CacheEntry.fitted_on_split` is recorded and `get()` refuses to serve
an entry fitted on a different split than the caller declares - a hard error,
not a warning, because silent contamination is the failure mode that invalidates
results.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


class CacheContaminationError(RuntimeError):
    """Raised when a cached artifact fitted on one split is requested for
    another - refusing is the whole point of the guard."""


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Content hash of a tensor: shape + dtype + raw bytes. Two different
    calibration splits therefore produce different keys by construction."""
    h = hashlib.sha256()
    h.update(str(tuple(tensor.shape)).encode())
    h.update(str(tensor.dtype).encode())
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:32]


def config_fingerprint(config: Dict[str, Any]) -> str:
    import json

    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


@dataclass
class CacheEntry:
    key: str
    artifact: Any
    fitted_on_split: str
    data_fingerprint: str
    config_fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    refusals: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits, "misses": self.misses, "refusals": self.refusals,
            "hit_rate": self.hit_rate,
        }


class FitCache:
    """In-process content-addressed cache for fitted artifacts (LBG codebooks,
    calibration ranges). Deliberately not persisted to disk: a stale on-disk
    cache surviving a code change is a contamination vector, and the speed win
    here is within-process anyway."""

    def __init__(self) -> None:
        self._entries: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    @staticmethod
    def make_key(data: torch.Tensor, config: Dict[str, Any], split: str) -> str:
        return f"{split}:{tensor_fingerprint(data)}:{config_fingerprint(config)}"

    def put(self, data: torch.Tensor, config: Dict[str, Any], split: str, artifact: Any) -> CacheEntry:
        entry = CacheEntry(
            key=self.make_key(data, config, split),
            artifact=artifact,
            fitted_on_split=split,
            data_fingerprint=tensor_fingerprint(data),
            config_fingerprint=config_fingerprint(config),
        )
        self._entries[entry.key] = entry
        return entry

    def get(self, data: torch.Tensor, config: Dict[str, Any], split: str) -> Optional[Any]:
        """Return the cached artifact, or None on a miss. Raises
        `CacheContaminationError` if an entry with the same data+config exists
        but was fitted on a DIFFERENT split - that is the contamination case,
        and it must be loud."""
        key = self.make_key(data, config, split)
        entry = self._entries.get(key)
        if entry is not None:
            self.stats.hits += 1
            return entry.artifact

        data_fp = tensor_fingerprint(data)
        config_fp = config_fingerprint(config)
        for other in self._entries.values():
            if other.data_fingerprint == data_fp and other.config_fingerprint == config_fp:
                self.stats.refusals += 1
                raise CacheContaminationError(
                    f"a cached artifact for this data+config exists but was fitted on split "
                    f"'{other.fitted_on_split}', not '{split}' - refusing to serve it, because "
                    f"reusing a fit across splits is exactly the leakage this cache guards against"
                )

        self.stats.misses += 1
        return None

    def get_or_fit(self, data: torch.Tensor, config: Dict[str, Any], split: str, fit_fn: Any) -> Any:
        cached = self.get(data, config, split)
        if cached is not None:
            return cached
        artifact = fit_fn(data)
        self.put(data, config, split, artifact)
        return artifact

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
