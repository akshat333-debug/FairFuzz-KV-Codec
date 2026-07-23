"""Per-cohort distortion-vs-bits curves.

Fit D(b) = alpha * exp(-beta * b) ONLY when the data actually support it
(monotone-decreasing points and a good log-linear fit); otherwise fall back to
non-parametric monotone interpolation. We never present an ill-fitting
parametric curve as if it were the truth.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

R2_THRESHOLD = 0.9


@dataclass
class ExpFit:
    alpha: float
    beta: float
    r2: float
    supported: bool


def fit_exponential(bits: Sequence[float], distortions: Sequence[float]) -> Optional[ExpFit]:
    """Log-linear fit of D = alpha*exp(-beta*b). Returns None if it can't even
    be attempted (non-positive distortions, <2 distinct points). `supported`
    marks whether the fit is trustworthy (monotone + R^2 >= threshold)."""
    b = np.asarray(bits, dtype=float)
    d = np.asarray(distortions, dtype=float)
    if len(b) < 2 or np.any(d <= 0) or len(np.unique(b)) < 2:
        return None
    logd = np.log(d)
    # linear regression logd = log(alpha) - beta*b
    A = np.vstack([b, np.ones_like(b)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, logd, rcond=None)
    beta = -slope
    alpha = float(np.exp(intercept))
    pred = intercept + slope * b
    ss_res = float(np.sum((logd - pred) ** 2))
    ss_tot = float(np.sum((logd - logd.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    monotone = bool(np.all(np.diff(d[np.argsort(b)]) <= 1e-12))
    supported = monotone and beta > 0 and r2 >= R2_THRESHOLD
    return ExpFit(alpha=alpha, beta=beta, r2=r2, supported=supported)


class DistortionCurve:
    """Predicts distortion at an arbitrary bit-width. Uses the exponential fit
    when supported, else monotone (isotonic-lite) interpolation of the measured
    points - which always exists and never over-claims a functional form."""

    def __init__(self, bits: Sequence[float], distortions: Sequence[float]):
        order = np.argsort(np.asarray(bits, dtype=float))
        self.bits = np.asarray(bits, dtype=float)[order]
        self.distortions = _enforce_monotone_decreasing(np.asarray(distortions, dtype=float)[order])
        self.fit = fit_exponential(self.bits.tolist(), self.distortions.tolist())

    @property
    def uses_exponential(self) -> bool:
        return self.fit is not None and self.fit.supported

    def predict(self, b: float) -> float:
        if self.uses_exponential:
            assert self.fit is not None
            return float(self.fit.alpha * np.exp(-self.fit.beta * b))
        # non-parametric monotone interpolation (flat extrapolation at the ends)
        return float(np.interp(b, self.bits, self.distortions))

    def diagnostics(self) -> dict:
        return {
            "bits": self.bits.tolist(),
            "distortions": self.distortions.tolist(),
            "uses_exponential": self.uses_exponential,
            "alpha": self.fit.alpha if self.fit else None,
            "beta": self.fit.beta if self.fit else None,
            "r2": self.fit.r2 if self.fit else None,
        }


def _enforce_monotone_decreasing(d: np.ndarray) -> np.ndarray:
    """Clamp measured distortions to be non-increasing in bits (pool small
    inversions from measurement noise). Keeps predict() sane without pretending
    the raw points were perfectly clean."""
    out = d.copy()
    for i in range(1, len(out)):
        if out[i] > out[i - 1]:
            out[i] = out[i - 1]
    return out


def marginal_decay(curve: DistortionCurve) -> List[Tuple[float, float]]:
    """Marginal distortion reduction per extra bit between successive measured
    points - the diagnostic that shows where spending bits stops paying off."""
    out: List[Tuple[float, float]] = []
    for i in range(1, len(curve.bits)):
        db = curve.bits[i] - curve.bits[i - 1]
        dd = curve.distortions[i - 1] - curve.distortions[i]
        out.append((float(curve.bits[i]), float(dd / db) if db else 0.0))
    return out
