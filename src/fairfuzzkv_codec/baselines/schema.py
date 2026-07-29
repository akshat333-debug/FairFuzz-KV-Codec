"""Common baseline-adapter contract (Prompt 16 item 109) and provenance
schema (item 113). Every baseline in the matrix - reproduced or not - gets
exactly one `BaselineCard`, so nothing is silently missing or silently
mislabeled.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class EvaluationRegime(str, Enum):
    """Prompt 16 item 108: compression/quantization, prefill-time selection,
    and decode-time selection are SEPARATE result regimes - never mixed into
    one table (a decode-time selector like H2O is not comparable to a
    prefill-time selector like SnapKV on the same table, since they operate
    on different information and at a different point in generation)."""

    COMPRESSION_QUANTIZATION = "compression_quantization"
    PREFILL_SELECTION = "prefill_selection"
    DECODE_TIME_SELECTION = "decode_time_selection"


class ReproductionStatus(str, Enum):
    """Item 111: never silently reimplement a different algorithm under the
    same published name. Every non-FAITHFUL entry must say why and what the
    nearest faithful configuration is instead."""

    FAITHFUL = "faithful"  # matches the published algorithm's defining mechanism exactly, to the best available specification
    APPROXIMATE = "approximate"  # core published mechanism reproduced; some implementation-level details (e.g. exact hyperparameter defaults) are not verified against the original reference code, because no network access was available to fetch it
    NOT_REPRODUCED = "not_reproduced"  # not implemented under this name; see nearest_faithful_configuration


@dataclass
class BaselineCard:
    """Provenance/configuration card - Prompt 16 acceptance gate: "Every
    baseline has a provenance/configuration card." Required for BOTH
    reproduced and not-reproduced baselines."""

    name: str
    regime: EvaluationRegime
    reproduction_status: ReproductionStatus
    version_note: str  # paper/mechanism cited, and what could/couldn't be verified without network access
    model_support: str
    context_limit_note: str
    deviations: str  # empty string if none
    limitations: str
    nearest_faithful_configuration: str = ""  # required (non-empty) when reproduction_status == NOT_REPRODUCED

    def __post_init__(self) -> None:
        if self.reproduction_status == ReproductionStatus.NOT_REPRODUCED and not self.nearest_faithful_configuration:
            raise ValueError(f"{self.name}: NOT_REPRODUCED baselines must state a nearest_faithful_configuration")

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name, "regime": self.regime.value, "reproduction_status": self.reproduction_status.value,
            "version_note": self.version_note, "model_support": self.model_support,
            "context_limit_note": self.context_limit_note, "deviations": self.deviations,
            "limitations": self.limitations, "nearest_faithful_configuration": self.nearest_faithful_configuration,
        }


@dataclass
class LatencyMeasurement:
    encode_seconds: float
    decode_seconds: float
    measured: bool = True  # False only if latency could not be measured (never estimated silently)


@dataclass
class AdapterResult:
    """One matched-bit comparison outcome for one baseline on one input."""

    baseline_name: str
    regime: EvaluationRegime
    target_bits_per_element: float
    actual_bits_per_element: float
    matched: bool  # within MatchedBitTuner's tolerance
    kv_mse: float
    latency: Optional[LatencyMeasurement] = None
    limitations_triggered: str = ""  # non-empty if this specific run hit a documented limitation
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "baseline_name": self.baseline_name, "regime": self.regime.value,
            "target_bits_per_element": self.target_bits_per_element,
            "actual_bits_per_element": self.actual_bits_per_element, "matched": self.matched,
            "kv_mse": self.kv_mse,
            "latency": None if self.latency is None else {
                "encode_seconds": self.latency.encode_seconds, "decode_seconds": self.latency.decode_seconds,
                "measured": self.latency.measured,
            },
            "limitations_triggered": self.limitations_triggered, "extra": self.extra,
        }
