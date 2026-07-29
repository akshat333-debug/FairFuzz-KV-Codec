"""Prompt 16 - full baseline matrix, regime-separated evaluation.

Compares this project's own scalar/LBG FairFuzzKV codecs against a mix of
FAITHFUL (this project's own established baselines), APPROXIMATE
(published-method core mechanisms, hyperparameters not verified against
reference code - no network access), and explicitly NOT_REPRODUCED
(documented reason + nearest faithful configuration, never silently
substituted under the original name) baselines. See `registry.py` for the
full matrix and `schema.py` for the card/result contract.
"""

from fairfuzzkv_codec.baselines.adapter import BaselineAdapter, run_matched_bit_comparison, tune_to_matched_bits
from fairfuzzkv_codec.baselines.schema import (
    AdapterResult, BaselineCard, EvaluationRegime, LatencyMeasurement, ReproductionStatus,
)

__all__ = [
    "BaselineAdapter", "run_matched_bit_comparison", "tune_to_matched_bits",
    "AdapterResult", "BaselineCard", "EvaluationRegime", "LatencyMeasurement", "ReproductionStatus",
]
