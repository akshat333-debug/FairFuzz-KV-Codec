from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# Bump whenever feature/score/cohort semantics change, so cohort manifests
# remain inspectable/versioned.
FRAGILITY_SCHEMA_VERSION = 1

# The only names allowed to flow into a risk score or cohort assignment.
# Raw language/script labels are intentionally excluded - they are
# descriptive-only (dashboard) and must never reach the allocation objective.
ALLOWED_FEATURE_NAMES = frozenset(
    {
        "num_subtokens",
        "chars_per_token",
        "bytes_per_token",
        "continuation_ratio",
        "script_transitions",
        "normalization_sensitivity",
        "rare_token_indicator",
        "boundary_mismatch",
        "token_cost_inflation",
    }
)


class FeatureVector(BaseModel):
    schema_version: int = FRAGILITY_SCHEMA_VERSION
    unit_char_span: Tuple[int, int]
    num_subtokens: float
    chars_per_token: float
    bytes_per_token: float
    continuation_ratio: float
    script_transitions: float
    normalization_sensitivity: float
    rare_token_indicator: float
    boundary_mismatch: float
    token_cost_inflation: float

    def to_feature_dict(self) -> Dict[str, float]:
        """Only the whitelisted numeric features - no span/version metadata,
        no script/language. This is what estimators/cohorts are allowed to see."""
        return {name: getattr(self, name) for name in ALLOWED_FEATURE_NAMES}


class RiskScore(BaseModel):
    schema_version: int = FRAGILITY_SCHEMA_VERSION
    unit_char_span: Tuple[int, int]
    score: float  # transparent monotone score, in [0, 1]
    feature_contributions: Dict[str, float]  # per-feature weighted contribution, sums to `score`


class CohortBand(BaseModel):
    label: str
    lower: float  # inclusive
    upper: float  # exclusive, except the top band which is inclusive
    count: int


class CohortDefinition(BaseModel):
    schema_version: int = FRAGILITY_SCHEMA_VERSION
    tokenizer_name: str
    corpus_id: str
    min_band_samples: int
    bands: List[CohortBand]


class CalibrationReport(BaseModel):
    schema_version: int = FRAGILITY_SCHEMA_VERSION
    model_name: str
    proxy_label_name: str
    train_size: int
    held_out_size: int
    held_out_auc: Optional[float] = None
    held_out_brier_score: Optional[float] = None
    transparent_baseline_auc: Optional[float] = None
    beats_transparent_baseline: bool = False
    reliability_curve_bin_true_frequency: List[float] = Field(default_factory=list)
    reliability_curve_bin_predicted_probability: List[float] = Field(default_factory=list)


class StabilityReport(BaseModel):
    schema_version: int = FRAGILITY_SCHEMA_VERSION
    tokenizer_a: str
    tokenizer_b: str
    num_units_compared: int
    cohort_agreement_rate: float
    verdict: str  # "universal" (high agreement) or "model_specific" (low agreement)
