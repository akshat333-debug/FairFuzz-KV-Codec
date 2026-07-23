from fairfuzzkv_codec.fragility_estimation.calibrated_model import fit_and_calibrate, proxy_label
from fairfuzzkv_codec.fragility_estimation.cohorts import (
    assign_cohort,
    assign_cohort_index,
    build_cohort_definition,
    load_cohort_manifest,
    write_cohort_manifest,
)
from fairfuzzkv_codec.fragility_estimation.dashboard import dominant_script_label, plot_risk_by_script
from fairfuzzkv_codec.fragility_estimation.features import compute_features, compute_features_for_records
from fairfuzzkv_codec.fragility_estimation.leakage import LeakageError, validate_no_leakage
from fairfuzzkv_codec.fragility_estimation.pipeline import FragilityReport, compute_fragility_report
from fairfuzzkv_codec.fragility_estimation.reference import get_reference_chars_per_token
from fairfuzzkv_codec.fragility_estimation.schema import (
    ALLOWED_FEATURE_NAMES,
    FRAGILITY_SCHEMA_VERSION,
    CalibrationReport,
    CohortBand,
    CohortDefinition,
    FeatureVector,
    RiskScore,
    StabilityReport,
)
from fairfuzzkv_codec.fragility_estimation.stability import compute_cross_tokenizer_stability
from fairfuzzkv_codec.fragility_estimation.transparent_score import TRANSPARENT_WEIGHTS, transparent_monotone_score

__all__ = [
    "ALLOWED_FEATURE_NAMES",
    "FRAGILITY_SCHEMA_VERSION",
    "TRANSPARENT_WEIGHTS",
    "CalibrationReport",
    "CohortBand",
    "CohortDefinition",
    "FeatureVector",
    "FragilityReport",
    "LeakageError",
    "RiskScore",
    "StabilityReport",
    "assign_cohort",
    "assign_cohort_index",
    "build_cohort_definition",
    "compute_cross_tokenizer_stability",
    "compute_features",
    "compute_features_for_records",
    "compute_fragility_report",
    "dominant_script_label",
    "fit_and_calibrate",
    "get_reference_chars_per_token",
    "load_cohort_manifest",
    "plot_risk_by_script",
    "proxy_label",
    "transparent_monotone_score",
    "validate_no_leakage",
    "write_cohort_manifest",
]
