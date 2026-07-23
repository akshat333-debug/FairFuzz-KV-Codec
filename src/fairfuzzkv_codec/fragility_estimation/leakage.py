from typing import Dict

from fairfuzzkv_codec.fragility_estimation.schema import ALLOWED_FEATURE_NAMES

# Names that must never reach a risk score or cohort assignment: raw
# language/script labels, task performance labels, and compression outcomes.
# This is a denylist kept separate from ALLOWED_FEATURE_NAMES so an
# unrecognized-but-suspicious key (typo aside) still gets caught explicitly.
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "language",
        "language_hint",
        "script",
        "script_profile",
        "task_accuracy",
        "task_label",
        "compression_ratio",
        "compression_bits",
        "bits_per_element",
        "distortion",
    }
)


class LeakageError(ValueError):
    pass


def validate_no_leakage(feature_dict: Dict[str, float]) -> None:
    """Raise if feature_dict contains anything outside the whitelisted
    fragility features - in particular a raw language/script label or a
    task/compression outcome. Called at the entry point of every risk
    score and cohort function; never bypassed."""
    keys = set(feature_dict.keys())

    forbidden_present = keys & FORBIDDEN_FEATURE_NAMES
    if forbidden_present:
        raise LeakageError(f"forbidden feature(s) passed into fragility scoring: {sorted(forbidden_present)}")

    unknown = keys - ALLOWED_FEATURE_NAMES
    if unknown:
        raise LeakageError(f"unrecognized feature(s) not in the fragility whitelist: {sorted(unknown)}")
