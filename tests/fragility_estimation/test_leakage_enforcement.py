import pytest

from fairfuzzkv_codec.fragility_estimation.leakage import (
    ALLOWED_FEATURE_NAMES,
    FORBIDDEN_FEATURE_NAMES,
    LeakageError,
    validate_no_leakage,
)


def test_forbidden_names_include_language_and_script():
    assert "language" in FORBIDDEN_FEATURE_NAMES
    assert "language_hint" in FORBIDDEN_FEATURE_NAMES
    assert "script" in FORBIDDEN_FEATURE_NAMES
    assert "script_profile" in FORBIDDEN_FEATURE_NAMES


def test_forbidden_names_include_task_and_compression_outcomes():
    assert "task_accuracy" in FORBIDDEN_FEATURE_NAMES
    assert "compression_ratio" in FORBIDDEN_FEATURE_NAMES
    assert "bits_per_element" in FORBIDDEN_FEATURE_NAMES


def test_allowed_and_forbidden_sets_are_disjoint():
    assert ALLOWED_FEATURE_NAMES.isdisjoint(FORBIDDEN_FEATURE_NAMES)


def test_valid_feature_dict_passes():
    valid = {name: 0.0 for name in ALLOWED_FEATURE_NAMES}
    validate_no_leakage(valid)  # must not raise


@pytest.mark.parametrize("leaked_key", sorted(FORBIDDEN_FEATURE_NAMES))
def test_every_forbidden_key_is_rejected(leaked_key):
    smuggled = {"num_subtokens": 1.0, leaked_key: "anything"}
    with pytest.raises(LeakageError):
        validate_no_leakage(smuggled)


def test_unknown_key_is_rejected_even_if_not_explicitly_forbidden():
    with pytest.raises(LeakageError):
        validate_no_leakage({"num_subtokens": 1.0, "some_new_field_nobody_reviewed": 1.0})
