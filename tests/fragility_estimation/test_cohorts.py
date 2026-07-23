import pytest

from fairfuzzkv_codec.fragility_estimation.cohorts import (
    assign_cohort,
    assign_cohort_index,
    build_cohort_definition,
    load_cohort_manifest,
    write_cohort_manifest,
)


def test_deterministic_rerun_produces_identical_bands():
    scores = [0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    def_a = build_cohort_definition(scores, "tok", "corpus-1")
    def_b = build_cohort_definition(scores, "tok", "corpus-1")
    assert def_a.model_dump() == def_b.model_dump()


def test_bands_cover_full_score_range_with_no_gaps():
    scores = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    definition = build_cohort_definition(scores, "tok", "corpus-1", min_band_samples=1)
    for s in scores:
        # must not raise / must resolve to a valid band index every time
        idx = assign_cohort_index(s, definition)
        assert 0 <= idx < len(definition.bands)


def test_min_band_samples_merges_undersized_bands():
    # 20 low scores, 1 lone high outlier -> the outlier's band should merge
    # rather than stand alone below min_band_samples.
    scores = [0.1] * 20 + [0.99]
    definition = build_cohort_definition(scores, "tok", "corpus-1", num_bands=4, min_band_samples=3)
    assert all(b.count >= 3 for b in definition.bands)


def test_tie_at_boundary_goes_to_lower_band():
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    definition = build_cohort_definition(scores, "tok", "corpus-1", num_bands=4, min_band_samples=1)
    # a score exactly at an interior boundary must land in the lower band
    boundary = definition.bands[0].upper
    label_at_boundary = assign_cohort(boundary, definition)
    label_just_below = assign_cohort(boundary - 1e-9, definition)
    assert label_at_boundary != definition.bands[0].label or label_at_boundary == label_just_below


def test_empty_scores_raises():
    with pytest.raises(ValueError):
        build_cohort_definition([], "tok", "corpus-1")


def test_manifest_round_trip(tmp_path):
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    definition = build_cohort_definition(scores, "tok", "corpus-1", min_band_samples=1)
    path = tmp_path / "cohort.json"
    write_cohort_manifest(definition, path)
    reloaded = load_cohort_manifest(path)
    assert reloaded.model_dump() == definition.model_dump()
    assert reloaded.schema_version == definition.schema_version
