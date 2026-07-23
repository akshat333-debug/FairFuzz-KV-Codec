from typing import Any

from fairfuzzkv_codec.fragility_estimation.cohorts import assign_cohort_index, build_cohort_definition
from fairfuzzkv_codec.fragility_estimation.features import compute_features_for_records
from fairfuzzkv_codec.fragility_estimation.schema import StabilityReport
from fairfuzzkv_codec.fragility_estimation.transparent_score import transparent_monotone_score
from fairfuzzkv_codec.unicode_grouping import GroupMapper

# Below this agreement rate, risk is reported as tokenizer-specific rather
# than a claim that generalizes across tokenizer families.
UNIVERSAL_AGREEMENT_THRESHOLD = 0.7


def compute_cross_tokenizer_stability(
    text: str,
    tokenizer_a: Any,
    tokenizer_b: Any,
    corpus_id: str,
) -> StabilityReport:
    """Compare cohort assignment for the SAME text under two tokenizer
    families. Surface units (Module 1) are tokenizer-independent, so the
    same char spans exist under both tokenizers - this lets us compare
    like-for-like ordinal risk-band assignment rather than approximate
    matching."""
    name_a = getattr(tokenizer_a, "name_or_path", type(tokenizer_a).__name__)
    name_b = getattr(tokenizer_b, "name_or_path", type(tokenizer_b).__name__)

    result_a = GroupMapper(tokenizer_a).map(text)
    result_b = GroupMapper(tokenizer_b).map(text)

    features_a = compute_features_for_records(result_a.records, tokenizer_a, text)
    features_b = compute_features_for_records(result_b.records, tokenizer_b, text)

    scores_a = [transparent_monotone_score(fv).score for fv in features_a]
    scores_b = [transparent_monotone_score(fv).score for fv in features_b]

    cohort_def_a = build_cohort_definition(scores_a, name_a, corpus_id)
    cohort_def_b = build_cohort_definition(scores_b, name_b, corpus_id)

    span_to_score_a = {fv.unit_char_span: s for fv, s in zip(features_a, scores_a)}
    span_to_score_b = {fv.unit_char_span: s for fv, s in zip(features_b, scores_b)}
    common_spans = sorted(set(span_to_score_a) & set(span_to_score_b))

    if not common_spans:
        return StabilityReport(
            tokenizer_a=name_a,
            tokenizer_b=name_b,
            num_units_compared=0,
            cohort_agreement_rate=0.0,
            verdict="model_specific",
        )

    agree = 0
    for span in common_spans:
        idx_a = assign_cohort_index(span_to_score_a[span], cohort_def_a)
        idx_b = assign_cohort_index(span_to_score_b[span], cohort_def_b)
        if idx_a == idx_b:
            agree += 1

    agreement_rate = agree / len(common_spans)
    verdict = "universal" if agreement_rate >= UNIVERSAL_AGREEMENT_THRESHOLD else "model_specific"

    return StabilityReport(
        tokenizer_a=name_a,
        tokenizer_b=name_b,
        num_units_compared=len(common_spans),
        cohort_agreement_rate=agreement_rate,
        verdict=verdict,
    )
