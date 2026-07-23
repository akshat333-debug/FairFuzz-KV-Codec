from typing import Any, List

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import TOKEN_COUNT_TOLERANCE
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.numeric_forms import measure_token_count, parse_value
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import (
    FRAGMENTATION_LEVELS,
    GroupValidationReport,
    MinPairGroup,
    ValidationResult,
)

CONTEXT_TOKEN_COUNT_TOLERANCE = 2


def validate_answer_equivalence(group: MinPairGroup) -> ValidationResult:
    """Every variant's rendering must parse back to the group's canonical
    value - this IS semantic equivalence for synthetic slot-filled content."""
    for n_g, variant in group.variants.items():
        parsed = parse_value(variant.evidence_rendering)
        if parsed != group.canonical_value:
            return ValidationResult(
                passed=False,
                check_name="answer_equivalence",
                detail=f"n_g={n_g} rendering {variant.evidence_rendering!r} parsed to {parsed}, expected {group.canonical_value}",
            )
    return ValidationResult(passed=True, check_name="answer_equivalence")


def validate_evidence_identity(group: MinPairGroup) -> ValidationResult:
    """All variants in a group must refer to the identical underlying fact:
    same subject, same canonical value."""
    for n_g, variant in group.variants.items():
        if variant.subject_name != group.subject_name or variant.canonical_value != group.canonical_value:
            return ValidationResult(
                passed=False,
                check_name="evidence_identity",
                detail=f"n_g={n_g} subject/value mismatch vs group ({variant.subject_name}, {variant.canonical_value}) != ({group.subject_name}, {group.canonical_value})",
            )
    return ValidationResult(passed=True, check_name="evidence_identity")


def validate_token_count_target(group: MinPairGroup, tokenizer: Any) -> ValidationResult:
    """Independently re-measure (don't trust the stored field) the evidence
    span's token count against its n_g target, and check total context
    length is matched across the group's variants within tolerance."""
    context_lengths = []
    for n_g, variant in group.variants.items():
        realized = measure_token_count(variant.evidence_rendering, tokenizer)
        tolerance = TOKEN_COUNT_TOLERANCE[n_g]
        if abs(realized - n_g) > tolerance:
            return ValidationResult(
                passed=False,
                check_name="token_count_target",
                detail=f"n_g={n_g} re-measured {realized}, outside tolerance {tolerance}",
            )
        if realized != variant.n_g_realized:
            return ValidationResult(
                passed=False,
                check_name="token_count_target",
                detail=f"n_g={n_g} stored n_g_realized={variant.n_g_realized} disagrees with re-measurement {realized}",
            )
        context_lengths.append(
            len(tokenizer(variant.context_text, add_special_tokens=False).input_ids)
        )

    if max(context_lengths) - min(context_lengths) > CONTEXT_TOKEN_COUNT_TOLERANCE:
        return ValidationResult(
            passed=False,
            check_name="token_count_target",
            detail=f"context lengths span {context_lengths}, exceeds tolerance {CONTEXT_TOKEN_COUNT_TOLERANCE}",
        )
    return ValidationResult(passed=True, check_name="token_count_target")


def validate_context_position_matching(group: MinPairGroup) -> ValidationResult:
    """The evidence fact must sit at the same sentence index, among the same
    distractor identities, across every variant of the group."""
    reference_variant = group.variants[FRAGMENTATION_LEVELS[0]]
    reference_names = reference_variant.provenance.get("distractor_names")
    reference_values = reference_variant.provenance.get("distractor_values")

    for n_g, variant in group.variants.items():
        if variant.evidence_position_index != group.evidence_position_index:
            return ValidationResult(
                passed=False,
                check_name="context_position_matching",
                detail=f"n_g={n_g} evidence_position_index {variant.evidence_position_index} != group {group.evidence_position_index}",
            )
        if variant.provenance.get("distractor_names") != reference_names:
            return ValidationResult(
                passed=False,
                check_name="context_position_matching",
                detail=f"n_g={n_g} distractor_names differ from n_g={FRAGMENTATION_LEVELS[0]}",
            )
        if variant.provenance.get("distractor_values") != reference_values:
            return ValidationResult(
                passed=False,
                check_name="context_position_matching",
                detail=f"n_g={n_g} distractor_values differ from n_g={FRAGMENTATION_LEVELS[0]}",
            )

        # Split on the literal "Fact: " marker rather than periods - several
        # renderings (digit_dot, word_dot_letters) contain embedded periods
        # that would otherwise fragment a period-based sentence split.
        fact_chunks = variant.context_text.split("Fact: ")[1:]
        target_idx = next(
            (i for i, chunk in enumerate(fact_chunks) if chunk.startswith(f"{variant.subject_name} ")), None
        )
        if target_idx != variant.evidence_position_index:
            return ValidationResult(
                passed=False,
                check_name="context_position_matching",
                detail=f"n_g={n_g} target fact found at sentence index {target_idx}, expected {variant.evidence_position_index}",
            )
    return ValidationResult(passed=True, check_name="context_position_matching")


def validate_no_answer_leakage(group: MinPairGroup) -> ValidationResult:
    """The canonical value must not be guessable from anything other than
    the designated evidence fact: no distractor may share it, and the
    question/filler text must not mention it."""
    for n_g, variant in group.variants.items():
        distractor_values: List[int] = variant.provenance.get("distractor_values", [])
        if group.canonical_value in distractor_values:
            return ValidationResult(
                passed=False,
                check_name="no_answer_leakage",
                detail=f"n_g={n_g} canonical value {group.canonical_value} also appears among distractor values {distractor_values}",
            )
        if parse_value(variant.question_text) is not None:
            return ValidationResult(
                passed=False,
                check_name="no_answer_leakage",
                detail=f"n_g={n_g} question text unexpectedly contains a parseable value",
            )
    return ValidationResult(passed=True, check_name="no_answer_leakage")


VALIDATORS = [
    validate_answer_equivalence,
    validate_evidence_identity,
    validate_context_position_matching,
    validate_no_answer_leakage,
]


def validate_group(group: MinPairGroup, tokenizer: Any) -> GroupValidationReport:
    results = [v(group) for v in VALIDATORS]
    results.append(validate_token_count_target(group, tokenizer))
    return GroupValidationReport(
        group_id=group.group_id,
        passed=all(r.passed for r in results),
        results=results,
    )


def validate_dataset(groups: List[MinPairGroup], tokenizer: Any) -> List[GroupValidationReport]:
    return [validate_group(g, tokenizer) for g in groups]
