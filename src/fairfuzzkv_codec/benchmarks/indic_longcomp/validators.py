"""Validators for IndicLongComp: parallelism (the non-negotiable "verified,
not loosely similar" check), answer auditability, PII scan, deduplication,
and a same-repo contamination self-check.
"""

import hashlib
import re
from typing import Iterable, List

from fairfuzzkv_codec.benchmarks.indic_longcomp.generator import ALL_LANGUAGES
from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import GroupValidationReport, IndicGroup, ValidationResult

# Simple, deliberately conservative patterns - this benchmark is fully
# synthetic (names from a fixed pool, template connective words, single
# digits), so a real match here would indicate a generation bug, not real PII.
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_PATTERN = re.compile(r"\b\d{7,}\b")


def validate_parallelism(group: IndicGroup) -> GroupValidationReport:
    """The non-negotiable check: every language variant must share the SAME
    canonical answer, evidence count, evidence position, and distractor
    count as the group - the mechanical definition of "parallel" this
    project uses (never translation-equivalence, which isn't verifiable
    without a professional reviewer we don't have - see package docstring)."""
    results: List[ValidationResult] = []

    languages_present = set(group.variants.keys())
    results.append(ValidationResult(
        passed=languages_present == set(ALL_LANGUAGES), check_name="language_coverage",
        detail=f"present={sorted(lang.value for lang in languages_present)}",
    ))

    identity_checks = (
        ("answer_identity", lambda v: v.canonical_answer, group.canonical_answer),
        ("evidence_count_identity", lambda v: v.evidence_count, group.evidence_count),
        ("evidence_position_identity", lambda v: v.evidence_position_index, group.evidence_position_index),
        ("distractor_count_identity", lambda v: v.distractor_count, group.distractor_count),
        ("task_family_identity", lambda v: v.task_family, group.task_family),
    )
    for check_name, extract, expected in identity_checks:
        mismatch = next(
            ((lang, extract(v)) for lang, v in group.variants.items() if extract(v) != expected), None,
        )
        if mismatch is None:
            results.append(ValidationResult(passed=True, check_name=check_name))
        else:
            lang, actual = mismatch
            results.append(ValidationResult(
                passed=False, check_name=check_name,
                detail=f"{lang.value}: {actual} != group {expected}",
            ))

    results.append(_context_length_alignment(group))

    return GroupValidationReport(group_id=group.group_id, passed=all(r.passed for r in results), results=results)


# Cross-language token-count inflation is the phenomenon this project STUDIES
# (see fragility_estimation's `token_cost_inflation`), so demanding equal token
# counts across languages would be wrong - Indic scripts legitimately tokenize
# longer than English for the same content. What "context length aligned"
# (Prompt 15 item 101) can mean here is: every variant is built from the SAME
# number of facts (guaranteed by construction and by the evidence/distractor
# identity checks above), and no variant's length is so far off the others that
# it signals a rendering bug rather than tokenizer inflation. This bound is a
# sanity ceiling, not a fairness claim.
MAX_CONTEXT_TOKEN_RATIO = 4.0


def _context_length_alignment(group: IndicGroup) -> ValidationResult:
    counts = {lang: v.context_token_count for lang, v in group.variants.items()}
    measured = [c for c in counts.values() if c > 0]
    if not measured:
        # no tokenizer was supplied at generation time - report that honestly
        # instead of silently "passing" an unmeasured check.
        return ValidationResult(
            passed=True, check_name="context_length_alignment",
            detail="not measured (no tokenizer supplied at generation time)",
        )
    ratio = max(measured) / max(1, min(measured))
    detail = (
        f"token counts by language={{{', '.join(f'{k.value}: {v}' for k, v in sorted(counts.items(), key=lambda kv: kv[0].value))}}}, "
        f"max/min ratio={ratio:.2f} (ceiling {MAX_CONTEXT_TOKEN_RATIO}; cross-language "
        f"inflation is expected and is itself under study, not a defect)"
    )
    return ValidationResult(passed=ratio <= MAX_CONTEXT_TOKEN_RATIO, check_name="context_length_alignment", detail=detail)


def validate_answer_auditability(group: IndicGroup) -> ValidationResult:
    """The canonical answer must be a plain single digit - exactly and
    trivially auditable, never a fuzzy-matched free-text answer."""
    if not (0 <= group.canonical_answer <= 9):
        return ValidationResult(
            passed=False, check_name="answer_auditability",
            detail=f"canonical_answer {group.canonical_answer} is not a single digit 0-9",
        )
    return ValidationResult(passed=True, check_name="answer_auditability")


def pii_scan(text: str) -> List[str]:
    findings = []
    if _EMAIL_PATTERN.search(text):
        findings.append("email_pattern")
    if _PHONE_PATTERN.search(text):
        findings.append("long_digit_run")
    return findings


def validate_no_answer_leakage(group: IndicGroup) -> ValidationResult:
    """The canonical answer must not be directly parseable out of the
    question text. Note: COUNTING questions legitimately state the target
    digit being counted (e.g. "how many people own code 8?") - that digit is
    a query PARAMETER, not the answer (the answer is the resulting COUNT),
    so this checks against canonical_answer specifically, not "any digit"."""
    from fairfuzzkv_codec.benchmarks.fragkv_minpairs.numeric_forms import parse_value

    for language, variant in group.variants.items():
        if parse_value(variant.question_text) == group.canonical_answer:
            return ValidationResult(
                passed=False, check_name="no_answer_leakage",
                detail=f"{language.value}: question text parses to the canonical answer {group.canonical_answer}",
            )
    return ValidationResult(passed=True, check_name="no_answer_leakage")


def validate_no_pii(groups: Iterable[IndicGroup]) -> ValidationResult:
    for group in groups:
        for variant in group.variants.values():
            findings = pii_scan(variant.context_text) + pii_scan(variant.question_text)
            if findings:
                return ValidationResult(
                    passed=False, check_name="pii_scan",
                    detail=f"{group.group_id}/{variant.language.value}: {findings}",
                )
    return ValidationResult(passed=True, check_name="pii_scan")


def find_duplicate_texts(groups: Iterable[IndicGroup]) -> List[str]:
    """Exact-duplicate context_text across all variants of all groups -
    hash-based, deterministic. Empty list means no duplicates."""
    seen: dict = {}
    duplicates = []
    for group in groups:
        for variant in group.variants.values():
            h = hashlib.sha256(variant.context_text.encode("utf-8")).hexdigest()
            key = f"{group.group_id}/{variant.language.value}"
            if h in seen:
                duplicates.append(f"{key} duplicates {seen[h]}")
            else:
                seen[h] = key
    return duplicates


def check_contamination_against(groups: Iterable[IndicGroup], other_texts: Iterable[str]) -> List[str]:
    """Self-check: none of this benchmark's context texts should appear
    verbatim in another dataset already committed to this repo (e.g.
    FragKV-MinPairs). Does NOT and cannot check against a model's actual
    pretraining corpus - no one outside the model provider has that data;
    documented as an explicit gap, not silently assumed clean."""
    other_set = set(other_texts)
    overlaps = []
    for group in groups:
        for variant in group.variants.values():
            if variant.context_text in other_set:
                overlaps.append(f"{group.group_id}/{variant.language.value}")
    return overlaps


def validate_dataset(groups: List[IndicGroup]) -> List[GroupValidationReport]:
    reports = []
    for group in groups:
        report = validate_parallelism(group)
        for extra in (validate_answer_auditability(group), validate_no_answer_leakage(group)):
            report.results.append(extra)
            report.passed = report.passed and extra.passed
        reports.append(report)
    return reports
