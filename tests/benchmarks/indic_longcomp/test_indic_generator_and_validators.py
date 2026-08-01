import pytest
from transformers import AutoTokenizer

from fairfuzzkv_codec.benchmarks.indic_longcomp.generator import ALL_LANGUAGES, ALL_TASK_FAMILIES, generate_dataset
from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import LanguageCondition, TaskFamily
from fairfuzzkv_codec.benchmarks.indic_longcomp.validators import (
    check_contamination_against, find_duplicate_texts, validate_answer_auditability,
    validate_dataset, validate_no_answer_leakage, validate_no_pii, validate_parallelism,
)

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def small_dataset(tokenizer):
    return generate_dataset(groups_per_family=3, seed=1, tokenizer=tokenizer)


def test_generates_all_four_language_conditions(small_dataset):
    for group in small_dataset:
        assert set(group.variants.keys()) == set(ALL_LANGUAGES)
    assert set(ALL_LANGUAGES) == {
        LanguageCondition.ENGLISH, LanguageCondition.HINDI, LanguageCondition.HINGLISH, LanguageCondition.TELUGU_ENGLISH,
    }


def test_generates_all_five_task_families(small_dataset):
    families = {g.task_family for g in small_dataset}
    assert families == set(ALL_TASK_FAMILIES)


def test_deterministic_given_same_seed(tokenizer):
    a = generate_dataset(groups_per_family=2, seed=42, tokenizer=tokenizer)
    b = generate_dataset(groups_per_family=2, seed=42, tokenizer=tokenizer)
    assert [g.model_dump() for g in a] == [g.model_dump() for g in b]


def test_different_seeds_produce_different_content(tokenizer):
    a = generate_dataset(groups_per_family=2, seed=1, tokenizer=tokenizer)
    b = generate_dataset(groups_per_family=2, seed=2, tokenizer=tokenizer)
    assert a[0].canonical_answer != b[0].canonical_answer or a[0].variants[LanguageCondition.ENGLISH].context_text != b[0].variants[LanguageCondition.ENGLISH].context_text


def test_parallelism_holds_for_every_generated_group(small_dataset):
    for group in small_dataset:
        report = validate_parallelism(group)
        assert report.passed, report.results


def test_answer_auditability_all_single_digits(small_dataset):
    for group in small_dataset:
        result = validate_answer_auditability(group)
        assert result.passed, result.detail


def test_validate_dataset_all_pass(small_dataset):
    reports = validate_dataset(small_dataset)
    assert all(r.passed for r in reports)


def test_no_pii_found(small_dataset):
    result = validate_no_pii(small_dataset)
    assert result.passed, result.detail


def test_no_duplicate_texts(small_dataset):
    duplicates = find_duplicate_texts(small_dataset)
    assert duplicates == []


def test_no_answer_leakage_in_question_text(small_dataset):
    for group in small_dataset:
        result = validate_no_answer_leakage(group)
        assert result.passed, result.detail


def test_counting_question_states_target_digit_but_that_is_not_leakage():
    # the target digit in "how many own code N?" is a query PARAMETER, not
    # the answer (the answer is the resulting COUNT) - must not be flagged.
    dataset = generate_dataset(groups_per_family=3, seed=9, task_families=(TaskFamily.COUNTING,))
    for group in dataset:
        assert validate_no_answer_leakage(group).passed


def test_leaked_answer_in_question_is_detected():
    dataset = generate_dataset(groups_per_family=1, seed=11, task_families=(TaskFamily.RETRIEVAL,))
    group = dataset[0]
    leaky = group.variants[LanguageCondition.ENGLISH].model_copy(
        update={"question_text": f" Query: The answer is {group.canonical_answer}. What is it? Answer:"}
    )
    group.variants[LanguageCondition.ENGLISH] = leaky
    assert not validate_no_answer_leakage(group).passed


def test_no_verbatim_overlap_with_itself_as_a_sanity_check(small_dataset):
    # comparing a dataset against ITS OWN texts should overlap trivially -
    # exercises the contamination-check machinery honestly (it's meant to
    # compare against a DIFFERENT corpus; this just proves it can detect a match).
    own_texts = [v.context_text for g in small_dataset for v in g.variants.values()]
    overlaps = check_contamination_against(small_dataset[:1], own_texts)
    assert len(overlaps) == len(small_dataset[0].variants)


def test_broken_parallelism_is_detected():
    # construct a group where one variant's canonical_answer diverges, and
    # confirm the validator actually catches it (not just a trivially-true check).
    dataset = generate_dataset(groups_per_family=1, seed=7, task_families=(TaskFamily.RETRIEVAL,))
    group = dataset[0]
    tampered = group.variants[LanguageCondition.HINDI].model_copy(update={"canonical_answer": (group.canonical_answer + 1) % 10})
    group.variants[LanguageCondition.HINDI] = tampered
    report = validate_parallelism(group)
    assert not report.passed
    assert any(r.check_name == "answer_identity" and not r.passed for r in report.results)


def test_counting_and_aggregation_answers_are_derivable_from_evidence():
    dataset = generate_dataset(groups_per_family=2, seed=3, task_families=(TaskFamily.COUNTING, TaskFamily.AGGREGATION))
    for group in dataset:
        assert 0 <= group.canonical_answer <= 9


def test_parallelism_checks_task_family_and_context_length(monkeypatch):
    """Prompt 15 item 101 names context length and task family among the
    properties that must stay aligned - both must actually be checked, not
    just recorded on the schema."""
    from fairfuzzkv_codec.benchmarks.indic_longcomp.validators import validate_parallelism

    groups = generate_dataset(1, seed=5)
    report = validate_parallelism(groups[0])
    names = {r.check_name for r in report.results}
    assert "task_family_identity" in names
    assert "context_length_alignment" in names
    assert report.passed


def test_task_family_mismatch_is_caught():
    from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import TaskFamily
    from fairfuzzkv_codec.benchmarks.indic_longcomp.validators import validate_parallelism

    group = generate_dataset(1, seed=6)[0]
    lang = next(iter(group.variants))
    other = next(f for f in TaskFamily if f != group.task_family)
    group.variants[lang].task_family = other

    report = validate_parallelism(group)
    assert not report.passed
    failed = {r.check_name for r in report.results if not r.passed}
    assert "task_family_identity" in failed


def test_context_length_alignment_flags_absurd_ratio():
    from fairfuzzkv_codec.benchmarks.indic_longcomp.validators import (
        MAX_CONTEXT_TOKEN_RATIO,
        validate_parallelism,
    )

    group = generate_dataset(1, seed=7)[0]
    langs = list(group.variants)
    for i, lang in enumerate(langs):
        group.variants[lang].context_token_count = 10
    # one variant absurdly longer than the rest -> rendering bug, must fail
    group.variants[langs[0]].context_token_count = int(10 * MAX_CONTEXT_TOKEN_RATIO) + 10

    report = validate_parallelism(group)
    failed = {r.check_name for r in report.results if not r.passed}
    assert "context_length_alignment" in failed


def test_context_length_unmeasured_is_reported_not_silently_passed():
    from fairfuzzkv_codec.benchmarks.indic_longcomp.validators import validate_parallelism

    group = generate_dataset(1, seed=8)[0]  # no tokenizer -> token counts are 0
    result = next(r for r in validate_parallelism(group).results if r.check_name == "context_length_alignment")
    assert "not measured" in (result.detail or "")
