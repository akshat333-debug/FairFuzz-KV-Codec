import copy

import pytest
from transformers import AutoTokenizer

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.dataset_card import compute_split_hash
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import generate_dataset, generate_validated_dataset
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import FRAGMENTATION_LEVELS
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.validators import validate_dataset, validate_group

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def small_dataset(tokenizer):
    return generate_validated_dataset(30, tokenizer, seed=101)


def test_generation_is_deterministic(tokenizer):
    a = generate_dataset(20, tokenizer, seed=5)
    b = generate_dataset(20, tokenizer, seed=5)
    assert compute_split_hash(a) == compute_split_hash(b)


def test_different_seeds_produce_different_datasets(tokenizer):
    a = generate_dataset(20, tokenizer, seed=5)
    b = generate_dataset(20, tokenizer, seed=6)
    assert compute_split_hash(a) != compute_split_hash(b)


def test_every_group_has_all_four_fragmentation_levels(small_dataset):
    for g in small_dataset:
        assert set(g.variants.keys()) == set(FRAGMENTATION_LEVELS)


def test_matched_fields_identical_across_variants_in_a_group(small_dataset):
    for g in small_dataset:
        for n_g, variant in g.variants.items():
            assert variant.subject_name == g.subject_name
            assert variant.canonical_value == g.canonical_value
            assert variant.distractor_count == g.distractor_count
            assert variant.difficulty == g.difficulty
            assert variant.evidence_position_index == g.evidence_position_index


def test_all_generated_groups_pass_validation(tokenizer, small_dataset):
    reports = validate_dataset(small_dataset, tokenizer)
    assert all(r.passed for r in reports)


def test_scalable_generator_hits_200_group_prototype_minimum(tokenizer):
    groups = generate_validated_dataset(200, tokenizer, seed=999)
    assert len(groups) >= 200


def test_validator_catches_broken_evidence_identity(tokenizer, small_dataset):
    group = copy.deepcopy(small_dataset[0])
    group.canonical_value = (group.canonical_value + 1) % 10  # desync group vs its own variants
    report = validate_group(group, tokenizer)
    assert not report.passed
    assert any(r.check_name == "evidence_identity" and not r.passed for r in report.results)


def test_validator_catches_answer_leakage(tokenizer, small_dataset):
    group = copy.deepcopy(small_dataset[0])
    for variant in group.variants.values():
        variant.provenance["distractor_values"] = [group.canonical_value] + variant.provenance.get(
            "distractor_values", []
        )[1:]
    report = validate_group(group, tokenizer)
    assert not report.passed
    assert any(r.check_name == "no_answer_leakage" and not r.passed for r in report.results)


def test_validator_catches_position_mismatch(tokenizer, small_dataset):
    group = copy.deepcopy(small_dataset[0])
    one_ng = next(iter(group.variants))
    group.variants[one_ng].evidence_position_index += 1
    report = validate_group(group, tokenizer)
    assert not report.passed
    assert any(r.check_name == "context_position_matching" and not r.passed for r in report.results)
