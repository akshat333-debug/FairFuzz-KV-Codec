"""Integration test for the real model + KV-splice study mechanism (runner.py).
Small and fast (2 groups) - the full 200-group Gate 1 study itself is run as
a one-off script (scripts/run_gate1_study.py), not as part of the test suite."""

import pytest
from transformers import AutoTokenizer

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import generate_validated_dataset
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.runner import FragKVRunner, build_codecs
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import FRAGMENTATION_LEVELS

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def runner():
    return FragKVRunner(MODEL_NAME)


def test_run_study_produces_one_record_per_group_ng_codec(tokenizer, runner):
    groups = generate_validated_dataset(2, tokenizer, seed=321)
    records = runner.run_study(groups, max_new_tokens=3)

    codec_names = [name for name, _ in build_codecs("x")]
    expected = len(groups) * len(FRAGMENTATION_LEVELS) * len(codec_names)
    assert len(records) == expected

    for r in records:
        assert r.codec_name in codec_names
        assert r.n_g in FRAGMENTATION_LEVELS
        assert r.kv_reconstruction_mse is not None
        assert r.kv_reconstruction_mse >= 0.0
        assert isinstance(r.correct, bool)


def test_fullkv_reconstruction_is_near_lossless(tokenizer, runner):
    """FullKV (fp16 cast, no compression) should reconstruct with only tiny
    fp16 rounding error - a sanity check that the splice mechanism isn't
    itself introducing distortion."""
    groups = generate_validated_dataset(1, tokenizer, seed=321)
    records = runner.run_study(groups, max_new_tokens=2)
    fullkv_records = [r for r in records if r.codec_name == "FullKV"]
    assert len(fullkv_records) == len(FRAGMENTATION_LEVELS)
    for r in fullkv_records:
        assert r.kv_reconstruction_mse < 0.01


def test_lossy_codecs_are_matched_at_8_bits_per_element(tokenizer, runner):
    groups = generate_validated_dataset(1, tokenizer, seed=321)
    records = runner.run_study(groups, max_new_tokens=2)
    for r in records:
        if r.codec_name in ("UniformQuant8", "TopK50"):
            assert abs(r.actual_bits_per_element - 8.0) < 0.5
        elif r.codec_name == "FullKV":
            assert abs(r.actual_bits_per_element - 16.0) < 0.5
