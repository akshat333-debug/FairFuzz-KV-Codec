import json

import pytest
from transformers import AutoTokenizer

from fairfuzzkv_codec.unicode_grouping import GroupMapper, NormalizationPolicy

MODEL = "yujiepan/qwen2-tiny-random"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def test_audit_report_is_json_serializable_and_versioned(tokenizer):
    mapper = GroupMapper(tokenizer)
    result = mapper.map("Hello world, visit https://example.com now! 123 😀")
    report = mapper.audit_report(result)

    # must round-trip through JSON (inspectable/versioned per non-negotiable instruction)
    serialized = json.dumps(report)
    reloaded = json.loads(serialized)
    assert reloaded["schema_version"] == 1
    assert reloaded["num_units"] == len(result.records)
    assert reloaded["num_quarantined_tokens"] == len(result.quarantine)


def test_render_debug_contains_all_units(tokenizer):
    mapper = GroupMapper(tokenizer)
    result = mapper.map("abc 123")
    debug_text = mapper.render_debug(result)
    for r in result.records:
        assert repr(r.original_text) in debug_text


def test_preserve_original_policy_never_mutates_text(tokenizer):
    text = "café naïve"
    mapper = GroupMapper(tokenizer, normalization_policy=NormalizationPolicy.PRESERVE_ORIGINAL)
    result = mapper.map(text)
    assert result.text == text
    for r in result.records:
        assert r.normalized_span == r.char_span


def test_nfc_policy_computes_normalized_spans_without_touching_original(tokenizer):
    text = "café"  # composed é
    mapper = GroupMapper(tokenizer, normalization_policy=NormalizationPolicy.NFC)
    result = mapper.map(text)
    # original_text on every record must remain untouched
    assert "".join(r.original_text for r in result.records) == text
    for r in result.records:
        assert r.normalized_span[1] - r.normalized_span[0] > 0 or r.original_text == ""


def test_curated_corpus_round_trip_coverage_meets_995_percent(tokenizer):
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "नमस्ते, आप कैसे हैं?",
        "తెలుగు భాష చాలా అందంగా ఉంటుంది.",
        "தமிழ் மொழி மிகவும் அழகானது.",
        "Mujhe ye movie बहुत पसंद आई!",
        "😀😃😄 party!! 🎉",
        "Visit https://example.com/path now.",
        "Price is 1234.56 and count is 007.",
    ]
    mapper = GroupMapper(tokenizer)
    passed = 0
    for text in corpus:
        result = mapper.map(text)
        recon = "".join(r.original_text for r in result.records)
        if recon == text:
            passed += 1
    assert passed / len(corpus) >= 0.995


def test_raises_on_slow_tokenizer_construction():
    class FakeSlowTokenizer:
        is_fast = False

    with pytest.raises(ValueError):
        GroupMapper(FakeSlowTokenizer())
