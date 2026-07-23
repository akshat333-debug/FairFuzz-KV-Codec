import pytest
from transformers import AutoTokenizer

from fairfuzzkv_codec.fragility_estimation import compute_fragility_report
from fairfuzzkv_codec.fragility_estimation.schema import ALLOWED_FEATURE_NAMES

BPE_MODEL = "yujiepan/qwen2-tiny-random"
SENTENCEPIECE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "नमस्ते, आप कैसे हैं?",
    "తెలుగు భాష చాలా అందంగా ఉంటుంది.",
    "Mujhe ye बहुत पसंद है, kal फिर से try करेंगे.",
    "Visit https://example.com/path now! Price: 1234.56 😀",
]


@pytest.fixture(scope="module", params=[BPE_MODEL, SENTENCEPIECE_MODEL])
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


@pytest.mark.parametrize("text", CORPUS)
def test_one_feature_vector_per_group_record(tokenizer, text):
    report = compute_fragility_report(text, tokenizer)
    assert len(report.feature_vectors) == len(report.mapper_result.records)
    assert len(report.risk_scores) == len(report.feature_vectors)


@pytest.mark.parametrize("text", CORPUS)
def test_feature_vector_only_exposes_whitelisted_features(tokenizer, text):
    report = compute_fragility_report(text, tokenizer)
    for fv in report.feature_vectors:
        assert set(fv.to_feature_dict().keys()) == ALLOWED_FEATURE_NAMES


@pytest.mark.parametrize("text", CORPUS)
def test_features_are_finite_numbers(tokenizer, text):
    report = compute_fragility_report(text, tokenizer)
    for fv in report.feature_vectors:
        for value in fv.to_feature_dict().values():
            assert value == value  # not NaN
            assert value != float("inf") and value != float("-inf")


def test_reruns_are_deterministic(tokenizer):
    text = CORPUS[3]
    report_a = compute_fragility_report(text, tokenizer)
    report_b = compute_fragility_report(text, tokenizer)
    scores_a = [r.score for r in report_a.risk_scores]
    scores_b = [r.score for r in report_b.risk_scores]
    assert scores_a == scores_b
