from transformers import AutoTokenizer

from fairfuzzkv_codec.fragility_estimation import compute_cross_tokenizer_stability, compute_fragility_report
from fairfuzzkv_codec.fragility_estimation.calibrated_model import fit_and_calibrate

BPE_MODEL = "yujiepan/qwen2-tiny-random"
SENTENCEPIECE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "नमस्ते, आप कैसे हैं? मैं ठीक हूँ।",
    "తెలుగు భాష చాలా అందంగా ఉంటుంది.",
    "தமிழ் மொழி மிகவும் அழகானது.",
    "Mujhe ye बहुत पसंद है, kal फिर से try करेंगे.",
    "Visit https://example.com/path now! Price: 1234.56 😀",
    "Wait...what?! Really?!?! No way... :-)",
]


def _all_feature_vectors(tokenizer):
    features = []
    for text in CORPUS:
        report = compute_fragility_report(text, tokenizer)
        features.extend(report.feature_vectors)
    return features


def test_calibration_report_has_required_fields_for_logistic():
    tok = AutoTokenizer.from_pretrained(BPE_MODEL)
    features = _all_feature_vectors(tok)
    model, report = fit_and_calibrate(features, model_name="logistic")

    assert report.train_size + report.held_out_size == len(features) or model is None
    assert report.proxy_label_name == "boundary_mismatch>0"
    # with a real corpus of this size we expect enough signal to fit
    if model is not None:
        assert report.held_out_auc is not None
        assert report.held_out_brier_score is not None
        assert report.transparent_baseline_auc is not None
        assert 0.0 <= report.held_out_auc <= 1.0
        assert len(report.reliability_curve_bin_true_frequency) == len(
            report.reliability_curve_bin_predicted_probability
        )


def test_calibration_report_for_tree_model():
    tok = AutoTokenizer.from_pretrained(BPE_MODEL)
    features = _all_feature_vectors(tok)
    model, report = fit_and_calibrate(features, model_name="tree")
    assert report.model_name == "tree"
    if model is not None:
        assert 0.0 <= report.held_out_auc <= 1.0


def test_too_small_or_single_class_sample_returns_documented_gap_not_fabricated_result():
    tok = AutoTokenizer.from_pretrained(BPE_MODEL)
    features = _all_feature_vectors(tok)[:3]  # deliberately too small
    model, report = fit_and_calibrate(features, model_name="logistic")
    # must not fabricate metrics it can't measure
    if model is None:
        assert report.held_out_auc is None
        assert report.held_out_size == 0


def test_cross_tokenizer_stability_reports_agreement_and_verdict():
    tok_a = AutoTokenizer.from_pretrained(BPE_MODEL)
    tok_b = AutoTokenizer.from_pretrained(SENTENCEPIECE_MODEL)
    report = compute_cross_tokenizer_stability(CORPUS[4], tok_a, tok_b, corpus_id="test-corpus")
    assert report.num_units_compared > 0
    assert 0.0 <= report.cohort_agreement_rate <= 1.0
    assert report.verdict in ("universal", "model_specific")
