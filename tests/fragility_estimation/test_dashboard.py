import os

from transformers import AutoTokenizer

from fairfuzzkv_codec.fragility_estimation import compute_fragility_report, dominant_script_label, plot_risk_by_script

MODEL = "yujiepan/qwen2-tiny-random"


def test_plot_risk_by_script_generates_file(tmp_path):
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "Hello नमस्ते world 😀"
    report = compute_fragility_report(text, tok)
    scores = [r.score for r in report.risk_scores]

    out_path = plot_risk_by_script(report.mapper_result.records, scores, str(tmp_path))
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_dominant_script_label_is_descriptive_only():
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "Hello नमस्ते"
    report = compute_fragility_report(text, tok)
    labels = [dominant_script_label(r) for r in report.mapper_result.records]
    assert "Latin" in labels
    assert "Devanagari" in labels
