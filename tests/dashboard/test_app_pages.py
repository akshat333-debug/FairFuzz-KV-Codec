"""Every dashboard page must actually render without raising.

Uses Streamlit's own AppTest harness, so this is a real render of the real app
against the real frozen artifacts - not a mock.
"""

import pytest

from fairfuzzkv_codec.dashboard.artifacts import REPO_ROOT

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP = str(REPO_ROOT / "dashboard_app.py")

# Pages that render purely from frozen artifacts (no button press required).
ARTIFACT_PAGES = [
    "Overview & Gate Decisions",
    "Claims & Limitations",
    "Dataset Inspection",
    "Tokenizer Fragmentation",
    "Compression Configuration",
    "Bitstream Anatomy",
    "Reconstructed-Cache Diagnostics",
    "Rate-Distortion Curves",
    "Fairness Trade-offs",
    "Baseline Matrix",
    "Systems Profiling",
]


def _run_page(name: str):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(name).run()
    return at


@pytest.mark.parametrize("page", ARTIFACT_PAGES)
def test_page_renders_without_exception(page):
    at = _run_page(page)
    assert not at.exception, f"{page} raised: {[str(e) for e in at.exception]}"


def test_overview_shows_every_gate_including_the_failures():
    at = _run_page("Overview & Gate Decisions")
    values = [m.value for m in at.metric]
    assert "WEAK_PASS" in values
    assert values.count("FAIL") >= 2  # Gate 2 and Gate 4 both FAILED
    assert "PASS" in values


def test_claims_page_exposes_negative_claims():
    at = _run_page("Claims & Limitations")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(str(m.value) for m in at.metric)
    assert "negative" in text.lower() or any("negative" in str(m.value).lower() for m in at.metric)


def test_fairness_page_states_the_gate2_failure_prominently():
    at = _run_page("Fairness Trade-offs")
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "Gate 2 FAILED" in errors


def test_reconstruction_page_flags_unmatched_budgets():
    """int8 vs int4 vs LBG are deliberately NOT matched, so the page must warn."""
    at = _run_page("Reconstructed-Cache Diagnostics")
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "UNMATCHED BUDGETS" in errors


def test_baseline_page_marks_not_reproduced_baselines():
    at = _run_page("Baseline Matrix")
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "NOT REPRODUCED" in errors
    assert "RateQuant" in errors


def test_systems_page_states_the_integration_boundary():
    at = _run_page("Systems Profiling")
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "Integration boundary" in warnings
    assert "not" in warnings.lower()  # decode is explicitly not a serving number


def test_interactive_demo_renders_and_exposes_byte_accounting():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("Interactive Text Demo").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception, f"demo raised: {[str(e) for e in at.exception]}"
    labels = [m.label for m in at.metric]
    # complete byte accounting must be on screen
    assert "Serialized bytes" in labels
    assert "Logical bits" in labels
    assert "Overhead bytes" in labels
    assert "KV MSE" in labels
