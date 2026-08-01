"""Proposition 1 (Fragility Distribution).

Stated precisely, as the property the project actually relies on - NOT the
stronger causal claim Gate 1 failed to establish:

    P1. The transparent fragility score induces a non-degenerate, reproducible
        ordering over surface units, in which units that are demonstrably more
        fragmented under a given tokenizer receive strictly higher scores, and
        quantile cohorts partition that ordering into ordered, covering bands.

P1 is a property of the ESTIMATOR (Module 2), not a claim that fragility causes
downstream compression failure - that is Gate 1's question, and Gate 1 came
back WEAK_PASS (RISK_REGISTER R-06). These tests must therefore never be cited
as evidence for the causal claim.
"""

import pytest
from transformers import AutoTokenizer

from fairfuzzkv_codec.fragility_estimation.cohorts import assign_cohort, build_cohort_definition
from fairfuzzkv_codec.fragility_estimation.pipeline import compute_fragility_report

BPE_MODEL = "yujiepan/qwen2-tiny-random"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(BPE_MODEL)


def _scores(text: str, tokenizer) -> list:
    return [rs.score for rs in compute_fragility_report(text, tokenizer).risk_scores]


def test_p1_scores_are_bounded_and_non_degenerate(tokenizer):
    """A score that is constant everywhere induces no ordering at all, so the
    distribution must actually spread."""
    text = "The report shows 4321 units. Mujhe ye बहुत पसंद है 😀 https://example.com/x?y=1"
    scores = _scores(text, tokenizer)
    assert len(scores) > 5
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert max(scores) > min(scores)  # non-degenerate


def test_p1_is_reproducible(tokenizer):
    """Same text + same tokenizer must give byte-identical scores; an estimator
    that drifts between runs cannot support cohort assignment."""
    text = "मेरा नाम राहुल है and I work in tech with 42 colleagues."
    assert _scores(text, tokenizer) == _scores(text, tokenizer)


def test_p1_more_fragmented_units_score_higher(tokenizer):
    """The directional core of P1: a unit the tokenizer shatters into many
    subtokens must not score BELOW a unit it keeps whole."""
    report = compute_fragility_report(
        "the cat sat on विश्वविद्यालयीन आंतरविद्याशाखीय", tokenizer,
    )
    pairs = [
        (fv.num_subtokens, rs.score)
        for fv, rs in zip(report.feature_vectors, report.risk_scores)
    ]
    fewest = min(pairs, key=lambda p: p[0])
    most = max(pairs, key=lambda p: p[0])
    assert most[0] > fewest[0], "test text failed to produce a fragmentation contrast"
    assert most[1] >= fewest[1]


def test_p1_cohorts_are_ordered_and_covering(tokenizer):
    """Quantile bands must tile the score range in order, with no gap that
    would leave a score unassignable."""
    text = (
        "Revenue rose to 4321 units. मेरा नाम राहुल है। "
        "नా పేరు రాము and the URL is https://example.com/docs 😀"
    )
    scores = _scores(text, tokenizer)
    definition = build_cohort_definition(scores, tokenizer_name=BPE_MODEL, corpus_id="p1")

    bands = definition.bands
    assert len(bands) >= 2
    for lower, upper in zip(bands, bands[1:]):
        assert lower.upper <= upper.lower + 1e-9  # ordered, non-overlapping
    assert sum(b.count for b in bands) == len(scores)  # covering

    # every observed score lands in some band
    for s in scores:
        assert assign_cohort(s, definition) is not None


def test_p1_cohort_assignment_is_deterministic(tokenizer):
    text = "Mujhe ye बहुत पसंद है 😀 and the total was 3.14 units."
    scores = _scores(text, tokenizer)
    d1 = build_cohort_definition(scores, tokenizer_name=BPE_MODEL, corpus_id="p1")
    d2 = build_cohort_definition(scores, tokenizer_name=BPE_MODEL, corpus_id="p1")
    assert [assign_cohort(s, d1) for s in scores] == [assign_cohort(s, d2) for s in scores]


def test_p1_does_not_imply_the_gate1_causal_claim():
    """Guard against the ledger drifting: P1 is an estimator property. If Gate 1
    is ever upgraded past WEAK_PASS that must happen in the Gate 1 report, not
    by quietly reinterpreting these tests."""
    from pathlib import Path

    ledger = Path("CLAIMS_LEDGER.md")
    if not ledger.exists():  # ledger lives at repo root; skip if run elsewhere
        pytest.skip("CLAIMS_LEDGER.md not reachable from this working directory")
    text = ledger.read_text(encoding="utf-8")
    assert "C-11 (Gate 1)" in text
    c11_line = next(line for line in text.splitlines() if line.startswith("| C-11 (Gate 1)"))
    assert "WEAK_PASS" in c11_line
