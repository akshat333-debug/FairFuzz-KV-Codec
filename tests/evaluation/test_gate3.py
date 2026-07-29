from fairfuzzkv_codec.evaluation.gate3 import (
    FamilyGateResult, Gate3Decision, decide_gate3, hierarchical_bootstrap,
)


def _family(name, tok, g1, g2, g1_eff=0.1, g2_ben=0.05):
    return FamilyGateResult(
        model_name=name, tokenizer_family=tok, gate1_decision=g1, gate2_decision=g2,
        gate1_effect_size=g1_eff, gate2_worst_cohort_benefit=g2_ben,
    )


def test_pass_when_both_gates_reproduce_category():
    a = _family("A", "bpe", "WEAK_PASS", "FAIL")
    b = _family("B", "sentencepiece", "PASS", "FAIL")
    report = decide_gate3(a, b)
    assert report.decision == Gate3Decision.PASS
    assert report.gate1_reproduces and report.gate2_reproduces


def test_pass_when_both_families_fail_both_gates_identically():
    # reproducing a NEGATIVE finding (both FAIL) is still a PASS on
    # reproducibility - the non-negotiable is about reproducing across
    # models, not about the finding being positive.
    a = _family("A", "bpe", "FAIL", "FAIL")
    b = _family("B", "sentencepiece", "FAIL", "FAIL")
    report = decide_gate3(a, b)
    assert report.decision == Gate3Decision.PASS


def test_weak_pass_when_only_one_gate_reproduces():
    a = _family("A", "bpe", "WEAK_PASS", "FAIL")
    b = _family("B", "sentencepiece", "PASS", "PASS")
    report = decide_gate3(a, b)
    assert report.decision == Gate3Decision.WEAK_PASS
    assert report.gate1_reproduces and not report.gate2_reproduces


def test_fail_when_neither_gate_reproduces():
    a = _family("A", "bpe", "PASS", "PASS")
    b = _family("B", "sentencepiece", "FAIL", "FAIL")
    report = decide_gate3(a, b)
    assert report.decision == Gate3Decision.FAIL
    assert not report.gate1_reproduces and not report.gate2_reproduces


def test_claim_scope_statement_narrows_on_fail():
    a = _family("A", "bpe", "PASS", "PASS")
    b = _family("B", "sentencepiece", "FAIL", "FAIL")
    report = decide_gate3(a, b)
    assert "do NOT report" in report.claim_scope_statement or "family-specific" in report.claim_scope_statement


def test_claim_scope_statement_mentions_cohort_transfer_when_model_specific():
    a = _family("A", "bpe", "PASS", "PASS")
    b = _family("B", "sentencepiece", "PASS", "PASS")
    report = decide_gate3(a, b, cohort_transfer_verdict="model_specific")
    assert "does NOT transfer" in report.claim_scope_statement or "not transfer" in report.claim_scope_statement.lower()
    assert "universal risk threshold" in report.claim_scope_statement


def test_claim_scope_statement_mentions_universal_cohort_transfer():
    a = _family("A", "bpe", "PASS", "PASS")
    b = _family("B", "sentencepiece", "PASS", "PASS")
    report = decide_gate3(a, b, cohort_transfer_verdict="universal")
    assert "transfer" in report.claim_scope_statement.lower()


def test_decision_is_deterministic_pure_function():
    a = _family("A", "bpe", "WEAK_PASS", "FAIL")
    b = _family("B", "sentencepiece", "PASS", "FAIL")
    r1 = decide_gate3(a, b)
    r2 = decide_gate3(a, b)
    assert r1.decision == r2.decision and r1.reasoning == r2.reasoning


def test_families_are_recorded_in_report():
    a = _family("A", "bpe", "WEAK_PASS", "FAIL")
    b = _family("B", "sentencepiece", "PASS", "FAIL")
    report = decide_gate3(a, b)
    assert len(report.families) == 2
    d = report.to_dict()
    assert d["families"][0]["model_name"] == "A"


# ---- hierarchical bootstrap -------------------------------------------------

def test_hierarchical_bootstrap_point_estimate_is_unweighted_family_mean():
    family_examples = {"A": [1.0, 1.0, 1.0, 1.0], "B": [0.0, 0.0]}
    point, lo, hi = hierarchical_bootstrap(family_examples, n_boot=200, seed=1)
    # unweighted mean of family means: (1.0 + 0.0) / 2 = 0.5, NOT the pooled
    # example mean (which would be (4*1+2*0)/6 = 0.667) - a larger family
    # must not dominate.
    assert abs(point - 0.5) < 1e-9
    assert lo <= point <= hi


def test_hierarchical_bootstrap_ci_widens_with_family_disagreement():
    agree = {"A": [1.0] * 10, "B": [1.0] * 10}
    disagree = {"A": [1.0] * 10, "B": [0.0] * 10}
    _, lo_a, hi_a = hierarchical_bootstrap(agree, n_boot=500, seed=2)
    _, lo_d, hi_d = hierarchical_bootstrap(disagree, n_boot=500, seed=2)
    assert (hi_d - lo_d) > (hi_a - lo_a)


def test_hierarchical_bootstrap_empty_input_is_safe():
    point, lo, hi = hierarchical_bootstrap({})
    assert point == 0.0 and lo == 0.0 and hi == 0.0


def test_hierarchical_bootstrap_deterministic_given_seed():
    family_examples = {"A": [0.2, 0.4, 0.6], "B": [0.1, 0.3, 0.9]}
    r1 = hierarchical_bootstrap(family_examples, n_boot=300, seed=7)
    r2 = hierarchical_bootstrap(family_examples, n_boot=300, seed=7)
    assert r1 == r2
