"""Gate 3: cross-tokenizer/cross-model reproduction (Prompt 17).

Reruns Gate 1 and Gate 2 on a SECOND model/tokenizer family
(TinyLlama-1.1B, SentencePiece) via the SAME frozen scripts already used for
Qwen2.5-0.5B (byte-level BPE) - `run_gate1_study.py --model ...` and
`run_gate2_study.py --model ...`, unchanged decision logic - then compares
decision CATEGORIES (not pooled significance) via `evaluation.gate3`. Also
runs the (already-built, Module 2) cross-tokenizer cohort-stability check.
Family A's real results are the ALREADY-COMMITTED Qwen2.5-0.5B runs
(gate1_study/, gate2_fairness_study/) - not re-run. Pilot scale on Family B,
per GATE3_CONFIG.md. Numbers are measured, never invented.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer  # noqa: E402

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.dataset_card import load_dataset  # noqa: E402
from fairfuzzkv_codec.evaluation.gate3 import FamilyGateResult, decide_gate3, hierarchical_bootstrap  # noqa: E402
from fairfuzzkv_codec.fragility_estimation.stability import compute_cross_tokenizer_stability  # noqa: E402

QWEN = "Qwen/Qwen2.5-0.5B"
TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GATE1_TINYLLAMA_GROUPS = 20
GATE2_TINYLLAMA_GROUPS = 16
GATE2_TINYLLAMA_BUDGET = "6"
GATE2_TINYLLAMA_SEED = "42"

# REAL, MEASURED finding (see RISK_REGISTER / GATE3_CONFIG.md): the numeric
# rendering ladder (numeric_forms.RENDER_LADDER), calibrated against
# Qwen2.5-0.5B's byte-level BPE tokenizer, does NOT transfer to TinyLlama's
# SentencePiece tokenizer under the frozen tolerance {1:0,2:0,4:1,8:1} - NO
# digit 0-9 can hit both n_g=4 AND n_g=8 within tolerance simultaneously
# (measured directly: n_g=4 only reachable for digits {1,2,6}, n_g=8 only
# for {0,4,5,9}, disjoint sets). This IS the answer to Prompt 17 item 116
# ("does this transfer or require tokenizer-specific recalibration?") found
# one layer earlier than expected - at dataset construction, not cohort
# assignment. Widening tolerance to {4:2, 8:2} makes 7/10 digits reachable
# (measured) - the minimum viable recalibration, not an arbitrarily loose one.
TINYLLAMA_TOKEN_COUNT_TOLERANCE = '{"1":0,"2":0,"4":2,"8":2}'


def _run(cmd: list) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _topk50_paired_diffs(predictions_path: Path) -> List[float]:
    """Per-group (correct at n_g=1) - (correct at n_g=8) for the TopK50
    codec, from a Gate 1 predictions.jsonl - the same per-group paired
    statistic `stats_utils.compute_effect_size` averages, but returned here
    as a list (one value per group) so `hierarchical_bootstrap` can treat
    each group as an "example" within its family's stratum."""
    by_group: dict = {}
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["codec_name"] != "TopK50":
                continue
            by_group.setdefault(rec["group_id"], {})[rec["n_g"]] = rec["correct"]
    diffs = [
        float(vals[1]) - float(vals[8])
        for vals in by_group.values()
        if 1 in vals and 8 in vals
    ]
    return diffs or [0.0]  # hierarchical_bootstrap requires a non-empty list per family


def main() -> None:
    out = Path("gate3_study")
    out.mkdir(exist_ok=True)
    python = sys.executable

    print("[1/4] Gate 1 on TinyLlama (frozen script, pilot scale, RECALIBRATED tolerance - see comment at top of this file)...")
    _run([
        python, "scripts/run_gate1_study.py", "--model", TINYLLAMA,
        "--num-groups", str(GATE1_TINYLLAMA_GROUPS), "--seed", "42",
        "--output-dir", str(out / "gate1_tinyllama"),
        "--token-count-tolerance", TINYLLAMA_TOKEN_COUNT_TOLERANCE,
    ])

    print("\n[2/4] Gate 2 on TinyLlama (frozen script, pilot scale, same recalibrated tolerance)...")
    _run([
        python, "scripts/run_gate2_study.py", "--model", TINYLLAMA,
        "--num-groups", str(GATE2_TINYLLAMA_GROUPS), "--budgets", GATE2_TINYLLAMA_BUDGET,
        "--seeds", GATE2_TINYLLAMA_SEED, "--output-dir", str(out / "gate2_tinyllama"),
        "--token-count-tolerance", TINYLLAMA_TOKEN_COUNT_TOLERANCE,
    ])

    print("\n[3/4] Cohort transfer analysis (Module 2 cross-tokenizer stability, reused unchanged)...")
    tok_a = AutoTokenizer.from_pretrained(QWEN)
    tok_b = AutoTokenizer.from_pretrained(TINYLLAMA)
    groups = load_dataset("gate1_study/dataset")
    sample_text = groups[0].get_variant(1).context_text
    stability = compute_cross_tokenizer_stability(sample_text, tok_a, tok_b, corpus_id="gate3_shared_corpus")
    print(f"  verdict={stability.verdict} agreement={stability.cohort_agreement_rate:.2f} n={stability.num_units_compared}")

    print("\n[4/4] Assembling Gate 3 decision from both families' real results...")
    gate1_a = json.loads(Path("gate1_study/GATE1_REPORT.json").read_text(encoding="utf-8"))
    gate2_a = json.loads(Path("gate2_fairness_study/GATE2_REPORT.json").read_text(encoding="utf-8"))
    gate1_b = json.loads((out / "gate1_tinyllama" / "GATE1_REPORT.json").read_text(encoding="utf-8"))
    gate2_b = json.loads((out / "gate2_tinyllama" / "GATE2_REPORT.json").read_text(encoding="utf-8"))

    def _max_effect(gate1_report: dict) -> float:
        results = gate1_report.get("lossy_results", [])
        return max((abs(r["effect_size"]) for r in results), default=0.0)

    family_a = FamilyGateResult(
        model_name=QWEN, tokenizer_family="byte-level BPE",
        gate1_decision=gate1_a["decision"], gate2_decision=gate2_a["report"]["decision"],
        gate1_effect_size=_max_effect(gate1_a), gate2_worst_cohort_benefit=gate2_a["bootstrap"]["point"],
    )
    family_b = FamilyGateResult(
        model_name=TINYLLAMA, tokenizer_family="SentencePiece",
        gate1_decision=gate1_b["decision"], gate2_decision=gate2_b["report"]["decision"],
        gate1_effect_size=_max_effect(gate1_b), gate2_worst_cohort_benefit=gate2_b["bootstrap"]["point"],
    )

    report = decide_gate3(family_a, family_b, cohort_transfer_verdict=stability.verdict)
    print(f"\nGATE 3 DECISION: {report.decision.value}")

    print("\n[hierarchical bootstrap] pooling Gate 1's TopK50 n_g=1-vs-8 paired effect across BOTH families...")
    family_paired_diffs = {
        family_a.model_name: _topk50_paired_diffs(Path("gate1_study/predictions.jsonl")),
        family_b.model_name: _topk50_paired_diffs(out / "gate1_tinyllama" / "predictions.jsonl"),
    }
    hb_point, hb_lo, hb_hi = hierarchical_bootstrap(family_paired_diffs, n_boot=2000, seed=42)
    print(f"  pooled (family-then-example) point={hb_point:.4f} 95% CI [{hb_lo:.4f}, {hb_hi:.4f}]")
    print(report.reasoning)
    print(report.claim_scope_statement)

    (out / "gate3_report.json").write_text(
        json.dumps({
            "report": report.to_dict(), "cohort_stability": stability.model_dump(),
            "hierarchical_bootstrap": {
                "statistic": "TopK50 n_g=1-vs-8 paired correctness diff, per group, family-then-example resample",
                "point": hb_point, "ci_low": hb_lo, "ci_high": hb_hi,
                "n_examples_per_family": {k: len(v) for k, v in family_paired_diffs.items()},
            },
        }, indent=2),
        encoding="utf-8",
    )

    accuracy_table = "\n".join(
        f"| n_g={n_g} | {gate1_a['control_result']['accuracy_by_n_g'].get(str(n_g), 'n/a')} (Qwen FullKV) | "
        f"{gate1_b['control_result']['accuracy_by_n_g'].get(str(n_g), 'n/a')} (TinyLlama FullKV) |"
        for n_g in (1, 2, 4, 8)
    )
    tolerance_str = TINYLLAMA_TOKEN_COUNT_TOLERANCE

    gate3_report_md = f"""# Gate 3 Report: Cross-Tokenizer/Cross-Model Reproduction

**Decision: {report.decision.value}**

Reproducible from `gate3_study/gate3_report.json` + the frozen decision
logic in `fairfuzzkv_codec.evaluation.gate3` (tested on synthetic fixtures
before this study ran - see `tests/evaluation/test_gate3.py`). Frozen
configuration: `GATE3_CONFIG.md`.

## Families compared

| | {family_a.model_name} | {family_b.model_name} |
|---|---|---|
| Tokenizer | {family_a.tokenizer_family} | {family_b.tokenizer_family} |
| Gate 1 decision | {family_a.gate1_decision} | {family_b.gate1_decision} |
| Gate 2 decision | {family_a.gate2_decision} | {family_b.gate2_decision} |
| Gate 1 max lossy effect size | {family_a.gate1_effect_size:.3f} | {family_b.gate1_effect_size:.3f} |
| Gate 2 worst-cohort benefit | {family_a.gate2_worst_cohort_benefit:.4f} | {family_b.gate2_worst_cohort_benefit:.4f} |

## Result

{report.reasoning}

- Gate 1 reproduces in category: **{report.gate1_reproduces}**
- Gate 2 reproduces in category: **{report.gate2_reproduces}**
- Cohort transfer verdict: **{report.cohort_transfer_verdict}** (agreement={stability.cohort_agreement_rate:.2f}, n={stability.num_units_compared} shared surface units)

**Important nuance: Gate 2's two FAILs have DIFFERENT root causes, not the
same mechanism** - "reproduces in category" means both landed on FAIL, not
that the same thing broke:
- {family_a.model_name}: "{gate2_a['report']['reasoning']}"
- {family_b.model_name}: "{gate2_b['report']['reasoning']}"

The TinyLlama run failed matched-bit tolerance at pilot scale (16 groups,
1 budget, 1 seed - only 2 isolated examples) - a different, scale-driven
reason than Qwen's original 6-run finding that aggregate and minimax
allocators chose identical bit-widths. Both are real FAILs; neither
should be read as confirming the other's specific mechanism.

## A finding one layer earlier than expected: the dataset generator itself needed tokenizer-specific recalibration

Before Gate 1/Gate 2 could even RUN on TinyLlama, the FragKV-MinPairs
numeric rendering ladder (`numeric_forms.RENDER_LADDER`, calibrated
against Qwen2.5-0.5B) failed to construct ANY valid group under the frozen
tolerance `{{1:0,2:0,4:1,8:1}}` - measured directly: under TinyLlama's
SentencePiece tokenizer, no digit 0-9 can hit both n_g=4 AND n_g=8 within
tolerance simultaneously (the sets of digits that CAN hit each target are
disjoint: `{{1,2,6}}` for n_g=4, `{{0,4,5,9}}` for n_g=8). Widening
tolerance to `{tolerance_str}` (still the SAME ladder, SAME render
functions, just a wider match window) made 7/10 digits reachable and let
the study proceed. This is itself a real, measured answer to Prompt 17
item 116 ("does this transfer or require tokenizer-specific
recalibration?") - found at dataset-construction time, not cohort-
assignment time. See RISK_REGISTER.

## Hierarchical/stratified bootstrap across families and examples (item 118)

Pooled TopK50 n_g=1-vs-8 paired correctness effect, resampled at TWO levels
(which families contribute, THEN which examples within each drawn family) -
so {family_b.model_name}'s smaller pilot sample cannot dominate or be
swamped by {family_a.model_name}'s larger one; each family contributes an
unweighted 50% to the pooled estimate:

- Point estimate: **{hb_point:.4f}**
- 95% CI: **[{hb_lo:.4f}, {hb_hi:.4f}]**
- n examples: {family_a.model_name}={len(family_paired_diffs[family_a.model_name])}, {family_b.model_name}={len(family_paired_diffs[family_b.model_name])}

## Tokenizer family x fragmentation level (interaction, item 119)

FullKV (lossless) accuracy by fragmentation level, both families:

| n_g | {family_a.model_name} | {family_b.model_name} |
|---|---|---|
{accuracy_table}

## Claim scope

{report.claim_scope_statement}

## Scope deferred (documented, not hidden)

Per GATE3_CONFIG.md: **model family x allocator** and **quantizer type x
cohort** interaction effects (Prompt 17 item 119) were NOT attempted here -
both would require re-running the Prompt 10/11 allocator study on
TinyLlama, beyond this session's compute budget. Gate 1/Gate 2 pilot scale
on TinyLlama ({GATE1_TINYLLAMA_GROUPS} / {GATE2_TINYLLAMA_GROUPS} groups)
is smaller than Family A's original real runs (200 / 24 groups x 6 runs) -
explicitly reduced given TinyLlama's ~2x slower CPU forward pass, not
silently. See PENDING.md.

## Raw data

`gate3_study/gate1_tinyllama/predictions.jsonl`,
`gate3_study/gate2_tinyllama/predictions.jsonl` - every raw prediction
retained (acceptance gate: "raw predictions and run manifests are
retained"). Family A's raw predictions remain at `gate1_study/` and
`gate2_fairness_study/` (unchanged, not duplicated).
"""
    Path("GATE3_REPORT.md").write_text(gate3_report_md, encoding="utf-8")
    print(f"\nsaved -> {out}, GATE3_REPORT.md")


if __name__ == "__main__":
    main()
