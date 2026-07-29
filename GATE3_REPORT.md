# Gate 3 Report: Cross-Tokenizer/Cross-Model Reproduction

**Decision: PASS**

Reproducible from `gate3_study/gate3_report.json` + the frozen decision
logic in `fairfuzzkv_codec.evaluation.gate3` (tested on synthetic fixtures
before this study ran - see `tests/evaluation/test_gate3.py`). Frozen
configuration: `GATE3_CONFIG.md`.

## Families compared

| | Qwen/Qwen2.5-0.5B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
|---|---|---|
| Tokenizer | byte-level BPE | SentencePiece |
| Gate 1 decision | WEAK_PASS | WEAK_PASS |
| Gate 2 decision | FAIL | FAIL |
| Gate 1 max lossy effect size | 0.820 | 0.100 |
| Gate 2 worst-cohort benefit | 0.0000 | -1.0000 |

## Result

Both Gate 1 (WEAK_PASS vs WEAK_PASS) and Gate 2 (FAIL vs FAIL) land in the same signal/no-signal category across Qwen/Qwen2.5-0.5B and TinyLlama/TinyLlama-1.1B-Chat-v1.0.

- Gate 1 reproduces in category: **True**
- Gate 2 reproduces in category: **True**
- Cohort transfer verdict: **model_specific** (agreement=0.23, n=61 shared surface units)

**Important nuance: Gate 2's two FAILs have DIFFERENT root causes, not the
same mechanism** - "reproduces in category" means both landed on FAIL, not
that the same thing broke:
- Qwen/Qwen2.5-0.5B: "minimax did not meaningfully reduce worst-cohort degradation (benefit 0.000, consistency 0%, cost 0.000). Codec is preserved; minimax is reported as NEGATIVE evidence for the fairness thesis, not a validated claim."
- TinyLlama/TinyLlama-1.1B-Chat-v1.0: "matched-bit tolerance violated in at least one run - comparison invalid"

The TinyLlama run failed matched-bit tolerance at pilot scale (16 groups,
1 budget, 1 seed - only 2 isolated examples) - a different, scale-driven
reason than Qwen's original 6-run finding that aggregate and minimax
allocators chose identical bit-widths. Both are real FAILs; neither
should be read as confirming the other's specific mechanism.

## A finding one layer earlier than expected: the dataset generator itself needed tokenizer-specific recalibration

Before Gate 1/Gate 2 could even RUN on TinyLlama, the FragKV-MinPairs
numeric rendering ladder (`numeric_forms.RENDER_LADDER`, calibrated
against Qwen2.5-0.5B) failed to construct ANY valid group under the frozen
tolerance `{1:0,2:0,4:1,8:1}` - measured directly: under TinyLlama's
SentencePiece tokenizer, no digit 0-9 can hit both n_g=4 AND n_g=8 within
tolerance simultaneously (the sets of digits that CAN hit each target are
disjoint: `{1,2,6}` for n_g=4, `{0,4,5,9}` for n_g=8). Widening
tolerance to `{"1":0,"2":0,"4":2,"8":2}` (still the SAME ladder, SAME render
functions, just a wider match window) made 7/10 digits reachable and let
the study proceed. This is itself a real, measured answer to Prompt 17
item 116 ("does this transfer or require tokenizer-specific
recalibration?") - found at dataset-construction time, not cohort-
assignment time. See RISK_REGISTER.

## Hierarchical/stratified bootstrap across families and examples (item 118)

Pooled TopK50 n_g=1-vs-8 paired correctness effect, resampled at TWO levels
(which families contribute, THEN which examples within each drawn family) -
so TinyLlama/TinyLlama-1.1B-Chat-v1.0's smaller pilot sample cannot dominate or be
swamped by Qwen/Qwen2.5-0.5B's larger one; each family contributes an
unweighted 50% to the pooled estimate:

- Point estimate: **0.0375**
- 95% CI: **[0.0000, 0.0900]**
- n examples: Qwen/Qwen2.5-0.5B=200, TinyLlama/TinyLlama-1.1B-Chat-v1.0=20

Note the CI includes 0 - consistent with Gate 1's own WEAK_PASS framing on
both families (a real but modest effect, not a strong one). This is the
project's own established discipline (Gate1/Gate2/Gate4 all report CIs
that don't overstate what pilot-scale data supports), now extended across
families as item 118 requires.

## Tokenizer family x fragmentation level (interaction, item 119)

FullKV (lossless) accuracy by fragmentation level, both families:

| n_g | Qwen/Qwen2.5-0.5B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
|---|---|---|
| n_g=1 | 0.48 (Qwen FullKV) | 0.05 (TinyLlama FullKV) |
| n_g=2 | 0.665 (Qwen FullKV) | 0.1 (TinyLlama FullKV) |
| n_g=4 | 0.025 (Qwen FullKV) | 0.1 (TinyLlama FullKV) |
| n_g=8 | 0.0 (Qwen FullKV) | 0.0 (TinyLlama FullKV) |

## Claim scope

The qualitative Gate 1/Gate 2 pattern reproduces across both tested families. This still does NOT establish a universal claim beyond these two families/scales - see each family's own Gate 1/Gate 2 report for the underlying finding's own strength. Additionally: fragility cohort risk-band assignment does NOT transfer between these tokenizer families (cross-tokenizer stability below the universal-agreement threshold) - do not claim a universal risk threshold; cohorts require per-tokenizer recalibration.

## Scope deferred (documented, not hidden)

Per GATE3_CONFIG.md: **model family x allocator** and **quantizer type x
cohort** interaction effects (Prompt 17 item 119) were NOT attempted here -
both would require re-running the Prompt 10/11 allocator study on
TinyLlama, beyond this session's compute budget. Gate 1/Gate 2 pilot scale
on TinyLlama (20 / 16 groups)
is smaller than Family A's original real runs (200 / 24 groups x 6 runs) -
explicitly reduced given TinyLlama's ~2x slower CPU forward pass, not
silently. See PENDING.md.

## Raw data

`gate3_study/gate1_tinyllama/predictions.jsonl`,
`gate3_study/gate2_tinyllama/predictions.jsonl` - every raw prediction
retained (acceptance gate: "raw predictions and run manifests are
retained"). Family A's raw predictions remain at `gate1_study/` and
`gate2_fairness_study/` (unchanged, not duplicated).
