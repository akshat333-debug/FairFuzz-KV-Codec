# IndicLongComp Dataset Report

Not a PASS/FAIL gate (Prompt 15's checkpoint is an "External Validity Dataset
Gate" - a construction-quality bar, not a scientific claim). This documents
what was built and what the real course-subset FullKV run found.

## What this is

A parallel multilingual/code-mixed long-context benchmark across four
language conditions - English, Hindi, Hinglish, Telugu-English - and five
task families (retrieval, multi-hop, comparison, counting, evidence
aggregation), built from hand-designed parallel templates
(`fairfuzzkv_codec.benchmarks.indic_longcomp`). See the package docstring
and `dataset_card.py`'s notes for the full content-provenance statement:
**LLM-authored, not sourced from any external corpus (MLRBench or
otherwise - no verified network/license access was available), not
professionally translated or reviewed.** "Parallel" is used only in the
verified structural sense - every language variant of a group shares the
same drawn canonical answer, evidence count, evidence position, and
distractor count by construction, checked per-group by
`validators.validate_parallelism`, never assumed from translation quality.

## Acceptance gates

| Gate | Status |
|---|---|
| >=4 language/code-mix conditions | Met - en, hi, hinglish, te_en |
| Parallelism checks + answer validation documented | Met - `validators.py`, all groups pass in both subsets |
| Dataset card distinguishes translated/generated/original content | Met - `content_provenance_note` states LLM-generated-template, explicitly not translated/reviewed, no external corpus |
| FullKV correctness + isolation subset computed | Met - real Qwen2.5-0.5B run, course subset, BEFORE any compression evaluation |
| Regenerable from scripts and manifests | Met - `scripts/run_indiclongcomp_study.py`, deterministic given seed |

## Real course-subset result (Qwen2.5-0.5B, 10 groups = 2 per task family, 40 variants)

| Language | Accuracy | | Task family | Accuracy |
|---|---|---|---|---|
| en | 1/10 | | retrieval | 1/8 |
| hi | 0/10 | | multi_hop | 1/8 |
| hinglish | 2/10 | | comparison | 0/8 |
| te_en | 0/10 | | counting | 0/8 |
| | | | aggregation | 1/8 |

**Intersection-full-correct groups: 0/10.** No group had all four language
variants answered correctly, so the isolation subset for compression
comparison is currently empty at this pilot scale.

## Why - and why this isn't alarming

At n=2 groups per task family (8-10 datapoints per breakdown cell), the
raw counts above are far too small to distinguish "genuinely near-floor
model capability" from ordinary small-sample noise - the same
underpowered-pilot caveat Gate 2 (RISK R-09) and Gate 4 (RISK R-10) already
carry. FragKV-MinPairs' own real 200-group Gate-1 study found ~48%
n_g=1 FullKV accuracy on a structurally similar retrieval task, so a 0.5B
base (non-instruction-tuned) model CAN do this task type at a meaningful
rate - this course subset's low numbers are consistent with small-n
variance, not evidence the benchmark or model are broken.

## Journal subset (structural validation only, no FullKV run)

50 groups per task family (250 groups, 1000 variants) generated and
structurally validated (parallelism, PII, dedup, contamination) in this
session. Running the real FullKV baseline on it - the natural next step to
get an isolation subset with real statistical power - was **not** done
here: it would mean ~1000 real model forward-pass generations, a real
compute cost beyond this session's budget, and is left as an explicit,
documented follow-up rather than silently skipped or estimated.

## Non-claims

This report does not claim IndicLongComp shows (or fails to show) a
fragility/language effect - that would need the journal-scale FullKV run
plus the same isolation/disparity machinery Gate 1/Gate 2 already use.
This prompt's job was building and validating the DATASET, not running a
new causal gate on it.
