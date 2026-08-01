# Journal Expansion Plan (4–6 months)

Everything in this document is **[future]** — a plan, not a result. It exists
because the pilot answered its questions honestly and *negatively*, and the
honest response to "the effect was not observed at this scale" is a study
designed to detect it if it exists, not a louder claim.

## Why the pilot could not answer the question

Two blockers, both **[measured]** in the pilot:

1. **The lossless control collapses.** FullKV accuracy falls 48% → 0% between
   `n_g=1` and `n_g=8`. Where the base model already fails, compression damage
   is unmeasurable. The intersection-full-correct subset therefore retained only
   low-fragmentation cohorts.
2. **The allocators never diverged.** Aggregate and minimax selected identical
   allocations in every run and on both tokenizer families. With
   monotone-consistent degradation curves, there is no cohort tension for a
   fairness objective to exploit.

**A bigger model alone does not fix (2).** The expansion must change the
*benchmark* and the *cohort structure*, not just the scale.

## Phase 1 (Weeks 1–4) — Fix the benchmark

- Replace ZWJ-heavy synthetic renderings with **naturalistic high-fragmentation
  text**: real morphologically rich Indic words, transliteration variants, and
  code-mixed spans that a 7B model can actually parse.
- Target: **FullKV accuracy ≥ 60% at every fragmentation level.** This is a
  hard gate — if the control still collapses, the study stops here and reports
  that, rather than proceeding to measure noise.
- Deliverable: IndicLongComp v2 dataset card + FullKV isolation report.

## Phase 2 (Weeks 5–10) — Scale models and contexts

| Axis | Pilot | Journal target |
|---|---|---|
| Model families | 2 (Qwen2 0.5B, TinyLlama 1.1B) | **3**: Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B — distinct tokenizers (BPE / tiktoken-style / SentencePiece) |
| Contexts | <300 tokens | **4K, 8K, 16K, 32K, 64K** |
| Language conditions | 4 | **6+**: en, hi, hinglish, te_en, ta_en, bn, mr |
| Hardware | CPU laptop | **A100 80GB** and **L40S**, with the existing hardware-manifest capture |

Reuse unchanged: `systems/benchmark.py` (p50/p95 + CIs), `systems/hardware.py`,
the FFK1 format, and every frozen gate decision function.

## Phase 3 (Weeks 11–16) — Re-run the frozen gates at scale

Run **the unchanged decision logic** from `gate1.py`, `gate2.py`, `gate3.py`,
`gate4.py`. Thresholds are not to be revised for the journal run; if they were
wrong, that is itself a finding to report.

- Gate 1: ≥1000 minimal-pair groups per model family.
- Gate 2: ≥5 budgets × ≥5 seeds × 3 model families, with **fragility-band
  cohorts** (not just fragmentation levels) so cohort curves can genuinely
  diverge.
- Gate 3: hierarchical bootstrap across families; report all interaction effects.
- Gate 4: re-test fuzzy scoring with a real repair-outcome dataset — the pilot's
  synthetic candidate signals were its main weakness.

**Powered for a 5-point effect** at α=0.05 with paired bootstrap CIs
(pilot was powered only for ≥10 points).

## Phase 4 (Weeks 17–22) — Systems and release

- GPU profiling on A100/L40S: encode/decode/prefill/decode-throughput, p50/p95,
  peak memory, across the full context sweep. This closes the pilot's
  **[future]** D9/D10 gaps.
- **Real serving integration** (vLLM paged-attention hook) so decode figures stop
  being replay-only. Until that lands, the replay labelling stays.
- Public artifact release: dataset + raw predictions + bitstreams + checksums on
  Zenodo with a DOI; camera-ready vector figures.

## Success criteria (pre-committed now)

| Outcome | Publication framing |
|---|---|
| Gate 1 PASS **and** Gate 2 PASS | Positive fairness result — the intended contribution |
| Gate 1 PASS, Gate 2 FAIL | "Fragmentation harms, but allocation cannot fix it" — a real, publishable negative |
| Gate 1 FAIL | "Tokenizer fragmentation does **not** drive disproportionate compression damage at scale" — a valuable null result that closes the question |
| Control still collapses | Report the benchmark-design failure; do not report a fairness result |

**All four outcomes are publishable.** Committing to that now is what makes the
study honest: there is no result the plan needs in order to be worth writing up.

## Risks

| Risk | Mitigation |
|---|---|
| 7B+ models exceed available compute | Phase 2 is the budget gate; fall back to 2 families and report the reduced scope explicitly |
| Naturalistic renderings still collapse the control | Phase 1's hard gate stops the study rather than producing uninterpretable numbers |
| Cohort curves still don't diverge | Report it as a structural finding about KV distortion — it would mean fairness-aware allocation is unnecessary, which is itself the answer |
| Licensing blocks a public dataset release | Content is already 100% project-original synthetic; no third-party license needed |
| Reviewers read the retained name as a claim | `PROJECT_IDENTITY.json` + `CLAIMS_AUDIT.md` state explicitly that the name is not evidence |
