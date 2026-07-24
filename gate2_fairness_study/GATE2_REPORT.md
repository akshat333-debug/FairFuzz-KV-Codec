# Gate 2 Report: Matched-Bit Fairness Experiment

**Decision: FAIL** (pilot scale — provisional negative evidence, not a final verdict)

Reproducible from `predictions.jsonl` + the frozen decision logic in
`fairfuzzkv_codec.evaluation.gate2` (pre-registered thresholds, committed and
unit-tested on synthetic fixtures **before** this study ran — see
`tests/evaluation/test_gate2.py`). Frozen configuration: `GATE2_CONFIG.md`.

## Question

Does the **minimax** allocator (minimize worst-cohort degradation) meaningfully
reduce cross-cohort compression-degradation **disparity** versus the
**aggregate** allocator (minimize total degradation), at **matched bits**?

## Setup (measured, Qwen2.5-0.5B)

- Cohorts = evidence-fragmentation levels `n_g ∈ {1,2,4,8}`.
- 16 FragKV-MinPairs groups per seed, split 8 calibrate / 8 evaluate.
- Quantizer menu per cohort: `INT4`, `INT8` (`UniformQuantCodec`, matched-bit
  compatible).
- **Multiple budgets and seeds (item 81):** per-cohort budgets `{5,6,7}`
  bits/element × seeds `{42,7}` = **6 runs**.
- Isolation: intersection-full-correct subset (keep only examples FullKV got
  right), so compression effects are separated from base-model inequality.

## Result (6 runs, 2 seeds × 3 budgets)

In **every** run the aggregate and minimax allocators chose **identical**
per-cohort bit-widths, so the worst-cohort fairness benefit was exactly 0 in all
6 runs:

| Run (seed, budget) | Aggregate | Minimax | Benefit |
|---|---|---|---|
| 42, 20 | `{1:int4,2:int8,4:int4,8:int4}` | identical | 0.000 |
| 42, 24 | `{1:int8,2:int8,4:int4,8:int4}` | identical | 0.000 |
| 42, 28 | `{1:int8,2:int8,4:int4,8:int4}` | identical | 0.000 |
| 7, 20 | `{1:int4,2:int8,4:int4,8:int4}` | identical | 0.000 |
| 7, 24 | `{1:int8,2:int8,4:int4,8:int4}` | identical | 0.000 |
| 7, 28 | `{1:int8,2:int8,4:int4,8:int4}` | identical | 0.000 |

| Pooled quantity | Value |
|---|---|
| Matched bits/element (agg / mm) | equal every run (e.g. 6.01 / 6.01) |
| Isolated examples (pooled) | 72 |
| Cohort counts (pooled) | `n_g=1`: 27, `n_g=2`: 39, `n_g=4`: 6, `n_g=8`: 0 |
| Worst-cohort fairness benefit | **0.000**, 95% CI [0.000, 0.000] |
| Directional consistency | 0% |

## Why FAIL

Two independent reasons, both honest:

1. **No allocation divergence — robustly, across 6 runs.** At every budget and
   seed the aggregate and minimax solvers chose the *same* per-cohort bit-widths,
   so there is *nothing* for a fairness benefit to come from — minimax cannot
   beat a control it is identical to. The per-cohort degradation curves on this
   data are monotone-consistent (INT8 never worse than INT4), so both a
   sum-objective and a max-objective pour the marginal bit into the same cohort;
   they only diverge when one cohort is *disproportionately* fragile, which does
   not occur here. This held across seeds {42,7} and budgets {5,6,7}/cohort.

2. **The isolation subset collapses to low-fragmentation cohorts.** `n_g=4` and
   `n_g=8` contributed *zero* full-correct examples, so the disparity question
   cannot even be posed across the intended fragmentation range. This is the
   **same base-model confound Gate 1 flagged** (RISK_REGISTER R-06): the base
   model already fails the high-fragmentation renderings, so there is no
   base-correct signal left to measure compression fairness on.

Per the Prompt 12 non-negotiable: the codec is preserved and minimax is reported
as **negative evidence** for the Q1 fairness thesis at this scale. No fairness
claim is fabricated.

## Power / caveats

- **PILOT SCALE, UNDERPOWERED.** 72 pooled isolated examples, but concentrated in
  low-fragmentation cohorts (`n_g=1,2`); `n_g=8` contributes 0. The bootstrap CI
  is degenerate `[0,0]` precisely because the two systems are identical in every
  run, not because disparity was measured and found tiny.
- **Cohort-threshold sensitivity (item 83):** N/A here — cohorts are the
  categorical fragmentation levels `n_g`, which have no tunable threshold. The
  threshold-sensitivity analysis applies to *fragility-band* cohorts (Module 2),
  which are not the cohort axis used in this Gate-2 run; wiring minimax to those
  bands (and sweeping their quantile thresholds) is the natural follow-up.
- This does **not** prove minimax can never help — it shows that on this small
  synthetic minimal-pair set, at this budget, (a) the allocators don't diverge
  and (b) the isolation protocol is starved by the same high-fragmentation
  base-model collapse as Gate 1.

## Next steps (to give the thesis a fair test)

1. Naturalistic (non-ZWJ-heavy) high-fragmentation renderings and/or a larger
   model, so `n_g=4,8` retain full-correct examples after isolation (shared with
   Gate 1's own next-steps).
2. A larger, more heterogeneous cohort set whose degradation curves actually
   diverge, so aggregate and minimax produce *different* allocations.
3. Scale groups/budgets/seeds up once (1) and (2) hold, then re-run this exact,
   unchanged decision logic.
