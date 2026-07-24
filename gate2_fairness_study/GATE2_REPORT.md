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
- 16 FragKV-MinPairs groups, seed 42, split 8 calibrate / 8 evaluate.
- Quantizer menu per cohort: `INT4`, `INT8` (`UniformQuantCodec`, matched-bit
  compatible). Per-cohort budget 6 bits/element (`budget = 24` over 4 cohorts).
- Isolation: intersection-full-correct subset (keep only examples FullKV got
  right), so compression effects are separated from base-model inequality.

## Result

| Quantity | Value |
|---|---|
| Aggregate allocation | `{1:int8, 2:int8, 4:int4, 8:int4}` |
| Minimax allocation | `{1:int8, 2:int8, 4:int4, 8:int4}` — **identical** |
| Matched bits/element (agg / mm) | 6.01 / 6.01 (ok) |
| Isolated examples | 12 |
| Cohorts with any full-correct examples | only `n_g=1` (5) and `n_g=2` (7) |
| Worst-cohort fairness benefit | **0.000**, 95% CI [0.000, 0.000] |

## Why FAIL

Two independent reasons, both honest:

1. **No allocation divergence.** At the frozen budget the aggregate and minimax
   solvers chose the *same* per-cohort bit-widths, so there is *nothing* for a
   fairness benefit to come from — minimax cannot beat a control it is identical
   to. The per-cohort degradation curves on this data did not create the tension
   (one clearly worst cohort that minimax would protect at the aggregate's
   expense) that the fairness thesis requires.

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

- **PILOT SCALE, UNDERPOWERED.** 12 isolated examples over 2 usable cohorts. The
  bootstrap CI is degenerate `[0,0]` precisely because the two systems are
  identical, not because disparity was measured and found tiny.
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
