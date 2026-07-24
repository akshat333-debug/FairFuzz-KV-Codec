# Gate 2 Experiment Configuration (FROZEN)

Pre-registered configuration for the Gate-2 comparison between the **aggregate**
rate-distortion allocator (Prompt 10, the control) and the **fairness-constrained
minimax** allocator (Prompt 11). Frozen **before** running the final comparison so
the objective and setup cannot be tuned to the result (mirrors the Gate-1
pre-registration discipline).

## Frozen decisions

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen2.5-0.5B` (real pretrained weights) |
| Cohorts | one per transformer layer (`L{layer}`) of the captured K cache |
| Quantizer menu | scalar `int4`, `int8` + LBG `(vector_dim=8, codebook_size∈{16,64})` |
| Distortion | test-split MSE, calibration fit on the **train** split only (leakage-safe) |
| Cost | full serialized bits per cohort (scale/zero-point/codebook overhead included) |
| Objective (default, FROZEN) | minimize **max-cohort** distortion (unweighted) |
| Control | aggregate allocator minimizing **sum** of distortion (Prompt 10) |
| Budget sweep | midpoint of `[all-cheapest, all-dearest]`, plus a Pareto sweep across that range |
| Solvers | exact epigraph reference vs continuous water-filling + discrete projection |
| Seed | 42 (calibration split is deterministic) |

## Frozen success criteria

* The minimax allocation's **worst-cohort** distortion is **≤** the aggregate
  allocation's worst-cohort distortion at the same budget (fairness improves the
  worst case, by construction — verified, not assumed).
* The production water-filling solver matches the exact reference's worst-case
  value within tolerance on the small per-layer instance.
* No allocation exceeds the encoded-bit budget after real serialization.
* The Pareto frontier (worst vs average distortion across budgets) is reported in
  full, **exposing** the cost of fairness (higher average distortion is the price
  of a lower worst case) rather than hiding it.

## Non-claims

The minimax allocator is the **engineering** control/experimental condition for
Gate 2. It does **not** rely on the Gate-1 causal claim (which is only
`WEAK_PASS`; see `RISK_REGISTER` R-06). "Fairness across cohorts" here means
equalizing achieved **distortion** across active cohorts (the KKT result in
`ALLOCATION_MATH.md`) — it is **not** a claim about downstream task fairness.
