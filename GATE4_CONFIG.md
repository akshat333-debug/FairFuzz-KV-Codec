# Gate 4 Experiment Configuration (FROZEN)

Pre-registered configuration for the Gate-4 comparison between the Module 3
**fuzzy** repair-priority scorer, its three **simple** competitors (monotone,
knapsack, logistic), and **no-repair** (initial pruning only). Frozen
**before** running the final comparison, mirroring the Gate-1/Gate-2
pre-registration discipline. The decision logic in `evaluation/gate4.py` is
committed and tested against synthetic fixtures (`tests/evaluation/test_gate4.py`)
before this config is ever pointed at a real study.

## Frozen decisions

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen2.5-0.5B` (real pretrained weights) |
| Dataset | fresh FragKV-MinPairs groups (`generate_validated_dataset`), 5 groups x 2 seeds = 10 pooled groups x 4 `n_g` variants each = 40 pooled variant-examples per (budget, system) - smaller pilot than Gate 2's 16/seed, kept tractable given Gate 4 multiplies by budget x system count; reported as pilot-scale, not a final-power study, matching the project's established honesty convention |
| Seeds | 42, 7 |
| Cohorts | evidence-fragmentation levels `n_g in {1,2,4,8}` (same axis as Gate 1/Gate 2) |
| Systems compared | `no_repair`, `fuzzy`, `monotone`, `knapsack`, `logistic` |
| Initial pruning | `TopAttentionMassSelector` ranked by the REAL captured attention row of the audited (layer, head) below |
| Budgets | initial retention ratio in `{0.3, 0.5}` (fraction of context positions kept before repair) |
| Repair swap size `n` | `floor(0.5 * num_evicted)` per system - half the evicted budget is repair-eligible, fixed regardless of scorer |
| Audited (layer, head) for repair scoring | layer 0, head 0 - head index 0 always maps to KV-head 0 under any GQA grouping ratio, so this stays correct regardless of the model's `num_key_value_heads` |
| Repair contract `delta` | 0.02 (the project's standing default, e.g. Prompt 9 examples) |
| Candidate inputs (Module 3) | fragility = real transparent monotone score (Module 2) broadcast from surface group to token position; evidence_importance = real captured attention mass on that position from the audited head; completion_cost = the position's surface group's token count (Module 1); staleness = `1 - position/context_length` (how early/far-back the position is) |
| Input normalization | train-only stats fit on the FIRST group's candidates per (seed, budget) run, applied unchanged to every group in that run |
| Matched bits | automatic by construction - repair swaps are budget-neutral (Prompt 9), so every system that accepts its proposed swap ends with the SAME retained-position count as `no_repair`; verified per run, not assumed |

## Metrics (per system, per run)

- **Task distortion**: real-model generation accuracy on the matched-bit reconstructed cache (splice into `DynamicCache`, greedy-decode, parse, compare to canonical value) - reused, unmodified `FragKVRunner`-style pipeline.
- **Numerical distortion**: K/V reconstruction MSE.
- **Worst-cohort distortion / disparity**: intersection-full-correct isolation + CDDB/worst-group-drop, reusing `evaluation.isolation` / `evaluation.disparity` (Prompt 12 machinery) with `n_g` as the cohort axis.
- **Repair overhead**: accepted-swap count / attempted-swap count, per system (0/0 for `no_repair`).
- **Inference latency**: measured wall-clock scorer latency per candidate (`repair_scoring.sensitivity.measure_complexity`), real, not estimated.

## Frozen decision thresholds

See `evaluation/gate4.py` for the exact frozen constants. Summary: **PASS**
requires fuzzy to beat `no_repair` by a practically meaningful margin on BOTH
task accuracy and worst-cohort degradation, directionally consistent across
>= 80% of (budget, seed) runs, AND fuzzy must not be dominated by the best
simple competitor on task accuracy (paired bootstrap CI on
`fuzzy - best_simple` does not sit meaningfully below zero). A single
favorable metric is not sufficient - per the Prompt 14 non-negotiable
instruction, "require consistent benefit, not a single favorable metric."

## Naming/claims switch (automatic, not manual)

`core/naming.py::resolve_project_identity` maps the frozen Gate 4 decision to
a project name/claims tuple:

| Gate 4 decision | Project name | Claim framing |
|---|---|---|
| PASS | FairFuzzKV-Codec | Fuzzy repair-priority scoring is validated over the simpler competitors |
| WEAK_PASS | FairFuzzKV-Codec | Fuzzy scoring shows a real but modest/inconsistent benefit - keep the name, soften the claim |
| FAIL | FairFuzzKV-Codec (name unchanged - owner-chosen identity, not evidence) | Fuzzy scoring is negative evidence; the codec (quantization/pruning/allocation/format) is preserved; the fuzzy scorer becomes/stays NON-DEFAULT and the 'Fuzzy' in the name is explicitly not a validated claim, not a fabricated fuzzy-fairness win |

`scripts/run_gate4_study.py` applies this automatically after freezing the
decision - it is not a manual follow-up step.

## Non-claims

Gate 4 does not resurrect the Gate-1 (`WEAK_PASS`, RISK R-06) or Gate-2
(`FAIL`, RISK R-09) results. It is a narrower question: given the SAME
budget-neutral repair mechanism (Prompt 9, unchanged), does the fuzzy
scorer (Prompt 13) pick better repair candidates than a plain weighted sum,
a sigmoid score, or a knapsack ratio? A PASS here is a scoring-mechanism
finding, not a restored fairness or causal-fragmentation claim.
