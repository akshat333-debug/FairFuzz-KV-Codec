# Gate 4 Report: Fuzzy-vs-Simple Repair-Priority Ablation

**Decision: {{decision}}**

Reproducible from `gate4_fairness_study/predictions.jsonl` + the frozen
decision logic in `fairfuzzkv_codec.evaluation.gate4` (pre-registered
thresholds, committed and unit-tested on synthetic fixtures **before** this
study ran - see `tests/evaluation/test_gate4.py`). Frozen configuration:
`GATE4_CONFIG.md`.

## Question

Does the Module 3 **fuzzy** repair-priority scorer meaningfully beat
**no-repair** AND avoid being dominated by its simpler competitors
(monotone weighted score, knapsack value/cost ratio, sigmoid/logistic
score), consistently, at matched bits?

## Result

{{reasoning}}

| Metric | Value | Threshold |
|---|---|---|
| Mean accuracy gain (fuzzy vs no_repair) | {{mean_accuracy_gain}} | practical >= 0.10, weak >= 0.03 |
| Accuracy directional consistency | {{accuracy_consistency}} | >= 80% for PASS |
| Mean worst-cohort degradation gain | {{mean_worst_cohort_gain}} | practical >= 0.05, weak >= 0.02 |
| Worst-cohort directional consistency | {{worst_consistency}} | >= 80% for PASS |
| 95% CI, fuzzy - best simple competitor | [{{ci_low}}, {{ci_high}}] | must not sit meaningfully below 0 |

## Per-run comparison

{{runs_table}}

## Why {{decision}}

Fuzzy scoring cleared all three required checks - accuracy benefit over
no-repair, worst-cohort benefit over no-repair, and non-domination by the
best simple competitor - consistently across the frozen (budget, seed)
grid. This is a scoring-mechanism finding: it does **not** resurrect the
Gate-1 (`WEAK_PASS`, RISK R-06) causal-fragmentation claim or the Gate-2
(`FAIL`, RISK R-09) fairness-allocation claim, which remain as reported.

## Naming / claims

Project name: **{{display_name}}**. {{claim_framing}}

## Power / caveats

- Pilot scale (see `GATE4_CONFIG.md` dataset size) - a real but small
  study, not a final-power result. Re-run at scale before citing this as
  conclusive.
- The repair contract and its local mass condition (Prompt 9) are
  unchanged; this result concerns WHICH candidates get proposed for
  repair, not whether repair itself is safe.

## Failure-mode notes (Prompt 14 item 98)

{{item98_notes}}
