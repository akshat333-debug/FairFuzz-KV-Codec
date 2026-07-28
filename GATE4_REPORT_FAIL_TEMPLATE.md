# Gate 4 Report: Fuzzy-vs-Simple Repair-Priority Ablation

**Decision: {{decision}}** (negative evidence - codec preserved, no fuzzy claim fabricated)

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

## Why FAIL

{{failure_notes}}

Per the Prompt 14 non-negotiable instruction: the codec (capture, Unicode
grouping, fragility estimation, quantization, pruning, allocation, metadata
coding, decoder) is **preserved and unaffected**. Only the fuzzy-scoring
claim and the project name/claims change. Fuzzy scoring remains available
in the codebase as an optional, non-default repair-priority scorer
(`repair_scoring.ablation.ScorerType.FUZZY`) - it is not deleted, just not
claimed superior.

## Naming / claims

Project name: **{{display_name}}**. {{claim_framing}}

## Power / caveats

- Pilot scale (see `GATE4_CONFIG.md` dataset size) - a negative pilot
  result, not proof fuzzy scoring can never help. A larger, more
  heterogeneous candidate pool (naturalistic high-fragmentation text,
  multiple audited heads/layers) is the natural follow-up before writing
  fuzzy scoring off entirely.
- The repair contract and its local mass condition (Prompt 9) are
  unchanged; this result concerns WHICH candidates get proposed for
  repair, not whether repair itself is safe.

## Failure-mode notes (Prompt 14 item 98)

{{item98_notes}}

See `gate4_fairness_study/predictions.jsonl` (one row per example/budget/
seed/system, includes `repair_accepted`/`repair_attempted` and `kv_mse`) for
the raw data backing this section.
