# Gate 4 Report: Fuzzy-vs-Simple Repair-Priority Ablation

**Decision: FAIL** (negative evidence - codec preserved, no fuzzy claim fabricated)

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

fuzzy did not show a consistent, practically meaningful benefit over no_repair (accuracy gain -0.013, 25% consistent; worst-cohort gain -0.050, 0% consistent) and/or was dominated by the best simple competitor (CI [-0.050, 0.037]). Codec is preserved; fuzzy scoring is reported as NEGATIVE evidence, not a validated claim.

| Metric | Value | Threshold |
|---|---|---|
| Mean accuracy gain (fuzzy vs no_repair) | -0.013 | practical >= 0.10, weak >= 0.03 |
| Accuracy directional consistency | 25% | >= 80% for PASS |
| Mean worst-cohort degradation gain | -0.050 | practical >= 0.05, weak >= 0.02 |
| Worst-cohort directional consistency | 0% | >= 80% for PASS |
| 95% CI, fuzzy - best simple competitor | [-0.050, 0.037] | must not sit meaningfully below 0 |

## Per-run comparison

| Budget | Seed | Matched | Acc gain | Worst gain |
|---|---|---|---|---|
| 0.3 | 42 | True | -0.100 | -0.200 |
| 0.5 | 42 | True | 0.050 | 0.000 |
| 0.3 | 7 | True | 0.000 | 0.000 |
| 0.5 | 7 | True | 0.000 | 0.000 |

## Why FAIL

- fuzzy INCREASED worst-cohort degradation on average (-0.050) - overprotecting some cohorts at the expense of others (Prompt 14 item 98 failure mode)

Per the Prompt 14 non-negotiable instruction: the codec (capture, Unicode
grouping, fragility estimation, quantization, pruning, allocation, metadata
coding, decoder) is **preserved and unaffected**. Only the fuzzy-scoring
claim and the project name/claims change. Fuzzy scoring remains available
in the codebase as an optional, non-default repair-priority scorer
(`repair_scoring.ablation.ScorerType.FUZZY`) - it is not deleted, just not
claimed superior.

## Naming / claims

Project name: **FairFuzzKV-Codec** (unchanged - an owner-chosen constant, not evidence; an earlier auto-rename to FragKV-Codec was reverted because a distribution name is branding, not a scientific claim, and it broke the build). Fuzzy repair-priority scoring did not beat no-repair and/or the simpler competitors (Gate 4 FAIL) - negative evidence, not fabricated into a claim. The 'Fuzzy' in the name is historical identity only and must NOT be read as validation: the fuzzy scorer remains an optional, NON-DEFAULT scorer. The codec (capture, Unicode grouping, fragility estimation, quantization, pruning, allocation, metadata coding, decoder) is unaffected and preserved; its surviving, evidence-grounded contribution is tokenizer-fragmentation-aware KV compression.

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

- **Overprotection regressions found: 3/80 pooled (example, budget, seed) cells.** no_repair answered correctly, but fuzzy's accepted repair swap flipped the answer to incorrect, e.g. `g000001_42_ng1` at budget=0.3, seed=42 (KV MSE barely moved: 33.576 -> 33.625, so the regression is a discrete generation-outcome flip, not a large numerical distortion increase - the swap reintroduced a token that changed the model's argmax path).
- **Accept/reject instability across budgets (same tokenizer/model):** fuzzy's swap-acceptance flipped between budget=0.3 and budget=0.5 for 0/40 (group, seed) pairs. Cross-TOKENIZER stability (item 98's other named failure mode) could not be tested inside this pilot, since it used one model/tokenizer (`Qwen/Qwen2.5-0.5B`); it is measured separately below.
- **Cross-tokenizer stability, measured** (`scripts/run_cross_tokenizer_stability.py` -> `cross_tokenizer_study/cross_tokenizer_stability.json`): Spearman rank correlation of each scorer's repair-priority ordering for the same surface units under a byte-level-BPE vs a SentencePiece tokenizer, over 6 multilingual/code-mixed texts (no model generation needed - Module 1/2 signals only). Measured: **fuzzy rho=+0.901** (min +0.680), monotone +0.942 (min +0.909), logistic +0.942 (min +0.909), knapsack +0.841 (min +0.553). So fuzzy reorders candidates somewhat more than the monotone/logistic scorers when the tokenizer family changes (delta -0.041) but is more stable than the knapsack ratio; at this sample size that difference is **not** large enough to call fuzzy unstable. Reported as-is: a small-sample stability probe, not a powered study.

See `gate4_fairness_study/predictions.jsonl` (one row per example/budget/
seed/system, includes `repair_accepted`/`repair_attempted` and `kv_mse`) for
the raw data backing this section.
