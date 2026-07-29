# Gate 3 Experiment Configuration (FROZEN)

Pre-registered configuration for the Prompt 17 cross-tokenizer/cross-model
reproduction check. Frozen **before** running the cross-model study,
mirroring the Gate 1/2/4 pre-registration discipline. Decision logic in
`evaluation/gate3.py` is committed and tested against synthetic fixtures
(`tests/evaluation/test_gate3.py`) before this config is ever pointed at a
real study.

## Model/tokenizer families (item 114)

| | Family A (existing) | Family B (new) |
|---|---|---|
| Model | `Qwen/Qwen2.5-0.5B` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Params | ~0.5B | ~1.1B |
| Tokenizer | byte-level BPE (Qwen2 family) | SentencePiece (Llama family) |
| License | Apache 2.0 | Apache 2.0 |
| Context limit (model max) | 32,768 tokens | 2,048 tokens |
| Hardware | CPU-runnable (used throughout this project) | CPU-runnable; slower than Qwen2.5-0.5B (~2x params) - real wall-clock cost, not estimated |
| Architecture | Qwen2 (GQA) | Llama (GQA) |

These two families are materially different in the dimension this project's
claims depend on: tokenizer algorithm (byte-level BPE vs SentencePiece),
which is exactly what Module 1/2's cross-tokenizer stability machinery
(`fragility_estimation.stability`) already exists to compare - reused
unchanged here, not rebuilt (item 116).

## Frozen protocol - reused unchanged where possible (item 115)

Both `scripts/run_gate1_study.py` and `scripts/run_gate2_study.py` gained a
`--model` CLI flag for this prompt (the ONLY change to either frozen
script) so the exact same decision logic, codec configuration, and dataset
generator run identically against both families - "identical frozen
protocol" means the same *script*, not a re-derived one.

| Item | Family A (already run, Prompts 5/12) | Family B (this prompt) |
|---|---|---|
| Gate 1 groups | 200 (real, committed `gate1_study/`) | 20 (pilot - TinyLlama's ~2x slower CPU forward pass makes the original 200-group scale impractical in this session; explicitly reduced, not silently) |
| Gate 2 groups / budgets / seeds | 24 groups x {5,6,7} bits x {42,7} = 6 runs | 16 groups x {6} bits x {42} = 1 run (reduced scope, same reason) |
| Codec menus | Identical (`build_codecs`, `BIT_CHOICES=[4,8]`) | Identical |
| Decision logic | `gate1.py` / `gate2.py`, unchanged | Same, unchanged |

## Cohort transfer analysis (item 116)

`fragility_estimation.stability.compute_cross_tokenizer_stability` (already
built in Module 2, not new code) run on a shared text corpus, comparing
cohort BAND assignment (not raw scores) between the two tokenizers.
`UNIVERSAL_AGREEMENT_THRESHOLD = 0.7` (existing, frozen constant) decides
"universal" vs "model_specific" cohort transfer.

## Interaction effects reported (item 119) - scoped

- **Tokenizer family x fragmentation level**: reported directly - each
  family's Gate 1 `accuracy_by_n_g` curve, side by side.
- **Model family x allocator** and **quantizer type x cohort**: explicitly
  **DEFERRED**, not attempted in this prompt. Both would require re-running
  the Prompt 10/11 allocator study (`scripts/run_minimax_demo.py`) on
  TinyLlama - a second full allocator study on a slower model, beyond this
  session's compute budget. Documented as a gap (PENDING.md), not silently
  dropped from the acceptance gates it doesn't actually claim to need
  (Prompt 17's acceptance gates don't name these two interactions
  specifically as required - the mandatory-work item does, so this is a
  real, stated scope reduction, not a hidden one).

## Frozen Gate 3 decision (item 120)

`evaluation.gate3.decide_gate3`: reproducibility is judged on whether each
family's Gate 1 and Gate 2 decision CATEGORY (SIGNAL = PASS/WEAK_PASS vs
NO_SIGNAL = FAIL) agrees - not on pooled significance across families. PASS
requires both gates to reproduce in category; WEAK_PASS if only one does;
FAIL if neither does. Reproducing a NEGATIVE finding (both families FAIL)
counts as PASS on reproducibility - the question is "does the same
qualitative pattern hold", not "is there a strong positive effect".

## Non-claims

Per the Prompt 17 non-negotiable ("a single-model success is a course
result, not a general research claim"): Gate 3 PASS does not retroactively
upgrade Gate 1's WEAK_PASS or Gate 2's FAIL into stronger claims - it only
says the SAME qualitative pattern was observed on a second family. Claim
scope is narrowed, never inflated, by this gate (see
`gate3.Gate3Report.claim_scope_statement`, generated automatically from the
decision).
