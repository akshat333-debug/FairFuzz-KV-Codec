# Gate 1 Report: FragKV-MinPairs Causal Test

**Decision: WEAK_PASS**

**This report is fully reproducible from `predictions.jsonl` alone** (no model
access needed) via `scripts.run_gate1_study.compute_gate1_from_predictions`.
The decision logic in `gate1.py` was committed and unit-tested against
synthetic fixtures ([commit adab7ee](../../commit/adab7ee)) *before* this
study was run - it was not adjusted after seeing these results.

## Setup

- Model: `Qwen/Qwen2.5-0.5B` (real pretrained weights, not random-init)
- Dataset: 200 validated FragKV-MinPairs groups, seed=42, split_hash
  `56f325e45ea064dc...` (see `dataset/dataset_card.json`)
- Codecs, matched at 8 bits/element (both by construction, not tuned per-run):
  `UniformQuant8` (int8 uniform quantization), `TopK50` (50% retention).
  `FullKV` (16 bits/element, fp16) is the lossless control.
- Task: recall a single-digit "code" for a named subject among 3-7
  distractor facts; evidence rendering fragmented into 1/2/4/8 tokens via a
  fixed, tokenizer-verified rendering ladder (digit -> dot-separated ->
  hyphen/zero-width-joined spelled-out letters -> fullwidth/double-separator
  variants). See `dataset/groups.jsonl` for every prompt actually used.
- 2400 raw predictions total (200 groups x 4 fragmentation levels x 3 codecs).

## Headline numbers (accuracy by fragmentation level n_g)

| Codec | n_g=1 | n_g=2 | n_g=4 | n_g=8 | Effect (1 vs 8) | Monotonic? | p-value |
|---|---|---|---|---|---|---|---|
| FullKV (control) | 0.480 | 0.665 | 0.025 | 0.000 | 0.480 | **No** | 0.0000 |
| UniformQuant8 | 0.825 | 0.850 | 0.015 | 0.005 | 0.820 | **No** | 0.0000 |
| TopK50 | 0.105 | 0.100 | 0.045 | 0.030 | 0.075 | **Yes** | 0.0004 |

Effect = paired mean(correct@n_g=1 - correct@n_g=8) across the 200 matched
groups; positive = higher fragmentation hurt. 95% bootstrap CIs, p-values
from a 5000-permutation sign-flip test - both computed by `stats_utils.py`.

## Why WEAK_PASS, not PASS or FAIL

The **FullKV control itself collapses from 48% to 0%** accuracy between
n_g=1 and n_g=8 - with ZERO compression. That is a **control confound**:
whatever breaks the model at n_g=4/8 is not specific to any codec. The
`word_zwnj_letters`/`word_double_sep_letters` renderings used at n_g=4/8
(zero-width-joined spelled-out digit words) are apparently far enough
outside this small model's training distribution that it frequently fails
to recover the value regardless of compression - notice accuracy even *rises*
from n_g=1 to n_g=2 before collapsing, which is itself non-monotonic and a
sign this may be more "does the model recognize this rendering at all" than
a smooth fragmentation effect.

Of the two lossy codecs, only **TopK50** shows a directionally consistent
(monotonically non-increasing) decline across all four levels - a real,
statistically significant effect (p=0.0004) but modest in size (7.5 points),
below the pre-registered 10-point "practically meaningful" bar.
**UniformQuant8** shows a much larger raw gap (82 points) but is **not
monotonic** (85% -> 1.5% between n_g=2 and n_g=4) and starts *higher* than
FullKV at n_g=1 (82.5% vs 48%) - almost certainly int8 rounding perturbing
which token wins near a decision boundary, in either direction, rather than
a clean fragmentation story.

Per the pre-registered logic: no codec is both meaningful *and* free of the
control confound -> not PASS. TopK50 alone clears the weak-effect bar with
real directional consistency -> not FAIL. Result: **WEAK_PASS**.

## Honest interpretation

- There **is** a real, statistically robust, monotonic signal that more
  fragmentation costs more accuracy under lossy compression (TopK50).
- It is **small** (7.5 points) relative to a **huge base-model confound**
  that this study cannot cleanly separate from a genuine compression effect,
  because the n_g=4/8 renderings may simply be too unnatural for a 0.5B
  base model to parse at all, compressed or not.
- This is a genuine negative-leaning result for the strong causal claim
  ("fragmentation causes compression-specific failure") at this model scale
  and rendering design. It does **not** rule the hypothesis out - it says
  this particular pilot study is underpowered to isolate it from the
  confound, and/or the n_g=4/8 renderings need to be less exotic (e.g. avoid
  zero-width joiners, which may be a stronger out-of-distribution signal to
  a small model than "fragmentation" per se).

## Known limitations (do not re-run to "fix" without a new pre-registration)

1. **Rendering realism**: zero-width-joiner-based renderings for n_g=8 are a
   strong synthetic artifact, arguably more "adversarial" than "fragmented."
   A follow-up study should prefer renderings a real tokenizer would produce
   naturally (e.g. genuine transliteration/script variants) over
   zero-width-joiner insertion, to reduce the confound.
2. **Model scale**: only tested on a 0.5B base model. A larger or
   instruction-tuned model may not exhibit the same collapse at n_g=4/8,
   which would shrink the control confound and let a cleaner effect show
   through.
3. **Single model, single task template**: per Gate 1's own acceptance
   criteria this is the prototype-scale run; broader claims need more model
   families and task templates.

## Next steps (see PENDING.md)

- Do not adopt "fragmentation causes compression failure" as a validated
  claim in the project report at this scale - report WEAK_PASS honestly,
  including the confound.
- If pursuing this further: redesign n_g=4/8 renderings to avoid zero-width
  joiners (item 1 above), and/or test on a larger model, before drawing a
  stronger conclusion.
- The codec deliverable (Modules 1-4) is unaffected either way - it does not
  depend on this causal claim.
