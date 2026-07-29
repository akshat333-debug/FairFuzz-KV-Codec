# Pending / Known Gaps

Status as of Prompt 6 completion. Everything below is either an environment
limitation, a deliberately-deferred scope item from a later prompt, or a
documented heuristic ceiling. Nothing here is a failing test or a broken
feature in the completed prompts (Prompts 1-6 all pass: 231/231 tests, ruff
clean, mypy clean).

## Gate 1 (Prompt 5) result: WEAK_PASS, not PASS - read before building Allocation

The real 200-group causal study (`gate1_study/GATE1_REPORT.md`) found only a
small (7.5-point), confound-entangled effect of evidence fragmentation on
compression-induced task failure - the lossless FullKV control itself
collapses 48%→0% between n_g=1 and n_g=8, so the study cannot cleanly
attribute that collapse to compression rather than the base model failing to
parse the (synthetic, zero-width-joiner-heavy) high-fragmentation renderings
at all. See RISK_REGISTER R-06. **Do not build an Allocation module premised
on "fragmentation causally requires protection" without first either (a)
re-running Gate 1 with more naturalistic high-fragmentation renderings and/or
a larger model, or (b) explicitly framing fragility-cohort-based allocation
as a heuristic choice, not an empirically validated one**, per the Prompt 5
non-negotiable instruction and `gate1_study/GATE1_REPORT.md`'s own next-steps
section.

## Prompt 6 real benchmark result: MSE does not always predict task accuracy

`quantization_benchmark/benchmark_results.json` (real, Qwen2.5-0.5B): at
~8 bits/element, INT8-per_tensor scored HIGHER task accuracy (84%, 42/50)
than INT8-per_channel (78%, 39/50) despite per_channel having much LOWER K
MSE (0.013 vs 0.234). At ~4 bits/element the ordering flips back to matching
MSE (per_channel 18% > groupwise16 8% > per_tensor 6%, tracking MSE 1.6 <
5.3 < 6.9). This is reported as-is - real, measured, not smoothed into a
tidy story. Plausible explanations (not verified): n=50 is small enough for
sampling noise to matter at 8 bits where accuracy is already high, and/or
overall MSE doesn't fully capture which specific values matter most for this
retrieval task. Do not assume per_channel granularity is strictly better for
task purposes at high bit-widths without a larger study.

## Environment-blocked (not code defects)

_(none currently)_

### Resolved: Docker build
`docker build -t fairfuzzkv-codec-test .` now **succeeds end-to-end**, and the
built image runs the CLI (`docker run --rm fairfuzzkv-codec-test inspect`
outputs a valid config). Earlier failures were sandbox container-registry
flakiness (pulls of `python:3.13-slim` / `ghcr.io/astral-sh/uv` timing out),
not a Dockerfile defect. Fixes applied: base image `python:3.10-slim` ->
`python:3.13-slim` (matches `requires-python >=3.12`), and `pip install uv`
instead of the `ghcr.io` image copy.

## Resolved in Prompt 6: INT4 genuinely packed now (new codec, old baseline unchanged)

`codec/baselines.py`'s original `UniformQuantCodec` still stores INT4 in an
int8 container (unchanged - it's the simple grade-floor baseline from Prompts
1-2 and other code depends on its exact behavior). Prompt 6 adds a SEPARATE,
stronger `codec/scalar_quant.py::ScalarQuantCodec` that packs INT4 two values
per byte for real (`quantization/packing.py`, verified: 231/231 tests pass,
including a byte-count regression test). Use `ScalarQuantCodec` for anything
needing genuine 4-bit storage; `UniformQuantCodec` remains available for
existing callers (`baselines.py`, `demo.py`, matched-bit evaluator) that
don't need packing.

## Prompt 13: fuzzy repair-priority scorer validated on synthetic candidates only

`repair_scoring_study/scorer_comparison.json` compares the fuzzy scorer
against its three competitors on synthetic candidate signals (random
fragility/evidence/cost/staleness values), not a real repair-outcome
dataset - none exists yet. Do not cite one scorer as "better" than another
for real repair decisions without a downstream-task study first. A small
calibrated tree/MLP competitor (Prompt 13 item 88, marked "if justified")
was deliberately NOT implemented for the same reason: fitting one against a
proxy label would fabricate a validation result the project has no real
data to support. See CLAIMS_LEDGER C-22.

## Prompt 14: Gate 4 FAIL - fuzzy scoring is negative evidence; Python import path NOT renamed

`GATE4_REPORT.md` / RISK_REGISTER R-10: on a real Qwen2.5-0.5B pilot (80
pooled cells), the fuzzy repair-priority scorer did not beat no-repair and
was not distinguishable from its simpler competitors. Do not cite fuzzy
scoring as validated. The automatic naming switch (`core/naming.py`)
renamed the PyPI/pip distribution name in `pyproject.toml` to
`fragkv-codec` and wrote `PROJECT_IDENTITY.json`, per the frozen decision -
but the Python IMPORT path (`fairfuzzkv_codec`, ~100+ source/test files)
was deliberately left unchanged. Renaming every import statement is a large,
mechanical, error-prone change disproportionate to what "the project name
and claims automatically follow the decision file" (Prompt 14's acceptance
gate) actually requires - it names PROJECT metadata, not the internal
package layout. If a full package rename is wanted later, it should be its
own reviewed change, not a side effect of a gate decision script.

## Prompt 15: IndicLongComp built at pilot scale - journal subset has no real FullKV run yet

`INDICLONGCOMP_REPORT.md` / RISK_REGISTER R-12: the course subset's real
Qwen2.5-0.5B FullKV run found an EMPTY intersection-full-correct isolation
subset (0/10 groups), consistent with small-sample noise (8-10 datapoints
per breakdown cell), not a validated capability or fragility-effect
finding. The 250-group journal subset is generated and structurally
validated (parallelism, PII, dedup, contamination) but has NOT had a real
FullKV run - that is the natural next step before any compression/fairness
comparison on this benchmark, left undone here given real per-example
model-forward-pass compute cost (~1000 generations) beyond this session's
budget. Also: every context/question string in this benchmark is
LLM-authored from hand-designed templates, not sourced from MLRBench or any
other external corpus (no verified network/license access was available)
and not professionally translated or reviewed - stated explicitly in every
dataset card's `content_provenance_note`, never presented as sourced or
reviewed content.

## Prompt 16: baseline matrix - RateQuant/RDKV/KVTuner/KVmix not reproduced; group-aware pruning not in the automated matrix

`BASELINE_MATRIX_REPORT.md` / RISK_REGISTER R-13/R-14: per the Prompt 16
non-negotiable ("reproducibility is more important than having every
baseline name in a table"), RateQuant, RDKV, KVTuner, and KVmix are
explicitly `NOT_REPRODUCED` - no verified network/license access was
available to confirm their exact published algorithms - each with a
documented reason and a nearest-faithful-configuration pointer to this
project's own real functionality (never silently substituted under those
names). SnapKV, PyramidKV, and H2O ARE implemented but marked
`APPROXIMATE`: their defining mechanism is reproduced, but hyperparameter
defaults (observation window/pooling kernel, pyramid ratio, heavy/recent
split) were not verified against reference code. Group-aware pruning
(Prompt 9) is also not in the automated matrix - it needs text-derived
surface-group IDs (from Module 1's GroupMapper) beyond the common
`Callable[[float], BaseCodec]` adapter shape used here; a text-aware
adapter variant is the natural follow-up, not attempted in this prompt.
The real run covers one matched-bit target (4.0 bits/element) on one
course subset - not a budget sweep.

## Prompt 17: Gate 3 PASS at pilot scale; model-family x allocator and quantizer x cohort interactions deferred

`GATE3_REPORT.md` / RISK_REGISTER R-15: Gate 1 and Gate 2 reproduced in
decision category across Qwen2.5-0.5B and TinyLlama-1.1B-Chat, but Family
B's runs are pilot scale (20/16 groups, 1 budget, 1 seed) - much smaller
than Family A's original real runs (200/24 groups x 6 runs) - and Gate 2's
FAIL on TinyLlama came from a matched-bit-tolerance violation at that small
scale, not the same "identical allocations" mechanism Qwen showed. Per
Prompt 17 item 119, **model-family x allocator** and **quantizer-type x
cohort** interaction effects were NOT attempted - both would need a second
full Prompt 10/11 allocator study on TinyLlama, beyond this session's
compute budget. Cohort risk-band assignment does NOT transfer between
these two tokenizer families (agreement 0.23, well below the 0.7 universal
threshold) - do not claim a universal risk threshold from Module 2's
cohorts without re-calibrating per tokenizer family. Also: the FragKV-
MinPairs numeric rendering ladder needed a documented tolerance widening to
even construct a TinyLlama dataset (see R-15) - any THIRD tokenizer family
added later should expect to need its own such check, not assume the
Qwen-calibrated ladder transfers by default.

## Deliberate heuristic ceilings (documented in code with `ponytail:` comments)

- **`ScalarQuantCodec` mixed-precision** (`codec/scalar_quant.py`) groups by
  LAYER on the fast path (byte-identical to the original implementation) and
  automatically switches to per-(layer, head) **cell grouping** when the
  `BitWidthMap` carries head-level overrides for that tensor, so individual
  heads can now carry distinct bit-widths. Each cell is quantized as a
  single-head slice, so `PER_HEAD` and `PER_TENSOR` granularity coincide
  within a cell (one scale per head). Tested:
  `test_head_override_uses_cell_grouping_and_distinct_bits_per_head`.
- **`language_hint` is always `None`** (`unicode_grouping`). No language-ID model
  is wired in; the field exists in the schema but is never populated. It is also
  intentionally forbidden from entering fragility scoring (see leakage.py).
- **Number surface units do not merge across decimal separators**
  (`unicode_grouping/surface_units.py`). `"3.14"` becomes
  `NUMBER('3') PUNCTUATION('.') NUMBER('14')`. Round-trip coverage is still exact.
- **Slow (non-fast) tokenizers unsupported** (`unicode_grouping/aligner.py`).
  Raises explicitly rather than guessing offsets. Both required tokenizer
  families (byte-level BPE, SentencePiece) ship as fast tokenizers, so this is
  not a gap for the acceptance gates.
- **`rare_token_indicator` uses token-id rank as a rarity proxy**
  (`fragility_estimation/features.py`). No real corpus-frequency table.
- **`continuation_ratio` mis-labels the first token of a sequence** as a
  continuation (it never carries a leading-space marker even when word-initial).
  Affects at most one token per text.

## Pilot-scale validation (real numbers, too small to generalize) - see RISK_REGISTER R-03

- **Fragility calibration** (`fragility_estimation/calibrated_model.py`) is
  validated on ~100 surface units from a 7-sentence curated corpus. Held-out
  AUC/Brier are real and measured, never fabricated, but the sample is too small
  to claim the learned model generalizes. Re-run against a large multilingual
  corpus before relying on the calibrated model for allocation.

## Deferred to later prompts (out of scope through Prompt 6)

- **Datasets** LongBench (Dataset 1) and PG-19 (Dataset 2): not integrated.
  Marked `(Pending)` in SPEC_TRACEABILITY.md; planned paths
  `fairfuzzkv_codec.benchmarks.longbench` / `.pg19`.
- **Downstream Task Accuracy metric** (exact match, F1): not implemented -
  depends on the dataset integrations above.
- **Propositions 1 & 2** (`tests/eval/test_prop1_fragility.py`,
  `test_prop2_allocation.py`): pending, depend on Allocation module.
- **Remaining spec modules not yet built** (empty/partial stubs):
  - `pruning` - COMPLETE (Prompt 9: recency/top-attention-mass/top-k/group-
    aware selectors, max/sum/normalized group aggregation, attention-mass
    repair contract with local `p_E^repair <= p_E^0 + delta` enforcement +
    per-head/query logging, local per-head bound validator that reports
    assumption failures). Local head-level verification ONLY - not an
    end-to-end guarantee (CLAIMS_LEDGER C-18/C-19).
  - `quantization` - scalar suite complete (Prompt 6: INT8/INT4,
    symmetric/asymmetric, per-tensor/head/channel/groupwise, percentile/
    MSE-optimal clipping, mixed precision incl. genuine per-head bits) AND
    LBG vector quantization complete (Prompt 7: `quantization/vector_quant.py`
    + `codec/vector_quant.py` - deterministic LBG training, head-block/
    cross-token vector formation, global/per-layer/per-head codebook scopes,
    chunked nearest-codeword with optional FAISS, serialized+counted codebook
    overhead, scalar-vs-LBG matched-bits benchmark `lbg_benchmark/`).
    Product/residual VQ (Prompt 7 item 49, optional) still pending.
  - `allocation` - COMPLETE as the AGGREGATE control condition (Prompt 10:
    train/val/test-separated distortion calibration, exp/monotone distortion
    curves, exact-DP + greedy water-filling allocator with optimality-gap
    validation, drives the real ScalarQuantCodec, `allocation_study/`
    artifacts). Deliberately framed as a heuristic aggregate baseline, NOT
    premised on the Gate-1 causal claim (WEAK_PASS caveat above). The
    FAIRNESS-constrained MINIMAX allocator that protects the worst cohort vs
    this aggregate control is now implemented (Prompt 11: `allocation/minimax.py`,
    `ALLOCATION_MATH.md`, frozen `GATE2_CONFIG.md`, `gate2_study/`). At a generous
    midpoint budget the two coincide (no cohort tension); the Pareto sweep shows
    where fairness trades average distortion to lower the worst case - reported,
    not hidden. Cohort-conditioned CODEBOOKS driven by fragility (tying minimax
    to Module 2 cohorts) remain a later-prompt deliverable.
  - `evaluation` Gate 2 (Prompt 12) - COMPLETE machinery, **FAIL** result.
    Isolation subset, robust disparity metrics (CDDB/range/std/worst/risk-
    coverage), paired bootstrap, frozen PASS/WEAK_PASS/FAIL logic (tested
    pre-run). Real pilot `gate2_fairness_study/`: aggregate and minimax chose
    identical allocations (zero fairness benefit) and isolation kept only
    n_g=1,2 (n_g=4,8 have ~0 base-correct - same Gate-1 confound). Reported as
    negative evidence (RISK_REGISTER R-09), codec preserved, no fabricated
    fairness claim. A properly powered re-test needs naturalistic high-frag
    renderings / larger model AND a heterogeneous cohort set whose degradation
    curves diverge - then re-run the unchanged frozen decision logic.
  - `metadata_coding` - COMPLETE (Prompt 8: FFKV binary format v1 container
    with CRC32 checksums + forward-compat section skipping, Golomb-Rice
    retention coding with bitmap/RLE fallback chosen by measured length,
    blockwise-Rice integer coders, LEB128 varints, `FORMAT.md` + golden
    vectors). Streaming/incremental append across decode steps not wired
    (prefill-regime container only); a future decode-regime writer can add
    sections without a format change (forward-compat directory already
    supports it).
  - `decoder` - COMPLETE for container reconstruction (Prompt 8:
    `decode_from_container` rebuilds scalar OR LBG payloads with a
    completeness report). Direct model-injection generation loop remains the
    version-sensitive path already covered by the attention-equivalence
    harness (Prompt 2), not this container decoder.
  - `experiment_tracking` - empty

## Verification status (through Prompt 6)

- **231/231 tests pass**, ruff clean, mypy clean (95 source files).
- All six prompt deliverables run end-to-end: grade-floor demo
  (`scripts/demo.py`), unicode grouping, fragility estimation, Gate 1 study
  (reproducible from committed `gate1_study/predictions.jsonl`), scalar
  quantization + real rate-distortion benchmark (`quantization_benchmark/`).
- Docker build verified working earlier this session; no dependency changes
  since Prompt 4, so still valid.
- `code-review-graph` clean and current: 98 files / 495 nodes / 3523 edges,
  0 errors.

## Resolved earlier (was pending, now done)

- transformers v5 `DynamicCache` capture break - fixed + tested (RISK R-02).
- Matched-bit evaluator unit bug - fixed + tested.
- TopK byte-accounting honesty bug - fixed + tested.
- Demo false-success message, CLI fake evaluate score - fixed.
- `code-review-graph` graph.db lock - resolved via MCP-server build tool
  (RISK R-05).
- INT4 genuine packing (Prompt 6 `ScalarQuantCodec`), groupwise scale
  storage bug (R-07), asymmetric int8 wraparound bug (R-08).
