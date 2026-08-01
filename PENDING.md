# Pending / Known Gaps

Status as of **Prompt 20 (final release)** completion. Everything below is either an
environment limitation, a deliberately-deferred scope item, a documented
heuristic ceiling, or a **negative/null scientific result that must NOT be
"fixed"** (fixing a FAIL by changing the number would be fabrication - those
entries are findings, not defects). Nothing here is a failing test or a broken
feature.

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

## Prompt 13: fuzzy scoring costs ~200x the simplest competitor (measured)

`repair_scoring_study/scorer_comparison.json` (measured with warm-up +
repeated timed runs, median reported): fuzzy Mamdani inference costs
~1.3e-4 s/candidate versus ~6.5e-7 s/candidate for the knapsack and
monotone scorers - a **~200x latency overhead** for 25 parameters vs 3-6.
Combined with the Gate 4 FAIL (fuzzy did not beat no-repair or its simpler
competitors on task accuracy), the honest reading is that fuzzy inference is
substantially more expensive AND not measurably better on this evidence.
It stays as an optional, non-default scorer. Note the absolute numbers are
tiny (sub-millisecond per candidate) and were measured on synthetic
candidates on CPU, so the ratio matters more than the magnitude.

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

## Prompt 14: Gate 4 FAIL - fuzzy scoring is negative evidence; project name is NOT switched by the gate

`GATE4_REPORT.md` / RISK_REGISTER R-10: on a real Qwen2.5-0.5B pilot (80
pooled cells), the fuzzy repair-priority scorer did not beat no-repair and
was not distinguishable from its simpler competitors. Do not cite fuzzy
scoring as validated; it stays in the codebase as an optional, NON-DEFAULT
scorer.

**Naming decision, revised (deliberate deviation from Prompt 14's automatic
rename).** An earlier revision let the gate rewrite the distribution name to
`fragkv-codec`. That has been reverted by the project owner and the name is
now a fixed constant, `FairFuzzKV-Codec` / `fairfuzzkv-codec`, for two
reasons: (1) a distribution name is identity/branding, not a scientific
claim, and (2) the rename **broke the build** - `uv_build` infers the module
directory from the project name, so `fragkv-codec` made it look for a
`src/fragkv_codec/` that does not exist, and the entire test suite failed to
install. `pyproject.toml` now pins `[tool.uv.build-backend] module-name`
explicitly so name and import package can never diverge again, and
`tests/core/test_naming.py` has a regression guard asserting
`package_name.replace("-","_") == "fairfuzzkv_codec"` for every decision.
What still follows the frozen decision automatically is the **claim
framing** in `PROJECT_IDENTITY.json` - a FAIL is reported as negative
evidence, and the "Fuzzy" in the name is explicitly documented as historical
identity, never as validation. The Python import path (`fairfuzzkv_codec`)
is likewise unchanged.

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

## Prompt 17: Gate 3 PASS at pilot scale; item-119 interactions now measured (both null)

`GATE3_REPORT.md` / RISK_REGISTER R-15: Gate 1 and Gate 2 reproduced in
decision category across Qwen2.5-0.5B and TinyLlama-1.1B-Chat, but Family
B's runs are pilot scale (20/16 groups, 1 budget, 1 seed) - much smaller
than Family A's original real runs (200/24 groups x 6 runs) - and Gate 2's
FAIL on TinyLlama came from a matched-bit-tolerance violation at that small
scale, not the same "identical allocations" mechanism Qwen showed. Per
Prompt 17 item 119, **model-family x allocator** and **quantizer-type x
cohort** interactions were initially deferred as needing "a second full
Prompt 10/11 allocator study on TinyLlama". That estimate was WRONG - the
allocator path is one prefill capture plus quantize/dequantize calibration,
with no autoregressive generation - so both were subsequently measured
(`scripts/run_gate3_interactions.py`). Both came back NULL: aggregate and
minimax chose identical allocations on both families (corroborating the
Gate 2 FAIL and showing it is not Qwen-specific), and `int8` won every
cohort on both families (no quantizer-by-cohort crossover). Single-text,
single-budget probes - absence of an interaction at this scale is not proof
none exists. Cohort risk-band assignment does NOT transfer between
these two tokenizer families (agreement 0.23, well below the 0.7 universal
threshold) - do not claim a universal risk threshold from Module 2's
cohorts without re-calibrating per tokenizer family. Also: the FragKV-
MinPairs numeric rendering ladder needed a documented tolerance widening to
even construct a TinyLlama dataset (see R-15) - any THIRD tokenizer family
added later should expect to need its own such check, not assume the
Qwen-calibrated ladder transfers by default.

## Prompt 18: decode-side numbers are attention REPLAY, not serving throughput

`PERFORMANCE.md` / `systems_profile/`: prefill is a real Hugging Face forward
pass, but decode-side consequences are measured through the attention replay
harness, NOT by integrating this codec into vLLM/TGI/HF-generate. So
"tokens/s" is a replay rate and is labelled that way everywhere; no end-to-end
serving speedup is claimed. Also measured and reported plainly: the codec is
NOT the bottleneck (prefill 115-208 ms vs 0.6-2.4 ms scalar encode), so this
codec buys memory, not speed. CPU-only, one machine; no GPU numbers are
reported because none were measured. `measure_peak_memory`'s CPU reading uses
tracemalloc and therefore does not see torch's C++ allocator - stated rather
than over-claimed.

## Prompt 19: dashboard is the sanctioned Streamlit fallback, not React/Next.js

Prompt 19 asks for React/Next.js + FastAPI with a "high-quality Streamlit
fallback only if schedule requires". This is the Streamlit fallback, chosen
because the interactive text demo must call the tokenizer, Module 1/2, and the
real codecs live, which is direct in-process and materially more moving parts
across an HTTP boundary plus a node build toolchain. Node IS available in this
environment (v22), so this is a scope/schedule judgement, not an environment
block - stated honestly. Trade-off: a less bespoke visual language than a
hand-built React product would give. The "fallback recorded assets" (item 134)
are a static HTML export (`scripts/export_demo_assets.py`), not a screen
recording - no video was produced.

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
- ~~**`continuation_ratio` mis-labels the first token of a sequence**~~ -
  **FIXED**. The sequence-initial token is now treated as word-initial by
  definition (`_is_continuation_piece(piece, sequence_index)`), using the
  absolute token index already carried on `GroupRecord.token_indices`.
  Regression tests: `test_sequence_initial_token_is_not_counted_as_a_continuation`,
  `test_first_unit_continuation_ratio_is_not_inflated`.

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
- ~~**Downstream Task Accuracy metric** (exact match, F1)~~ - **DONE**:
  `evaluation/downstream.py`. It was previously blocked on the dataset
  integrations, but the metric has no dataset dependency - it is a pure
  function of predicted/gold strings - so it ships independently and works on
  any benchmark with auditable answers (FragKV-MinPairs, IndicLongComp).
  Unicode-aware normalization (NFKC, Unicode-category punctuation stripping,
  English-only article removal so Indic text is not corrupted), multi-reference
  support, explicit empty-answer conventions. Tested in
  `tests/evaluation/test_downstream.py`.
- ~~**Propositions 1 & 2**~~ - **DONE**. Both now exist and pass:
  `tests/eval/test_prop1_fragility.py` (P1, Fragility Distribution: the
  transparent score induces a non-degenerate, reproducible ordering in which
  more-fragmented units score no lower, and quantile cohorts partition it into
  ordered, covering bands) and `tests/eval/test_prop2_allocation.py` (P2,
  Allocation Optimality: the exact solver matches independent brute-force
  enumeration, greedy stays within a bounded gap and never beats the optimum,
  no allocation exceeds budget, and both objectives are monotone in budget).
  Each file carries a guard test asserting the proposition is NOT read as the
  corresponding gate claim (Gate 1 WEAK_PASS / Gate 2 FAIL).
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
  - `experiment_tracking` - **DONE** (`experiment_tracking/registry.py`):
    append-only JSONL run index tying each study run to its git commit (with
    `-dirty` marker), config, seeds, metrics, and artifact paths, plus
    per-study querying and metric history. Dependency-free by choice (no
    MLflow/W&B); append-only so a rerun can never quietly restate an earlier
    result. Tested in `tests/experiment_tracking/`.

## Verification status (through Prompt 20 - FINAL RELEASE)

- **548/548 tests pass**, ruff clean, mypy clean (197 source files).
- **Release checklist: 10/10 passed, 0 failed, 0 skipped** (`scripts/run_release_checklist.py --full`),
  covering clean install, lint/types, full suite, binary compatibility, gate
  decision records, gate reproduction from raw predictions, dashboard render,
  report generation, required documents, dataset regeneration, and core
  experiments.
- **62 released artifacts checksum-verified** against `release/CHECKSUMS.sha256`.
- Every prompt deliverable runs end-to-end on real models: grade-floor demo
  (`scripts/demo.py`), unicode grouping, fragility estimation, Gate 1 study
  (reproducible from committed `gate1_study/predictions.jsonl` - re-run from
  scratch and matched exactly), scalar + LBG quantization benchmarks, FFKV
  binary format with golden vectors, pruning/repair/bound, aggregate + minimax
  allocators, Gate 2 / Gate 3 / Gate 4 studies, IndicLongComp, baseline matrix.
- All four gate decisions are reproducible from raw predictions alone, without
  model access (`compute_gate1_from_predictions`, `compute_gate4_from_predictions`,
  and the Gate 2/Gate 3 report paths).
- Docker build verified working; no dependency changes since Prompt 4.

**Gate results, stated plainly:** Gate 1 WEAK_PASS, Gate 2 FAIL, Gate 3 PASS
(pilot scale), Gate 4 FAIL. Two of four are negative. They are recorded as
negative evidence and are not to be "resolved" by re-running until a nicer
number appears.

## Resolved earlier (was pending, now done)

- transformers v5 `DynamicCache` capture break - fixed + tested (RISK R-02).
- Matched-bit evaluator unit bug - fixed + tested.
- TopK byte-accounting honesty bug - fixed + tested.
- Demo false-success message, CLI fake evaluate score - fixed.
- `code-review-graph` graph.db lock - resolved via MCP-server build tool
  (RISK R-05).
- INT4 genuine packing (Prompt 6 `ScalarQuantCodec`), groupwise scale
  storage bug (R-07), asymmetric int8 wraparound bug (R-08).
