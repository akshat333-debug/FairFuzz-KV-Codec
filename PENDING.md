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
  - `pruning` - partial (top-k baseline exists; policy layer pending)
  - `quantization` - scalar suite complete (Prompt 6: INT8/INT4,
    symmetric/asymmetric, per-tensor/head/channel/groupwise, percentile/
    MSE-optimal clipping, mixed precision incl. genuine per-head bits) AND
    LBG vector quantization complete (Prompt 7: `quantization/vector_quant.py`
    + `codec/vector_quant.py` - deterministic LBG training, head-block/
    cross-token vector formation, global/per-layer/per-head codebook scopes,
    chunked nearest-codeword with optional FAISS, serialized+counted codebook
    overhead, scalar-vs-LBG matched-bits benchmark `lbg_benchmark/`).
    Product/residual VQ (Prompt 7 item 49, optional) still pending.
  - `allocation` - empty (consumes fragility cohorts; blocked on Gate 1
    WEAK_PASS caveat above)
  - `metadata_coding` - empty (Golomb-Rice / entropy coding, later prompt)
  - `decoder` - empty (full reconstruction path, later prompt)
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
