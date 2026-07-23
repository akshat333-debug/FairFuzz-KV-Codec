# FairFuzzKV-Codec

FairFuzzKV-Codec is a research project for memory-conscious compression of Key-Value (KV) caches in Large Language Models.

## Current Project Status: Scalar Quantization Suite Complete (Prompts 1-6)

We are actively building the infrastructure and baseline evaluation pipeline. The repository has completed the **Vertical Skeleton Initialization**, the **Grade-Floor Baseline Gate**, the **Unicode-Aware Group Mapper (Module 1)**, the **Tokenizer Fragility Estimator & Cohort Builder (Module 2)**, the **FragKV-MinPairs Gate 1 Causal Test (Prompt 5)**, and the **Scalar Quantization Suite (Prompt 6)**.

> See [PENDING.md](PENDING.md) for the honest list of known gaps, deferred scope, and heuristic ceilings.
> **Read [gate1_study/GATE1_REPORT.md](gate1_study/GATE1_REPORT.md) before building Allocation** - Gate 1 came back **WEAK_PASS**, not PASS: fragmentation shows only a small, confound-entangled effect on compression failure at this model scale.

**Completed through Prompt 6.** Verification: 231/231 tests pass, `ruff` and `mypy` clean, all six prompt deliverables run end-to-end, Docker image builds and runs the CLI. Prompts 1-2 (grade-floor codec + honest byte accounting), 3 (Module 1: Unicode grouping), 4 (Module 2: fragility cohorts), 5 (Gate 1 causal test → WEAK_PASS), 6 (scalar quantization suite).

### What has been implemented so far:

1. **Robust Execution & Configuration Infrastructure**
   - Deterministic execution management with robust seed tracking and JSON configuration hashing.
   - Traceable manifests generated for every run to prevent silent overwrites.
   - Command-Line Interface (CLI) built with `typer` for capturing, encoding, decoding, and evaluating.
   
2. **Model Integration & KV Capture**
   - Memory-conscious KV cache capture from real open-weight Hugging Face models (e.g., Qwen, TinyLlama).
   - Support for granular Layer and Head selection for focused testing.
   
3. **Strict Binary Serializer & Byte Accountant**
   - `BinarySerializer` for exact byte-aligned layout serialization (Magic Header, Config Hash, JSON Metadata, and Raw Tensors).
   - `ByteAccountant` to strictly track serialized file sizes and report exact storage overheads against theoretical logical bit-budgets, explicitly forbidding sparsity-mask substitutions.

4. **Foundational Baseline Codecs**
   - `FullKVFP16Codec`: Exact FP16 standard baseline.
   - `UniformQuantCodec`: Uniform quantization supporting INT8 and INT4 with per-channel scaling options and saturation handling.
   - `TopKCodec`: Top-K token magnitude pruning baseline.

5. **Evaluation, Profiling, and Telemetry**
   - `MatchedBitEvaluator`: Enforces strict total-bit budget limits (e.g., 4 bits/element) by binary-searching retention ratios or tuning bit-widths, refusing to compare out-of-tolerance codecs.
   - `AttentionVerificationHarness`: A standalone mathematical verification harness to test L2/MSE attention distortion directly without requiring version-sensitive HF replacement.
   - `TelemetryTracker`: Measures peak CPU/GPU memory usage, prefill latency, and encode/decode times across warm-up and repeated sampling steps.
   - Matplotlib Dashboarding: Plots KV Distortion vs Bit Budget and Byte-level Compression Ratios.

6. **Module 1 — Unicode-Aware Surface-Unit & Group Mapper** (`fairfuzzkv_codec.unicode_grouping`)
   - Extended grapheme cluster segmentation (UAX #29) — never splits Indic combining sequences or emoji ZWJ sequences.
   - Surface units for words, punctuation, whitespace, numbers, URLs, and emoji, with structurally exact round-trip coverage (no gaps/overlaps).
   - `GroupMapper` aligns tokenizer subtokens to surface units via fast-tokenizer offset mappings, with deterministic repair rules and an explicit quarantine path (never guesses silently).
   - Works across two distinct tokenizer families (byte-level BPE, SentencePiece); versioned, JSON-serializable audit reports.

7. **Module 2 — Tokenizer Fragility Estimator & Cohort Builder** (`fairfuzzkv_codec.fragility_estimation`)
   - Per-group features (subtokens, chars/bytes-per-token, continuation ratio, script transitions, normalization sensitivity, rare-token proxy, boundary mismatch, token-cost inflation vs. English reference).
   - **Transparent monotone risk score** as the mandatory audit baseline (fixed, human-specified weights — never presented as learned), plus calibrated logistic/tree models validated against the transparent baseline on held-out data with reliability curves.
   - Quantile risk cohorts with minimum-sample-size merge and deterministic tie rules, stored in JSON manifests; cross-tokenizer stability analysis.
   - **Leakage guard**: raw language/script labels and task/compression outcomes are structurally forbidden from reaching any risk score or cohort — script grouping is descriptive-dashboard-only.

8. **Prompt 5 — FragKV-MinPairs & Gate 1 Causal Test** (`fairfuzzkv_codec.benchmarks.fragkv_minpairs`)
   - Minimal-pair benchmark generator: matched subject/distractors/position/difficulty, manipulating only evidence-unit fragmentation n_g ∈ {1,2,4,8} via a tokenizer-verified rendering ladder; every rendering round-trips to its canonical value.
   - Automated validators (answer equivalence, evidence identity, token-count target, context-position matching, no-answer-leakage) and an immutable sha256 split hash.
   - Real-model runner: captures a genuine prefill KV cache, compresses/decompresses it through FullKV + 2 codecs matched at 8 bits/element, splices the reconstruction back into the model (`DynamicCache(ddp_cache_data=...)`) to continue generation, and grades the result.
   - **Pre-registered** PASS/WEAK_PASS/FAIL decision logic (`gate1.py`), committed and tested on synthetic fixtures *before* the real 200-group study ran.
   - Real result: **WEAK_PASS** — see [gate1_study/GATE1_REPORT.md](gate1_study/GATE1_REPORT.md).

9. **Prompt 6 — Scalar Quantization Suite** (`fairfuzzkv_codec.quantization`, `fairfuzzkv_codec.codec.scalar_quant.ScalarQuantCodec`)
   - Symmetric and asymmetric INT8/INT4 for K and V independently, with per-tensor/per-head/per-channel/groupwise scale granularity.
   - Percentile and MSE-optimal clipping (small grid search), outlier saturation diagnostics, deterministic calibration-set selection.
   - **Genuine INT4 nibble packing** (two values per byte, verified by a byte-count test) - never stored one-per-byte in an int8 container.
   - Mixed K/V precision and per-layer bit-width configuration via a compact `BitWidthMap` (sparse overrides only, not one entry per layer).
   - Real distortion metrics (MSE, normalized L2, cosine drift, attention-output drift) and a real rate-distortion benchmark on captured Qwen2.5-0.5B caches: [quantization_benchmark/](quantization_benchmark/).
   - Two real bugs caught and fixed during development via the benchmark run itself, not just unit tests — see RISK_REGISTER R-07/R-08.
   - Fully standalone: no dependency on fragility cohorts, LBG/vector quantization, or fairness allocation, per this prompt's own acceptance gate.

### Next Steps:
- **Allocation (Module, Prompt 7+):** per Gate 1's WEAK_PASS result, do not premise allocation on "fragmentation causally requires protection" without re-reading `gate1_study/GATE1_REPORT.md`'s caveats first.

## Usage

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run the Grade-Floor baseline demo (real HF capture -> encode -> decode -> matched-bit eval)
uv run python scripts/demo.py

# Run the full test suite (unit + property-based, both tokenizer families)
uv run pytest

# Lint and type-check
uv run ruff check .
uv run mypy .
```

The demo runs the capture, encode, decode, and evaluation loop and writes artifacts (manifests and plots) to the `results/` directory.

### Using Module 1 & 2 programmatically

```python
from transformers import AutoTokenizer
from fairfuzzkv_codec.fragility_estimation import compute_fragility_report, build_cohort_definition, assign_cohort

tok = AutoTokenizer.from_pretrained("yujiepan/qwen2-tiny-random")
report = compute_fragility_report("Mujhe ye बहुत पसंद है 😀", tok)

for fv, rs in zip(report.feature_vectors, report.risk_scores):
    print(fv.unit_char_span, round(rs.score, 3), rs.feature_contributions)

scores = [rs.score for rs in report.risk_scores]
cohorts = build_cohort_definition(scores, tokenizer_name="qwen2", corpus_id="demo")
```
