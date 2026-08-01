# FairFuzzKV-Codec

> **The project name is a fixed, owner-chosen constant — it is NOT evidence.**
> An earlier revision let the frozen Gate 4 decision rename the project
> (`FairFuzzKV-Codec` → `FragKV-Codec`); that rename has been reverted by the
> project owner. A distribution name is identity/branding, not a scientific
> claim (and the rename also broke the build, since the import package is
> `fairfuzzkv_codec`). What still follows the evidence automatically is the
> **claim framing**: `core/naming.py` maps the frozen Gate 4 decision to the
> wording in [PROJECT_IDENTITY.json](PROJECT_IDENTITY.json).
>
> **Gate 4 came back FAIL** ([GATE4_REPORT.md](GATE4_REPORT.md)): the Module 3
> fuzzy repair-priority scorer did not beat no-repair. The "Fuzzy" in the name
> is historical only and must **not** be read as a validated claim — the fuzzy
> scorer remains in the codebase as an optional, **non-default** scorer. See
> RISK_REGISTER R-10 and CLAIMS_LEDGER C-23.

FairFuzzKV-Codec is a research project for memory-conscious compression of Key-Value (KV) caches in Large Language Models.

## Current Project Status: FINAL RELEASE (Prompts 1-20)

This is the final release. The repository completed the **Vertical Skeleton Initialization**, the **Grade-Floor Baseline Gate**, the **Unicode-Aware Group Mapper (Module 1)**, the **Tokenizer Fragility Estimator & Cohort Builder (Module 2)**, the **FragKV-MinPairs Gate 1 Causal Test (Prompt 5)**, the **Scalar Quantization Suite (Prompt 6)**, **LBG Vector Quantization (Prompt 7)**, the **Unified Binary Format + Golomb-Rice Metadata Coding + Streaming Decoder (Prompt 8)**, **Pruning Selectors + Attention-Mass Repair + Local Bound Validation (Prompt 9)**, the **Aggregate Rate-Distortion Allocator (Prompt 10)**, the **Fairness-Constrained Minimax Water-Filling Allocator (Prompt 11)**, the **Gate 2 Matched-Bit Fairness Experiment (Prompt 12)**, the **Fuzzy Repair-Priority Scorer & Competitors (Prompt 13)**, the **Gate 4 Ablation & Naming Decision (Prompt 14)**, the **IndicLongComp Benchmark (Prompt 15)**, the **Baseline Matrix (Prompt 16)**, **Gate 3 Cross-Model Reproduction (Prompt 17)**, **Systems Profiling & Hardening (Prompt 18)**, the **Research Dashboard (Prompt 19)**, and the **Final Reproducibility Release (Prompt 20)**.

> **Headline result, stated up front: two of the four pre-registered gates FAILED.** Gate 1 WEAK_PASS, **Gate 2 FAIL**, Gate 3 PASS, **Gate 4 FAIL**. The fairness hypothesis this project set out to test is **not supported at this scale**, and the codec deliverable is independent of it. See [FINAL_REPORT.md](FINAL_REPORT.md) and [CLAIMS_AUDIT.md](CLAIMS_AUDIT.md).

> See [PENDING.md](PENDING.md) for the honest list of known gaps, deferred scope, and heuristic ceilings.
> **Read [gate1_study/GATE1_REPORT.md](gate1_study/GATE1_REPORT.md) and [ALLOCATION_MATH.md](ALLOCATION_MATH.md) before relying on allocation** - Gate 1 came back **WEAK_PASS**, not PASS: fragmentation shows only a small, confound-entangled effect on compression failure at this model scale, so the allocators are framed as engineering controls, not validated causal-fairness claims.

**Completed through Prompt 20 (final release).** Verification: 548 tests pass, `ruff` and `mypy` clean, all deliverables run end-to-end on a real captured Qwen2.5-0.5B cache, Docker image builds and runs the CLI. The Gate-1 200-group causal study was re-run from scratch on the real model and reproduced the committed result exactly (2400 predictions, WEAK_PASS).

> **Gate 2 came back FAIL at pilot scale** ([gate2_fairness_study/GATE2_REPORT.md](gate2_fairness_study/GATE2_REPORT.md)): the aggregate and minimax allocators chose identical per-cohort bit-widths (zero worst-cohort fairness benefit), and the intersection-full-correct isolation retained only low-fragmentation cohorts (the same base-model confound as Gate 1). Minimax is reported as **negative evidence** for the fairness thesis at this scale — the codec is preserved and no fairness claim is fabricated. See RISK_REGISTER R-09.

> **Gate 4 came back FAIL** ([GATE4_REPORT.md](GATE4_REPORT.md)): on a real Qwen2.5-0.5B pilot (5 groups x 2 seeds x 4 fragmentation levels x 2 budgets, matched bits verified in every one of 80 runs), the Module 3 fuzzy repair-priority scorer did not beat no-repair on task accuracy (-0.013 mean gain, 25% directionally consistent) or worst-cohort degradation (-0.050 mean gain — fuzzy made the worst cohort WORSE on average, an overprotection failure mode), and was not distinguishable from its simplest competitors (95% CI on fuzzy-vs-best-simple accuracy: [-0.050, 0.037]). Fuzzy scoring is reported as **negative evidence**; it remains in the codebase as an optional, non-default scorer. See RISK_REGISTER R-10 and CLAIMS_LEDGER C-23.

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

9. **Prompt 7 — LBG Vector Quantization** (`fairfuzzkv_codec.quantization.vector_quant`, `fairfuzzkv_codec.codec.vector_quant.LBGVectorQuantCodec`)
   - Deterministic Linde-Buzo-Gray/k-means codebook training (split perturbation, empty-cluster recovery, mini-batch), head-block and cross-token vector formation, global/per-layer/per-head codebook scopes, calibration-fit leakage guard, chunked nearest-codeword with an optional FAISS path behind a CPU-authoritative interface.
   - Codebook overhead is serialized and fully counted; scalar-vs-LBG matched-total-bit benchmark in [lbg_benchmark/](lbg_benchmark/), which honestly reports small-corpus cases where codebook overhead makes VQ worse.

10. **Prompt 8 — Unified Binary Format, Golomb-Rice Metadata Coding, Streaming Decoder** (`fairfuzzkv_codec.metadata_coding`, `fairfuzzkv_codec.decoder`)
    - FairFuzzKV binary format v1: magic/version/endianness, geometry + tokenizer-hash header, section directory, CRC32 checksums, forward-compatible unknown-section skipping, safe rejection of corrupt/truncated/fuzzed input. Byte-level spec + golden vectors in [FORMAT.md](FORMAT.md).
    - Golomb-Rice retention coding (blockwise-adaptive Rice over sorted gaps) with bitmap and run-length fallbacks chosen by *measured* length; blockwise-Rice integer coders for indices/bit-width maps/cohort IDs; container decodes both scalar and LBG payloads.

11. **Prompt 9 — Pruning Selectors, Attention-Mass Repair, Local Bound** (`fairfuzzkv_codec.pruning`)
    - Recency / top-attention-mass / top-k-score / group-aware selectors (quantization-independent); coherent surface-group retention; attention-mass repair contract enforcing a **local** `p_E^repair ≤ p_E^0 + δ` per audited head/query with full accept/reject logging; local per-head bound validator `‖O−Ô‖₂ ≤ 2·M·p_E` that reports assumption failures instead of manufacturing a pass. **Local head-level only — not an end-to-end guarantee.**

12. **Prompt 10 — Aggregate Rate-Distortion Allocator** (`fairfuzzkv_codec.allocation`) — the Gate-2 control condition
    - Train/val/test-separated per-cohort distortion calibration (scalar + LBG options), exp/monotone distortion curves, exact-DP + greedy water-filling allocator validated by optimality gap, driving the real `ScalarQuantCodec`. Live study in [allocation_study/](allocation_study/).

13. **Prompt 11 — Fairness-Constrained Minimax Allocator** (`fairfuzzkv_codec.allocation.minimax`)
    - Minimizes the worst-cohort distortion at a fixed complete bit budget. Continuous water-filling derived from KKT conditions ([ALLOCATION_MATH.md](ALLOCATION_MATH.md); the optimum equalizes achieved **distortion**, not beta) projected to discrete scalar/LBG choices, with an exact epigraph reference solver. Reports worst/average distortion, the Pareto frontier (cost of fairness), and allocation shifts vs the aggregate control. Frozen setup in [GATE2_CONFIG.md](GATE2_CONFIG.md); live study in [gate2_study/](gate2_study/).

14. **Prompt 13 — Fuzzy Repair-Priority Scorer and Simpler Competitor Suite** (`fairfuzzkv_codec.repair_scoring`) — Module 3
    - A real, inspectable Mamdani fuzzy inference system: triangular membership functions, a documented 10-rule base, min/max aggregation, centroid defuzzification, and per-candidate rule traces — not a neural network renamed "fuzzy" (`fuzzy.py`).
    - Three non-fuzzy competitors on the identical normalized input contract: monotone weighted score, sigmoid (logistic-shaped) score, and knapsack value/cost ratio (`competitors.py`). A calibrated tree/MLP competitor is deliberately skipped — no real repair-outcome labels exist to validate one against without fabricating a result (see PENDING.md).
    - Inputs (fragility, evidence/attention importance, completion cost, staleness, optional uncertainty) are normalized with train-only min-max statistics (`inputs.py`), the same train/apply discipline as `quantization/calibration.py`.
    - Config-selectable ablation registry (`ablation.py`) runs every scorer on identical candidates/budgets; sensitivity analysis over fuzzy breakpoints and rules, plus measured latency/parameter-count complexity per scorer (`sensitivity.py`).
    - Drop-in integration with the **unchanged** Prompt 9 `RepairContract` — any scorer's output proposes a budget-neutral swap (`integration.py`) that the frozen local mass condition still accepts/rejects; the codec is byte-identical if this module is removed.
    - Live demo on synthetic candidates (scores structural signals, not raw KV tensors, so no HF capture is needed): `scripts/run_repair_scoring_demo.py` → [repair_scoring_study/](repair_scoring_study/).

15. **Prompt 14 — Gate 4 Fuzzy-vs-Simple Ablation and Naming Decision** (`fairfuzzkv_codec.evaluation.gate4`) — Gate 4 Decision: **FAIL**
    - Real Qwen2.5-0.5B pilot comparing `{no_repair, fuzzy, monotone, knapsack, logistic}` at 2 budgets (retention ratio 0.3/0.5) x 2 seeds (42, 7) x the 4 `n_g` fragmentation cohorts, using REAL per-position candidate signals (fragility from Module 2, attention mass from a real captured audited head via a `q_proj` forward hook + `output_attentions=True`, surface-group token count, positional staleness) — never synthetic inputs (contrast with Prompt 13's demo). Frozen pre-registered thresholds tested on synthetic fixtures before the real run (`tests/evaluation/test_gate4.py`); frozen setup in [GATE4_CONFIG.md](GATE4_CONFIG.md).
    - `ExplicitMaskCodec` (`codec/explicit_mask.py`) applies a caller-supplied retention mask instead of an internally-computed one, so every scorer's bits/element is matched BY CONSTRUCTION (Prompt 9's repair swaps are budget-neutral) — verified true in all 80 real runs, not assumed.
    - Result: fuzzy did not beat no-repair on task accuracy (-0.013 mean gain, 25% consistent) or worst-cohort degradation (-0.050 mean gain — fuzzy INCREASED the worst cohort's degradation, an overprotection failure mode per item 98), and was statistically indistinguishable from its simplest competitors (95% CI [-0.050, 0.037]). Full report: [GATE4_REPORT.md](GATE4_REPORT.md); raw predictions: [gate4_fairness_study/predictions.jsonl](gate4_fairness_study/predictions.jsonl).
    - **Automatic naming/claims switch** (`core/naming.py`): the frozen decision drives the claim framing and writes [PROJECT_IDENTITY.json](PROJECT_IDENTITY.json) — not a manual follow-up. (The project *name* is deliberately excluded from the switch: it is owner-chosen identity, not evidence, and an earlier auto-rename broke the build.) Two report templates ([GATE4_REPORT_PASS_TEMPLATE.md](GATE4_REPORT_PASS_TEMPLATE.md), [GATE4_REPORT_FAIL_TEMPLATE.md](GATE4_REPORT_FAIL_TEMPLATE.md)) were prepared before the run so either outcome could be completed immediately.

16. **Prompt 15 — IndicLongComp Parallel Multilingual and Code-Mixed Benchmark** (`fairfuzzkv_codec.benchmarks.indic_longcomp`)
    - Four language/code-mix conditions (English, Hindi, Hinglish, Telugu-English) x five task families (retrieval, multi-hop, comparison, counting, evidence aggregation), built from hand-designed parallel templates. **Content is LLM-authored, not sourced from any external corpus (MLRBench or otherwise — no verified network/license access was available) and not professionally translated or reviewed** — stated explicitly in every dataset card, never presented as sourced/reviewed content.
    - "Parallel" used only in the **verified structural sense** per the Prompt 15 non-negotiable: every language variant of a group is rendered from ONE shared random draw (names, digits, evidence position, distractor count), so the canonical answer — always a language-independent single digit — is identical by construction across all 4 languages, checked per-group by `validators.validate_parallelism` (not assumed from translation quality).
    - Full dataset-card discipline: license inventory (100% project-original, no external corpus), automated PII regex scan, exact-duplicate dedup, a real contamination self-check against FragKV-MinPairs' 800 already-committed texts (0 overlaps found), and a checksum split-hash — all with `encoding="utf-8"` explicit on every file write (unlike `fragkv_minpairs.dataset_card`, which crashed on Windows for exactly this reason — fixed as part of this prompt since it became a hard blocker for the contamination check; see RISK_REGISTER).
    - Per-language tokenizer-fragility distributions and quantile-cohort coverage, reusing Module 2's real pipeline unchanged (`fragility_report.py`).
    - Real FullKV baseline run on Qwen2.5-0.5B, intersection-full-correct isolation tagged BEFORE any compression evaluation (`runner.py`). Course subset (10 groups, real FullKV run) + journal subset (250 groups, structurally validated, FullKV run left as documented follow-up given compute budget). Full report: [INDICLONGCOMP_REPORT.md](INDICLONGCOMP_REPORT.md).

17. **Prompt 16 — Full Baseline Matrix and Regime-Separated Evaluation** (`fairfuzzkv_codec.baselines`)
    - Regime-separated result tables (Prompt 16 item 108, never mixed): **compression/quantization**, **prefill-time selection**, **decode-time selection**. Every baseline gets exactly one provenance/configuration card (item 113), reproduced or not.
    - Real, faithful adapters for this project's own codecs (UniformINT8/INT4, `FairFuzzKV-Scalar`, `FairFuzzKV-LBG`, `TopK-L2`) plus core-mechanism (`approximate`) reproductions of **SnapKV** (observation-window attention voting + 1D max-pooling), **PyramidKV** (per-layer pyramid retention budget — required extending `ExplicitMaskCodec` to a 2D per-layer mask), and **H2O** (heavy-hitter ∪ recency union, decode-time regime only). **RateQuant, RDKV, KVTuner, KVmix are explicitly `not_reproduced`** — no verified network/license access to confirm their exact algorithms, each with a stated nearest-faithful-configuration pointer to this project's own real functionality, never silently reimplemented under those names (item 111, the non-negotiable: "reproducibility over having every baseline name in a table").
    - Generalized matched-bit tuner (`adapter.py`) automates verification across BOTH discrete (quantization bit-width) and continuous (retention-ratio, binary-searched) baselines — an unmatched baseline is reported as unmatched, never dropped.
    - Real run on Qwen2.5-0.5B against IndicLongComp's course subset (40 variants): **compression/quantization** — FairFuzzKV-Scalar (MSE 0.907) and FairFuzzKV-LBG (1.508) both clearly beat UniformINT8/INT4 (4.091) at matched ~4.05 bits/element; **prefill-selection** — PyramidKV (MSE 30.9) beat SnapKV (35.9) and TopK-L2 (36.0); **decode-time** — H2O (36.1), kept in its own table, never compared directly against the prefill numbers. All 40/40 matched in every regime. Full report: [BASELINE_MATRIX_REPORT.md](BASELINE_MATRIX_REPORT.md).

18. **Prompt 17 — Cross-Tokenizer and Cross-Model Reproduction** (`fairfuzzkv_codec.evaluation.gate3`) — Gate 3 Decision: **PASS**
    - Reran Gate 1 and Gate 2 on a second, materially different model/tokenizer family — **TinyLlama-1.1B-Chat (SentencePiece)** vs the existing **Qwen2.5-0.5B (byte-level BPE)** — via the SAME frozen scripts (`run_gate1_study.py` / `run_gate2_study.py` gained a `--model` flag, the only change to either), at pilot scale (20 / 16 groups vs the original 200 / 24×6, explicitly reduced given TinyLlama's ~2x slower CPU forward pass).
    - **Real, unexpected finding one layer earlier than planned**: the FragKV-MinPairs numeric rendering ladder (calibrated against Qwen) could not construct ANY valid group under TinyLlama's SentencePiece tokenizer — no digit 0-9 hits both `n_g=4` and `n_g=8` within the frozen tolerance simultaneously (measured: the digit sets that reach each target are disjoint). Required a documented, transparent, tokenizer-specific tolerance widening (`{4:2, 8:2}` instead of `{4:1, 8:1}`) — itself a direct, concrete answer to item 116 ("does this transfer or require recalibration?"), found at dataset-construction time rather than cohort-assignment time. `generator.build_group`/`generate_dataset`/`generate_validated_dataset` and `validators.validate_group` gained an optional `token_count_tolerance` override (default `None` = frozen Qwen behavior, unchanged — regression-verified).
    - **Result**: Gate 1 reproduced in category (WEAK_PASS vs WEAK_PASS) and Gate 2 reproduced in category (FAIL vs FAIL) — **but Gate 2's two FAILs have different root causes** (Qwen: aggregate/minimax picked identical allocations; TinyLlama: matched-bit tolerance violated at pilot scale) — reported explicitly, not glossed over. Cohort transfer verdict: **`model_specific`** (agreement 0.23, well below the 0.7 universal threshold, reusing Module 2's existing `compute_cross_tokenizer_stability` unchanged) — fragility risk bands do NOT transfer between these tokenizer families.
    - Hierarchical/stratified bootstrap (`gate3.hierarchical_bootstrap`) resamples both families AND examples within each family, so a larger family can't dominate the pooled estimate — applied to the real data (TopK50's n_g=1-vs-8 paired effect, pooled across both families' 200 and 20 groups respectively): point **0.0375**, 95% CI **[0.0, 0.09]** — CI includes 0, consistent with Gate 1's own WEAK_PASS framing on both families. Full report: [GATE3_REPORT.md](GATE3_REPORT.md), frozen config: [GATE3_CONFIG.md](GATE3_CONFIG.md).

### Next Steps:
- **Cohort-conditioned fairness (later prompts):** tie the minimax allocator to fragility-derived cohorts / codebooks. Per Gate 1's WEAK_PASS result, do not premise this on "fragmentation causally requires protection" without re-reading `gate1_study/GATE1_REPORT.md`'s caveats first — keep the framing as an engineering control.
- **Repair-priority scorer validation at scale:** Gate 4's real-data comparison (`gate4_fairness_study/`) is pilot-scale (80 pooled cells) and came back FAIL — do not cite the fuzzy scorer as superior to its competitors. A larger, more heterogeneous candidate pool (naturalistic high-fragmentation text, multiple audited heads/layers) is the natural follow-up before writing fuzzy scoring off entirely.
- **IndicLongComp at scale:** the journal subset (250 groups) is generated and structurally validated but has NOT had a real FullKV run — do that next to get a non-empty, adequately-powered isolation subset before running any compression/fairness comparison on this benchmark (the course subset's isolation subset is empty at 0/10 full-correct groups, consistent with small-n noise, not a validated capability finding — see INDICLONGCOMP_REPORT.md).
- **Baseline matrix gaps:** RateQuant/RDKV/KVTuner/KVmix remain unreproduced (see BASELINE_MATRIX_REPORT.md's cards) — do not cite results under those names. Group-aware pruning (Prompt 9) isn't in the automated matrix — it needs text-derived surface-group IDs beyond the common `Callable[[float], BaseCodec]` adapter shape; a text-aware adapter variant is the natural follow-up. SnapKV/PyramidKV/H2O hyperparameter defaults are this implementation's best-effort choices, not verified against reference code.
- **Gate 3 scope deferred:** model-family × allocator and quantizer-type × cohort interaction effects (item 119) were NOT attempted — both need a second full Prompt 10/11 allocator study on TinyLlama, beyond this session's compute budget. Gate 3 itself is pilot-scale on Family B (20/16 groups, 1 budget, 1 seed) — do not treat as a well-powered cross-model study. See GATE3_REPORT.md.

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

# Research dashboard (13 pages, loads frozen artifacts only)
uv run streamlit run dashboard_app.py

# Systems profiling -> systems_profile/ + see PERFORMANCE.md
uv run python scripts/run_systems_profile.py

# Offline demo snapshot (fallback for presenting without the live app)
uv run python scripts/export_demo_assets.py   # -> demo_assets/demo.html
```

### Release documents

| Document | What it is |
|---|---|
| [FINAL_REPORT.md](FINAL_REPORT.md) | Full experimental report: problem, algorithms, derivations, all four gate outcomes, limitations, ethics |
| [CLAIMS_AUDIT.md](CLAIMS_AUDIT.md) | Every claim labelled measured / derived / subset / future |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Fresh-machine guide + 16-point reproducibility checklist |
| [VIVA_PACK.md](VIVA_PACK.md) | 100 Q&A, proof walkthroughs, failure scenarios, demo recovery plan |
| [MODEL_CARD.md](MODEL_CARD.md) | Codec card: intended use, evaluation, limits, ethical caveats |
| [JOURNAL_EXPANSION_PLAN.md](JOURNAL_EXPANSION_PLAN.md) | 4-6 month plan to properly test what the pilot could not |
| [PERFORMANCE.md](PERFORMANCE.md) | Measured latency/memory profile, bottleneck analysis, troubleshooting |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | 9-minute demonstration script |
| [FORMAT.md](FORMAT.md) | FFK1 byte-level format spec with golden vectors |

Verify the release yourself:

```bash
uv run python scripts/run_release_checklist.py --skip-install   # 8 automated checks
uv run python scripts/export_release_package.py                 # checksums, bitstreams, vector figures
shasum -a 256 -c release/CHECKSUMS.sha256                       # verify artifact integrity
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
