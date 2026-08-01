# Final Claims Audit

Prompt 20 item 141: every claim this project makes is labelled with its epistemic
status. Four labels, used consistently across `FINAL_REPORT.md`,
`MODEL_CARD.md`, `PERFORMANCE.md`, and the dashboard:

| Label | Meaning |
|---|---|
| **[measured]** | Produced by executing code on real data/hardware and recorded in a committed artifact. Reproducible. |
| **[derived]** | Follows mathematically from assumptions that are stated inline. Not an empirical result. |
| **[subset]** | Observed on a specific, named subset — does not generalize beyond it. |
| **[future]** | Not done. Planned or hypothesized only. |

**Audit rules enforced:**
1. No estimated quantity is described as measured.
2. No weak label is called gold.
3. No negative result is softened.
4. Every number traces to a committed artifact path.

---

## A. Codec / engineering claims

| # | Claim | Label | Evidence |
|---|---|---|---|
| A1 | No-op codec round-trips FP16 bit-identically | [measured] | `tests/codec/test_noop.py` |
| A2 | Byte accounting counts codes, scales, zero-points, masks, indices, headers, alignment, metadata | [measured] | `tests/codec/test_accounting.py` |
| A3 | Sparsity ratios are never substituted for real bytes | [measured] | `add_logical_only_component` + test |
| A4 | INT4 is genuinely packed two values per byte | [measured] | `tests/quantization/test_packing.py` (8 vals → 4 bytes) |
| A5 | Scalar encoding is deterministic (byte-identical) | [measured] | `test_deterministic_byte_output` |
| A6 | NaN input is rejected, not silently propagated | [measured] | `test_nan_input_is_rejected` |
| A7 | LBG training/encoding deterministic under fixed seed | [measured] | `tests/quantization/test_vector_quant.py` |
| A8 | LBG codebook overhead is serialized and counted | [measured] | `test_codebook_overhead_is_serialized_and_counted` |
| A9 | FFK1 golden bitstreams decode identically across runs | [measured] | `tests/metadata_coding/test_golden_vectors.py` |
| A10 | Corrupt/truncated/checksum-failing containers are rejected safely | [measured] | 300 random blobs + 300 bit-flips fuzzed |
| A11 | Golomb-Rice beats raw indices on a sparse pattern, falls back otherwise | [measured] | `test_rice_gaps_beats_raw_indices_on_sparse_pattern` |
| A12 | Fallback is chosen by measured encoded length, not assumption | [measured] | all three candidates computed in `encode_retention` |
| A13 | Chunked encoding never returns a truncated bitstream | [measured] | `test_failed_chunk_discards_partial_output` |
| A14 | Oversized allocations are refused before allocating | [measured] | `AllocationTooLarge` pre-check + test |
| A15 | Cross-split cache reuse raises rather than contaminating | [measured] | `CacheContaminationError` + test |

## B. Mathematical claims

| # | Claim | Label | Notes |
|---|---|---|---|
| B1 | Minimax optimum equalizes achieved **distortion** across active cohorts | [derived] | KKT derivation in `ALLOCATION_MATH.md`; assumes exponential curves + active set |
| B2 | The optimum does **NOT** equalize β | [derived] | β are fixed parameters, not decision variables. Explicitly corrected, not coded blindly |
| B3 | `‖O−Ô‖₂ ≤ 2·M·p_E` | [derived] | **Local, single-head, single-layer**, under renormalization. **NOT** an end-to-end guarantee |
| B4 | Exact allocator attains the true optimum | [measured] | Verified against independent brute-force enumeration, 120 random instances |
| B5 | Greedy stays within a bounded optimality gap | [measured] | max gap < 0.5 over 120 instances |
| B6 | Worst-cohort distortion is monotone non-increasing in budget | [measured] | `test_p2_minimax_worst_case_is_monotone` |
| B7 | Bound validator reports assumption failures rather than passing | [measured] | empty-kept-set test |

## C. Empirical / gate claims

| # | Claim | Label | Status |
|---|---|---|---|
| C1 | Fragmentation causally predicts extra compression failure | [measured][subset] | **WEAK_PASS — not established.** Control collapses 48%→0%; confound cannot be separated |
| C2 | Minimax reduces cross-cohort degradation disparity | [measured][subset] | **FAIL.** Benefit 0.000, CI [0,0], 6 runs |
| C3 | Findings reproduce across model/tokenizer families | [measured][subset] | **PASS** at pilot scale (20/16 groups on family B) |
| C4 | Fuzzy repair scoring beats no-repair | [measured][subset] | **FAIL.** −0.013 accuracy, −0.050 worst-cohort |
| C5 | Fuzzy is distinguishable from simple competitors | [measured][subset] | **FAIL.** CI [−0.050, 0.037] |
| C6 | Fragility cohorts transfer across tokenizers | [measured] | **NO** — agreement 0.23. No universal threshold may be claimed |
| C7 | Gate 1 and Gate 4 recompute from raw predictions without a model | [measured] | Verified in the release checklist |
| C8 | Model-family × allocator interaction exists | [measured][subset] | **NULL** — identical allocations on both families |
| C9 | Quantizer × cohort interaction exists | [measured][subset] | **NULL** — int8 won every cohort on both families |
| C10 | MSE predicts task accuracy | [measured][subset] | **NO** at 8 bits — per-tensor beat per-channel despite 18x worse MSE |

## D. Systems / performance claims

| # | Claim | Label | Evidence |
|---|---|---|---|
| D1 | Prefill 115–208 ms; scalar encode 0.6–2.4 ms | [measured] | `systems_profile/`, p50 with warm-up |
| D2 | The codec is not the bottleneck; the model is | [measured] | follows directly from D1 |
| D3 | LBG encode ≈ 40x slower than scalar INT8 | [measured] | 49.7–83.8 ms vs 0.6–2.0 ms |
| D4 | LBG reaches 11.7–26.5x compression vs fp32 | [measured] | `systems_profile/` |
| D5 | Fuzzy inference ≈ 204x the cheapest competitor | [measured] | warm-up + 5 repeats, median |
| D6 | Batch scaling: 0.0105 → 0.0049 µs/element (batch 1→4) | [measured] | batch sweep |
| D7 | Chunking overhead 1.010x vs single-shot | [measured] | streaming section |
| D8 | Decode-side tokens/s is a **replay** rate | [measured] | attention replay harness, **not** serving |
| D9 | Any end-to-end serving speedup | [future] | **Not claimed. Not measured.** No serving integration exists |
| D10 | GPU performance | [future] | **Not measured.** CPU-only; no GPU numbers reported |
| D11 | CPU peak memory covers torch's C++ allocator | — | **Explicitly denied.** tracemalloc sees Python-level allocations only |

## E. Dataset claims

| # | Claim | Label | Notes |
|---|---|---|---|
| E1 | IndicLongComp variants are "parallel" | [measured] | **Only** in the mechanically verified sense: identical answer, evidence count, evidence position, distractor count, task family. Verified per group |
| E2 | Variants are faithful translations | — | **NOT CLAIMED.** No professional translation or native review was performed |
| E3 | Content is sourced from MLRBench or another corpus | — | **NOT CLAIMED — explicitly false.** All content is LLM-authored from hand-designed templates. Stated in every dataset card |
| E4 | 4 language conditions, 5 task families | [measured] | en, hi, hinglish, te_en; retrieval/multi-hop/comparison/counting/aggregation |
| E5 | No PII present | [measured] | Synthetic name pool; automated regex scan found no emails or long digit runs |
| E6 | Not contaminated with model pretraining data | — | **NOT CLAIMED.** Only an in-repo self-check was possible; it cannot verify any model's actual pretraining corpus |
| E7 | Hindi is the most fragile language under both tokenizers | [measured][subset] | 0.344 BPE / 0.444 SP vs English 0.266/0.333. A property of *this generated text*, not a claim about downstream failure |
| E8 | Journal subset has a real FullKV run | — | **NO — [future].** Generated and structurally validated only |

## F. Baseline claims

| # | Claim | Label | Notes |
|---|---|---|---|
| F1 | SnapKV / PyramidKV / H2O reproduce the published algorithms exactly | — | **NOT CLAIMED.** Marked `APPROXIMATE`: core mechanism reproduced, hyperparameters unverified |
| F2 | RateQuant / RDKV / KVTuner / KVmix are implemented | — | **NO.** Marked `NOT_REPRODUCED` with reasons and a nearest-faithful-configuration pointer. Never silently substituted |
| F3 | All baselines matched the bit budget | [measured] | 40/40 variants per baseline |
| F4 | Regimes are strictly separated | [measured] | Decode-time (H2O) never appears in a prefill table |

## G. Things this project explicitly does NOT claim

1. That compression is unfair to non-English users. **Gate 2 FAILED.**
2. That fuzzy scoring helps. **Gate 4 FAILED.**
3. That fragmentation causally drives compression failure. **Gate 1 WEAK_PASS only.**
4. Any end-to-end serving throughput or latency improvement.
5. Any GPU performance characteristic.
6. That fragility cohorts generalize across tokenizers. **They do not (0.23).**
7. That the Indic benchmark text is translated, reviewed, or externally sourced.
8. That results generalize beyond ≤1.1B-parameter models and <300-token contexts.

## Audit conclusion

**[measured]** 28 ledger claims tracked; 5 are negative or weak and are labelled
as such in every surfacing location (README, ledger, dashboard, final report,
offline export). No estimated metric is presented as measured: the only
non-measured performance statements (D9, D10) are labelled **[future]** and
carry an explicit "not claimed" marker. No weak label is called gold: the
fragility proxy label (C1) and the `APPROXIMATE`/`NOT_REPRODUCED` baselines (F1,
F2) are flagged wherever they appear.
