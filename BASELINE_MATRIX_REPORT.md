# Baseline Matrix Report (Prompt 16)

Real Qwen2.5-0.5B run against IndicLongComp's course subset (10 groups x 4
languages = 40 variants). Target: 4.0 bits/element, 15% tolerance.
Matched-bit tuning is automated per baseline (`baselines.adapter.tune_to_matched_bits`)
- an unmatched baseline is reported AS unmatched, never silently excluded.

Regimes are kept STRICTLY separate per Prompt 16 item 108 - decode-time
results (H2O) never appear in a prefill-selection table, and vice versa.

## Result tables

### Compression / Quantization

| Baseline | Matched | Mean bits/element | Mean KV MSE | Mean encode (s) |
|---|---|---|---|---|
| FairFuzzKV-LBG | 40/40 | 4.090 | 1.5076 | 0.9123 |
| FairFuzzKV-Scalar | 40/40 | 4.048 | 0.9072 | 0.0055 |
| UniformINT8/INT4 | 40/40 | 4.011 | 4.0913 | 0.0016 |

### Prefill-Time Selection

| Baseline | Matched | Mean bits/element | Mean KV MSE | Mean encode (s) |
|---|---|---|---|---|
| PyramidKV | 40/40 | 4.000 | 30.9084 | 0.0012 |
| SnapKV | 40/40 | 4.012 | 35.9477 | 0.0011 |
| TopK-L2 | 40/40 | 4.000 | 36.0463 | 0.0017 |

### Decode-Time Selection

| Baseline | Matched | Mean bits/element | Mean KV MSE | Mean encode (s) |
|---|---|---|---|---|
| H2O | 40/40 | 3.980 | 36.0754 | 0.0012 |

## Baseline cards (provenance/configuration)

Every baseline - reproduced or not - has exactly one card (Prompt 16 item 113).

| Name | Regime | Status | Deviations | Nearest faithful config |
|---|---|---|---|---|
| FairFuzzKV-LBG | compression_quantization | faithful | Fixed (vector_dim, codebook_size) rather than dynamically tuned per target bits/element - LBG's discrete grid makes exact tuning impractical; reported bits/element may miss the run's target and is marked unmatched rather than hidden. | - |
| FairFuzzKV-Scalar | compression_quantization | faithful | - | - |
| KVTuner | compression_quantization | not_reproduced | Not implemented under this name. | This project's own `BitWidthMap` sparse per-layer/per-head bit-width override mechanism (Prompt 6, `ScalarQuantCodec`) already supports mixed-precision configuration search; it is the nearest REAL functionality in this codebase, not a verified reproduction of KVTuner's specific search algorithm. |
| KVmix | compression_quantization | not_reproduced | Not implemented under this name. | This project's `ScalarQuantCodec` already supports independent K/V bit-width configuration (Prompt 6) - the nearest real functionality, not a verified reproduction of KVmix's specific policy for choosing those bit-widths. |
| RDKV | compression_quantization | not_reproduced | Not implemented under this name. | This project's rate-distortion allocator (Prompt 10) plus `ScalarQuantCodec` covers the general rate-distortion-vs-KV-quantization design space this name likely refers to, but again is not a verified reproduction of a specific published RDKV method. |
| RateQuant | compression_quantization | not_reproduced | Not implemented under this name. | This project's own Prompt 10 aggregate rate-distortion allocator (`fairfuzzkv_codec.allocation`) solves the conceptually similar problem (minimize distortion subject to a total bit budget) and is a REAL, already-benchmarked deliverable (see allocation_study/) - but it is this project's own method, not a reproduction of a paper named RateQuant, and must not be relabeled as such. |
| UniformINT8/INT4 | compression_quantization | faithful | - | - |
| H2O | decode_time_selection | approximate | heavy_ratio/recent_ratio scaled from the single target retention ratio (0.7x / 0.5x split) rather than the original's own budget-split hyperparameter, which was not verified against reference code. | - |
| PyramidKV | prefill_selection | approximate | pyramid_ratio=2.0 (linear interpolation of per-layer multiplier) is this implementation's best-effort default, not verified against the original reference code's exact schedule. | - |
| SnapKV | prefill_selection | approximate | observation_window=16, pooling_kernel=5 are this implementation's best-effort defaults, not verified against the original reference code. | - |
| TopK-L2 | prefill_selection | faithful | - | - |

## Non-negotiable compliance

- **Reproducibility over completeness**: RateQuant, RDKV, KVTuner, KVmix are
  explicitly `not_reproduced` with a stated reason (no verified network/
  license access to the original paper or reference code in this
  environment) and a nearest-faithful-configuration pointer to this
  project's own real functionality - never silently reimplemented under
  those names.
- SnapKV, PyramidKV, H2O are marked `approximate`: their DEFINING mechanism
  (observation-window voting + pooling; per-layer pyramid budget; heavy-
  hitter + recency union) is reproduced, but specific hyperparameter
  defaults were not verified against reference code - see each card's
  `deviations` field.
- UniformINT8/INT4, FairFuzzKV-Scalar, FairFuzzKV-LBG, TopK-L2 are this
  project's own already-real, already-tested codecs (Prompts 2/6/7) -
  `faithful` by definition, not literature reproductions.

## Raw data

`baseline_matrix_study/raw_results.jsonl` - one row per (variant, baseline)
matched-bit comparison. Regenerable via `scripts/run_baseline_matrix.py`.
