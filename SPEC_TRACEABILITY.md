# Specification Traceability

This document maps the frozen specification components to concrete code paths.

## Modules
1. **Cache Capture**: `fairfuzzkv_codec.cache_capture`
2. **Unicode Grouping**: `fairfuzzkv_codec.unicode_grouping` (GroupMapper: grapheme segmentation, surface units, tokenizer alignment, quarantine, versioned audit report). Tested in `tests/unicode_grouping/`.
3. **Fragility Estimation**: `fairfuzzkv_codec.fragility_estimation` (per-group features, transparent monotone risk score as audit baseline, calibrated logistic/tree models validated against transparent baseline on held-out data, quantile risk cohorts with min-sample merge, cross-tokenizer stability, leakage enforcement). Tested in `tests/fragility_estimation/`.
4. **Pruning**: `fairfuzzkv_codec.pruning`
5. **Quantization**: `fairfuzzkv_codec.quantization` + `fairfuzzkv_codec.codec.scalar_quant.ScalarQuantCodec` (symmetric/asymmetric INT8/INT4, per-tensor/per-head/per-channel/groupwise granularity, percentile/MSE-optimal clipping, calibration-set selection, saturation diagnostics, genuine INT4 nibble packing, mixed K/V and per-layer bit-width configuration via `BitWidthMap`). Tested in `tests/quantization/` and `tests/codec/test_scalar_quant.py` (58 tests). Real rate-distortion benchmark: `quantization_benchmark/` (K/V MSE and task-accuracy curves, Qwen2.5-0.5B).
6. **Allocation**: `fairfuzzkv_codec.allocation`
7. **Metadata Coding**: `fairfuzzkv_codec.metadata_coding`
8. **Decoder/Reconstruction**: `fairfuzzkv_codec.decoder`
9. **FragKV-MinPairs / Gate 1 Causal Test**: `fairfuzzkv_codec.benchmarks.fragkv_minpairs` (dataset generator, validators, immutable split hash, pre-registered Gate 1 decision logic, real-model runner via KV-cache splicing). Tested in `tests/benchmarks/fragkv_minpairs/`. Real 200-group study result: `gate1_study/GATE1_REPORT.md` (**WEAK_PASS** - see RISK_REGISTER R-06 and CLAIMS_LEDGER C-11).

## Propositions
- **Proposition 1 (Fragility Distribution)**: Tested in `tests/eval/test_prop1_fragility.py` (Pending)
- **Proposition 2 (Allocation Optimality)**: Tested in `tests/eval/test_prop2_allocation.py` (Pending)

## Datasets
- **Dataset 1 (LongBench)**: Not yet integrated. Planned path: `fairfuzzkv_codec.benchmarks.longbench` (Pending)
- **Dataset 2 (PG-19)**: Not yet integrated. Planned path: `fairfuzzkv_codec.benchmarks.pg19` (Pending)

## Metric Families
- **Compression Efficiency**: Measured via exact byte accounting (not sparsity masks). Implemented in `fairfuzzkv_codec.codec.binary_serializer.ByteAccountant`, tested in `tests/codec/test_accounting.py`.
- **Reconstruction Error**: L2/MSE distortion via attention-equivalence check. Implemented in `fairfuzzkv_codec.benchmarks.attention_harness.AttentionVerificationHarness`, tested in `tests/cache_capture/test_hf_capture.py`.
- **Downstream Task Accuracy**: Exact match, F1 on real benchmark datasets. Not yet implemented — requires Dataset 1/2 integration above. Planned path: `fairfuzzkv_codec.evaluation.downstream` (Pending)
- **Throughput/Latency**: Prefill vs Decode speeds. Implemented in `fairfuzzkv_codec.evaluation.profiler.TelemetryTracker`.

## Go/No-Go Gates
See `EXECUTION_GATES.md` for the gates and their corresponding validation scripts.
