# Performance Profile, Bottleneck Analysis, and Troubleshooting Guide

All numbers below are **measured** on the hardware manifest recorded in
`systems_profile/systems_profile.json` — warm-up runs, synchronization, 10
repeats, median (p50) and p95 reported, bootstrap 95% CIs stored alongside.
Nothing here is inferred, extrapolated, or projected. Where a number is a
replay-harness rate rather than a serving throughput, it says so.

Regenerate with:

```bash
uv run python scripts/run_systems_profile.py
```

## Integration boundary (read this before quoting any number)

| Phase | What is actually integrated |
|---|---|
| **Prefill** | **Real.** A genuine Hugging Face forward pass on `Qwen/Qwen2.5-0.5B` produces the KV cache that is encoded and decoded. Prefill latency is measured on that real model. |
| **Decode** | **Attention replay harness only.** Decode-side consequences are measured by replaying attention over the reconstructed cache (`benchmarks/attention_harness`), **not** by swapping this codec into vLLM/TGI/HF generate. |

Consequently **"tokens/s" below is a replay rate, not a serving throughput.**
This project does not claim an end-to-end serving speedup, and
`systems/benchmark.speedup()` refuses to call a ratio "significant" when the
two confidence intervals overlap.

## Measured results

Hardware: macOS arm64, 4 torch threads, power mode `normal`, CPU device.

### Latency and real serialized size by context length

Encode/decode p50, and **actual serialized bytes** (not a theoretical bit count):

| Context (seq) | Codec | Encode p50 | Decode p50 | Serialized bytes | Ratio vs fp32 |
|---|---|---|---|---|---|
| 13 | scalar_int8 | 0.60 ms | 0.22 ms | 41,303 | 3.87x |
| 13 | scalar_int4 | 1.00 ms | 0.99 ms | 21,336 | 7.49x |
| 13 | lbg_vd8_cb256 | 49.65 ms | 0.21 ms | 13,679 | 11.68x |
| 44 | scalar_int8 | 1.26 ms | 0.60 ms | 136,537 | 3.96x |
| 44 | scalar_int4 | 1.64 ms | 1.37 ms | 68,954 | 7.84x |
| 44 | lbg_vd8_cb256 | 60.27 ms | 0.30 ms | 25,585 | 21.13x |
| 109 | scalar_int8 | 1.97 ms | 0.82 ms | 336,220 | 3.98x |
| 109 | scalar_int4 | 2.43 ms | 1.81 ms | 168,797 | 7.93x |
| 109 | lbg_vd8_cb256 | 83.84 ms | 0.72 ms | 50,547 | 26.50x |

### Prefill and attention replay

| Context (seq) | Prefill p50 | Prefill p95 | Replay p50 | Replay rate |
|---|---|---|---|---|
| 13 | 114.7 ms | 160.2 ms | 0.30 ms | ~43.9k tok/s |
| 44 | 137.3 ms | 147.9 ms | 0.58 ms | ~75.6k tok/s |
| 109 | 207.6 ms | 208.1 ms | 1.26 ms | ~86.8k tok/s |

### Batch scaling (scalar INT8)

| Batch | Elements | Encode p50 | µs/element |
|---|---|---|---|
| 1 | 135,168 | 1.41 ms | 0.0105 |
| 2 | 270,336 | 1.72 ms | 0.0064 |
| 4 | 540,672 | 2.64 ms | 0.0049 |

### Streaming

2 chunks of 22 tokens: **1.010x** the single-shot payload — per-chunk header
overhead is real, counted, and small.

## Honest bottleneck analysis

1. **The codec is not the bottleneck; the model is.** Prefill costs 115–208 ms
   while scalar encode costs 0.6–2.4 ms — roughly **two orders of magnitude**
   apart. Any claim that this codec meaningfully changes end-to-end latency on
   this setup would be false. Its value is **memory footprint**, not speed.

2. **LBG buys compression with encode time.** LBG reaches 11.7–26.5x vs fp32
   where scalar INT8 reaches ~4x, but costs 50–84 ms to encode — **~40x
   slower** than scalar INT8, because codebook training (Lloyd iterations) runs
   at encode time. LBG *decode* is as fast as scalar (0.2–0.7 ms): it is a
   table lookup. So LBG suits write-once/read-many caches and is a poor fit for
   latency-sensitive per-request encoding.

3. **INT4 decode is slower than INT8 decode** (1.81 ms vs 0.82 ms at seq=109)
   because nibble unpacking is real work. Genuine 4-bit packing is not free —
   it trades decode CPU for bytes.

4. **Batching amortizes fixed overhead well**: 0.0105 → 0.0049 µs/element from
   batch 1 → 4, i.e. per-element cost roughly halves. Fixed per-call costs
   (metadata construction, serialization headers) dominate at batch 1.

5. **Compression ratio improves with context length** (3.87x → 3.98x for INT8,
   11.68x → 26.50x for LBG) because fixed overhead — headers, scales,
   codebooks — amortizes over more tokens. Short contexts are the worst case,
   which is exactly what `lbg_benchmark/` already reported for tiny corpora.

## Troubleshooting guide

| Symptom | Likely cause | What to do |
|---|---|---|
| Encode much slower than this table | LBG codebook training on every call | Reuse a fitted codebook via `systems.cache.FitCache`, or call `LBGVectorQuantCodec.fit()` once on calibration data and encode many caches with it |
| `AllocationTooLarge` raised | Your `AllocationBudget` ceiling is below one chunk | Lower `chunk_tokens` (see `recommended_chunk_tokens`) or raise the ceiling deliberately — the guard fired *before* allocating, which is the intended behavior |
| `StreamingEncodeError` mid-encode | A chunk failed; all partial output was discarded | Check the wrapped cause in the message. A truncated bitstream is never returned, so nothing on disk is corrupt |
| `CacheContaminationError` | A cached fit from one split was requested for another | This is the leakage guard doing its job. Fit separately per split; do not bypass it |
| Latency numbers noisy / non-reproducible | No warm-up, or a throttled machine | Use `systems.benchmark.measure_latency` (warm-up + repeats + percentiles) and check `power_mode` in the hardware manifest — `low_power` numbers are not comparable to `normal` |
| p95 ≫ p50 | GC or thermal interference | Increase `repeats`; `measure_latency` already collects garbage between timed runs. Report p95 too, don't hide it |
| Compression ratio worse than expected | Short context — fixed overhead dominates | Expected. See finding 5 and `lbg_benchmark/lbg_vs_scalar.json`, which reports small-corpus cases where VQ is *worse* than scalar |
| CUDA timings look impossibly fast | Missing synchronization | `measure_latency(..., device="cuda")` synchronizes; hand-rolled timers usually don't |

## Limits and what is NOT claimed

- No end-to-end serving-engine integration exists. Decode figures are replay.
- CPU-only measurements on one machine; no GPU numbers are reported because
  none were measured (rather than estimated).
- `measure_peak_memory` CPU readings come from `tracemalloc`, which sees
  Python-level allocations only — **not** torch's C++ allocator. GPU peak (when
  present) uses torch's own counter and is authoritative.
- Regression thresholds in `tests/systems/test_systems.py` are deliberately
  loose ceilings to catch order-of-magnitude regressions in CI. They are
  guardrails, **not** performance claims.
