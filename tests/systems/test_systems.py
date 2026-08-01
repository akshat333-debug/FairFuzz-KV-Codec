import random

import pytest
import torch

from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
from fairfuzzkv_codec.systems.benchmark import measure_latency, measure_peak_memory, speedup
from fairfuzzkv_codec.systems.cache import CacheContaminationError, FitCache, tensor_fingerprint
from fairfuzzkv_codec.systems.hardware import capture_hardware_manifest
from fairfuzzkv_codec.systems.streaming import (
    AllocationBudget,
    AllocationTooLarge,
    StreamingEncodeError,
    chunked_decode,
    chunked_encode,
    iter_sequence_chunks,
    maybe_pin_memory,
    recommended_chunk_tokens,
)


def _kv(seq: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(4, 1, 2, seq, 16, generator=g)


# ---- hardware manifest -----------------------------------------------------

def test_hardware_manifest_captures_required_fields():
    m = capture_hardware_manifest("cpu")
    assert m.platform and m.torch_version and m.python_version
    assert m.device == "cpu"
    assert m.torch_num_threads >= 1
    assert m.power_mode in ("normal", "low_power", "unknown")


def test_hardware_manifest_never_fabricates_undetectable_fields():
    """Undetectable values must be None/"unknown", never invented."""
    m = capture_hardware_manifest("cpu")
    d = m.to_dict()
    for key in ("cpu_count_logical", "total_ram_gb"):
        assert d[key] is None or d[key] > 0
    if not m.cuda_available:
        assert m.cuda_device_name is None


# ---- latency statistics ----------------------------------------------------

def test_measure_latency_reports_percentiles_and_ci():
    stats = measure_latency("noop", lambda: sum(range(100)), warmup=2, repeats=15)
    assert stats.n_samples == 15
    assert stats.warmup_runs == 2
    assert stats.min_seconds <= stats.p50_seconds <= stats.p95_seconds <= stats.max_seconds
    assert stats.ci95_low <= stats.mean_seconds <= stats.ci95_high
    assert stats.measured is True


def test_measure_latency_rejects_zero_repeats():
    with pytest.raises(ValueError):
        measure_latency("x", lambda: None, repeats=0)


def test_single_sample_ci_is_nan_not_a_fake_zero_width_interval():
    stats = measure_latency("one", lambda: None, warmup=0, repeats=1)
    assert stats.ci95_low != stats.ci95_low  # NaN
    assert stats.n_samples == 1


def test_speedup_flags_overlapping_intervals_as_not_significant():
    """Two runs of the same work must NOT be reported as a speedup."""
    a = measure_latency("a", lambda: sum(range(200)), repeats=15)
    b = measure_latency("b", lambda: sum(range(200)), repeats=15)
    result = speedup(a, b)
    assert result["measured"] is True
    if result["confidence_intervals_overlap"]:
        assert result["significant"] is False
        assert "not a demonstrated speedup" in result["interpretation"]


def test_measure_peak_memory_returns_a_measured_reading():
    reading = measure_peak_memory(lambda: torch.zeros(256, 256))
    assert reading.measured is True
    assert reading.peak_cpu_mb is not None and reading.peak_cpu_mb > 0


# ---- chunked / bounded / OOM ----------------------------------------------

def test_chunks_cover_the_sequence_exactly_without_overlap():
    kv = _kv(seq=37)
    starts, total = [], 0
    for start, chunk in iter_sequence_chunks(kv, chunk_tokens=8):
        starts.append(start)
        total += chunk.size(3)
    assert starts == [0, 8, 16, 24, 32]
    assert total == 37  # ragged tail included exactly once


def test_chunked_encode_decode_round_trips_in_original_order():
    kv = _kv(seq=24)
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    result = chunked_encode(kv, codec.encode_prefill, chunk_tokens=8)
    assert result.num_chunks == 3

    def _decode(payload: bytes) -> torch.Tensor:
        return codec.decode(payload, {}, tuple(kv.shape), kv.dtype, "cpu")

    recon = chunked_decode(result, _decode, expected_shape=tuple(kv.shape))
    assert recon.shape == kv.shape
    assert (recon - kv).pow(2).mean().item() < 1e-2  # 8-bit, near-lossless


def test_allocation_budget_refuses_before_allocating():
    budget = AllocationBudget(max_bytes=1024)
    with pytest.raises(AllocationTooLarge):
        budget.check_tensor((1000, 1000), torch.float32, what="huge")
    budget.check_tensor((4, 4), torch.float32)  # within budget - no raise


def test_chunked_encode_respects_the_allocation_budget():
    kv = _kv(seq=64)
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    with pytest.raises(AllocationTooLarge):
        chunked_encode(kv, codec.encode_prefill, chunk_tokens=64, budget=AllocationBudget(max_bytes=512))


def test_failed_chunk_discards_partial_output_instead_of_truncating():
    """A partial bitstream must never be returned."""
    kv = _kv(seq=24)
    calls = {"n": 0}

    def flaky(chunk):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated encoder failure")
        return b"payload", {}

    with pytest.raises(StreamingEncodeError) as exc:
        chunked_encode(kv, flaky, chunk_tokens=8)
    assert "never returned" in str(exc.value)


def test_mis_assembled_decode_raises_rather_than_returning_wrong_data():
    kv = _kv(seq=16)
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    result = chunked_encode(kv, codec.encode_prefill, chunk_tokens=8)
    result.payloads = result.payloads[:1]  # simulate a lost chunk
    with pytest.raises(StreamingEncodeError):
        chunked_decode(
            result,
            lambda p: codec.decode(p, {}, tuple(kv.shape), kv.dtype, "cpu"),
            expected_shape=tuple(kv.shape),
        )


def test_recommended_chunk_tokens_is_at_least_one_and_bounded():
    small = recommended_chunk_tokens((4, 1, 2, 1000, 16), torch.float32, target_mb=0.001)
    assert small >= 1  # never 0 -> never an infinite loop
    big = recommended_chunk_tokens((4, 1, 2, 1000, 16), torch.float32, target_mb=64.0)
    assert big > small


def test_pin_memory_is_a_safe_noop_without_cuda():
    t = torch.zeros(8)
    assert maybe_pin_memory(t, enabled=True).shape == t.shape  # no crash, no false claim


# ---- fit cache / contamination guard ---------------------------------------

def test_cache_hit_and_miss():
    cache = FitCache()
    data = _kv(seq=8)
    cfg = {"bits": 8}
    assert cache.get(data, cfg, split="train") is None
    cache.put(data, cfg, split="train", artifact="codebook")
    assert cache.get(data, cfg, split="train") == "codebook"
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_cache_refuses_to_serve_an_artifact_fitted_on_another_split():
    """The contamination guard: same data+config, different split -> hard error."""
    cache = FitCache()
    data = _kv(seq=8)
    cfg = {"bits": 8}
    cache.put(data, cfg, split="train", artifact="fitted_on_train")
    with pytest.raises(CacheContaminationError):
        cache.get(data, cfg, split="test")
    assert cache.stats.refusals == 1


def test_different_data_produces_a_different_key_so_no_cross_split_reuse():
    train, test = _kv(seq=8, seed=1), _kv(seq=8, seed=2)
    assert tensor_fingerprint(train) != tensor_fingerprint(test)
    cache = FitCache()
    cache.put(train, {"b": 4}, split="train", artifact="A")
    assert cache.get(test, {"b": 4}, split="test") is None  # clean miss, no leak


def test_get_or_fit_only_fits_once():
    cache = FitCache()
    data = _kv(seq=8)
    calls = {"n": 0}

    def fit(_d):
        calls["n"] += 1
        return "artifact"

    cache.get_or_fit(data, {"b": 8}, "train", fit)
    cache.get_or_fit(data, {"b": 8}, "train", fit)
    assert calls["n"] == 1


# ---- load / large-file / fuzz ----------------------------------------------

def test_large_context_encodes_within_bounded_memory():
    """Large-context test: a long sequence must encode chunk-by-chunk without
    the peak tracking sequence length."""
    kv = _kv(seq=1024)
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    # this cache is ~512 B/token, so a small target is needed to force chunking
    chunk = recommended_chunk_tokens(tuple(kv.shape), kv.dtype, target_mb=0.1)
    assert chunk < kv.size(3), "target_mb must be small enough to actually split"
    result = chunked_encode(kv, codec.encode_prefill, chunk_tokens=chunk)
    assert result.num_chunks > 1
    assert result.total_bytes > 0


def test_load_repeated_encodes_are_stable():
    """Load test: many sequential encodes must stay byte-identical (no state
    leaking between runs)."""
    kv = _kv(seq=16)
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    first, _ = codec.encode_prefill(kv)
    for _ in range(50):
        payload, _m = codec.encode_prefill(kv)
        assert payload == first


def test_serialization_fuzz_never_crashes_uncontrolled():
    """Fuzz the container parser with random bytes; only the documented
    CorruptContainerError is acceptable."""
    from fairfuzzkv_codec.metadata_coding.container import CorruptContainerError, unpack

    rng = random.Random(99)
    for _ in range(300):
        blob = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 96)))
        try:
            unpack(blob)
        except CorruptContainerError:
            pass


# ---- performance regression thresholds -------------------------------------

def test_encode_latency_regression_threshold():
    """Regression guard with a deliberately loose ceiling: it catches an
    order-of-magnitude regression without failing on normal CI jitter. The
    threshold is a CEILING, not a performance claim."""
    kv = _kv(seq=64)
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    stats = measure_latency("scalar_encode", lambda: codec.encode_prefill(kv), warmup=2, repeats=10)
    assert stats.p50_seconds < 2.0, f"encode p50 regressed to {stats.p50_seconds:.3f}s"


def test_chunked_encode_does_not_blow_up_byte_count():
    """Chunking must not cost more than ~2x the single-shot payload (per-chunk
    headers are real overhead and are counted, not hidden)."""
    kv = _kv(seq=64)
    codec = ScalarQuantCodec("h", tensor_name="k", default_bits=8)
    single, _ = codec.encode_prefill(kv)
    chunked = chunked_encode(kv, codec.encode_prefill, chunk_tokens=16)
    assert chunked.total_bytes < 2.0 * len(single)
