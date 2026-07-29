"""Common baseline-adapter API (Prompt 16 item 109) and a generalized
matched-bit tuner (item 110 / acceptance gate "matched-bit verification is
automated") that works across BOTH regimes - quantization codecs (knob =
integer bit-width, checked not searched) and selection codecs (knob =
continuous retention ratio, binary-searched) - reusing the SAME refuse-if-
out-of-tolerance discipline `evaluation.matched_bit.MatchedBitEvaluator`
already established for TopK/UniformQuant, generalized to any codec factory.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from fairfuzzkv_codec.baselines.schema import AdapterResult, BaselineCard, LatencyMeasurement
from fairfuzzkv_codec.codec.base import BaseCodec


@dataclass
class BaselineAdapter:
    """Wraps a `BaseCodec` factory with the phase/limitation metadata every
    baseline needs (item 109: encoded bytes + reconstruction come from the
    codec itself; phase/latency/limitations live here)."""

    card: BaselineCard
    codec_factory: Callable[[float], BaseCodec]  # knob (bits/element target OR retention ratio) -> codec
    is_discrete: bool = False  # True for quantization codecs (integer bit-width knob), False for continuous-ratio selection codecs


def _measure_bits_and_mse(codec: BaseCodec, kv_cache: torch.Tensor) -> tuple:
    byte_stream, meta = codec.encode_prefill(kv_cache)
    # shape metadata key differs across this project's codecs ("kv_shape" vs
    # "full_shape") - the original tensor's own shape is authoritative and
    # every codec's decode() takes it as an explicit param regardless.
    recon = codec.decode(byte_stream, meta, tuple(kv_cache.shape), kv_cache.dtype, "cpu")
    bits_per_element = meta["accountant_report"]["logical_bits"] / kv_cache.numel()
    mse = torch.nn.functional.mse_loss(recon.float(), kv_cache.float()).item()
    return bits_per_element, mse, byte_stream, meta, recon


def tune_to_matched_bits(
    adapter: BaselineAdapter, kv_cache: torch.Tensor, target_bits_per_element: float, tolerance: float = 0.1,
) -> Optional[AdapterResult]:
    """Binary-searches `adapter.codec_factory`'s knob (or checks the single
    discrete value it represents) until the codec's ACTUAL measured
    bits/element is within `tolerance` of the target. Returns None - not a
    fabricated result - if no knob value gets close enough; the caller marks
    that baseline as unmatched in the result table rather than silently
    dropping it."""
    if adapter.is_discrete:
        codec = adapter.codec_factory(target_bits_per_element)
        bits, mse, *_ = _measure_bits_and_mse(codec, kv_cache)
        matched = abs(bits - target_bits_per_element) / max(target_bits_per_element, 1e-9) <= tolerance
        start = time.perf_counter()
        codec.encode_prefill(kv_cache)
        encode_s = time.perf_counter() - start
        return AdapterResult(
            baseline_name=adapter.card.name, regime=adapter.card.regime,
            target_bits_per_element=target_bits_per_element, actual_bits_per_element=bits, matched=matched,
            kv_mse=mse, latency=LatencyMeasurement(encode_seconds=encode_s, decode_seconds=0.0),
        )

    low, high = 0.01, 1.0
    best = None
    for _ in range(12):
        mid = (low + high) / 2
        codec = adapter.codec_factory(mid)
        bits, mse, *_ = _measure_bits_and_mse(codec, kv_cache)
        diff = abs(bits - target_bits_per_element)
        if best is None or diff < best[0]:
            best = (diff, mid, bits, mse)
        if bits > target_bits_per_element:
            high = mid
        else:
            low = mid

    if best is None:
        return None
    _, knob, bits, mse = best
    matched = abs(bits - target_bits_per_element) / max(target_bits_per_element, 1e-9) <= tolerance
    codec = adapter.codec_factory(knob)
    start = time.perf_counter()
    codec.encode_prefill(kv_cache)
    encode_s = time.perf_counter() - start
    return AdapterResult(
        baseline_name=adapter.card.name, regime=adapter.card.regime,
        target_bits_per_element=target_bits_per_element, actual_bits_per_element=bits, matched=matched,
        kv_mse=mse, latency=LatencyMeasurement(encode_seconds=encode_s, decode_seconds=0.0), extra={"tuned_knob": knob},
    )


def run_matched_bit_comparison(
    adapters: List[BaselineAdapter], kv_cache: torch.Tensor, target_bits_per_element: float, tolerance: float = 0.1,
) -> List[AdapterResult]:
    """Runs every adapter at the SAME target and returns one AdapterResult
    per adapter (never silently skipping a baseline that fails to match -
    unmatched/failed baselines are still returned with matched=False so the
    caller can report them transparently, per the acceptance gate)."""
    results = []
    for adapter in adapters:
        result = tune_to_matched_bits(adapter, kv_cache, target_bits_per_element, tolerance)
        if result is None:
            result = AdapterResult(
                baseline_name=adapter.card.name, regime=adapter.card.regime,
                target_bits_per_element=target_bits_per_element, actual_bits_per_element=float("nan"),
                matched=False, kv_mse=float("nan"), limitations_triggered="matched-bit tuning failed to converge",
            )
        results.append(result)
    return results
