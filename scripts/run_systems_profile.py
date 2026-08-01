"""Systems profiling on a real captured KV cache (Prompt 18).

Integration boundary, stated up front (item 121):
  * PREFILL is real. A real Hugging Face forward pass produces the KV cache
    that is encoded/decoded here, and prefill latency is measured on that real
    model.
  * DECODE consequences are measured through the ATTENTION REPLAY harness
    (`benchmarks/attention_harness.AttentionVerificationHarness`), NOT by
    swapping this codec into a production serving engine's attention kernel.
    So "decode tokens/s" below is the rate of the replay loop over a
    reconstructed cache, not an end-to-end vLLM/TGI serving number, and it is
    labelled that way everywhere. No inferred or extrapolated speedup is
    reported anywhere in this script.

Everything reported is measured: p50/p95 with warm-up, synchronization, and
bootstrap CIs, alongside a hardware manifest.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from fairfuzzkv_codec.benchmarks.attention_harness import AttentionVerificationHarness  # noqa: E402
from fairfuzzkv_codec.cache_capture.hf_capture import HFCapture  # noqa: E402
from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec  # noqa: E402
from fairfuzzkv_codec.codec.vector_quant import LBGVectorQuantCodec  # noqa: E402
from fairfuzzkv_codec.core.config import LayerHeadSelection  # noqa: E402
from fairfuzzkv_codec.systems.benchmark import measure_latency, measure_peak_memory  # noqa: E402
from fairfuzzkv_codec.systems.hardware import capture_hardware_manifest  # noqa: E402
from fairfuzzkv_codec.systems.streaming import chunked_encode, recommended_chunk_tokens  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cpu"
CONTEXT_WORDS = [12, 40, 100]  # produces materially different context lengths
BATCH_SIZES = [1, 2, 4]


def _text(n_words: int) -> str:
    base = (
        "the compression system encodes key value cache tensors from a transformer "
        "model into a compact serialized representation and reconstructs them again "
    ).split()
    return " ".join((base * 40)[:n_words])


def _profile_codec(name: str, codec, kv: torch.Tensor) -> Dict[str, Any]:
    payload, meta = codec.encode_prefill(kv)
    report = meta["accountant_report"]

    enc = measure_latency(f"{name}_encode", lambda: codec.encode_prefill(kv), warmup=2, repeats=10, device=DEVICE)
    dec = measure_latency(
        f"{name}_decode",
        lambda: codec.decode(payload, {}, tuple(kv.shape), kv.dtype, DEVICE),
        warmup=2, repeats=10, device=DEVICE,
    )
    mem = measure_peak_memory(lambda: codec.encode_prefill(kv), device=DEVICE)

    return {
        "codec": name,
        # ACTUAL serialized bytes - the acceptance gate requires measurements on
        # the real compressed representation, not a theoretical bit count.
        "serialized_bytes": report["serialized_bytes"],
        "logical_bits": report["logical_bits"],
        "bits_per_element": report["serialized_bytes"] * 8 / kv.numel(),
        "compression_ratio_vs_fp32": (kv.numel() * 4) / report["serialized_bytes"],
        "encode_latency": enc.to_dict(),
        "decode_latency": dec.to_dict(),
        "encode_peak_memory": mem.to_dict(),
    }


def main() -> None:
    out = Path("systems_profile")
    out.mkdir(exist_ok=True)

    hardware = capture_hardware_manifest(DEVICE)
    print(f"hardware: {hardware.platform} | threads={hardware.torch_num_threads} | power={hardware.power_mode}")

    capture = HFCapture(MODEL_NAME, device=DEVICE, dtype=torch.float32)
    results: Dict[str, Any] = {
        "hardware_manifest": hardware.to_dict(),
        "integration_boundary": {
            "prefill": "REAL - Hugging Face forward pass on " + MODEL_NAME,
            "decode": (
                "ATTENTION REPLAY harness over the reconstructed cache - NOT an "
                "end-to-end serving-engine integration. Decode tokens/s below is "
                "the replay rate, not a production serving throughput."
            ),
            "inferred_speedups_reported": False,
        },
        "context_sweep": [],
        "batch_sweep": [],
    }

    # ---- context-length sweep ---------------------------------------------
    for n_words in CONTEXT_WORDS:
        text = _text(n_words)
        def _do_prefill(t: str = text) -> Any:
            return capture.capture_prefill_kv(t, LayerHeadSelection())

        prefill = measure_latency(
            f"prefill_{n_words}w", _do_prefill, warmup=1, repeats=3, device=DEVICE,
        )
        K, V = capture.capture_prefill_kv(text, LayerHeadSelection())
        seq_len = K.size(3)
        print(f"\ncontext {n_words} words -> seq={seq_len}, K={tuple(K.shape)}")
        print(f"  prefill p50={prefill.p50_seconds*1000:.1f}ms p95={prefill.p95_seconds*1000:.1f}ms")

        codecs = [
            ("scalar_int8", ScalarQuantCodec("prof", tensor_name="k", default_bits=8)),
            ("scalar_int4", ScalarQuantCodec("prof", tensor_name="k", default_bits=4)),
            ("lbg_vd8_cb256", LBGVectorQuantCodec("prof", tensor_name="k", vector_dim=8, codebook_size=256, minibatch=4096, max_iters=10)),
        ]
        codec_rows: List[Dict[str, Any]] = []
        for name, codec in codecs:
            row = _profile_codec(name, codec, K)
            codec_rows.append(row)
            print(f"  {name:14s} enc p50={row['encode_latency']['p50_seconds']*1000:8.2f}ms "
                  f"dec p50={row['decode_latency']['p50_seconds']*1000:7.2f}ms "
                  f"bytes={row['serialized_bytes']:>8} ratio={row['compression_ratio_vs_fp32']:.2f}x")

        # decode-side consequence via the attention replay harness
        harness = AttentionVerificationHarness(head_dim=K.size(4))
        q = torch.randn_like(K)
        scalar = ScalarQuantCodec("prof", tensor_name="k", default_bits=8)
        payload, meta = scalar.encode_prefill(K)
        K_recon = scalar.decode(payload, {}, tuple(K.shape), K.dtype, DEVICE)
        replay = measure_latency(
            "attention_replay_step",
            lambda: harness.compute_attention(q, K_recon, V),
            warmup=2, repeats=10, device=DEVICE,
        )
        replay_tokens_per_s = seq_len / replay.p50_seconds if replay.p50_seconds > 0 else float("nan")
        print(f"  attention replay p50={replay.p50_seconds*1000:.2f}ms "
              f"-> {replay_tokens_per_s:,.0f} tok/s (REPLAY, not serving)")

        results["context_sweep"].append({
            "context_words": n_words,
            "sequence_length": seq_len,
            "kv_shape": list(K.shape),
            "prefill_latency": prefill.to_dict(),
            "codecs": codec_rows,
            "attention_replay_latency": replay.to_dict(),
            "attention_replay_tokens_per_second": replay_tokens_per_s,
            "replay_is_not_serving_throughput": True,
        })

    # ---- batch-size sweep (synthetic batching of the captured cache) -------
    K_base, _V = capture.capture_prefill_kv(_text(40), LayerHeadSelection())
    print()
    for batch in BATCH_SIZES:
        kv = K_base.repeat(1, batch, 1, 1, 1)
        codec = ScalarQuantCodec("prof", tensor_name="k", default_bits=8)
        enc = measure_latency(f"batch{batch}_encode", lambda: codec.encode_prefill(kv), warmup=2, repeats=8, device=DEVICE)
        payload, meta = codec.encode_prefill(kv)
        per_elem_us = enc.p50_seconds / kv.numel() * 1e6
        print(f"  batch={batch} elements={kv.numel():>9} enc p50={enc.p50_seconds*1000:7.2f}ms "
              f"({per_elem_us:.4f} us/element)")
        results["batch_sweep"].append({
            "batch_size": batch,
            "num_elements": kv.numel(),
            "encode_latency": enc.to_dict(),
            "microseconds_per_element": per_elem_us,
            "serialized_bytes": meta["accountant_report"]["serialized_bytes"],
        })

    # ---- chunked/streaming encode on the longest context -------------------
    chunk = recommended_chunk_tokens(tuple(K_base.shape), K_base.dtype, target_mb=1.0)
    codec = ScalarQuantCodec("prof", tensor_name="k", default_bits=8)
    single, single_meta = codec.encode_prefill(K_base)
    chunked = chunked_encode(K_base, codec.encode_prefill, chunk_tokens=max(1, min(chunk, K_base.size(3) // 2 or 1)))
    results["streaming"] = {
        "chunk_tokens": chunked.chunk_tokens,
        "num_chunks": chunked.num_chunks,
        "single_shot_bytes": len(single),
        "chunked_total_bytes": chunked.total_bytes,
        "chunking_overhead_ratio": chunked.total_bytes / len(single),
    }
    print(f"\nstreaming: {chunked.num_chunks} chunks of {chunked.chunk_tokens} tokens, "
          f"overhead {chunked.total_bytes / len(single):.3f}x vs single-shot")

    (out / "systems_profile.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out/'systems_profile.json'}")


if __name__ == "__main__":
    main()
