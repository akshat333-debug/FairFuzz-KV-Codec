import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import generate_validated_dataset  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.numeric_forms import parse_value  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.runner import FragKVRunner  # noqa: E402
from fairfuzzkv_codec.cache_capture.hf_capture import HFCapture  # noqa: E402
from fairfuzzkv_codec.core.config import LayerHeadSelection  # noqa: E402
from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec  # noqa: E402
from fairfuzzkv_codec.dashboard.plots import plot_rate_distortion_curves  # noqa: E402
from fairfuzzkv_codec.quantization.metrics import compute_distortion  # noqa: E402
from fairfuzzkv_codec.quantization.scales import Granularity  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Configs benchmarked across K/V numerical distortion AND real task accuracy.
# Kept small and explicit rather than a combinatorial sweep - enough points
# to draw a real rate-distortion curve without an excessive runtime.
CONFIGS: List[Dict[str, Any]] = [
    {"name": "INT8-per_channel", "bits": 8, "granularity": Granularity.PER_CHANNEL, "symmetric": True, "group_size": None},
    {"name": "INT8-per_tensor", "bits": 8, "granularity": Granularity.PER_TENSOR, "symmetric": True, "group_size": None},
    {"name": "INT4-per_channel", "bits": 4, "granularity": Granularity.PER_CHANNEL, "symmetric": True, "group_size": None},
    {"name": "INT4-groupwise16", "bits": 4, "granularity": Granularity.GROUPWISE, "symmetric": True, "group_size": 16},
    {"name": "INT4-per_tensor", "bits": 4, "granularity": Granularity.PER_TENSOR, "symmetric": True, "group_size": None},
]


def benchmark_kv_distortion(output_dir: Path) -> List[Dict[str, Any]]:
    print("[1] Capturing real K/V from Qwen2.5-0.5B...")
    capture = HFCapture(MODEL_NAME, device="cpu", dtype=torch.float32)
    text = (
        "FairFuzzKV-Codec is a research project for memory-conscious compression of "
        "Key-Value caches in large language models, aiming to preserve attention "
        "fidelity while reducing serialized storage footprint significantly."
    )
    K, V = capture.capture_prefill_kv(text, LayerHeadSelection())
    print(f"  K shape={tuple(K.shape)} V shape={tuple(V.shape)}")

    q_dummy = torch.randn_like(K)  # dummy Q for attention-shape verification, matching scripts/demo.py's precedent

    records = []
    for cfg in CONFIGS:
        codec_k = ScalarQuantCodec(
            "bench", tensor_name="k", granularity=cfg["granularity"], symmetric=cfg["symmetric"],
            group_size=cfg["group_size"], default_bits=cfg["bits"],
        )
        codec_v = ScalarQuantCodec(
            "bench", tensor_name="v", granularity=cfg["granularity"], symmetric=cfg["symmetric"],
            group_size=cfg["group_size"], default_bits=cfg["bits"],
        )

        stream_k, meta_k = codec_k.encode_prefill(K)
        recon_k = codec_k.decode(stream_k, meta_k, tuple(meta_k["full_shape"]), K.dtype, "cpu")
        bits_k = meta_k["accountant_report"]["logical_bits"] / K.numel()

        stream_v, meta_v = codec_v.encode_prefill(V)
        recon_v = codec_v.decode(stream_v, meta_v, tuple(meta_v["full_shape"]), V.dtype, "cpu")
        bits_v = meta_v["accountant_report"]["logical_bits"] / V.numel()

        dist_k = compute_distortion(K, recon_k, q_for_attention=q_dummy, v_original=V, v_reconstructed=recon_v)
        dist_v = compute_distortion(V, recon_v)

        print(f"  {cfg['name']}: K bits/elem={bits_k:.2f} mse={dist_k.mse:.6f} | V bits/elem={bits_v:.2f} mse={dist_v.mse:.6f}")

        records.append({"series_name": "K", "config": cfg["name"], "bits_per_element": bits_k, "mse": dist_k.mse,
                         "normalized_l2": dist_k.normalized_l2, "cosine_drift": dist_k.cosine_drift,
                         "attention_output_mse": dist_k.attention_output_mse})
        records.append({"series_name": "V", "config": cfg["name"], "bits_per_element": bits_v, "mse": dist_v.mse,
                         "normalized_l2": dist_v.normalized_l2, "cosine_drift": dist_v.cosine_drift})

    return records


def benchmark_task_accuracy(output_dir: Path, num_groups: int = 20) -> List[Dict[str, Any]]:
    print(f"\n[2] Real task-level accuracy benchmark ({num_groups} FragKV-MinPairs groups)...")
    runner = FragKVRunner(MODEL_NAME)
    groups = generate_validated_dataset(num_groups, runner.tokenizer, seed=7)
    print(f"  generated {len(groups)} validated groups")

    records = []
    for cfg in CONFIGS:
        codec_k = ScalarQuantCodec(
            "bench-task", tensor_name="k", granularity=cfg["granularity"], symmetric=cfg["symmetric"],
            group_size=cfg["group_size"], default_bits=cfg["bits"],
        )
        codec_v = ScalarQuantCodec(
            "bench-task", tensor_name="v", granularity=cfg["granularity"], symmetric=cfg["symmetric"],
            group_size=cfg["group_size"], default_bits=cfg["bits"],
        )

        correct = 0
        total = 0
        bits_sum = 0.0
        for group in groups:
            variant = group.get_variant(1)  # n_g=1 (least fragmented) for a clean task-level rate-distortion read
            K, V = runner._capture_context_kv(variant.context_text)

            stream_k, meta_k = codec_k.encode_prefill(K)
            K_recon = codec_k.decode(stream_k, meta_k, tuple(meta_k["full_shape"]), K.dtype, "cpu")
            stream_v, meta_v = codec_v.encode_prefill(V)
            V_recon = codec_v.decode(stream_v, meta_v, tuple(meta_v["full_shape"]), V.dtype, "cpu")

            bits_sum += (meta_k["accountant_report"]["logical_bits"] + meta_v["accountant_report"]["logical_bits"]) / (
                K.numel() + V.numel()
            )

            suffix_ids = runner.tokenizer(
                variant.question_text, return_tensors="pt", add_special_tokens=False
            ).input_ids
            generated_text = runner._generate_from_reconstructed_kv(K_recon, V_recon, suffix_ids, max_new_tokens=3)
            parsed = parse_value(generated_text)
            if parsed == group.canonical_value:
                correct += 1
            total += 1

        accuracy = correct / total if total else 0.0
        avg_bits = bits_sum / total if total else 0.0
        print(f"  {cfg['name']}: bits/elem={avg_bits:.2f} accuracy={accuracy:.3f} ({correct}/{total})")
        records.append({"series_name": "task", "config": cfg["name"], "bits_per_element": avg_bits,
                         "accuracy": accuracy, "task_distortion": 1.0 - accuracy})

    return records


def main():
    output_dir = Path("quantization_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    kv_records = benchmark_kv_distortion(output_dir)
    task_records = benchmark_task_accuracy(output_dir, num_groups=50)
    elapsed = time.time() - t0

    all_records = {"kv_distortion": kv_records, "task_accuracy": task_records, "elapsed_seconds": elapsed}
    (output_dir / "benchmark_results.json").write_text(json.dumps(all_records, indent=2))

    print("\n[3] Generating rate-distortion plots...")
    k_records = [r for r in kv_records if r["series_name"] == "K"]
    v_records = [r for r in kv_records if r["series_name"] == "V"]
    for r in k_records:
        r["series_name"] = "K"
    for r in v_records:
        r["series_name"] = "V"

    plot_rate_distortion_curves(k_records + v_records, str(output_dir), metric_key="mse", ylabel="MSE")
    plot_rate_distortion_curves(task_records, str(output_dir), metric_key="task_distortion", ylabel="Task Distortion (1 - accuracy)")

    print(f"\nDone in {elapsed:.1f}s. Results in {output_dir}/")


if __name__ == "__main__":
    main()
