import torch
import sys
from pathlib import Path

# Adjust path to import package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairfuzzkv_codec.core.config import FairFuzzKVConfig, ModelIdentity, CodecBudget, QuantizerConfig, QuantizerType, HardwareManifest, LayerHeadSelection
from fairfuzzkv_codec.core.execution import ExecutionManager, compute_config_hash
from fairfuzzkv_codec.cache_capture.hf_capture import HFCapture
from fairfuzzkv_codec.benchmarks.attention_harness import AttentionVerificationHarness
from fairfuzzkv_codec.codec.baselines import FullKVFP16Codec, UniformQuantCodec, TopKCodec
from fairfuzzkv_codec.evaluation.profiler import TelemetryTracker
from fairfuzzkv_codec.dashboard.plots import plot_distortion_vs_budget, plot_compression_ratio
from fairfuzzkv_codec.evaluation.matched_bit import MatchedBitEvaluator

def main():
    print("=== FairFuzzKV-Codec: Grade-Floor Baseline Gate ===")
    
    # 1. Config
    config = FairFuzzKVConfig(
        model=ModelIdentity(model_name="Qwen/Qwen2.5-0.5B", layer_count=24, head_count=14, head_dim=64),
        selection=LayerHeadSelection(layers=[0], heads=[0, 1]), # Select layer 0, heads 0, 1 for fast CPU smoke test
        budget=CodecBudget(total_bits_per_element=8.0),
        quantizer=QuantizerConfig(quantizer_type=QuantizerType.UNIFORM),
        hardware=HardwareManifest(device="cpu", dtype="float32")
    )
    
    manager = ExecutionManager(base_dir="results")
    run_dir = manager.create_run_directory(config, "grade_floor_demo")
    config_hash = compute_config_hash(config)
    
    print(f"\n[1] Capturing KV Cache from {config.model.model_name}...")
    telemetry = TelemetryTracker("cpu")
    real_capture_succeeded = False

    try:
        capture = HFCapture(config.model.model_name, device="cpu", dtype=torch.float32)
        text = "FairFuzzKV-Codec is a research project for compressing KV caches."

        telemetry.start_recording()
        K, V = capture.capture_prefill_kv(text, config.selection)
        metrics = telemetry.end_recording("hf_capture")
        print(f"Captured K shape: {K.shape}, V shape: {V.shape} in {metrics['elapsed_ms']:.2f} ms")
        real_capture_succeeded = True
    except Exception as e:
        print(f"Failed to load model from HF. Error: {e}")
        print("Fallback to synthetic data for demo...")
        from fairfuzzkv_codec.fixtures.synthetic import generate_synthetic_kv_cache
        K = generate_synthetic_kv_cache(1, 1, 2, 16, 64, dtype=torch.float32, device="cpu")
        V = generate_synthetic_kv_cache(1, 1, 2, 16, 64, dtype=torch.float32, device="cpu")
        
    print("\n[2] Initializing Baseline Codecs...")
    codecs = [
        ("FP16 Full", FullKVFP16Codec(config_hash)),
        ("INT8 Uniform", UniformQuantCodec(config_hash, num_bits=8)),
        ("INT4 Uniform", UniformQuantCodec(config_hash, num_bits=4)),
        ("Top-K (50%)", TopKCodec(config_hash, retention_ratio=0.5))
    ]
    
    # 3. Attention Harness
    harness = AttentionVerificationHarness(head_dim=K.size(-1))
    q_dummy = torch.randn_like(K) # Dummy Q for MSE test
    
    results = []
    ratio_results = []
    
    original_size_bytes = K.numel() * K.element_size()
    
    for name, codec in codecs:
        print(f"\nEvaluating {name}...")
        telemetry.start_recording()
        byte_stream, meta = codec.encode_prefill(K)
        enc_metrics = telemetry.end_recording(f"{name}_encode")
        
        serialized_size = len(byte_stream)
        comp_ratio = original_size_bytes / serialized_size
        
        print(f"  Serialized size: {serialized_size} bytes (Overhead: {meta['accountant_report']['overhead_bytes']} bytes)")
        print(f"  Compression Ratio: {comp_ratio:.2f}x")
        print(f"  Encode time: {enc_metrics['elapsed_ms']:.2f} ms")
        
        telemetry.start_recording()
        K_recon = codec.decode(byte_stream, meta, tuple(meta['kv_shape']), K.dtype, "cpu")
        dec_metrics = telemetry.end_recording(f"{name}_decode")
        
        mse = harness.verify_reconstruction(q_dummy, K, V, K_recon, V)
        print(f"  Decode time: {dec_metrics['elapsed_ms']:.2f} ms")
        print(f"  Attention KV Distortion (MSE): {mse:.6f}")
        
        bpe = meta['accountant_report']['logical_bits'] / K.numel()

        results.append({
            'codec_name': name,
            'bits_per_token': bpe,
            'distortion': mse
        })
        
        ratio_results.append({
            'codec_name': name,
            'compression_ratio': comp_ratio
        })
        
    print("\n[4] Matched-Bit Evaluator (Target: 4 bits/element)...")
    evaluator = MatchedBitEvaluator(target_bits_per_token=4.0, tolerance=0.10)
    matched_bit_succeeded = False
    try:
        tuned_topk = evaluator.tune_topk(K, config_hash)
        bpe = evaluator._eval_size(tuned_topk, K)
        print(f"Successfully tuned Top-K to {bpe:.2f} bits/element!")
        matched_bit_succeeded = True
    except Exception as e:
        print(f"Matched bit evaluator failed: {e}")

    print("\n[5] Generating Plots and Dashboard...")
    plot_distortion_vs_budget(results, str(run_dir))
    plot_compression_ratio(ratio_results, str(run_dir))

    if real_capture_succeeded and matched_bit_succeeded:
        print(f"\n[DEMO] Success! Grade-Floor Baseline Gate passed. Check results in: {run_dir}")
    else:
        print(f"\n[DEMO] PARTIAL: real_capture={real_capture_succeeded}, matched_bit={matched_bit_succeeded}. Check results in: {run_dir}")

if __name__ == "__main__":
    main()
