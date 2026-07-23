import torch
from fairfuzzkv_codec.fixtures.synthetic import generate_synthetic_kv_cache
from fairfuzzkv_codec.codec.baselines import FullKVFP16Codec, UniformQuantCodec, TopKCodec
from fairfuzzkv_codec.evaluation.matched_bit import MatchedBitEvaluator

def test_baseline_codecs():
    tensor = generate_synthetic_kv_cache(1, 1, 1, 32, 64, device="cpu")
    config_hash = "test_hash"
    
    codecs = [
        FullKVFP16Codec(config_hash),
        UniformQuantCodec(config_hash, num_bits=8),
        UniformQuantCodec(config_hash, num_bits=4),
        TopKCodec(config_hash, retention_ratio=0.5)
    ]
    
    for codec in codecs:
        byte_stream, meta = codec.encode_prefill(tensor)
        
        # Test Serialization
        assert len(byte_stream) > 0
        assert "accountant_report" in meta
        
        # Test Deserialization
        reconstructed = codec.decode(byte_stream, meta, tuple(meta["kv_shape"]), torch.float16, "cpu")
        
        # Shape should match if not pruned (for Top-K we kept shape same in our baseline dummy)
        assert reconstructed.shape == tensor.shape

def test_matched_bit_evaluator():
    tensor = generate_synthetic_kv_cache(1, 1, 1, 32, 64, device="cpu")
    config_hash = "test_hash"
    
    # Target the exact bits/element that Uniform 4-bit produces to test the checker
    test_codec = UniformQuantCodec(config_hash, num_bits=4)
    byte_stream, meta = test_codec.encode_prefill(tensor)
    actual_bpe = meta["accountant_report"]["logical_bits"] / tensor.numel()

    evaluator = MatchedBitEvaluator(target_bits_per_token=actual_bpe, tolerance=0.1)
    
    # Uniform 4-bit should match exactly
    codec = evaluator.check_quant_codec(tensor, config_hash, 4)
    assert codec.num_bits == 4
    
    # TopK should be tuned to retention ratio ~ 0.25 (since FP16 is 16 bits, 16 * 0.25 = 4)
    tuned = evaluator.tune_topk(tensor, config_hash)
    assert abs(tuned.retention_ratio - 0.25) < 0.1
