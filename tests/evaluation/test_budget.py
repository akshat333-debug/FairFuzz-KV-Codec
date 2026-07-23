import pytest

from fairfuzzkv_codec.fixtures.synthetic import generate_synthetic_kv_cache
from fairfuzzkv_codec.evaluation.matched_bit import MatchedBitEvaluator


def test_matched_bit_evaluator_refuses_unreachable_target():
    """Gate 3: the evaluator must refuse to compare codecs when the requested
    bit budget cannot be matched within tolerance, instead of silently
    reporting a bogus result."""
    tensor = generate_synthetic_kv_cache(1, 1, 1, 32, 64, device="cpu")
    config_hash = "test_hash"

    # 0.001 bits/element is far below what any of these baselines can reach.
    evaluator = MatchedBitEvaluator(target_bits_per_token=0.001, tolerance=0.05)

    with pytest.raises(ValueError):
        evaluator.tune_topk(tensor, config_hash)

    with pytest.raises(ValueError):
        evaluator.check_quant_codec(tensor, config_hash, bits=4)


def test_matched_bit_evaluator_accepts_reachable_target():
    """A target that sits on an achievable retention ratio must succeed and
    return a codec whose real logical bits/element fall within tolerance."""
    tensor = generate_synthetic_kv_cache(1, 1, 1, 32, 64, device="cpu")
    config_hash = "test_hash"

    # FP16 storage is 16 bits/element; retention_ratio=0.25 -> 4 bits/element.
    evaluator = MatchedBitEvaluator(target_bits_per_token=4.0, tolerance=0.1)
    tuned = evaluator.tune_topk(tensor, config_hash)

    actual_bpe = evaluator._eval_size(tuned, tensor)
    assert abs(actual_bpe - 4.0) / 4.0 <= 0.1


def test_matched_bit_evaluator_matches_topk_to_real_quant_footprint():
    """Fair matched-bit comparison: measure a real UniformQuant codec's actual
    bits/element (which includes real per-channel scale/zero-point overhead,
    not a theoretical num_bits estimate), then verify TopK can be tuned to
    that same real target within tolerance."""
    tensor = generate_synthetic_kv_cache(1, 1, 1, 32, 64, device="cpu")
    config_hash = "test_hash"

    probe = MatchedBitEvaluator(target_bits_per_token=0.0, tolerance=1.0)
    from fairfuzzkv_codec.codec.baselines import UniformQuantCodec
    measured_bpe = probe._eval_size(UniformQuantCodec(config_hash, num_bits=4), tensor)

    evaluator = MatchedBitEvaluator(target_bits_per_token=measured_bpe, tolerance=0.1)
    quant_codec = evaluator.check_quant_codec(tensor, config_hash, bits=4)
    assert quant_codec.num_bits == 4

    tuned = evaluator.tune_topk(tensor, config_hash)
    actual_bpe = evaluator._eval_size(tuned, tensor)
    assert abs(actual_bpe - measured_bpe) / measured_bpe <= 0.1
