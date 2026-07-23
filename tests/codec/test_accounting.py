from fairfuzzkv_codec.fixtures.synthetic import generate_synthetic_kv_cache
from fairfuzzkv_codec.codec.baselines import FullKVFP16Codec, UniformQuantCodec, TopKCodec


def test_serialized_bytes_matches_sum_of_components():
    """Claim C-01: the byte accountant must reflect real serialized bytes, never
    a substitute like a sparsity ratio. serialized_bytes must equal the actual
    length of the encoded byte stream and the sum of its component sizes."""
    tensor = generate_synthetic_kv_cache(2, 1, 2, 16, 32, device="cpu")
    config_hash = "test_hash"

    codecs = [
        FullKVFP16Codec(config_hash),
        UniformQuantCodec(config_hash, num_bits=8),
        UniformQuantCodec(config_hash, num_bits=4),
        TopKCodec(config_hash, retention_ratio=0.5),
    ]

    for codec in codecs:
        byte_stream, meta = codec.encode_prefill(tensor)
        report = meta["accountant_report"]

        assert report["serialized_bytes"] == len(byte_stream)
        assert sum(report["components"].values()) == report["serialized_bytes"]
        assert report["logical_bits"] > 0


def test_overhead_bytes_reflects_true_gap_between_logical_and_serialized():
    """overhead_bytes must be the real gap between serialized size and the
    theoretical logical bit-budget, not a hidden/omitted value."""
    tensor = generate_synthetic_kv_cache(1, 1, 1, 16, 32, device="cpu")
    codec = UniformQuantCodec("test_hash", num_bits=4)
    byte_stream, meta = codec.encode_prefill(tensor)
    report = meta["accountant_report"]

    import math
    expected_overhead = report["serialized_bytes"] - math.ceil(report["logical_bits"] / 8.0)
    assert report["overhead_bytes"] == expected_overhead
