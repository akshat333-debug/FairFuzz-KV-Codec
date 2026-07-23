import torch
from fairfuzzkv_codec.fixtures.synthetic import generate_synthetic_kv_cache
from fairfuzzkv_codec.codec.noop import NoOpCodec

def test_noop_roundtrip():
    tensor = generate_synthetic_kv_cache(layers=4, batch=1, heads=4, sequence=128, head_dim=64, device="cpu")
    codec = NoOpCodec()
    
    # Encode
    byte_stream, metadata = codec.encode_prefill(tensor)
    
    # Assert exact byte accounting matches
    assert len(byte_stream) == metadata["exact_bytes"]
    
    # Decode
    reconstructed = codec.decode(byte_stream, metadata, tuple(metadata["shape"]), torch.float16, "cpu")
    
    # Assert exact bit-identical parity
    assert torch.all(tensor == reconstructed).item() is True
