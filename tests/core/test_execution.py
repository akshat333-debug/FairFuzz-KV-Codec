from fairfuzzkv_codec.core.execution import set_seed, compute_config_hash
from fairfuzzkv_codec.core.config import FairFuzzKVConfig, ModelIdentity, CodecBudget, QuantizerConfig, QuantizerType, HardwareManifest
import torch

def test_determinism_seeding():
    set_seed(42)
    tensor1 = torch.randn(10, 10)
    
    set_seed(42)
    tensor2 = torch.randn(10, 10)
    
    assert torch.all(tensor1 == tensor2).item() is True

def test_config_hashing():
    config1 = FairFuzzKVConfig(
        model=ModelIdentity(model_name="test", layer_count=2, head_count=2, head_dim=64),
        budget=CodecBudget(total_bits_per_element=8.0),
        quantizer=QuantizerConfig(quantizer_type=QuantizerType.NOOP),
        hardware=HardwareManifest(device="cpu", dtype="float16")
    )
    
    config2 = FairFuzzKVConfig(
        model=ModelIdentity(model_name="test", layer_count=2, head_count=2, head_dim=64),
        budget=CodecBudget(total_bits_per_element=8.0),
        quantizer=QuantizerConfig(quantizer_type=QuantizerType.NOOP),
        hardware=HardwareManifest(device="cpu", dtype="float16")
    )
    
    hash1 = compute_config_hash(config1)
    hash2 = compute_config_hash(config2)
    
    assert hash1 == hash2
