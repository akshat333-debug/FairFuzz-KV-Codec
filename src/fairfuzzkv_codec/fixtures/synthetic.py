import torch

def generate_synthetic_kv_cache(
    layers: int,
    batch: int,
    heads: int,
    sequence: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Generate a synthetic KV-cache fixture.
    Shape: [layers, batch, heads, sequence, head_dim]
    """
    # Ensure exact shapes
    shape = (layers, batch, heads, sequence, head_dim)
    
    # Generate random normally distributed data typical of post-RoPE activations
    tensor = torch.randn(shape, dtype=dtype, device=device)
    
    return tensor
