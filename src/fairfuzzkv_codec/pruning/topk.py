import torch

def compute_topk_mask(tensor: torch.Tensor, retention_ratio: float) -> torch.Tensor:
    """
    Computes a boolean mask retaining the top-k tokens per head based on L2 norm.
    tensor shape: [layers, batch, heads, seq_len, head_dim]
    mask shape: [layers, batch, heads, seq_len, 1]
    """
    seq_len = tensor.size(3)
    k = max(1, int(seq_len * retention_ratio))
    
    # Calculate magnitude of each token representation (L2 norm over head_dim)
    # norms shape: [layers, batch, heads, seq_len]
    norms = torch.norm(tensor, p=2, dim=-1)
    
    # Get top-k indices along the sequence dimension
    # topk_indices shape: [layers, batch, heads, k]
    _, topk_indices = torch.topk(norms, k, dim=3)
    
    # Create mask
    mask = torch.zeros_like(norms, dtype=torch.bool)
    mask.scatter_(3, topk_indices, True)
    
    # Expand mask to match tensor for broadcasting if needed
    # shape: [layers, batch, heads, seq_len, 1]
    mask = mask.unsqueeze(-1)
    
    return mask

def apply_topk_pruning(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies the mask to the tensor. Tokens not in mask are zeroed out or dropped.
    For standard dense storage compatibility, we usually zero them out, but true
    compression requires storing only the non-zero indices and values.
    """
    return tensor * mask
