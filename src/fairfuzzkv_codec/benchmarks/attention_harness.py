import torch
import torch.nn.functional as F

class AttentionVerificationHarness:
    """
    Verifies that the reconstructed KV cache behaves correctly in a standard
    scaled dot-product attention calculation.
    """
    def __init__(self, head_dim: int):
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Standard attention.
        q, k, v expected shape: [batch, heads, seq_len, head_dim]
        For Q, seq_len is usually 1 during decode, or N during prefill.
        """
        # q: [batch, heads, q_len, head_dim]
        # k: [batch, heads, kv_len, head_dim]
        # v: [batch, heads, kv_len, head_dim]
        
        # attn_weights: [batch, heads, q_len, kv_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # out: [batch, heads, q_len, head_dim]
        out = torch.matmul(attn_probs, v)
        return out

    def verify_reconstruction(self, q: torch.Tensor, k_orig: torch.Tensor, v_orig: torch.Tensor, 
                              k_recon: torch.Tensor, v_recon: torch.Tensor) -> float:
        """
        Computes L2 distance between attention outputs using original vs reconstructed KV.
        Returns the MSE error.
        """
        out_orig = self.compute_attention(q, k_orig, v_orig)
        out_recon = self.compute_attention(q, k_recon, v_recon)
        
        mse = F.mse_loss(out_orig, out_recon).item()
        return mse
