from typing import Optional

import torch
from pydantic import BaseModel

from fairfuzzkv_codec.benchmarks.attention_harness import AttentionVerificationHarness


class DistortionReport(BaseModel):
    mse: float
    normalized_l2: float  # ||orig - recon|| / ||orig||, scale-invariant
    cosine_drift: float  # 1 - cosine_similarity, 0 = identical direction
    attention_output_mse: Optional[float] = None
    task_distortion: Optional[float] = None  # e.g. 1 - task accuracy, filled in by the caller if available


def compute_distortion(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    q_for_attention: Optional[torch.Tensor] = None,
    v_original: Optional[torch.Tensor] = None,
    v_reconstructed: Optional[torch.Tensor] = None,
    task_distortion: Optional[float] = None,
) -> DistortionReport:
    """Core numerical distortion metrics between an original tensor and its
    quantize/dequantize reconstruction. Passing q_for_attention plus a
    matching (v_original, v_reconstructed) pair also computes real
    attention-output MSE via the existing AttentionVerificationHarness
    (treating `original`/`reconstructed` as K); omit them to skip that part.
    task_distortion is a pass-through - filled in by callers that ran a real
    downstream task (e.g. fairfuzzkv_codec.benchmarks.fragkv_minpairs), never
    computed here (this module has no task-execution logic of its own)."""
    diff = (reconstructed - original).float()
    orig_f = original.float()

    mse = diff.pow(2).mean().item()

    orig_norm = orig_f.norm().item()
    normalized_l2 = (diff.norm().item() / orig_norm) if orig_norm > 0 else float("inf")

    orig_flat = orig_f.reshape(-1)
    recon_flat = reconstructed.float().reshape(-1)
    denom = orig_flat.norm() * recon_flat.norm()
    cosine_sim = (orig_flat @ recon_flat / denom).item() if denom > 0 else 0.0
    cosine_drift = 1.0 - cosine_sim

    attention_output_mse = None
    if q_for_attention is not None and v_original is not None and v_reconstructed is not None:
        harness = AttentionVerificationHarness(head_dim=original.size(-1))
        attention_output_mse = harness.verify_reconstruction(
            q_for_attention, original, v_original, reconstructed, v_reconstructed
        )

    return DistortionReport(
        mse=mse,
        normalized_l2=normalized_l2,
        cosine_drift=cosine_drift,
        attention_output_mse=attention_output_mse,
        task_distortion=task_distortion,
    )
