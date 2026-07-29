"""Core selection mechanisms for three published KV-cache methods, built by
COMPOSING this project's existing `pruning.selectors.attention_mass`
(query-axis reduction) rather than reimplementing attention-mass scoring
from scratch - reuse, not duplication. Head/batch/layer axes are then
aggregated (mean) down to ONE position-level importance score BEFORE
top-k selection, so the resulting retention ratio is exact (selecting
top-k independently per head and unioning the results would inflate the
final retained count past the target, since different heads favor
different positions - the matched-bit budget must be exact at the
position level, not a per-head approximation).

Every function here reproduces the DEFINING mechanism of its named method as
described in its publication, to the best of what is reliably known without
network access to the original paper PDF or reference repository in this
environment. Exact hyperparameter defaults (window sizes, pooling kernels)
used by the original authors' code could NOT be verified here and are
documented as explicit, overridable config with a stated best-effort value -
see each function's docstring and the corresponding `BaselineCard` in
`registry.py`. This is why these are marked `ReproductionStatus.APPROXIMATE`,
not `FAITHFUL`.
"""

import torch

SEQ_AXIS = 3


def _topk_mask_1d(score: torch.Tensor, keep: int) -> torch.Tensor:
    seq_len = score.shape[-1]
    keep = max(1, min(seq_len, keep))
    _, idx = torch.topk(score, keep)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[idx] = True
    return mask


def h2o_mask(
    attn_weights: torch.Tensor, heavy_ratio: float = 0.5, recent_ratio: float = 0.5,
) -> torch.Tensor:
    """H2O (Zhang et al., 2023) - decode-time eviction: keep the union of
    (a) "heavy hitter" positions with the highest cumulative attention mass
    received so far (mean-aggregated across layers/heads/batch to one
    position-level score), and (b) a fixed-size RECENT window.
    `heavy_ratio`/`recent_ratio` are each a fraction of the sequence length;
    they need not sum to 1.0 - the union is what matters, matching the
    original method's design that heavy-hitters and recent tokens can
    overlap. Returns a 1D [seq] bool keep-mask."""
    from fairfuzzkv_codec.pruning.selectors import attention_mass

    mass = attention_mass(attn_weights)  # [L,B,H,Sk]
    seq_len = mass.size(SEQ_AXIS)
    position_score = mass.mean(dim=(0, 1, 2))  # [Sk]

    heavy_keep = max(1, int(seq_len * heavy_ratio))
    recent_keep = max(1, int(seq_len * recent_ratio))

    heavy_mask = _topk_mask_1d(position_score, heavy_keep)
    recent_mask = torch.zeros(seq_len, dtype=torch.bool)
    recent_mask[seq_len - recent_keep :] = True

    return heavy_mask | recent_mask


def snapkv_mask(
    attn_weights: torch.Tensor, keep_ratio: float, observation_window: int = 16, pooling_kernel: int = 5,
) -> torch.Tensor:
    """SnapKV (Li et al., 2024) - prefill-time compression: use only the
    LAST `observation_window` query positions (the "voting" window nearest
    generation) to score every prior key position, mean-aggregated across
    layers/heads/batch to one position-level score, then apply 1D
    max-pooling over the key axis so locally-clustered important regions
    survive intact rather than being punctured by single-position gaps,
    before taking the top-k by pooled score.

    `observation_window=16` and `pooling_kernel=5` are this implementation's
    best-effort defaults, not verified against the original reference
    implementation (no network access) - override them explicitly if a
    specific reproduction target is needed; see the BaselineCard."""
    if attn_weights.dim() != 5:
        raise ValueError("attn_weights must be [layers, batch, heads, q, k]")
    seq_len = attn_weights.size(-1)
    window = min(observation_window, attn_weights.size(3))

    # score: attention mass each key received, summed over ONLY the last
    # `window` query positions (the observation window), not the full q axis,
    # then mean-aggregated across layers/heads/batch to one position score.
    window_attn = attn_weights[:, :, :, -window:, :]
    score = window_attn.sum(dim=3).mean(dim=(0, 1, 2))  # [Sk]

    pooled = _max_pool_1d(score, pooling_kernel)
    keep = max(1, int(seq_len * keep_ratio))
    return _topk_mask_1d(pooled, keep)


def _max_pool_1d(score: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Symmetric 1D max-pool over a [seq] score, same length out as in
    (matches SnapKV's clustering-preservation intent: a position's pooled
    score is the max score in its local neighborhood, so an isolated
    high-score spike still "protects" its neighbors from eviction)."""
    if kernel_size <= 1:
        return score
    pad = kernel_size // 2
    flat = score.reshape(1, 1, -1)
    pooled = torch.nn.functional.max_pool1d(flat, kernel_size=kernel_size, stride=1, padding=pad)
    pooled = pooled[..., : score.shape[-1]]  # even kernel sizes can overshoot by one; trim back to seq_len
    return pooled.reshape(score.shape)


def pyramidkv_mask(
    attn_weights: torch.Tensor, total_keep_ratio: float, pyramid_ratio: float = 2.0,
) -> torch.Tensor:
    """PyramidKV (2024) - prefill-time compression with a PER-LAYER budget:
    early layers keep MORE positions, later layers keep FEWER, following the
    empirical finding that attention concentrates more sharply in later
    layers (so they tolerate more aggressive pruning). Budget schedule:
    layer l's retention count is linearly interpolated from
    `pyramid_ratio * base` (layer 0) down to `base / pyramid_ratio` (last
    layer), scaled so the AVERAGE across layers hits `total_keep_ratio`
    exactly (the matched-bit target). `pyramid_ratio=2.0` is this
    implementation's best-effort default (early layers keep ~2x, late layers
    ~0.5x the average), not verified against the original reference
    implementation - see the BaselineCard.

    Returns a 2D [layers, seq] bool mask (ExplicitMaskCodec's per-layer
    mask path), since the whole point of this method is a DIFFERENT budget
    per layer - a 1D mask couldn't represent it."""
    if attn_weights.dim() != 5:
        raise ValueError("attn_weights must be [layers, batch, heads, q, k]")
    from fairfuzzkv_codec.pruning.selectors import attention_mass

    mass = attention_mass(attn_weights)  # [L,B,H,Sk]
    num_layers, seq_len = mass.size(0), mass.size(SEQ_AXIS)
    per_layer_score = mass.mean(dim=(1, 2))  # [L, Sk]

    if num_layers == 1:
        multipliers = [1.0]
    else:
        multipliers = [pyramid_ratio + (1.0 / pyramid_ratio - pyramid_ratio) * (layer / (num_layers - 1)) for layer in range(num_layers)]
    mean_multiplier = sum(multipliers) / num_layers
    base_keep = total_keep_ratio * seq_len / mean_multiplier

    out = torch.zeros(num_layers, seq_len, dtype=torch.bool)
    for layer in range(num_layers):
        keep = max(1, min(seq_len, int(round(base_keep * multipliers[layer]))))
        out[layer] = _topk_mask_1d(per_layer_score[layer], keep)
    return out
