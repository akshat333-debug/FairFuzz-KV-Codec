"""The Prompt 16 baseline matrix: one `BaselineCard` per named method,
whether or not it has a working adapter. `REPRODUCED_ADAPTERS` builds the
adapters that actually run; `NOT_REPRODUCED_CARDS` documents the rest with a
reason and a nearest faithful configuration - never silently substituted
under the original name (item 111).
"""

from typing import List

import torch

from fairfuzzkv_codec.baselines.adapter import BaselineAdapter
from fairfuzzkv_codec.baselines.schema import BaselineCard, EvaluationRegime, ReproductionStatus
from fairfuzzkv_codec.baselines.selection_methods import h2o_mask, pyramidkv_mask, snapkv_mask
from fairfuzzkv_codec.codec.baselines import TopKCodec, UniformQuantCodec
from fairfuzzkv_codec.codec.explicit_mask import ExplicitMaskCodec
from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
from fairfuzzkv_codec.codec.vector_quant import LBGVectorQuantCodec

_NO_NETWORK_NOTE = (
    "No verified network/license access was available in this environment to fetch the "
    "original paper or reference implementation - see PENDING.md."
)

MODEL_SUPPORT_NOTE = "Runs against any captured KV cache tensor [layers, batch, heads, seq, head_dim]; validated on Qwen2.5-0.5B in this project's studies."
CONTEXT_LIMIT_NOTE = "No hard context-length limit in this implementation beyond available memory; not benchmarked past this project's short (<300 token) study contexts."


def build_quantization_adapters(config_hash: str) -> List[BaselineAdapter]:
    def uniform_int(target: float) -> UniformQuantCodec:
        return UniformQuantCodec(config_hash, num_bits=8 if target >= 6.0 else 4)

    def scalar_genuine(target: float) -> ScalarQuantCodec:
        bits = 8 if target >= 6.0 else 4
        return ScalarQuantCodec(config_hash, tensor_name="k", symmetric=True, default_bits=bits)

    def lbg_fixed(_target: float) -> LBGVectorQuantCodec:
        # vector_dim=2, codebook_size=256 -> 8 index bits / 2 dims = 4.0
        # bits/element before codebook overhead - close to this run's 4.0
        # bits/element target by construction, not dynamically solved (LBG's
        # discrete (vector_dim, codebook_size) grid makes exact per-target
        # tuning impractical here - see BaselineCard deviations). minibatch
        # caps per-Lloyd-iteration vector count - without it, a real Qwen
        # K/V tensor at vector_dim=2 yields ~150k+ vectors, and LBG's split+
        # Lloyd training (up to codebook_size's log2 splits x max_iters each)
        # becomes prohibitively slow on CPU; self-calibrating fresh on every
        # variant (80x in this study) makes that cost multiply badly.
        return LBGVectorQuantCodec(config_hash, tensor_name="k", vector_dim=2, codebook_size=256, minibatch=4096, max_iters=15)

    return [
        BaselineAdapter(
            card=BaselineCard(
                name="UniformINT8/INT4", regime=EvaluationRegime.COMPRESSION_QUANTIZATION,
                reproduction_status=ReproductionStatus.FAITHFUL,
                version_note="This project's own grade-floor baseline codec (Prompt 1-2), not a literature reproduction.",
                model_support=MODEL_SUPPORT_NOTE, context_limit_note=CONTEXT_LIMIT_NOTE,
                deviations="", limitations="Per-tensor scale only; no percentile/MSE-optimal clipping.",
            ),
            codec_factory=uniform_int, is_discrete=True,
        ),
        BaselineAdapter(
            card=BaselineCard(
                name="FairFuzzKV-Scalar", regime=EvaluationRegime.COMPRESSION_QUANTIZATION,
                reproduction_status=ReproductionStatus.FAITHFUL,
                version_note="This project's own stronger scalar codec (Prompt 6): symmetric INT4/INT8, genuine nibble packing, percentile-clip capable.",
                model_support=MODEL_SUPPORT_NOTE, context_limit_note=CONTEXT_LIMIT_NOTE,
                deviations="", limitations="",
            ),
            codec_factory=scalar_genuine, is_discrete=True,
        ),
        BaselineAdapter(
            card=BaselineCard(
                name="FairFuzzKV-LBG", regime=EvaluationRegime.COMPRESSION_QUANTIZATION,
                reproduction_status=ReproductionStatus.FAITHFUL,
                version_note="This project's own LBG vector-quantization codec (Prompt 7), fixed config vector_dim=2/codebook_size=256.",
                model_support=MODEL_SUPPORT_NOTE, context_limit_note=CONTEXT_LIMIT_NOTE,
                deviations="Fixed (vector_dim, codebook_size) rather than dynamically tuned per target bits/element - LBG's discrete grid makes exact tuning impractical; reported bits/element may miss the run's target and is marked unmatched rather than hidden.",
                limitations="Self-calibrates on the same tensor it encodes (no separate calibration split in this quick per-example comparison) - codebook overhead is still fully counted, but this is not the leakage-safe calibration protocol used in lbg_benchmark/.",
            ),
            codec_factory=lbg_fixed, is_discrete=True,
        ),
    ]


def build_prefill_selection_adapters(config_hash: str, attn_weights: torch.Tensor) -> List[BaselineAdapter]:
    # is_discrete=False -> `adapter.tune_to_matched_bits` binary-searches
    # this argument directly AS a retention ratio in [0.01, 1.0] (mirroring
    # `evaluation.matched_bit.MatchedBitEvaluator.tune_topk`'s own
    # `TopKCodec(retention_ratio=mid)` pattern) - it is NOT the outer target
    # bits/element, which the search is solving FOR, not given as an input.
    def topk_l2(ratio: float) -> TopKCodec:
        return TopKCodec(config_hash, retention_ratio=ratio)

    def snapkv(ratio: float) -> ExplicitMaskCodec:
        mask = snapkv_mask(attn_weights, keep_ratio=ratio)
        return ExplicitMaskCodec(config_hash, mask)

    def pyramidkv(ratio: float) -> ExplicitMaskCodec:
        mask = pyramidkv_mask(attn_weights, total_keep_ratio=ratio)
        return ExplicitMaskCodec(config_hash, mask)

    return [
        BaselineAdapter(
            card=BaselineCard(
                name="TopK-L2", regime=EvaluationRegime.PREFILL_SELECTION,
                reproduction_status=ReproductionStatus.FAITHFUL,
                version_note="This project's own grade-floor magnitude baseline (Prompt 2): retains highest-L2-norm token representations.",
                model_support=MODEL_SUPPORT_NOTE, context_limit_note=CONTEXT_LIMIT_NOTE,
                deviations="", limitations="Ignores attention entirely - a pure representation-magnitude heuristic.",
            ),
            codec_factory=topk_l2, is_discrete=False,
        ),
        BaselineAdapter(
            card=BaselineCard(
                name="SnapKV", regime=EvaluationRegime.PREFILL_SELECTION,
                reproduction_status=ReproductionStatus.APPROXIMATE,
                version_note=f"Li et al. 2024 - core mechanism (observation-window attention voting + 1D max-pooling before top-k). {_NO_NETWORK_NOTE}",
                model_support=MODEL_SUPPORT_NOTE, context_limit_note=CONTEXT_LIMIT_NOTE,
                deviations="observation_window=16, pooling_kernel=5 are this implementation's best-effort defaults, not verified against the original reference code.",
                limitations="Position-level (mean-aggregated across layers/heads), not the original's exact per-head pooling if that differs.",
            ),
            codec_factory=snapkv, is_discrete=False,
        ),
        BaselineAdapter(
            card=BaselineCard(
                name="PyramidKV", regime=EvaluationRegime.PREFILL_SELECTION,
                reproduction_status=ReproductionStatus.APPROXIMATE,
                version_note=f"2024 - core mechanism (per-layer pyramid retention budget schedule, more budget for early layers). {_NO_NETWORK_NOTE}",
                model_support=MODEL_SUPPORT_NOTE, context_limit_note=CONTEXT_LIMIT_NOTE,
                deviations="pyramid_ratio=2.0 (linear interpolation of per-layer multiplier) is this implementation's best-effort default, not verified against the original reference code's exact schedule.",
                limitations="Requires a 2D per-layer mask (ExplicitMaskCodec extension); within-layer selection is plain top-attention-mass, not any additional intra-layer heuristic the original might use.",
            ),
            codec_factory=pyramidkv, is_discrete=False,
        ),
    ]


def build_decode_time_selection_adapters(config_hash: str, attn_weights: torch.Tensor) -> List[BaselineAdapter]:
    # See build_prefill_selection_adapters' comment: `ratio` is the searched
    # knob itself (a retention-ratio guess in [0.01, 1.0]), not a bits/element
    # value to re-derive a ratio from.
    def h2o(ratio: float) -> ExplicitMaskCodec:
        mask = h2o_mask(attn_weights, heavy_ratio=ratio * 0.7, recent_ratio=ratio * 0.5)
        return ExplicitMaskCodec(config_hash, mask)

    return [
        BaselineAdapter(
            card=BaselineCard(
                name="H2O", regime=EvaluationRegime.DECODE_TIME_SELECTION,
                reproduction_status=ReproductionStatus.APPROXIMATE,
                version_note=f"Zhang et al. 2023 - core mechanism (union of heavy-hitter cumulative-attention-mass positions and a fixed recent window). {_NO_NETWORK_NOTE}",
                model_support=MODEL_SUPPORT_NOTE, context_limit_note=CONTEXT_LIMIT_NOTE,
                deviations="heavy_ratio/recent_ratio scaled from the single target retention ratio (0.7x / 0.5x split) rather than the original's own budget-split hyperparameter, which was not verified against reference code.",
                limitations="DECODE-TIME regime only - never compared against prefill-time selection results in the same table (item 108). Applied here to a single prefill capture as a structural stand-in for iterative decode-time eviction, since this project's runners do real single-shot generation, not long multi-step decode loops.",
            ),
            codec_factory=h2o, is_discrete=False,
        ),
    ]


NOT_REPRODUCED_CARDS: List[BaselineCard] = [
    BaselineCard(
        name="RateQuant", regime=EvaluationRegime.COMPRESSION_QUANTIZATION, reproduction_status=ReproductionStatus.NOT_REPRODUCED,
        version_note=f"Name refers to a rate-distortion-optimized quantization method; the exact allocation formula could not be verified. {_NO_NETWORK_NOTE}",
        model_support="N/A - not implemented", context_limit_note="N/A",
        deviations="Not implemented under this name.",
        limitations="Not reproduced - do not cite results under the name RateQuant.",
        nearest_faithful_configuration="This project's own Prompt 10 aggregate rate-distortion allocator (`fairfuzzkv_codec.allocation`) solves the conceptually similar problem (minimize distortion subject to a total bit budget) and is a REAL, already-benchmarked deliverable (see allocation_study/) - but it is this project's own method, not a reproduction of a paper named RateQuant, and must not be relabeled as such.",
    ),
    BaselineCard(
        name="RDKV", regime=EvaluationRegime.COMPRESSION_QUANTIZATION, reproduction_status=ReproductionStatus.NOT_REPRODUCED,
        version_note=f"Exact method specification could not be verified. {_NO_NETWORK_NOTE}",
        model_support="N/A - not implemented", context_limit_note="N/A",
        deviations="Not implemented under this name.",
        limitations="Not reproduced - do not cite results under the name RDKV.",
        nearest_faithful_configuration="This project's rate-distortion allocator (Prompt 10) plus `ScalarQuantCodec` covers the general rate-distortion-vs-KV-quantization design space this name likely refers to, but again is not a verified reproduction of a specific published RDKV method.",
    ),
    BaselineCard(
        name="KVTuner", regime=EvaluationRegime.COMPRESSION_QUANTIZATION, reproduction_status=ReproductionStatus.NOT_REPRODUCED,
        version_note=f"Exact per-layer/per-head mixed-precision search procedure could not be verified. {_NO_NETWORK_NOTE}",
        model_support="N/A - not implemented", context_limit_note="N/A",
        deviations="Not implemented under this name.",
        limitations="Not reproduced - do not cite results under the name KVTuner.",
        nearest_faithful_configuration="This project's own `BitWidthMap` sparse per-layer/per-head bit-width override mechanism (Prompt 6, `ScalarQuantCodec`) already supports mixed-precision configuration search; it is the nearest REAL functionality in this codebase, not a verified reproduction of KVTuner's specific search algorithm.",
    ),
    BaselineCard(
        name="KVmix", regime=EvaluationRegime.COMPRESSION_QUANTIZATION, reproduction_status=ReproductionStatus.NOT_REPRODUCED,
        version_note=f"Exact K/V-differentiated mixed-precision policy could not be verified. {_NO_NETWORK_NOTE}",
        model_support="N/A - not implemented", context_limit_note="N/A",
        deviations="Not implemented under this name.",
        limitations="Not reproduced - do not cite results under the name KVmix.",
        nearest_faithful_configuration="This project's `ScalarQuantCodec` already supports independent K/V bit-width configuration (Prompt 6) - the nearest real functionality, not a verified reproduction of KVmix's specific policy for choosing those bit-widths.",
    ),
]
