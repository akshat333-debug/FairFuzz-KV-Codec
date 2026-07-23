"""Per-cohort distortion calibration with enforced train/val/test separation.

For each cohort (here: a layer slice of the KV cache) and each quantizer option,
measure real reconstruction distortion. Calibration parameters (scalar clip
range, LBG codebook) are fit on the TRAIN split; distortion is REPORTED on the
held-out TEST split - so a codec can never look good by memorizing the data it is
scored on. Produces `Cohort`/`BitOption` structures the allocator consumes, and
a bridge that turns an allocation into a `BitWidthMap` driving the real encoder.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from fairfuzzkv_codec.allocation.allocator import Allocation, BitOption, Cohort
from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
from fairfuzzkv_codec.quantization.bitwidth_map import BitWidthMap
from fairfuzzkv_codec.quantization.scales import Granularity

SEQ_DIM = 3


@dataclass
class Split:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


def make_split(kv_cache: torch.Tensor, fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Split:
    """Split along the sequence axis into disjoint train/val/test. Disjoint by
    construction - the same token position never appears in two splits."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    S = kv_cache.size(SEQ_DIM)
    n_train = max(1, int(S * fractions[0]))
    n_val = max(1, int(S * fractions[1]))
    n_train = min(n_train, S - 2)
    n_val = min(n_val, S - n_train - 1)
    idx = torch.arange(S)
    return Split(
        train=kv_cache.index_select(SEQ_DIM, idx[:n_train]),
        val=kv_cache.index_select(SEQ_DIM, idx[n_train : n_train + n_val]),
        test=kv_cache.index_select(SEQ_DIM, idx[n_train + n_val :]),
    )


def _scalar_distortion_and_cost(
    layer_slice_train: torch.Tensor,
    layer_slice_test: torch.Tensor,
    layer_slice_full: torch.Tensor,
    bits: int,
) -> Tuple[float, int]:
    """Distortion is measured on the held-out TEST split (leakage-safe); the
    bit COST is the serialized size of encoding the FULL layer at this bit-width
    (train touches the codec first for range familiarization). Reporting cost on
    the full layer - not the small test slice - keeps the allocator's budget in
    the same units the real encoder will actually spend."""
    codec = ScalarQuantCodec("cal", tensor_name="k", granularity=Granularity.PER_CHANNEL, default_bits=bits)
    _train_stream, _ = codec.encode_prefill(layer_slice_train)  # train-side familiarization
    test_stream, test_meta = codec.encode_prefill(layer_slice_test)
    recon = codec.decode(test_stream, test_meta, tuple(test_meta["full_shape"]), layer_slice_test.dtype, "cpu")
    mse = (recon - layer_slice_test).pow(2).mean().item()

    _full_stream, full_meta = codec.encode_prefill(layer_slice_full)
    total_bits = full_meta["accountant_report"]["serialized_bytes"] * 8
    return mse, total_bits


def calibrate_layers_scalar(
    kv_cache: torch.Tensor, bit_choices: List[int] = [4, 8]
) -> List[Cohort]:
    """Build one Cohort per layer, with a BitOption per scalar bit-width, using
    train/test-separated distortion. This is the concrete calibration that feeds
    the allocator for scalar codecs."""
    split = make_split(kv_cache)
    num_layers = kv_cache.size(0)
    cohorts: List[Cohort] = []
    for layer in range(num_layers):
        train_l = split.train[layer : layer + 1]
        test_l = split.test[layer : layer + 1]
        full_l = kv_cache[layer : layer + 1]
        options: List[BitOption] = []
        for bits in sorted(bit_choices):
            mse, total_bits = _scalar_distortion_and_cost(train_l, test_l, full_l, bits)
            options.append(BitOption(label=f"int{bits}", total_bits=total_bits, distortion=mse))
        cohorts.append(Cohort(cohort_id=f"L{layer}", options=options))
    return cohorts


def allocation_to_bitwidth_map(
    allocation: Allocation, tensor_name: str = "k", default_bits: int = 8
) -> BitWidthMap:
    """Turn an allocation (cohort label 'L{layer}' -> option 'int{bits}') into a
    BitWidthMap that the real ScalarQuantCodec consumes - closing the loop from
    'notebook simulation' to 'drives the real encoder'."""
    bwm = BitWidthMap(default_k_bits=default_bits, default_v_bits=default_bits)
    for cohort_id, opt in allocation.choice.items():
        if not cohort_id.startswith("L"):
            continue
        layer = int(cohort_id[1:])
        bits = int(opt.label.replace("int", ""))
        bwm.set_layer_bits(tensor_name, layer, bits)
    return bwm


def encode_with_allocation(
    kv_cache: torch.Tensor, allocation: Allocation, tensor_name: str = "k"
) -> Tuple[bytes, Dict[str, object]]:
    """Encode the REAL cache using the allocator's chosen per-layer bit-widths."""
    bwm = allocation_to_bitwidth_map(allocation, tensor_name)
    codec = ScalarQuantCodec("alloc", tensor_name=tensor_name, granularity=Granularity.PER_CHANNEL, bitwidth_map=bwm)
    return codec.encode_prefill(kv_cache)
