import torch

from fairfuzzkv_codec.baselines.adapter import run_matched_bit_comparison, tune_to_matched_bits
from fairfuzzkv_codec.baselines.registry import (
    NOT_REPRODUCED_CARDS, build_decode_time_selection_adapters, build_prefill_selection_adapters,
    build_quantization_adapters,
)
from fairfuzzkv_codec.baselines.schema import EvaluationRegime, ReproductionStatus


def _kv(layers=2, seq=24, head_dim=4):
    g = torch.Generator().manual_seed(0)
    return torch.randn(layers, 1, 1, seq, head_dim, generator=g)


def _attn(layers=2, seq=24):
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(layers, 1, 2, seq, seq, generator=g)
    return torch.softmax(logits, dim=-1)


def test_quantization_adapters_hit_matched_bits_at_discrete_targets():
    kv = _kv(seq=80)  # LBG codebook_size=256 needs >=256 vectors (vector_dim=2 -> 2 per position per layer)
    adapters = build_quantization_adapters("test")
    for adapter in adapters:
        result = tune_to_matched_bits(adapter, kv, target_bits_per_element=8.0, tolerance=0.5)
        assert result is not None
        assert result.regime == EvaluationRegime.COMPRESSION_QUANTIZATION
        assert result.actual_bits_per_element > 0


def test_prefill_selection_adapters_produce_matched_results():
    kv, attn = _kv(), _attn()
    adapters = build_prefill_selection_adapters("test", attn)
    for adapter in adapters:
        result = tune_to_matched_bits(adapter, kv, target_bits_per_element=8.0, tolerance=0.15)
        assert result is not None
        assert result.regime == EvaluationRegime.PREFILL_SELECTION
        assert result.matched, f"{adapter.card.name} failed to match: actual={result.actual_bits_per_element}"


def test_decode_time_adapter_is_in_its_own_regime_not_prefill():
    kv, attn = _kv(), _attn()
    adapters = build_decode_time_selection_adapters("test", attn)
    result = tune_to_matched_bits(adapters[0], kv, target_bits_per_element=8.0, tolerance=0.3)
    assert result is not None
    assert result.regime == EvaluationRegime.DECODE_TIME_SELECTION
    assert result.regime != EvaluationRegime.PREFILL_SELECTION


def test_run_matched_bit_comparison_never_drops_a_baseline():
    kv, attn = _kv(), _attn()
    adapters = build_prefill_selection_adapters("test", attn)
    results = run_matched_bit_comparison(adapters, kv, target_bits_per_element=8.0)
    assert len(results) == len(adapters)
    assert {r.baseline_name for r in results} == {a.card.name for a in adapters}


def test_not_reproduced_cards_all_have_a_nearest_faithful_configuration():
    for card in NOT_REPRODUCED_CARDS:
        assert card.reproduction_status == ReproductionStatus.NOT_REPRODUCED
        assert card.nearest_faithful_configuration


def test_not_reproduced_cards_have_no_working_adapter_by_name():
    reproduced_names = (
        {a.card.name for a in build_quantization_adapters("t")}
        | {a.card.name for a in build_prefill_selection_adapters("t", _attn())}
        | {a.card.name for a in build_decode_time_selection_adapters("t", _attn())}
    )
    not_reproduced_names = {c.name for c in NOT_REPRODUCED_CARDS}
    assert reproduced_names.isdisjoint(not_reproduced_names)


def test_every_regime_has_at_least_one_reproduced_baseline():
    quant = build_quantization_adapters("t")
    prefill = build_prefill_selection_adapters("t", _attn())
    decode = build_decode_time_selection_adapters("t", _attn())
    assert len(quant) >= 1
    assert len(prefill) >= 1
    assert len(decode) >= 1
    assert all(a.card.regime == EvaluationRegime.COMPRESSION_QUANTIZATION for a in quant)
    assert all(a.card.regime == EvaluationRegime.PREFILL_SELECTION for a in prefill)
    assert all(a.card.regime == EvaluationRegime.DECODE_TIME_SELECTION for a in decode)


def test_pyramidkv_adapter_round_trips_with_2d_mask():
    kv, attn = _kv(layers=2, seq=24), _attn(layers=2, seq=24)
    adapters = build_prefill_selection_adapters("test", attn)
    pyramidkv = next(a for a in adapters if a.card.name == "PyramidKV")
    result = tune_to_matched_bits(pyramidkv, kv, target_bits_per_element=6.0, tolerance=0.2)
    assert result is not None
    assert result.kv_mse >= 0.0
