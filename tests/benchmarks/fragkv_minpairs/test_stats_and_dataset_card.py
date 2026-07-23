from transformers import AutoTokenizer

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.dataset_card import (
    build_dataset_card,
    compute_split_hash,
    load_dataset,
    load_dataset_card,
    write_dataset,
)
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import generate_validated_dataset
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.stats_utils import PredictionRecord, compute_effect_size

MODEL = "Qwen/Qwen2.5-0.5B"


def _record(group_id, n_g, codec, correct):
    return PredictionRecord(
        group_id=group_id, n_g=n_g, codec_name=codec, target_bits_per_element=8.0,
        actual_bits_per_element=8.0, generated_text="x", parsed_value=None, correct=correct,
    )


def test_compute_effect_size_detects_strong_consistent_effect():
    records = []
    for i in range(150):
        gid = f"g{i}"
        records.append(_record(gid, 1, "codecA", True))
        records.append(_record(gid, 8, "codecA", i % 10 != 0))  # ~90% fail at n_g=8 -> True kept sparse
    # Rebuild cleanly: n_g=1 always correct, n_g=8 correct only 10% of the time
    records = []
    for i in range(150):
        gid = f"g{i}"
        records.append(_record(gid, 1, "codecA", True))
        records.append(_record(gid, 8, "codecA", i % 10 == 0))

    result = compute_effect_size(records, "codecA", low_n_g=1, high_n_g=8)
    assert result.n_paired_groups == 150
    assert result.effect_size > 0.8
    assert result.p_value < 0.01
    assert result.ci_low > 0.5


def test_compute_effect_size_no_effect_gives_high_p_value():
    records = []
    for i in range(150):
        gid = f"g{i}"
        correct = i % 2 == 0
        records.append(_record(gid, 1, "codecA", correct))
        records.append(_record(gid, 8, "codecA", correct))  # identical outcome -> zero effect

    result = compute_effect_size(records, "codecA", low_n_g=1, high_n_g=8)
    assert abs(result.effect_size) < 1e-9
    assert result.p_value > 0.5


def test_compute_effect_size_handles_no_paired_groups():
    result = compute_effect_size([], "codecA")
    assert result.n_paired_groups == 0
    assert result.p_value == 1.0


def test_write_and_load_dataset_round_trips(tmp_path):
    tok = AutoTokenizer.from_pretrained(MODEL)
    groups = generate_validated_dataset(10, tok, seed=55)
    card = build_dataset_card(groups, MODEL, seed=55)

    write_dataset(groups, card, tmp_path)
    reloaded_groups = load_dataset(tmp_path)
    reloaded_card = load_dataset_card(tmp_path)

    assert len(reloaded_groups) == len(groups)
    assert compute_split_hash(reloaded_groups) == card.split_hash
    assert reloaded_card.split_hash == card.split_hash
    assert reloaded_card.num_groups == len(groups)
