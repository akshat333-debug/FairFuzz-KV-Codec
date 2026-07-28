import pytest

from fairfuzzkv_codec.benchmarks.indic_longcomp.generator import generate_dataset
from fairfuzzkv_codec.benchmarks.indic_longcomp.runner import IndicLongCompRunner, full_correct_group_ids
from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import TaskFamily

TINY_MODEL = "yujiepan/qwen2-tiny-random"


@pytest.fixture(scope="module")
def runner():
    return IndicLongCompRunner(TINY_MODEL, device="cpu")


def test_run_group_produces_one_record_per_language(runner):
    dataset = generate_dataset(groups_per_family=1, seed=1, tokenizer=runner.tokenizer, task_families=(TaskFamily.RETRIEVAL,))
    records = runner.run_group(dataset[0])
    assert len(records) == 4
    assert {r.language for r in records} == {"en", "hi", "hinglish", "te_en"}
    for r in records:
        assert isinstance(r.correct, bool)


def test_run_dataset_and_full_correct_group_ids(runner):
    dataset = generate_dataset(groups_per_family=2, seed=2, tokenizer=runner.tokenizer, task_families=(TaskFamily.RETRIEVAL,))
    records = runner.run_dataset(dataset)
    assert len(records) == len(dataset) * 4
    full_correct = full_correct_group_ids(records)
    assert full_correct.issubset({g.group_id for g in dataset})


def test_full_correct_requires_every_language_correct():
    from fairfuzzkv_codec.benchmarks.indic_longcomp.runner import IndicPredictionRecord

    records = [
        IndicPredictionRecord("g1", "en", "retrieval", True, "3", 3),
        IndicPredictionRecord("g1", "hi", "retrieval", True, "3", 3),
        IndicPredictionRecord("g1", "hinglish", "retrieval", False, "5", 5),
        IndicPredictionRecord("g1", "te_en", "retrieval", True, "3", 3),
    ]
    assert full_correct_group_ids(records) == set()  # one wrong language excludes the whole group

    records[2] = IndicPredictionRecord("g1", "hinglish", "retrieval", True, "3", 3)
    assert full_correct_group_ids(records) == {"g1"}
