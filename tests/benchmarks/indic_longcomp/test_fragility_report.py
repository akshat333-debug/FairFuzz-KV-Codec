import pytest
from transformers import AutoTokenizer

from fairfuzzkv_codec.benchmarks.indic_longcomp.fragility_report import per_language_fragility
from fairfuzzkv_codec.benchmarks.indic_longcomp.generator import generate_dataset
from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import LanguageCondition

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def test_reports_a_distribution_for_every_language(tokenizer):
    groups = generate_dataset(groups_per_family=2, seed=13, tokenizer=tokenizer)
    distributions = per_language_fragility(groups, tokenizer, tokenizer_name=MODEL)
    assert set(distributions.keys()) == set(LanguageCondition)
    for dist in distributions.values():
        assert dist.num_units_scored > 0
        assert 0.0 <= dist.mean_score <= 1.0
        assert dist.min_score <= dist.mean_score <= dist.max_score
        assert sum(dist.cohort_counts.values()) == dist.num_units_scored


def test_non_english_languages_show_real_measured_fragility(tokenizer):
    # not a claim about WHICH language is more fragile (that would need a
    # much larger study) - just that Module 2's real pipeline actually ran
    # and produced non-trivial, non-identical numbers per language.
    groups = generate_dataset(groups_per_family=3, seed=21, tokenizer=tokenizer)
    distributions = per_language_fragility(groups, tokenizer, tokenizer_name=MODEL)
    scores = {lang: d.mean_score for lang, d in distributions.items()}
    assert len(set(round(s, 6) for s in scores.values())) > 1  # not all identical by coincidence
