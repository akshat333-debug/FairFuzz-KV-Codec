from fairfuzzkv_codec.benchmarks.indic_longcomp.dataset_card import (
    build_dataset_card, compute_split_hash, load_dataset, load_dataset_card, write_dataset,
)
from fairfuzzkv_codec.benchmarks.indic_longcomp.generator import generate_dataset
from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import LanguageCondition


def _dataset():
    return generate_dataset(groups_per_family=1, seed=5)


def test_split_hash_deterministic_and_content_sensitive():
    a = _dataset()
    b = _dataset()
    assert compute_split_hash(a) == compute_split_hash(b)

    tampered = list(b)
    tampered[0] = tampered[0].model_copy(update={"canonical_answer": (tampered[0].canonical_answer + 1) % 10})
    assert compute_split_hash(a) != compute_split_hash(tampered)


def test_dataset_card_covers_all_languages_and_documents_provenance():
    groups = _dataset()
    card = build_dataset_card(groups, seed=5, scale="course")
    assert set(card.languages) == set(LanguageCondition)
    assert "LLM-authored" in card.content_provenance_note
    assert "not" in card.content_provenance_note.lower()  # "not reviewed" / "not sourced"
    assert card.num_groups == len(groups)
    assert card.num_variants == len(groups) * 4


def test_write_and_load_dataset_round_trips_with_real_unicode(tmp_path):
    groups = _dataset()
    card = build_dataset_card(groups, seed=5, scale="course")
    write_dataset(groups, card, tmp_path)

    loaded_groups = load_dataset(tmp_path)
    loaded_card = load_dataset_card(tmp_path)

    assert len(loaded_groups) == len(groups)
    assert loaded_card.split_hash == card.split_hash

    # the actual Devanagari text must survive the round trip byte-for-byte -
    # this is exactly the class of bug PENDING.md documents for fragkv_minpairs.
    original_hindi = next(g for g in groups).variants[LanguageCondition.HINDI].context_text
    loaded_hindi = next(g for g in loaded_groups).variants[LanguageCondition.HINDI].context_text
    assert original_hindi == loaded_hindi
    assert "क" in loaded_hindi or "त" in loaded_hindi  # sanity: real Devanagari made it through
