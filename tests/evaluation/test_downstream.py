import pytest

from fairfuzzkv_codec.evaluation.downstream import (
    best_over_references,
    exact_match,
    normalize_answer,
    score_dataset,
    token_f1,
)


def test_normalization_casefolds_and_strips_punctuation_and_articles():
    assert normalize_answer("  The Cat, sat! ") == "cat sat"
    assert normalize_answer("A dog") == "dog"


def test_normalization_is_unicode_aware_not_ascii_only():
    # Devanagari danda is Unicode punctuation - must be stripped like a period.
    assert normalize_answer("मेरा नाम राहुल है।") == normalize_answer("मेरा नाम राहुल है")
    # NFKC folds fullwidth digits to ASCII
    assert normalize_answer("４２") == "42"


def test_normalization_does_not_strip_non_english_articles():
    """Applying an English article list to Indic text would corrupt answers;
    only a/an/the are removed."""
    assert "ye" in normalize_answer("Mujhe ye pasand hai")


def test_exact_match_basic():
    assert exact_match("The answer", "answer") == 1.0
    assert exact_match("42", "43") == 0.0


def test_token_f1_rewards_partial_overlap():
    assert token_f1("the red cat", "red cat") == 1.0  # articles stripped
    partial = token_f1("red cat sat", "red cat")
    assert 0.0 < partial < 1.0
    assert token_f1("dog", "cat") == 0.0


def test_token_f1_empty_conventions_are_explicit():
    assert token_f1("", "") == 1.0        # both empty -> match
    assert token_f1("cat", "") == 0.0     # one empty -> no credit
    assert token_f1("", "cat") == 0.0


def test_best_over_references_takes_the_max():
    assert best_over_references("cat", ["dog", "cat"], metric="em") == 1.0
    assert best_over_references("cat", ["dog", "bird"], metric="em") == 0.0


def test_best_over_references_rejects_empty_reference_list():
    """An empty gold list must raise, not silently look like a wrong answer."""
    with pytest.raises(ValueError):
        best_over_references("cat", [])


def test_score_dataset_means_and_alignment_check():
    scores = score_dataset(["cat", "dog"], ["cat", "bird"])
    assert scores.num_examples == 2
    assert scores.exact_match == 0.5
    with pytest.raises(ValueError):
        score_dataset(["a"], ["a", "b"])


def test_score_dataset_handles_empty_corpus():
    scores = score_dataset([], [])
    assert scores.num_examples == 0 and scores.exact_match == 0.0
