from typing import Any, Dict

# ponytail: a small fixed canonical English paragraph, not a downloaded
# corpus - good enough to establish a "matched English" chars-per-token
# baseline per tokenizer. Extend with a real reference corpus if the
# baseline needs to be more representative.
REFERENCE_ENGLISH_TEXT = (
    "The quick brown fox jumps over the lazy dog. This sentence contains "
    "common English words and standard punctuation, providing a simple "
    "reference point for measuring how many characters a typical tokenizer "
    "spends per subword token on ordinary English text."
)

_cache: Dict[int, float] = {}


def get_reference_chars_per_token(tokenizer: Any) -> float:
    """Chars-per-token for REFERENCE_ENGLISH_TEXT under the given tokenizer,
    cached per tokenizer identity for the process lifetime."""
    key = id(tokenizer)
    if key not in _cache:
        encoding = tokenizer(REFERENCE_ENGLISH_TEXT, return_special_tokens_mask=True)
        special_mask = encoding["special_tokens_mask"]
        num_real_tokens = sum(1 for m in special_mask if not m)
        _cache[key] = len(REFERENCE_ENGLISH_TEXT) / num_real_tokens if num_real_tokens else 1.0
    return _cache[key]
