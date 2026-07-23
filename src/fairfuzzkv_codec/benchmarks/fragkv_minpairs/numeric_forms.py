import re
import unicodedata
from typing import Any, List, Optional, Tuple

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import TransformationType

WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
DIGIT_BY_WORD = {v: k for k, v in WORDS.items()}

ZWNJ = "‌"
ZWJ = "‍"
DOUBLE_SEP = ZWNJ + ZWJ

# Every rendering is a value in a specific TransformationType member. This is
# a small, fixed, empirically-verified ladder (see conversation record: real
# token counts measured against Qwen2.5-0.5B's tokenizer) - not a general
# search algorithm, since the domain (10 digits) is tiny and closed.
# ponytail: exhaustive combinatorial search over separator/fullwidth states
# would also work but costs far more tokenizer calls for no real benefit here.


def _fullwidth_letter(c: str) -> str:
    if c.isalpha():
        return chr(0xFF41 + (ord(c) - ord("a")))
    return c


def render_digit(value: int) -> str:
    return str(value)


def render_digit_dot(value: int) -> str:
    return f"{value}."


def _word_joined(value: int, sep: str, fullwidth: bool) -> str:
    word = WORDS[str(value)]
    letters = [_fullwidth_letter(c) if fullwidth else c for c in word]
    return sep.join(letters)


def render_word_hyphen_letters(value: int) -> str:
    return _word_joined(value, "-", fullwidth=False)


def render_word_dot_letters(value: int) -> str:
    return _word_joined(value, ".", fullwidth=False)


def render_word_zwnj_letters(value: int) -> str:
    return _word_joined(value, ZWNJ, fullwidth=False)


def render_word_zwnj_fullwidth_letters(value: int) -> str:
    return _word_joined(value, ZWNJ, fullwidth=True)


def render_word_double_sep_letters(value: int) -> str:
    return _word_joined(value, DOUBLE_SEP, fullwidth=False)


def render_word_double_sep_fullwidth_letters(value: int) -> str:
    return _word_joined(value, DOUBLE_SEP, fullwidth=True)


# Ladder order matters: this is the fixed escalation sequence tried when
# searching for a rendering that measures to a target token count.
RENDER_LADDER: List[Tuple[TransformationType, Any]] = [
    (TransformationType.DIGIT, render_digit),
    (TransformationType.DIGIT_DOT, render_digit_dot),
    (TransformationType.WORD_HYPHEN_LETTERS, render_word_hyphen_letters),
    (TransformationType.WORD_DOT_LETTERS, render_word_dot_letters),
    (TransformationType.WORD_ZWNJ_LETTERS, render_word_zwnj_letters),
    (TransformationType.WORD_ZWNJ_FULLWIDTH_LETTERS, render_word_zwnj_fullwidth_letters),
    (TransformationType.WORD_DOUBLE_SEP_LETTERS, render_word_double_sep_letters),
    (TransformationType.WORD_DOUBLE_SEP_FULLWIDTH_LETTERS, render_word_double_sep_fullwidth_letters),
]


def measure_token_count(rendering: str, tokenizer: Any) -> int:
    """Token count of `rendering` as it appears after the shared 'code '
    delimiter in context - measured with a leading space so BPE merge
    behavior matches real placement, minus the one token that space alone
    contributes (shared across every rendering, so it's not part of the
    evidence span's own fragmentation)."""
    with_space = tokenizer(" " + rendering, add_special_tokens=False).input_ids
    bare_space = tokenizer(" ", add_special_tokens=False).input_ids
    return len(with_space) - len(bare_space)


def find_rendering_for_target(
    value: int, target_n_g: int, tokenizer: Any, tolerance: int = 0
) -> Optional[Tuple[str, TransformationType, int]]:
    """Walk the fixed ladder, measuring real token counts, and return the
    first rendering within `tolerance` of target_n_g. Returns None if nothing
    on the ladder gets close enough - an honest, reportable failure rather
    than a fabricated match."""
    for transformation_type, render_fn in RENDER_LADDER:
        rendering = render_fn(value)
        realized = measure_token_count(rendering, tokenizer)
        if abs(realized - target_n_g) <= tolerance:
            return rendering, transformation_type, realized
    return None


_WORD_PATTERN = re.compile("|".join(sorted(DIGIT_BY_WORD, key=len, reverse=True)))


def parse_value(text: str) -> Optional[int]:
    """Recover the canonical single digit from arbitrary (possibly noisy,
    model-generated) text. Tries digit-run extraction first (after NFKC
    normalization, which folds fullwidth digits/letters back to ASCII), then
    falls back to number-word matching. Returns None if nothing parses -
    graded as incorrect, never guessed."""
    if not text:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    # Strip the separators our own renderings use (and plain whitespace) so
    # zero-width-joined or hyphen/dot-joined letters read as one contiguous
    # word again before matching - NFKC alone only folds fullwidth forms.
    cleaned = normalized
    for sep in (ZWNJ, ZWJ, "-", ".", " "):
        cleaned = cleaned.replace(sep, "")

    digit_match = re.search(r"\d", cleaned)
    word_match = _WORD_PATTERN.search(cleaned.lower())

    # Prefer whichever match starts earliest, since the answer is expected
    # at the front of the model's continuation.
    candidates = []
    if digit_match:
        candidates.append((digit_match.start(), int(digit_match.group())))
    if word_match:
        candidates.append((word_match.start(), int(DIGIT_BY_WORD[word_match.group()])))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]
