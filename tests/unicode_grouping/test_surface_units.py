import pytest
from hypothesis import given, settings, strategies as st

from fairfuzzkv_codec.unicode_grouping.schema import SurfaceUnitType
from fairfuzzkv_codec.unicode_grouping.surface_units import build_surface_units

# Curated corpus spanning the categories named in Prompt 3 item 21: English,
# Hindi, Telugu, Tamil, Hinglish (code-mixed), Telugu-English (code-mixed),
# emoji, punctuation-heavy text, URLs, numbers, malformed Unicode.
CURATED_CASES = [
    "The quick brown fox jumps over the lazy dog.",
    "नमस्ते, आप कैसे हैं? मैं ठीक हूँ।",
    "తెలుగు భాష చాలా అందంగా ఉంటుంది.",
    "தமிழ் மொழி மிகவும் அழகானது.",
    "Mujhe ye movie बहुत पसंद आई, kal फिर से देखूंगा!",
    "Nenu ఈరోజు office కి late గా వెళ్ళాను, chala tired ga undi.",
    "😀😃😄😁 party time!! 🎉🎊",
    "Wait...what?! Really?!?! No way... :-)",
    "Visit https://example.com/path?q=1&x=2 or www.test.org now.",
    "The year is 2026, price is 1234.56, and count is 007.",
    "Family emoji: \U0001F468‍\U0001F469‍\U0001F467 and flag: \U0001F1EE\U0001F1F3",
    "Zero-width test: a​b‌c‍d",  # ZWSP, ZWNJ, ZWJ
    "́‍﻿ combining marks and BOM-ish chars ​",
]


@pytest.mark.parametrize("text", CURATED_CASES)
def test_round_trip_coverage_exact(text):
    """Acceptance gate: round-trip span coverage must be exact - concatenating
    every surface unit's original_text in order must reconstruct the input
    exactly, with no gaps or overlaps."""
    units = build_surface_units(text)
    assert "".join(u.text for u in units) == text

    # contiguous, ordered, no gaps/overlaps
    assert units[0].start == 0
    assert units[-1].end == len(text)
    for a, b in zip(units, units[1:]):
        assert a.end == b.start


def test_round_trip_coverage_rate_meets_gate():
    failures = 0
    for text in CURATED_CASES:
        units = build_surface_units(text)
        if "".join(u.text for u in units) != text:
            failures += 1
    rate = 1 - (failures / len(CURATED_CASES))
    assert rate >= 0.995


def test_url_becomes_single_unit():
    text = "check https://example.com/a/b?c=1 out"
    units = build_surface_units(text)
    url_units = [u for u in units if u.unit_type == SurfaceUnitType.URL]
    assert len(url_units) == 1
    assert url_units[0].text == "https://example.com/a/b?c=1"


def test_consecutive_digits_merge_into_one_number_unit():
    units = build_surface_units("abc 12345 def")
    numbers = [u for u in units if u.unit_type == SurfaceUnitType.NUMBER]
    assert len(numbers) == 1
    assert numbers[0].text == "12345"


def test_emoji_units_never_merge():
    units = build_surface_units("😀😃")
    emoji_units = [u for u in units if u.unit_type == SurfaceUnitType.EMOJI]
    assert len(emoji_units) == 2


def test_code_mixed_text_word_units_have_multiple_scripts_overall():
    text = "Mujhe ये पसंद है"
    units = build_surface_units(text)
    scripts_seen = set()
    from fairfuzzkv_codec.unicode_grouping.scripts import script_profile_for_text
    for u in units:
        if u.unit_type == SurfaceUnitType.WORD:
            scripts_seen.update(script_profile_for_text(u.text))
    assert "Latin" in scripts_seen
    assert "Devanagari" in scripts_seen


@given(
    st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Po", "Zs"),
            max_codepoint=0x1F64F,
        ),
        max_size=60,
    )
)
@settings(max_examples=150, deadline=None)
def test_property_full_coverage_never_gaps_or_overlaps(text):
    units = build_surface_units(text)
    assert "".join(u.text for u in units) == text
    if units:
        assert units[0].start == 0
        assert units[-1].end == len(text)
        for a, b in zip(units, units[1:]):
            assert a.end == b.start
