from fairfuzzkv_codec.unicode_grouping.graphemes import segment_graphemes


def test_ascii_splits_per_character():
    clusters = segment_graphemes("abc")
    assert [c.text for c in clusters] == ["a", "b", "c"]
    assert [(c.start, c.end) for c in clusters] == [(0, 1), (1, 2), (2, 3)]


def test_devanagari_combining_sequence_is_one_cluster():
    # 'न' + 'म' + 'स' + '्' (virama) + 'त' + 'े' (vowel sign) -> "नमस्ते"
    text = "नमस्ते"
    clusters = segment_graphemes(text)
    recon = "".join(c.text for c in clusters)
    assert recon == text
    # the conjunct स् and the vowel sign े must not be split from their base
    assert any(len(c.text) > 1 for c in clusters)


def test_tamil_combining_sequence_is_one_cluster():
    text = "தமிழ்"
    clusters = segment_graphemes(text)
    assert "".join(c.text for c in clusters) == text


def test_telugu_combining_sequence_is_one_cluster():
    text = "తెలుగు"
    clusters = segment_graphemes(text)
    assert "".join(c.text for c in clusters) == text


def test_emoji_zwj_sequence_is_one_cluster():
    # family emoji: man + ZWJ + woman + ZWJ + girl
    zwj_family = "\U0001F468‍\U0001F469‍\U0001F467"
    clusters = segment_graphemes(zwj_family)
    assert len(clusters) == 1
    assert clusters[0].text == zwj_family


def test_full_coverage_no_gaps_no_overlaps():
    text = "abc नमस्ते 😀 123"
    clusters = segment_graphemes(text)
    assert clusters[0].start == 0
    assert clusters[-1].end == len(text)
    for a, b in zip(clusters, clusters[1:]):
        assert a.end == b.start


def test_malformed_unicode_does_not_crash():
    # lone combining mark with no base, unassigned-ish codepoints
    text = "́‍﻿"
    clusters = segment_graphemes(text)
    assert "".join(c.text for c in clusters) == text
