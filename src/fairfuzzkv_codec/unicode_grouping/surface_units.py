import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

import regex

from fairfuzzkv_codec.unicode_grouping.graphemes import GraphemeCluster, segment_graphemes
from fairfuzzkv_codec.unicode_grouping.scripts import is_emoji_cluster
from fairfuzzkv_codec.unicode_grouping.schema import SurfaceUnitType

# ponytail: pragmatic scheme/www prefix match, not a full RFC 3986 URL
# validator. Good enough to keep a URL as one surface unit instead of
# shredding it into dozens of word/punctuation units.
_URL_PATTERN = regex.compile(r"(?:https?://|www\.)\S+", regex.IGNORECASE)


@dataclass(frozen=True)
class SurfaceUnitDraft:
    unit_type: SurfaceUnitType
    start: int  # codepoint offset, inclusive
    end: int    # codepoint offset, exclusive
    text: str


def _find_url_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _URL_PATTERN.finditer(text)]


def _classify_cluster(cluster: GraphemeCluster) -> SurfaceUnitType:
    if is_emoji_cluster(cluster.text):
        return SurfaceUnitType.EMOJI
    if cluster.text.isspace():
        return SurfaceUnitType.WHITESPACE
    category = unicodedata.category(cluster.text[0])
    if category.startswith("N"):
        return SurfaceUnitType.NUMBER
    if category.startswith("P"):
        return SurfaceUnitType.PUNCTUATION
    if category.startswith("L") or category.startswith("M"):
        return SurfaceUnitType.WORD
    return SurfaceUnitType.OTHER


def build_surface_units(text: str) -> List[SurfaceUnitDraft]:
    """Partition text into surface units. Structurally guaranteed to cover the
    entire input exactly once, in order, with no gaps or overlaps, and to
    never split a grapheme cluster across two units - every cluster is
    consumed exactly once via URL-capture, an EMOJI singleton, or a same-class
    run merge."""
    clusters = segment_graphemes(text)
    url_spans = _find_url_spans(text)

    units: List[SurfaceUnitDraft] = []
    i = 0
    url_idx = 0
    n = len(clusters)

    while i < n:
        cluster = clusters[i]

        if url_idx < len(url_spans) and cluster.start == url_spans[url_idx][0]:
            _, u_end = url_spans[url_idx]
            j = i
            while j < n and clusters[j].start < u_end:
                j += 1
            # Grapheme-cluster integrity wins over the regex's exact end: if a
            # cluster straddles the URL boundary, include it whole.
            run_end = clusters[j - 1].end
            units.append(SurfaceUnitDraft(SurfaceUnitType.URL, cluster.start, run_end, text[cluster.start:run_end]))
            i = j
            url_idx += 1
            continue

        cls = _classify_cluster(cluster)

        if cls == SurfaceUnitType.EMOJI:
            units.append(SurfaceUnitDraft(cls, cluster.start, cluster.end, cluster.text))
            i += 1
            continue

        j = i
        run_end = cluster.end
        while j + 1 < n:
            nxt = clusters[j + 1]
            if url_idx < len(url_spans) and nxt.start == url_spans[url_idx][0]:
                break
            if _classify_cluster(nxt) != cls:
                break
            j += 1
            run_end = nxt.end

        units.append(SurfaceUnitDraft(cls, cluster.start, run_end, text[cluster.start:run_end]))
        i = j + 1

    return units
