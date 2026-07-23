from dataclasses import dataclass
from typing import List

import regex

# \X matches one Unicode extended grapheme cluster per UAX #29 - this already
# keeps base+combining-mark sequences (Devanagari/Tamil/Telugu matras) and
# emoji ZWJ sequences together as a single unit, so no custom Indic/emoji
# clustering logic is needed here.
_GRAPHEME_PATTERN = regex.compile(r"\X")


@dataclass(frozen=True)
class GraphemeCluster:
    text: str
    start: int  # codepoint offset, inclusive
    end: int    # codepoint offset, exclusive


def segment_graphemes(text: str) -> List[GraphemeCluster]:
    """Partition text into extended grapheme clusters. Guaranteed to cover
    every character exactly once, in order, with no gaps - this is the
    invariant the rest of the module (surface units, alignment) relies on
    to guarantee round-trip coverage and to never split a grapheme cluster."""
    return [
        GraphemeCluster(text=m.group(), start=m.start(), end=m.end())
        for m in _GRAPHEME_PATTERN.finditer(text)
    ]
