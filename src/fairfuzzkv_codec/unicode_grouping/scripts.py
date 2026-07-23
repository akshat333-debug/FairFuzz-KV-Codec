import unicodedata
from typing import List

import regex

# Emoji detection via regex's Unicode property support (\p{Emoji_Presentation}
# covers default-emoji-style codepoints; \p{Extended_Pictographic} catches the
# rest, including ZWJ-joined sequences already grouped into one grapheme
# cluster upstream).
_EMOJI_PATTERN = regex.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")

# ponytail: coarse Unicode block ranges covering only the scripts named in the
# Prompt 3 spec (Latin, Devanagari/Hindi, Telugu, Tamil). Extend this table if
# more scripts are needed rather than pulling in a full ICU/unicodedata2
# dependency for a handful of ranges.
_SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Latin", 0x0041, 0x024F),
]


def detect_script(char: str) -> str:
    if _EMOJI_PATTERN.match(char):
        return "Emoji"
    cp = ord(char)
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    category = unicodedata.category(char)
    if category.startswith("N"):
        return "Common_Number"
    if category[0] in ("P", "Z", "C"):
        return "Common"
    return "Unknown"


def is_emoji_cluster(cluster_text: str) -> bool:
    return bool(_EMOJI_PATTERN.search(cluster_text))


def script_profile_for_text(text: str) -> List[str]:
    scripts: List[str] = []
    for ch in text:
        s = detect_script(ch)
        if s not in scripts:
            scripts.append(s)
    return scripts
