from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

# Bump whenever GroupRecord/QuarantineRecord field semantics change, so audit
# reports remain inspectable/versioned per the module's non-negotiable contract.
MAPPER_SCHEMA_VERSION = 1


class NormalizationPolicy(str, Enum):
    PRESERVE_ORIGINAL = "preserve_original"
    NFC = "nfc"
    NFKC_ANALYSIS_ONLY = "nfkc_analysis_only"


class SurfaceUnitType(str, Enum):
    WORD = "word"
    PUNCTUATION = "punctuation"
    WHITESPACE = "whitespace"
    NUMBER = "number"
    URL = "url"
    EMOJI = "emoji"
    OTHER = "other"


class QuarantineReason(str, Enum):
    INVALID_OFFSET = "invalid_offset"
    OUT_OF_BOUNDS = "out_of_bounds"
    NO_OVERLAP = "no_overlap"


class GroupRecord(BaseModel):
    schema_version: int = MAPPER_SCHEMA_VERSION
    unit_type: SurfaceUnitType
    char_span: Tuple[int, int]        # [start, end) codepoint offsets into the original text
    byte_span: Tuple[int, int]        # [start, end) UTF-8 byte offsets into the original text
    normalized_span: Tuple[int, int]  # [start, end) offsets into the normalization-only analysis copy
    original_text: str                # verbatim slice of the untouched original text
    script_profile: List[str]
    language_hint: Optional[str] = None
    token_indices: List[int] = Field(default_factory=list)
    token_count: int = 0
    is_special: bool = False
    alignment_confidence: float = 1.0


class QuarantineRecord(BaseModel):
    schema_version: int = MAPPER_SCHEMA_VERSION
    token_index: int
    token_text: str
    reason: QuarantineReason
    detail: str
