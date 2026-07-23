from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

# Bump whenever generator/variant semantics change, so dataset manifests
# remain inspectable/versioned and split hashes stay meaningful.
FRAGKV_SCHEMA_VERSION = 1

FRAGMENTATION_LEVELS = (1, 2, 4, 8)


class TransformationType(str, Enum):
    """How the evidence value's surface form was rendered. Every member must
    be exactly reversible back to the canonical digit via numeric_forms.parse_value
    - this is what "semantic equivalence verified" means for synthetic,
    slot-filled content (there is no natural-language paraphrase to check
    with an NLI model; equivalence is a deterministic round-trip)."""

    DIGIT = "digit"
    DIGIT_DOT = "digit_dot"
    WORD_HYPHEN_LETTERS = "word_hyphen_letters"
    WORD_DOT_LETTERS = "word_dot_letters"
    WORD_ZWNJ_LETTERS = "word_zwnj_letters"
    WORD_ZWNJ_FULLWIDTH_LETTERS = "word_zwnj_fullwidth_letters"
    WORD_DOUBLE_SEP_LETTERS = "word_double_sep_letters"
    WORD_DOUBLE_SEP_FULLWIDTH_LETTERS = "word_double_sep_fullwidth_letters"


class MinPairVariant(BaseModel):
    schema_version: int = FRAGKV_SCHEMA_VERSION
    group_id: str
    n_g_target: int
    n_g_realized: int
    canonical_value: int  # the single digit 0-9 this variant's evidence encodes
    evidence_rendering: str
    transformation_type: TransformationType
    subject_name: str
    context_text: str
    question_text: str
    gold_answer_rendering: str  # == evidence_rendering; what a faithful copy would echo
    evidence_position_index: int  # index of the evidence fact among all facts in context
    distractor_count: int
    difficulty: str
    context_token_count: int
    evidence_token_count: int  # == n_g_realized, measured independently as a cross-check
    provenance: Dict[str, Any] = Field(default_factory=dict)


class MinPairGroup(BaseModel):
    schema_version: int = FRAGKV_SCHEMA_VERSION
    group_id: str
    subject_name: str
    canonical_value: int
    distractor_count: int
    difficulty: str
    evidence_position_index: int
    variants: Dict[int, MinPairVariant]  # keyed by n_g_target

    def get_variant(self, n_g: int) -> MinPairVariant:
        return self.variants[n_g]


class ValidationResult(BaseModel):
    passed: bool
    check_name: str
    detail: str = ""


class GroupValidationReport(BaseModel):
    group_id: str
    passed: bool
    results: List[ValidationResult]


class DatasetCard(BaseModel):
    schema_version: int = FRAGKV_SCHEMA_VERSION
    dataset_name: str = "FragKV-MinPairs"
    tokenizer_name: str
    num_groups: int
    fragmentation_levels: Tuple[int, ...] = FRAGMENTATION_LEVELS
    generation_seed: int
    split_hash: str
    transformation_counts: Dict[str, int]
    difficulty_counts: Dict[str, int]
    token_count_tolerance: Dict[int, int]  # n_g_target -> allowed |realized - target|
    notes: str = ""
