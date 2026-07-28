from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

INDIC_SCHEMA_VERSION = 1


class LanguageCondition(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"
    TELUGU_ENGLISH = "te_en"


class TaskFamily(str, Enum):
    RETRIEVAL = "retrieval"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    COUNTING = "counting"
    AGGREGATION = "aggregation"


class ContentProvenance(str, Enum):
    LLM_GENERATED_TEMPLATE = "llm_generated_template"


class IndicVariant(BaseModel):
    schema_version: int = INDIC_SCHEMA_VERSION
    group_id: str
    language: LanguageCondition
    task_family: TaskFamily
    context_text: str
    question_text: str
    canonical_answer: int  # single digit 0-9, language-independent by design
    evidence_count: int
    evidence_position_index: int
    distractor_count: int
    context_token_count: int
    provenance: ContentProvenance = ContentProvenance.LLM_GENERATED_TEMPLATE
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


class IndicGroup(BaseModel):
    schema_version: int = INDIC_SCHEMA_VERSION
    group_id: str
    task_family: TaskFamily
    canonical_answer: int  # the parallelism anchor - identical across every language variant
    evidence_count: int
    evidence_position_index: int
    distractor_count: int
    variants: Dict[LanguageCondition, IndicVariant]

    def get_variant(self, language: LanguageCondition) -> IndicVariant:
        return self.variants[language]


class ValidationResult(BaseModel):
    passed: bool
    check_name: str
    detail: str = ""


class GroupValidationReport(BaseModel):
    group_id: str
    passed: bool
    results: List[ValidationResult]


class FragilityDistribution(BaseModel):
    language: LanguageCondition
    tokenizer_name: str
    num_units_scored: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    cohort_counts: Dict[str, int]  # cohort band label -> count


class DatasetCard(BaseModel):
    schema_version: int = INDIC_SCHEMA_VERSION
    dataset_name: str = "IndicLongComp"
    scale: str  # "course" or "journal"
    languages: Tuple[LanguageCondition, ...]
    task_families: Tuple[TaskFamily, ...]
    num_groups: int
    num_variants: int
    generation_seed: int
    split_hash: str
    content_provenance_note: str
    license_note: str
    pii_review_note: str
    contamination_check_note: str
    dedup_note: str
    parallelism_check_note: str
    fragility_distributions: List[FragilityDistribution] = Field(default_factory=list)
