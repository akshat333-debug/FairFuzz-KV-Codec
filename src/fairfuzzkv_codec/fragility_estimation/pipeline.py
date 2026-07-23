from dataclasses import dataclass
from typing import Any, List

from fairfuzzkv_codec.fragility_estimation.features import compute_features_for_records
from fairfuzzkv_codec.fragility_estimation.schema import FeatureVector, RiskScore
from fairfuzzkv_codec.fragility_estimation.transparent_score import transparent_monotone_score
from fairfuzzkv_codec.unicode_grouping import GroupMapper, MapperResult


@dataclass
class FragilityReport:
    mapper_result: MapperResult
    feature_vectors: List[FeatureVector]
    risk_scores: List[RiskScore]


def compute_fragility_report(text: str, tokenizer: Any) -> FragilityReport:
    """One-shot entry point: Module 1 (surface units + token alignment) ->
    Module 2 features -> transparent audit-baseline risk score, per unit."""
    mapper = GroupMapper(tokenizer)
    result = mapper.map(text)
    feature_vectors = compute_features_for_records(result.records, tokenizer, text)
    risk_scores = [transparent_monotone_score(fv) for fv in feature_vectors]
    return FragilityReport(mapper_result=result, feature_vectors=feature_vectors, risk_scores=risk_scores)
