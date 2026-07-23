from typing import Dict, Tuple

from fairfuzzkv_codec.fragility_estimation.leakage import validate_no_leakage
from fairfuzzkv_codec.fragility_estimation.schema import FeatureVector, RiskScore

# Hand-specified, fixed weights - NOT learned/fit. This is the mandatory
# audit baseline: every number here is chosen and documented by a human,
# not "presented as learned truth" (non-negotiable instruction). Weights
# sum to 1.0; every term is monotone (higher feature value -> higher or
# equal risk contribution, never decreasing).
TRANSPARENT_WEIGHTS: Dict[str, float] = {
    "num_subtokens": 0.15,
    "continuation_ratio": 0.15,
    "script_transitions": 0.10,
    "normalization_sensitivity": 0.10,
    "rare_token_indicator": 0.15,
    "boundary_mismatch": 0.15,
    "token_cost_inflation": 0.20,
}
assert abs(sum(TRANSPARENT_WEIGHTS.values()) - 1.0) < 1e-9

# Fixed, documented caps used to map an unbounded feature onto [0, 1] before
# weighting. Not fit to data - chosen as generous-but-finite bounds so a
# single extreme unit can't dominate the score.
_CAPS = {
    "num_subtokens": 8.0,
    "script_transitions": 3.0,
}


def _normalize(name: str, value: float) -> float:
    if name == "num_subtokens":
        return min(value / _CAPS["num_subtokens"], 1.0)
    if name == "script_transitions":
        return min(value / _CAPS["script_transitions"], 1.0)
    if name == "token_cost_inflation":
        # 1.0x (matches English reference) -> 0 risk; 3.0x -> max risk
        return min(max((value - 1.0) / 2.0, 0.0), 1.0)
    # continuation_ratio, normalization_sensitivity, rare_token_indicator,
    # boundary_mismatch are already defined on [0, 1]
    return min(max(value, 0.0), 1.0)


def transparent_monotone_score(feature_vector: FeatureVector) -> RiskScore:
    """The mandatory transparent audit-baseline estimator: a fixed, monotone,
    human-specified weighted sum. score is always in [0, 1]; contributions
    sum exactly to score, so every point is fully feature-level explainable."""
    feature_dict = feature_vector.to_feature_dict()
    validate_no_leakage(feature_dict)

    contributions: Dict[str, float] = {}
    total = 0.0
    for name, weight in TRANSPARENT_WEIGHTS.items():
        normalized = _normalize(name, feature_dict[name])
        contribution = weight * normalized
        contributions[name] = contribution
        total += contribution

    return RiskScore(
        unit_char_span=feature_vector.unit_char_span,
        score=total,
        feature_contributions=contributions,
    )


def transparent_scores_and_labels(
    feature_vectors: list, proxy_labels: list
) -> Tuple[list, list]:
    """Helper for calibration comparisons: transparent score for each unit
    plus its aligned proxy label, as parallel lists."""
    scores = [transparent_monotone_score(fv).score for fv in feature_vectors]
    return scores, list(proxy_labels)
