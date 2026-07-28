"""Per-language tokenizer-fragility distributions and cohort coverage
(Prompt 15 item 105) - entirely reused Module 2 infrastructure
(`fragility_estimation.compute_fragility_report` + `.cohorts`), just wired
across this benchmark's four language conditions. No new fragility
algorithm; this module only aggregates and reports.
"""

import statistics
from typing import Any, Dict, List

from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import FragilityDistribution, IndicGroup, LanguageCondition
from fairfuzzkv_codec.fragility_estimation.cohorts import build_cohort_definition
from fairfuzzkv_codec.fragility_estimation.pipeline import compute_fragility_report


def per_language_fragility(
    groups: List[IndicGroup], tokenizer: Any, tokenizer_name: str, min_band_samples: int = 3,
) -> Dict[LanguageCondition, FragilityDistribution]:
    """Runs Module 2's real fragility pipeline on every context text, pooled
    by language, and reports the score distribution plus quantile-cohort
    coverage - the same cohort machinery Module 2 already uses elsewhere."""
    scores_by_language: Dict[LanguageCondition, List[float]] = {}
    for group in groups:
        for language, variant in group.variants.items():
            report = compute_fragility_report(variant.context_text, tokenizer)
            scores_by_language.setdefault(language, []).extend(rs.score for rs in report.risk_scores)

    distributions: Dict[LanguageCondition, FragilityDistribution] = {}
    for language, scores in scores_by_language.items():
        if not scores:
            continue
        definition = build_cohort_definition(
            scores, tokenizer_name=tokenizer_name, corpus_id=f"indic_longcomp_{language.value}",
            min_band_samples=min(min_band_samples, max(1, len(scores) // 4)),
        )
        distributions[language] = FragilityDistribution(
            language=language, tokenizer_name=tokenizer_name, num_units_scored=len(scores),
            mean_score=statistics.mean(scores), std_score=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            min_score=min(scores), max_score=max(scores),
            cohort_counts={band.label: band.count for band in definition.bands},
        )
    return distributions
