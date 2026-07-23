import os
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt

from fairfuzzkv_codec.unicode_grouping.schema import GroupRecord


def dominant_script_label(record: GroupRecord) -> str:
    """Descriptive-only label for the dashboard panel: NOT a fragility
    feature, never passed to transparent_monotone_score/cohorts (see
    leakage.py's ALLOWED_FEATURE_NAMES, which excludes script/language)."""
    if not record.script_profile:
        return "Unknown"
    if len(record.script_profile) > 1:
        return "Mixed"
    return record.script_profile[0]


def plot_risk_by_script(records: List[GroupRecord], risk_scores: List[float], output_dir: str) -> str:
    """Descriptive-only panel: risk score distribution grouped by script.
    Non-negotiable per Prompt 4 item 29: this grouping is for human audit
    only. The optimization objective must use risk cohorts (fragility_estimation.cohorts),
    never this script grouping - enforced structurally by leakage.py."""
    by_script = defaultdict(list)
    for record, score in zip(records, risk_scores):
        by_script[dominant_script_label(record)].append(score)

    scripts = sorted(by_script.keys())
    data = [by_script[s] for s in scripts]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=scripts)
    plt.ylabel("Transparent Risk Score")
    plt.xlabel("Script (descriptive grouping only - not used in optimization)")
    plt.title("Fragility Risk Score Distribution by Script (Descriptive Only)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fragility_risk_by_script.png")
    plt.savefig(out_path)
    plt.close()
    return out_path
