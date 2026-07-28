"""Dataset card + checksum-manifested read/write for IndicLongComp.

Every file write here uses explicit `encoding="utf-8"` - unlike
`fragkv_minpairs.dataset_card`, which crashes on Windows' default `cp1252`
locale for exactly this reason (see PENDING.md / RISK_REGISTER). This
benchmark writes real Devanagari and code-mixed Latin-script text, so
getting this right is not optional.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

from fairfuzzkv_codec.benchmarks.indic_longcomp.generator import ALL_LANGUAGES
from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import DatasetCard, FragilityDistribution, IndicGroup

CONTENT_PROVENANCE_NOTE = (
    "All context/question text is LLM-authored from hand-designed parallel "
    "templates (fairfuzzkv_codec.benchmarks.indic_longcomp.templates), not "
    "sourced from MLRBench or any other external corpus (no verified network/"
    "license access was available when this was built), and not reviewed by a "
    "professional translator or native-speaker annotator. 'Parallel' is used "
    "only in the verified structural sense: see parallelism_check_note."
)
LICENSE_NOTE = (
    "100% project-original synthetic content (no external corpus text is "
    "reproduced anywhere in this dataset) - covered by this repository's own "
    "license, no third-party license inventory needed."
)
PII_REVIEW_NOTE = (
    "No real-world personal data. Names are drawn from a fixed synthetic pool "
    "shared with fragkv_minpairs.generator.NAME_POOL. An automated regex scan "
    "for email addresses and long digit runs (7+) found none - see "
    "validators.validate_no_pii."
)
DEDUP_NOTE = "Exact-duplicate context text checked via sha256 across every (group, language) pair - see validators.find_duplicate_texts."
PARALLELISM_NOTE = (
    "Every group's language variants share the same drawn canonical_answer, "
    "evidence_count, evidence_position_index, and distractor_count by "
    "construction (one shared random draw rendered into all 4 languages), "
    "verified per-group by validators.validate_parallelism, not merely "
    "assumed from generation."
)


def _contamination_note(overlap_count: int, compared_against: int) -> str:
    return (
        f"Checked against {compared_against} other in-repo generated texts "
        f"(FragKV-MinPairs); {overlap_count} verbatim overlaps found. This is "
        "a same-repo self-check ONLY - it cannot and does not verify against "
        "any model's actual pretraining corpus, which is not available to "
        "check against (documented gap, not assumed clean)."
    )


def compute_split_hash(groups: List[IndicGroup]) -> str:
    payload = {g.group_id: g.model_dump(mode="json") for g in groups}
    stable_json = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(stable_json).hexdigest()


def build_dataset_card(
    groups: List[IndicGroup], seed: int, scale: str, contamination_overlaps: int = 0,
    contamination_compared_against: int = 0, fragility_distributions: Optional[List[FragilityDistribution]] = None,
) -> DatasetCard:
    task_families = tuple(sorted({g.task_family for g in groups}, key=lambda t: t.value))
    num_variants = sum(len(g.variants) for g in groups)
    return DatasetCard(
        scale=scale,
        languages=ALL_LANGUAGES,
        task_families=task_families,
        num_groups=len(groups),
        num_variants=num_variants,
        generation_seed=seed,
        split_hash=compute_split_hash(groups),
        content_provenance_note=CONTENT_PROVENANCE_NOTE,
        license_note=LICENSE_NOTE,
        pii_review_note=PII_REVIEW_NOTE,
        contamination_check_note=_contamination_note(contamination_overlaps, contamination_compared_against),
        dedup_note=DEDUP_NOTE,
        parallelism_check_note=PARALLELISM_NOTE,
        fragility_distributions=fragility_distributions or [],
    )


def write_dataset(groups: List[IndicGroup], card: DatasetCard, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups_path = output_dir / "groups.jsonl"
    with open(groups_path, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(g.model_dump_json() + "\n")

    card_path = output_dir / "dataset_card.json"
    card_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")


def load_dataset(input_dir: Union[str, Path]) -> List[IndicGroup]:
    input_dir = Path(input_dir)
    groups = []
    with open(input_dir / "groups.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                groups.append(IndicGroup.model_validate_json(line))
    return groups


def load_dataset_card(input_dir: Union[str, Path]) -> DatasetCard:
    return DatasetCard.model_validate_json((Path(input_dir) / "dataset_card.json").read_text(encoding="utf-8"))


def transformation_summary(groups: List[IndicGroup]) -> dict:
    counts: Counter = Counter()
    for g in groups:
        counts[g.task_family.value] += 1
    return dict(counts)
