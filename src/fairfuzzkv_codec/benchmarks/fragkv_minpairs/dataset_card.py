import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import List, Union

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import TOKEN_COUNT_TOLERANCE
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import DatasetCard, MinPairGroup


def compute_split_hash(groups: List[MinPairGroup]) -> str:
    """Immutable content hash of the dataset: sha256 over the sorted,
    canonical JSON of every group, keyed by group_id so ordering never
    affects the hash. Any change to any group's content changes this."""
    payload = {g.group_id: g.model_dump(mode="json") for g in groups}
    stable_json = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(stable_json).hexdigest()


def build_dataset_card(
    groups: List[MinPairGroup], tokenizer_name: str, seed: int
) -> DatasetCard:
    transformation_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()
    for g in groups:
        difficulty_counts[g.difficulty] += 1
        for variant in g.variants.values():
            transformation_counts[variant.transformation_type.value] += 1

    return DatasetCard(
        tokenizer_name=tokenizer_name,
        num_groups=len(groups),
        generation_seed=seed,
        split_hash=compute_split_hash(groups),
        transformation_counts=dict(transformation_counts),
        difficulty_counts=dict(difficulty_counts),
        token_count_tolerance=dict(TOKEN_COUNT_TOLERANCE),
        notes=(
            "Canonical evidence values are single digits 0-9 (see numeric_forms.py). "
            "n_g=1/2 hit their token-count target exactly for every digit; n_g=4/8 "
            "are matched within +-1 token (empirically verified against this tokenizer, "
            "not assumed) - see token_count_tolerance."
        ),
    )


def write_dataset(groups: List[MinPairGroup], card: DatasetCard, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups_path = output_dir / "groups.jsonl"
    with open(groups_path, "w") as f:
        for g in groups:
            f.write(g.model_dump_json() + "\n")

    card_path = output_dir / "dataset_card.json"
    card_path.write_text(card.model_dump_json(indent=2))


def load_dataset(input_dir: Union[str, Path]) -> List[MinPairGroup]:
    input_dir = Path(input_dir)
    groups = []
    with open(input_dir / "groups.jsonl") as f:
        for line in f:
            if line.strip():
                groups.append(MinPairGroup.model_validate_json(line))
    return groups


def load_dataset_card(input_dir: Union[str, Path]) -> DatasetCard:
    return DatasetCard.model_validate_json((Path(input_dir) / "dataset_card.json").read_text())
