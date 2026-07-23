import random
from typing import Any, List, Optional

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.numeric_forms import (
    find_rendering_for_target,
    render_digit,
)
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import (
    FRAGMENTATION_LEVELS,
    MinPairGroup,
    MinPairVariant,
)

# Token-count tolerance per fragmentation target, empirically justified
# (see numeric_forms.RENDER_LADDER measurements against Qwen2.5-0.5B):
# n_g=1 and n_g=2 hit exactly for every digit 0-9; n_g=4 and n_g=8 land
# within +-1 token for every digit. Fixed and documented, not tuned per run.
TOKEN_COUNT_TOLERANCE = {1: 0, 2: 0, 4: 1, 8: 1}

DIFFICULTY_BY_DISTRACTOR_COUNT = {3: "easy", 5: "medium", 7: "hard"}

NAME_POOL = [
    "Marcus", "Elena", "Tomas", "Priya", "Kwame", "Ingrid", "Diego", "Yuki",
    "Fatima", "Lars", "Amara", "Hiroshi", "Sofia", "Kofi", "Anya", "Ravi",
    "Chidi", "Ines", "Bjorn", "Nadia", "Sven", "Zara", "Oscar", "Leila",
    "Niko", "Aisha", "Petr", "Mei", "Dmitri", "Talia", "Felix", "Rina",
    "Adrian", "Noor", "Casimir", "Yara", "Gustav", "Lena", "Otis", "Mira",
    "Rasheed", "Dana", "Emil", "Sana", "Viggo", "Nia", "Ansel", "Tara",
    "Kian", "Elsa", "Rufus", "Amina", "Silas", "Vera", "Omar", "Petra",
    "Bram", "Zola", "Ivo", "Nyla", "Frey", "Maya",
]


def _pad_to_token_count(tokenizer: Any, facts: List[str], target_token_count: int) -> str:
    """Append a neutral filler sentence after all real facts so total context
    token count matches a target within a couple tokens - keeps every real
    fact's relative position identical across a group's variants while
    absorbing the token-count delta caused by different evidence renderings."""
    base_text = " ".join(facts)
    base_len = len(tokenizer(base_text, add_special_tokens=False).input_ids)
    if base_len >= target_token_count:
        return base_text

    needed = target_token_count - base_len
    filler_word = "la"
    filler = " ".join([filler_word] * max(needed, 1))
    candidate = base_text + f" Note: {filler}."
    # trim/grow one filler word at a time until within +-1 of target
    for _ in range(20):
        cur_len = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        if abs(cur_len - target_token_count) <= 1:
            break
        words = filler.split()
        if cur_len < target_token_count:
            words.append(filler_word)
        elif len(words) > 1:
            words.pop()
        else:
            break
        filler = " ".join(words)
        candidate = base_text + f" Note: {filler}."
    return candidate


def build_group(
    group_id: str,
    rng: random.Random,
    tokenizer: Any,
) -> Optional[MinPairGroup]:
    """Build one matched minimal-pair group across all FRAGMENTATION_LEVELS.
    Returns None if the tokenizer-aware search couldn't hit every target
    within tolerance for the sampled canonical value - an honest, retryable
    failure (generator.py's caller just tries a new value/seed draw)."""
    subject_name = rng.choice(NAME_POOL)
    canonical_value = rng.randint(0, 9)
    distractor_count = rng.choice(sorted(DIFFICULTY_BY_DISTRACTOR_COUNT))
    difficulty = DIFFICULTY_BY_DISTRACTOR_COUNT[distractor_count]
    evidence_position_index = rng.randint(0, distractor_count)

    renderings = {}
    for n_g in FRAGMENTATION_LEVELS:
        result = find_rendering_for_target(
            canonical_value, n_g, tokenizer, tolerance=TOKEN_COUNT_TOLERANCE[n_g]
        )
        if result is None:
            return None
        renderings[n_g] = result

    # Shared distractor identity (names + values) across all variants of this
    # group - only the target's evidence rendering may differ between them.
    shared_rng = random.Random(rng.random())
    other_names = [n for n in NAME_POOL if n != subject_name]
    shared_rng.shuffle(other_names)
    distractor_names = other_names[:distractor_count]
    used_values: set = {canonical_value}  # distractors must never repeat the answer
    distractor_values = []
    for _ in distractor_names:
        v = shared_rng.randint(0, 9)
        tries = 0
        while v in used_values and tries < 20:
            v = shared_rng.randint(0, 9)
            tries += 1
        used_values.add(v)
        distractor_values.append(v)

    def _build_facts(rendering: str) -> List[str]:
        facts = [
            f"Fact: {name} owns code {render_digit(v)}."
            for name, v in zip(distractor_names, distractor_values)
        ]
        target_fact = f"Fact: {subject_name} owns code {rendering}."
        facts.insert(min(evidence_position_index, len(facts)), target_fact)
        return facts

    # Context length target: the longest real-fact token count across the
    # 4 variants, so every variant pads UP (never needs to shrink).
    lengths = {
        n_g: len(tokenizer(" ".join(_build_facts(rendering)), add_special_tokens=False).input_ids)
        for n_g, (rendering, _ttype, _realized) in renderings.items()
    }
    target_token_count = max(lengths.values())

    variants = {}
    for n_g, (rendering, ttype, realized) in renderings.items():
        facts = _build_facts(rendering)
        context_text = _pad_to_token_count(tokenizer, facts, target_token_count)
        question_text = f" Query: What is the code for {subject_name}? Answer:"
        context_token_count = len(tokenizer(context_text, add_special_tokens=False).input_ids)
        evidence_token_count = len(
            tokenizer(" " + rendering, add_special_tokens=False).input_ids
        ) - len(tokenizer(" ", add_special_tokens=False).input_ids)

        variants[n_g] = MinPairVariant(
            group_id=group_id,
            n_g_target=n_g,
            n_g_realized=realized,
            canonical_value=canonical_value,
            evidence_rendering=rendering,
            transformation_type=ttype,
            subject_name=subject_name,
            context_text=context_text,
            question_text=question_text,
            gold_answer_rendering=rendering,
            evidence_position_index=evidence_position_index,
            distractor_count=distractor_count,
            difficulty=difficulty,
            context_token_count=context_token_count,
            evidence_token_count=evidence_token_count,
            provenance={
                "distractor_names": distractor_names,
                "distractor_values": distractor_values,
                "seed_draw": group_id,
            },
        )

    return MinPairGroup(
        group_id=group_id,
        subject_name=subject_name,
        canonical_value=canonical_value,
        distractor_count=distractor_count,
        difficulty=difficulty,
        evidence_position_index=evidence_position_index,
        variants=variants,
    )


def generate_dataset(
    num_groups: int, tokenizer: Any, seed: int, max_attempts_multiplier: int = 5
) -> List[MinPairGroup]:
    """Deterministic given (num_groups, tokenizer, seed): generates candidate
    groups until num_groups succeed or the attempt budget is exhausted."""
    rng = random.Random(seed)
    groups: List[MinPairGroup] = []
    attempts = 0
    max_attempts = num_groups * max_attempts_multiplier
    while len(groups) < num_groups and attempts < max_attempts:
        group_id = f"g{attempts:06d}_{seed}"
        group = build_group(group_id, rng, tokenizer)
        if group is not None:
            groups.append(group)
        attempts += 1
    return groups


def generate_validated_dataset(
    num_groups: int, tokenizer: Any, seed: int, max_attempts_multiplier: int = 10
) -> List[MinPairGroup]:
    """generate_dataset, but every returned group has also passed the full
    validator suite (validators.validate_group) - backfills with additional
    candidates if some fail validation, up to the attempt budget."""
    from fairfuzzkv_codec.benchmarks.fragkv_minpairs.validators import validate_group

    rng = random.Random(seed)
    groups: List[MinPairGroup] = []
    attempts = 0
    max_attempts = num_groups * max_attempts_multiplier
    while len(groups) < num_groups and attempts < max_attempts:
        group_id = f"g{attempts:06d}_{seed}"
        group = build_group(group_id, rng, tokenizer)
        attempts += 1
        if group is None:
            continue
        if validate_group(group, tokenizer).passed:
            groups.append(group)
    return groups
