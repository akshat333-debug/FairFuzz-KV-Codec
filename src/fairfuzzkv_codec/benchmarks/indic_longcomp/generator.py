"""Parallel group generator.

For each group, one shared configuration (names, digits, distractor count,
evidence position) is drawn ONCE, then rendered into all four
`LanguageCondition` variants via `templates.TEMPLATES`. Because every
variant is built from the SAME drawn values, `canonical_answer` (always a
single digit) is identical by construction across languages - this is what
makes the group "parallel" in the verified sense (see package docstring),
not because the sentences are claimed to be faithful translations of one
another.
"""

import random
from typing import Any, Dict, List, Optional

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import NAME_POOL
from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import IndicGroup, IndicVariant, LanguageCondition, TaskFamily
from fairfuzzkv_codec.benchmarks.indic_longcomp.templates import TEMPLATES

DISTRACTOR_COUNT_CHOICES = (3, 5, 7)
ALL_LANGUAGES = tuple(LanguageCondition)
ALL_TASK_FAMILIES = tuple(TaskFamily)


def _draw_names(rng: random.Random, n: int) -> List[str]:
    return rng.sample(NAME_POOL, n)


def _draw_config(task_family: TaskFamily, rng: random.Random) -> Dict[str, Any]:
    distractor_count = rng.choice(DISTRACTOR_COUNT_CHOICES)

    if task_family == TaskFamily.RETRIEVAL:
        names = _draw_names(rng, 1 + distractor_count)
        target_names, distractor_names = names[:1], names[1:]
        digit = rng.randint(0, 9)
        used = {digit}
        distractor_digits = []
        for _ in distractor_names:
            d = rng.randint(0, 9)
            while d in used:
                d = rng.randint(0, 9)
            distractor_digits.append(d)
            used.add(d)
        canonical_answer = digit
        evidence_count = 1
        target_digits = [digit]

    elif task_family == TaskFamily.MULTI_HOP:
        names = _draw_names(rng, 2 + distractor_count)
        target_names, distractor_names = names[:2], names[2:]
        digit = rng.randint(0, 9)
        used = {digit}
        distractor_digits = []
        for _ in distractor_names:
            d = rng.randint(0, 9)
            while d in used:
                d = rng.randint(0, 9)
            distractor_digits.append(d)
            used.add(d)
        canonical_answer = digit
        evidence_count = 2
        target_digits = [digit]  # only name2 (target_names[1]) owns the code

    elif task_family == TaskFamily.COMPARISON:
        names = _draw_names(rng, 2 + distractor_count)
        target_names, distractor_names = names[:2], names[2:]
        d1 = rng.randint(0, 9)
        d2 = rng.randint(0, 9)
        while d2 == d1:
            d2 = rng.randint(0, 9)
        canonical_answer = max(d1, d2)
        used = {d1, d2, canonical_answer}
        distractor_digits = []
        for _ in distractor_names:
            d = rng.randint(0, 9)
            while d in used:
                d = rng.randint(0, 9)
            distractor_digits.append(d)
            used.add(d)
        evidence_count = 2
        target_digits = [d1, d2]

    elif task_family == TaskFamily.COUNTING:
        match_count = rng.choice((2, 3))
        # target_digit must not coincide with match_count (the answer) - a
        # "how many own code 3?" question whose answer IS 3 is a spurious
        # mechanical collision the no_answer_leakage validator would (rightly)
        # flag, even though it isn't semantic leakage. Avoided by construction.
        target_digit = rng.randint(0, 9)
        while target_digit == match_count:
            target_digit = rng.randint(0, 9)
        names = _draw_names(rng, match_count + distractor_count)
        target_names, distractor_names = names[:match_count], names[match_count:]
        used = {target_digit}
        distractor_digits = []
        for _ in distractor_names:
            d = rng.randint(0, 9)
            while d in used:
                d = rng.randint(0, 9)
            distractor_digits.append(d)
            used.add(d)
        canonical_answer = match_count
        evidence_count = match_count
        target_digits = [target_digit] * match_count

    elif task_family == TaskFamily.AGGREGATION:
        names = _draw_names(rng, 2 + distractor_count)
        target_names, distractor_names = names[:2], names[2:]
        d1 = rng.randint(0, 4)
        d2 = rng.randint(0, 4)
        canonical_answer = d1 + d2
        used = {d1, d2, canonical_answer}
        distractor_digits = []
        for _ in distractor_names:
            d = rng.randint(0, 9)
            while d in used:
                d = rng.randint(0, 9)
            distractor_digits.append(d)
            used.add(d)
        evidence_count = 2
        target_digits = [d1, d2]

    else:
        raise ValueError(f"unknown task_family: {task_family}")

    evidence_position_index = rng.randint(0, distractor_count)
    return {
        "target_names": target_names, "target_digits": target_digits,
        "distractor_names": distractor_names, "distractor_digits": distractor_digits,
        "canonical_answer": canonical_answer, "evidence_count": evidence_count,
        "distractor_count": distractor_count, "evidence_position_index": evidence_position_index,
    }


def _target_facts(task_family: TaskFamily, tpl: Any, config: Dict[str, Any]) -> List[str]:
    names, digits = config["target_names"], config["target_digits"]
    if task_family == TaskFamily.RETRIEVAL:
        return [tpl.fact_owns.format(name=names[0], digit=digits[0])]
    if task_family == TaskFamily.MULTI_HOP:
        return [
            tpl.fact_hop_link.format(name1=names[0], name2=names[1]),
            tpl.fact_owns.format(name=names[1], digit=digits[0]),
        ]
    if task_family in (TaskFamily.COMPARISON, TaskFamily.AGGREGATION):
        return [tpl.fact_owns.format(name=n, digit=d) for n, d in zip(names, digits)]
    if task_family == TaskFamily.COUNTING:
        return [tpl.fact_owns.format(name=n, digit=d) for n, d in zip(names, digits)]
    raise ValueError(f"unknown task_family: {task_family}")


def _question(task_family: TaskFamily, tpl: Any, config: Dict[str, Any]) -> str:
    names = config["target_names"]
    if task_family == TaskFamily.RETRIEVAL:
        return tpl.question_retrieval.format(name=names[0])
    if task_family == TaskFamily.MULTI_HOP:
        return tpl.question_multihop.format(name1=names[0])
    if task_family == TaskFamily.COMPARISON:
        return tpl.question_comparison.format(name1=names[0], name2=names[1])
    if task_family == TaskFamily.COUNTING:
        return tpl.question_counting.format(digit=config["target_digits"][0])
    if task_family == TaskFamily.AGGREGATION:
        return tpl.question_aggregation.format(name1=names[0], name2=names[1])
    raise ValueError(f"unknown task_family: {task_family}")


def _render_variant(
    language: LanguageCondition, task_family: TaskFamily, config: Dict[str, Any],
    group_id: str, tokenizer: Optional[Any],
) -> IndicVariant:
    tpl = TEMPLATES[language]
    target = _target_facts(task_family, tpl, config)
    distractors = [
        tpl.fact_owns.format(name=n, digit=d)
        for n, d in zip(config["distractor_names"], config["distractor_digits"])
    ]
    facts = list(distractors)
    facts[config["evidence_position_index"]:config["evidence_position_index"]] = target
    context_text = " ".join(facts)
    question_text = _question(task_family, tpl, config)

    token_count = len(tokenizer(context_text, add_special_tokens=False).input_ids) if tokenizer is not None else 0

    return IndicVariant(
        group_id=group_id, language=language, task_family=task_family,
        context_text=context_text, question_text=question_text,
        canonical_answer=config["canonical_answer"], evidence_count=config["evidence_count"],
        evidence_position_index=config["evidence_position_index"], distractor_count=config["distractor_count"],
        context_token_count=token_count,
        generation_metadata={
            "target_names": config["target_names"], "distractor_names": config["distractor_names"],
            "distractor_digits": config["distractor_digits"],
        },
    )


def build_group(group_id: str, rng: random.Random, task_family: TaskFamily, tokenizer: Optional[Any] = None) -> IndicGroup:
    config = _draw_config(task_family, rng)
    variants = {
        lang: _render_variant(lang, task_family, config, group_id, tokenizer)
        for lang in ALL_LANGUAGES
    }
    return IndicGroup(
        group_id=group_id, task_family=task_family, canonical_answer=config["canonical_answer"],
        evidence_count=config["evidence_count"], evidence_position_index=config["evidence_position_index"],
        distractor_count=config["distractor_count"], variants=variants,
    )


def generate_dataset(
    groups_per_family: int, seed: int, tokenizer: Optional[Any] = None,
    task_families: tuple = ALL_TASK_FAMILIES,
) -> List[IndicGroup]:
    """Deterministic given (groups_per_family, seed, task_families)."""
    rng = random.Random(seed)
    groups: List[IndicGroup] = []
    for task_family in task_families:
        for i in range(groups_per_family):
            group_id = f"indic_{task_family.value}_{i:04d}_{seed}"
            groups.append(build_group(group_id, rng, task_family, tokenizer))
    return groups
