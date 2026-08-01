"""IndicLongComp course + journal subsets (Prompt 15).

Course subset: small, full pipeline including a REAL FullKV baseline run on
Qwen2.5-0.5B (isolation subset tagged before any compression evaluation, per
the item's own ordering requirement).

Journal subset: larger, structurally validated (parallelism, PII, dedup,
contamination self-check, fragility distributions), but WITHOUT a real-model
FullKV run - explicitly documented as a follow-up given this session's real
per-example model-forward-pass compute budget, not silently skipped.

Numbers are measured, never invented.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer  # noqa: E402

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.dataset_card import load_dataset as load_fragkv_dataset  # noqa: E402
from fairfuzzkv_codec.benchmarks.indic_longcomp.dataset_card import build_dataset_card, write_dataset  # noqa: E402
from fairfuzzkv_codec.benchmarks.indic_longcomp.fragility_report import per_language_fragility  # noqa: E402
from fairfuzzkv_codec.benchmarks.indic_longcomp.generator import generate_dataset  # noqa: E402
from fairfuzzkv_codec.benchmarks.indic_longcomp.runner import IndicLongCompRunner, full_correct_group_ids  # noqa: E402
from fairfuzzkv_codec.benchmarks.indic_longcomp.validators import (  # noqa: E402
    check_contamination_against, find_duplicate_texts, validate_dataset, validate_no_pii,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
# Item 105 asks for fragility distributions + cohort coverage "for each
# tokenizer family". The project targets two distinct families throughout
# (see tests/unicode_grouping/test_aligner.py): byte-level BPE and
# SentencePiece. Both are reported here; tokenization only, no model run.
FRAGILITY_TOKENIZERS = [
    ("Qwen/Qwen2.5-0.5B", "byte_level_bpe"),
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", "sentencepiece"),
]
COURSE_GROUPS_PER_FAMILY = 2
JOURNAL_GROUPS_PER_FAMILY = 10
SEED = 42


def _other_repo_texts() -> list:
    try:
        fragkv_groups = load_fragkv_dataset("gate1_study/dataset")
    except FileNotFoundError:
        return []
    return [v.context_text for g in fragkv_groups for v in g.variants.values()]


def _structural_checks(groups, tag: str, other_texts: list) -> dict:
    reports = validate_dataset(groups)
    all_passed = all(r.passed for r in reports)
    pii = validate_no_pii(groups)
    duplicates = find_duplicate_texts(groups)
    overlaps = check_contamination_against(groups, other_texts)
    print(f"[{tag}] parallelism/answer validation: {sum(r.passed for r in reports)}/{len(reports)} groups passed")
    print(f"[{tag}] PII scan passed: {pii.passed}")
    print(f"[{tag}] duplicate texts: {len(duplicates)}")
    print(f"[{tag}] contamination overlaps vs FragKV-MinPairs: {len(overlaps)}/{len(other_texts)} compared")
    return {
        "all_groups_passed_validation": all_passed,
        "failed_group_ids": [r.group_id for r in reports if not r.passed],
        "pii_passed": pii.passed,
        "duplicate_count": len(duplicates),
        "contamination_overlap_count": len(overlaps),
        "contamination_compared_against": len(other_texts),
    }


def _fragility_across_tokenizer_families(groups, tag: str) -> list:
    """Item 105: per-language fragility distributions + cohort coverage for
    EVERY tokenizer family, not just the study model's. Tokenization only -
    no model forward passes - so this is cheap enough to run on both the
    course and journal subsets."""
    distributions: list = []
    for tokenizer_name, family in FRAGILITY_TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        per_language = per_language_fragility(groups, tok, tokenizer_name=tokenizer_name)
        for lang, dist in per_language.items():
            print(f"  [{tag}] fragility[{family}/{lang.value}]: mean={dist.mean_score:.3f} "
                  f"n={dist.num_units_scored} cohorts={dist.cohort_counts}")
        distributions.extend(per_language.values())
    return distributions


def main() -> None:
    out = Path("indic_longcomp_study")
    (out / "course").mkdir(parents=True, exist_ok=True)
    (out / "journal").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    other_texts = _other_repo_texts()

    # ---- course subset: full pipeline incl. real FullKV isolation ----
    course_groups = generate_dataset(COURSE_GROUPS_PER_FAMILY, seed=SEED, tokenizer=tokenizer)
    print(f"course: generated {len(course_groups)} groups ({len(course_groups) * 4} variants)")
    course_checks = _structural_checks(course_groups, "course", other_texts)

    course_fragility = _fragility_across_tokenizer_families(course_groups, "course")

    course_card = build_dataset_card(
        course_groups, seed=SEED, scale="course",
        contamination_overlaps=course_checks["contamination_overlap_count"],
        contamination_compared_against=course_checks["contamination_compared_against"],
        fragility_distributions=course_fragility,
    )
    write_dataset(course_groups, course_card, out / "course")

    print("course: running REAL FullKV baseline on Qwen2.5-0.5B (before any compression evaluation)...")
    runner = IndicLongCompRunner(MODEL_NAME, device="cpu")
    predictions = runner.run_dataset(course_groups)
    full_correct = full_correct_group_ids(predictions)
    per_lang_accuracy = {}
    for lang_code in ("en", "hi", "hinglish", "te_en"):
        lang_records = [p for p in predictions if p.language == lang_code]
        per_lang_accuracy[lang_code] = sum(r.correct for r in lang_records) / len(lang_records)
    print(f"course: FullKV per-language accuracy: {per_lang_accuracy}")
    print(f"course: intersection-full-correct groups: {len(full_correct)}/{len(course_groups)}")

    (out / "course" / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p.__dict__) for p in predictions), encoding="utf-8",
    )
    (out / "course" / "isolation_summary.json").write_text(
        json.dumps({
            "num_groups": len(course_groups),
            "full_correct_group_count": len(full_correct),
            "full_correct_group_ids": sorted(full_correct),
            "per_language_accuracy": per_lang_accuracy,
            "structural_checks": course_checks,
        }, indent=2),
        encoding="utf-8",
    )

    # ---- journal subset: larger, structurally validated, no real-model run ----
    journal_groups = generate_dataset(JOURNAL_GROUPS_PER_FAMILY, seed=SEED, tokenizer=tokenizer)
    print(f"journal: generated {len(journal_groups)} groups ({len(journal_groups) * 4} variants) - structural validation only")
    journal_checks = _structural_checks(journal_groups, "journal", other_texts)
    journal_fragility = _fragility_across_tokenizer_families(journal_groups, "journal")
    journal_card = build_dataset_card(
        journal_groups, seed=SEED, scale="journal",
        contamination_overlaps=journal_checks["contamination_overlap_count"],
        contamination_compared_against=journal_checks["contamination_compared_against"],
        fragility_distributions=journal_fragility,
    )
    write_dataset(journal_groups, journal_card, out / "journal")
    (out / "journal" / "structural_checks.json").write_text(json.dumps(journal_checks, indent=2), encoding="utf-8")

    print(f"saved -> {out/'course'} and {out/'journal'}")


if __name__ == "__main__":
    main()
