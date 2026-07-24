"""Gate 2 matched-bit fairness study (real model).

Cohorts = evidence-fragmentation levels n_g in {1,2,4,8}. Two systems allocate a
shared per-cohort bit budget over {INT4, INT8}:
  * aggregate  - minimize SUM of cohort degradation (Prompt 10 control)
  * minimax    - minimize the WORST cohort degradation (Prompt 11)
Degradation is calibrated on a train split and the two systems are compared on a
held-out eval split, restricted to the intersection-full-correct subset. Raw
predictions are preserved; the pre-registered Gate 2 decision (gate2.py) is then
applied. Numbers are measured on Qwen2.5-0.5B - never invented. Pilot scale is
configurable; small runs are UNDERPOWERED and reported as such.

Usage: python scripts/run_gate2_study.py --num-groups 24 --budget-bits 6
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer  # noqa: E402

from fairfuzzkv_codec.allocation.allocator import BitOption, Cohort, solve_exact  # noqa: E402
from fairfuzzkv_codec.allocation.minimax import solve_minimax_exact  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import generate_validated_dataset  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.runner import FragKVRunner  # noqa: E402
from fairfuzzkv_codec.codec.baselines import FullKVFP16Codec, UniformQuantCodec  # noqa: E402
from fairfuzzkv_codec.evaluation.gate2 import (  # noqa: E402
    decide_gate2,
    paired_bootstrap_worst_benefit,
    run_comparison_from_records,
)
from fairfuzzkv_codec.evaluation.isolation import PredictionRecord, cohort_counts, isolate  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
BIT_CHOICES = [4, 8]


def _accuracy_by_cohort(fragkv_records) -> Dict[str, Dict[str, float]]:
    """codec_name -> {cohort(n_g) -> accuracy}."""
    agg: Dict[str, Dict[str, List[int]]] = {}
    for r in fragkv_records:
        agg.setdefault(r.codec_name, {}).setdefault(str(r.n_g), []).append(int(r.correct))
    return {c: {k: sum(v) / len(v) for k, v in d.items()} for c, d in agg.items()}


def _calibrate_cohorts(runner: FragKVRunner, ch: str, calib_groups) -> List[Cohort]:
    """Per-cohort INT4/INT8 degradation curves from the calibration split."""
    calib_codecs = [
        ("FullKV", FullKVFP16Codec(ch)),
        ("int4", UniformQuantCodec(ch, num_bits=4)),
        ("int8", UniformQuantCodec(ch, num_bits=8)),
    ]
    calib_records = []
    for g in calib_groups:
        for n_g in g.variants:
            calib_records.extend(runner.run_variant(g, n_g, calib_codecs))
    acc = _accuracy_by_cohort(calib_records)
    cohorts: List[Cohort] = []
    for cid in sorted(acc["FullKV"].keys(), key=int):
        full_acc = acc["FullKV"].get(cid, 1.0)
        options = [
            BitOption(label=f"int{b}", total_bits=b, distortion=max(0.0, full_acc - acc[f"int{b}"].get(cid, 0.0)))
            for b in BIT_CHOICES
        ]
        cohorts.append(Cohort(cohort_id=cid, options=options))
    return cohorts


def _eval_one(
    runner: FragKVRunner, ch: str, eval_groups, cohorts: List[Cohort],
    budget: int, seed: int, tag: str,
) -> tuple:
    """Allocate at `budget`, apply per-cohort bits on the eval split, return
    (records, RunComparison). example_ids are tagged unique per (seed, budget)."""
    aggregate = solve_exact(cohorts, budget)
    minimax = solve_minimax_exact(cohorts, budget)
    agg_choice = {c: aggregate.choice[c].label for c in aggregate.choice}
    mm_choice = {c: minimax.allocation.choice[c].label for c in minimax.allocation.choice}

    def _codec_for(label: str) -> UniformQuantCodec:
        return UniformQuantCodec(ch, num_bits=int(label.replace("int", "")))

    records: List[PredictionRecord] = []
    agg_bits, mm_bits = [], []
    for g in eval_groups:
        for n_g in g.variants:
            cid = str(n_g)
            eval_codecs = [
                ("full", FullKVFP16Codec(ch)),
                ("aggregate", _codec_for(agg_choice[cid])),
                ("minimax", _codec_for(mm_choice[cid])),
            ]
            for fr in runner.run_variant(g, n_g, eval_codecs):
                records.append(PredictionRecord(
                    example_id=f"{tag}_{fr.group_id}_{n_g}", cohort=cid, system=fr.codec_name,
                    correct=bool(fr.correct), bits_per_element=float(fr.actual_bits_per_element),
                ))
                if fr.codec_name == "aggregate":
                    agg_bits.append(fr.actual_bits_per_element)
                elif fr.codec_name == "minimax":
                    mm_bits.append(fr.actual_bits_per_element)

    agg_mean = sum(agg_bits) / len(agg_bits) if agg_bits else 0.0
    mm_mean = sum(mm_bits) / len(mm_bits) if mm_bits else 0.0
    matched_ok = abs(agg_mean - mm_mean) <= 0.5
    cmp = run_comparison_from_records(isolate(records), budget=budget, seed=seed, matched_bits_ok=matched_ok)
    print(f"  seed={seed} budget={budget} agg={agg_choice} mm={mm_choice} "
          f"bits(agg/mm)={agg_mean:.2f}/{mm_mean:.2f} benefit={cmp.fairness_benefit_worst:.4f}")
    return records, cmp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-groups", type=int, default=24)
    p.add_argument("--budgets", type=str, default="5,6,7", help="comma list of per-cohort avg bit budgets in [4,8]")
    p.add_argument("--seeds", type=str, default="42", help="comma list of dataset seeds")
    p.add_argument("--output-dir", type=str, default="gate2_fairness_study")
    args = p.parse_args()

    budgets_bits = [int(x) for x in args.budgets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runner = FragKVRunner(MODEL_NAME)
    ch = runner.config_hash
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_records: List[PredictionRecord] = []
    runs = []
    n_cohorts = 0
    for seed in seeds:
        groups = generate_validated_dataset(args.num_groups, tok, seed=seed)
        half = len(groups) // 2
        calib_groups, eval_groups = groups[:half], groups[half:]
        cohorts = _calibrate_cohorts(runner, ch, calib_groups)
        n_cohorts = max(n_cohorts, len(cohorts))
        print(f"seed={seed}: {len(groups)} groups, {len(cohorts)} cohorts")
        for bb in budgets_bits:
            budget = bb * len(cohorts)
            recs, cmp = _eval_one(runner, ch, eval_groups, cohorts, budget, seed, tag=f"s{seed}b{bb}")
            all_records.extend(recs)
            runs.append(cmp)

    # pooled isolation + bootstrap across every (seed, budget) run
    iso = isolate(all_records)
    ids = sorted({r.example_id for r in iso})
    point, lo, hi = paired_bootstrap_worst_benefit(iso, ids, n_boot=2000, seed=seeds[0])
    report = decide_gate2(runs, ci_low=lo, ci_high=hi)
    counts = cohort_counts(iso, "minimax")

    print(f"\n=== {len(runs)} runs ({len(seeds)} seeds x {len(budgets_bits)} budgets) ===")
    print(f"isolated examples: {len(ids)}  cohort counts: {counts}")
    print(f"pooled worst-cohort benefit={point:.4f}  CI[{lo:.4f},{hi:.4f}]")
    print(f"directional consistency={report.directional_consistency:.0%}")
    print(f"DECISION: {report.decision.value}")
    print(report.reasoning)

    with open(out / "predictions.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r.__dict__) + "\n")
    (out / "GATE2_REPORT.json").write_text(json.dumps({
        "num_groups": args.num_groups, "budgets_bits": budgets_bits, "seeds": seeds,
        "n_runs": len(runs), "isolated_examples": len(ids), "cohort_counts": counts,
        "bootstrap": {"point": point, "ci_low": lo, "ci_high": hi},
        "per_run": [r.__dict__ for r in runs],
        "report": report.to_dict(),
        "power_note": (
            f"PILOT SCALE: {len(ids)} isolated examples across {n_cohorts} cohorts, "
            f"{len(runs)} runs. Underpowered - provisional evidence, not a final verdict."
        ),
    }, indent=2))
    print(f"saved -> {out}/GATE2_REPORT.json, predictions.jsonl")


if __name__ == "__main__":
    main()
