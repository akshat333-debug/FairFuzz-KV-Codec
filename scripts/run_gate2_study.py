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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-groups", type=int, default=24)
    p.add_argument("--budget-bits", type=int, default=6, help="per-cohort avg bit budget (between 4 and 8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="gate2_fairness_study")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runner = FragKVRunner(MODEL_NAME)
    ch = runner.config_hash
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    groups = generate_validated_dataset(args.num_groups, tok, seed=args.seed)
    half = len(groups) // 2
    calib_groups, eval_groups = groups[:half], groups[half:]
    print(f"groups: {len(groups)} (calib {len(calib_groups)}, eval {len(eval_groups)})")

    # ---- calibration: per-cohort degradation for INT4/INT8 -----------------
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
    cohorts_ids = sorted(acc["FullKV"].keys(), key=int)

    cohorts: List[Cohort] = []
    for cid in cohorts_ids:
        full_acc = acc["FullKV"].get(cid, 1.0)
        options = []
        for bits in BIT_CHOICES:
            sys_acc = acc[f"int{bits}"].get(cid, 0.0)
            degradation = max(0.0, full_acc - sys_acc)
            options.append(BitOption(label=f"int{bits}", total_bits=bits, distortion=degradation))
        cohorts.append(Cohort(cohort_id=cid, options=options))

    budget = args.budget_bits * len(cohorts)
    aggregate = solve_exact(cohorts, budget)
    minimax = solve_minimax_exact(cohorts, budget)
    agg_choice = {c: aggregate.choice[c].label for c in aggregate.choice}
    mm_choice = {c: minimax.allocation.choice[c].label for c in minimax.allocation.choice}
    print(f"budget={budget}  aggregate={agg_choice}  minimax={mm_choice}")

    # ---- eval: apply each system's per-cohort bits on held-out groups -------
    def _codec_for(label: str) -> UniformQuantCodec:
        return UniformQuantCodec(ch, num_bits=int(label.replace("int", "")))

    records: List[PredictionRecord] = []
    agg_bits_acc: List[float] = []
    mm_bits_acc: List[float] = []
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
                    example_id=f"{fr.group_id}_{n_g}", cohort=cid, system=fr.codec_name,
                    correct=bool(fr.correct), bits_per_element=float(fr.actual_bits_per_element),
                ))
                if fr.codec_name == "aggregate":
                    agg_bits_acc.append(fr.actual_bits_per_element)
                elif fr.codec_name == "minimax":
                    mm_bits_acc.append(fr.actual_bits_per_element)

    # matched-bit check: both systems' realized mean bits/element within tolerance
    agg_mean = sum(agg_bits_acc) / len(agg_bits_acc) if agg_bits_acc else 0.0
    mm_mean = sum(mm_bits_acc) / len(mm_bits_acc) if mm_bits_acc else 0.0
    matched_ok = abs(agg_mean - mm_mean) <= 0.5  # within half a bit/element

    # ---- isolate, compare, decide ------------------------------------------
    iso = isolate(records)
    ids = sorted({r.example_id for r in iso})
    cmp = run_comparison_from_records(iso, budget=budget, seed=args.seed, matched_bits_ok=matched_ok)
    point, lo, hi = paired_bootstrap_worst_benefit(iso, ids, n_boot=2000, seed=args.seed)
    report = decide_gate2([cmp], ci_low=lo, ci_high=hi)

    counts = cohort_counts(iso, "minimax")
    print(f"isolated examples: {len(ids)}  cohort counts: {counts}")
    print(f"matched bits: agg={agg_mean:.2f} mm={mm_mean:.2f} ok={matched_ok}")
    print(f"worst-cohort fairness benefit={point:.4f}  CI[{lo:.4f},{hi:.4f}]")
    print(f"DECISION: {report.decision.value}")
    print(report.reasoning)

    # preserve raw predictions + report
    with open(out / "predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r.__dict__) + "\n")
    (out / "GATE2_REPORT.json").write_text(json.dumps({
        "num_groups": args.num_groups, "budget": budget,
        "aggregate_choice": agg_choice, "minimax_choice": mm_choice,
        "matched_bits": {"aggregate": agg_mean, "minimax": mm_mean, "ok": matched_ok},
        "isolated_examples": len(ids), "cohort_counts": counts,
        "bootstrap": {"point": point, "ci_low": lo, "ci_high": hi},
        "report": report.to_dict(),
        "power_note": (
            f"PILOT SCALE: {len(ids)} isolated examples across {len(cohorts)} cohorts. "
            "Underpowered - treat the decision as provisional evidence, not a final verdict."
        ),
    }, indent=2))
    print(f"saved -> {out}/GATE2_REPORT.json, predictions.jsonl")


if __name__ == "__main__":
    main()
