import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer  # noqa: E402

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.dataset_card import (  # noqa: E402
    build_dataset_card,
    write_dataset,
)
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.gate1 import decide_gate1  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import generate_validated_dataset  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.pivot import generate_pivot_plan  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.runner import FragKVRunner, build_codecs  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.stats_utils import (  # noqa: E402
    PredictionRecord,
    compute_effect_size,
)
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.validators import validate_dataset  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # default; overridable via --model for cross-model reproduction (Prompt 17 / Gate 3)


def compute_gate1_from_predictions(predictions_path: Path):
    """Reproduce the Gate 1 report purely from raw predictions on disk - no
    model inference. This function is what "Gate 1 report is reproducible
    from raw predictions" means operationally: rerun this on the same
    predictions.jsonl and get the identical decision."""
    records = []
    with open(predictions_path) as f:
        for line in f:
            if line.strip():
                records.append(PredictionRecord.model_validate_json(line))

    codec_names = sorted({r.codec_name for r in records})
    lossy_names = [c for c in codec_names if c != "FullKV"]

    control_result = compute_effect_size(records, "FullKV", low_n_g=1, high_n_g=8)
    lossy_results = [compute_effect_size(records, name, low_n_g=1, high_n_g=8) for name in lossy_names]

    return decide_gate1(control_result, lossy_results)


def main():
    parser = argparse.ArgumentParser(description="Run the FragKV-MinPairs Gate 1 causal study")
    parser.add_argument("--num-groups", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="results/fragkv_gate1_study")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="override for cross-model reproduction (Gate 3)")
    parser.add_argument(
        "--token-count-tolerance", type=str, default=None,
        help='JSON dict override e.g. \'{"1":0,"2":0,"4":2,"8":2}\' - a tokenizer-specific recalibration of the '
             "numeric rendering ladder's matching tolerance (see generator.build_group's docstring). "
             "Default: None, i.e. the frozen Qwen-calibrated tolerance, UNCHANGED.",
    )
    args = parser.parse_args()
    model_name = args.model
    token_count_tolerance = {int(k): v for k, v in json.loads(args.token_count_tolerance).items()} if args.token_count_tolerance else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== FragKV-MinPairs Gate 1 Study ===")
    print(f"model={model_name} num_groups={args.num_groups} seed={args.seed} token_count_tolerance={token_count_tolerance or 'default (frozen)'}")

    print("\n[1] Loading tokenizer and generating validated dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    t0 = time.time()
    groups = generate_validated_dataset(args.num_groups, tokenizer, seed=args.seed, token_count_tolerance=token_count_tolerance)
    print(f"  generated {len(groups)} validated groups in {time.time() - t0:.1f}s")
    if len(groups) < args.num_groups:
        print(f"  WARNING: requested {args.num_groups} but only {len(groups)} passed validation within budget")

    print("\n[2] Re-validating full dataset (independent re-check, not trusting generator)...")
    reports = validate_dataset(groups, tokenizer, token_count_tolerance)
    n_failed = sum(1 for r in reports if not r.passed)
    if n_failed:
        print(f"  WARNING: {n_failed} groups failed independent re-validation - dropping them")
        passing_ids = {r.group_id for r in reports if r.passed}
        groups = [g for g in groups if g.group_id in passing_ids]
    print(f"  {len(groups)} groups confirmed valid")

    card = build_dataset_card(groups, model_name, seed=args.seed)
    write_dataset(groups, card, output_dir / "dataset")
    print(f"  dataset written to {output_dir / 'dataset'}, split_hash={card.split_hash[:16]}...")

    print("\n[3] Running real study: FullKV + 2 compression baselines, matched at 8 bits/element...")
    print(f"  codecs: {[name for name, _ in build_codecs('x')]}")
    runner = FragKVRunner(model_name)
    t0 = time.time()
    records = runner.run_study(groups, max_new_tokens=args.max_new_tokens)
    elapsed = time.time() - t0
    print(f"  {len(records)} raw predictions in {elapsed:.1f}s ({elapsed / max(len(groups), 1):.2f}s/group)")

    predictions_path = output_dir / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    print(f"  raw predictions written to {predictions_path}")

    print("\n[4] Computing Gate 1 decision from raw predictions (pre-registered logic, unchanged)...")
    report = compute_gate1_from_predictions(predictions_path)

    report_path = output_dir / "GATE1_REPORT.json"
    report_path.write_text(report.model_dump_json(indent=2))
    (output_dir / "RUN_METADATA.json").write_text(
        json.dumps({"model": model_name, "num_groups": args.num_groups, "seed": args.seed, "token_count_tolerance": token_count_tolerance}, indent=2)
    )
    print(f"  {report.decision.value}")
    print(f"  {report.reasoning}")
    print(f"  full report: {report_path}")

    if report.decision.value == "FAIL":
        pivot_path = output_dir / "PIVOT_PLAN.md"
        pivot_path.write_text(generate_pivot_plan(report))
        print(f"\n[5] Gate 1 = FAIL -> pivot plan written to {pivot_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
