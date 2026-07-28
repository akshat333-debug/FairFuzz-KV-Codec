"""Gate 4 real study: fuzzy vs monotone/knapsack/logistic/no-repair, on a
real captured Qwen2.5-0.5B cache, at 2 budgets x 2 seeds (frozen in
GATE4_CONFIG.md). Freezes the Gate 4 decision, applies the automatic
naming/claims switch, and writes the appropriate report template filled in.
Numbers are measured, never invented.
"""

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.gate4_runner import SYSTEMS, Gate4Runner  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.generator import generate_validated_dataset  # noqa: E402
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import FRAGMENTATION_LEVELS  # noqa: E402
from fairfuzzkv_codec.core.naming import apply_project_identity, resolve_project_identity  # noqa: E402
from fairfuzzkv_codec.evaluation.disparity import compute_disparity  # noqa: E402
from fairfuzzkv_codec.evaluation.gate4 import (  # noqa: E402
    RunComparison, SystemMetrics, decide_gate4, paired_bootstrap_fuzzy_vs_best_simple,
)
from fairfuzzkv_codec.evaluation.isolation import PredictionRecord as IsoRecord  # noqa: E402
from fairfuzzkv_codec.evaluation.isolation import degradation_per_cohort, full_correct_ids  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
GROUPS_PER_SEED = 5
SEEDS = (42, 7)
BUDGETS = (0.3, 0.5)  # initial retention ratio
REPAIR_SWAP_FRACTION = 0.5
BITS_TOLERANCE = 1e-6


def _run_one(seed: int, budget: float, runner: Gate4Runner, groups) -> tuple:
    all_records = []
    latencies_accum = []
    norm_stats = None
    for group in groups:
        for n_g in FRAGMENTATION_LEVELS:
            records, latencies, norm_stats = runner.run_variant(
                group, n_g, retention_ratio=budget, repair_swap_fraction=REPAIR_SWAP_FRACTION,
                seed=seed, norm_stats=norm_stats,
            )
            all_records.extend(records)
            latencies_accum.append(latencies)
    return all_records, latencies_accum


def _matched_bits_ok(records) -> bool:
    by_example: dict = {}
    for r in records:
        if r.system == "full":
            continue
        by_example.setdefault(r.example_id, []).append(r.bits_per_element)
    return all(max(v) - min(v) < BITS_TOLERANCE for v in by_example.values())


def _build_run_comparison(seed: int, budget: float, records, latencies_accum) -> RunComparison:
    iso_records = [
        IsoRecord(example_id=r.example_id, cohort=str(r.n_g), system=r.system, correct=r.correct, bits_per_element=r.bits_per_element)
        for r in records
    ]
    fc = full_correct_ids(iso_records)

    metrics = {}
    for system in SYSTEMS:
        sys_records = [r for r in records if r.system == system]
        accuracy = sum(1 for r in sys_records if r.correct) / len(sys_records)
        degr = degradation_per_cohort(iso_records, system, fc)
        disparity = compute_disparity(degr)
        mean_mse = mean(r.kv_mse for r in sys_records)
        attempted = sum(r.repair_attempted for r in sys_records)
        accepted = sum(r.repair_accepted for r in sys_records)
        accept_rate = accepted / attempted if attempted > 0 else 0.0
        latency = mean(lat[system] for lat in latencies_accum) if system != "no_repair" else 0.0
        metrics[system] = SystemMetrics(
            system=system, task_accuracy=accuracy, worst_cohort_degradation=disparity.worst_group_drop,
            cddb=disparity.cddb, mean_kv_mse=mean_mse, repair_accept_rate=accept_rate,
            mean_latency_seconds_per_candidate=latency,
        )

    return RunComparison(
        budget_retention_ratio=budget, seed=seed, matched_bits_ok=_matched_bits_ok(records), metrics=metrics,
    )


def _build_item98_notes(all_predictions: list) -> str:
    """Real, grounded failure-mode notes (Prompt 14 item 98), derived from
    the raw per-example records - never a generic placeholder. Looks for
    concrete regressions (no_repair correct, fuzzy incorrect after an
    accepted swap - "overprotected the wrong candidate") and cases where
    fuzzy's accept/reject behavior flips between the two frozen budgets on
    the SAME group (an instability signal; this study uses one tokenizer/
    model, so this is same-tokenizer budget-instability, not a claim about
    cross-tokenizer stability, which was not tested here)."""
    by_key: dict = {}
    for r in all_predictions:
        key = (r["example_id"], r["budget_retention_ratio"], r["seed"])
        by_key.setdefault(key, {})[r["system"]] = r

    regressions = []
    for (example_id, budget, seed), systems in sorted(by_key.items()):
        if "no_repair" not in systems or "fuzzy" not in systems:
            continue
        nr, f = systems["no_repair"], systems["fuzzy"]
        if nr["correct"] and not f["correct"] and f["repair_accepted"] > 0:
            regressions.append((example_id, budget, seed, nr["kv_mse"], f["kv_mse"]))

    lines = []
    if regressions:
        lines.append(
            f"- **Overprotection regressions found: {len(regressions)}/{len(by_key)} pooled "
            f"(example, budget, seed) cells.** no_repair answered correctly, but fuzzy's "
            f"accepted repair swap flipped the answer to incorrect, e.g. "
            f"`{regressions[0][0]}` at budget={regressions[0][1]}, seed={regressions[0][2]} "
            f"(KV MSE barely moved: {regressions[0][3]:.3f} -> {regressions[0][4]:.3f}, so the "
            f"regression is a discrete generation-outcome flip, not a large numerical distortion "
            f"increase - the swap reintroduced a token that changed the model's argmax path)."
        )
    else:
        lines.append("- No no_repair-correct -> fuzzy-incorrect regressions found in this run.")

    unstable = 0
    by_group_ng_seed: dict = {}
    for (example_id, budget, seed), systems in by_key.items():
        if "fuzzy" not in systems:
            continue
        by_group_ng_seed.setdefault((example_id, seed), {})[budget] = systems["fuzzy"]["repair_accepted"] > 0
    for (example_id, seed), by_budget in by_group_ng_seed.items():
        if len(set(by_budget.values())) > 1:
            unstable += 1
    lines.append(
        f"- **Accept/reject instability across budgets (same tokenizer/model):** fuzzy's "
        f"swap-acceptance flipped between budget=0.3 and budget=0.5 for {unstable}/"
        f"{len(by_group_ng_seed)} (group, seed) pairs. Cross-TOKENIZER stability "
        f"(item 98's other named failure mode) was not tested in this pilot - only "
        f"`Qwen/Qwen2.5-0.5B` was used; see GATE4_CONFIG.md."
    )
    return "\n".join(lines)


def _render_report(template_path: Path, report, identity, runs_table: str, failure_notes_text: str, item98_notes: str) -> str:
    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "{{decision}}": report.decision.value,
        "{{reasoning}}": report.reasoning,
        "{{mean_accuracy_gain}}": f"{report.mean_accuracy_gain:.3f}",
        "{{accuracy_consistency}}": f"{report.accuracy_directional_consistency:.0%}",
        "{{mean_worst_cohort_gain}}": f"{report.mean_worst_cohort_gain:.3f}",
        "{{worst_consistency}}": f"{report.worst_cohort_directional_consistency:.0%}",
        "{{ci_low}}": f"{report.ci_low_vs_best_simple:.3f}",
        "{{ci_high}}": f"{report.ci_high_vs_best_simple:.3f}",
        "{{runs_table}}": runs_table,
        "{{display_name}}": identity.display_name,
        "{{claim_framing}}": identity.claim_framing,
        "{{failure_notes}}": failure_notes_text,
        "{{item98_notes}}": item98_notes,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def main() -> None:
    out = Path("gate4_fairness_study")
    out.mkdir(exist_ok=True)
    repo_root = Path(__file__).parent.parent

    runner = Gate4Runner(MODEL_NAME, device="cpu")

    runs = []
    all_predictions: list = []
    pooled_outcomes: dict = {}

    for seed in SEEDS:
        groups = generate_validated_dataset(GROUPS_PER_SEED, runner.tokenizer, seed=seed)
        print(f"seed={seed}: generated {len(groups)} groups")
        for budget in BUDGETS:
            records, latencies_accum = _run_one(seed, budget, runner, groups)
            all_predictions.extend(r.__dict__ for r in records)
            run = _build_run_comparison(seed, budget, records, latencies_accum)
            runs.append(run)
            print(
                f"  budget={budget}: matched_bits_ok={run.matched_bits_ok} "
                f"acc_gain={run.fuzzy_vs_norepair_accuracy_gain():.3f} "
                f"worst_gain={run.fuzzy_vs_norepair_worst_cohort_gain():.3f}"
            )

            by_example: dict = {}
            for r in records:
                if r.system == "full":
                    continue
                key = f"{r.example_id}_b{budget}"
                by_example.setdefault(key, {})[r.system] = r.correct
            for key, outcomes in by_example.items():
                if all(s in outcomes for s in ("fuzzy", "monotone", "knapsack", "logistic")):
                    pooled_outcomes[key] = outcomes

    (out / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in all_predictions))

    point, ci_low, ci_high = paired_bootstrap_fuzzy_vs_best_simple(pooled_outcomes, n_boot=2000, seed=0)
    print(f"fuzzy - best_simple accuracy: {point:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")

    report = decide_gate4(runs, ci_low, ci_high)
    print(f"GATE 4 DECISION: {report.decision.value}")
    print(report.reasoning)

    identity = resolve_project_identity(report.decision)
    changed = apply_project_identity(identity, repo_root)
    print(f"naming switch applied: {changed} -> {identity.display_name}")

    (out / "gate4_report.json").write_text(json.dumps(report.to_dict(), indent=2))

    runs_table_lines = ["| Budget | Seed | Matched | Acc gain | Worst gain |", "|---|---|---|---|---|"]
    for r in runs:
        runs_table_lines.append(
            f"| {r.budget_retention_ratio} | {r.seed} | {r.matched_bits_ok} | "
            f"{r.fuzzy_vs_norepair_accuracy_gain():.3f} | {r.fuzzy_vs_norepair_worst_cohort_gain():.3f} |"
        )
    runs_table = "\n".join(runs_table_lines)

    failure_notes_text = "\n".join(f"- {n}" for n in report.failure_notes) if report.failure_notes else "None observed in this run."
    item98_notes = _build_item98_notes(all_predictions)

    from fairfuzzkv_codec.evaluation.gate4 import Gate4Decision

    template_name = "GATE4_REPORT_PASS_TEMPLATE.md" if report.decision != Gate4Decision.FAIL else "GATE4_REPORT_FAIL_TEMPLATE.md"
    rendered = _render_report(repo_root / template_name, report, identity, runs_table, failure_notes_text, item98_notes)
    (repo_root / "GATE4_REPORT.md").write_text(rendered, encoding="utf-8")
    print(f"saved -> {repo_root/'GATE4_REPORT.md'}")
    print(f"saved -> {out/'gate4_report.json'}")
    print(f"saved -> {out/'predictions.jsonl'}")


if __name__ == "__main__":
    main()
