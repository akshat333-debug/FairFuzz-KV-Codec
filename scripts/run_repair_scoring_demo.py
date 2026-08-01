"""Module 3 demo: compare the fuzzy repair-priority scorer against its three
simpler competitors on synthetic candidate groups (this module scores
already-computed structural signals, not raw KV tensors, so no HF model
capture is needed here - unlike Prompts 5/6). Numbers are measured, never
invented.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from fairfuzzkv_codec.dashboard.plots import plot_fuzzy_membership_functions  # noqa: E402
from fairfuzzkv_codec.pruning.repair import RepairContract  # noqa: E402
from fairfuzzkv_codec.repair_scoring.ablation import ScorerConfig, ScorerType, run_ablation, score_candidates  # noqa: E402
from fairfuzzkv_codec.repair_scoring.competitors import monotone_weighted_score  # noqa: E402
from fairfuzzkv_codec.repair_scoring.fuzzy import fuzzy_priority_scores, fuzzy_repair_priority  # noqa: E402
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs, fit_input_normalizers, normalize_inputs  # noqa: E402
from fairfuzzkv_codec.repair_scoring.integration import propose_repair_swap  # noqa: E402
from fairfuzzkv_codec.repair_scoring.sensitivity import (  # noqa: E402
    fuzzy_num_parameters, measure_complexity, sensitivity_to_breakpoints, sensitivity_to_rules,
)


def _synthetic_candidates(n: int, seed: int) -> ScorerInputs:
    g = torch.Generator().manual_seed(seed)
    return ScorerInputs(
        fragility=torch.rand(n, generator=g) * 5,
        evidence_importance=torch.rand(n, generator=g) * 5,
        completion_cost=torch.rand(n, generator=g) * 5,
        staleness=torch.rand(n, generator=g) * 5,
    )


def main() -> None:
    out = Path("repair_scoring_study")
    out.mkdir(exist_ok=True)

    train = _synthetic_candidates(n=40, seed=1)
    eval_candidates = _synthetic_candidates(n=16, seed=2)
    stats = fit_input_normalizers(train)  # train-only normalization
    normalized_eval = normalize_inputs(eval_candidates, stats)

    ablation = run_ablation(normalized_eval)
    print("scorer priorities (first 5 candidates):")
    for name, scores in ablation.items():
        print(f"  {name}: {[round(s, 3) for s in scores[:5].tolist()]}")

    # Latency/parameter complexity for EVERY scorer in the ablation (not just
    # fuzzy vs one competitor), so the fuzzy overhead is quantified against all
    # the alternatives it is being compared against.
    complexity = [
        measure_complexity("fuzzy", fuzzy_priority_scores, normalized_eval, fuzzy_num_parameters()),
        measure_complexity("monotone", monotone_weighted_score, normalized_eval, num_parameters=4),
        measure_complexity(
            "logistic",
            lambda x: score_candidates(x, ScorerConfig(ScorerType.LOGISTIC)),
            normalized_eval,
            num_parameters=6,  # 4 field weights + bias + steepness
        ),
        measure_complexity(
            "knapsack",
            lambda x: score_candidates(x, ScorerConfig(ScorerType.KNAPSACK)),
            normalized_eval,
            num_parameters=3,  # value weights over the non-cost fields
        ),
    ]
    print("\nscorer complexity (median of repeated timed runs, after warm-up):")
    for report in complexity:
        print(f"  {report.scorer_name}: {report.latency_seconds_per_candidate:.2e}s/candidate "
              f"(min {report.latency_min_seconds_per_candidate:.2e}, max {report.latency_max_seconds_per_candidate:.2e}), "
              f"{report.num_parameters} params")
    fuzzy_lat = complexity[0].latency_seconds_per_candidate
    cheapest = min(complexity[1:], key=lambda c: c.latency_seconds_per_candidate)
    print(f"  -> fuzzy inference overhead vs cheapest competitor ({cheapest.scorer_name}): "
          f"{fuzzy_lat / max(cheapest.latency_seconds_per_candidate, 1e-12):.1f}x")

    bp_sensitivity = sensitivity_to_breakpoints(normalized_eval)
    rule_sensitivity = sensitivity_to_rules(normalized_eval)

    # Human-readable rule traces, exported for dashboard inspection (Prompt 13
    # acceptance gate). Every candidate's full per-rule firing strengths are
    # written to the artifact; a readable summary of which rules actually fired
    # is printed for the first few.
    fuzzy_results = fuzzy_repair_priority(normalized_eval)
    rule_traces: List[Dict[str, Any]] = [
        {
            "candidate_index": i,
            "inputs": {name: round(float(values[i].item()), 4)
                       for name, values in normalized_eval.as_dict().items()},
            "priority": round(r.priority, 4),
            "fired_rules": [t.to_dict() for t in r.fired_rules()],
            "all_rules": [t.to_dict() for t in r.rule_trace],
        }
        for i, r in enumerate(fuzzy_results)
    ]
    print("\nrule traces (first 3 candidates, fired rules only):")
    for entry in rule_traces[:3]:
        fired = ", ".join(f"{t['rule_name']}->{t['consequent_level']}@{t['firing_strength']:.2f}"
                          for t in entry["fired_rules"])
        print(f"  candidate {entry['candidate_index']}: priority={entry['priority']:.3f} | {fired or 'no rule fired'}")

    # end-to-end: fuzzy scorer output drives the UNCHANGED Prompt 9 repair contract.
    q = torch.ones(2, 4)
    g = torch.Generator().manual_seed(3)
    k = torch.randn(16, 4, generator=g)
    v = torch.randn(16, 4, generator=g)
    evicted0 = torch.rand(16, generator=g) < 0.4
    priority = score_candidates(normalized_eval, ScorerConfig(ScorerType.FUZZY))
    reintroduce, evict = propose_repair_swap(priority, evicted0, n=2)
    contract = RepairContract(delta=0.05)
    new_evicted = contract.evaluate_swap(q, k, v, evicted0, reintroduce=reintroduce, evict=evict)
    swap_accepted = bool(contract.accepted_actions())
    print(f"proposed swap accepted by RepairContract: {swap_accepted}")

    membership_plot = None
    try:
        membership_plot = plot_fuzzy_membership_functions(str(out))
        print(f"saved plot -> {membership_plot}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")

    result = {
        "num_train_candidates": train.fragility.shape[0],
        "num_eval_candidates": eval_candidates.fragility.shape[0],
        "ablation_priorities": {name: scores.tolist() for name, scores in ablation.items()},
        "complexity": [c.to_dict() for c in complexity],
        "breakpoint_sensitivity": bp_sensitivity,
        "rule_sensitivity": rule_sensitivity,
        "repair_integration": {
            "reintroduce": reintroduce, "evict": evict, "swap_accepted": swap_accepted,
            "kept_count_preserved": bool(int((~new_evicted).sum().item()) == int((~evicted0).sum().item())),
        },
        "rule_traces": rule_traces,
        "membership_plot": membership_plot,
    }
    (out / "scorer_comparison.json").write_text(json.dumps(result, indent=2))
    print(f"saved -> {out/'scorer_comparison.json'}")


if __name__ == "__main__":
    main()
