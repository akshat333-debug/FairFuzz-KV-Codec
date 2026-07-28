"""Module 3 demo: compare the fuzzy repair-priority scorer against its three
simpler competitors on synthetic candidate groups (this module scores
already-computed structural signals, not raw KV tensors, so no HF model
capture is needed here - unlike Prompts 5/6). Numbers are measured, never
invented.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from fairfuzzkv_codec.dashboard.plots import plot_fuzzy_membership_functions  # noqa: E402
from fairfuzzkv_codec.pruning.repair import RepairContract  # noqa: E402
from fairfuzzkv_codec.repair_scoring.ablation import ScorerConfig, ScorerType, run_ablation, score_candidates  # noqa: E402
from fairfuzzkv_codec.repair_scoring.competitors import monotone_weighted_score  # noqa: E402
from fairfuzzkv_codec.repair_scoring.fuzzy import fuzzy_priority_scores  # noqa: E402
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

    complexity = [
        measure_complexity("fuzzy", fuzzy_priority_scores, normalized_eval, fuzzy_num_parameters()),
        measure_complexity("monotone", monotone_weighted_score, normalized_eval, num_parameters=4),
    ]
    for report in complexity:
        print(f"  {report.scorer_name}: {report.latency_seconds_per_candidate:.2e}s/candidate, "
              f"{report.num_parameters} params")

    bp_sensitivity = sensitivity_to_breakpoints(normalized_eval)
    rule_sensitivity = sensitivity_to_rules(normalized_eval)

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
        "membership_plot": membership_plot,
    }
    (out / "scorer_comparison.json").write_text(json.dumps(result, indent=2))
    print(f"saved -> {out/'scorer_comparison.json'}")


if __name__ == "__main__":
    main()
