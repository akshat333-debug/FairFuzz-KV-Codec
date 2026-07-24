"""Gate-2 comparison: aggregate (control) vs fairness-constrained minimax.

Runs the FROZEN configuration in GATE2_CONFIG.md on a real captured cache:
calibrates per-layer cohorts (train/test separated), solves the aggregate
allocation and the minimax allocation at the same budget, reports worst /
average distortion, the Pareto frontier, and the per-cohort allocation shift,
and saves artifacts + a plot. Numbers are measured, never invented.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from fairfuzzkv_codec.allocation import (  # noqa: E402
    allocation_shift,
    calibrate_layers_mixed,
    pareto_frontier,
    solve_exact,
    solve_minimax_exact,
    solve_minimax_waterfill,
    worst_distortion,
)
from fairfuzzkv_codec.cache_capture.hf_capture import HFCapture  # noqa: E402
from fairfuzzkv_codec.core.config import LayerHeadSelection  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    out = Path("gate2_study")
    out.mkdir(exist_ok=True)

    capture = HFCapture(MODEL_NAME, device="cpu", dtype=torch.float32)
    K, _V = capture.capture_prefill_kv(
        "Fairness-constrained minimax allocation minimizes the worst-cohort "
        "distortion at a fixed KV-cache bit budget, as the Gate-2 experimental "
        "condition against the aggregate-optimal control.",
        LayerHeadSelection(),
    )
    print(f"captured K shape={tuple(K.shape)}")

    cohorts = calibrate_layers_mixed(K, scalar_bits=[4, 8], lbg_configs=[(8, 16), (8, 64)])
    lo = sum(min(o.total_bits for o in c.options) for c in cohorts)
    hi = sum(max(o.total_bits for o in c.options) for c in cohorts)
    budget = (lo + hi) // 2
    print(f"budget window [{lo}, {hi}], using B={budget}")

    aggregate = solve_exact(cohorts, budget)
    mm_exact = solve_minimax_exact(cohorts, budget)
    mm_wf = solve_minimax_waterfill(cohorts, budget)

    agg_worst = worst_distortion(aggregate, cohorts)
    print(f"aggregate: worst={agg_worst:.6f} avg={aggregate.total_distortion / len(cohorts):.6f} bits={aggregate.total_bits}")
    print(f"minimax exact: worst={mm_exact.worst_distortion:.6f} avg={mm_exact.average_distortion:.6f} bits={mm_exact.allocation.total_bits}")
    print(f"minimax waterfill: worst={mm_wf.worst_distortion:.6f} avg={mm_wf.average_distortion:.6f} bits={mm_wf.allocation.total_bits}")

    fairness_helps = mm_exact.worst_distortion <= agg_worst + 1e-9
    wf_matches = abs(mm_wf.worst_distortion - mm_exact.worst_distortion) < 0.5
    no_overflow = mm_exact.allocation.total_bits <= budget and mm_wf.allocation.total_bits <= budget
    shift = allocation_shift(aggregate, mm_exact.allocation)
    print(f"fairness improves worst case: {fairness_helps}")
    print(f"waterfill matches exact: {wf_matches}")
    print(f"no budget overflow: {no_overflow}")
    print(f"allocation shifts (aggregate -> minimax): {shift}")

    budgets = [lo + int((hi - lo) * f / 8) for f in range(9)]
    frontier = pareto_frontier(cohorts, budgets)

    result = {
        "frozen_config": "GATE2_CONFIG.md",
        "budget": budget,
        "aggregate": {"worst": agg_worst, "avg": aggregate.total_distortion / len(cohorts),
                      "bits": aggregate.total_bits, "choice": aggregate.to_dict()["choice"]},
        "minimax_exact": mm_exact.to_dict(),
        "minimax_waterfill": mm_wf.to_dict(),
        "checks": {"fairness_improves_worst": bool(fairness_helps),
                   "waterfill_matches_exact": bool(wf_matches),
                   "no_budget_overflow": bool(no_overflow)},
        "allocation_shift": {k: list(v) for k, v in shift.items()},
        "pareto_frontier": frontier,
    }
    (out / "gate2_result.json").write_text(json.dumps(result, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feas = [p for p in frontier if p["feasible"] > 0]
        b = [p["budget"] for p in feas]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(b, [p["worst_distortion"] for p in feas], "o-", label="worst-cohort (minimax)")
        ax.plot(b, [p["average_distortion"] for p in feas], "s--", label="average")
        ax.set_xlabel("total bit budget")
        ax.set_ylabel("distortion (MSE)")
        ax.set_title("Gate-2 Pareto: cost of fairness (worst vs average)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "pareto_frontier.png", dpi=100)
        print(f"saved plot -> {out/'pareto_frontier.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")

    print(f"saved -> {out/'gate2_result.json'}")


if __name__ == "__main__":
    main()
