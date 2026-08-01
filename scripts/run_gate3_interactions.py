"""Gate 3 item-119 interaction effects: model family x allocator, and
quantizer type x cohort.

These two were previously deferred as needing "a second full allocator study"
on the slower model. That estimate was wrong: the Prompt 10/11 allocator path
is ONE prefill capture plus quantize/dequantize calibration - it never runs
autoregressive generation - so both interactions are cheap enough to measure
directly. This script does exactly that, on the same two model/tokenizer
families frozen in GATE3_CONFIG.md.

Reported, never inferred:
  * model family x allocator - does the aggregate-vs-minimax allocator
    comparison behave the same way on both families, or does the allocator
    ranking depend on the model?
  * quantizer type x cohort - is the best quantizer option the SAME for every
    cohort, or does the winning quantizer depend on the cohort? (A crossover
    is the interaction; a uniform winner is its absence.)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from fairfuzzkv_codec.allocation.allocator import solve_exact  # noqa: E402
from fairfuzzkv_codec.allocation.calibration import calibrate_layers_mixed  # noqa: E402
from fairfuzzkv_codec.allocation.minimax import solve_minimax_exact, worst_distortion  # noqa: E402
from fairfuzzkv_codec.cache_capture.hf_capture import HFCapture  # noqa: E402
from fairfuzzkv_codec.core.config import LayerHeadSelection  # noqa: E402

# The two frozen Gate 3 families (GATE3_CONFIG.md).
FAMILIES = [
    ("Qwen/Qwen2.5-0.5B", "byte_level_bpe"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "sentencepiece"),
]

TEXT = (
    "Cross-model reproduction of the rate-distortion allocator: the same frozen "
    "calibration and allocation protocol is applied to two tokenizer families to "
    "test whether the allocator's behaviour transfers or is model-specific."
)


def _family_result(model_name: str, tokenizer_family: str) -> Dict[str, Any]:
    print(f"\n=== {model_name} ({tokenizer_family}) ===")
    capture = HFCapture(model_name, device="cpu", dtype=torch.float32)
    K, _V = capture.capture_prefill_kv(TEXT, LayerHeadSelection())
    print(f"  captured K shape={tuple(K.shape)}")

    cohorts = calibrate_layers_mixed(K, scalar_bits=[4, 8], lbg_configs=[(8, 16), (8, 64)])
    lo = sum(min(o.total_bits for o in c.options) for c in cohorts)
    hi = sum(max(o.total_bits for o in c.options) for c in cohorts)
    budget = (lo + hi) // 2

    aggregate = solve_exact(cohorts, budget)
    minimax = solve_minimax_exact(cohorts, budget)
    agg_worst = worst_distortion(aggregate, cohorts)
    agg_mean = aggregate.total_distortion / len(cohorts)

    print(f"  budget={budget}  aggregate: worst={agg_worst:.6f} mean={agg_mean:.6f}")
    print(f"                   minimax:   worst={minimax.worst_distortion:.6f} mean={minimax.average_distortion:.6f}")

    # ---- quantizer type x cohort -------------------------------------------
    # For every cohort, which quantizer option achieves the lowest distortion?
    # If the winner is identical everywhere there is no interaction; if it
    # changes by cohort, that crossover IS the interaction.
    best_by_cohort = {}
    per_cohort_options = {}
    for c in cohorts:
        ranked = sorted(c.options, key=lambda o: o.distortion)
        best_by_cohort[c.cohort_id] = ranked[0].label
        per_cohort_options[c.cohort_id] = {o.label: o.distortion for o in c.options}
    winners = sorted(set(best_by_cohort.values()))
    quantizer_cohort_interaction = len(winners) > 1
    print(f"  quantizer x cohort: {len(winners)} distinct winning quantizer(s) across "
          f"{len(cohorts)} cohorts -> interaction={quantizer_cohort_interaction} ({winners})")

    return {
        "model_name": model_name,
        "tokenizer_family": tokenizer_family,
        "num_cohorts": len(cohorts),
        "budget": budget,
        "allocator": {
            "aggregate": {"worst": agg_worst, "mean": agg_mean, "bits": aggregate.total_bits},
            "minimax": {
                "worst": minimax.worst_distortion, "mean": minimax.average_distortion,
                "bits": minimax.allocation.total_bits,
            },
            "minimax_improves_worst": bool(minimax.worst_distortion <= agg_worst + 1e-12),
            "allocations_identical": bool(
                {k: v.label for k, v in aggregate.choice.items()}
                == {k: v.label for k, v in minimax.allocation.choice.items()}
            ),
        },
        "quantizer_x_cohort": {
            "best_quantizer_by_cohort": best_by_cohort,
            "distinct_winners": winners,
            "interaction_present": quantizer_cohort_interaction,
            "per_cohort_distortion": per_cohort_options,
        },
    }


def main() -> None:
    out = Path("gate3_study")
    out.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for model_name, family in FAMILIES:
        try:
            results.append(_family_result(model_name, family))
        except Exception as e:  # noqa: BLE001
            # A family that cannot be run is recorded as such, never silently
            # dropped or replaced with the other family's numbers.
            print(f"  FAILED for {model_name}: {e}")
            results.append({"model_name": model_name, "tokenizer_family": family, "error": str(e)})

    ok = [r for r in results if "error" not in r]

    # ---- model family x allocator interaction ------------------------------
    interaction: Dict[str, Any] = {"analyzed": len(ok) >= 2}
    if len(ok) >= 2:
        same_direction = len({r["allocator"]["minimax_improves_worst"] for r in ok}) == 1
        same_identity = len({r["allocator"]["allocations_identical"] for r in ok}) == 1
        interaction.update({
            "minimax_improves_worst_by_family": {
                r["model_name"]: r["allocator"]["minimax_improves_worst"] for r in ok
            },
            "allocations_identical_by_family": {
                r["model_name"]: r["allocator"]["allocations_identical"] for r in ok
            },
            "allocator_behaviour_consistent_across_families": bool(same_direction and same_identity),
        })
        print("\n=== model family x allocator ===")
        for r in ok:
            print(f"  {r['model_name']}: minimax_improves_worst="
                  f"{r['allocator']['minimax_improves_worst']} "
                  f"allocations_identical={r['allocator']['allocations_identical']}")
        print(f"  -> allocator behaviour consistent across families: {same_direction and same_identity}")

    # ---- quantizer x cohort, across families -------------------------------
    if ok:
        print("\n=== quantizer type x cohort ===")
        for r in ok:
            qc = r["quantizer_x_cohort"]
            print(f"  {r['model_name']}: interaction_present={qc['interaction_present']} "
                  f"winners={qc['distinct_winners']}")

    (out / "gate3_interactions.json").write_text(json.dumps({
        "frozen_families": [{"model": m, "tokenizer_family": f} for m, f in FAMILIES],
        "per_family": results,
        "model_family_x_allocator": interaction,
        "note": (
            "Capture + quantization only - no autoregressive generation - which is why "
            "these interactions are cheap to measure despite being deferred earlier. "
            "Single text, single budget per family: a pilot-scale interaction probe."
        ),
    }, indent=2))
    print(f"\nsaved -> {out/'gate3_interactions.json'}")


if __name__ == "__main__":
    main()
