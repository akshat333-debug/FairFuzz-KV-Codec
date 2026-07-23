"""Aggregate rate-distortion allocation on a real captured cache.

Calibrates per-layer distortion (train/val/test separated), fits distortion
curves, solves the aggregate allocation exactly and greedily, reports the
optimality gap, saves diagnostics + a budget-allocation plot, and drives the
REAL encoder with the chosen bit-widths. This is the Gate-2 control condition.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from fairfuzzkv_codec.allocation import (  # noqa: E402
    calibrate_layers_scalar,
    encode_with_allocation,
    optimality_gap,
    solve_exact,
    solve_greedy,
)
from fairfuzzkv_codec.allocation.curves import DistortionCurve, marginal_decay  # noqa: E402
from fairfuzzkv_codec.cache_capture.hf_capture import HFCapture  # noqa: E402
from fairfuzzkv_codec.core.config import LayerHeadSelection  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    out = Path("allocation_study")
    out.mkdir(exist_ok=True)

    capture = HFCapture(MODEL_NAME, device="cpu", dtype=torch.float32)
    K, _V = capture.capture_prefill_kv(
        "Aggregate rate-distortion allocation over transformer KV cache layers, "
        "calibrated per layer and solved to optimality as the fairness control.",
        LayerHeadSelection(),
    )
    print(f"captured K shape={tuple(K.shape)}")

    cohorts = calibrate_layers_scalar(K, bit_choices=[4, 8])
    lo = sum(min(o.total_bits for o in c.options) for c in cohorts)
    hi = sum(max(o.total_bits for o in c.options) for c in cohorts)
    budget = (lo + hi) // 2
    print(f"budget window [{lo}, {hi}], using B={budget}")

    exact = solve_exact(cohorts, budget)
    greedy = solve_greedy(cohorts, budget)
    gap = optimality_gap(exact, greedy)
    print(f"exact distortion={exact.total_distortion:.6f} bits={exact.total_bits}")
    print(f"greedy distortion={greedy.total_distortion:.6f} bits={greedy.total_bits}")
    print(f"optimality gap={gap:.4f}")

    # curve diagnostics per cohort
    curves = {}
    for c in cohorts:
        bits = [o.total_bits for o in c.options]
        dist = [o.distortion for o in c.options]
        # normalize bits to per-element for a readable curve
        curve = DistortionCurve([b / K[0:1].numel() for b in bits], dist)
        curves[c.cohort_id] = {
            "diagnostics": curve.diagnostics(),
            "marginal_decay": marginal_decay(curve),
        }

    # drive the real encoder with the exact allocation
    stream, meta = encode_with_allocation(K, exact, tensor_name="k")
    report = meta["accountant_report"]
    assert isinstance(report, dict)
    real_bits = report["serialized_bytes"] * 8
    print(f"REAL encode with allocation: {real_bits} bits (<= budget {budget}: {real_bits <= budget})")

    result = {
        "budget": budget,
        "exact": exact.to_dict(),
        "greedy": greedy.to_dict(),
        "optimality_gap": gap,
        "real_encode_bits": real_bits,
        "real_encode_within_budget": bool(real_bits <= budget),
        "curves": curves,
    }
    (out / "allocation_result.json").write_text(json.dumps(result, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(exact.choice.keys())
        bits = [exact.choice[k].total_bits for k in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, bits)
        ax.set_ylabel("allocated serialized bits")
        ax.set_title(f"Exact allocation (B={budget}, gap vs greedy={gap:.3f})")
        fig.tight_layout()
        fig.savefig(out / "budget_allocation.png", dpi=100)
        print(f"saved plot -> {out/'budget_allocation.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")

    print(f"saved -> {out/'allocation_result.json'}")


if __name__ == "__main__":
    main()
