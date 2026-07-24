from fairfuzzkv_codec.allocation.allocator import (
    Allocation,
    BitOption,
    Cohort,
    optimality_gap,
    solve_exact,
    solve_greedy,
)
from fairfuzzkv_codec.allocation.calibration import (
    Split,
    allocation_to_bitwidth_map,
    calibrate_layers_mixed,
    calibrate_layers_scalar,
    encode_with_allocation,
    make_split,
)
from fairfuzzkv_codec.allocation.curves import (
    DistortionCurve,
    ExpFit,
    fit_exponential,
    marginal_decay,
)
from fairfuzzkv_codec.allocation.minimax import (
    MinimaxResult,
    allocation_shift,
    pareto_frontier,
    solve_continuous_minimax,
    solve_minimax_exact,
    solve_minimax_waterfill,
    worst_distortion,
)

__all__ = [
    "Allocation", "BitOption", "Cohort", "solve_exact", "solve_greedy", "optimality_gap",
    "Split", "make_split", "calibrate_layers_scalar", "calibrate_layers_mixed",
    "allocation_to_bitwidth_map", "encode_with_allocation",
    "DistortionCurve", "ExpFit", "fit_exponential", "marginal_decay",
    "MinimaxResult", "solve_minimax_exact", "solve_minimax_waterfill",
    "solve_continuous_minimax", "worst_distortion", "pareto_frontier", "allocation_shift",
]
