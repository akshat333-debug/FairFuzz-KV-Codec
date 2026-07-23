from enum import Enum
from typing import List

from pydantic import BaseModel

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.stats_utils import EffectSizeResult

# =============================================================================
# PRE-REGISTERED Gate 1 decision thresholds.
#
# These numbers are fixed HERE, committed to git, and tested against synthetic
# fixtures (tests/benchmarks/fragkv_minpairs/test_gate1.py) BEFORE this module
# is ever pointed at real study predictions. Per Prompt 5's non-negotiable
# instruction: "PASS/FAIL logic is explicit and not rewritten after seeing
# results." Do not edit these constants after a real Gate 1 study has run -
# if the thresholds turn out to be wrong, that is itself a finding to report,
# not a reason to move the goalposts retroactively.
# =============================================================================

# A paired accuracy drop (n_g=1 vs n_g=8, same codec) at or above this is
# "practically meaningful" - not just statistically distinguishable from zero.
PRACTICAL_EFFECT_THRESHOLD = 0.10

# Below PRACTICAL but at or above this: a real but small signal.
WEAK_EFFECT_THRESHOLD = 0.03

# The FullKV (lossless) control must show LESS than this paired effect, or
# the study can't attribute any lossy-codec effect specifically to
# compression rather than to fragmentation confusing the base model/tokenizer
# regardless of compression.
CONTROL_CONFOUND_THRESHOLD = 0.03


class Gate1Decision(str, Enum):
    PASS = "PASS"
    WEAK_PASS = "WEAK_PASS"
    FAIL = "FAIL"


class Gate1Report(BaseModel):
    decision: Gate1Decision
    control_result: EffectSizeResult
    lossy_results: List[EffectSizeResult]
    control_confound: bool
    reasoning: str


def _qualifies(result: EffectSizeResult, threshold: float) -> bool:
    return result.n_paired_groups > 0 and result.monotonic_non_increasing and result.effect_size >= threshold


def decide_gate1(control_result: EffectSizeResult, lossy_results: List[EffectSizeResult]) -> Gate1Report:
    """Pure function of its inputs - no I/O, no randomness, no hidden state.
    Given the same EffectSizeResults it always returns the same decision."""
    control_confound = abs(control_result.effect_size) >= CONTROL_CONFOUND_THRESHOLD

    meaningful = [r for r in lossy_results if _qualifies(r, PRACTICAL_EFFECT_THRESHOLD)]
    weak = [r for r in lossy_results if _qualifies(r, WEAK_EFFECT_THRESHOLD) and r not in meaningful]

    if meaningful and not control_confound:
        decision = Gate1Decision.PASS
        reasoning = (
            f"{len(meaningful)} lossy codec(s) show a directionally consistent, practically "
            f"meaningful paired effect (>= {PRACTICAL_EFFECT_THRESHOLD:.0%}), and the FullKV "
            f"control shows no comparable confound (|effect|={abs(control_result.effect_size):.3f} "
            f"< {CONTROL_CONFOUND_THRESHOLD:.0%})."
        )
    elif meaningful and control_confound:
        decision = Gate1Decision.WEAK_PASS
        reasoning = (
            f"{len(meaningful)} lossy codec(s) show a meaningful paired effect, but the FullKV "
            f"control ALSO shows a comparable effect (|effect|={abs(control_result.effect_size):.3f} "
            f">= {CONTROL_CONFOUND_THRESHOLD:.0%}), so the effect cannot be cleanly attributed to "
            "compression rather than fragmentation confusing the base model regardless of codec."
        )
    elif weak:
        decision = Gate1Decision.WEAK_PASS
        reasoning = (
            f"{len(weak)} lossy codec(s) show a directionally consistent effect between "
            f"{WEAK_EFFECT_THRESHOLD:.0%} and {PRACTICAL_EFFECT_THRESHOLD:.0%} - real signal, "
            "but below the practically-meaningful bar."
        )
    else:
        decision = Gate1Decision.FAIL
        reasoning = (
            "No lossy codec showed a directionally consistent effect of at least "
            f"{WEAK_EFFECT_THRESHOLD:.0%}. Fragmentation does not appear to causally predict "
            "additional compression failure in this study."
        )

    return Gate1Report(
        decision=decision,
        control_result=control_result,
        lossy_results=lossy_results,
        control_confound=control_confound,
        reasoning=reasoning,
    )
