from fairfuzzkv_codec.benchmarks.fragkv_minpairs.gate1 import Gate1Decision, Gate1Report

PIVOT_PLAN_TEMPLATE = """\
# Pivot Plan: Gate 1 = FAIL

Gate 1 (FragKV-MinPairs causal test) did not find that evidence-unit
fragmentation causally predicts additional compression failure. Per the
Prompt 5 non-negotiable instruction, we do not continue by assuming the
fairness/fragility hypothesis anyway.

## What Gate 1 found

{reasoning}

## What is preserved

The codec deliverable (Modules 1-4: capture, unicode grouping, fragility
features, quantization/pruning baselines, matched-bit evaluation, byte
accounting) is unaffected by this result. All of it stands on its own as a
KV-cache compression system with honest accounting - none of that work
depended on fragmentation being a causal driver of failure.

## What must change in the project report

- Remove or qualify any claim that tokenizer fragmentation "causes",
  "drives", or "explains" compression degradation for any language/script
  group. That specific causal claim is not supported by this study.
- Fragility cohorts (Module 2, `fragility_estimation`) may still be reported
  as a *descriptive* risk-scoring mechanism (per-unit features, transparent
  score, cohort bands) - just not as a *causally validated* basis for
  allocation decisions.
- Any planned allocation/pruning policy that spends compute or bit-budget on
  the premise "high-fragility cohorts need protection" should be re-labeled
  as a *heuristic* choice, not an empirically justified one, or dropped.

## Recommended next steps

1. Report this as a negative/null result for the causal hypothesis - that is
   itself a valid scientific contribution, not a failure of the project.
2. If pursuing fairness claims further, scale FragKV-MinPairs to more
   groups/codecs/models before concluding the null result generalizes (see
   the power_note on each EffectSizeResult - this study may simply be
   under-powered rather than truly null).
3. Otherwise, proceed with the codec engineering roadmap (allocation,
   pruning, quantization, decoder) purely as a compression system, without
   fairness-causal framing.
"""


def generate_pivot_plan(report: Gate1Report) -> str:
    if report.decision != Gate1Decision.FAIL:
        raise ValueError(
            f"generate_pivot_plan should only be called on a FAIL decision, got {report.decision}"
        )
    return PIVOT_PLAN_TEMPLATE.format(reasoning=report.reasoning)
