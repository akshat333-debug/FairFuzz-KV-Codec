"""IndicLongComp: a parallel multilingual/code-mixed long-context benchmark.

**Content provenance (read before using):** every context/question string in
this benchmark is LLM-authored from hand-designed parallel templates, not
sourced from any external corpus (MLRBench or otherwise - no verified
network/license access was available when this was built) and not reviewed
by a professional translator or native-speaker annotator. See
`schema.ContentProvenance` and every `DatasetCard.content_provenance_note`.

"Parallel" here is used in the STRONG, VERIFIED sense the project's own
non-negotiable instruction requires: every language variant of a group is
generated from the SAME drawn random values (names, digits, distractor
count, evidence position) through a template with the SAME slot structure,
so the canonical answer - always a single language-independent digit - is
IDENTICAL by construction across all four language conditions, not merely
"translated" and assumed equivalent. `validators.validate_parallelism`
checks this mechanically on every group, not just at generation time.
"""
