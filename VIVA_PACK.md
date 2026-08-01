# Viva Defence Pack

100 likely questions with concise, defensible answers, plus architecture
walkthrough, proof sketches, failure scenarios, and a live-demo recovery plan.

**The one-sentence defence:** *The engineering deliverable works and is measured
honestly; the research hypothesis was pre-registered, tested, and largely not
supported — and the project reports that rather than hiding it.*

---

## A. Problem & motivation (1–8)

**1. What problem does this solve?**
KV cache dominates memory in long-context LLM serving. Compressing it raises
serving capacity per unit of memory.

**2. Why the fairness angle?**
Tokenizers fragment languages unequally. If compression damages fragmented
evidence more, it degrades non-English users silently.

**3. Did you prove that?**
No. Gate 1 was WEAK_PASS and Gate 2 FAILED. I report it as unsupported at this
scale.

**4. So what survives?**
The codec: format, quantizers, allocators, pruning, metadata coding, byte
accounting — all independent of the fairness claim.

**5. Why is memory the right target, not speed?**
Measured: prefill 115–208 ms vs 0.6–2.4 ms encode. The codec is ~100x cheaper
than the model, so it cannot meaningfully change latency.

**6. Who would use this?**
Researchers studying compression/fairness interactions, and as a teaching
artifact for rate-distortion and VQ.

**7. Is it production-ready?**
No. There is no serving-engine integration; decode numbers are replay-harness.

**8. What is the single most important result?**
That two of four pre-registered gates failed and the project says so.

## B. Architecture (9–18)

**9. Walk me through the pipeline.**
Model → capture → [Unicode grouping + fragility] → allocation → [pruning +
quantization] → metadata coding → FFK1 container → decoder → reconstructed cache.

**10. Why separate prefill and decode from commit one?**
They have different constraints — prefill is bulk, decode is incremental. Mixing
them produces misleading benchmarks. `encode_prefill` vs `encode_decode_step`.

**11. What is Module 1?**
Unicode grouping: UAX #29 grapheme clusters → surface units → tokenizer
alignment via offset mappings, with an explicit quarantine path.

**12. Module 2?**
Fragility estimation: per-group features → transparent monotone risk score →
quantile cohorts.

**13. Why a *transparent* score if you also have a calibrated model?**
The transparent score is the mandatory audit baseline. A learned model is only
used if it beats it on held-out data.

**14. What is the leakage guard?**
Raw language/script labels and task/compression outcomes are structurally
forbidden from reaching any risk score or cohort. It raises, not warns.

**15. Why does the group mapper quarantine instead of guessing?**
A wrong alignment silently corrupts every downstream feature. Explicit failure
is recoverable; a silent guess is not.

**16. How do modules stay decoupled?**
Selectors are quantization-independent; the codec works if `repair_scoring` is
deleted entirely.

**17. Where is the byte accountant?**
`codec/binary_serializer.py::ByteAccountant` — counts codes, scales,
zero-points, masks, indices, headers, alignment, metadata.

**18. Can I remove the fuzzy module?**
Yes — that was a stated design requirement, and the codec is byte-identical
without it.

## C. Quantization (19–32)

**19. What scalar options exist?**
INT8/INT4, symmetric/asymmetric, per-tensor/head/channel/groupwise, percentile
and MSE-optimal clipping.

**20. Is INT4 really 4-bit?**
Yes — two values per byte, verified by a byte-count test (8 values → 4 bytes).

**21. Why does that matter?**
Storing INT4 in an int8 container wastes half the bits while claiming 4-bit
compression. That would be dishonest accounting.

**22. Explain LBG.**
Start from the global centroid; repeatedly split each centroid (c → c(1±ε)) and
run Lloyd iterations until the codebook reaches the target size.

**23. Why splitting rather than random init?**
Deterministic and reproducible; avoids the seed-sensitivity of random k-means
init.

**24. Empty clusters?**
Reseeded by splitting the most-populated centroid, so the codebook stays fully
utilized instead of carrying dead entries.

**25. Is the codebook counted in the bits?**
Yes — serialized and counted. Otherwise the compression ratio would be fiction.

**26. When is LBG worse than scalar?**
Small corpora. Measured: on a 3-token cache LBG was infeasible or worse than
INT4, because fixed codebook cost has nothing to amortize over.

**27. LBG's cost?**
~40x slower encode (49.7–83.8 ms vs 0.6–2.0 ms) because codebook training
happens at encode. Decode is as fast as scalar — a table lookup.

**28. So when would you use LBG?**
Write-once/read-many caches where 2.24 bits/element at lower MSE than INT4-at-4
bits justifies encode cost.

**29. Groupwise scales — what bug did you hit?**
Scales were stored broadcast to every element, inflating INT4-groupwise from ~4
to ~68 bits/element. Caught by a benchmark, not the unit tests.

**30. Lesson from that?**
Round-trip correctness tests don't catch storage-accounting bugs. Always assert
bits/element too — there's now a regression test.

**31. The asymmetric INT8 bug?**
Unsigned values 0–255 were cast to signed int8, wrapping anything ≥128. Fixed by
matching the container dtype to signedness.

**32. Does MSE predict task accuracy?**
Measured: not always. At 8 bits, per-tensor scored *higher* accuracy than
per-channel despite 18x worse MSE. Reported as-is.

## D. Binary format (33–42)

**33. Why not pickle?**
Pickle is version-fragile, unreadable across languages, and executes code. The
prompt forbade it and it's the right call.

**34. Format structure?**
Magic `FFK1`, version, endianness byte, geometry+tokenizer-hash header, section
directory, per-section + geometry + whole-file CRC32, trailer.

**35. Forward compatibility?**
Readers skip unknown section types using the directory's offset/length. A v1
reader tolerates v2 sections.

**36. Corruption handling?**
Three CRC layers plus bounds checks on every offset. Any inconsistency raises
`CorruptContainerError`.

**37. How do you know it's robust?**
Fuzz tested: 300 random blobs, 300 single-bit flips. Only the documented
exception is acceptable.

**38. Golden vectors?**
Exact hex bytes pinned in a test, so any accidental layout change fails loudly.

**39. Golomb-Rice — what and why?**
Retention positions gap-coded then Rice-coded; near-optimal for geometrically
distributed gaps, i.e. sparse retention.

**40. Adaptive k?**
5-bit k written inline per 64-value block — its side information is counted.

**41. What if Rice is a bad fit?**
Bitmap and run-length fallbacks. The encoder computes all three and emits the
**shortest measured**, never assumed.

**42. Evidence it works?**
5%-dense over 10,000 positions beats raw 32-bit indices; dense alternating falls
back to bitmap. Both tested.

## E. Pruning & the bound (43–52)

**43. Selectors?**
Recency, top-attention-mass, top-k score, group-aware — all
quantization-independent.

**44. Why group-aware?**
Retaining half a surface unit is incoherent: partial evidence can be worse than
none.

**45. State the bound.**
For one head, evicting E and renormalizing: ‖O−Ô‖₂ ≤ 2·M·p_E, where M =
max‖v_i‖ and p_E is evicted attention mass.

**46. Prove it.**
Ô is a renormalized convex combination over kept positions. O−Ô is a difference
of two convex combinations of the same value vectors; each has total mass 1, and
they differ on at most mass p_E on each side. Norm ≤ p_E·M + p_E·M = 2·M·p_E.

**47. Is that end-to-end?**
**No.** Local, single-head, single-layer, under the renormalization assumption.
I never call it a generation guarantee.

**48. What if the kept set is empty?**
Renormalization is undefined. The validator reports an assumption failure rather
than manufacturing a pass — tested.

**49. Is the bound tight?**
It's an upper bound; observed slack is logged per query so tightness is
inspectable.

**50. What is the repair contract?**
Budget-neutral swaps accepted only if worst-query p_E^repair ≤ p_E^0 + δ.

**51. Why budget-neutral?**
Otherwise "repair" would just spend more bits, and any improvement would be
trivially explained by budget rather than by the policy.

**52. Adversarial cases you tested?**
Concentrated attention, huge value norms, duplicated subtokens, zero-attention
groups, and evicting the sole dominant key.

## F. Allocation (53–64)

**53. What's the aggregate allocator?**
Minimize Σ D_l(b_l) subject to Σ cost_l ≤ B — a multiple-choice knapsack.

**54. Solvers?**
Exact DP reference plus a greedy water-filling approximation.

**55. How do you know the exact solver is exact?**
Verified against independent brute-force enumeration on 120 random instances.

**56. Greedy quality?**
Never beats the optimum; max observed gap < 0.5 across 120 instances.

**57. What does minimax optimize?**
The worst-cohort distortion at fixed budget.

**58. Derive it.**
Epigraph form, KKT: Σμ_l = 1 and λ = μ_l β_l D_l(x_l). On the active set
D_l(x_l) = t, so x_l = (ln α_l − Λ)/β_l with Λ fixed by the budget.

**59. Does it equalize β?**
**No** — and this is the question I most expect. β_l are fixed curve parameters,
not decision variables. What is equalized is the achieved **distortion**.

**60. Why does the distinction matter?**
Saying "equalizes β" would be mathematically false; I corrected it rather than
coding the informal statement.

**61. Active-set iteration?**
Any cohort with x_l < 0 already sits below the floor (α_l ≤ t); drop it, set
x_l = 0, re-solve. Terminates in ≤ L steps.

**62. Infeasible budgets?**
Reported as infeasible, not silently clamped.

**63. Does allocation drive the real encoder?**
Yes — `allocation_to_bitwidth_map` → `ScalarQuantCodec`, verified within budget
on a real capture.

**64. Is allocation premised on Gate 1?**
No. Gate 1 was WEAK_PASS, so allocation is framed as an engineering control, not
a validated causal remedy.

## G. Gates & methodology (65–80)

**65. Why pre-register?**
So thresholds cannot be tuned after seeing results. They're frozen in code and
unit-tested against synthetic fixtures **before** each real run.

**66. Gate 1 result?**
WEAK_PASS. Only TopK50 showed a directional effect (7.5 pts, p=0.0004), below
the 10-pt practical bar.

**67. Why not PASS with p=0.0004?**
p-values alone were explicitly insufficient by pre-registration. The effect was
below the practical threshold, and the control was confounded.

**68. What confound?**
FullKV — the *lossless* control — collapsed 48% → 0% between n_g=1 and n_g=8. If
the base model fails without compression, compression damage is unmeasurable.

**69. Gate 2 result?**
FAIL. Aggregate and minimax chose identical allocations in all 6 runs; benefit
0.000, CI [0,0].

**70. Why identical?**
Degradation curves were monotone-consistent, so sum- and max-objectives pour the
marginal bit into the same cohort. No cohort tension existed.

**71. Isn't a zero-width CI suspicious?**
It's degenerate precisely *because* the two systems are identical — not because
disparity was measured and found tiny. I state that explicitly.

**72. Gate 3?**
PASS at pilot scale — Gate 1/Gate 2 reproduced in decision category on TinyLlama
(SentencePiece).

**73. Gate 4?**
FAIL. Fuzzy: −0.013 accuracy vs no-repair, −0.050 worst-cohort (it made the worst
cohort *worse*), CI vs best simple [−0.050, 0.037].

**74. Is the fuzzy engine actually fuzzy?**
Yes — triangular membership functions, an explicit 10-rule base, min/max Mamdani
aggregation, centroid defuzzification, per-candidate rule traces. Not a renamed
neural net.

**75. Fuzzy's cost?**
~204x the cheapest competitor (1.3e-4 vs 6.5e-7 s/candidate), 25 parameters vs 3.

**76. So why keep it?**
Honesty: deleting a failed method hides the negative result. It stays
**non-default** and documented as negative evidence.

**77. Isolation subset — what and why?**
Keep only examples FullKV answers correctly, so compression damage is separated
from base-model inequality.

**78. Statistical methods?**
Permutation tests (Gate 1), paired bootstrap CIs (Gates 2/4), hierarchical
bootstrap across families (Gate 3).

**79. Why not just report p-values?**
Practical effect size plus directional consistency was pre-registered. A
significant but tiny effect is not a useful claim.

**80. Are gate results reproducible?**
Gates 1 and 4 recompute exactly from committed raw predictions with **no model
access**. Verified in the release checklist.

## H. Systems & measurement (81–90)

**81. How do you measure latency?**
Warm-up runs, CUDA synchronization, 10 repeats, p50/p95, bootstrap CIs, hardware
manifest.

**82. Why warm-up?**
First-call overhead (imports, allocator warm-up, caches) dominates a single
timing and isn't representative.

**83. Why p95 as well as p50?**
Tail latency is what users feel. Hiding it behind a mean would be marketing.

**84. Do you claim a speedup?**
No. `speedup()` refuses to call a ratio significant when confidence intervals
overlap.

**85. What's the integration boundary?**
Prefill is a real HF forward pass. Decode is an **attention replay harness**, not
a serving engine. Stated everywhere the number appears.

**86. Peak memory caveat?**
CPU readings use tracemalloc, which sees Python-level allocations only — not
torch's C++ allocator. GPU peak (when present) uses torch's own counter and is
authoritative.

**87. Large contexts?**
Chunked encoding bounds peak memory by chunk size, with overhead measured at
1.010x vs single-shot.

**88. OOM behaviour?**
`AllocationBudget` refuses an oversized allocation *before* attempting it,
raising a catchable `AllocationTooLarge`.

**89. Partial output on failure?**
Never returned. A failed chunk discards all prior chunks and raises — a
truncated bitstream is never emitted.

**90. Benchmark contamination?**
The fit cache is content-addressed and **raises** if an artifact fitted on one
split is requested for another.

## I. Honesty, ethics, limitations (91–100)

**91. Why keep "Fair"/"Fuzzy" in the name after two failures?**
The name is owner-chosen identity, not evidence. It's documented as historical
in the README, `PROJECT_IDENTITY.json`, and the claims audit; the claim framing
*does* follow the gate.

**92. Isn't that misleading?**
It would be if the claims followed the name. They don't — they follow the
decision file, and every surfacing location says the fuzzy scorer failed.

**93. Biggest weakness?**
Pilot scale plus the base-model confound. Both are in `PENDING.md` and
`RISK_REGISTER.md`.

**94. Why are some baselines missing?**
RateQuant/RDKV/KVTuner/KVmix are `NOT_REPRODUCED` — no verified access to
reference implementations. Reproducibility over table-filling.

**95. Why not reimplement them from the papers?**
Then the table would carry their names attached to my guesses. That's worse than
an honest gap.

**96. Is the Indic benchmark "parallel"?**
Only in the mechanically verified sense — identical answer, evidence count,
position, distractor count, task family. I do **not** claim translation
equivalence.

**97. Who wrote the Indic text?**
It's LLM-authored from hand-designed templates — not sourced, not professionally
translated, not native-reviewed. Stated in every dataset card.

**98. Contamination checks?**
Only an in-repo self-check was possible. It cannot verify any model's actual
pretraining corpus, and I say so.

**99. What would change your conclusion?**
A benchmark where the lossless control doesn't collapse, plus cohorts whose
degradation curves genuinely diverge. That's Phase 1–3 of the expansion plan.

**100. If you had one more month?**
Naturalistic high-fragmentation renderings and a 7B model — fix the benchmark
before touching the allocator. The allocator isn't the bottleneck; the
experiment design is.

---

## Live-demo recovery plan

| Failure | Recovery |
|---|---|
| Streamlit won't start | `open demo_assets/demo.html` — static, no deps, same frozen numbers |
| No network / model download hangs | Every §2 command in `REPRODUCIBILITY.md` runs offline, including both gate reproductions |
| A page errors | Switch to the offline export; the page-level cause is in `tests/dashboard/` |
| Laptop dies | Figures in `release/figures/` (SVG+PDF) and tables in `FINAL_REPORT.md` carry the whole story |
| Asked for a number I don't recall | `release/CHECKSUMS.sha256` lists every artifact; open the JSON directly rather than guessing |
| Asked to prove a result live | `compute_gate1_from_predictions` / `compute_gate4_from_predictions` — runs in seconds, no model |
| Interactive demo too slow | Use the smaller tokenizer option; or show the pre-rendered byte-accounting table |

**Golden rule under pressure:** if a number isn't on screen from an artifact,
say "that's in `<file>`, let me open it" — never estimate aloud. Estimating from
memory is exactly the failure mode this project's methodology exists to prevent.
