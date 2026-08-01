# FairFuzzKV-Codec — Final Experimental Report

**Title note (required by the pre-registered naming rule).** The Gate 4 decision
governs the *claims*, and Gate 4 returned **FAIL**. Accordingly this report makes
**no fuzzy-scoring claim and no fairness claim**. The project name is retained as
owner-chosen identity and is explicitly *not* evidence — see §9 and
`PROJECT_IDENTITY.json`. The surviving, evidence-grounded contribution is
**tokenizer-fragmentation-aware KV-cache compression with exact byte accounting**.

Every statement below is labelled **[measured]**, **[derived]** (follows from an
assumption stated inline), **[subset]** (observed on a stated subset only), or
**[future]**. Sentence-level audit: `CLAIMS_AUDIT.md`.

---

## 1. Problem statement

Serving long-context LLMs is dominated by the Key-Value cache, which grows
linearly in context length and batch size. Compressing it directly increases
serving capacity per unit of memory.

This project asked a second, less-studied question: **tokenizers fragment
languages unequally.** A word that is one token in English may become many
subtokens in Hindi or Telugu. If lossy KV compression damages fragmented
evidence disproportionately, then compression would silently degrade service
quality for non-English users — a fairness problem hidden inside a systems
optimization.

The project therefore had two deliverables: (a) a real, honestly-accounted
codec, and (b) a pre-registered empirical test of the fairness hypothesis.
**[measured]** (a) succeeded; (b) largely did not, and this report says so.

## 2. Syllabus alignment

| Syllabus topic | Where it is realized |
|---|---|
| Scalar quantization | `quantization/`, `codec/scalar_quant.py` — INT8/INT4, symmetric/asymmetric, four granularities, percentile & MSE-optimal clipping |
| **Linde-Buzo-Gray vector quantization** | `quantization/vector_quant.py` — split perturbation, empty-cluster recovery, mini-batch, codebook scopes |
| Rate-distortion theory | `allocation/curves.py` (D(b)=αe^{−βb} with a monotone fallback), rate-distortion benchmarks |
| Entropy / Golomb-Rice coding | `metadata_coding/golomb_rice.py` — blockwise-adaptive Rice, bitmap & RLE fallbacks |
| Constrained optimization / KKT | `ALLOCATION_MATH.md`, `allocation/minimax.py` — epigraph form, active-set water-filling |
| Dynamic programming | `allocation/allocator.py` — exact multiple-choice knapsack solver |
| Fuzzy inference (Mamdani) | `repair_scoring/fuzzy.py` — triangular MFs, 10-rule base, centroid defuzzification |
| Unicode text processing | `unicode_grouping/` — UAX #29 grapheme clusters, surface units |
| Statistical hypothesis testing | permutation tests, paired/hierarchical bootstrap CIs across all four gates |

## 3. Architecture

```
Model ──► Capture ──► [Unicode Grouping + Fragility Estimation]
                                   │
                                   ▼
                            Allocation (aggregate | minimax)
                                   │
                    ┌──────────────┴───────────────┐
                    ▼                              ▼
              Pruning (+repair)            Quantization (scalar | LBG)
                    └──────────────┬───────────────┘
                                   ▼
                        Metadata Coding (Golomb-Rice)
                                   ▼
                        FFK1 container (CRC32, versioned)
                                   ▼
                     Decoder ──► Reconstructed cache
```

Prefill and decode regimes are separated from the first commit
(`codec/base.py`: `encode_prefill` vs `encode_decode_step`). **[measured]**

## 4. Algorithms and mathematical derivations

### 4.1 Minimax allocation — the KKT result

Minimize the worst-cohort distortion at fixed budget, with
`D_l(x_l) = α_l e^{−β_l x_l}`:

```
minimize_{x,t}  t    s.t.  D_l(x_l) ≤ t,  Σ x_l ≤ B,  x_l ≥ 0
```

Stationarity gives `Σ μ_l = 1` and `λ = μ_l β_l D_l(x_l)`. **[derived]** At the
optimum, **the achieved distortion `D_l` is equalized across all active
cohorts** — `D_l(x_l) = t`. Solving on the active set:

```
x_l = (ln α_l − Λ)/β_l ,   Λ = ( Σ_A (ln α_l)/β_l − B ) / ( Σ_A 1/β_l )
```

with an active-set iteration dropping any cohort whose `x_l < 0` (i.e.
`α_l ≤ t`, already below the floor).

> **Correction made during implementation.** A common informal statement is that
> such an optimum "equalizes β". That is **wrong** and was not coded: the `β_l`
> are fixed curve parameters, not decision variables. What is equalized is the
> achieved distortion. **[derived]** Test: `test_p2_minimax_...` and
> `test_continuous_equalizes_distortion_not_beta`.

### 4.2 Local attention-mass bound

For one head, evicting set `E` and renormalizing the kept weights:

```
‖O − Ô‖₂ ≤ 2 · M · p_E ,   M = max_i ‖v_i‖₂ ,  p_E = Σ_{i∈E} a_i
```

**[derived]** — a **local, single-head, single-layer** statement under the
renormalization assumption. It is **not** an end-to-end generation guarantee.
The validator reports assumption failures (e.g. an empty kept set makes
renormalization undefined) instead of manufacturing a pass. **[measured]**

### 4.3 Golomb-Rice retention coding

Retained positions are gap-coded (`g_i = p_i − p_{i−1} − 1`) and Rice-coded with
a 5-bit `k` written inline per 64-value block, so its side information is
counted. The encoder computes **all three** candidate encodings — Rice gaps,
bitmap, run-length — and emits the shortest by **measured** length, never by
assumption. **[measured]** On a realistic 5%-dense pattern over 10,000
positions, Rice gaps beat raw 32-bit indices; on a dense alternating pattern the
selector falls back to bitmap.

## 5. Gate outcomes

All four gates used **thresholds frozen in code and unit-tested against
synthetic fixtures before the real study ran**, so the decision logic could not
be tuned to the result. **[measured]**

| Gate | Question | Decision | Key evidence |
|---|---|---|---|
| **1** | Does fragmentation causally predict extra compression failure? | **WEAK_PASS** | Only TopK50 showed a directional effect (7.5 pts, p=0.0004), below the pre-registered 10-pt bar; the **lossless control itself collapsed 48%→0%** across the same range |
| **2** | Does minimax allocation reduce cross-cohort disparity at matched bits? | **FAIL** | Aggregate and minimax chose **identical allocations in all 6 runs**; worst-cohort benefit 0.000, CI [0,0] |
| **3** | Do the findings reproduce across model/tokenizer families? | **PASS** | Gate 1 and Gate 2 reproduced in decision category on TinyLlama (SentencePiece) |
| **4** | Does fuzzy repair scoring beat no-repair and simpler scorers? | **FAIL** | Accuracy gain −0.013 (25% consistent); worst-cohort gain −0.050 (fuzzy made the worst cohort **worse**); CI vs best simple [−0.050, 0.037] |

**[measured]** Two of four gates are negative. **[measured]** Gate 1 and Gate 4
decisions recompute exactly from committed raw predictions with no model access.

### Interpretation, stated plainly

The fairness hypothesis is **not supported at this scale**. **[measured]** Two
distinct reasons, both documented rather than explained away:

1. **A base-model confound swallows the high-fragmentation cohorts.** The
   lossless FullKV control drops to 0% accuracy at `n_g=8`, so there is no
   base-correct signal left on which compression damage could be measured.
2. **The allocators never diverged.** Because the per-cohort degradation curves
   are monotone-consistent, a sum-objective and a max-objective pour the
   marginal bit into the same cohort. Minimax cannot beat a control it is
   identical to. **[measured]** This held across budgets, seeds, *and* both
   tokenizer families (Gate 3 interactions).

## 6. Scalar vs LBG comparison

**[measured]** At matched *total serialized* bits, including codebook overhead:

| Corpus | Codec | Effective bits/element | K MSE | Encode p50 |
|---|---|---|---|---|
| Long | INT8-per_channel | 8.07 | 0.013 | 0.60–1.97 ms |
| Long | INT4-per_channel | 4.07 | 1.63 | 1.00–2.43 ms |
| Long | LBG vd4 cb256 | **2.24** | **0.73** | — |
| Long | LBG vd8 cb256 | 1.46 | 1.58 | 49.7–83.8 ms |
| Tiny (3 tokens) | INT4-per_channel | 56.50 | 0.18 | — |
| Tiny (3 tokens) | LBG (several configs) | **INFEASIBLE / worse** | — | — |

Three honest findings:
1. **[measured]** LBG dominates on rate: 2.24 bits/element at *lower* MSE than
   INT4 at 4.07.
2. **[measured]** LBG costs ~40x more encode time — codebook training happens at
   encode. Its *decode* is as fast as scalar (table lookup).
3. **[measured]** On a tiny corpus LBG is **worse than scalar or infeasible**,
   because the fixed codebook cost has nothing to amortize over. Reported, not
   hidden.

## 7. Matched-bit evaluation

**[measured]** No comparison in this project is made without matched budgets:
- `MatchedBitEvaluator` **refuses** to return a codec outside tolerance.
- Baseline matrix: automated per-baseline tuning; **40/40 variants matched** for
  every baseline.
- Gate 4: matched bits verified in **all 80 runs**; a violation would have forced
  an automatic FAIL.
- The dashboard **refuses** to present unmatched budgets as like-for-like,
  rendering an explicit `UNMATCHED BUDGETS` error instead.

Byte accounting counts codes, scales, zero-points, masks, indices, headers,
alignment, and metadata. **[measured]** Sparsity ratios are never substituted
for real bytes — a dedicated test enforces this.

## 8. Systems results

**[measured]** Warm-up, synchronization, 10 repeats, p50/p95, bootstrap CIs,
hardware manifest with power mode.

- Prefill: 115–208 ms. Scalar encode: 0.6–2.4 ms. **The codec is ~2 orders of
  magnitude cheaper than the model — it buys memory, not speed.**
- **[measured]** Decode-side figures come from an **attention replay harness**,
  not a serving engine. No end-to-end serving speedup is claimed anywhere.
- **[measured]** `speedup()` refuses to call a ratio significant when the
  confidence intervals overlap.

## 9. Limitations

1. **[measured]** Gates 2 and 4 FAILED; Gate 1 is only WEAK_PASS.
2. **[subset]** All studies are pilot scale, ≤1.1B params, CPU-only, <300-token
   contexts.
3. **[measured]** Fragility cohorts do **not** transfer across tokenizer
   families (agreement 0.23) — no universal risk threshold may be claimed.
4. **[subset]** IndicLongComp content is LLM-authored from templates — **not**
   sourced from an external corpus, **not** professionally translated or
   reviewed. "Parallel" is claimed **only** for mechanically verified properties
   (answer, evidence count/position, distractors, task family); translation
   equivalence is explicitly *not* claimed.
5. **[measured]** RateQuant, RDKV, KVTuner, KVmix are marked `NOT_REPRODUCED`
   with reasons — never silently reimplemented under their names.
6. **[measured]** Heuristic ceilings remain: token-id-rank rarity proxy,
   unpopulated `language_hint`, no decode-regime streaming writer.
7. **[future]** No production serving integration exists.

## 10. Ethical and fairness caveats

- **[measured]** No fairness benefit was demonstrated. Any deployment claiming
  this codec improves equity for non-English users would be unsupported by this
  evidence.
- **[measured]** A leakage guard structurally forbids raw language/script labels
  and task outcomes from entering fragility scores or cohorts; script grouping is
  descriptive-only.
- **[measured]** No real personal data: synthetic name pool, automated PII scan
  clean.
- **[measured]** The name retains "Fair"/"Fuzzy" as historical identity only;
  the fuzzy scorer is **non-default** and documented as negative evidence.
- **[subset]** Because Indic content is unreviewed by native speakers, it must
  not ground claims about real speaker experience.
- **[future]** A defensible fairness claim requires the journal-scale study in
  `JOURNAL_EXPANSION_PLAN.md`.

## 11. Conclusion

**[measured]** The engineering deliverable succeeded: a versioned, checksummed,
fuzz-tested binary format; scalar and LBG quantization with genuine bit packing;
Golomb-Rice metadata coding with measured-length fallback selection; group-aware
pruning with a locally verified bound; aggregate and minimax allocators proven
optimal against brute force; 548 passing tests; every gate reproducible from raw
predictions.

**[measured]** The research hypothesis was tested honestly and largely **not**
supported. Two of four gates failed, and those failures are reported in the
README, the claims ledger, the dashboard, and this report — not buried.

**[future]** The most valuable next step is not a bigger allocator but a better
benchmark: naturalistic high-fragmentation text and a model large enough that
the lossless control does not collapse, so the question can actually be asked.
