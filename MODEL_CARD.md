# Model / Codec Card — FairFuzzKV-Codec

This project ships a **codec**, not a trained model. This card follows model-card
conventions so its intended use, inputs, evaluation, and limits are legible in
the same format reviewers expect.

Every statement is labelled: **[measured]**, **[derived]** (follows from an
assumption stated inline), **[subset]** (observed on a specific subset only), or
**[future]**. The full sentence-level audit is in `CLAIMS_AUDIT.md`.

## Overview

| Field | Value |
|---|---|
| Artifact type | KV-cache compression codec + evaluation harness |
| Version | 1.0.0 |
| License | MIT |
| Primary contact | see `CITATION.cff` |
| Format specification | `FORMAT.md` (FFK1 v1, golden vectors included) |

## Intended use

**In scope.** Compressing transformer Key-Value caches produced during prefill,
for memory-constrained serving and for research on compression/fairness
interactions. Research and teaching use.

**Out of scope.**
- Production serving without further integration work — **[measured]** decode-side
  numbers come from an attention replay harness, not a serving engine
  (`PERFORMANCE.md`).
- Any claim that the codec makes inference *faster*. **[measured]** Prefill costs
  115–208 ms while scalar encode costs 0.6–2.4 ms; the codec buys memory, not
  speed.
- Any fairness guarantee. **[measured]** Gate 2 FAILED; no fairness benefit was
  demonstrated.

## Inputs and outputs

- **Input:** KV tensors shaped `[layers, batch, heads, seq, head_dim]`, fp16/fp32.
- **Output:** an FFK1 container — magic bytes, semantic version, endianness,
  geometry + tokenizer hash, section directory, per-section and whole-file CRC32,
  forward-compatible unknown-section skipping.
- **Reconstruction:** approximate (lossy) for quantized payloads; **[measured]**
  bit-exact for the no-op FP16 codec.

## Algorithms

| Component | Method |
|---|---|
| Scalar quantization | INT8/INT4, symmetric & asymmetric, per-tensor/head/channel/groupwise, percentile & MSE-optimal clipping, genuine two-per-byte INT4 nibble packing |
| Vector quantization | Linde-Buzo-Gray with deterministic split perturbation, empty-cluster recovery, mini-batch mode; global/per-layer/per-head codebook scopes |
| Metadata coding | Golomb-Rice over sorted gaps with blockwise-adaptive `k`; bitmap and run-length fallbacks selected by **measured** encoded length |
| Pruning | recency, top-attention-mass, top-k score, group-aware; budget-neutral repair under a local mass condition |
| Allocation | aggregate (exact DP + greedy water-filling) and fairness-constrained minimax (epigraph reference + discrete water-filling) |

## Evaluation data

- **FragKV-MinPairs** — synthetic minimal pairs varying only evidence
  fragmentation `n_g ∈ {1,2,4,8}`. 200 groups, immutable split hash.
- **IndicLongComp** — 4 language conditions (en, hi, hinglish, te_en), 5 task
  families. **[subset]** All content is LLM-authored from hand-designed
  templates — **not** sourced from an external corpus, **not** professionally
  translated or reviewed. Stated in every dataset card.
- **Models** — `Qwen/Qwen2.5-0.5B` (byte-level BPE) and
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (SentencePiece), both Apache-2.0, both
  CPU-runnable.

## Measured results (headline)

| Result | Value | Label |
|---|---|---|
| Scalar INT8 compression vs fp32 | 3.87–3.98x | [measured] |
| LBG (vd8, cb256) compression vs fp32 | 11.68–26.50x | [measured] |
| LBG encode cost vs scalar INT8 | ~40x slower | [measured] |
| Fuzzy scorer latency vs simplest competitor | ~204x | [measured] |
| Golden bitstreams decode identically across runs | yes | [measured] |
| Gate 1 (fragmentation causes compression failure) | **WEAK_PASS** | [measured, subset] |
| Gate 2 (minimax reduces cross-cohort disparity) | **FAIL** | [measured, subset] |
| Gate 3 (cross-model reproduction) | **PASS** | [measured, subset] |
| Gate 4 (fuzzy beats no-repair and simple scorers) | **FAIL** | [measured, subset] |

## Limitations and risks

1. **[measured]** Two of four pre-registered gates returned negative results. The
   fairness thesis this project set out to test is **not supported** at this scale.
2. **[measured]** A base-model confound dominates the high-fragmentation cohorts:
   the lossless FullKV control itself collapses from 48% to 0% accuracy between
   `n_g=1` and `n_g=8`, so compression effects cannot be cleanly separated there.
3. **[subset]** All studies are pilot scale on ≤1.1B-parameter models, CPU-only,
   short contexts (<300 tokens).
4. **[measured]** Fragility cohorts do **not** transfer across tokenizer families
   (agreement 0.23) — do not claim a universal risk threshold.
5. **[measured]** `rare_token_indicator` uses token-id rank as a rarity proxy; no
   corpus-frequency table is used. `language_hint` is never populated.
6. **[derived]** The attention-mass bound `‖O−Ô‖₂ ≤ 2·M·p_E` holds **per head,
   per layer**, under the stated renormalization assumption. It is **not** an
   end-to-end generation guarantee, and the validator reports assumption
   failures rather than manufacturing a pass.

## Ethical and fairness caveats

- The project name contains "Fair" and "Fuzzy" for **historical identity only**.
  **[measured]** Neither a fairness benefit nor fuzzy-scoring superiority was
  demonstrated; the fuzzy scorer is non-default.
- **[measured]** A leakage guard structurally forbids raw language/script labels
  and task outcomes from entering fragility scores or cohort assignment. Script
  grouping is descriptive-dashboard-only.
- **[measured]** IndicLongComp contains no real personal data; names come from a
  fixed synthetic pool and an automated PII scan found no emails or long digit
  runs.
- **[subset]** Because the Indic content is LLM-authored and unreviewed by native
  speakers, it must not be used to make claims about real-world language quality
  or speaker experience.
- **[future]** Any deployment claim about fairness for non-English users would
  require the journal-scale study in `JOURNAL_EXPANSION_PLAN.md`, not this
  pilot evidence.
