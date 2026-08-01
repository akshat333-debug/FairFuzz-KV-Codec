# FairFuzzKV-Codec — 9-Minute Demonstration Script

Covers, in order: **problem → codec → syllabus algorithms → result → failure/pivot logic.**

**Setup (before the room):**

```bash
uv sync
uv run streamlit run dashboard_app.py
```

**Offline fallback** (no live app, no network — run this beforehand and present
the exported page):

```bash
uv run python scripts/export_demo_assets.py
# then open demo_assets/demo.html
```

---

## 0:00–1:00 — The problem

> "Serving a long-context LLM, the KV cache dominates memory — it grows linearly
> with context and with batch size. Compress it and you serve more users per GPU.
>
> But there's a fairness question nobody was checking: tokenizers don't treat all
> languages equally. 'university' is one token in English; the Hindi equivalent
> may shatter into eight. If compression damages fragmented text more, then
> compression silently degrades service for non-English users.
>
> That's the hypothesis this project set out to test — and, as you'll see, it is
> largely **not** what the evidence showed."

*Dashboard: **Overview & Gate Decisions**. Point at the four gate tiles.*

## 1:00–2:30 — The codec (what actually works)

*Dashboard: **Bitstream Anatomy**.*

> "First, the engineering deliverable, which stands independent of the research
> question. This is a real binary format — magic bytes, versioning, endianness,
> a section directory, three CRC32 layers, and forward-compatible section
> skipping. Not a pickle. `FORMAT.md` has byte-level diagrams and golden test
> vectors, and the parser is fuzz-tested: 300 random blobs and 300 bit-flips
> either parse correctly or raise a documented error. It never crashes and never
> returns a truncated stream."

*Switch to **Reconstructed-Cache Diagnostics**.*

> "Here's the byte accounting — and note the red banner. INT8, INT4 and LBG are
> at *different* bits/element, so the dashboard **refuses** to present this as a
> like-for-like comparison. That guard is deliberate: the whole project bans
> comparing codecs at unmatched budgets."

## 2:30–4:00 — Syllabus algorithms

*Dashboard: **Rate-Distortion Curves**.*

> "Two quantizer families. Scalar: INT8/INT4, symmetric and asymmetric, four
> granularities, with genuine nibble packing — INT4 is two values per byte, not
> int8 with wasted bits, and there's a byte-count test enforcing that.
>
> Vector: Linde-Buzo-Gray, the syllabus algorithm. Deterministic training with
> split perturbation and empty-cluster recovery. On a long corpus LBG hits 26x
> compression against fp32 where INT8 hits 4x.
>
> But scroll down —" *(point to the small-corpus banner)* "— on a tiny corpus LBG
> is **worse** than INT4, because the codebook overhead has nothing to amortize
> over. We report that case rather than only showing the flattering one."

*Optional 20s: **Interactive Text Demo** — type a Hinglish sentence, show surface
groups → subtoken counts → fragility scores → retained/evicted → full byte
accounting live.*

## 4:00–6:00 — The results, including the negative ones

*Dashboard: **Fairness Trade-offs**.*

> "Now the science. Four pre-registered gates. Every threshold was committed and
> unit-tested against synthetic fixtures **before** the real study ran, so the
> decision logic could not be tuned to the outcome.
>
> **Gate 1 — WEAK_PASS.** Fragmentation showed only a 7.5-point effect, and the
> lossless control itself collapsed from 48% to 0% accuracy across the same
> range. That's a base-model confound we could not separate from a compression
> effect. So: not a validated causal claim.
>
> **Gate 2 — FAIL.** The fairness allocator. The aggregate and minimax allocators
> chose **identical** allocations in all six runs — zero worst-cohort benefit.
> There was no cohort tension for fairness to exploit.
>
> **Gate 4 — FAIL.** The fuzzy repair scorer didn't beat no-repair; it actually
> made the worst cohort *worse* on average, and cost about 200x the latency of a
> three-parameter weighted sum.
>
> **Gate 3 — PASS.** Reproducibility across two tokenizer families. The findings,
> including the negative ones, reproduce."

*Switch to **Claims & Limitations**, tick "Show only negative / weak claims".*

> "This panel is generated from `CLAIMS_LEDGER.md` — it can't drift from the
> source of truth. Five of twenty-eight claims are negative or weak, and they're
> displayed as prominently as the positive ones."

## 6:00–7:30 — Failure and pivot logic

> "So what happens when your central hypothesis fails?
>
> The naming decision was pre-committed: PASS keeps the fairness framing, FAIL
> drops it. Gate 4 failed, so the fuzzy scorer became **non-default** and the
> claim framing in `PROJECT_IDENTITY.json` switched automatically to negative
> evidence. The scorer is still in the codebase — it's just not claimed to be
> better, because it isn't.
>
> The pivot is that the **codec survives independently of the fairness thesis**.
> Format, quantizers, allocators, pruning, bound validation, baseline matrix —
> all still stand. What we lost was a hypothesis; what we kept is a working,
> honestly-measured system.
>
> And Gate 3 tells us the negatives aren't a fluke of one model: the allocators
> produce identical allocations on TinyLlama too."

## 7:30–8:45 — Systems reality

*Dashboard: **Systems Profiling**.*

> "Measured, not projected: warm-up, synchronization, ten repeats, p50 and p95,
> bootstrap confidence intervals, full hardware manifest.
>
> The honest headline is that **the codec is not the bottleneck — the model is.**
> Prefill costs 115–208 ms; scalar encode costs 0.6–2.4 ms. Two orders of
> magnitude. So this codec buys you *memory*, not speed, and we say so instead of
> quoting a speedup.
>
> Note the boundary banner too: decode numbers come from an attention **replay
> harness**, not a production serving engine. We don't claim an end-to-end
> serving win we didn't measure. And our speedup helper refuses to call a ratio
> significant when the confidence intervals overlap."

## 8:45–9:00 — Close

> "Roughly 490 tests, ruff and mypy clean, every gate reproducible from raw
> predictions without model access. Two of four gates failed, and that's in the
> README, not buried.
>
> The deliverable is a working KV codec with real byte accounting — plus honest
> evidence about what fragmentation-aware fairness compression does and doesn't
> buy you at this scale."

---

## Q&A quick references

| Question | Answer |
|---|---|
| "Why keep 'Fuzzy' in the name if Gate 4 failed?" | Name is owner-chosen identity, not evidence. Documented explicitly in README + `PROJECT_IDENTITY.json`; the scorer is non-default. |
| "Did you tune thresholds after seeing results?" | No. Frozen in `gate*.py`, unit-tested pre-run. Gate 1 reproduces exactly from committed predictions. |
| "Why is RateQuant missing from the baselines?" | Marked `NOT_REPRODUCED` with a reason — no verified access to the reference implementation. Reproducibility over table-filling. |
| "Is the fairness idea dead?" | Not proven dead — proven *unsupported at this scale*. `GATE2_REPORT.md` lists what a properly powered re-test needs. |
| "Biggest weakness?" | Pilot scale throughout, and a base-model confound that swallows the high-fragmentation cohorts. Both are in `PENDING.md` and `RISK_REGISTER.md`. |
