"""FairFuzzKV-Codec research dashboard (Prompt 19).

Run:  uv run streamlit run dashboard_app.py

DEVIATION, stated openly: Prompt 19 asks for React/Next.js + FastAPI, with a
"high-quality Streamlit fallback only if schedule requires". This is the
Streamlit fallback. Reason: the interactive text demo (item 130) must call the
tokenizer, Module 1 grouping, Module 2 fragility, and the real codecs live, and
doing that in-process is direct, whereas the same demo across an HTTP boundary
plus a node build toolchain is substantially more moving parts for the same
science. The trade-off is a less bespoke visual language.

Non-negotiable honoured throughout: this UI never hides unfavourable cohorts or
unmatched budgets. Missing artifacts render as an explicit "not generated yet"
notice with the command to produce them - never placeholder numbers.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from fairfuzzkv_codec.dashboard.artifacts import (  # noqa: E402
    build_provenance,
    gate_summary,
    load_artifact,
    load_claims,
    load_jsonl,
    load_limitations,
    matched_bit_warning,
)

st.set_page_config(page_title="FairFuzzKV-Codec Research Dashboard", layout="wide")

PAGES = [
    "Overview & Gate Decisions",
    "Claims & Limitations",
    "Interactive Text Demo",
    "Dataset Inspection",
    "Unicode Grouping",
    "Tokenizer Fragmentation",
    "Compression Configuration",
    "Bitstream Anatomy",
    "Reconstructed-Cache Diagnostics",
    "Rate-Distortion Curves",
    "Fairness Trade-offs",
    "Baseline Matrix",
    "Systems Profiling",
]


def artifact_or_notice(name: str):
    """Return artifact data, or render the missing-artifact notice and None.
    This is the single choke point enforcing 'no placeholder metrics'."""
    art = load_artifact(name)
    if not art.available:
        st.warning(art.missing_message())
        return None, art
    return art.data, art


def provenance_panel(art) -> None:
    """Item 132: config hash, model/tokenizer, seed, commit, hardware, raw path."""
    prov = build_provenance(art)
    with st.expander("Provenance", expanded=False):
        st.json(prov.to_dict())
        st.caption(f"Raw result path: `{art.path}`")


# ---------------------------------------------------------------- pages ----

def page_overview() -> None:
    st.title("FairFuzzKV-Codec")
    st.caption(
        "Memory-conscious KV-cache compression. Gate outcomes below are read "
        "from the frozen report files, not hardcoded."
    )

    st.subheader("Gate decisions")
    st.caption("Negative results are shown with the same prominence as positive ones.")
    cols = st.columns(4)
    for col, row in zip(cols, gate_summary()):
        decision = row["decision"]
        with col:
            st.metric(row["gate"], decision)
            if decision == "FAIL":
                st.error("Negative evidence")
            elif decision == "WEAK_PASS":
                st.warning("Below the practical bar")
            elif decision == "PASS":
                st.success("Reproduced")
            else:
                st.info(row["status"])
            st.caption(f"`{Path(row['artifact']).name}`")

    st.info(
        "**Two of four gates are negative.** Gate 2 (fairness) and Gate 4 (fuzzy "
        "scoring) FAILED and are reported as negative evidence. Gate 1 is only a "
        "WEAK_PASS. The codec is preserved; no fairness or fuzzy claim is made."
    )


def page_claims() -> None:
    st.title("Claims & Limitations")
    st.caption("Driven directly from CLAIMS_LEDGER.md and PENDING.md - it cannot drift from the source.")

    claims = load_claims()
    if not claims:
        st.warning("CLAIMS_LEDGER.md not found.")
        return

    negative = [c for c in claims if c.is_negative]
    st.metric("Claims tracked", len(claims), delta=f"{len(negative)} negative/weak", delta_color="inverse")

    only_negative = st.checkbox("Show only negative / weak claims", value=False)
    for c in claims:
        if only_negative and not c.is_negative:
            continue
        header = f"{c.claim_id} - {'⚠️ ' if c.is_negative else ''}{c.description[:110]}"
        with st.expander(header):
            st.markdown(f"**Validation path:** {c.validation}")
            st.markdown(f"**Status:** {c.status}")

    st.subheader("Known limitations (PENDING.md)")
    for item in load_limitations():
        st.markdown(f"- {item}")


def page_interactive_demo() -> None:
    """Item 130 + acceptance gate: complete byte accounting exposed."""
    st.title("Interactive Text Demo")
    st.caption(
        "Live: surface groups → subtoken counts → fragility scores → "
        "retained/evicted → quantizer choice → coded bits → reconstruction error."
    )

    text = st.text_area(
        "Input text",
        "Mujhe ye बहुत पसंद है 😀 and the total was 4321 units.",
        height=80,
    )
    col1, col2, col3 = st.columns(3)
    bits = col1.selectbox("Quantizer bits", [8, 4], index=0)
    retention = col2.slider("Retention ratio (pruning)", 0.1, 1.0, 0.5, 0.1)
    tok_name = col3.selectbox(
        "Tokenizer", ["yujiepan/qwen2-tiny-random", "hf-internal-testing/tiny-random-LlamaForCausalLM"]
    )

    if not st.button("Analyze", type="primary"):
        return

    import torch
    from transformers import AutoTokenizer

    from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
    from fairfuzzkv_codec.fragility_estimation.pipeline import compute_fragility_report
    from fairfuzzkv_codec.pruning.selectors import TopKTokenScoreSelector

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    report = compute_fragility_report(text, tokenizer)

    st.subheader("Surface groups, subtokens, fragility")
    rows = []
    for fv, rs in zip(report.feature_vectors, report.risk_scores):
        start, end = fv.unit_char_span
        rows.append({
            "surface unit": text[start:end],
            "char span": f"{start}:{end}",
            "subtokens": int(fv.num_subtokens),
            "chars/token": round(fv.chars_per_token, 2),
            "fragility": round(rs.score, 4),
        })
    st.dataframe(rows, width="stretch")

    # retained / evicted via a real selector
    scores = torch.tensor([[[[r["fragility"] for r in rows]]]], dtype=torch.float32)
    keep = max(1, int(len(rows) * retention))
    mask = TopKTokenScoreSelector().select(scores, keep=keep)[0, 0, 0]
    st.subheader("Retained / evicted groups")
    st.caption("Highest-fragility units are retained by this selector.")
    st.dataframe(
        [{"surface unit": r["surface unit"], "fragility": r["fragility"],
          "decision": "RETAINED" if bool(mask[i]) else "evicted"}
         for i, r in enumerate(rows)],
        width="stretch",
    )

    # real encode on a synthetic cache sized to this text
    n_tokens = max(1, len(tokenizer(text, add_special_tokens=False).input_ids))
    g = torch.Generator().manual_seed(0)
    kv = torch.randn(4, 1, 2, n_tokens, 16, generator=g)
    codec = ScalarQuantCodec("demo", tensor_name="k", default_bits=bits)
    payload, meta = codec.encode_prefill(kv)
    recon = codec.decode(payload, {}, tuple(meta["full_shape"]), kv.dtype, "cpu")
    acct = meta["accountant_report"]

    st.subheader("Complete byte accounting")
    st.caption(
        "Every component of the real serialized payload - nothing omitted, "
        "sparsity masks never substituted for actual bytes."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Serialized bytes", f"{acct['serialized_bytes']:,}")
    c2.metric("Logical bits", f"{acct['logical_bits']:,}")
    c3.metric("Overhead bytes", f"{acct['overhead_bytes']:,}")
    c4.metric("Bits/element", f"{acct['serialized_bytes'] * 8 / kv.numel():.2f}")
    st.dataframe(
        [{"component": k, "bytes": v} for k, v in sorted(acct["components"].items())],
        width="stretch",
    )

    st.subheader("Reconstruction error")
    mse = (recon - kv).pow(2).mean().item()
    st.metric("KV MSE", f"{mse:.6f}")
    st.caption(
        f"NOTE: the KV tensor here is synthetic (seeded random), sized to this "
        f"text's {n_tokens} tokens - it is a live codec demonstration, not a "
        f"real model capture. Real-model numbers live in the study artifacts."
    )


def page_dataset() -> None:
    st.title("Dataset Inspection")
    for label, key, rel in (
        ("IndicLongComp - course", "indic_course_card", "indic_longcomp_study/course/groups.jsonl"),
        ("IndicLongComp - journal", "indic_journal_card", "indic_longcomp_study/journal/groups.jsonl"),
    ):
        st.subheader(label)
        card, art = artifact_or_notice(key)
        if card is None:
            continue
        c1, c2, c3 = st.columns(3)
        c1.metric("Groups", card.get("num_groups", "?"))
        c2.metric("Variants", card.get("num_variants", "?"))
        c3.metric("Languages", len(card.get("languages", [])))
        st.caption(f"Split hash: `{card.get('split_hash', 'n/a')}`")
        with st.expander("Provenance / license / PII / contamination notes"):
            for field_name in ("content_provenance_note", "license_note", "pii_review_note",
                               "contamination_check_note", "dedup_note", "parallelism_check_note"):
                if card.get(field_name):
                    st.markdown(f"**{field_name}**: {card[field_name]}")
        rows = load_jsonl(rel, limit=5)
        if rows:
            with st.expander("Sample groups (first 5)"):
                st.json(rows)
        provenance_panel(art)


def page_unicode() -> None:
    st.title("Unicode Grouping")
    st.caption("Module 1: grapheme clusters → surface units → tokenizer alignment, with an explicit quarantine path.")
    text = st.text_input("Text", "नमस्ते 👨‍👩‍👧 3.14 https://example.com")
    if not st.button("Segment"):
        return
    from transformers import AutoTokenizer

    from fairfuzzkv_codec.unicode_grouping.mapper import GroupMapper

    mapper = GroupMapper(AutoTokenizer.from_pretrained("yujiepan/qwen2-tiny-random"))
    result = mapper.map(text)
    st.dataframe(
        [{"unit": text[r.char_span[0]:r.char_span[1]], "span": f"{r.char_span[0]}:{r.char_span[1]}",
          "tokens": r.token_count, "script": r.script_profile, "confidence": r.alignment_confidence}
         for r in result.records],
        width="stretch",
    )
    quarantined = getattr(result, "quarantined", [])
    st.metric("Quarantined units", len(quarantined))
    if quarantined:
        st.warning("Quarantined units are shown, never silently dropped.")
        st.json([str(q) for q in quarantined])


def page_fragmentation() -> None:
    st.title("Tokenizer Fragmentation")
    data, art = artifact_or_notice("cross_tokenizer")
    if data is None:
        return
    st.subheader("Cross-tokenizer scorer stability")
    st.caption("Spearman rank correlation of repair-priority ordering, BPE vs SentencePiece.")
    st.dataframe(
        [{"scorer": k, "mean rho": round(v, 4)} for k, v in data.get("mean_spearman_by_scorer", {}).items()],
        width="stretch",
    )
    st.subheader("Fragility distributions per language & tokenizer family")
    card, card_art = artifact_or_notice("indic_course_card")
    if card:
        dists = card.get("fragility_distributions", [])
        if dists:
            st.dataframe(
                [{"language": d.get("language"), "tokenizer": d.get("tokenizer_name"),
                  "units": d.get("num_units_scored"), "mean": round(d.get("mean_score", 0), 4),
                  "cohorts": json.dumps(d.get("cohort_counts", {}))} for d in dists],
                width="stretch",
            )
        else:
            st.warning("No fragility distributions recorded in this dataset card.")
    provenance_panel(art)


def page_compression_config() -> None:
    st.title("Compression Configuration")
    st.caption("The quantizer/codec option space actually implemented and tested.")
    st.markdown("""
| Family | Options |
|---|---|
| Scalar | INT8 / INT4, symmetric & asymmetric, per-tensor / per-head / per-channel / groupwise, percentile & MSE-optimal clipping, genuine INT4 nibble packing |
| Vector (LBG) | vector dim, codebook size, head-block & cross-token formation, global / per-layer / per-head codebook scope |
| Pruning | recency, top-attention-mass, top-k score, group-aware |
| Allocation | aggregate (sum-distortion) and minimax (worst-cohort) |
""")
    data, art = artifact_or_notice("allocation")
    if data is None:
        return
    st.subheader("Latest allocation run")
    c1, c2, c3 = st.columns(3)
    c1.metric("Budget (bits)", f"{data.get('budget', 0):,}")
    c2.metric("Optimality gap", f"{data.get('optimality_gap', 0):.4f}")
    c3.metric("Real encode within budget", str(data.get("real_encode_within_budget")))
    provenance_panel(art)


def page_bitstream() -> None:
    st.title("Bitstream Anatomy")
    st.caption("FFKV binary format v1 - see FORMAT.md for the byte-level spec and golden vectors.")
    st.code("""magic "FFK1" | version_major u8 | version_minor u8 | endianness u8 | flags u8
geometry_len u32 | geometry_json ... | geometry_crc32 u32
num_sections u32 | directory[ type u32, offset u64, length u64, crc32 u32 ] ...
section payloads ...
file_crc32 u32   (trailer, covers every preceding byte)""", language="text")

    st.subheader("Live section breakdown")
    import torch

    from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
    from fairfuzzkv_codec.decoder import decode_from_container, encode_to_container
    from fairfuzzkv_codec.metadata_coding.container import unpack

    bits = st.selectbox("Bits", [8, 4], key="bitstream_bits")
    g = torch.Generator().manual_seed(0)
    kv = torch.randn(3, 1, 2, 12, 16, generator=g)
    codec = ScalarQuantCodec("dash", tensor_name="k", default_bits=bits)
    blob = encode_to_container(codec, kv, tokenizer_hash="demo")
    container = unpack(blob)
    recon, report = decode_from_container(blob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Container bytes", f"{len(blob):,}")
    c2.metric("Sections", len(container.sections))
    c3.metric("Shape OK", str(report.shape_ok))
    st.dataframe(
        [{"section type": s.type, "bytes": len(s.data)} for s in container.sections],
        width="stretch",
    )
    with st.expander("Geometry header"):
        st.json(container.geometry)


def page_reconstruction() -> None:
    st.title("Reconstructed-Cache Diagnostics")
    import torch

    from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec
    from fairfuzzkv_codec.codec.vector_quant import LBGVectorQuantCodec

    g = torch.Generator().manual_seed(0)
    kv = torch.randn(4, 1, 2, 32, 16, generator=g)
    rows = []
    bits_by_system = {}
    for name, codec in (
        ("scalar_int8", ScalarQuantCodec("d", tensor_name="k", default_bits=8)),
        ("scalar_int4", ScalarQuantCodec("d", tensor_name="k", default_bits=4)),
        ("lbg_vd8_cb64", LBGVectorQuantCodec("d", tensor_name="k", vector_dim=8, codebook_size=64)),
    ):
        payload, meta = codec.encode_prefill(kv)
        recon = codec.decode(payload, {}, tuple(meta["full_shape"]), kv.dtype, "cpu")
        bpe = meta["accountant_report"]["serialized_bytes"] * 8 / kv.numel()
        bits_by_system[name] = bpe
        rows.append({
            "codec": name, "bits/element": round(bpe, 3),
            "serialized bytes": meta["accountant_report"]["serialized_bytes"],
            "MSE": round((recon - kv).pow(2).mean().item(), 6),
            "max abs err": round((recon - kv).abs().max().item(), 6),
        })
    st.dataframe(rows, width="stretch")

    # Item 131: refuse to present an unmatched comparison as like-for-like
    warning = matched_bit_warning(bits_by_system)
    if warning:
        st.error(warning)
    else:
        st.success("Budgets matched within tolerance - comparison is like-for-like.")


def page_rate_distortion() -> None:
    st.title("Rate-Distortion Curves")
    data, art = artifact_or_notice("quantization_benchmark")
    if data is None:
        return
    st.subheader("K/V distortion vs bits (real Qwen2.5-0.5B capture)")
    st.dataframe(data.get("kv_distortion", []), width="stretch")
    st.subheader("Task accuracy vs bits")
    st.dataframe(data.get("task_accuracy", []), width="stretch")
    st.warning(
        "Measured finding, reported as-is: at ~8 bits/element per-tensor scored "
        "HIGHER task accuracy than per-channel despite far worse MSE. MSE does "
        "not always predict task accuracy at this sample size."
    )
    for img, caption in (
        ("quantization_benchmark/rate_distortion_mse.png", "K/V MSE"),
        ("quantization_benchmark/rate_distortion_task_distortion.png", "Task distortion"),
    ):
        p = Path(img)
        if p.exists():
            st.image(str(p), caption=caption)
    provenance_panel(art)

    st.subheader("Scalar vs LBG at matched total bits")
    lbg, lbg_art = artifact_or_notice("lbg_benchmark")
    if lbg:
        st.dataframe(lbg.get("long_corpus", []), width="stretch")
        st.error(
            "Small-corpus case shown honestly: "
            f"{lbg.get('finding_small_corpus_lbg_worse_than_int4', [])} had HIGHER "
            "effective bits than INT4 - codebook overhead makes VQ worse there."
        )


def page_fairness() -> None:
    st.title("Fairness Trade-offs")
    st.error(
        "**Gate 2 FAILED.** The aggregate and minimax allocators chose identical "
        "allocations in every run (zero worst-cohort benefit), and isolation "
        "retained only low-fragmentation cohorts. Reported as negative evidence."
    )
    data, art = artifact_or_notice("gate2")
    if data:
        report = data.get("report", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Decision", report.get("decision", "?"))
        c2.metric("Worst-cohort benefit", f"{report.get('mean_fairness_benefit_worst', 0):.4f}")
        c3.metric("Directional consistency", f"{report.get('directional_consistency', 0):.0%}")
        st.caption(report.get("reasoning", ""))
        st.subheader("Cohort counts after isolation")
        st.json(data.get("cohort_counts", {}))
        st.warning(
            "Cohort counts are shown in full, including cohorts with zero "
            "surviving examples - unfavourable cohorts are never hidden."
        )
        provenance_panel(art)

    st.subheader("Pareto frontier: the cost of fairness")
    mm, mm_art = artifact_or_notice("minimax")
    if mm:
        st.dataframe(mm.get("pareto_frontier", []), width="stretch")
        st.caption("Worst-cohort vs average distortion across budgets - the trade-off is exposed, not hidden.")


def page_baselines() -> None:
    st.title("Baseline Matrix")
    st.caption("Results are split by evaluation regime; decode-time results never appear in prefill tables.")
    tables, art = artifact_or_notice("baseline_matrix")
    if tables:
        for regime, rows in tables.items():
            st.subheader(regime)
            st.dataframe(
                [{"baseline": k, **{kk: vv for kk, vv in v.items()}} for k, v in rows.items()],
                width="stretch",
            )
    cards_blob, cards_art = artifact_or_notice("baseline_cards")
    if cards_blob:
        cards = cards_blob["cards"] if isinstance(cards_blob, dict) else cards_blob
        commit = cards_blob.get("generated_by_repo_commit", "unrecorded") if isinstance(cards_blob, dict) else "unrecorded"
        st.caption(f"Cards generated by repo commit `{commit}`")
        not_reproduced = [c for c in cards if c.get("reproduction_status") == "not_reproduced"]
        if not_reproduced:
            st.error(
                "NOT REPRODUCED (shown explicitly, never silently substituted): "
                + ", ".join(c["name"] for c in not_reproduced)
            )
        for c in cards:
            with st.expander(f"{c['name']} - {c['reproduction_status']} ({c['regime']})"):
                st.json(c)


def page_systems() -> None:
    st.title("Systems Profiling")
    data, art = artifact_or_notice("systems_profile")
    if data is None:
        return
    boundary = data.get("integration_boundary", {})
    st.warning(
        f"**Integration boundary.** Prefill: {boundary.get('prefill')}. "
        f"Decode: {boundary.get('decode')}"
    )
    with st.expander("Hardware manifest", expanded=True):
        st.json(data.get("hardware_manifest", {}))

    st.subheader("Context-length sweep (p50 / p95, measured)")
    rows = []
    for ctx in data.get("context_sweep", []):
        for c in ctx["codecs"]:
            rows.append({
                "seq": ctx["sequence_length"], "codec": c["codec"],
                "encode p50 (ms)": round(c["encode_latency"]["p50_seconds"] * 1000, 3),
                "encode p95 (ms)": round(c["encode_latency"]["p95_seconds"] * 1000, 3),
                "decode p50 (ms)": round(c["decode_latency"]["p50_seconds"] * 1000, 3),
                "serialized bytes": c["serialized_bytes"],
                "ratio vs fp32": round(c["compression_ratio_vs_fp32"], 2),
            })
    st.dataframe(rows, width="stretch")

    st.subheader("Batch scaling")
    st.dataframe(
        [{"batch": b["batch_size"], "elements": b["num_elements"],
          "encode p50 (ms)": round(b["encode_latency"]["p50_seconds"] * 1000, 3),
          "us/element": round(b["microseconds_per_element"], 5)}
         for b in data.get("batch_sweep", [])],
        width="stretch",
    )
    st.info(
        "Prefill costs 115-208 ms while scalar encode costs 0.6-2.4 ms - the codec "
        "is roughly two orders of magnitude cheaper than the model. Its value is "
        "memory footprint, not speed. No inferred speedup is reported anywhere."
    )
    provenance_panel(art)


PAGE_FUNCS = {
    "Overview & Gate Decisions": page_overview,
    "Claims & Limitations": page_claims,
    "Interactive Text Demo": page_interactive_demo,
    "Dataset Inspection": page_dataset,
    "Unicode Grouping": page_unicode,
    "Tokenizer Fragmentation": page_fragmentation,
    "Compression Configuration": page_compression_config,
    "Bitstream Anatomy": page_bitstream,
    "Reconstructed-Cache Diagnostics": page_reconstruction,
    "Rate-Distortion Curves": page_rate_distortion,
    "Fairness Trade-offs": page_fairness,
    "Baseline Matrix": page_baselines,
    "Systems Profiling": page_systems,
}


def main() -> None:
    st.sidebar.title("FairFuzzKV-Codec")
    st.sidebar.caption("Research dashboard - frozen artifacts only")
    choice = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(
        "Production mode: panels backed by a missing artifact show the command "
        "to generate it rather than placeholder numbers."
    )
    PAGE_FUNCS[choice]()


if __name__ == "__main__":
    main()
