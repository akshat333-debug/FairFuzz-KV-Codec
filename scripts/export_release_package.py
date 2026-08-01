"""Build the release package: checksums, sample bitstreams, vector figures.

Prompt 20 items 137/139. Produces `release/`:
  * `CHECKSUMS.sha256` - sha256 of every committed artifact, so a reproducer can
    verify they received exactly what was released.
  * `sample_bitstreams/` - real FFK1 containers (scalar + LBG) plus a manifest
    recording each one's geometry and byte count, so the format can be exercised
    without running a model.
  * `figures/` - publication-grade VECTOR figures (SVG + PDF) regenerated from
    the frozen artifact JSON. Vector, not raster, so they scale in a paper.

Every figure is drawn from a real artifact file. If an artifact is missing the
figure is skipped with a printed notice - never drawn from placeholder data.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec  # noqa: E402
from fairfuzzkv_codec.codec.vector_quant import LBGVectorQuantCodec  # noqa: E402
from fairfuzzkv_codec.dashboard.artifacts import load_artifact  # noqa: E402
from fairfuzzkv_codec.decoder import decode_from_container, encode_to_container  # noqa: E402

RELEASE = REPO / "release"

CHECKSUM_TARGETS = [
    "gate1_study", "gate2_fairness_study", "gate2_study", "gate3_study",
    "gate4_fairness_study", "quantization_benchmark", "lbg_benchmark",
    "allocation_study", "baseline_matrix_study", "indic_longcomp_study",
    "systems_profile", "repair_scoring_study", "cross_tokenizer_study",
]

DOCS = [
    "README.md", "ARCHITECTURE.md", "FORMAT.md", "CLAIMS_LEDGER.md", "RISK_REGISTER.md",
    "PENDING.md", "SPEC_TRACEABILITY.md", "PERFORMANCE.md", "ALLOCATION_MATH.md",
    "EXECUTION_GATES.md", "GATE2_CONFIG.md", "GATE3_CONFIG.md", "GATE4_CONFIG.md",
    "GATE3_REPORT.md", "GATE4_REPORT.md", "BASELINE_MATRIX_REPORT.md",
    "INDICLONGCOMP_REPORT.md", "DEMO_SCRIPT.md", "pyproject.toml", "uv.lock", "Dockerfile",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_checksums() -> int:
    lines: List[str] = []
    for rel in DOCS:
        p = REPO / rel
        if p.is_file():
            lines.append(f"{sha256_file(p)}  {rel}")
    for d in CHECKSUM_TARGETS:
        base = REPO / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".png", ".ffkv"}:
                lines.append(f"{sha256_file(p)}  {p.relative_to(REPO)}")
    (RELEASE / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def write_sample_bitstreams() -> Dict[str, Any]:
    """Real FFK1 containers a reproducer can decode without any model."""
    out = RELEASE / "sample_bitstreams"
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(20)
    kv = torch.randn(4, 1, 2, 24, 32, generator=g)

    manifest: List[Dict[str, Any]] = []
    for name, codec in (
        ("scalar_int8", ScalarQuantCodec("release", tensor_name="k", default_bits=8)),
        ("scalar_int4", ScalarQuantCodec("release", tensor_name="k", default_bits=4)),
        ("lbg_vd8_cb64", LBGVectorQuantCodec("release", tensor_name="k", vector_dim=8, codebook_size=64)),
    ):
        blob = encode_to_container(codec, kv, tokenizer_hash="release-sample")
        path = out / f"{name}.ffkv"
        path.write_bytes(blob)
        recon, report = decode_from_container(blob)
        manifest.append({
            "file": path.name,
            "codec": name,
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "kv_shape": list(kv.shape),
            "decodes_to_expected_shape": report.shape_ok,
            "reconstruction_mse": float((recon - kv).pow(2).mean().item()),
            "bits_per_element": len(blob) * 8 / kv.numel(),
        })
        print(f"  {name}: {len(blob):,} bytes, mse={manifest[-1]['reconstruction_mse']:.6f}")

    payload = {
        "note": (
            "Real FFK1 v1 containers over a seeded synthetic KV tensor (seed=20). "
            "Synthetic so the sample is reproducible byte-for-byte without a model "
            "download - this is a FORMAT sample, not a model measurement."
        ),
        "format_spec": "FORMAT.md",
        "samples": manifest,
    }
    (out / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _save_vector(fig, name: str) -> None:
    figs = RELEASE / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(figs / f"{name}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {name}.svg / .pdf")


def figure_rate_distortion() -> bool:
    art = load_artifact("quantization_benchmark", REPO)
    if not art.available:
        print(f"  SKIP rate_distortion: {art.status.value}")
        return False
    rows = [r for r in (art.data or {}).get("kv_distortion", []) if r.get("series_name") == "K"]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter([r["bits_per_element"] for r in rows], [r["mse"] for r in rows], s=70, zorder=3)
    for r in rows:
        ax.annotate(r["config"], (r["bits_per_element"], r["mse"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlabel("bits / element (measured, serialized)")
    ax.set_ylabel("K reconstruction MSE (log scale)")
    ax.set_title("Rate-distortion, K cache (Qwen2.5-0.5B, measured)")
    ax.grid(alpha=0.3, zorder=0)
    _save_vector(fig, "rate_distortion_K")
    return True


def figure_gate_outcomes() -> bool:
    from fairfuzzkv_codec.dashboard.artifacts import gate_summary

    rows = gate_summary(REPO)
    colors = {"PASS": "#34a853", "WEAK_PASS": "#fbbc04", "FAIL": "#ea4335"}
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    names = [r["gate"] for r in rows]
    vals = [{"PASS": 3, "WEAK_PASS": 2, "FAIL": 1}.get(r["decision"], 0) for r in rows]
    ax.bar(names, vals, color=[colors.get(r["decision"], "#9aa0a6") for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, vals[i] + 0.06, r["decision"], ha="center", fontsize=9, fontweight="bold")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["FAIL", "WEAK_PASS", "PASS"])
    ax.set_ylim(0, 3.6)
    ax.set_title("Pre-registered gate outcomes (2 of 4 negative)")
    ax.grid(axis="y", alpha=0.3)
    _save_vector(fig, "gate_outcomes")
    return True


def figure_systems_latency() -> bool:
    art = load_artifact("systems_profile", REPO)
    if not art.available:
        print(f"  SKIP systems_latency: {art.status.value}")
        return False
    sweep = (art.data or {}).get("context_sweep", [])
    if not sweep:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    codec_names = [c["codec"] for c in sweep[0]["codecs"]]
    for codec in codec_names:
        xs, ys, y95 = [], [], []
        for ctx in sweep:
            row = next(c for c in ctx["codecs"] if c["codec"] == codec)
            xs.append(ctx["sequence_length"])
            ys.append(row["encode_latency"]["p50_seconds"] * 1000)
            y95.append(row["encode_latency"]["p95_seconds"] * 1000)
        ax.plot(xs, ys, "o-", label=f"{codec} (p50)")
        ax.fill_between(xs, ys, y95, alpha=0.15)
    prefill = [c["prefill_latency"]["p50_seconds"] * 1000 for c in sweep]
    ax.plot([c["sequence_length"] for c in sweep], prefill, "k--", label="model prefill (p50)")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("latency, ms (log scale)")
    ax.set_title("Encode latency vs model prefill - the codec is not the bottleneck")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save_vector(fig, "systems_latency")
    return True


def figure_pareto() -> bool:
    art = load_artifact("minimax", REPO)
    if not art.available:
        print(f"  SKIP pareto: {art.status.value}")
        return False
    frontier = [p for p in (art.data or {}).get("pareto_frontier", []) if p.get("feasible", 0) > 0]
    if not frontier:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    budgets = [p["budget"] for p in frontier]
    ax.plot(budgets, [p["worst_distortion"] for p in frontier], "o-", label="worst cohort (minimax)")
    ax.plot(budgets, [p["average_distortion"] for p in frontier], "s--", label="average")
    ax.set_yscale("log")
    ax.set_xlabel("total serialized bit budget")
    ax.set_ylabel("distortion (log scale)")
    ax.set_title("Pareto frontier: the cost of fairness")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save_vector(fig, "fairness_pareto")
    return True


def main() -> None:
    RELEASE.mkdir(exist_ok=True)
    print("sample bitstreams:")
    samples = write_sample_bitstreams()
    print("figures (vector SVG + PDF):")
    made = sum([figure_rate_distortion(), figure_gate_outcomes(),
                figure_systems_latency(), figure_pareto()])
    print("checksums:")
    n = write_checksums()
    print(f"  {n} files hashed -> release/CHECKSUMS.sha256")

    (RELEASE / "release_manifest.json").write_text(json.dumps({
        "sample_bitstreams": len(samples["samples"]),
        "vector_figures": made,
        "checksummed_files": n,
        "note": "All figures derive from frozen artifact files; none use placeholder data.",
    }, indent=2), encoding="utf-8")
    print(f"\nrelease package -> {RELEASE}")


if __name__ == "__main__":
    main()
