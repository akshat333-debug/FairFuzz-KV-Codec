"""Export a self-contained offline snapshot for presenting without the live app.

Prompt 19 item 134's "fallback recorded assets": if Streamlit, the network, or
the machine misbehaves during a presentation, `demo_assets/demo.html` renders
the same frozen numbers as a single static page with no dependencies.

Same honesty rules as the dashboard: a missing artifact is exported as an
explicit "not generated" notice, never as placeholder data.
"""

import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairfuzzkv_codec.dashboard.artifacts import (  # noqa: E402
    gate_summary,
    load_artifact,
    load_claims,
    load_limitations,
)

OUT = Path("demo_assets")

FIGURES = [
    "quantization_benchmark/rate_distortion_mse.png",
    "quantization_benchmark/rate_distortion_task_distortion.png",
    "gate2_study/pareto_frontier.png",
    "allocation_study/budget_allocation.png",
    "repair_scoring_study/fuzzy_membership.png",
]

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
max-width:1100px;margin:2rem auto;padding:0 1.5rem;line-height:1.55;color:#1a1a1a}
h1{border-bottom:3px solid #222;padding-bottom:.4rem}
h2{margin-top:2.5rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
th,td{border:1px solid #ddd;padding:.5rem .6rem;text-align:left}
th{background:#f5f5f5}
.gate{display:inline-block;padding:.9rem 1.2rem;margin:.4rem .6rem .4rem 0;
border-radius:8px;font-weight:600;min-width:150px}
.PASS{background:#e6f4ea;border:2px solid #34a853}
.FAIL{background:#fce8e6;border:2px solid #ea4335}
.WEAK_PASS{background:#fef7e0;border:2px solid #fbbc04}
.unknown{background:#f1f3f4;border:2px solid #9aa0a6}
.warn{background:#fce8e6;border-left:4px solid #ea4335;padding:.9rem;margin:1rem 0}
.note{background:#e8f0fe;border-left:4px solid #1a73e8;padding:.9rem;margin:1rem 0}
.missing{background:#f1f3f4;border-left:4px solid #9aa0a6;padding:.9rem;margin:1rem 0;
font-family:ui-monospace,monospace;font-size:.87rem}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:1rem 0}
code{background:#f1f3f4;padding:.12rem .35rem;border-radius:3px}
"""


def esc(x: Any) -> str:
    return html.escape(str(x))


def table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "<p><em>No rows.</em></p>"
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in columns) + "</tr>"
        for r in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def missing_block(art) -> str:
    return f'<div class="missing">{esc(art.missing_message())}</div>'


def main() -> None:
    OUT.mkdir(exist_ok=True)
    parts: List[str] = ["<h1>FairFuzzKV-Codec — Offline Demo Snapshot</h1>"]
    parts.append(
        "<p>Static export of the frozen study artifacts, for presenting without "
        "the live dashboard. Every number here comes from a committed artifact "
        "file; missing artifacts are shown as such, never as placeholder data.</p>"
    )

    # ---- gates ----
    parts.append("<h2>Gate decisions</h2>")
    gates = gate_summary()
    chips = "".join(
        f'<div class="gate {esc(g["decision"]) if g["decision"] in ("PASS","FAIL","WEAK_PASS") else "unknown"}">'
        f'{esc(g["gate"])}<br>{esc(g["decision"])}</div>'
        for g in gates
    )
    parts.append(chips)
    n_fail = sum(1 for g in gates if g["decision"] == "FAIL")
    parts.append(
        f'<div class="warn"><strong>{n_fail} of {len(gates)} gates FAILED.</strong> '
        "Gate 2 (fairness) and Gate 4 (fuzzy scoring) are reported as negative "
        "evidence. Gate 1 is a WEAK_PASS only. The codec is preserved; no fairness "
        "or fuzzy superiority claim is made.</div>"
    )

    # ---- systems ----
    parts.append("<h2>Systems profile (measured)</h2>")
    art = load_artifact("systems_profile")
    if not art.available:
        parts.append(missing_block(art))
    else:
        data = art.data or {}
        b = data.get("integration_boundary", {})
        parts.append(
            f'<div class="note"><strong>Integration boundary.</strong> '
            f'Prefill: {esc(b.get("prefill"))}<br>Decode: {esc(b.get("decode"))}</div>'
        )
        hw = data.get("hardware_manifest", {})
        parts.append(
            f"<p><strong>Hardware:</strong> {esc(hw.get('platform'))} — "
            f"{esc(hw.get('torch_num_threads'))} torch threads, power mode "
            f"<code>{esc(hw.get('power_mode'))}</code></p>"
        )
        rows = []
        for ctx in data.get("context_sweep", []):
            for c in ctx["codecs"]:
                rows.append({
                    "seq": ctx["sequence_length"],
                    "codec": c["codec"],
                    "encode p50 (ms)": round(c["encode_latency"]["p50_seconds"] * 1000, 2),
                    "encode p95 (ms)": round(c["encode_latency"]["p95_seconds"] * 1000, 2),
                    "decode p50 (ms)": round(c["decode_latency"]["p50_seconds"] * 1000, 2),
                    "serialized bytes": c["serialized_bytes"],
                    "ratio vs fp32": round(c["compression_ratio_vs_fp32"], 2),
                })
        parts.append(table(rows, ["seq", "codec", "encode p50 (ms)", "encode p95 (ms)",
                                  "decode p50 (ms)", "serialized bytes", "ratio vs fp32"]))
        parts.append(
            '<div class="note">The codec is <strong>not</strong> the bottleneck: prefill '
            "costs 115–208 ms versus 0.6–2.4 ms to encode. This codec buys memory, "
            "not speed. No inferred speedup is reported anywhere.</div>"
        )

    # ---- claims ----
    parts.append("<h2>Claims ledger</h2>")
    claims = load_claims()
    negative = [c for c in claims if c.is_negative]
    parts.append(f"<p>{len(claims)} claims tracked, <strong>{len(negative)} negative or weak</strong>.</p>")
    parts.append(table(
        [{"id": c.claim_id, "claim": c.description[:150], "status": c.status[:200]} for c in negative],
        ["id", "claim", "status"],
    ))

    # ---- limitations ----
    parts.append("<h2>Known limitations (PENDING.md)</h2>")
    parts.append("<ul>" + "".join(f"<li>{esc(x)}</li>" for x in load_limitations()) + "</ul>")

    # ---- figures ----
    parts.append("<h2>Figures</h2>")
    copied = 0
    for rel in FIGURES:
        src = Path(rel)
        if not src.exists():
            parts.append(f'<div class="missing">Figure not generated: <code>{esc(rel)}</code></div>')
            continue
        dest = OUT / src.name
        shutil.copy2(src, dest)
        copied += 1
        parts.append(f"<h3>{esc(src.stem)}</h3><img src='{esc(src.name)}' alt='{esc(src.stem)}'>")

    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>FairFuzzKV-Codec — Offline Demo</title>"
        f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>"
    )
    (OUT / "demo.html").write_text(page, encoding="utf-8")

    # machine-readable companion
    (OUT / "demo_data.json").write_text(json.dumps({
        "gates": gates,
        "num_claims": len(claims),
        "num_negative_claims": len(negative),
        "limitations": load_limitations(),
    }, indent=2), encoding="utf-8")

    print(f"exported -> {OUT/'demo.html'} ({copied}/{len(FIGURES)} figures copied)")
    print(f"gates: {', '.join(g['gate'] + '=' + g['decision'] for g in gates)}")


if __name__ == "__main__":
    main()
