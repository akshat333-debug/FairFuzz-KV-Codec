"""Scalar (INT8/INT4) vs LBG vector quantization at matched TOTAL bits.

The rate axis for every codec is effective_bits_per_element = total SERIALIZED
bytes * 8 / num_elements. That counts ALL overhead honestly on both sides -
scalar scales/zero-points AND LBG codebook bytes - so the comparison is fair
and codebook amortization is visible. A deliberately tiny "small-corpus" case
is included to show where LBG's codebook overhead makes it WORSE than scalar;
that case is reported, never hidden (Prompt 7 non-negotiable).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from fairfuzzkv_codec.cache_capture.hf_capture import HFCapture  # noqa: E402
from fairfuzzkv_codec.codec.scalar_quant import ScalarQuantCodec  # noqa: E402
from fairfuzzkv_codec.codec.vector_quant import LBGVectorQuantCodec  # noqa: E402
from fairfuzzkv_codec.core.config import LayerHeadSelection  # noqa: E402
from fairfuzzkv_codec.quantization.scales import Granularity  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# 3 vector dimensions x 2 codebook sizes = 6 LBG configs (acceptance gate).
VECTOR_DIMS = [4, 8, 16]
CODEBOOK_SIZES = [16, 256]


def _eff_bits(meta: Dict[str, Any], numel: int) -> float:
    return meta["accountant_report"]["serialized_bytes"] * 8.0 / numel


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).pow(2).mean().item()


def _eval_scalar(tensor: torch.Tensor, name: str, bits: int, gran: Granularity) -> Dict[str, Any]:
    codec = ScalarQuantCodec("bench", tensor_name="k", granularity=gran, default_bits=bits)
    t0 = time.perf_counter()
    stream, meta = codec.encode_prefill(tensor)
    enc_ms = (time.perf_counter() - t0) * 1000
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), tensor.dtype, "cpu")
    return {
        "codec": name, "family": "scalar",
        "eff_bits_per_element": _eff_bits(meta, tensor.numel()),
        "mse": _mse(tensor, recon), "encode_ms": enc_ms,
    }


def _eval_lbg(tensor: torch.Tensor, vector_dim: int, codebook_size: int) -> Dict[str, Any]:
    codec = LBGVectorQuantCodec("bench", tensor_name="k", vector_dim=vector_dim, codebook_size=codebook_size)
    t0 = time.perf_counter()
    try:
        stream, meta = codec.encode_prefill(tensor)
    except ValueError as e:
        # Too few vectors to even fill the codebook - the extreme small-corpus
        # failure mode. Reported explicitly with infinite effective rate.
        return {
            "codec": f"LBG-vd{vector_dim}-cb{codebook_size}", "family": "lbg",
            "vector_dim": vector_dim, "codebook_size": codebook_size,
            "eff_bits_per_element": float("inf"), "mse": float("nan"),
            "infeasible": str(e),
        }
    enc_ms = (time.perf_counter() - t0) * 1000
    recon = codec.decode(stream, meta, tuple(meta["full_shape"]), tensor.dtype, "cpu")
    diag = codec.last_diagnostics.get("g")
    return {
        "codec": f"LBG-vd{vector_dim}-cb{codebook_size}", "family": "lbg",
        "vector_dim": vector_dim, "codebook_size": codebook_size,
        "eff_bits_per_element": _eff_bits(meta, tensor.numel()),
        "mse": _mse(tensor, recon), "encode_ms": enc_ms,
        "utilization": diag.utilization if diag else None,
        "dead_codewords": diag.dead_codewords if diag else None,
    }


def _run_suite(tensor: torch.Tensor, label: str) -> List[Dict[str, Any]]:
    print(f"\n=== {label}: K shape={tuple(tensor.shape)} numel={tensor.numel()} ===")
    rows: List[Dict[str, Any]] = []
    rows.append(_eval_scalar(tensor, "INT8-per_channel", 8, Granularity.PER_CHANNEL))
    rows.append(_eval_scalar(tensor, "INT4-per_channel", 4, Granularity.PER_CHANNEL))
    for vd in VECTOR_DIMS:
        for cb in CODEBOOK_SIZES:
            rows.append(_eval_lbg(tensor, vd, cb))
    for r in rows:
        if r.get("infeasible"):
            print(f"  {r['codec']:<22} INFEASIBLE: {r['infeasible']}")
            continue
        extra = ""
        if r["family"] == "lbg":
            extra = f" util={r['utilization']:.2f} dead={r['dead_codewords']}"
        print(f"  {r['codec']:<22} eff_bits/elem={r['eff_bits_per_element']:6.2f}  mse={r['mse']:.6f}{extra}")
    return rows


def main() -> None:
    out_dir = Path("lbg_benchmark")
    out_dir.mkdir(exist_ok=True)
    capture = HFCapture(MODEL_NAME, device="cpu", dtype=torch.float32)

    long_text = (
        "FairFuzzKV-Codec is a research project for memory-conscious compression of "
        "Key-Value caches in large language models, aiming to preserve attention "
        "fidelity while reducing serialized storage footprint significantly across "
        "many tokens so that per-vector codebook overhead amortizes well."
    )
    K_long, _V = capture.capture_prefill_kv(long_text, LayerHeadSelection())
    long_rows = _run_suite(K_long, "LONG corpus (codebook amortizes)")

    # Small corpus: only a handful of tokens -> few vectors -> the fixed
    # codebook cost is spread over almost nothing, so LBG effective bits blow up.
    K_small, _ = capture.capture_prefill_kv("Short prompt.", LayerHeadSelection(layers=[0], heads=[0]))
    small_rows = _run_suite(K_small, "SMALL corpus (codebook overhead dominates)")

    # Explicit finding: does any LBG config beat INT4 scalar at <= its bits, and
    # where does the small corpus flip it?
    int4_small = next(r for r in small_rows if r["codec"] == "INT4-per_channel")
    lbg_worse = [
        r["codec"] for r in small_rows
        if r["family"] == "lbg" and r["eff_bits_per_element"] > int4_small["eff_bits_per_element"]
    ]

    result = {
        "long_corpus": long_rows,
        "small_corpus": small_rows,
        "finding_small_corpus_lbg_worse_than_int4": lbg_worse,
    }
    (out_dir / "lbg_vs_scalar.json").write_text(json.dumps(result, indent=2))
    print(f"\nSmall-corpus LBG configs with HIGHER eff bits than INT4-per_channel "
          f"({int4_small['eff_bits_per_element']:.2f}): {lbg_worse}")
    print(f"Saved -> {out_dir/'lbg_vs_scalar.json'}")


if __name__ == "__main__":
    main()
