"""Prompt 16: full baseline matrix, regime-separated, on IndicLongComp's
course subset (real Qwen2.5-0.5B captures). Compares this project's own
scalar/LBG FairFuzzKV codecs against the baseline matrix in
`fairfuzzkv_codec.baselines.registry` at one matched-bit target. Compression/
quantization, prefill-selection, and decode-time-selection results are kept
in SEPARATE tables (Prompt 16 item 108) - never mixed. Numbers are measured,
never invented.
"""

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from fairfuzzkv_codec.baselines.adapter import run_matched_bit_comparison  # noqa: E402
from fairfuzzkv_codec.baselines.registry import (  # noqa: E402
    NOT_REPRODUCED_CARDS, build_decode_time_selection_adapters, build_prefill_selection_adapters,
    build_quantization_adapters,
)
from fairfuzzkv_codec.baselines.schema import EvaluationRegime  # noqa: E402
from fairfuzzkv_codec.benchmarks.indic_longcomp.dataset_card import load_dataset  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TARGET_BITS_PER_ELEMENT = 4.0
TOLERANCE = 0.15
COURSE_SUBSET_DIR = "indic_longcomp_study/course"


@torch.no_grad()
def _capture_kv_and_attention(model, tokenizer, text: str):
    ids = tokenizer(text, return_tensors="pt")
    out = model(**ids, use_cache=True, output_attentions=True)
    pkv = out.past_key_values
    num_layers = len(pkv.layers)
    K = torch.stack([pkv.layers[i].keys for i in range(num_layers)], dim=0)
    V = torch.stack([pkv.layers[i].values for i in range(num_layers)], dim=0)
    attn = torch.stack([out.attentions[i][0] for i in range(num_layers)], dim=0).unsqueeze(1)  # [L,1,H,Sq,Sk]
    return K, V, attn


def _regime_table(results: list, regime: EvaluationRegime) -> dict:
    by_name: dict = {}
    for r in results:
        if r.regime != regime:
            continue
        by_name.setdefault(r.baseline_name, []).append(r)
    table = {}
    for name, rs in by_name.items():
        matched_count = sum(1 for r in rs if r.matched)
        table[name] = {
            "n_variants": len(rs),
            "matched_count": matched_count,
            "mean_actual_bits_per_element": mean(r.actual_bits_per_element for r in rs if r.actual_bits_per_element == r.actual_bits_per_element),
            "mean_kv_mse": mean(r.kv_mse for r in rs if r.kv_mse == r.kv_mse),
            "mean_encode_seconds": mean(r.latency.encode_seconds for r in rs if r.latency is not None),
        }
    return table


def main() -> None:
    out = Path("baseline_matrix_study")
    out.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    model.eval()

    groups = load_dataset(COURSE_SUBSET_DIR)
    print(f"loaded {len(groups)} groups ({sum(len(g.variants) for g in groups)} variants) from {COURSE_SUBSET_DIR}")

    import time

    all_results: list = []
    for gi, group in enumerate(groups):
        for language, variant in group.variants.items():
            t0 = time.perf_counter()
            K, V, attn = _capture_kv_and_attention(model, tokenizer, variant.context_text)
            print(f"    [{group.group_id}/{language.value}] captured K{tuple(K.shape)} attn{tuple(attn.shape)} in {time.perf_counter() - t0:.1f}s", flush=True)

            quant_adapters = build_quantization_adapters("baseline-matrix")
            prefill_adapters = build_prefill_selection_adapters("baseline-matrix", attn)
            decode_adapters = build_decode_time_selection_adapters("baseline-matrix", attn)
            adapters = quant_adapters + prefill_adapters + decode_adapters

            for adapter in adapters:
                t1 = time.perf_counter()
                k_result = run_matched_bit_comparison([adapter], K, TARGET_BITS_PER_ELEMENT, TOLERANCE)[0]
                v_result = run_matched_bit_comparison([adapter], V, TARGET_BITS_PER_ELEMENT, TOLERANCE)[0]
                k_result.kv_mse = (k_result.kv_mse + v_result.kv_mse) / 2.0
                all_results.append(k_result)
                print(f"      {adapter.card.name}: {time.perf_counter() - t1:.1f}s", flush=True)
        print(f"  group {gi + 1}/{len(groups)} ({group.group_id}) done", flush=True)

    tables = {
        "compression_quantization": _regime_table(all_results, EvaluationRegime.COMPRESSION_QUANTIZATION),
        "prefill_selection": _regime_table(all_results, EvaluationRegime.PREFILL_SELECTION),
        "decode_time_selection": _regime_table(all_results, EvaluationRegime.DECODE_TIME_SELECTION),
    }
    for regime_name, table in tables.items():
        print(f"\n[{regime_name}]")
        for name, row in table.items():
            print(f"  {name}: matched={row['matched_count']}/{row['n_variants']} "
                  f"bits={row['mean_actual_bits_per_element']:.2f} mse={row['mean_kv_mse']:.4f}")

    all_cards = (
        [a.card for a in build_quantization_adapters("baseline-matrix")]
        + [a.card for a in build_prefill_selection_adapters("baseline-matrix", torch.zeros(1, 1, 1, 2, 2))]
        + [a.card for a in build_decode_time_selection_adapters("baseline-matrix", torch.zeros(1, 1, 1, 2, 2))]
        + NOT_REPRODUCED_CARDS
    )

    (out / "result_tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    (out / "raw_results.jsonl").write_text(
        "\n".join(json.dumps(r.to_dict()) for r in all_results), encoding="utf-8",
    )
    (out / "baseline_cards.json").write_text(
        json.dumps([c.to_dict() for c in all_cards], indent=2), encoding="utf-8",
    )
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
