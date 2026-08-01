"""Cross-tokenizer stability of the repair-priority scorers (Prompt 14 item 98).

Item 98 names two fuzzy failure modes: overprotecting expensive low-value
groups (already measured in `GATE4_REPORT.md` from the real study) and
**becoming unstable across tokenizers** - which the Gate 4 pilot did NOT test,
because it ran a single model/tokenizer. This script closes that gap without
needing any model generation: it only needs tokenization + Module 1/2 features.

Method. For each text, surface units are segmented by Module 1 and scored by
Module 2 under TWO tokenizer families (byte-level BPE and SentencePiece).
Surface-unit boundaries are text-derived, so the same units align 1:1 across
tokenizers and can be compared pairwise. Two of the four scorer signals are
genuinely tokenizer-dependent (fragility risk score, completion cost from
subtoken count) and two are not (evidence importance, staleness), so any
ranking instability observed is attributable to the tokenizer.

Metric: Spearman rank correlation between a scorer's priority ranking under
tokenizer A and under tokenizer B. 1.0 = perfectly stable ordering; lower =
the scorer reorders repair candidates when the tokenizer changes. Numbers are
measured, never invented.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from fairfuzzkv_codec.fragility_estimation import compute_fragility_report  # noqa: E402
from fairfuzzkv_codec.repair_scoring.ablation import ScorerConfig, ScorerType, score_candidates  # noqa: E402
from fairfuzzkv_codec.repair_scoring.inputs import (  # noqa: E402
    ScorerInputs,
    fit_input_normalizers,
    normalize_inputs,
)

BPE_MODEL = "yujiepan/qwen2-tiny-random"  # byte-level BPE (Qwen2/GPT2 family)
SENTENCEPIECE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # SentencePiece

TEXTS = [
    "The quarterly report shows revenue of 4321 units across three regions.",
    "Mujhe ye बहुत पसंद है और मैं इसे रोज़ इस्तेमाल करता हूँ 😀",
    "मेरा नाम राहुल है और मैं दिल्ली में रहता हूँ।",
    "నా పేరు రాము, నేను హైదరాబాద్‌లో ఉంటాను and I work in tech.",
    "என் பெயர் குமார், நான் சென்னையில் வசிக்கிறேன் with my family.",
    "Visit https://example.com/docs?id=42 for the API keys and 3.14 constants.",
]


def _spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation = Pearson correlation of ranks. Implemented
    here (a few lines) rather than adding a scipy dependency; ties get average
    ranks, the standard definition."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")

    def _ranks(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    if va == 0 or vb == 0:
        return float("nan")  # degenerate (all-tied) ranking - undefined, not 1.0
    return cov / (va * vb)


def _inputs_for(text: str, tokenizer) -> ScorerInputs:
    """Build candidate signals for one text under one tokenizer. Fragility and
    completion cost vary with the tokenizer; evidence importance and staleness
    are text-derived and do not."""
    report = compute_fragility_report(text, tokenizer)
    fragility = [rs.score for rs in report.risk_scores]
    subtokens = [fv.num_subtokens for fv in report.feature_vectors]
    spans = [fv.unit_char_span for fv in report.feature_vectors]
    n = len(fragility)
    # evidence importance: longer surface units carry more evidence (text-derived)
    evidence = [float(e - s) for (s, e) in spans]
    # staleness: earlier units are staler (text-derived position)
    staleness = [1.0 - (i / max(1, n - 1)) for i in range(n)]
    return ScorerInputs(
        fragility=torch.tensor(fragility, dtype=torch.float32),
        evidence_importance=torch.tensor(evidence, dtype=torch.float32),
        completion_cost=torch.tensor(subtokens, dtype=torch.float32),
        staleness=torch.tensor(staleness, dtype=torch.float32),
    )


def main() -> None:
    out = Path("cross_tokenizer_study")
    out.mkdir(exist_ok=True)

    tok_a = AutoTokenizer.from_pretrained(BPE_MODEL)
    tok_b = AutoTokenizer.from_pretrained(SENTENCEPIECE_MODEL)

    per_text: List[Dict[str, object]] = []
    per_scorer: Dict[str, List[float]] = {t.value: [] for t in ScorerType}

    for text in TEXTS:
        in_a = _inputs_for(text, tok_a)
        in_b = _inputs_for(text, tok_b)
        if in_a.fragility.shape[0] != in_b.fragility.shape[0]:
            # surface units must align 1:1 for a paired comparison; if a
            # tokenizer quarantines differently the text is skipped and SAID SO,
            # never silently truncated to force a comparison.
            per_text.append({"text": text, "skipped": "surface-unit counts differ across tokenizers"})
            continue

        # normalization is fit per tokenizer on that tokenizer's own candidates
        norm_a = normalize_inputs(in_a, fit_input_normalizers(in_a))
        norm_b = normalize_inputs(in_b, fit_input_normalizers(in_b))

        entry: Dict[str, object] = {"text": text, "num_units": int(in_a.fragility.shape[0])}
        for scorer in ScorerType:
            sa = score_candidates(norm_a, ScorerConfig(scorer)).tolist()
            sb = score_candidates(norm_b, ScorerConfig(scorer)).tolist()
            rho = _spearman(sa, sb)
            entry[scorer.value] = rho
            if rho == rho:  # not NaN
                per_scorer[scorer.value].append(rho)
        per_text.append(entry)

    print("Spearman rank correlation of repair-priority ranking, BPE vs SentencePiece")
    print("(1.0 = identical candidate ordering under both tokenizers)\n")
    summary: Dict[str, float] = {}
    for name, values in per_scorer.items():
        if not values:
            print(f"  {name:10s}: no comparable texts")
            continue
        mean_rho = sum(values) / len(values)
        summary[name] = mean_rho
        print(f"  {name:10s}: mean rho={mean_rho:+.3f}  min={min(values):+.3f}  (n={len(values)} texts)")

    if "fuzzy" in summary:
        simple = {k: v for k, v in summary.items() if k != "fuzzy"}
        if simple:
            best_simple = max(simple, key=lambda k: simple[k])
            delta = summary["fuzzy"] - simple[best_simple]
            verdict = (
                "fuzzy is LESS stable across tokenizers than the best simple scorer"
                if delta < -0.05 else
                "fuzzy is not measurably less stable than the best simple scorer"
            )
            print(f"\n  fuzzy {summary['fuzzy']:+.3f} vs best simple ({best_simple}) "
                  f"{simple[best_simple]:+.3f} -> delta {delta:+.3f}: {verdict}")

    (out / "cross_tokenizer_stability.json").write_text(json.dumps({
        "tokenizers": {"bpe": BPE_MODEL, "sentencepiece": SENTENCEPIECE_MODEL},
        "per_text": per_text,
        "mean_spearman_by_scorer": summary,
        "note": (
            "Measured on Module 1/2 surface-unit signals only - no model generation. "
            "Small text sample: a stability probe, not a powered study."
        ),
    }, indent=2))
    print(f"\nsaved -> {out/'cross_tokenizer_stability.json'}")


if __name__ == "__main__":
    main()
