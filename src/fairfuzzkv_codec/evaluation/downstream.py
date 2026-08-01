"""Downstream task-accuracy metrics: exact match and token-level F1.

The standard SQuAD-style pair, implemented directly (no third-party metric
package) so the normalization rules are inspectable rather than inherited from
an opaque dependency:

  * exact_match - normalized strings identical.
  * token_f1    - harmonic mean of token precision/recall over the normalized
                  token multisets, which credits partial answers.

Normalization is deliberately conservative and Unicode-aware: casefold, strip
articles (English only), drop punctuation, collapse whitespace. It does NOT
transliterate or translate - this project never claims cross-script answer
equivalence it has not verified (see the IndicLongComp parallelism note).

`SPEC_TRACEABILITY.md` lists this as the "Downstream Task Accuracy" metric
family. It was previously blocked on LongBench/PG-19 integration, but the
metric itself has no dataset dependency - it is a pure function of predicted
and gold strings - so it ships here and is usable on any benchmark with
auditable answers (FragKV-MinPairs and IndicLongComp both qualify).
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

# English articles only. Applying an article list to Hindi/Telugu/Tamil text
# would be wrong, so normalization stays language-neutral apart from this.
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Casefold, NFKC-normalize, drop punctuation/articles, collapse spaces."""
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    # strip punctuation by Unicode category (P*) rather than an ASCII-only list,
    # so Devanagari danda etc. are handled too.
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    tokens = [t for t in re.split(r"\s+", text) if t and t not in _ARTICLES]
    return " ".join(tokens)


def answer_tokens(text: str) -> List[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str, gold: str) -> float:
    """1.0 iff the normalized strings are identical, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Token-multiset F1. Two empty answers count as a match (1.0); one empty
    and one not counts as 0.0 - the standard SQuAD convention, stated rather
    than left implicit."""
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_references(prediction: str, golds: Sequence[str], metric: str = "f1") -> float:
    """Score against the BEST of several acceptable gold answers (the standard
    multi-reference convention). Raises on an empty reference list instead of
    silently returning 0.0, which would look like a wrong answer."""
    if not golds:
        raise ValueError("at least one gold reference is required")
    fn = token_f1 if metric == "f1" else exact_match
    return max(fn(prediction, gold) for gold in golds)


@dataclass
class DownstreamScores:
    num_examples: int
    exact_match: float
    token_f1: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "num_examples": float(self.num_examples),
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
        }


def score_dataset(predictions: Sequence[str], golds: Sequence[str]) -> DownstreamScores:
    """Corpus-level EM and F1 (means over examples)."""
    if len(predictions) != len(golds):
        raise ValueError(f"predictions ({len(predictions)}) and golds ({len(golds)}) must align")
    if not predictions:
        return DownstreamScores(0, 0.0, 0.0)
    em = sum(exact_match(p, g) for p, g in zip(predictions, golds)) / len(predictions)
    f1 = sum(token_f1(p, g) for p, g in zip(predictions, golds)) / len(predictions)
    return DownstreamScores(num_examples=len(predictions), exact_match=em, token_f1=f1)
