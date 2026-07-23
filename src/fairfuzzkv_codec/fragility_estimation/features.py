import unicodedata
from typing import Any, List, Sequence

from fairfuzzkv_codec.fragility_estimation.schema import FeatureVector
from fairfuzzkv_codec.unicode_grouping.schema import GroupRecord
from fairfuzzkv_codec.unicode_grouping.scripts import detect_script

# Leading-space / meta-space markers used by the two supported tokenizer
# families to signal "this piece starts a new word": byte-level BPE uses
# 'Ġ' (GPT2/Qwen2), SentencePiece uses '▁' (Llama/T5).
# ponytail: a piece lacking this marker is treated as a continuation
# subtoken. Known ceiling: the very first token of an entire sequence never
# carries the marker even when it IS word-initial, so it is slightly
# mis-counted as "continuation" - a documented edge case affecting at most
# one token per text, not engineered around here.
_NEW_WORD_MARKERS = ("Ġ", "▁")


def _is_continuation_piece(piece: str) -> bool:
    return not (piece.startswith(_NEW_WORD_MARKERS) or piece.startswith(" "))


def _count_script_transitions(text: str) -> int:
    transitions = 0
    prev = None
    for ch in text:
        s = detect_script(ch)
        if prev is not None and s != prev:
            transitions += 1
        prev = s
    return transitions


def _normalization_sensitivity(text: str) -> float:
    return 1.0 if unicodedata.normalize("NFC", text) != text else 0.0


def _rare_token_indicator(token_ids: Sequence[int], vocab_size: int) -> float:
    """ponytail: no real corpus-frequency table available, so this uses
    token-id rank as a cheap proxy for rarity (higher ids in a BPE/unigram
    vocab tend to be later, less-frequent merges). Documented heuristic, not
    a learned frequency estimate - extend with a real frequency table if
    accuracy matters more than auditability here."""
    if not token_ids or vocab_size <= 0:
        return 0.0
    rare_count = sum(1 for t in token_ids if t / vocab_size > 0.9)
    return rare_count / len(token_ids)


def compute_features(
    record: GroupRecord,
    token_pieces: Sequence[str],
    token_ids: Sequence[int],
    reference_chars_per_token: float,
    vocab_size: int,
) -> FeatureVector:
    """Compute the per-group feature vector for one surface unit's GroupRecord.
    token_pieces/token_ids are the full-sequence tokenizer output; only the
    positions in record.token_indices are used for this unit."""
    unit_pieces = [token_pieces[i] for i in record.token_indices]
    unit_ids = [token_ids[i] for i in record.token_indices]
    num_subtokens = float(record.token_count)

    char_len = record.char_span[1] - record.char_span[0]
    byte_len = record.byte_span[1] - record.byte_span[0]
    chars_per_token = char_len / num_subtokens if num_subtokens > 0 else float(char_len)
    bytes_per_token = byte_len / num_subtokens if num_subtokens > 0 else float(byte_len)

    continuation_ratio = (
        sum(1 for p in unit_pieces if _is_continuation_piece(p)) / len(unit_pieces) if unit_pieces else 0.0
    )

    token_cost_inflation = (
        reference_chars_per_token / chars_per_token if chars_per_token > 0 else 1.0
    )

    return FeatureVector(
        unit_char_span=record.char_span,
        num_subtokens=num_subtokens,
        chars_per_token=chars_per_token,
        bytes_per_token=bytes_per_token,
        continuation_ratio=continuation_ratio,
        script_transitions=float(_count_script_transitions(record.original_text)),
        normalization_sensitivity=_normalization_sensitivity(record.original_text),
        rare_token_indicator=_rare_token_indicator(unit_ids, vocab_size),
        boundary_mismatch=1.0 - record.alignment_confidence,
        token_cost_inflation=token_cost_inflation,
    )


def compute_features_for_records(
    records: List[GroupRecord], tokenizer: Any, text: str
) -> List[FeatureVector]:
    """Convenience batch entry point: re-runs the tokenizer once (deterministic,
    same text+tokenizer as GroupMapper used) to get piece strings, then
    computes one FeatureVector per GroupRecord."""
    from fairfuzzkv_codec.fragility_estimation.reference import get_reference_chars_per_token

    encoding = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    token_ids = encoding["input_ids"]
    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    reference_cpt = get_reference_chars_per_token(tokenizer)
    vocab_size = getattr(tokenizer, "vocab_size", 0)

    return [
        compute_features(r, token_pieces, token_ids, reference_cpt, vocab_size)
        for r in records
    ]
