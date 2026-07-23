from typing import Any, List, Tuple

from fairfuzzkv_codec.unicode_grouping.schema import GroupRecord, QuarantineRecord, QuarantineReason
from fairfuzzkv_codec.unicode_grouping.scripts import script_profile_for_text
from fairfuzzkv_codec.unicode_grouping.surface_units import SurfaceUnitDraft


def align_tokens_to_units(
    text: str, units: List[SurfaceUnitDraft], tokenizer: Any
) -> Tuple[List[GroupRecord], List[QuarantineRecord]]:
    """Align tokenizer subtokens to surface units via fast-tokenizer offset
    mappings. Requires a fast tokenizer: offset_mapping is not available on
    slow (pure-Python) tokenizers, and guessing offsets there would violate
    the "never guess silently" rule - unsupported case is documented, not
    faked."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "GroupMapper alignment requires a fast tokenizer (offset_mapping support); "
            f"{type(tokenizer).__name__} is not fast. Slow-tokenizer alignment is not implemented."
        )

    encoding = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    offsets = encoding["offset_mapping"]
    special_mask = encoding.get("special_tokens_mask", [0] * len(offsets))
    input_ids = encoding["input_ids"]

    quarantined: List[QuarantineRecord] = []
    unit_token_indices: List[List[int]] = [[] for _ in units]
    # How many units each token overlapped - used for a deterministic,
    # non-guessing confidence score (an even split of ambiguity, not a pick).
    token_overlap_count: List[int] = [0] * len(offsets)

    def token_text(idx: int) -> str:
        try:
            return tokenizer.convert_ids_to_tokens([input_ids[idx]])[0]
        except Exception:
            return f"<id:{input_ids[idx]}>"

    for tok_idx, (start, end) in enumerate(offsets):
        if special_mask[tok_idx]:
            continue  # special tokens carry no real text span; intentionally excluded, not quarantined

        if end <= start:
            quarantined.append(
                QuarantineRecord(
                    token_index=tok_idx,
                    token_text=token_text(tok_idx),
                    reason=QuarantineReason.INVALID_OFFSET,
                    detail=f"empty or inverted offset ({start}, {end})",
                )
            )
            continue

        if start < 0 or end > len(text):
            quarantined.append(
                QuarantineRecord(
                    token_index=tok_idx,
                    token_text=token_text(tok_idx),
                    reason=QuarantineReason.OUT_OF_BOUNDS,
                    detail=f"offset ({start}, {end}) outside text length {len(text)}",
                )
            )
            continue

        overlapping = [i for i, u in enumerate(units) if not (end <= u.start or start >= u.end)]
        if not overlapping:
            quarantined.append(
                QuarantineRecord(
                    token_index=tok_idx,
                    token_text=token_text(tok_idx),
                    reason=QuarantineReason.NO_OVERLAP,
                    detail=f"offset ({start}, {end}) overlaps no surface unit",
                )
            )
            continue

        token_overlap_count[tok_idx] = len(overlapping)
        # Repair rule for a token spanning >1 surface unit (tokenizer merges
        # don't always respect our boundaries): assign to every unit it
        # genuinely overlaps rather than picking one arbitrarily.
        for i in overlapping:
            unit_token_indices[i].append(tok_idx)

    records: List[GroupRecord] = []
    for i, u in enumerate(units):
        toks = unit_token_indices[i]
        if toks:
            confidence = sum(1.0 / token_overlap_count[t] for t in toks) / len(toks)
        else:
            confidence = 1.0

        byte_start = len(text[: u.start].encode("utf-8"))
        byte_end = len(text[: u.end].encode("utf-8"))

        records.append(
            GroupRecord(
                unit_type=u.unit_type,
                char_span=(u.start, u.end),
                byte_span=(byte_start, byte_end),
                normalized_span=(u.start, u.end),  # overwritten by GroupMapper per normalization_policy
                original_text=u.text,
                script_profile=script_profile_for_text(u.text),
                language_hint=None,  # ponytail: no language-ID model wired in; documented gap, not guessed
                token_indices=toks,
                token_count=len(toks),
                is_special=False,
                alignment_confidence=confidence,
            )
        )

    return records, quarantined
