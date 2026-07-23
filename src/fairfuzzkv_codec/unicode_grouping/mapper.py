import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from fairfuzzkv_codec.unicode_grouping.aligner import align_tokens_to_units
from fairfuzzkv_codec.unicode_grouping.schema import (
    MAPPER_SCHEMA_VERSION,
    GroupRecord,
    NormalizationPolicy,
    QuarantineRecord,
)
from fairfuzzkv_codec.unicode_grouping.surface_units import build_surface_units


@dataclass
class MapperResult:
    text: str
    normalization_policy: NormalizationPolicy
    tokenizer_name: str
    records: List[GroupRecord]
    quarantine: List[QuarantineRecord]


class GroupMapper:
    """Module 1: aligns original text, Unicode surface units, tokenizer
    subtokens, and (eventually) KV positions. Infrastructure, not a
    heuristic - every output is a versioned, inspectable GroupRecord or an
    explicit QuarantineRecord; nothing is silently guessed."""

    def __init__(self, tokenizer: Any, normalization_policy: NormalizationPolicy = NormalizationPolicy.PRESERVE_ORIGINAL):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                f"GroupMapper requires a fast tokenizer; {type(tokenizer).__name__} is not fast."
            )
        self.tokenizer = tokenizer
        self.normalization_policy = normalization_policy

    def map(self, text: str) -> MapperResult:
        drafts = build_surface_units(text)
        records, quarantine = align_tokens_to_units(text, drafts, self.tokenizer)
        self._apply_normalization_spans(records)
        tokenizer_name = getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__name__)
        return MapperResult(
            text=text,
            normalization_policy=self.normalization_policy,
            tokenizer_name=tokenizer_name,
            records=records,
            quarantine=quarantine,
        )

    def _apply_normalization_spans(self, records: List[GroupRecord]) -> None:
        """Compute normalized_span as an analysis-only side channel; original_text
        and char_span are never mutated. Each unit is normalized independently
        and spans accumulate in unit order.
        ponytail: doesn't model cross-unit normalization interactions (a
        combining sequence split across two adjacent units would normalize
        differently as one string) - vanishingly rare given units already
        preserve whole grapheme clusters; extend if it ever bites."""
        if self.normalization_policy == NormalizationPolicy.PRESERVE_ORIGINAL:
            for r in records:
                r.normalized_span = r.char_span
            return

        form: Literal["NFC", "NFKC"] = "NFC" if self.normalization_policy == NormalizationPolicy.NFC else "NFKC"
        cursor = 0
        for r in records:
            normalized_unit_text = unicodedata.normalize(form, r.original_text)
            length = len(normalized_unit_text)
            r.normalized_span = (cursor, cursor + length)
            cursor += length

    def audit_report(self, result: MapperResult) -> Dict[str, Any]:
        return {
            "schema_version": MAPPER_SCHEMA_VERSION,
            "tokenizer": result.tokenizer_name,
            "normalization_policy": result.normalization_policy.value,
            "text_length_chars": len(result.text),
            "text_length_bytes": len(result.text.encode("utf-8")),
            "num_units": len(result.records),
            "num_quarantined_tokens": len(result.quarantine),
            "units": [r.model_dump() for r in result.records],
            "quarantine": [q.model_dump() for q in result.quarantine],
        }

    def render_debug(self, result: MapperResult) -> str:
        lines = [f"Original: {result.text!r}", f"Tokenizer: {result.tokenizer_name}", ""]
        for r in result.records:
            lines.append(
                f"[{r.unit_type.value:11s}] chars={r.char_span} bytes={r.byte_span} "
                f"scripts={r.script_profile} tokens={r.token_indices} "
                f"conf={r.alignment_confidence:.2f}  text={r.original_text!r}"
            )
        if result.quarantine:
            lines.append("")
            lines.append("QUARANTINED TOKENS:")
            for q in result.quarantine:
                lines.append(f"  token[{q.token_index}] {q.token_text!r}: {q.reason.value} ({q.detail})")
        return "\n".join(lines)
