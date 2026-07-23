import pytest
import torch  # noqa: F401  ensures transformers' torch backend is initialized before tokenizer loads
from transformers import AutoTokenizer

from fairfuzzkv_codec.unicode_grouping.aligner import align_tokens_to_units
from fairfuzzkv_codec.unicode_grouping.surface_units import build_surface_units

# Two tokenizer families with distinct schemes (Prompt 3 acceptance gate):
# byte-level BPE (Qwen2/GPT2-style, 'Ġ' leading-space marker) and
# SentencePiece (Llama-style, '▁' meta-space marker).
BPE_MODEL = "yujiepan/qwen2-tiny-random"
SENTENCEPIECE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

CASES = [
    "Hello world, this is a test.",
    "नमस्ते दुनिया, कैसे हो?",
    "Mujhe ye बहुत पसंद है, kal फिर से try करेंगे.",
    "Visit https://example.com now! Price: 1234.56 😀",
]


@pytest.fixture(scope="module", params=[BPE_MODEL, SENTENCEPIECE_MODEL])
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


@pytest.mark.parametrize("text", CASES)
def test_every_non_special_token_is_placed_or_quarantined(tokenizer, text):
    """Never-guess-silently invariant: every real (non-special) token index
    must end up either attached to at least one surface unit or explicitly
    quarantined - never silently dropped."""
    units = build_surface_units(text)
    records, quarantine = align_tokens_to_units(text, units, tokenizer)

    encoding = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
    special_mask = encoding["special_tokens_mask"]
    num_tokens = len(encoding["input_ids"])

    assigned = set()
    for r in records:
        assigned.update(r.token_indices)
    quarantined_indices = {q.token_index for q in quarantine}

    for tok_idx in range(num_tokens):
        if special_mask[tok_idx]:
            continue
        assert tok_idx in assigned or tok_idx in quarantined_indices, (
            f"token {tok_idx} neither assigned nor quarantined for text={text!r}"
        )


@pytest.mark.parametrize("text", CASES)
def test_alignment_confidence_in_valid_range(tokenizer, text):
    units = build_surface_units(text)
    records, _ = align_tokens_to_units(text, units, tokenizer)
    for r in records:
        assert 0.0 <= r.alignment_confidence <= 1.0


def test_raises_on_slow_tokenizer():
    class FakeSlowTokenizer:
        is_fast = False

    with pytest.raises(ValueError):
        align_tokens_to_units("hello", build_surface_units("hello"), FakeSlowTokenizer())
