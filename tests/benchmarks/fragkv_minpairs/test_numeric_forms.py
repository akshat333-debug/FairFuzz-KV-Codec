import pytest
from transformers import AutoTokenizer

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.numeric_forms import (
    RENDER_LADDER,
    find_rendering_for_target,
    parse_value,
)

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.mark.parametrize("value", range(10))
def test_every_ladder_rendering_round_trips(value):
    for _ttype, render_fn in RENDER_LADDER:
        rendering = render_fn(value)
        assert parse_value(rendering) == value


@pytest.mark.parametrize("target,tolerance", [(1, 0), (2, 0), (4, 1), (8, 1)])
def test_find_rendering_hits_every_digit_within_tolerance(tokenizer, target, tolerance):
    for value in range(10):
        result = find_rendering_for_target(value, target, tokenizer, tolerance=tolerance)
        assert result is not None, f"digit {value} could not reach n_g={target}"
        rendering, _ttype, realized = result
        assert abs(realized - target) <= tolerance
        assert parse_value(rendering) == value


def test_parse_value_handles_noisy_model_output():
    assert parse_value(" 7") == 7
    assert parse_value(" three<|endoftext|>") == 3
    assert parse_value(" four eight two.") == 4  # first parseable value wins
    assert parse_value("") is None
    assert parse_value("no digits or words here") is None


def test_parse_value_prefers_earliest_match():
    assert parse_value("blah blah seven blah two") == 7
