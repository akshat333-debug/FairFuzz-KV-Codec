import pytest

from fairfuzzkv_codec.baselines.schema import BaselineCard, EvaluationRegime, ReproductionStatus


def test_not_reproduced_requires_nearest_faithful_configuration():
    with pytest.raises(ValueError):
        BaselineCard(
            name="X", regime=EvaluationRegime.COMPRESSION_QUANTIZATION,
            reproduction_status=ReproductionStatus.NOT_REPRODUCED,
            version_note="", model_support="", context_limit_note="", deviations="", limitations="",
        )


def test_not_reproduced_with_configuration_is_fine():
    card = BaselineCard(
        name="X", regime=EvaluationRegime.COMPRESSION_QUANTIZATION,
        reproduction_status=ReproductionStatus.NOT_REPRODUCED,
        version_note="", model_support="", context_limit_note="", deviations="", limitations="",
        nearest_faithful_configuration="Use Y instead.",
    )
    assert card.nearest_faithful_configuration == "Use Y instead."


def test_faithful_card_does_not_require_configuration():
    card = BaselineCard(
        name="X", regime=EvaluationRegime.PREFILL_SELECTION, reproduction_status=ReproductionStatus.FAITHFUL,
        version_note="", model_support="", context_limit_note="", deviations="", limitations="",
    )
    assert card.nearest_faithful_configuration == ""


def test_card_to_dict_roundtrips_all_fields():
    card = BaselineCard(
        name="X", regime=EvaluationRegime.DECODE_TIME_SELECTION, reproduction_status=ReproductionStatus.APPROXIMATE,
        version_note="v", model_support="m", context_limit_note="c", deviations="d", limitations="l",
    )
    d = card.to_dict()
    assert d["name"] == "X" and d["regime"] == "decode_time_selection" and d["reproduction_status"] == "approximate"
