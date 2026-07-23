from fairfuzzkv_codec.quantization.bitwidth_map import BitWidthMap


def test_default_bits_used_when_no_override():
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=4)
    assert bwm.bits_for_layer("k", 5) == 8
    assert bwm.bits_for_layer("v", 5) == 4


def test_layer_override_takes_effect():
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=8)
    bwm.set_layer_bits("k", 3, 4)
    assert bwm.bits_for_layer("k", 3) == 4
    assert bwm.bits_for_layer("k", 4) == 8  # unaffected layer keeps default


def test_head_override_takes_precedence_over_layer_override():
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=8)
    bwm.set_layer_bits("k", 2, 4)
    bwm.set_head_bits("k", 2, 5, 2)
    assert bwm.bits_for("k", 2, 5) == 2  # head override wins
    assert bwm.bits_for("k", 2, 0) == 4  # falls back to layer override


def test_k_and_v_overrides_are_independent():
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=8)
    bwm.set_layer_bits("k", 1, 4)
    assert bwm.bits_for_layer("k", 1) == 4
    assert bwm.bits_for_layer("v", 1) == 8


def test_mixed_precision_map_is_compact_when_serialized():
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=8)
    bwm.set_layer_bits("k", 10, 4)
    dumped = bwm.model_dump()
    # only the one override should be present, not one entry per layer
    assert dumped["k_overrides"] == {"10": 4}
    assert dumped["v_overrides"] == {}


def test_json_round_trip():
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=4)
    bwm.set_layer_bits("k", 2, 4)
    bwm.set_head_bits("v", 5, 3, 2)
    reloaded = BitWidthMap.model_validate_json(bwm.model_dump_json())
    assert reloaded.bits_for_layer("k", 2) == 4
    assert reloaded.bits_for("v", 5, 3) == 2
    assert reloaded.bits_for_layer("v", 0) == 4


def test_invalid_tensor_name_rejected():
    bwm = BitWidthMap(default_k_bits=8, default_v_bits=8)
    try:
        bwm.bits_for_layer("q", 0)
        assert False, "expected ValueError for invalid tensor_name"
    except ValueError:
        pass
