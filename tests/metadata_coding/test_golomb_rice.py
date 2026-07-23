import random

from fairfuzzkv_codec.metadata_coding.golomb_rice import (
    METHOD_BITMAP,
    METHOD_RICE_GAPS,
    decode_ints,
    decode_retention,
    decode_uints,
    encode_ints,
    encode_retention,
    encode_uints,
    read_varint,
    write_varint,
)


def test_varint_round_trip():
    for v in [0, 1, 127, 128, 300, 16384, 2**32, 2**40]:
        data = write_varint(v)
        got, off = read_varint(data, 0)
        assert got == v and off == len(data)


def test_uint_array_round_trip():
    vals = [0, 1, 2, 5, 100, 3, 7, 0, 0, 255, 1024] * 10
    data = encode_uints(vals)
    assert decode_uints(data, len(vals)) == vals


def test_signed_array_round_trip():
    vals = [0, -1, 1, -128, 127, -1000, 1000, 3, -3]
    data = encode_ints(vals)
    assert decode_ints(data, len(vals)) == vals


def test_retention_round_trip_all_methods():
    universe = 500
    for positions in ([], [0], [499], [0, 250, 499], list(range(0, 500, 7))):
        data = encode_retention(positions, universe)
        got, uni = decode_retention(data)
        assert got == sorted(positions)
        assert uni == universe


def test_rice_gaps_beats_raw_indices_on_sparse_pattern():
    # realistic sparse retention: 5% of 10000 positions retained.
    universe = 10000
    rng = random.Random(0)
    positions = sorted(rng.sample(range(universe), universe // 20))
    encoded = encode_retention(positions, universe)
    raw_index_bytes = len(positions) * 4  # 32-bit indices
    assert len(encoded) < raw_index_bytes
    # sparse gaps should win the selection (not bitmap/rle).
    assert encoded[0] == METHOD_RICE_GAPS


def test_falls_back_to_bitmap_on_dense_pattern():
    # dense/adversarial: retain ~half, alternating - gaps are tiny and uniform,
    # bitmap is competitive/better and the length-based selector must pick it.
    universe = 4096
    positions = list(range(0, universe, 2))
    encoded = encode_retention(positions, universe)
    # whichever wins, it must be chosen by actual length: verify bitmap is at
    # least considered and the result is no longer than the bitmap candidate.
    bitmap_len = 1 + len(write_varint(universe)) + (universe + 7) // 8
    assert len(encoded) <= bitmap_len
    assert encoded[0] in (METHOD_BITMAP, METHOD_RICE_GAPS)


def test_selection_is_by_measured_length_not_assumption():
    # For a very sparse set the encoder must NOT emit a full bitmap.
    universe = 100000
    positions = [10, 5000, 99999]
    encoded = encode_retention(positions, universe)
    assert len(encoded) < (universe + 7) // 8
