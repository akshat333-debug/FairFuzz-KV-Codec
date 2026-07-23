import torch

from fairfuzzkv_codec.quantization.packing import pack_int4, pack_int8, unpack_int4, unpack_int8


def test_signed_known_vector_round_trips():
    vals = torch.tensor([-8, -3, -1, 0, 1, 3, 7, -5], dtype=torch.int8)
    packed = pack_int4(vals, signed=True)
    assert packed.dtype == torch.uint8
    assert packed.numel() == 4  # 8 values -> 4 bytes, genuinely 2 per byte
    unpacked = unpack_int4(packed, num_elements=8, signed=True)
    assert torch.equal(unpacked, vals)


def test_unsigned_known_vector_round_trips():
    vals = torch.tensor([0, 15, 7, 8, 3], dtype=torch.uint8)
    packed = pack_int4(vals, signed=False)
    assert packed.numel() == 3  # 5 values (odd) -> 3 bytes (one padded nibble)
    unpacked = unpack_int4(packed, num_elements=5, signed=False)
    assert torch.equal(unpacked.to(torch.uint8), vals)


def test_signed_boundary_values():
    vals = torch.tensor([-8, 7], dtype=torch.int8)  # extremes of the 4-bit signed range
    packed = pack_int4(vals, signed=True)
    unpacked = unpack_int4(packed, num_elements=2, signed=True)
    assert torch.equal(unpacked, vals)


def test_unsigned_boundary_values():
    vals = torch.tensor([0, 15], dtype=torch.uint8)  # extremes of the 4-bit unsigned range
    packed = pack_int4(vals, signed=False)
    unpacked = unpack_int4(packed, num_elements=2, signed=False)
    assert torch.equal(unpacked.to(torch.uint8), vals)


def test_byte_count_is_exactly_half_rounded_up():
    for n in range(1, 21):
        vals = torch.zeros(n, dtype=torch.int8)
        packed = pack_int4(vals, signed=True)
        assert packed.numel() == (n + 1) // 2, f"n={n} expected {(n + 1) // 2} bytes, got {packed.numel()}"


def test_all_16_nibble_values_round_trip_unsigned():
    vals = torch.arange(16, dtype=torch.uint8)
    packed = pack_int4(vals, signed=False)
    unpacked = unpack_int4(packed, num_elements=16, signed=False)
    assert torch.equal(unpacked.to(torch.uint8), vals)


def test_all_16_signed_nibble_values_round_trip():
    vals = torch.tensor(list(range(-8, 8)), dtype=torch.int8)
    packed = pack_int4(vals, signed=True)
    unpacked = unpack_int4(packed, num_elements=16, signed=True)
    assert torch.equal(unpacked, vals)


def test_packing_is_deterministic_across_repeated_calls():
    torch.manual_seed(0)
    vals = torch.randint(-8, 8, (101,), dtype=torch.int8)
    packed_a = pack_int4(vals, signed=True)
    packed_b = pack_int4(vals, signed=True)
    assert torch.equal(packed_a, packed_b)


def test_packing_byte_content_is_reproducible_bytes():
    """Cross-platform byte equality: the packed tensor's raw bytes must be
    identical given identical input, independent of any process-local state."""
    vals = torch.tensor([-8, 7, 0, -1], dtype=torch.int8)
    packed_a = pack_int4(vals, signed=True).numpy().tobytes()
    packed_b = pack_int4(vals, signed=True).numpy().tobytes()
    assert packed_a == packed_b
    assert packed_a == bytes([(8 & 0xF) | ((7 & 0xF) << 4), (0 & 0xF) | ((15 & 0xF) << 4)])


def test_pack_int8_preserves_unsigned_values_without_wraparound():
    vals = torch.tensor([0, 128, 200, 255], dtype=torch.uint8)
    packed = pack_int8(vals)
    assert packed.dtype == torch.uint8
    unpacked = unpack_int8(packed)
    assert torch.equal(unpacked, vals)


def test_pack_int8_preserves_signed_values():
    vals = torch.tensor([-128, -1, 0, 127], dtype=torch.int8)
    packed = pack_int8(vals)
    assert packed.dtype == torch.int8
    unpacked = unpack_int8(packed)
    assert torch.equal(unpacked, vals)
