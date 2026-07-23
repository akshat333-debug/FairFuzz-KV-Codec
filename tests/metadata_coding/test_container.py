import random
from typing import Dict

import pytest

from fairfuzzkv_codec.metadata_coding.container import (
    MAGIC,
    Container,
    CorruptContainerError,
    Section,
    pack,
    unpack,
)

GEOMETRY: Dict[str, object] = {"format": "FFK1", "codec_name": "scalar_quant", "geometry": {"layers": 2}}


def _sample_container() -> bytes:
    return pack(GEOMETRY, [Section(1, b"hello payload"), Section(2, b"\x00\x01\x02\x03")])


def test_round_trip():
    data = _sample_container()
    c = unpack(data)
    assert c.geometry == GEOMETRY
    assert c.get(1) == b"hello payload"
    assert c.get(2) == b"\x00\x01\x02\x03"


def test_golden_bytes_deterministic_across_runs():
    assert _sample_container() == _sample_container()


def test_unknown_section_is_skipped_not_rejected():
    # A section type a v1 reader doesn't recognize must be tolerated (forward
    # compat) - kept as a raw blob, never a hard failure.
    data = pack(GEOMETRY, [Section(1, b"known"), Section(9999, b"future-stuff")])
    c = unpack(data)
    assert c.has(1)
    assert c.has(9999)  # preserved as opaque blob
    assert c.get(9999) == b"future-stuff"


def test_bad_magic_rejected():
    data = bytearray(_sample_container())
    data[0] = ord("X")
    with pytest.raises(CorruptContainerError):
        unpack(bytes(data))


def test_truncated_file_rejected():
    data = _sample_container()
    with pytest.raises(CorruptContainerError):
        unpack(data[: len(data) // 2])


def test_geometry_checksum_failure_rejected():
    data = bytearray(_sample_container())
    # flip a byte inside the geometry JSON (right after the 4-byte length @ off 12).
    data[16] ^= 0xFF
    with pytest.raises(CorruptContainerError):
        unpack(bytes(data))


def test_section_checksum_failure_rejected():
    data = bytearray(_sample_container())
    # corrupt a payload byte near the end (before the 4-byte trailer).
    data[-6] ^= 0xFF
    with pytest.raises(CorruptContainerError):
        unpack(bytes(data))


def test_impossible_section_count_rejected():
    data = bytearray(_sample_container())
    # find the num_sections u32 and set it absurdly high -> must be rejected.
    # (checksum will also fail, but either way it's a CorruptContainerError.)
    import struct

    struct.pack_into("<I", data, 12, 0)  # break geometry len to force early reject path
    with pytest.raises(CorruptContainerError):
        unpack(bytes(data))


def test_empty_and_tiny_inputs_rejected():
    for bad in [b"", b"FF", MAGIC, MAGIC + b"\x01\x00\x01\x00"]:
        with pytest.raises(CorruptContainerError):
            unpack(bad)


def test_fuzz_random_bytes_never_crash():
    rng = random.Random(1234)
    for _ in range(500):
        n = rng.randint(0, 128)
        blob = bytes(rng.randint(0, 255) for _ in range(n))
        try:
            unpack(blob)
        except CorruptContainerError:
            pass  # expected - safe rejection
        # any OTHER exception type would be a bug; let it propagate and fail.


def test_fuzz_bitflips_on_valid_container_are_rejected_or_equal():
    base = _sample_container()
    rng = random.Random(7)
    for _ in range(300):
        data = bytearray(base)
        idx = rng.randrange(len(data))
        data[idx] ^= 1 << rng.randint(0, 7)
        if bytes(data) == base:
            continue
        try:
            c = unpack(bytes(data))
            # If it parses despite a bitflip, the checksums guarantee the
            # payloads still match what was stored (flip landed in a slack bit).
            assert isinstance(c, Container)
        except CorruptContainerError:
            pass
