"""Golomb-Rice coding for retention positions and integer metadata arrays.

All methods are self-contained and well-documented (Rice codes, unary, LEB128
varints, zig-zag for signed values) - no opaque third-party compressor is the
sole implementation, per Prompt 8's non-negotiable. Retention positions are
gap-coded with blockwise-adaptive Rice, and the encoder picks Rice vs bitmap vs
run-length by ACTUAL encoded length, never by assumption.
"""

from typing import List, Sequence, Tuple

from fairfuzzkv_codec.metadata_coding.bitio import BitReader, BitWriter

BLOCK_SIZE = 64
MAX_K = 30

# retention method tags
METHOD_RICE_GAPS = 0
METHOD_BITMAP = 1
METHOD_RLE = 2


# ---- LEB128 varints --------------------------------------------------------

def write_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint is unsigned")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated varint")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, offset
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 63) if value < 0 else (value << 1)


def zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


# ---- Rice primitives -------------------------------------------------------

def _rice_write(w: BitWriter, value: int, k: int) -> None:
    w.write_unary(value >> k)
    if k:
        w.write_bits(value & ((1 << k) - 1), k)


def _rice_read(r: BitReader, k: int) -> int:
    q = r.read_unary()
    rem = r.read_bits(k) if k else 0
    return (q << k) | rem


def _best_k(values: Sequence[int]) -> int:
    """Rice parameter minimizing total bits for this block (exhaustive over a
    small range - cheap and exact)."""
    best_k, best_bits = 0, None
    for k in range(MAX_K + 1):
        bits = sum((v >> k) + 1 + k for v in values)
        if best_bits is None or bits < best_bits:
            best_k, best_bits = k, bits
    return best_k


def encode_uints(values: Sequence[int]) -> bytes:
    """Blockwise-adaptive Rice coding of a non-negative integer array. Per
    block, the chosen k (5 bits) is written INLINE, so its side information is
    part of the counted byte length."""
    if any(v < 0 for v in values):
        raise ValueError("encode_uints requires non-negative values")
    w = BitWriter()
    for start in range(0, len(values), BLOCK_SIZE):
        block = values[start : start + BLOCK_SIZE]
        k = _best_k(block)
        w.write_bits(k, 5)
        for v in block:
            _rice_write(w, v, k)
    return w.getvalue()


def decode_uints(data: bytes, count: int) -> List[int]:
    r = BitReader(data)
    out: List[int] = []
    while len(out) < count:
        k = r.read_bits(5)
        for _ in range(min(BLOCK_SIZE, count - len(out))):
            out.append(_rice_read(r, k))
    return out


def encode_ints(values: Sequence[int]) -> bytes:
    """Signed integer array via zig-zag then Rice - used for scale deltas,
    repair metadata, and any centered quantity."""
    return encode_uints([zigzag_encode(v) for v in values])


def decode_ints(data: bytes, count: int) -> List[int]:
    return [zigzag_decode(v) for v in decode_uints(data, count)]


# ---- retention positions ---------------------------------------------------

def _gaps_from_positions(positions: Sequence[int]) -> List[int]:
    gaps: List[int] = []
    prev = -1
    for p in positions:
        gaps.append(p - prev - 1)
        prev = p
    return gaps


def _positions_from_gaps(gaps: Sequence[int]) -> List[int]:
    positions: List[int] = []
    prev = -1
    for g in gaps:
        prev = prev + g + 1
        positions.append(prev)
    return positions


def _runs_from_positions(positions: Sequence[int], universe: int) -> List[int]:
    """Alternating run lengths of a 0/1 membership bitmap, starting with a run
    of 0s (which may be length 0)."""
    runs: List[int] = []
    cur_val = 0
    cur_len = 0
    pos_set = iter(sorted(positions))
    nxt = next(pos_set, None)
    for i in range(universe):
        val = 1 if (nxt is not None and i == nxt) else 0
        if val == 1:
            nxt = next(pos_set, None)
        if val == cur_val:
            cur_len += 1
        else:
            runs.append(cur_len)
            cur_val = val
            cur_len = 1
    runs.append(cur_len)
    return runs


def encode_retention(positions: Sequence[int], universe: int) -> bytes:
    """Encode a sorted set of retained positions in [0, universe). Tries
    Rice-coded gaps, a raw bitmap, and run-length coding, and returns whichever
    is SHORTEST (measured, not assumed), tagged with a 1-byte method id."""
    positions = sorted(positions)
    if positions and (positions[0] < 0 or positions[-1] >= universe):
        raise ValueError("positions out of [0, universe)")
    count = len(positions)
    prefix = write_varint(universe) + write_varint(count)

    # candidate A: Rice gaps
    gaps_bytes = encode_uints(_gaps_from_positions(positions))
    cand_rice = bytes([METHOD_RICE_GAPS]) + prefix + gaps_bytes

    # candidate B: bitmap
    bitmap = bytearray((universe + 7) // 8)
    for p in positions:
        bitmap[p // 8] |= 1 << (7 - (p % 8))
    cand_bitmap = bytes([METHOD_BITMAP]) + write_varint(universe) + bytes(bitmap)

    # candidate C: run-length (Rice-coded run lengths)
    runs = _runs_from_positions(positions, universe)
    runs_bytes = encode_uints(runs)
    cand_rle = bytes([METHOD_RLE]) + write_varint(universe) + write_varint(len(runs)) + runs_bytes

    return min(cand_rice, cand_bitmap, cand_rle, key=len)


def decode_retention(data: bytes) -> Tuple[List[int], int]:
    """Return (positions, universe)."""
    if not data:
        raise ValueError("empty retention payload")
    method = data[0]
    off = 1
    if method == METHOD_RICE_GAPS:
        universe, off = read_varint(data, off)
        count, off = read_varint(data, off)
        gaps = decode_uints(data[off:], count)
        return _positions_from_gaps(gaps), universe
    if method == METHOD_BITMAP:
        universe, off = read_varint(data, off)
        bitmap = data[off:]
        positions = [
            i for i in range(universe)
            if i // 8 < len(bitmap) and (bitmap[i // 8] >> (7 - (i % 8))) & 1
        ]
        return positions, universe
    if method == METHOD_RLE:
        universe, off = read_varint(data, off)
        nruns, off = read_varint(data, off)
        runs = decode_uints(data[off:], nruns)
        positions = []
        idx = 0
        val = 0
        for run in runs:
            if val == 1:
                positions.extend(range(idx, idx + run))
            idx += run
            val ^= 1
        return positions, universe
    raise ValueError(f"unknown retention method tag: {method}")
