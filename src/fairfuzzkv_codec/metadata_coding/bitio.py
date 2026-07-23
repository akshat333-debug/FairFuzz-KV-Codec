"""Minimal MSB-first bit writer/reader.

Used by the Golomb-Rice and unary coders. MSB-first (big-endian bit order
within each byte) so the byte stream is stable and easy to diagram in FORMAT.md.
"""

from typing import List


class BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._cur = 0
        self._nbits = 0  # bits currently filled in _cur (0..7)

    def write_bit(self, bit: int) -> None:
        self._cur = (self._cur << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._bytes.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def write_bits(self, value: int, count: int) -> None:
        """Write the low `count` bits of `value`, MSB first."""
        for i in range(count - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def write_unary(self, q: int) -> None:
        """`q` zero bits followed by a terminating one bit."""
        for _ in range(q):
            self.write_bit(0)
        self.write_bit(1)

    def getvalue(self) -> bytes:
        """Flush: pad the final partial byte with zero bits on the right."""
        if self._nbits == 0:
            return bytes(self._bytes)
        pad = self._cur << (8 - self._nbits)
        return bytes(self._bytes) + bytes([pad])


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0  # absolute bit position

    @property
    def total_bits(self) -> int:
        return len(self._data) * 8

    def read_bit(self) -> int:
        if self._pos >= self.total_bits:
            raise EOFError("bit stream exhausted")
        byte = self._data[self._pos // 8]
        offset = 7 - (self._pos % 8)
        self._pos += 1
        return (byte >> offset) & 1

    def read_bits(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | self.read_bit()
        return value

    def read_unary(self) -> int:
        q = 0
        while self.read_bit() == 0:
            q += 1
            if q > self.total_bits:
                raise EOFError("unrun exceeded stream length")
        return q


def bits_to_bytes(bits: List[int]) -> bytes:
    w = BitWriter()
    for b in bits:
        w.write_bit(b)
    return w.getvalue()
