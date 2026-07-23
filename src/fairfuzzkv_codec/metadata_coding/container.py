"""FairFuzzKV binary container format v1 ("FFK1").

A real, versioned, self-describing container - NOT a Python pickle. Layout
(all multi-byte integers little-endian, declared by an explicit endianness byte):

    magic            4 bytes   b"FFK1"
    version_major    u8
    version_minor    u8
    endianness       u8        0x01 = little-endian (only value defined in v1)
    flags            u8        reserved, 0 in v1
    geometry_len     u32
    geometry_json    geometry_len bytes   (model/cache geometry, tokenizer hash,
                                           codec name + config - UTF-8 JSON)
    geometry_crc32   u32       CRC32 of geometry_json (early corruption detect)
    num_sections     u32
    directory        num_sections x (type:u32, offset:u64, length:u64, crc32:u32)
    ... section payload bytes (referenced by absolute offset/length) ...
    file_crc32       u32       CRC32 of every byte before this trailer

Forward compatibility: a reader iterates the directory and SKIPS any section
whose `type` it does not recognize (offset+length tell it exactly how many
bytes to jump), so v1 readers tolerate sections added by later versions.
"""

import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

MAGIC = b"FFK1"
VERSION_MAJOR = 1
VERSION_MINOR = 0
ENDIAN_LE = 0x01

# known section types
SECTION_CODEC_PAYLOAD = 1  # raw bytes from a codec.encode_prefill (scalar or LBG)
SECTION_RETENTION = 2  # Golomb-Rice retention payload (metadata_coding.golomb_rice)

# safety limits - a malformed file must be REJECTED, never allowed to allocate
# unbounded memory or run off the end of the buffer.
MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
MAX_SECTIONS = 4096
MAX_GEOMETRY_LEN = 1 * 1024 * 1024  # 1 MiB of JSON header is already absurd

_DIR_ENTRY = struct.Struct("<IQQI")  # type, offset, length, crc32


class CorruptContainerError(ValueError):
    """Raised for any malformed / corrupt / out-of-bounds container. Callers
    can treat every rejection uniformly; the reader never crashes or segfaults
    on hostile input, it raises this."""


@dataclass
class Section:
    type: int
    data: bytes


@dataclass
class Container:
    geometry: Dict[str, object]
    sections: List[Section] = field(default_factory=list)

    def get(self, section_type: int) -> bytes:
        for s in self.sections:
            if s.type == section_type:
                return s.data
        raise KeyError(f"no section of type {section_type}")

    def has(self, section_type: int) -> bool:
        return any(s.type == section_type for s in self.sections)


def pack(geometry: Dict[str, object], sections: List[Section]) -> bytes:
    """Serialize a container to bytes. Deterministic: identical inputs (with
    sorted-key geometry JSON) always produce identical bytes - the basis for
    the golden-vector test."""
    geometry_json = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(geometry_json) > MAX_GEOMETRY_LEN:
        raise ValueError("geometry JSON exceeds limit")
    if len(sections) > MAX_SECTIONS:
        raise ValueError("too many sections")

    head = bytearray()
    head += MAGIC
    head += bytes([VERSION_MAJOR, VERSION_MINOR, ENDIAN_LE, 0])
    head += struct.pack("<I", len(geometry_json))
    head += geometry_json
    head += struct.pack("<I", zlib.crc32(geometry_json) & 0xFFFFFFFF)
    head += struct.pack("<I", len(sections))

    # Directory offsets are absolute from file start; payload begins right
    # after the directory, and the trailer is appended last.
    dir_size = len(sections) * _DIR_ENTRY.size
    payload_start = len(head) + dir_size

    directory = bytearray()
    payload = bytearray()
    cursor = payload_start
    for s in sections:
        crc = zlib.crc32(s.data) & 0xFFFFFFFF
        directory += _DIR_ENTRY.pack(s.type, cursor, len(s.data), crc)
        payload += s.data
        cursor += len(s.data)

    body = bytes(head) + bytes(directory) + bytes(payload)
    trailer = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    return body + trailer


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CorruptContainerError(msg)


def unpack(data: bytes) -> Container:
    """Parse and fully validate a container. Every field is bounds-checked;
    any inconsistency raises CorruptContainerError rather than crashing."""
    _require(len(data) <= MAX_FILE_SIZE, "file exceeds size limit")
    _require(len(data) >= 4 + 4 + 4 + 4 + 4 + 4, "file too small for header")
    _require(data[:4] == MAGIC, "bad magic bytes")

    ver_major, ver_minor, endian, _flags = data[4], data[5], data[6], data[7]
    _require(ver_major == VERSION_MAJOR, f"unsupported major version {ver_major}")
    _require(endian == ENDIAN_LE, f"unsupported endianness {endian}")
    _ = ver_minor  # minor bumps are forward-compatible

    off = 8
    (geo_len,) = struct.unpack_from("<I", data, off)
    off += 4
    _require(0 <= geo_len <= MAX_GEOMETRY_LEN, "geometry length out of range")
    _require(off + geo_len + 4 <= len(data), "geometry runs past end of file")
    geometry_json = data[off : off + geo_len]
    off += geo_len
    (geo_crc,) = struct.unpack_from("<I", data, off)
    off += 4
    _require(zlib.crc32(geometry_json) & 0xFFFFFFFF == geo_crc, "geometry checksum mismatch")
    try:
        geometry = json.loads(geometry_json.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CorruptContainerError(f"geometry JSON invalid: {e}") from e

    # trailer first, so we validate the whole-file checksum before trusting the
    # directory offsets.
    _require(len(data) >= off + 4 + 4, "file too small for directory + trailer")
    (file_crc,) = struct.unpack_from("<I", data, len(data) - 4)
    _require(zlib.crc32(data[: len(data) - 4]) & 0xFFFFFFFF == file_crc, "file checksum mismatch")

    (num_sections,) = struct.unpack_from("<I", data, off)
    off += 4
    _require(0 <= num_sections <= MAX_SECTIONS, "section count out of range")
    _require(off + num_sections * _DIR_ENTRY.size + 4 <= len(data), "directory runs past end of file")

    entries: List[Tuple[int, int, int, int]] = []
    for _ in range(num_sections):
        stype, soff, slen, scrc = _DIR_ENTRY.unpack_from(data, off)
        off += _DIR_ENTRY.size
        entries.append((stype, soff, slen, scrc))

    body_end = len(data) - 4  # everything before trailer
    sections: List[Section] = []
    for stype, soff, slen, scrc in entries:
        _require(soff >= 0 and slen >= 0, "negative section geometry")
        _require(soff + slen <= body_end, "section runs past end of file")
        _require(soff >= off, "section overlaps directory/header")
        blob = data[soff : soff + slen]
        _require(zlib.crc32(blob) & 0xFFFFFFFF == scrc, f"section {stype} checksum mismatch")
        sections.append(Section(stype, blob))

    return Container(geometry=geometry, sections=sections)
