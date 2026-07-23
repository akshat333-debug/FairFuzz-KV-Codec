# FairFuzzKV Binary Format v1 (`FFK1`)

A real, versioned, self-describing container. **Not** a Python pickle — every
field is byte-addressable and documented here. Implemented in
`fairfuzzkv_codec.metadata_coding.container`.

All multi-byte integers are **little-endian**, declared explicitly by an
endianness byte so a reader never has to guess.

## Overall layout

```
+--------------------------------------------------------------+
| HEADER                                                       |
|   magic          4  b"FFK1"                                  |
|   version_major  u8 (=1)                                     |
|   version_minor  u8 (=0)                                     |
|   endianness     u8 (0x01 = little-endian)                   |
|   flags          u8 (reserved, 0)                            |
|   geometry_len   u32                                         |
|   geometry_json  geometry_len bytes  (UTF-8 JSON, sorted)    |
|   geometry_crc32 u32  (CRC32 of geometry_json)               |
+--------------------------------------------------------------+
| SECTION DIRECTORY                                            |
|   num_sections   u32                                         |
|   entry[i]:  type:u32  offset:u64  length:u64  crc32:u32     |
|              (offset is absolute from file start)            |
+--------------------------------------------------------------+
| SECTION PAYLOADS (contiguous, referenced by directory)      |
|   ... section 0 bytes ...                                    |
|   ... section 1 bytes ...                                    |
+--------------------------------------------------------------+
| TRAILER                                                      |
|   file_crc32     u32  (CRC32 of every byte before trailer)   |
+--------------------------------------------------------------+
```

### Geometry JSON

Records cache geometry, tokenizer hash, and the codec name so the decoder is
fully self-describing:

```json
{"codec_name":"scalar_quant","format":"FFK1",
 "geometry":{"batch":1,"head_dim":4,"heads":1,"layers":1,"seq":2},
 "tokenizer_hash":"...","tensor_name":"k","logical_bits":...}
```

Keys are emitted sorted with compact separators, so the serialization is
deterministic (basis of the golden-vector test).

### Known section types

| type | name                 | payload                                            |
|------|----------------------|----------------------------------------------------|
| 1    | `SECTION_CODEC_PAYLOAD` | raw bytes from `codec.encode_prefill` (scalar or LBG) |
| 2    | `SECTION_RETENTION`  | Golomb-Rice retention payload (see below)           |

**Forward compatibility:** a v1 reader iterates the directory and *skips* any
section whose `type` it doesn't recognize (the `offset`/`length` say exactly how
far to jump), so files carrying sections added by later versions still parse.

### Integrity & safe rejection

Three independent CRC32s (geometry, each section, whole file) plus bounds checks
on every offset/length and hard limits (`MAX_FILE_SIZE`, `MAX_SECTIONS`,
`MAX_GEOMETRY_LEN`). Any inconsistency raises `CorruptContainerError` — the
reader never crashes on hostile input (fuzz-tested with random bytes and
single-bit flips).

## Retention position coding (`metadata_coding.golomb_rice`)

Retained positions in `[0, universe)` are encoded as whichever of three
candidates is **shortest by measured length** (never assumed), tagged with a
1-byte method id:

```
byte 0: method   (0=RICE_GAPS, 1=BITMAP, 2=RLE)
RICE_GAPS: varint(universe) varint(count) <blockwise-Rice gaps>
BITMAP:    varint(universe) <ceil(universe/8) bitmap bytes, MSB-first>
RLE:       varint(universe) varint(num_runs) <blockwise-Rice run lengths>
```

Gaps are `p[i] - p[i-1] - 1`. Blockwise Rice writes a 5-bit `k` inline per
64-value block, so its side information is part of the counted length. Sparse
patterns pick `RICE_GAPS` (beats raw 32-bit indices); dense/alternating
patterns fall back to `BITMAP`.

Integer metadata arrays (codebook indices, bit-width maps, cohort IDs, repair
metadata) use the same blockwise-Rice coder (`encode_uints`); signed quantities
use zig-zag first (`encode_ints`). LEB128 varints carry lengths and universes.
No opaque third-party compressor is the sole implementation.

## Golden test vectors

Pinned in `tests/metadata_coding/test_golden_vectors.py` so a byte-layout
change can never pass silently.

**Container** — geometry
`{"codec_name":"scalar_quant","format":"FFK1","geometry":{"batch":1,"head_dim":4,"heads":1,"layers":1,"seq":2}}`,
sections `[(1, b"PAYLOAD"), (2, encode_retention([0,3,7], 8))]`:

```
len = 192 bytes
46464b31010001006e0000007b22636f6465635f6e616d65223a227363616c6172
5f7175616e74222c22666f726d6174223a2246464b31222c2267656f6d65747279
223a7b226261746368223a312c22686561645f64696d223a342c226865616473223a
312c226c6179657273223a312c22736571223a327d7de6070f18020000000100
0000b20000000000000007000000000000000f94697d02000000b9000000000000
000300000000000000ff9a52b15041594c4f41440108913db2b4af
```

**Retention** — `encode_retention([0, 3, 7], universe=8)` → `010891`
(method `01`=BITMAP, `varint(8)=08`, bitmap `0x91` = bits 0,3,7 set, MSB-first).
