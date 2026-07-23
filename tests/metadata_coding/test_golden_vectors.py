"""Golden byte vectors - pin the on-disk format so any accidental layout change
fails loudly. See FORMAT.md for the human-readable diagrams these correspond to.
"""

from fairfuzzkv_codec.metadata_coding.container import Section, pack, unpack
from fairfuzzkv_codec.metadata_coding.golomb_rice import decode_retention, encode_retention

GEOMETRY = {
    "codec_name": "scalar_quant",
    "format": "FFK1",
    "geometry": {"batch": 1, "head_dim": 4, "heads": 1, "layers": 1, "seq": 2},
}

GOLDEN_CONTAINER_HEX = (
    "46464b31010001006e0000007b22636f6465635f6e616d65223a227363616c61"
    "725f7175616e74222c22666f726d6174223a2246464b31222c2267656f6d6574"
    "7279223a7b226261746368223a312c22686561645f64696d223a342c22686561"
    "6473223a312c226c6179657273223a312c22736571223a327d7de6070f180200"
    "000001000000b20000000000000007000000000000000f94697d02000000b900"
    "0000000000000300000000000000ff9a52b15041594c4f41440108913db2b4af"
)

GOLDEN_RETENTION_HEX = "010891"


def test_golden_container_bytes_stable():
    ret = encode_retention([0, 3, 7], 8)
    data = pack(GEOMETRY, [Section(1, b"PAYLOAD"), Section(2, ret)])
    assert data.hex() == GOLDEN_CONTAINER_HEX
    # and it decodes identically across runs
    c = unpack(data)
    assert c.geometry == GEOMETRY
    assert c.get(1) == b"PAYLOAD"


def test_golden_retention_bytes_stable():
    ret = encode_retention([0, 3, 7], 8)
    assert ret.hex() == GOLDEN_RETENTION_HEX
    positions, universe = decode_retention(ret)
    assert positions == [0, 3, 7] and universe == 8
