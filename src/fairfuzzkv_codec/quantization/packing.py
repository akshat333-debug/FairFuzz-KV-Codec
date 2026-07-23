import torch

# ponytail: numpy/torch have no native int4 dtype, so a "packed" int4 tensor
# is represented as a 1D uint8 tensor with two nibbles per byte
# (low-nibble-first). Shape and original element count are carried in
# metadata by the caller (codec/binary_serializer.py's existing scheme) -
# this module only does the bit-exact pack/unpack.


def pack_int4(values: torch.Tensor, signed: bool) -> torch.Tensor:
    """Pack a tensor of 4-bit-range integer values two-per-byte.
    values: any-shape integer tensor. If signed, values must be in [-8, 7]
    (symmetric quant range); if unsigned, in [0, 15] (asymmetric range).
    Returns a 1D uint8 tensor of ceil(numel/2) bytes - genuinely half the
    footprint of one-value-per-byte storage, never inflated with int8
    padding per value."""
    flat = values.reshape(-1).to(torch.int64)
    nibbles = (flat & 0xF).to(torch.uint8)  # two's-complement bit pattern for negatives

    if nibbles.numel() % 2 == 1:
        nibbles = torch.cat([nibbles, torch.zeros(1, dtype=torch.uint8)])

    lo = nibbles[0::2]
    hi = nibbles[1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, num_elements: int, signed: bool) -> torch.Tensor:
    """Inverse of pack_int4. num_elements is the true (possibly odd) count
    before any padding, so the trailing pad nibble (if any) is dropped."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    nibbles = torch.stack([lo, hi], dim=1).reshape(-1)[:num_elements].to(torch.int64)

    if signed:
        nibbles = torch.where(nibbles >= 8, nibbles - 16, nibbles)

    return nibbles.to(torch.int8)


def pack_int8(values: torch.Tensor) -> torch.Tensor:
    """INT8 needs no packing (already one value per byte) - identity, kept
    for API symmetry with pack_int4 so callers don't special-case bit-width.
    Does NOT coerce dtype: asymmetric (unsigned, 0-255) values must already
    be a uint8 tensor and symmetric (signed, -128-127) values an int8 tensor
    - forcing int8 here would silently wrap unsigned values >=128 negative."""
    return values


def unpack_int8(packed: torch.Tensor) -> torch.Tensor:
    return packed
