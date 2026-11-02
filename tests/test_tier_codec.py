"""Independent bit and sparse-position oracles for every token tier."""

import numpy as np
import pytest

from tiktoken import _tiktoken as N

BOOK = {"order": [(i * 7919) % (1 << 17) for i in range(1056)], "a": 5, "b": 10}


@pytest.fixture(scope="module")
def codec():
    return N._TokenTierCodec(BOOK["order"], BOOK["a"], BOOK["b"])


def oracle(ids):
    ranks = np.full(1 << 17, -1, dtype=np.int32)
    ranks[BOOK["order"]] = np.arange(len(BOOK["order"]))
    r = ranks[ids]
    a = BOOK["a"]
    b = BOOK["b"]
    small = (r >= 0) & (r < (1 << a))
    medium = r >= (1 << a)
    widths = np.where(small, a + 1, np.where(medium, b + 2, 19))
    codes = np.where(
        small, r, np.where(medium, (2 << b) + (r - (1 << a)), (3 << 17) + ids)
    ).astype(np.uint32)
    columns = np.arange(18, -1, -1)
    bits = ((codes[:, None] >> columns) & 1).astype(np.uint8)
    return np.packbits(
        bits[columns[None, :] < widths[:, None]], bitorder="big"
    ).tobytes()


def test_all_original_ids(codec):
    ids = np.arange(1 << 17, dtype="<u4")
    encoded = codec.encode(ids.tobytes())
    assert encoded == oracle(ids)
    assert bytes(memoryview(codec.decode(encoded, len(ids)))) == ids.tobytes()


@pytest.mark.parametrize(
    "count", [0, 1, 2, 7, 8, 9, 63, 64, 65, 2047, 2048, 8191, 8192, 8193]
)
def test_boundaries(codec, count):
    choices = np.array(
        BOOK["order"][:32] + BOOK["order"][32:] + [131071], dtype=np.uint32
    )
    ids = choices[(np.arange(count) * 57 + 93) % len(choices)].astype("<u4")
    encoded = codec.encode(ids.tobytes())
    assert encoded == oracle(ids)
    out = codec.decode(encoded, count)
    assert bytes(memoryview(out)) == ids.tobytes()
    codec.encode(b"")
    assert bytes(memoryview(out)) == ids.tobytes()


def test_bad_input(codec):
    for raw in [b"x", np.array([131072], dtype="<u4").tobytes()]:
        with pytest.raises(ValueError):
            codec.encode(raw)
    with pytest.raises(ValueError):
        codec.decode(b"x", 10**9)
    with pytest.raises(ValueError):
        N._TokenTierCodec([0] * 1056, 5, 10)


def test_padding_and_extent(codec):
    raw = np.array([BOOK["order"][0]], dtype="<u4").tobytes()
    encoded = codec.encode(raw)
    with pytest.raises(ValueError):
        codec.decode(encoded + b"\0", 1)
    with pytest.raises(ValueError):
        codec.decode(encoded[:-1], 1)
    with pytest.raises(ValueError):
        codec.decode(encoded[:-1] + bytes([encoded[-1] | 1]), 1)


@pytest.mark.parametrize("count", [0, 1, 255, 256, 257, 1024, 8192, 8193])
def test_sparse_index_boundaries(codec, count):
    ids = ((np.arange(count, dtype=np.uint32) * 173 + 3) % (1 << 17)).astype("<u4")
    raw = ids.tobytes()
    encoded, index = codec.encode_indexed(raw)
    assert encoded == oracle(ids)
    assert len(index) == 4 * ((count + 255) // 256)
    for start in sorted(
        set(
            [
                0,
                min(count, 1),
                min(count, 255),
                min(count, 256),
                max(0, count - 2048),
                count,
            ]
        )
    ):
        for width in sorted(
            set([0, min(1, count - start), min(2048, count - start), count - start])
        ):
            actual = codec.decode_range(encoded, index, count, start, width)
            assert bytes(memoryview(actual)) == raw[start * 4 : (start + width) * 4]


def test_sparse_index_exact_positions(codec):
    ids = np.array(
        (BOOK["order"][:32] + BOOK["order"][32:] + [131071]) * 8, dtype="<u4"
    )
    ranks = np.full(1 << 17, -1, dtype=np.int32)
    ranks[BOOK["order"]] = np.arange(len(BOOK["order"]))
    selected = ranks[ids]
    widths = np.where(
        (selected >= 0) & (selected < 32), 6, np.where(selected >= 32, 12, 19)
    )
    offsets = np.r_[0, np.cumsum(widths, dtype=np.int64)]
    encoded, index = codec.encode_indexed(ids.tobytes())
    assert np.frombuffer(index, dtype="<u4").tolist() == offsets[:-1:256].tolist()
    for start in range(0, len(ids), 137):
        width = min(2048, len(ids) - start)
        assert (
            bytes(
                memoryview(codec.decode_range(encoded, index, len(ids), start, width))
            )
            == ids[start : start + width].tobytes()
        )


def test_sparse_index_bad_extents(codec):
    ids = np.arange(8192, dtype="<u4")
    encoded, index = codec.encode_indexed(ids.tobytes())
    for data, positions, total, start, width in [
        (encoded, b"", 8192, 1, 1),
        (encoded, (2**32 - 1).to_bytes(4, "little"), 8192, 0, 1),
        (encoded, index, 8192, 8191, 2),
        (encoded, index, 8192, 8193, 0),
        (encoded[:1], index, 8192, 0, 20),
    ]:
        with pytest.raises(ValueError):
            codec.decode_range(data, positions, total, start, width)


def test_sparse_index_is_readonly_le_bytes(codec):
    data, index = codec.encode_indexed(np.arange(1025, dtype="<u4").tobytes())
    assert isinstance(index, bytes) and len(index) == 20
    assert np.frombuffer(index, dtype="<u4")[0] == 0
    with pytest.raises(TypeError):
        codec.decode_range(data, [0], 1025, 0, 1)
    for invalid in [b"", b"x", b"xyz"]:
        with pytest.raises(ValueError):
            codec.decode_range(data, invalid, 1025, 0, 1)
