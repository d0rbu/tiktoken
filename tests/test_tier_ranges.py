"""Batched ranges retain scalar values, errors and independent output ownership."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tiktoken import _tiktoken as native


def make_fixture():
    codec = native._TokenTierCodec(list(range(1056)), 5, 10)
    ids = ((np.arange(16385, dtype=np.uint32) * 503) % 131072).astype("<u4")
    blocks = []
    for offset in range(0, len(ids), 8192):
        block = ids[offset : offset + 8192]
        data, positions = codec.encode_indexed(block.tobytes())
        blocks.append((data, positions, len(block), block))
    return codec, blocks


def test_many_ranges_match_scalar_and_original_tokens():
    codec, blocks = make_fixture()
    ranges = []
    wanted = []
    for index, start, count in [
        (1, 255, 3000),
        (0, 0, 8192),
        (2, 0, 1),
        (1, 8192, 0),
        (0, 4095, 9),
    ]:
        data, positions, tokens, ids = blocks[index]
        ranges.append((data, positions, tokens, start, count))
        wanted.extend(ids[start : start + count])
    result = codec.decode_ranges(ranges)
    assert bytes(memoryview(result)) == np.array(wanted, dtype="<u4").tobytes()
    assert bytes(memoryview(result)) == b"".join(
        bytes(memoryview(codec.decode_range(*item))) for item in ranges
    )
    assert memoryview(result).readonly
    assert bytes(memoryview(codec.decode_ranges([]))) == b""
    codec.decode_ranges(ranges)
    assert bytes(memoryview(result)) == np.array(wanted, dtype="<u4").tobytes()


@pytest.mark.parametrize(
    "invalid", ["extent", "position", "short", "padding", "huge", "bytearray"]
)
def test_invalid_later_range_does_not_return_partial_output(invalid):
    codec, blocks = make_fixture()
    data, positions, tokens, _ = blocks[0]
    first = (data, positions, tokens, 0, 256)
    bad = {
        "extent": (data, positions, tokens, tokens, 1),
        "position": (data, b"", tokens, 1, 1),
        "short": (data[:1], positions, tokens, 0, 2),
        "padding": (data + b"\0", positions, tokens, 0, tokens),
        "huge": (data, positions, tokens, 0, 2**63),
        "bytearray": (bytearray(data), positions, tokens, 0, 1),
    }[invalid]
    error = TypeError if invalid == "bytearray" else ValueError
    with pytest.raises(error):
        codec.decode_ranges([first, bad])
    assert bytes(memoryview(codec.decode_ranges([first]))) == bytes(
        memoryview(codec.decode_range(*first))
    )


def test_batched_range_owners_are_independent_across_threads():
    codec, blocks = make_fixture()
    ranges = [
        (data, positions, tokens, 0, tokens) for data, positions, tokens, _ in blocks
    ]
    expected = b"".join(ids.tobytes() for _, _, _, ids in blocks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(codec.decode_ranges, [ranges] * 32))
    del codec, blocks, ranges, pool
    assert all(bytes(memoryview(output)) == expected for output in outputs)
