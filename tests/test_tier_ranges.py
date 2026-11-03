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


@pytest.mark.parametrize("a,b", [(1, 2), (5, 10), (8, 14)])
def test_direct_ranges_match_original_ids_and_keep_destination_tail(a, b):
    codec = native._TokenTierCodec(list(range((1 << a) + (1 << b))), a, b)
    ids = ((np.arange(8193, dtype=np.uint32) * 503) % 131072).astype("<u4")
    data, positions = codec.encode_indexed(ids.tobytes())
    selections = [(255, 3000), (0, 7), (8192, 1), (8193, 0), (4095, 19)]
    ranges = [(data, positions, len(ids), start, count) for start, count in selections]
    expected = np.concatenate(
        [ids[start : start + count] for start, count in selections]
    )
    storage = np.full(len(expected) + 7, -777, dtype=np.int64)
    written = codec.decode_ranges_into(ranges, storage[2:-2])
    assert written == len(expected)
    np.testing.assert_array_equal(storage[2 : written + 2], expected)
    assert np.all(storage[:2] == -777)
    assert np.all(storage[written + 2 :] == -777)
    assert codec.decode_ranges_into([], np.empty(0, dtype=np.int64)) == 0
    with pytest.raises(ValueError, match="invalid indexed token range"):
        codec.decode_ranges_into(ranges, np.empty(2, dtype=np.int64))


@pytest.mark.parametrize(
    "kind", ["readonly", "stride", "reversed", "endian", "wrong_type", "unaligned"]
)
def test_direct_ranges_reject_unsupported_destinations_without_writes(kind):
    codec, blocks = make_fixture()
    data, positions, tokens, _ = blocks[0]
    storage = np.full(64, -777, dtype=np.int64)
    output = storage
    if kind == "readonly":
        output = storage.view()
        output.flags.writeable = False
    elif kind == "stride":
        output = storage[::2]
    elif kind == "reversed":
        output = storage[::-1]
    elif kind == "endian":
        output = storage.view(storage.dtype.newbyteorder("S"))
    elif kind == "wrong_type":
        output = storage.view(np.uint32)
    elif kind == "unaligned":
        output = np.ndarray(
            (16,), dtype=np.int64, buffer=bytearray(16 * 8 + 1), offset=1
        )
    with pytest.raises((ValueError, BufferError)):
        codec.decode_ranges_into([(data, positions, tokens, 0, 8)], output)
    assert np.all(storage == -777)


@pytest.mark.parametrize(
    "invalid", ["extent", "position", "short", "padding", "huge", "bytearray"]
)
def test_direct_range_errors_keep_existing_returned_buffer_contract(invalid):
    codec, blocks = make_fixture()
    data, positions, tokens, ids = blocks[0]
    bad = {
        "extent": (data, positions, tokens, tokens, 1),
        "position": (data, b"", tokens, 1, 1),
        "short": (data[:1], positions, tokens, 0, 2),
        "padding": (data + b"\0", positions, tokens, 0, tokens),
        "huge": (data, positions, tokens, 0, 2**63),
        "bytearray": (bytearray(data), positions, tokens, 0, 1),
    }[invalid]
    first = (data, positions, tokens, 0, 256)
    error = TypeError if invalid == "bytearray" else ValueError
    with pytest.raises(error):
        codec.decode_ranges_into([first, bad], np.empty(tokens + 256, dtype=np.int64))
    np.testing.assert_array_equal(
        np.frombuffer(codec.decode_ranges([first]), dtype="<u4"), ids[:256]
    )


def test_direct_destinations_are_independent_across_owners():
    codec, blocks = make_fixture()
    ranges = [
        (data, positions, tokens, 0, tokens) for data, positions, tokens, _ in blocks
    ]
    expected = np.concatenate([ids for _, _, _, ids in blocks])

    def read(_):
        output = np.empty(len(expected), dtype=np.int64)
        assert codec.decode_ranges_into(ranges, output) == len(expected)
        return output

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(read, range(16)))
    for output in outputs:
        np.testing.assert_array_equal(output, expected)
    outputs[0][0] = -1
    assert outputs[1][0] == expected[0]
