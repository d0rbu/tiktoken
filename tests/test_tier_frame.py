"""Contiguous frames preserve every indexed block and own their result bytes."""

import concurrent.futures
import struct
import sys

import pytest
from hypothesis import given, settings, strategies as st

from tiktoken import _tiktoken


def codec():
    return _tiktoken._TokenTierCodec(list(reversed(range(1056))), 5, 10)


def check_frame(instance, raw, block_tokens):
    expected = instance.encode_blocks(raw, block_tokens)
    data, index = instance.encode_frame(raw, block_tokens)
    assert type(data) is bytes and type(index) is bytes
    assert len(index) == len(expected) * 24
    begin = previous_tokens = 0
    restored = []
    for (end_tokens, payload_end, end), (payload, positions) in zip(
        struct.iter_unpack("<QQQ", index), expected, strict=True
    ):
        assert data[begin:payload_end] == payload
        assert data[payload_end:end] == positions
        tokens = end_tokens - previous_tokens
        assert tokens == min(block_tokens, len(raw) // 4 - previous_tokens)
        restored.append(bytes(memoryview(instance.decode_range(payload, positions, tokens, 0, tokens))))
        begin, previous_tokens = end, end_tokens
    assert begin == len(data)
    assert previous_tokens == len(raw) // 4
    assert b"".join(restored) == raw
    return data, index


@pytest.mark.parametrize("block_tokens", [1, 255, 256, 257, 8192, 131071, 131072, 131073])
def test_all_ids_and_boundaries(block_tokens):
    check_frame(codec(), struct.pack("<131072I", *range(131072)), block_tokens)


@settings(max_examples=40)
@given(st.lists(st.integers(0, 131071), max_size=2049), st.integers(1, 512))
def test_generated_frames(ids, block_tokens):
    check_frame(codec(), struct.pack(f"<{len(ids)}I", *ids), block_tokens)


@pytest.mark.parametrize("count", [0, 1, 255, 256, 257, 8191, 8192, 8193, 16385])
def test_owned_output_and_input_release(count):
    instance = codec()
    raw = struct.pack(f"<{count}I", *[i % 131072 for i in range(count)])
    before = sys.getrefcount(raw)
    result = check_frame(instance, raw, 8192)
    assert sys.getrefcount(raw) == before
    saved = tuple(bytes(bytearray(part)) for part in result)
    instance.encode_frame(b"", 8192)
    del instance, raw
    assert result == saved


@pytest.mark.parametrize("raw", [b"x", b"xyz", struct.pack("<3I", 1, 2, 131072)])
def test_late_invalid_input_returns_no_frame(raw):
    instance = codec()
    before = sys.getrefcount(raw)
    with pytest.raises(ValueError, match="^invalid uint32 token block$"):
        instance.encode_frame(raw, 1)
    assert sys.getrefcount(raw) == before
    check_frame(instance, struct.pack("<2I", 31, 65536), 1)


@pytest.mark.parametrize("block_tokens", [0, sys.maxsize])
def test_invalid_block_size(block_tokens):
    with pytest.raises(ValueError, match="^invalid token block size$"):
        codec().encode_frame(b"", block_tokens)


@pytest.mark.parametrize("raw", [bytearray(4), memoryview(b"\0" * 4), [0]])
def test_only_immutable_input(raw):
    with pytest.raises(TypeError):
        codec().encode_frame(raw, 8192)


def test_shared_frozen_codec():
    instance = codec()
    frames = [struct.pack("<8193I", *[(i + lane * 13) % 131072 for i in range(8193)]) for lane in range(32)]
    expected = [check_frame(instance, raw, 8192) for raw in frames]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        actual = list(executor.map(lambda raw: instance.encode_frame(raw, 8192), frames))
    assert actual == expected
