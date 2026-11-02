"""Frame batching must preserve each existing indexed block, including failures."""

import concurrent.futures
import struct
import sys

import pytest
from hypothesis import given, settings, strategies as st

from tiktoken import _tiktoken


def codec():
    return _tiktoken._TokenTierCodec(list(reversed(range(1056))), 5, 10)


def check_blocks(instance, raw, block_tokens):
    expected = [
        instance.encode_indexed(raw[start : start + block_tokens * 4])
        for start in range(0, len(raw), block_tokens * 4)
    ]
    actual = instance.encode_blocks(raw, block_tokens)
    assert actual == expected
    assert all(isinstance(part, bytes) for block in actual for part in block)
    restored = []
    for index, (data, positions) in enumerate(actual):
        count = min(block_tokens, len(raw) // 4 - index * block_tokens)
        restored.append(bytes(memoryview(instance.decode_range(data, positions, count, 0, count))))
    assert b"".join(restored) == raw
    return actual


@pytest.mark.parametrize("block_tokens", [1, 255, 256, 8192, 131071, 131072, 131073])
def test_every_id_and_partial_block(block_tokens):
    raw = struct.pack("<131072I", *range(131072))
    check_blocks(codec(), raw, block_tokens)


@settings(max_examples=40)
@given(st.lists(st.integers(0, 131071), max_size=2049), st.integers(1, 512))
def test_generated_frames(ids, block_tokens):
    check_blocks(codec(), struct.pack(f"<{len(ids)}I", *ids), block_tokens)


@pytest.mark.parametrize("count", [0, 1, 255, 256, 257, 8191, 8192, 8193, 16385])
def test_original_input_not_retained_and_outputs_owned(count):
    instance = codec()
    raw = struct.pack(f"<{count}I", *[i % 131072 for i in range(count)])
    before = sys.getrefcount(raw)
    output = check_blocks(instance, raw, 8192)
    assert sys.getrefcount(raw) == before
    saved = list(output)
    instance.encode_blocks(b"", 8192)
    del raw, instance
    assert output == saved


@pytest.mark.parametrize("raw", [b"x", b"xyz", struct.pack("<3I", 1, 2, 131072)])
def test_later_invalid_block_has_no_partial_return(raw):
    instance = codec()
    before = sys.getrefcount(raw)
    with pytest.raises(ValueError, match="^invalid uint32 token block$"):
        instance.encode_blocks(raw, 1)
    assert sys.getrefcount(raw) == before
    check_blocks(instance, struct.pack("<2I", 31, 65536), 1)


@pytest.mark.parametrize("block_tokens", [0, sys.maxsize])
def test_invalid_block_size(block_tokens):
    with pytest.raises(ValueError, match="^invalid token block size$"):
        codec().encode_blocks(b"", block_tokens)


@pytest.mark.parametrize("raw", [bytearray(4), memoryview(b"\0" * 4), [0]])
def test_input_stays_immutable_bytes_only(raw):
    with pytest.raises(TypeError):
        codec().encode_blocks(raw, 8192)


def test_frozen_codec_shared_by_many_callers():
    instance = codec()
    frames = [struct.pack("<8193I", *[(i + lane * 13) % 131072 for i in range(8193)]) for lane in range(32)]
    expected = [check_blocks(instance, raw, 8192) for raw in frames]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        actual = list(executor.map(lambda raw: instance.encode_blocks(raw, 8192), frames))
    assert actual == expected
