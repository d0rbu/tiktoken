"""Private compact-storage decoder; no tokenizer downloads or model registry."""

import gc
import mmap
import sys
import weakref

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tiktoken import _tiktoken


def literal(ids: list[int], leading_bits: int = 0) -> bytes:
    word = sum(value << (leading_bits + 17 * i) for i, value in enumerate(ids))
    return word.to_bytes((leading_bits + len(ids) * 17 + 7) // 8, "little")


@given(
    ids=st.lists(st.integers(0, 131071), max_size=300),
    leading=st.integers(0, 7),
    cut=st.integers(0, 300),
)
@settings(max_examples=200, deadline=None)
def test_decode_matches_independent_literal_without_touching_canaries(
    ids: list[int], leading: int, cut: int
) -> None:
    start = min(cut, len(ids))
    payload = literal(ids, leading)
    output = np.full(len(ids) - start + 4, -37, dtype=np.int64)
    _tiktoken._unpack_bits17_into(payload, leading + start * 17, output[2:-2])
    assert output.tolist() == [-37, -37, *ids[start:], -37, -37]


@pytest.mark.parametrize("length", range(67))
def test_all_word_tails_offsets_and_domain_extremes(length: int) -> None:
    ids = ([0, 131071, 1, 65536, 131070, 17, 32768, 0] * 9)[:length]
    payload = literal(ids)
    for start in range(length + 1):
        output = np.empty(length - start, dtype=np.int64)
        _tiktoken._unpack_bits17_into(payload, start * 17, output)
        assert output.tolist() == ids[start:]


@pytest.mark.parametrize("start", [7, 8, 17, sys.maxsize, 2**64 - 1])
def test_bounds_failure_precedes_any_destination_write(start: int) -> None:
    output = np.full(2, -37, dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds its byte buffer"):
        _tiktoken._unpack_bits17_into(bytes(5), start, output)
    assert output.tolist() == [-37, -37]


@pytest.mark.parametrize("start", [-1, 2**128])
def test_python_index_overflow_precedes_any_destination_write(start: int) -> None:
    output = np.full(1, -37, dtype=np.int64)
    with pytest.raises(OverflowError):
        _tiktoken._unpack_bits17_into(bytes(3), start, output)
    assert output.tolist() == [-37]


def test_empty_destination_has_no_reads_or_writes() -> None:
    output = np.empty(0, dtype=np.int64)
    _tiktoken._unpack_bits17_into(b"", 0, output)
    with pytest.raises(ValueError, match="exceeds its byte buffer"):
        _tiktoken._unpack_bits17_into(b"", 1, output)


def test_rejects_strides_and_readonly_destination_without_partial_write() -> None:
    output = np.full(6, -37, dtype=np.int64)
    with pytest.raises(ValueError, match="writable and contiguous"):
        _tiktoken._unpack_bits17_into(bytes(14), 0, output[::2])
    with pytest.raises(ValueError, match="bytes must be contiguous"):
        _tiktoken._unpack_bits17_into(np.zeros(28, dtype=np.uint8)[::2], 0, output)
    output.flags.writeable = False
    with pytest.raises(ValueError, match="writable and contiguous"):
        _tiktoken._unpack_bits17_into(bytes(14), 0, output)
    assert output.tolist() == [-37] * 6


@pytest.mark.parametrize(
    "dtype", [np.int32, np.uint64, np.float64, np.dtype(np.int64).newbyteorder("S")]
)
def test_rejects_incompatible_destination_buffer(dtype: object) -> None:
    output = np.zeros(1, dtype=dtype)
    with pytest.raises(BufferError):
        _tiktoken._unpack_bits17_into(bytes(3), 0, output)
    assert output.tolist() == [0]


def test_rejects_wrong_source_type_and_unaligned_destination() -> None:
    output = np.full(1, -37, dtype=np.int64)
    with pytest.raises(BufferError):
        _tiktoken._unpack_bits17_into(np.zeros(3, dtype=np.int8), 0, output)
    assert output.tolist() == [-37]
    backing = np.full(9, 37, dtype=np.uint8)
    with pytest.raises(BufferError):
        _tiktoken._unpack_bits17_into(bytes(3), 0, backing[1:].view(np.int64))
    assert backing.tolist() == [37] * 9


@pytest.mark.parametrize("offset", [0, 8, 16, 24])
def test_readonly_alias_rejected_before_writing(offset: int) -> None:
    backing = np.arange(256, dtype=np.uint8)
    before = backing.copy()
    source = backing.view()
    source.flags.writeable = False
    output = np.ndarray((8,), dtype=np.int64, buffer=backing, offset=offset)
    with pytest.raises(BufferError, match="must not overlap"):
        _tiktoken._unpack_bits17_into(source, 0, output)
    np.testing.assert_array_equal(backing, before)


def test_exports_are_released_on_success_and_failure(tmp_path) -> None:
    path = tmp_path / "tokens.bits17"
    path.write_bytes(literal([1, 131071, 65536]))
    with path.open("rb") as stream:
        source = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    output = np.empty(3, dtype=np.int64)
    weak_output = weakref.ref(output)
    _tiktoken._unpack_bits17_into(source, 0, output)
    assert output.tolist() == [1, 131071, 65536]
    with pytest.raises(ValueError):
        _tiktoken._unpack_bits17_into(source, 1000, output)
    source.close()
    del output
    gc.collect()
    assert weak_output() is None
