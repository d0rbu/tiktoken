"""Independent storage bytes and retained-buffer contracts, without tokenizer I/O."""

import gc
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tiktoken._token_storage import TokenStorageDecoder, TokenStorageEncoder


def oracle(ids: list[int], width: int) -> bytes:
    value = sum(token << (i * width) for i, token in enumerate(ids))
    return value.to_bytes((len(ids) * width + 7) // 8, "little")


def encode(ids: list[int], width: int, steps: list[int]) -> bytes:
    encoder = TokenStorageEncoder(width)
    raw = b"".join(token.to_bytes(4, "little") for token in ids)
    output = []
    at = 0
    for step in cycle(steps):
        if at == len(raw):
            break
        end = min(len(raw), at + step * 4)
        output.append(encoder.push(raw[at:end]))
        at = end
    output.append(encoder.finish())
    assert encoder.finish() == b""
    assert encoder.push(b"") == b""
    return b"".join(output)


@pytest.mark.parametrize("width", range(8, 33))
@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 65])
def test_widths_tails_and_consumed_ranges(width: int, length: int) -> None:
    mask = (1 << width) - 1
    ids = [
        0 if i % 3 == 0 else mask if i % 3 == 1 else (i * 7919) & mask
        for i in range(length)
    ]
    packed = encode(ids, width, [1, 7, 9])
    assert packed == oracle(ids, width)
    decoder = TokenStorageDecoder(width)
    for start in range(length + 1):
        for wanted in [0, 1, 7, 8, 9, length]:
            end = min(length, start + wanted)
            assert (
                decoder.read(packed, start * width, end - start).tolist()
                == ids[start:end]
            )


@given(
    st.integers(8, 32),
    st.lists(st.integers(0, 2**32 - 1), max_size=129),
    st.lists(st.integers(1, 33), min_size=1, max_size=7),
)
def test_frame_grouping_never_changes_bit_bytes(
    width: int, values: list[int], steps: list[int]
) -> None:
    ids = [value & ((1 << width) - 1) for value in values]
    packed = encode(ids, width, steps)
    assert packed == oracle(ids, width)
    assert TokenStorageDecoder(width).read(packed, 0, len(ids)).tolist() == ids


@pytest.mark.parametrize("constructor", [TokenStorageEncoder, TokenStorageDecoder])
@pytest.mark.parametrize("width", [0, 7, 33, 255])
def test_storage_width_admission(
    constructor: type[TokenStorageEncoder] | type[TokenStorageDecoder], width: int
) -> None:
    with pytest.raises(ValueError, match="between 8 and 32"):
        constructor(width)


def test_consumed_truncated_range_fails() -> None:
    packed = oracle([131071] * 8, 17)
    decoder = TokenStorageDecoder(17)
    for data, offset, count in [(packed[:-1], 0, 8), (packed, 0, 9), (packed, 1, 8)]:
        with pytest.raises(ValueError, match="packed token range exceeds stored bytes"):
            decoder.read(data, offset, count)


def test_outputs_keep_their_own_bytes_and_buffer_owners() -> None:
    ids = [i * 131 for i in range(17)]
    raw = b"".join(token.to_bytes(4, "little") for token in ids)
    encoder = TokenStorageEncoder(17)
    first = encoder.push(raw[:32])
    assert type(first) is bytes
    rest = encoder.push(raw[32:]) + encoder.finish()
    packed = first + rest
    decoder = TokenStorageDecoder(17)
    view = decoder.read(packed, 0, 8)
    other = decoder.read(packed, 8 * 17, len(ids) - 8)
    assert view.readonly and view.format == "I" and view.itemsize == 4
    assert view.shape == (8,) and view.strides == (4,) and view.c_contiguous
    assert view.nbytes == 32
    del encoder, decoder, raw, packed
    gc.collect()
    assert first == oracle(ids[:8], 17)
    assert view.tolist() == ids[:8]
    assert other.tolist() == ids[8:]
    with pytest.raises(TypeError):
        view[0] = 1


def test_immutable_decoder_has_no_shared_mutable_results() -> None:
    ids = [(i * 7919) & 131071 for i in range(1024)]
    packed = oracle(ids, 17)
    decoder = TokenStorageDecoder(17)

    def read(start: int) -> memoryview:
        return decoder.read(packed, start * 17, 128)

    with ThreadPoolExecutor(max_workers=2) as workers:
        views = list(workers.map(read, range(64)))
    gc.collect()
    assert [v.tolist() for v in views] == [
        ids[start : start + 128] for start in range(64)
    ]
