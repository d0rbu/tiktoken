"""Local native-buffer contracts; no registry lookup or tokenizer download."""

import gc

import numpy as np
import pytest

from tiktoken import _tiktoken


def core() -> _tiktoken.CoreBPE:
    ranks = {bytes([value]): value for value in range(256)}
    ranks.update({b"ab": 256, b"abc": 257, b"aa": 258, b"  ": 259})
    return _tiktoken.CoreBPE(ranks, {"<S>": 300, "<T>": 301}, r"[\s\S]+")


@pytest.mark.parametrize(
    "allowed", [frozenset(), frozenset({"<S>"}), frozenset({"<S>", "<T>"})]
)
def test_owned_batch_matches_scalar_document_bytes(allowed: frozenset[str]) -> None:
    model = core()
    texts = ("", "a", "b", "ab", "abc", "a b", "é😀\0", "<S>aa<T>") * 8
    expected = [
        np.frombuffer(model.encode_to_tiktoken_buffer(text, allowed), dtype=np.uint32)
        for text in texts
    ]
    buffers = model.encode_to_tiktoken_buffers(texts, allowed)
    assert isinstance(buffers, list)
    assert len(buffers) == len(texts)
    for buffer, row in zip(buffers, expected, strict=True):
        actual = np.frombuffer(buffer, dtype=np.uint32)
        assert actual.tobytes() == row.tobytes()
        assert not actual.flags.writeable
    # Joining documents would permit the "ab" merge across the first boundary.
    assert expected[1].tolist() == [97]
    assert expected[2].tolist() == [98]
    assert expected[3].tolist() == [256]


def test_owned_batch_preserves_empty_batch_and_empty_documents() -> None:
    model = core()
    assert model.encode_to_tiktoken_buffers((), frozenset()) == []
    buffers = model.encode_to_tiktoken_buffers(("", "", ""), frozenset())
    assert len(buffers) == 3
    assert all(np.frombuffer(buffer, dtype=np.uint32).size == 0 for buffer in buffers)


def test_owned_batch_views_keep_independent_readonly_owners_alive() -> None:
    model = core()
    texts = ("ab", "ab", "é😀")
    buffers = model.encode_to_tiktoken_buffers(texts, frozenset())
    assert buffers[0] is not buffers[1]
    rows = [np.frombuffer(buffer, dtype=np.uint32) for buffer in buffers]
    expected = [row.tobytes() for row in rows]
    assert not np.shares_memory(rows[0], rows[1])
    del buffers, model, texts
    gc.collect()
    assert [row.tobytes() for row in rows] == expected
    with pytest.raises(ValueError, match="read-only"):
        rows[0][0] = 7


@pytest.mark.parametrize("invalid", ["\ud800", 7])
def test_owned_batch_rejects_invalid_text_without_lossy_replacement(invalid: str | int) -> None:
    model = core()
    with pytest.raises((UnicodeEncodeError, TypeError)) as scalar:
        model.encode_to_tiktoken_buffer(invalid, frozenset())
    with pytest.raises(type(scalar.value)):
        model.encode_to_tiktoken_buffers(("ab", invalid, "later"), frozenset())
    # The model remains usable after argument extraction fails. The RMCP caller
    # preflights the whole frame for surrogates before invoking this private API.
    row = model.encode_to_tiktoken_buffers(("ab",), frozenset())[0]
    assert np.frombuffer(row, dtype=np.uint32).tolist() == [256]
