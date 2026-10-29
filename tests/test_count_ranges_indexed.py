"""The indexed binding keeps full-ID counts, extraction and first-error behavior."""

from functools import lru_cache
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tiktoken import _tiktoken

PATTERN = (Path(__file__).parents[1] / "src/rmcp_pattern.txt").read_text()


@lru_cache(maxsize=1)
def core():
    ranks = {bytes([byte]): byte for byte in range(256)}
    ranks.update({b"ab": 256, b"abc": 257, b"aa": 258, b"  ": 259, b"\r\n": 260, "界".encode(): 261})
    return _tiktoken.CoreBPE(ranks, {"<S>": 400, "<T>": 401, "aba": 402, "bab": 403, "<S>long": 404, "éS": 405}, PATTERN)


@pytest.mark.parametrize("allowed", [frozenset(), frozenset({"<S>", "aba", "éS"}), frozenset({"<S>", "<T>", "aba", "bab", "<S>long", "éS"})])
@given(text=st.text(max_size=180), intervals=st.lists(st.tuples(st.integers(0, 180), st.integers(0, 180)), max_size=50))
@settings(max_examples=500, deadline=None)
def test_indexed_ranges_equal_independent_complete_ids(text, intervals, allowed):
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode()))
    spans = [tuple(sorted((a % len(offsets), b % len(offsets)))) for a, b in intervals]
    spans += [(0, len(text))] * 2  # force index amortization for every nonempty string
    ranges = [(offsets[a], offsets[b]) for a, b in spans]
    expected = [len(core().encode(text[a:b], allowed)) for a, b in spans]
    assert core().count_tokens_ranges(text, ranges, allowed) == expected
    assert core().count_tokens_ranges(text, ranges[::-1], allowed) == expected[::-1]


@pytest.mark.parametrize("text", ["ababa" * 10 + "<S>long<T>", "  a\t\r\n  'sfoo'rd!", "界😀é\u0301\x00\u0085 1a", "abc" * 1024, "x" * 65536, "x" * 65537])
def test_difficult_and_size_boundary_ranges(text):
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode()))
    short = min(len(text), 24)
    spans = [(i, j) for i in range(short + 1) for j in range(i, short + 1)]
    spans += [(0, len(text))] * 4
    for allowed in [frozenset(), frozenset({"<S>", "aba", "éS"}), frozenset({"<S>", "<T>", "aba", "bab", "<S>long", "éS"})]:
        expected = [len(core().encode(text[a:b], allowed)) for a, b in spans]
        assert core().count_tokens_ranges(text, [(offsets[a], offsets[b]) for a, b in spans], allowed) == expected


def test_partial_vocabulary_does_not_encode_unrequested_invalid_bytes():
    allowed = frozenset({"<S>"})
    model = _tiktoken.CoreBPE({b"a": 0}, {"<S>": 1}, PATTERN)
    assert model.count_tokens_ranges("a!", [(0, 1)] * 8, allowed) == [1] * 8
    with pytest.raises(BaseException) as scalar:
        model.count_tokens("!", allowed)
    with pytest.raises(type(scalar.value)) as actual:
        model.count_tokens_ranges("a!", [(0, 1)] * 8 + [(1, 2), (0, 99)], allowed)
    assert str(actual.value) == str(scalar.value)
    with pytest.raises(ValueError, match="ordered, in bounds and on UTF-8 boundaries"):
        model.count_tokens_ranges("a!", [(0, 99), (1, 2)], allowed)


@pytest.mark.parametrize("span", [(0, 1), (1, 1), (2, 0), (0, 9)])
def test_index_preflight_retains_invalid_utf8_and_range_errors(span):
    with pytest.raises(ValueError, match="ordered, in bounds and on UTF-8 boundaries"):
        core().count_tokens_ranges("é", [(0, 2)] * 8 + [span], frozenset())
    assert core().count_tokens_ranges("é", [(0, 2)] * 8, frozenset()) == [2] * 8


def test_empty_registry_retains_scalar_panic_for_unknown_empty_special():
    model = _tiktoken.CoreBPE({bytes([byte]): byte for byte in range(256)}, {}, PATTERN)
    assert model.count_tokens_ranges("abc", [(0, 3)] * 2, frozenset()) == [3, 3]
    with pytest.raises(BaseException) as scalar:
        model.count_tokens("abc", frozenset({""}))
    with pytest.raises(type(scalar.value)) as actual:
        model.count_tokens_ranges("abc", [(0, 3)] * 2, frozenset({""}))
    assert str(actual.value) == str(scalar.value)
