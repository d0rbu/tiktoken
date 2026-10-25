"""UTF-8 byte ranges retain independent scalar encoding, order, policy and errors."""

import pytest
from hypothesis import given, settings, strategies as st

from tiktoken import _tiktoken


def core():
    ranks = {bytes([value]): value for value in range(256)}
    ranks.update({b"ab": 256, b"abc": 257, b"aa": 258, b"  ": 259})
    return _tiktoken.CoreBPE(ranks, {"<S>": 300, "<T>": 301}, r"[\s\S]+")


@pytest.mark.parametrize("allowed", [frozenset(), frozenset({"<S>"}), frozenset({"<S>", "<T>"})])
@given(text=st.text(max_size=150), intervals=st.lists(st.tuples(st.integers(0, 150), st.integers(0, 150)), max_size=20))
@settings(max_examples=100, deadline=None)
def test_ranges_match_independent_ids(text, intervals, allowed):
    model = core()
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode("utf-8")))
    spans = [tuple(sorted((a % len(offsets), b % len(offsets)))) for a, b in intervals]
    byte_ranges = [(offsets[a], offsets[b]) for a, b in spans]
    expected = [len(model.encode(text[a:b], allowed)) for a, b in spans]
    assert model.count_tokens_ranges(text, byte_ranges, allowed) == expected
    assert model.count_tokens_ranges(text, list(reversed(byte_ranges)), allowed) == list(reversed(expected))


@pytest.mark.parametrize("length", [0, 1, 2, 98, 99, 100, 101, 1024])
def test_ranges_empty_duplicate_unicode_and_partial_specials(length):
    model = core()
    text = "é<S>" + "abc" * length + "<T>🦊"
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode()))
    intervals = [(i, j) for i in range(min(len(text), 5) + 1) for j in range(i, min(len(text), 5) + 1)]
    intervals += [(0, len(text)), (1, len(text)), (0, len(text))]
    ranges = [(offsets[a], offsets[b]) for a, b in intervals]
    for allowed in (frozenset(), frozenset({"<S>"}), frozenset({"<S>", "<T>"})):
        assert model.count_tokens_ranges(text, ranges, allowed) == [
            model.count_tokens(text[a:b], allowed) for a, b in intervals
        ]
    assert model.count_tokens_ranges(text, [], frozenset()) == []
    assert model.count_tokens_ranges("", [(0, 0)], frozenset()) == [0]


@pytest.mark.parametrize("span", [(0, 1), (1, 2), (1, 1), (2, 0), (0, 3), (3, 3)])
def test_ranges_reject_non_boundaries_order_and_bounds_without_partial_output(span):
    model = core()
    with pytest.raises(ValueError, match="ordered, in bounds and on UTF-8 boundaries"):
        model.count_tokens_ranges("é", [(0, 2), span], frozenset())
    assert model.count_tokens_ranges("é", [(0, 2)], frozenset()) == [2]


@pytest.mark.parametrize("span", [(-1, 0), (0, -1), (0, 2**100)])
def test_ranges_reject_indices_outside_usize(span):
    with pytest.raises(OverflowError):
        core().count_tokens_ranges("abc", [span], frozenset())


@pytest.mark.parametrize("invalid", ["\ud800", 7])
def test_ranges_preserve_text_extraction_errors(invalid):
    model = core()
    with pytest.raises((UnicodeEncodeError, TypeError)) as scalar:
        model.count_tokens(invalid, frozenset())
    with pytest.raises(type(scalar.value)):
        model.count_tokens_ranges(invalid, [(0, 0)], frozenset())
    assert model.count_tokens_ranges("abc", [(0, 3)], frozenset()) == [1]


def test_ranges_reject_malformed_indices_and_special_allowance():
    model = core()
    for ranges in [[("zero", 1)], [(0, 1, 2)], [None]]:
        with pytest.raises((TypeError, ValueError)):
            model.count_tokens_ranges("abc", ranges, frozenset())
    with pytest.raises(TypeError):
        model.count_tokens_ranges("abc", [(0, 3)], frozenset({7}))
    # Like Python indexing, the native integer interface accepts bool as 0/1.
    assert model.count_tokens_ranges("abc", [(False, True)], frozenset()) == [1]


def test_ranges_preserve_scalar_native_failure_and_leave_encoder_usable():
    model = _tiktoken.CoreBPE({b"a": 0}, {}, r"[\s\S]+")
    with pytest.raises(BaseException) as scalar:
        model.count_tokens("b", frozenset())
    assert type(scalar.value).__name__ == "PanicException"
    with pytest.raises(type(scalar.value)) as ranges:
        model.count_tokens_ranges("ab", [(0, 1), (1, 2)], frozenset())
    assert str(ranges.value) == str(scalar.value)
    assert model.count_tokens_ranges("a", [(0, 1)], frozenset()) == [1]
