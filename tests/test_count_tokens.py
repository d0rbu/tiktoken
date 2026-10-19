"""Count-only traversal agrees with IDs on complete byte vocabularies."""
import pytest
from hypothesis import given, settings, strategies as st

from tiktoken import _tiktoken


def core() -> _tiktoken.CoreBPE:
    ranks = {bytes([value]): value for value in range(256)}
    ranks.update({b"ab": 256, b"abc": 257, b"aa": 258, b"  ": 259})
    return _tiktoken.CoreBPE(ranks, {"<S>": 300, "<T>": 301}, r"[\s\S]+")


@pytest.mark.parametrize("allowed", [frozenset(), frozenset({"<S>"}), frozenset({"<S>", "<T>"})])
@given(text=st.text(max_size=300))
@settings(max_examples=80, deadline=None)
def test_counts_match_ids(text: str, allowed: frozenset[str]) -> None:
    model = core()
    assert model.count_tokens(text, allowed) == len(model.encode(text, allowed))


@pytest.mark.parametrize("length", [0, 1, 2, 98, 99, 100, 101, 1024])
def test_count_boundaries_and_specials(length: int) -> None:
    model = core()
    for text in ("a" * length, "é" * length, " " * length, "<S>" + "abc" * length + "<T>"):
        for allowed in (frozenset(), frozenset({"<S>", "<T>"})):
            assert model.count_tokens(text, allowed) == len(model.encode(text, allowed))


@pytest.mark.parametrize("invalid", ["\ud800", 7])
def test_count_preserves_text_extraction_errors(invalid: str | int) -> None:
    model = core()
    with pytest.raises((UnicodeEncodeError, TypeError)) as encoded:
        model.encode(invalid, frozenset())
    with pytest.raises(type(encoded.value)):
        model.count_tokens(invalid, frozenset())
    assert model.count_tokens("abc", frozenset()) == 1
