"""One-call endpoint encoding retains independent-prefix EOF semantics."""

from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tiktoken import _tiktoken

PATTERN = (Path(__file__).parents[1] / "src/rmcp_pattern.txt").read_text()
SPECIALS = {"<S>": 400, "<T>": 401, "aba": 402, "bab": 403, "<S>long": 404, "éS": 405}


def core(pattern=PATTERN, specials=SPECIALS):
    ranks = {bytes([byte]): byte for byte in range(256)}
    ranks.update({b"ab": 256, b"abc": 257, b"aa": 258, b"  ": 259, b"\r\n": 260, "界".encode(): 261})
    return _tiktoken.CoreBPE(ranks, specials, pattern)


@pytest.mark.parametrize("allowed", [frozenset(), frozenset({"<S>", "aba", "éS"}), frozenset(SPECIALS)])
@given(text=st.text(max_size=200), positions=st.lists(st.integers(0, 200), max_size=40))
@settings(max_examples=500, deadline=None)
def test_full_ids_and_independently_tokenized_prefixes(text, positions, allowed):
    model = core()
    chars = [p % (len(text) + 1) for p in positions] + [len(text)] * 3
    byte_ends = [len(text[:p].encode()) for p in chars]
    ids, counts = model.encode_with_prefix_counts(text, byte_ends, allowed)
    assert list(memoryview(ids)) == model.encode(text, allowed)
    assert counts == [len(model.encode(text[:p], allowed)) for p in chars]


@pytest.mark.parametrize("text", ["ababa" * 10 + "<S>long<T>", "  a\t\r\n  'sfoo'rd!", "界😀é\u0301\x00\u0085 1a", "abc" * 1024, "x" * 65536, "x" * 65537])
@pytest.mark.parametrize("pattern", [PATTERN, r"[\s\S]+"])
def test_eof_specials_unicode_and_fallbacks(text, pattern):
    model = core(pattern)
    chars = list(range(min(len(text), 50) + 1)) + [len(text)] * 3
    ids, counts = model.encode_with_prefix_counts(text, [len(text[:p].encode()) for p in chars], frozenset(SPECIALS))
    assert list(memoryview(ids)) == model.encode(text, frozenset(SPECIALS))
    assert counts == [len(model.encode(text[:p], frozenset(SPECIALS))) for p in chars]


@pytest.mark.parametrize("end", [1, 3, 100])
def test_invalid_byte_ends_keep_first_error(end):
    with pytest.raises(ValueError, match="prefix end must"):
        core().encode_with_prefix_counts("é", [2] * 3 + [end], frozenset(SPECIALS))


def test_partial_vocabulary_and_empty_registry_preserve_native_failure():
    model = _tiktoken.CoreBPE({b"a": 0}, {"<S>": 1}, PATTERN)
    with pytest.raises(BaseException) as scalar:
        model.count_tokens("!", frozenset({"<S>"}))
    with pytest.raises(type(scalar.value)) as actual:
        model.encode_with_prefix_counts("a!", [1] * 8 + [2, 99], frozenset({"<S>"}))
    assert str(actual.value) == str(scalar.value)
    empty = core(specials={})
    ids, counts = empty.encode_with_prefix_counts("abc", [0, 3], frozenset())
    assert list(memoryview(ids)) == empty.encode("abc", frozenset())
    assert counts == [0, 1]
