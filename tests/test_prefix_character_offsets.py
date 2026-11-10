from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st
from tiktoken import _tiktoken

from .test_character_offsets import expected


def model(guarded=True):
    ranks = {bytes([value]): value for value in range(256)}
    ranks.update(
        {
            b"ab": 256,
            b"abc": 257,
            b"aa": 258,
            b"  ": 259,
            "é".encode(): 260,
            b"\xf0\x9f": 261,
            b"\x98\x80": 262,
        }
    )
    specials = {"<S>": 300, "<T>": 301, "aba": 302, "<S>long": 303, "éS": 304}
    pattern = (
        (Path(__file__).parents[1] / "src/rmcp_pattern.txt").read_text() if guarded else r"[\s\S]+"
    )
    return (
        _tiktoken.CoreBPE(ranks, specials, pattern),
        {**{v: k for k, v in ranks.items()}, **{v: k.encode() for k, v in specials.items()}},
        frozenset(specials),
    )


def reference(core, decoder, text, ends, allowed, limit):
    result = []
    for end in ends:
        if end > len(text):
            raise ValueError("prefix end exceeds the original character bounds")
        row = expected(core, decoder, text[:end], allowed)
        result.append(row)
        if not row[0] or len(row[0]) > limit:
            break
    return result


@pytest.mark.parametrize("guarded", [True, False])
@pytest.mark.parametrize("policy", ["all", "partial", "none"])
@settings(max_examples=100, deadline=None)
@given(text=st.text(st.characters(exclude_categories=("Cs",)), max_size=100))
def test_every_unicode_character_prefix_matches_independent_scalar(text, guarded, policy):
    core, decoder, allowed = model(guarded)
    allowed = (
        allowed
        if policy == "all"
        else frozenset({"<S>", "aba"})
        if policy == "partial"
        else frozenset()
    )
    ends = list(range(1, len(text) + 1)) + [len(text)] * 2
    for ends in [ends, list(reversed(ends))]:
        assert core._encode_with_character_offsets_prefixes(
            text, ends, allowed, 100000
        ) == reference(core, decoder, text, ends, allowed, 100000)


@pytest.mark.parametrize(
    "text",
    ["éS界😀\0 a\u0085\u2000\r\n 're  ", "ababa ababababab <S>long<T>abc", "a" * 101 + "<S>é😀"],
)
@pytest.mark.parametrize("limit", [0, 1, 4, 100000])
def test_limits_empty_unsorted_and_special_boundaries(text, limit):
    core, decoder, allowed = model()
    for ends in [[], [0, len(text) + 1], list(range(1, len(text) + 1)), [len(text), 1, len(text)]]:
        assert core._encode_with_character_offsets_prefixes(
            text, ends, allowed, limit
        ) == reference(core, decoder, text, ends, allowed, limit)
    with pytest.raises(ValueError, match="prefix end exceeds"):
        core._encode_with_character_offsets_prefixes(text, [len(text) + 1], allowed, limit)


def test_first_ineligible_prefix_precedes_later_bad_end():
    core, decoder, allowed = model()
    assert core._encode_with_character_offsets_prefixes("abc", [1, 999], allowed, 0) == [
        expected(core, decoder, "a", allowed)
    ]


def test_concurrent_results_are_independent():
    core, decoder, allowed = model()
    text = "aéS😀 abc<S>long  \n" * 3
    ends = [len(text), 1, 5, len(text)]
    wanted = reference(core, decoder, text, ends, allowed, 100000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(
            pool.map(
                lambda _: core._encode_with_character_offsets_prefixes(text, ends, allowed, 100000),
                range(32),
            )
        )
    assert actual == [wanted] * 32
    actual[0][0][0].append(999)
    assert actual[1] == wanted


@pytest.mark.parametrize("guarded", [True, False])
def test_prefix_values_are_dense_independently_mutable_lists(guarded):
    core, decoder, allowed = model(guarded)
    text = "abc é😀 abc <S>long  \r\n" * 4
    ends = [len(text), len(text), len(text) - 3, 1, len(text), 0]
    wanted = reference(core, decoder, text, ends, allowed, 100000)
    actual = core._encode_with_character_offsets_prefixes(text, ends, allowed, 100000)
    assert actual == wanted
    assert type(actual) is list
    for ids, offsets in actual:
        assert type(ids) is list and type(offsets) is list
        assert all(type(value) is int for value in ids)
        assert all(
            type(span) is tuple and all(type(value) is int for value in span)
            for span in offsets
        )
    assert all(type(row) is tuple for row in actual)
    assert len({id(row[0]) for row in actual}) == len(actual)
    assert len({id(row[1]) for row in actual}) == len(actual)
    actual[0][0][0] = -1
    actual[0][1][0] = (-1, -1)
    actual[0][0].append(999)
    actual[0][1].clear()
    assert actual[1:] == wanted[1:]
    assert core._encode_with_character_offsets_prefixes(text, ends, allowed, 100000) == wanted


@pytest.mark.parametrize("size", [65536, 65537])
def test_size_boundary_retains_independent_prefixes(size):
    core, decoder, allowed = model()
    text = ("abc " * (size // 4 + 1))[:size]
    ends = [1, 33, 64, size, size]
    assert core._encode_with_character_offsets_prefixes(text, ends, allowed, 100000) == reference(
        core, decoder, text, ends, allowed, 100000
    )


def test_partial_vocabulary_preserves_first_error_and_limit_order():
    pattern = (Path(__file__).parents[1] / "src/rmcp_pattern.txt").read_text()
    core = _tiktoken.CoreBPE({b"a": 0}, {"<S>": 1}, pattern)
    allowed = frozenset({"<S>"})
    assert core._encode_with_character_offsets_prefixes("a!", [1, 2, 999], allowed, 0) == [
        ([0], [(0, 1)])
    ]
    with pytest.raises(BaseException) as scalar:
        core._encode_with_character_offsets("!", allowed)
    with pytest.raises(type(scalar.value)) as actual:
        core._encode_with_character_offsets_prefixes("a!", [1, 2, 999], allowed, 100)
    assert str(actual.value) == str(scalar.value)
