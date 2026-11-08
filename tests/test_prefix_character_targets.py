"""Consumed target bounds match independent scalar offsets without Python offset lists."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given, settings, strategies as st

from .test_prefix_character_offsets import model, reference


def wanted(core, decoder, text, spans, allowed, limit):
    rows = reference(core, decoder, text, [end for _, end in spans], allowed, limit)
    return [
        (
            ids,
            next(
                (i for i, (_, end) in enumerate(offsets) if end > start), len(offsets)
            ),
            sum(begin < stop for begin, _ in offsets),
        )
        for (ids, offsets), (start, stop) in zip(rows, spans)
    ]


@pytest.mark.parametrize("guarded", [True, False])
@pytest.mark.parametrize("policy", ["all", "partial", "none"])
@settings(max_examples=100, deadline=None)
@given(text=st.text(st.characters(exclude_categories=("Cs",)), max_size=100))
def test_all_character_targets_match_scalar_offsets(text, guarded, policy):
    core, decoder, allowed = model(guarded)
    allowed = (
        allowed
        if policy == "all"
        else frozenset({"<S>", "aba"})
        if policy == "partial"
        else frozenset()
    )
    spans = [(max(0, end - 1), end) for end in range(len(text) + 1)]
    spans += [(len(text), len(text)), (0, len(text)), (len(text) + 1, len(text))]
    for order in (spans, list(reversed(spans))):
        assert core._encode_with_character_targets_prefixes(
            text, order, allowed, 100000
        ) == wanted(core, decoder, text, order, allowed, 100000)


@pytest.mark.parametrize("limit", [0, 1, 4, 100000])
@pytest.mark.parametrize("text", ["", "aé😀\0<S>long a\r\n", "ababa 're  " * 8])
def test_limits_empty_unsorted_duplicate_and_first_bad_end(text, limit):
    core, decoder, allowed = model()
    for spans in [
        [],
        [(0, 0), (0, len(text) + 1)],
        [(0, len(text)), (0, 1), (0, len(text))],
    ]:
        if len(text) == 0 and len(spans) == 3:
            spans[1] = (0, 0)
        assert core._encode_with_character_targets_prefixes(
            text, spans, allowed, limit
        ) == wanted(core, decoder, text, spans, allowed, limit)
    with pytest.raises(ValueError, match="prefix end exceeds"):
        core._encode_with_character_targets_prefixes(
            text, [(0, len(text) + 1)], allowed, limit
        )


def test_first_ineligible_prefix_precedes_later_bad_end():
    core, decoder, allowed = model()
    spans = [(0, 1), (0, 999)]
    assert core._encode_with_character_targets_prefixes(
        "abc", spans, allowed, 0
    ) == wanted(core, decoder, "abc", spans, allowed, 0)


@pytest.mark.parametrize("guarded", [True, False])
def test_concurrent_ids_are_independent_mutable_lists(guarded):
    core, decoder, allowed = model(guarded)
    text = "aéS😀 abc<S>long  \n" * 3
    spans = [(0, len(text)), (1, len(text)), (0, 1), (3, len(text))]
    expected = wanted(core, decoder, text, spans, allowed, 100000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(
            pool.map(
                lambda _: core._encode_with_character_targets_prefixes(
                    text, spans, allowed, 100000
                ),
                range(16),
            )
        )
    assert actual == [expected] * 16
    for rows in actual:
        assert type(rows) is list
        assert all(type(row) is tuple and type(row[0]) is list for row in rows)
        assert len({id(row[0]) for row in rows}) == len(rows)
        assert all(type(value) is int for row in rows for value in row[0])
    actual[0][0][0].clear()
    actual[0][1][0].append(999)
    assert actual[0][2:] == expected[2:]
    assert actual[1:] == [expected] * 15


@pytest.mark.parametrize("size", [65536, 65537])
def test_existing_prefix_size_boundary(size):
    core, decoder, allowed = model()
    text = ("abc " * (size // 4 + 1))[:size]
    spans = [(0, 1), (2, 33), (32, 64), (size - 8, size), (0, size)]
    assert core._encode_with_character_targets_prefixes(
        text, spans, allowed, 100000
    ) == wanted(core, decoder, text, spans, allowed, 100000)


@pytest.mark.parametrize("guarded", [True, False])
@settings(max_examples=150, deadline=None)
@given(
    text=st.text(st.characters(exclude_categories=("Cs",)), max_size=40),
    data=st.data(),
)
def test_arbitrary_target_starts_and_unsorted_prefixes(text, data, guarded):
    core, decoder, allowed = model(guarded)
    spans = data.draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=len(text) + 10),
                st.integers(min_value=0, max_value=len(text)),
            ),
            max_size=20,
        )
    )
    limit = data.draw(st.integers(min_value=0, max_value=200))
    assert core._encode_with_character_targets_prefixes(
        text, spans, allowed, limit
    ) == wanted(core, decoder, text, spans, allowed, limit)
