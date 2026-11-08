"""Consumed target bounds match independent scalar offsets without Python offset lists."""

import json
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
@pytest.mark.parametrize("policy", ["all", "partial", "none"])
@settings(max_examples=100, deadline=None)
@given(text=st.text(st.characters(exclude_categories=("Cs",)), max_size=100))
def test_json_targets_match_complete_scalar_ids_and_observed_metadata(text, guarded, policy):
    core, decoder, allowed = model(guarded)
    allowed = allowed if policy == "all" else frozenset({"<S>"}) if policy == "partial" else frozenset()
    spans = [(max(0, end - 1), end) for end in range(len(text) + 1)]
    spans += [(0, len(text)), (0, len(text)), (len(text) + 1, len(text))]
    for order in (spans, list(reversed(spans))):
        for limit in (0, 4, 100000):
            expected = wanted(core, decoder, text, order, allowed, limit)
            result = core._encode_with_character_targets_prefixes(text, order, allowed, limit, as_json=True)
            assert [(json.loads(payload), first, last) for payload, count, first, last, maximum in result] == expected
            for row, (ids, first, last) in zip(result, expected):
                payload, count, _, _, maximum = row
                assert type(payload) is bytes
                assert payload == json.dumps(ids, separators=(",", ":")).encode()
                assert count == len(ids)
                assert maximum == max(ids, default=None)
            assert core._encode_with_character_targets_prefixes(text, order, allowed, limit) == expected


def test_json_target_guard_order_empty_and_reserved_actual_id():
    from tiktoken import _tiktoken
    core = _tiktoken.CoreBPE({bytes([value]): value for value in range(256)}, {"<S>": 100295}, r"[\s\S]+")
    rows = core._encode_with_character_targets_prefixes("abc<S>d", [(0, 1), (1, 3), (3, 6), (6, 7)], frozenset({"<S>"}), 100, as_json=True)
    assert [row[4] for row in rows] == [97, 99, 100295, 100295]
    assert all(row[1] == len(json.loads(row[0])) and row[4] == max(json.loads(row[0])) for row in rows)
    assert core._encode_with_character_targets_prefixes("a", [], frozenset(), 100, as_json=True) == []
    with pytest.raises(ValueError, match="prefix end exceeds"):
        core._encode_with_character_targets_prefixes("a", [(0, 2)], frozenset(), 100, as_json=True)
    expected = wanted(core, {i: bytes([i]) for i in range(256)}, "abc", [(0, 1), (0, 999)], frozenset(), 0)
    assert [(json.loads(row[0]),row[2],row[3]) for row in core._encode_with_character_targets_prefixes("abc", [(0, 1), (0, 999)], frozenset(), 0, as_json=True)] == expected


@pytest.mark.parametrize(
    "special,text,spans,expected",
    [
        ("<S>", "aaa", [(3, 3)], [([97, 97, 97], 3, 1)]),
        ("<S>", "aéa😀a", [(6, 5)], [([97, 195, 169, 97, 240, 159, 152, 128, 97], 9, 9)]),
        ("<S>", "aaaa", [(0, 3), (3, 3), (0, 4)],
         [([97, 97, 97], 0, 1), ([97, 97, 97], 3, 1), ([97, 97, 97, 97], 0, 4)]),
        ("a", "aéa😀a", [(0, 5), (3, 5)],
         [([97, 195, 169, 97, 240, 159, 152, 128, 97], 0, 9),
          ([97, 195, 169, 97, 240, 159, 152, 128, 97], 4, 9)]),
    ],
)
def test_direct_targets_keep_original_decoder_alias_geometry(special, text, spans, expected):
    from tiktoken import _tiktoken

    core = _tiktoken.CoreBPE(
        {bytes([value]): value for value in range(256)}, {special: 97}, r"[\s\S]+"
    )
    # Captured from prechange 126: aliased decoder spans can be nonmonotone,
    # so preserve the original Rust partition_point result rather than linear counts.
    for as_json in (False, True):
        actual = core._encode_with_character_targets_prefixes(
            text, spans, frozenset(), 100000, as_json=as_json
        )
        if as_json:
            actual = [(json.loads(row[0]), row[2], row[3]) for row in actual]
        assert actual == expected


@pytest.mark.parametrize("guarded", [True, False])
@pytest.mark.parametrize("limit", [0, 1, 7, 100000])
def test_suffix_projection_matches_full_offsets_for_every_start(guarded, limit):
    core, decoder, allowed = model(guarded)
    text = "aé😀 ababa<S>long\0 \r\n" * 4
    spans = [(start, end) for end in range(len(text) + 1)
             for start in (0, max(0, end - 1), end, end + 1, len(text) + 2)]
    for ordered in (spans, list(reversed(spans))):
        actual = core._encode_with_character_targets_prefixes(text, ordered, allowed, limit)
        assert actual == wanted(core, decoder, text, ordered, allowed, limit)
        assert core._encode_with_character_targets_prefixes(
            text, ordered, allowed, limit, as_json=True
        ) == [(json.dumps(ids, separators=(",", ":")).encode(), len(ids), first, last,
               max(ids, default=None)) for ids, first, last in actual]
