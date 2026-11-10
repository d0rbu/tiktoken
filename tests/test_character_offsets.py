from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given, settings, strategies as st
from tiktoken import _tiktoken


def model() -> tuple[_tiktoken.CoreBPE, dict[int, bytes]]:
    ranks = {bytes([value]): value for value in range(256)}
    ranks.update({b"ab": 256, b"abc": 257, b"aa": 258, b"  ": 259})
    specials = {"<S>": 300, "<T>": 301, "<é😀>": 302}
    return _tiktoken.CoreBPE(ranks, specials, r"[\s\S]+"), {
        **{v: k for k, v in ranks.items()},
        **{v: k.encode() for k, v in specials.items()},
    }


def expected(
    core: _tiktoken.CoreBPE,
    decoder: dict[int, bytes],
    text: str,
    allowed: set[str] | frozenset[str],
) -> tuple[list[int], list[tuple[int, int]]]:
    ids = core.encode(text, allowed)
    positions = [i for i, c in enumerate(text) for _ in c.encode()]
    spans = []
    byte = 0
    for token in ids:
        end = byte + len(decoder[token])
        spans.append((positions[byte], positions[end - 1] + 1))
        byte = end
    assert byte == len(text.encode())
    return ids, spans


@pytest.mark.parametrize(
    "text", ["", "abc aa  x", "\0", "é😀", "e\u0301\n中", "<S>é<T><é😀>", "\U0010ffff\x7f\x80"]
)
@pytest.mark.parametrize(
    "allowed", [frozenset(), frozenset({"<S>"}), frozenset({"<S>", "<T>", "<é😀>"})]
)
def test_offsets_and_ids(text: str, allowed: frozenset[str]) -> None:
    core, decoder = model()
    assert core._encode_with_character_offsets(text, allowed) == expected(
        core, decoder, text, allowed
    )


@settings(max_examples=300, deadline=None)
@given(
    st.text(st.characters(exclude_categories=("Cs",)), max_size=300),
    st.sets(st.sampled_from(("<S>", "<T>", "<é😀>"))),
)
def test_unicode_property(text: str, allowed: set[str]) -> None:
    core, decoder = model()
    assert core._encode_with_character_offsets(text, allowed) == expected(
        core, decoder, text, allowed
    )


@pytest.mark.parametrize("bad", ["\ud800", "\udfff", 7, b"abc", None])
def test_argument_failures_preserved(bad: str | int | bytes | None) -> None:
    core, _ = model()
    with pytest.raises((UnicodeEncodeError, TypeError)) as scalar:
        core.encode(bad, frozenset())
    with pytest.raises(type(scalar.value)):
        core._encode_with_character_offsets(bad, frozenset())
    assert core._encode_with_character_offsets("abc", frozenset()) == ([257], [(0, 3)])


def test_shared_model_independent_calls() -> None:
    core, decoder = model()
    inputs = ["é😀<S>", "abc", "", "中\0e\u0301"] * 32
    allowed = frozenset({"<S>"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(
            pool.map(lambda text: core._encode_with_character_offsets(text, allowed), inputs)
        )
    assert actual == [expected(core, decoder, text, allowed) for text in inputs]
