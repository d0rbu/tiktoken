"""The guarded ASCII shortcut agrees with the unchanged fancy-regex path."""

from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tiktoken import _tiktoken


@pytest.fixture(scope="module")
def models() -> tuple[_tiktoken.CoreBPE, _tiktoken.CoreBPE]:
    pattern = (Path(__file__).parents[1] / "src/rmcp_pattern.txt").read_text()
    ranks = {bytes([value]): value for value in range(256)}
    ranks.update({b"ab": 256, b"'s": 257, b"  ": 258, b" a": 259, b" '": 260})
    special = {"<S>": 300, "<T>": 301}
    return (_tiktoken.CoreBPE(ranks, special, pattern),
            _tiktoken.CoreBPE(ranks, special, "(?:" + pattern + ")"))


@pytest.mark.parametrize("allowed", [frozenset(), frozenset({"<S>"}), frozenset({"<S>", "<T>"})])
@given(text=st.text(max_size=300))
@settings(max_examples=180, deadline=None)
def test_guarded_matches_original_fancy_path(models: tuple[_tiktoken.CoreBPE, _tiktoken.CoreBPE], allowed: frozenset[str], text: str) -> None:
    guarded, original = models
    for value in (text, " \t 's  're\r\nabc123! " + text + "  <S>\t 've<T>"):
        expected = original.encode(value, allowed)
        assert guarded.encode(value, allowed) == expected
        assert guarded.count_tokens(value, allowed) == len(expected)
