"""UTF-8 preparation converts the whole batch without invoking tokenization."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tiktoken.core import _prepare_utf8


class _Text(str):
    def encode(self, *_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("preparation must use the original Unicode value")


@given(
    st.lists(st.text(alphabet=st.characters(exclude_categories=("Cs",))), max_size=12)
)
@settings(max_examples=100, deadline=None)
def test_valid_batch_preparation_has_no_result(texts: list[str]) -> None:
    assert _prepare_utf8(texts) is None
    assert _prepare_utf8(tuple(texts)) is None


@pytest.mark.parametrize("point", range(0xD800, 0xE000))
def test_surrogate_anywhere_rejects_complete_batch(point: int) -> None:
    bad = chr(point)
    for texts in ((bad,), ("valid", bad), ("valid", "before" + bad + "after", "tail")):
        with pytest.raises(UnicodeEncodeError):
            _prepare_utf8(texts)


def test_subclasses_use_the_builtin_unicode_storage() -> None:
    assert _prepare_utf8((_Text("ASCII\0é🙂"),)) is None
    with pytest.raises(UnicodeEncodeError):
        _prepare_utf8((_Text("before\ud800after"),))
