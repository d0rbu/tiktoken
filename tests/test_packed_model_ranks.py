import pickle
import struct

import pytest

import tiktoken


def packed(ranks):
    return b"".join(
        struct.pack("<II", rank, len(word)) + word for word, rank in ranks.items()
    )


@pytest.mark.parametrize("extra", [{}, {b"ab": 256, b"abc": 257}])
def test_packed_encoding_preserves_public_operations(extra):
    ranks = {bytes([i]): i for i in range(256)} | extra
    special = {"<one>": 1000, "<two>": 1001}
    normal = tiktoken.Encoding(
        "normal", pat_str=r".+", mergeable_ranks=ranks, special_tokens=special
    )
    actual = tiktoken.Encoding._from_packed_ranks(
        "packed", pat_str=r".+", ranks=packed(ranks), special_tokens=special
    )
    assert "_mergeable_ranks" not in actual.__dict__
    assert actual.max_token_value == normal.max_token_value
    for text in ["abcab", "é🙂", "x<one>y", ""]:
        expected = normal.encode(text, allowed_special="all")
        assert actual.encode(text, allowed_special="all") == expected
        assert actual.decode(expected) == normal.decode(expected)
        assert actual._core_bpe.count_tokens(text, set(special)) == len(expected)
    assert "_mergeable_ranks" not in actual.__dict__
    assert actual._mergeable_ranks == ranks
    assert actual.__dict__["_mergeable_ranks"] is actual._mergeable_ranks
    copied = pickle.loads(pickle.dumps(actual))
    assert copied._mergeable_ranks == ranks
    assert copied.encode("abc") == normal.encode("abc")
    assert normal._mergeable_ranks is ranks


@pytest.mark.parametrize(
    "data",
    [b"", b"1", struct.pack("<II", 0, 2) + b"x", struct.pack("<II", 0, 1) + b"x0"],
)
def test_packed_extent_errors(data):
    with pytest.raises(ValueError, match="packed rank"):
        tiktoken.Encoding._from_packed_ranks(
            "bad", pat_str=".", ranks=data, special_tokens={}
        )


def test_packed_pattern_errors():
    with pytest.raises(ValueError):
        tiktoken.Encoding._from_packed_ranks(
            "bad", pat_str="[", ranks=packed({b"a": 0}), special_tokens={}
        )
