"""Private model admission preserves the original Python loop contracts."""

from types import SimpleNamespace

import pytest

from tiktoken import _tiktoken


def alphabet() -> dict[str, int]:
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = dict(zip(visible, visible))
    for byte in range(256):
        if byte not in codes:
            codes[byte] = 256 + sum(c >= 256 for c in codes.values())
    return {chr(code): byte for byte, code in codes.items()}


@pytest.mark.parametrize("value", [True, -1, 2**32, 1.0])
def test_model_admission_rejects_non_uint32_ids(value: object) -> None:
    vocab = alphabet()
    vocab["a"] = value
    assert _tiktoken._admit_model_words(vocab, [], alphabet(), 4096, 2**24) is None


def test_model_admission_and_byte_conversion_preserve_order() -> None:
    vocab = alphabet()
    vocab.update({"ab": 256, "abc": 257})
    admitted = _tiktoken._admit_model_words(
        vocab, [["a", "b"], ["ab", "c"]], alphabet(), 4096, 2**24
    )
    assert admitted is not None
    copied, words = admitted
    assert copied == vocab and copied is not vocab
    assert list(words)[-2:] == ["ab", "abc"]
    encoded = _tiktoken._model_word_bytes(words)
    assert encoded[b"ab"] == 256 and encoded[b"abc"] == 257
    assert all(encoded[bytes([byte])] == byte for byte in range(256))


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_admission_preserves_size_rejection(limit: int) -> None:
    vocab = alphabet()
    vocab["abc"] = 256
    assert (
        _tiktoken._admit_model_words(vocab, [["a", "bc"]], alphabet(), limit, 256)
        is None
    )


@pytest.mark.parametrize("pair", [[], ["a"], ["a", "b", "c"]])
def test_invalid_merge_arity_preserves_value_error(pair: list[str]) -> None:
    with pytest.raises(ValueError):
        _tiktoken._admit_model_words(alphabet(), [pair], alphabet(), 4096, 2**24)


@pytest.mark.parametrize("quota, expected_calls", [(0, ["a"]), (1, ["a", "ab"])])
def test_word_proof_preserves_quota_and_callback_prefix(
    quota: int, expected_calls: list[str]
) -> None:
    calls = []
    words = {"a": 0, "ab": 256, "abc": 257}

    def tokenize(word: str) -> list[SimpleNamespace]:
        calls.append(word)
        return [SimpleNamespace(id=words[word])]

    assert not _tiktoken._prove_model_words(words, tokenize, quota)
    assert calls == expected_calls


def test_word_proof_stops_at_first_mismatch() -> None:
    calls = []

    def tokenize(word: str) -> list[SimpleNamespace]:
        calls.append(word)
        return [SimpleNamespace(id=-1)]

    assert not _tiktoken._prove_model_words({"a": 0, "ab": 256}, tokenize, 100)
    assert calls == ["a"]


def test_bounded_vocab_and_unknown_word_errors() -> None:
    assert _tiktoken._bounded_model_vocab({"é": 0}, 2, 1, 2)
    assert not _tiktoken._bounded_model_vocab({"é": 0}, 2, 1, 1)
    assert not _tiktoken._bounded_model_vocab({"a": 0, "b": 0}, 2, 2, 100)
    assert not _tiktoken._bounded_model_vocab({"a": True}, 2, 2, 100)
    with pytest.raises(UnicodeEncodeError):
        _tiktoken._bounded_model_vocab({"\ud800": 0}, 2, 2, 100)
    with pytest.raises(KeyError, match="☃"):
        _tiktoken._model_word_bytes({"☃": 0})
