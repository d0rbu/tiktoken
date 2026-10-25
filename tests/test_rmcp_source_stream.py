"""Bounded source batches, before-encoding replay and shared-Core concurrency."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from tiktoken import _tiktoken


@pytest.fixture
def core():
    return _tiktoken.CoreBPE({bytes([i]): i for i in range(256)}, {"<|s|>": 256}, "(?s).")


def encoded(core, cursor, provenance=False):
    output = []
    lengths = []
    source = []
    while (batch := cursor.read_batch(provenance)) is not None:
        buffer, sizes = core._encode_source_batch(batch, {"<|s|>"}, 257)
        with memoryview(buffer) as view:
            output.extend(view.tolist())
        lengths.extend(sizes)
        source.extend(batch.provenance)
    return output, lengths, source


def test_all_range_cuts_preserve_ids_lengths_and_raw_metadata(core, tmp_path):
    path = tmp_path / "source"
    text = "é😀<|s|>\n"
    row = {
        "split": "train",
        "text": text,
        "token_provenance": {"original_source_tokens": 2**53 + 1},
    }
    data = (
        json.dumps(row, ensure_ascii=False)
        + '\r\n{"split":"eval","text":null}\n'
        + json.dumps(row, ensure_ascii=False)
    ).encode()
    path.write_bytes(data)
    reference = core.encode(text, {"<|s|>"}) + [257]
    for width in (1, 7, 64, 1024):
        ids, lengths, provenance = [], [], []
        for start in range(0, len(data), width):
            with closing(
                core._open_source(str(path), "train", start, min(start + width, len(data)))
            ) as cursor:
                output = encoded(core, cursor, True)
                ids.extend(output[0])
                lengths.extend(output[1])
                provenance.extend(output[2])
        assert ids == reference * 2
        assert lengths == [len(reference)] * 2
        assert [json.loads(p) for p in provenance] == [row["token_provenance"]] * 2


@pytest.mark.parametrize(
    "bad",
    [
        b"{broken}\n",
        b"[]\n",
        b'{"split":"train","text":"\\ud800"}\n',
        b'{"split":"train","text":null}\n',
        b'{"text":"missing split"}\n',
        b'{"split":"train","text":"bad\xff"}\n',
    ],
)
def test_decline_retains_exact_unencoded_current_batch_and_suffix(core, tmp_path, bad):
    path = tmp_path / "source"
    good = b'{"split":"train","text":"ok"}\n'
    suffix = b'{"split":"eval"}\n'
    path.write_bytes(good * 65 + bad + suffix)
    with closing(core._open_source(str(path), "train", 0)) as cursor:
        first = cursor.read_batch(False)
        buffer, lengths = core._encode_source_batch(first, set(), 257)
        assert lengths == [3] * 64
        assert memoryview(buffer).tolist() == [111, 107, 257] * 64
        with pytest.raises(NotImplementedError, match="before encoding"):
            cursor.read_batch(False)
        assert b"".join(cursor) == good + bad + suffix


def test_empty_filtered_missing_provenance_and_explicit_close(core, tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b'{"split":"eval","text":null}\n')
    with closing(core._open_source(str(path), "train", 0)) as cursor:
        assert cursor.read_batch(True) is None
    path.write_bytes(b'{"split":"train","text":""}\n')
    cursor = core._open_source(str(path), "train", 0)
    output = encoded(core, cursor, True)
    assert output == ([257], [1], [None])
    cursor.close()
    cursor.close()
    assert cursor.read_batch(False) is None
    assert list(cursor) == []


def test_open_error_preserves_python_errno_strerror_and_filename(core, tmp_path):
    path = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError) as python_error:
        open(path, "rb")
    with pytest.raises(FileNotFoundError) as native_error:
        core._open_source(path, "train", 0)
    assert native_error.value.args == python_error.value.args
    assert native_error.value.filename == python_error.value.filename
    assert native_error.value.strerror == python_error.value.strerror
    assert str(native_error.value) == str(python_error.value)


def test_shared_core_has_independent_bounded_cursors_across_threads(core, tmp_path):
    path = tmp_path / "source"
    texts = ["a" * 17, "b" * (1024 * 1024), "é😀<|s|>"]
    path.write_text("\n".join(json.dumps({"split": "train", "text": text}) for text in texts))
    expected = [token for text in texts for token in [*core.encode(text, {"<|s|>"}), 257]]

    def worker(_):
        with closing(core._open_source(str(path), "train", 0)) as cursor:
            result = encoded(core, cursor)
        assert result[0] == expected
        assert result[1] == [18, 1024 * 1024 + 1, 8]
        assert result[2] == []

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(worker, range(32)))
