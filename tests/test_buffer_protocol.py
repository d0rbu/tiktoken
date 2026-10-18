"""Shape, request flags, and lifetime of immutable native token buffers."""

import ctypes
import gc
import struct

import numpy as np
import pytest

from tiktoken import _tiktoken


class Buffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


get_buffer = ctypes.pythonapi.PyObject_GetBuffer
get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(Buffer), ctypes.c_int]
get_buffer.restype = ctypes.c_int
release_buffer = ctypes.pythonapi.PyBuffer_Release
release_buffer.argtypes = [ctypes.POINTER(Buffer)]
release_buffer.restype = None


def make_buffers(texts: tuple[str, ...], batch: bool) -> list[object]:
    model = _tiktoken.CoreBPE(
        {bytes([value]): value for value in range(256)}, {"<S>": 300}, r"[\s\S]+"
    )
    if batch:
        return model.encode_to_tiktoken_buffers(texts, {"<S>"})
    return [model.encode_to_tiktoken_buffer(text, {"<S>"}) for text in texts]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("text", ["", "a", "abcd", "é😀\0", "<S>"])
def test_buffer_shape_bytes_and_readonly_consumers(text: str, batch: bool) -> None:
    expected = [300] if text == "<S>" else list(text.encode())
    buffer = make_buffers((text,), batch)[0]
    view = memoryview(buffer)
    # Check bounds before any shaped traversal, including on an old extension.
    assert view.shape == (len(expected),)
    assert view.itemsize == 4 and view.nbytes == 4 * len(expected)
    assert view.format == "I" and view.ndim == 1 and view.strides == (4,)
    assert view.readonly and view.c_contiguous and view.f_contiguous
    assert view.tolist() == expected
    assert view.tobytes() == struct.pack(f"={len(expected)}I", *expected)
    assert np.asarray(buffer).tolist() == expected
    array = np.frombuffer(buffer, dtype=np.uint32)
    assert array.tolist() == expected
    assert isinstance(array.base, memoryview) and array.base.obj is buffer
    assert not np.asarray(buffer).flags.writeable
    with pytest.raises(TypeError):
        view[0:0] = b""


# All valid readonly request forms: SIMPLE, ND, STRIDES, C/F/ANY contiguous,
# and INDIRECT, with their defined optional FORMAT combinations.
@pytest.mark.parametrize(
    "flags", [0, 8, 12, 24, 28, 56, 60, 88, 92, 152, 156, 280, 284]
)
@pytest.mark.parametrize("text", ["", "a", "abcd"])
def test_c_buffer_request_metadata(flags: int, text: str) -> None:
    buffer = make_buffers((text,), True)[0]
    view = Buffer()
    assert get_buffer(buffer, ctypes.byref(view), flags) == 0
    try:
        assert view.len == 4 * len(text) and view.itemsize == 4
        assert view.readonly == 1 and view.ndim == 1 and view.obj
        assert view.format == (b"I" if flags & 4 else None)
        assert bool(view.shape) == bool(flags & 8)
        if view.shape:
            assert view.shape[0] == len(text)
        assert bool(view.strides) == (flags & 24 == 24)
        if view.strides:
            assert view.strides[0] == 4
        assert not view.suboffsets and not view.internal
    finally:
        release_buffer(ctypes.byref(view))
    assert not view.obj


@pytest.mark.parametrize(
    "flags", [1, 9, 13, 25, 29, 57, 61, 89, 93, 153, 157, 281, 285]
)
def test_c_buffer_rejects_writable_requests(flags: int) -> None:
    view = Buffer()
    with pytest.raises(BufferError, match="Object is not writable"):
        get_buffer(make_buffers(("a",), True)[0], ctypes.byref(view), flags)
    assert not view.obj


@pytest.mark.parametrize("batch", [False, True])
def test_views_retain_independent_owners_after_other_exports_close(batch: bool) -> None:
    buffers = make_buffers(("abc", "x"), batch)
    views = [memoryview(buffer) for buffer in buffers]
    assert [view.shape for view in views] == [(3,), (1,)]
    first = memoryview(buffers[0])
    rows = [np.asarray(buffer) for buffer in buffers]
    views[0].release()
    del buffers
    gc.collect()
    assert first.tolist() == [97, 98, 99]
    assert views[1].tolist() == [120]
    assert [row.tolist() for row in rows] == [[97, 98, 99], [120]]
    first.release()
    views[1].release()
    gc.collect()
    assert [row.tolist() for row in rows] == [[97, 98, 99], [120]]


def test_c_buffer_metadata_survives_struct_relocation() -> None:
    buffer = make_buffers(("abc",), True)[0]
    original = Buffer()
    assert get_buffer(buffer, ctypes.byref(original), 284) == 0
    # Transfer ownership of the one export to the copied struct. Do not release
    # the original: that would invalidate both copies of the same export.
    relocated = Buffer.from_buffer_copy(original)
    ctypes.memset(ctypes.byref(original), 0, ctypes.sizeof(original))
    del original, buffer
    gc.collect()
    try:
        assert relocated.shape[0] == 3
        assert relocated.strides[0] == 4
        assert relocated.len == 12 and relocated.format == b"I"
        assert ctypes.string_at(relocated.buf, relocated.len) == struct.pack(
            "=3I", 97, 98, 99
        )
    finally:
        release_buffer(ctypes.byref(relocated))
    assert not relocated.obj
