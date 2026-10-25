"""Private CPU storage conversion for producer-admitted little-endian uint32 IDs.

The caller owns vocabulary-width admission and complete aligned input frames.
Only a successful source/shard end calls finish. Decoder views own their buffers;
neither conversion shares mutable output with a later call.
"""

from tiktoken import _tiktoken


class TokenStorageEncoder:
    def __init__(self, width: int) -> None:
        self._encoder = _tiktoken._TokenStorageEncoder(width)

    def push(self, raw: bytes) -> bytes:
        return self._encoder.push(raw)

    def finish(self) -> bytes:
        return self._encoder.finish()


class TokenStorageDecoder:
    def __init__(self, width: int) -> None:
        self._decoder = _tiktoken._TokenStorageDecoder(width)

    def read(self, data: bytes, bit_offset: int, count: int) -> memoryview:
        return memoryview(self._decoder.read(data, bit_offset, count))
