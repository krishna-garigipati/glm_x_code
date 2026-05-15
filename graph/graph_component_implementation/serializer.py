import pickle
from typing import Any

from .errors import SerializationFailedError


class GraphSerializer:
    def __init__(self, compression: str, compression_level: int) -> None:
        self.compression = compression
        self.compression_level = compression_level

    def serialize(self, obj: Any) -> bytes:
        try:
            payload = pickle.dumps(obj, protocol=4)
            return self._compress(payload)
        except Exception as exc:
            raise SerializationFailedError(str(exc)) from exc

    def deserialize(self, data: bytes) -> Any:
        try:
            payload = self._decompress(data)
            return pickle.loads(payload)
        except Exception as exc:
            raise SerializationFailedError(str(exc)) from exc

    def _compress(self, data: bytes) -> bytes:
        if self.compression == "none":
            return data
        if self.compression == "lz4":
            try:
                import lz4.frame
            except Exception as exc:
                raise SerializationFailedError("lz4 not available") from exc
            return lz4.frame.compress(data, compression_level=self.compression_level)
        if self.compression == "zstd":
            try:
                import zstandard as zstd
            except Exception as exc:
                raise SerializationFailedError("zstd not available") from exc
            cctx = zstd.ZstdCompressor(level=self.compression_level)
            return cctx.compress(data)
        raise SerializationFailedError(f"Unknown compression: {self.compression}")

    def _decompress(self, data: bytes) -> bytes:
        if self.compression == "none":
            return data
        if self.compression == "lz4":
            try:
                import lz4.frame
            except Exception as exc:
                raise SerializationFailedError("lz4 not available") from exc
            return lz4.frame.decompress(data)
        if self.compression == "zstd":
            try:
                import zstandard as zstd
            except Exception as exc:
                raise SerializationFailedError("zstd not available") from exc
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        raise SerializationFailedError(f"Unknown compression: {self.compression}")
