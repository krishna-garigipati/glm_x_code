import logging
import pickle
from typing import Any, Tuple

from .errors import SerializationFailedError

logger = logging.getLogger("glmx.serializer")


def _check_lz4() -> bool:
    try:
        import lz4.frame  # noqa: F401
        return True
    except Exception:
        return False


def _check_zstd() -> bool:
    try:
        import zstandard  # noqa: F401
        return True
    except Exception:
        return False


_LZ4_AVAILABLE = _check_lz4()
_ZSTD_AVAILABLE = _check_zstd()


class GraphSerializer:
    def __init__(self, compression: str, compression_level: int) -> None:
        self.compression, self.compression_level = self._resolve(compression, compression_level)

    @staticmethod
    def _resolve(compression: str, level: int) -> Tuple[str, int]:
        """Gracefully fall back to plain pickle when a configured codec is
        missing from the environment (e.g. lz4 absent). Round-trip stays
        symmetric because both compress and decompress use the resolved codec."""
        if compression == "lz4" and not _LZ4_AVAILABLE:
            logger.warning("lz4 unavailable in this environment; falling back to compression='none'")
            return "none", level
        if compression == "zstd" and not _ZSTD_AVAILABLE:
            logger.warning("zstandard unavailable in this environment; falling back to compression='none'")
            return "none", level
        return compression, level

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
