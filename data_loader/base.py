from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class LoadedData:
    def __init__(
        self,
        sentences: Optional[List[str]] = None,
        triples: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.sentences = sentences or []
        self.triples = triples or []
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"LoadedData(sentences={len(self.sentences)}, "
            f"triples={len(self.triples)}, "
            f"metadata={self.metadata})"
        )


class DataLoader(ABC):
    @abstractmethod
    def load(self, path: str, **kwargs) -> LoadedData:
        pass

    @abstractmethod
    def detect_format(self, path: str) -> str:
        pass

    @classmethod
    def create(cls, format: str) -> "DataLoader":
        from .parquet_loader import ParquetLoader
        from .json_loader import JsonLoader
        from .csv_loader import CsvLoader
        from .sqlite_loader import SqliteLoader
        loaders = {
            "parquet": ParquetLoader(),
            "json": JsonLoader(),
            "csv": CsvLoader(),
            "sqlite": SqliteLoader(),
        }
        loader = loaders.get(format)
        if loader is None:
            raise ValueError(f"Unknown format: {format}. Supported: {list(loaders.keys())}")
        return loader
