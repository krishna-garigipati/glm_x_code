import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class LoadedData:
    def __init__(
        self,
        sentences: Optional[List[str]] = None,
        triples: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
    ):
        self.sentences = sentences or []
        self.triples = triples or []
        self.documents = documents or []
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"LoadedData(documents={len(self.documents)}, "
            f"sentences={len(self.sentences)}, "
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

    @staticmethod
    def auto_detect_format(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".json":
            return "json"
        if ext == ".jsonl":
            return "json"
        if ext == ".csv":
            return "csv"
        if ext == ".tsv":
            return "csv"
        if ext == ".parquet":
            return "parquet"
        if ext == ".db":
            return "sqlite"
        if ext in (".txt", ".text"):
            return "text"
        raise ValueError(f"Cannot auto-detect format for: {path}")


def load_data(path: str, format: Optional[str] = None,
              start: int = 0, end: Optional[int] = None, **kwargs) -> LoadedData:
    """Load a slice of data from any supported format."""
    if format is None:
        format = DataLoader.auto_detect_format(path)

    if format == "text":
        return _load_text_documents(path, start, end, **kwargs)

    loader = DataLoader.create(format)

    if format in ("json",):
        data = loader.load(path, start=start, end=end, **kwargs)
    else:
        data = loader.load(path, **kwargs)
        if end is not None:
            if data.documents:
                data.documents = data.documents[start:end]
            elif data.sentences:
                data.sentences = data.sentences[start:end]
            elif data.triples:
                data.triples = data.triples[start:end]
        elif start > 0:
            if data.documents:
                data.documents = data.documents[start:]
            elif data.sentences:
                data.sentences = data.sentences[start:]
            elif data.triples:
                data.triples = data.triples[start:]

    return data


def _load_text_documents(path: str, start: int = 0,
                         end: Optional[int] = None, **kwargs) -> LoadedData:
    import time
    t0 = time.time()
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            line = line.strip()
            if line:
                documents.append(line)
    elapsed = time.time() - t0
    logger = logging.getLogger("data_loader")
    logger.info(f"TextLoader: {len(documents)} documents from {path} in {elapsed:.2f}s")
    return LoadedData(
        documents=documents,
        metadata={"source": str(path), "format": "text", "num_entries": len(documents),
                  "load_time_seconds": round(elapsed, 2)},
    )
