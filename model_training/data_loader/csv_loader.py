import time, csv, logging
from pathlib import Path
from typing import Optional
from .base import DataLoader, LoadedData
logger = logging.getLogger(__name__)


class CsvLoader(DataLoader):
    def detect_format(self, path: str) -> str:
        return "csv"

    def load(self, path: str, **kwargs) -> LoadedData:
        t0 = time.time()
        path = Path(path)
        max_entries = kwargs.get("max_entries", None)
        triples, sentences = [], []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_entries is not None and i >= max_entries:
                    break
                if all(k in row for k in ("head", "relation", "tail")):
                    triples.append((row["head"], row["relation"], row["tail"]))
                elif all(k in row for k in ("subject", "predicate", "object")):
                    triples.append((row["subject"], row["predicate"], row["object"]))
                elif "sentence" in row:
                    sentences.append(row["sentence"])
        elapsed = time.time() - t0
        logger.info(f"CsvLoader: {len(triples)} triples, {len(sentences)} sentences in {elapsed:.2f}s")
        return LoadedData(sentences=sentences, triples=triples,
                         metadata={"source": str(path), "format": "csv",
                                   "num_entries": len(triples) + len(sentences),
                                   "load_time_seconds": round(elapsed, 2)})
