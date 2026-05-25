import time
import csv
import logging
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

        triples = []
        sentences = []
        documents = []

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
                else:
                    for doc_key in ("article", "document", "text"):
                        val = row.get(doc_key)
                        if val and len(val) > 20:
                            documents.append(val)
                            break

        elapsed = time.time() - t0
        logger.info(
            f"CsvLoader: {len(documents)} documents, {len(sentences)} sentences, "
            f"{len(triples)} triples from {path} in {elapsed:.2f}s"
        )

        metadata = {
            "source": str(path),
            "format": "csv",
            "num_entries": len(triples) + len(sentences) + len(documents),
            "load_time_seconds": round(elapsed, 2),
        }

        return LoadedData(
            documents=documents,
            sentences=sentences,
            triples=triples,
            metadata=metadata,
        )
