import time
import sqlite3
import logging
from pathlib import Path
from typing import Optional

from .base import DataLoader, LoadedData

logger = logging.getLogger(__name__)


class SqliteLoader(DataLoader):
    def detect_format(self, path: str) -> str:
        return "sqlite"

    def load(self, path: str, **kwargs) -> LoadedData:
        t0 = time.time()
        path = Path(path)

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}

        triples = []
        sentences = []
        metadata_info = {}

        if "nodes" in tables and "edges" in tables:
            cursor.execute("SELECT n1.label AS src, e.relation, n2.label AS tgt "
                          "FROM edges e "
                          "JOIN nodes n1 ON e.source_id = n1.id "
                          "JOIN nodes n2 ON e.target_id = n2.id")
            for row in cursor.fetchall():
                triples.append((row["src"], row["relation"], row["tgt"]))

            if "metadata" in tables:
                cursor.execute("SELECT key, value FROM metadata")
                for row in cursor.fetchall():
                    metadata_info[row["key"]] = row["value"]

        if "sentences" in tables:
            cursor.execute("SELECT text FROM sentences")
            for row in cursor.fetchall():
                sentences.append(row["text"])

        conn.close()

        elapsed = time.time() - t0
        logger.info(
            f"SqliteLoader: {len(triples)} triples, {len(sentences)} sentences "
            f"from {path} in {elapsed:.2f}s"
        )

        metadata = {
            "source": str(path),
            "format": "sqlite",
            "num_entries": len(triples) + len(sentences),
            "load_time_seconds": round(elapsed, 2),
            **metadata_info,
        }

        return LoadedData(
            sentences=sentences,
            triples=triples,
            metadata=metadata,
        )
