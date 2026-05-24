import time
import logging
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from .base import DataLoader, LoadedData

logger = logging.getLogger(__name__)


def _concept_label(uri: str) -> str:
    parts = uri.strip("/").split("/")
    name_idx = 4 if parts[0] == "http:" else 3
    if len(parts) > name_idx:
        return parts[name_idx].replace("_", " ")
    return uri


def _concept_lang(uri: str) -> str:
    parts = uri.strip("/").split("/")
    return parts[4] if len(parts) >= 5 and parts[0] == "http:" else ""


class ParquetLoader(DataLoader):
    def __init__(self, english_only: bool = True):
        self.english_only = english_only

    def detect_format(self, path: str) -> str:
        return "parquet"

    def load(self, path: str, **kwargs) -> LoadedData:
        t0 = time.time()
        path = Path(path)
        max_entries = kwargs.get("max_entries", None)

        if path.is_dir():
            parquet_files = sorted(path.glob("*.parquet"))
        else:
            parquet_files = [path]

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found at {path}")

        all_triples = []
        skipped_non_en = 0
        seen_pairs = set()

        for fpath in parquet_files:
            if max_entries is not None and len(all_triples) >= max_entries:
                break
            pf = pq.ParquetFile(fpath)
            for gi in range(pf.metadata.num_row_groups):
                if max_entries is not None and len(all_triples) >= max_entries:
                    break
                tbl = pf.read_row_groups([gi], columns=["subject", "predicate", "object"])
                for row in tbl.to_pylist():
                    if max_entries is not None and len(all_triples) >= max_entries:
                        break
                    head_uri = row["subject"]
                    tail_uri = row["object"]
                    rel = row["predicate"]

                    if self.english_only:
                        hl = _concept_lang(head_uri)
                        tl = _concept_lang(tail_uri)
                        if hl != "en" or tl != "en":
                            skipped_non_en += 1
                            continue

                    head = _concept_label(head_uri)
                    tail = _concept_label(tail_uri)
                    pair = (head, rel, tail)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    all_triples.append((head, rel, tail))

        elapsed = time.time() - t0
        logger.info(
            f"ParquetLoader: {len(all_triples)} triples from {len(parquet_files)} files "
            f"in {elapsed:.2f}s (skipped {skipped_non_en} non-en)"
        )
        metadata = {
            "source": str(path), "format": "parquet",
            "num_entries": len(all_triples),
            "load_time_seconds": round(elapsed, 2),
            "skipped_non_english": skipped_non_en,
            "files_loaded": len(parquet_files),
        }
        return LoadedData(sentences=[], triples=all_triples, metadata=metadata)
