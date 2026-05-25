import time
import json
import logging
from pathlib import Path
from typing import List, Optional

from .base import DataLoader, LoadedData

logger = logging.getLogger(__name__)


class JsonLoader(DataLoader):
    def detect_format(self, path: str) -> str:
        return "json"

    def _is_jsonl(self, path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunk = f.read(4096)
                f.seek(0)
                first = chunk.splitlines()[0] if chunk.splitlines() else ""
                return first.lstrip() and not first.lstrip().startswith("[")
        except Exception:
            return False

    def load(self, path: str, **kwargs) -> LoadedData:
        t0 = time.time()
        path = Path(path)
        max_entries = kwargs.get("max_entries", None)
        start = kwargs.get("start", 0)
        end = kwargs.get("end", None)

        if self._is_jsonl(path):
            return self._load_jsonl(path, max_entries, t0, start, end)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        if end is not None:
            data = data[start:end]
        elif start > 0:
            data = data[start:]

        if max_entries:
            data = data[:max_entries]

        return self._parse_items(data, path, t0)

    def _parse_items(self, items: list, path: Path, t0: float) -> LoadedData:
        triples = []
        sentences = []
        documents = []
        for item in items:
            if isinstance(item, dict):
                if all(k in item for k in ("head", "relation", "tail")):
                    triples.append((item["head"], item["relation"], item["tail"]))
                elif all(k in item for k in ("subject", "predicate", "object")):
                    triples.append((item["subject"], item["predicate"], item["object"]))
                elif "sentence" in item:
                    sentences.append(item["sentence"])
                else:
                    for doc_key in ("article", "document", "text"):
                        val = item.get(doc_key)
                        if val and isinstance(val, str) and len(val) > 20:
                            documents.append(val)
                            break
            elif isinstance(item, str):
                documents.append(item)

        if not documents and not triples and not sentences:
            for item in items:
                if isinstance(item, str):
                    documents.append(item)

        elapsed = time.time() - t0
        logger.info(
            f"JsonLoader: {len(documents)} documents, {len(sentences)} sentences, "
            f"{len(triples)} triples from {path} in {elapsed:.2f}s"
        )

        metadata = {
            "source": str(path),
            "format": "json",
            "num_entries": len(items),
            "load_time_seconds": round(elapsed, 2),
        }

        return LoadedData(
            documents=documents,
            sentences=sentences,
            triples=triples,
            metadata=metadata,
        )

    def _load_jsonl(self, path: Path, max_entries, t0: float,
                     start: int = 0, end: Optional[int] = None) -> LoadedData:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                if end is not None and i >= end:
                    break
                if max_entries is not None and len(items) >= max_entries:
                    break
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        items.append(line)
        return self._parse_items(items, path, t0)
