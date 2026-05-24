import time, json, logging
from pathlib import Path
from typing import Optional
from .base import DataLoader, LoadedData
logger = logging.getLogger(__name__)


class JsonLoader(DataLoader):
    def detect_format(self, path: str) -> str:
        return "json"

    def load(self, path: str, **kwargs) -> LoadedData:
        t0 = time.time()
        path = Path(path)
        max_entries = kwargs.get("max_entries", None)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        if max_entries:
            data = data[:max_entries]
        triples, sentences = [], []
        for item in data:
            if all(k in item for k in ("head", "relation", "tail")):
                triples.append((item["head"], item["relation"], item["tail"]))
            elif all(k in item for k in ("subject", "predicate", "object")):
                triples.append((item["subject"], item["predicate"], item["object"]))
            elif "sentence" in item:
                sentences.append(item["sentence"])
        elapsed = time.time() - t0
        logger.info(f"JsonLoader: {len(triples)} triples, {len(sentences)} sentences in {elapsed:.2f}s")
        return LoadedData(sentences=sentences, triples=triples,
                         metadata={"source": str(path), "format": "json",
                                   "num_entries": len(data), "load_time_seconds": round(elapsed, 2)})
