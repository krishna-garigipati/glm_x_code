#!/usr/bin/env python
"""Universal data ingestion entry point.
Any format (.parquet / .json / .csv / .db) -> unified SQLite .db.

Usage:
    python scripts/ingest.py --input dataset.parquet --output kg/dataset.db
    python scripts/ingest.py --input dataset.json --output kg/dataset.db
    python scripts/ingest.py --input dataset.csv --output kg/dataset.db
    python scripts/ingest.py --input dataset.db --output kg/dataset.db
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest")

import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import DataLoader
from kg_builder import KGBuilderPipeline, KGBuilderConfig
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore


def detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".parquet",):
        return "parquet"
    if ext in (".json", ".jsonl"):
        return "json"
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext == ".db":
        return "sqlite"
    raise ValueError(f"Unknown format for {path}")


def ingest(
    input_path: str,
    output_path: str,
    max_entries: Optional[int] = None,
    english_only: bool = True,
    embed_model: str = "BAAI/bge-small-en-v1.5",
):
    t_start = time.time()
    input_path = str(input_path)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fmt = detect_format(input_path)
    logger.info(f"Format detected: {fmt}")

    if fmt == "sqlite":
        import shutil
        shutil.copy2(input_path, output_path)
        logger.info(f"Copied {input_path} -> {output_path}")
        store = SQLiteGraphStore.load_state(output_path)
        elapsed = time.time() - t_start
        logger.info(f"Ingest complete: {store.get_node_count()} nodes, {store.get_edge_count()} edges in {elapsed:.1f}s")
        return store

    loader = DataLoader.create(fmt)
    loaded = loader.load(input_path, max_entries=max_entries)
    logger.info(f"Loaded: {loaded}")

    if loaded.triples:
        logger.info(f"Building KG from {len(loaded.triples)} triples...")
        config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False, embed_merge_threshold=0.92)
        pipeline = KGBuilderPipeline(config)
        store = pipeline.process_and_store_to_db(loaded, db_path=output_path, show_progress=True)
    elif loaded.sentences:
        logger.info(f"Building KG from {len(loaded.sentences)} sentences...")
        config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False, embed_merge_threshold=0.92)
        pipeline = KGBuilderPipeline(config)
        result = pipeline.process_corpus(loaded.sentences, show_progress=True)
        store = pipeline.build_graph_store(result["graph_data"], store_type="sqlite", db_path=output_path)
    else:
        raise ValueError("No triples or sentences found in data")

    logger.info(f"Computing BGE embeddings for {store.get_node_count()} nodes...")
    sbert = SentenceTransformer(embed_model)
    nids = sorted(store._nodes.keys())
    labels = [store._nodes[nid].label for nid in nids]
    embs = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    for nid, emb in zip(nids, embs):
        store._embeddings[nid] = emb.astype(np.float32)
        node = store._nodes[nid]
        store._nodes[nid] = type(node)(
            id=node.id, label=node.label, node_type=node.node_type,
            embedding=emb.astype(np.float32), activation=node.activation,
            use_count=node.use_count, create_time=node.create_time,
        )

    store.set_metadata("dataset_name", Path(input_path).stem)
    store.set_metadata("source", input_path)
    store.set_metadata("format", fmt)
    store.set_metadata("embed_model", embed_model)
    store.set_metadata("build_time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    store.save_state(output_path)

    elapsed = time.time() - t_start
    logger.info(
        f"Ingest complete: {store.get_node_count()} nodes, {store.get_edge_count()} edges "
        f"in {elapsed:.1f}s -> {output_path}"
    )
    return store


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest any dataset into unified .db format")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output .db path")
    parser.add_argument("--max-entries", type=int, default=None, help="Max entries to load")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", help="BGE model name")
    args = parser.parse_args()

    ingest(
        input_path=args.input,
        output_path=args.output,
        max_entries=args.max_entries,
        embed_model=args.embed_model,
    )


if __name__ == "__main__":
    main()
