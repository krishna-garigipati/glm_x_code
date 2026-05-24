#!/usr/bin/env python
"""Run full GLM-X pipeline end-to-end on a corpus file.

Usage:
    python _run_pipeline.py --corpus path/to/corpus.txt [--db path/to/output.db]
"""
import sys, json, time, logging, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from kg_builder import KGBuilderPipeline, KGBuilderConfig
from scripts.universal_train import train_on_db, run_qa_eval
from model_training.training.metrics_tracker import MetricsTracker


def load_corpus(path: str) -> list:
    path = Path(path)
    if path.suffix == ".json":
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif path.suffix == ".jsonl":
        import json
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("article", obj.get("text", line.strip())))
        return texts
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run full GLM-X pipeline")
    parser.add_argument("--corpus", "-c", required=True, help="Path to corpus file (.txt, .json, .jsonl)")
    parser.add_argument("--db", "-d", default="checkpoints/unified/graphs/true_facts.db", help="Output .db path")
    parser.add_argument("--checkpoint", default="checkpoints/unified", help="Checkpoint directory")
    parser.add_argument("--sbert", default="BAAI/bge-small-en-v1.5", help="SBERT model name")
    parser.add_argument("--spacy", default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process")
    args = parser.parse_args()

    t_start = time.time()

    # Step 1: Load corpus
    print(f"=== Loading corpus from {args.corpus} ===")
    texts = load_corpus(args.corpus)
    if args.max_docs:
        texts = texts[:args.max_docs]
    print(f"Loaded {len(texts)} texts")

    # Step 2: Build KG
    print("=== Step 1: Build KG ===")
    config = KGBuilderConfig(spaCy_model=args.spacy, verbose=False, embed_merge_threshold=0.92)
    pipeline = KGBuilderPipeline(config)
    result = pipeline.process_corpus(texts, show_progress=True)
    gd = result["graph_data"]
    stats = result["stats"]
    print(f"KG built: {stats['nodes']} nodes, {stats['edges']} edges, {stats['relation_types']} types in {stats['time_seconds']}s")

    # Step 3: Save to SQLite
    print(f"\n=== Step 2: Save to SQLite .db ===")
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    store = pipeline.build_graph_store(gd, store_type="sqlite", db_path=args.db)
    print(f"Saved to {args.db}: {store.get_node_count()} nodes, {store.get_edge_count()} edges")

    # Step 4: Compute BGE embeddings
    print("\n=== Step 3: Compute BGE embeddings ===")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(args.sbert)
    nids = sorted(store._nodes.keys())
    labels = [store._nodes[nid].label for nid in nids]
    embs = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=True)
    for nid, emb in zip(nids, embs):
        store._embeddings[nid] = emb.astype(np.float32)
        node = store._nodes[nid]
        store._nodes[nid] = type(node)(
            id=node.id, label=node.label, node_type=node.node_type,
            embedding=emb.astype(np.float32), activation=node.activation,
            use_count=node.use_count, create_time=node.create_time,
        )
    store.set_metadata("dataset_name", Path(args.corpus).stem)
    store.set_metadata("source", args.corpus)
    store.save_state(args.db)
    print(f"Embeddings computed: {len(store._embeddings)} nodes")

    # Step 5: Train model
    print("\n=== Step 4: Train model (fully adaptive) ===")
    metrics = MetricsTracker(
        run_id=time.strftime("%Y%m%d_%H%M%S") + f"_{Path(args.corpus).stem}",
        checkpoint_dir=args.checkpoint,
    )
    train_result = train_on_db(
        db_path=args.db,
        checkpoint_dir=args.checkpoint,
        sbert_model=args.sbert,
        metrics=metrics,
    )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s total")
    print(json.dumps(train_result, indent=2))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
