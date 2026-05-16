#!/usr/bin/env python
"""
GLM-X ConceptNet Training Pipeline (Direct Triple Classification)

Each triple (head, relation, tail) is encoded as text via Sentence-BERT.
The target intent is determined by CONCEPTNET_RELATION_MAP[relation].

Usage:
    python model_training\train_conceptnet.py --triples 5000 --epochs 50
"""

import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_conceptnet")

import pyarrow.parquet as pq
from model_training.models.intent_ffn import IntentFFN
from model_training.training.trainer import TrainingRun
from model_training.training.evaluate import run_evaluation
from model_training.dataset_conceptnet.relation_map import CONCEPTNET_RELATION_MAP

DATA_DIR = Path(__file__).parent / "dataset_conceptnet" / "conceptnet" / "data"

RELATION_SHARDS = {
    "Synonym": [8],
    "RelatedTo": [5, 6, 7],
    "Antonym": [0],
}


def concept_label(uri: str) -> str:
    parts = uri.strip("/").split("/")
    name_idx = 5 if len(parts) > 5 else 4
    if len(parts) > name_idx:
        return parts[name_idx].replace("_", " ")
    return uri


def concept_lang(uri: str) -> str:
    parts = uri.strip("/").split("/")
    if len(parts) >= 5 and parts[0] == "http:":
        return parts[4]
    return ""


def load_english_triples(
    max_triples: int = 5000,
    english_only: bool = True,
) -> list:
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    index_to_file = {int(f.stem.split("-")[1]): f for f in parquet_files}

    all_triples = []
    per_rel = max(1, max_triples * 2 // len(RELATION_SHARDS))

    for rel_name, shard_indices in RELATION_SHARDS.items():
        needed = per_rel
        for sidx in shard_indices:
            if needed <= 0:
                break
            fpath = index_to_file.get(sidx)
            if fpath is None:
                continue
            pf = pq.ParquetFile(fpath)
            for gi in range(pf.metadata.num_row_groups):
                if needed <= 0:
                    break
                tbl = pf.read_row_groups([gi], columns=["subject", "predicate", "object"])
                for row in tbl.to_pylist():
                    head_lang = concept_lang(row["subject"])
                    tail_lang = concept_lang(row["object"])
                    if english_only and (head_lang != "en" or tail_lang != "en"):
                        continue
                    head = concept_label(row["subject"])
                    tail = concept_label(row["object"])
                    all_triples.append({
                        "head": head,
                        "relation": rel_name,
                        "tail": tail,
                        "head_lang": head_lang,
                        "tail_lang": tail_lang,
                    })
                    needed -= 1
                    if needed <= 0:
                        break

    rng = np.random.RandomState(42)
    rng.shuffle(all_triples)
    all_triples = all_triples[:max_triples]
    logger.info(
        f"Loaded {len(all_triples)} English ConceptNet triples "
        f"(out of {max_triples * 2} sampled): "
        + str(dict(Counter(t["relation"] for t in all_triples).most_common()))
    )
    return all_triples


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLM-X on ConceptNet")
    parser.add_argument("--triples", type=int, default=5000, help="ConceptNet triples to load")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="saved_models/conceptnet", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).parent.resolve()
    output_dir = base_dir / args.output_dir
    unified_metrics_path = base_dir / "unified_metrics.json"
    config_path = base_dir / "config.yaml"
    step_start = time.time()

    # Step 1: Load English triples
    logger.info("=" * 60)
    logger.info("STEP 1/4: Loading English ConceptNet triples")
    logger.info("=" * 60)
    triples = load_english_triples(max_triples=args.triples)

    # Step 2: Encode triples directly via Sentence-BERT
    logger.info("=" * 60)
    logger.info("STEP 2/4: Encoding triples via Sentence-BERT")
    logger.info("=" * 60)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [f"{t['head']} {t['relation']} {t['tail']}" for t in triples]
    logger.info(f"Encoding {len(texts)} triples (sample: '{texts[0]}')")

    X = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    y = np.array([CONCEPTNET_RELATION_MAP[t["relation"]] for t in triples], dtype=np.int64)
    logger.info(f"Encoded {len(X)}/{len(triples)} triples")

    intent_dist = np.bincount(y, minlength=16).tolist()
    logger.info(f"Intent distribution: {intent_dist}")

    # Step 3: Split & Train
    logger.info("=" * 60)
    logger.info("STEP 3/4: Training IntentFFN")
    logger.info("=" * 60)
    n = len(X)
    indices = np.random.RandomState(42).permutation(n)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    X_train, y_train = X[indices[:train_end]], y[indices[:train_end]]
    X_val, y_val = X[indices[train_end:val_end]], y[indices[train_end:val_end]]
    X_test, y_test = X[indices[val_end:]], y[indices[val_end:]]
    logger.info(f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    intent_ffn = IntentFFN()
    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0001,
        "loss": "focal",
        "focal_gamma": 2.0,
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": 15,
    }
    training_run = TrainingRun(
        model=intent_ffn,
        dataset_name="conceptnet",
        config=train_config,
        output_dir=output_dir,
    )
    metrics = training_run.run(X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 4: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 4/4: Evaluating")
    logger.info("=" * 60)
    eval_metrics = run_evaluation(intent_ffn, X_test, y_test)
    metrics["evaluation"] = eval_metrics
    metrics["dataset_stats"] = {
        "total_triples": len(triples),
        "encoded": len(X),
        "intent_distribution": intent_dist,
    }
    metrics["total_pipeline_time_seconds"] = round(time.time() - step_start, 2)

    intent_ffn.add_trained_dataset("conceptnet")
    intent_ffn.save(output_dir / "intent_ffn_best.pt", metadata={
        "dataset": "conceptnet",
        "triples": len(triples),
        "test_accuracy": eval_metrics["test_top1_accuracy"],
    })
    training_run.save_results(unified_metrics_path)

    logger.info("=" * 60)
    logger.info("CONCEPTNET TRAINING COMPLETE")
    logger.info(f"  Model: {output_dir}")
    logger.info(f"  Test Top-1: {eval_metrics['test_top1_accuracy']:.4f}")
    logger.info(f"  Test Top-3: {eval_metrics['test_top3_accuracy']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
