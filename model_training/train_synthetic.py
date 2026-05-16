#!/usr/bin/env python
"""
GLM-X Synthetic Baseline Training Pipeline

Trains the IntentFFN on synthetic data covering all 16 intents.
Serves as the baseline before expanding with real datasets like ConceptNet.

Usage:
    python model_training\train_synthetic.py --samples 5000 --epochs 50
"""

import argparse
import logging
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_synthetic")

from g2p.types import Subgraph
from g2p.train import generate_synthetic_data
from g2p.config import G2PConfig
from model_training.models.glm_x_model import GLMXModel
from model_training.models.intent_ffn import IntentFFN
from model_training.training.trainer import TrainingRun
from model_training.training.evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLM-X on synthetic data")
    parser.add_argument("--samples", type=int, default=5000, help="Number of synthetic samples")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="saved_models/synthetic", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).parent.resolve()
    output_dir = base_dir / args.output_dir
    unified_metrics_path = base_dir / "unified_metrics.json"
    config_path = base_dir / "config.yaml"
    step_start = time.time()

    # Step 1: Generate synthetic data
    logger.info("=" * 60)
    logger.info("STEP 1/4: Generating synthetic training data")
    logger.info("=" * 60)
    g2p_config = G2PConfig.from_yaml(str(config_path))
    g2p_config.training.synthetic_data.num_samples = args.samples
    raw_data = generate_synthetic_data(g2p_config)
    logger.info(f"Generated {len(raw_data)} synthetic (Subgraph, intent_sequence) pairs")

    intent_dist = np.zeros(16, dtype=np.int64)
    for _, seq in raw_data:
        if seq:
            intent_dist[seq[0]] += 1
    logger.info(f"Intent distribution: {intent_dist.tolist()}")

    # Step 2: Encode via GLMXModel pipeline
    logger.info("=" * 60)
    logger.info("STEP 2/4: Encoding subgraphs via Sentence-BERT")
    logger.info("=" * 60)
    glm_x = GLMXModel(g2p_config_path=config_path)
    glm_x.initialize_pipeline()

    X_list, y_list = [], []
    for sg, intent_seq in raw_data:
        try:
            emb = glm_x.encode_subgraph(sg)
            target = intent_seq[0] if intent_seq else 0
            X_list.append(emb)
            y_list.append(target)
        except Exception as e:
            logger.warning(f"Encode error: {e}")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    logger.info(f"Encoded {len(X)}/{len(raw_data)} samples")

    # Step 3: Split
    n = len(X)
    indices = np.random.RandomState(42).permutation(n)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    X_train, y_train = X[indices[:train_end]], y[indices[:train_end]]
    X_val, y_val = X[indices[train_end:val_end]], y[indices[train_end:val_end]]
    X_test, y_test = X[indices[val_end:]], y[indices[val_end:]]
    logger.info(f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Step 4: Train
    logger.info("=" * 60)
    logger.info("STEP 3/4: Training IntentFFN")
    logger.info("=" * 60)
    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0001,
        "loss": "focal",
        "focal_gamma": 2.0,
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": 10,
    }
    training_run = TrainingRun(
        model=glm_x.intent_ffn,
        dataset_name="synthetic",
        config=train_config,
        output_dir=output_dir,
    )
    metrics = training_run.run(X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 5: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 4/4: Evaluating on test set")
    logger.info("=" * 60)
    eval_metrics = run_evaluation(glm_x.intent_ffn, X_test, y_test)
    metrics["evaluation"] = eval_metrics
    metrics["dataset_stats"] = {
        "total_samples": len(raw_data),
        "encoded": len(X),
        "intent_distribution": intent_dist.tolist(),
    }
    metrics["total_pipeline_time_seconds"] = round(time.time() - step_start, 2)

    # Save model + metrics
    glm_x.add_trained_dataset("synthetic")
    glm_x.save(output_dir, model_name="intent_ffn_best.pt", metadata={
        "dataset": "synthetic",
        "samples": len(raw_data),
        "test_accuracy": eval_metrics["test_top1_accuracy"],
    })
    training_run.save_results(unified_metrics_path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Model: {output_dir}")
    logger.info(f"  Metrics: {unified_metrics_path}")
    logger.info(f"  Test Top-1: {eval_metrics['test_top1_accuracy']:.4f}")
    logger.info(f"  Test Top-3: {eval_metrics['test_top3_accuracy']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
