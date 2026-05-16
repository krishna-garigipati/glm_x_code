#!/usr/bin/env python
"""
Test saved GLM-X model: load, evaluate on test data, generate predictions.
"""

import argparse
import logging
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_model")

from model_training.models.glm_x_model import GLMXModel
from model_training.training.evaluate import run_evaluation
from model_training.dataset_conceptnet.loader import download_conceptnet, filter_known_relations, build_concept_graph
from model_training.dataset_conceptnet.preprocessor import generate_training_data


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained GLM-X model")
    parser.add_argument("--model-dir", type=str, default="saved_models/conceptnet", help="Saved model directory")
    parser.add_argument("--max-samples", type=int, default=500, help="Test samples from ConceptNet")
    parser.add_argument("--seed-limit", type=int, default=200, help="Seed concepts for test set")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).parent.resolve()
    model_dir = base_dir / args.model_dir

    logger.info(f"Loading GLMXModel from {model_dir}")
    glm_x = GLMXModel.load(model_dir, g2p_config_path=base_dir / "config.yaml")
    glm_x.initialize_pipeline()

    logger.info("Loading ConceptNet test triples...")
    triples = download_conceptnet(max_samples=args.max_samples)
    filtered = filter_known_relations(triples)
    concept_to_id, edges = build_concept_graph(filtered)
    id_to_concept = {v: k for k, v in concept_to_id.items()}

    raw_samples = generate_training_data(
        edges=edges,
        concept_to_id=concept_to_id,
        id_to_concept=id_to_concept,
        seed_limit=args.seed_limit,
        max_hops=2,
        min_nodes=2,
    )
    logger.info(f"Generated {len(raw_samples)} test samples")

    X_list, y_list = [], []
    for sg, intent_seq, seed_name in raw_samples:
        try:
            emb = glm_x.encode_subgraph(sg)
            target = intent_seq[0] if intent_seq else 0
            X_list.append(emb)
            y_list.append(target)
        except Exception as e:
            logger.warning(f"Encode error: {e}")

    X_test = np.stack(X_list)
    y_test = np.array(y_list, dtype=np.int64)
    logger.info(f"Encoded {len(X_test)} test samples")

    eval_metrics = run_evaluation(glm_x.intent_ffn, X_test, y_test)

    logger.info("=" * 50)
    logger.info("TEST RESULTS")
    logger.info(f"  Test Top-1 Accuracy: {eval_metrics['test_top1_accuracy']:.4f}")
    logger.info(f"  Test Top-3 Accuracy: {eval_metrics['test_top3_accuracy']:.4f}")
    logger.info(f"  Test Top-5 Accuracy: {eval_metrics['test_top5_accuracy']:.4f}")
    logger.info(f"  Test Loss: {eval_metrics['test_loss']:.4f}")

    best_class = max(eval_metrics["per_class_metrics"].items(), key=lambda x: x[1]["f1"])
    worst_class = min(eval_metrics["per_class_metrics"].items(), key=lambda x: x[1]["f1"])
    logger.info(f"  Best class: {best_class[0]} (F1={best_class[1]['f1']:.4f})")
    logger.info(f"  Worst class: {worst_class[0]} (F1={worst_class[1]['f1']:.4f})")

    results_path = base_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"Test results saved to {results_path}")

    gen_tests = [
        ("cat", "IsA", "mammal"),
        ("dog", "IsA", "canine"),
        ("fire", "Causes", "smoke"),
        ("water", "IsA", "liquid"),
        ("car", "HasA", "wheel"),
    ]
    logger.info("\n--- Sample predictions ---")
    from model_training.dataset_conceptnet.relation_map import INTENT_VOCAB

    test_edges = [
        {"head_id": 0, "head": h, "relation": r, "tail_id": 1, "tail": t}
        for h, r, t in gen_tests
    ]
    dummy_nodes = {0, 1}
    for e in test_edges:
        sg = _make_dummy_subgraph(dummy_nodes, [e])
        emb = glm_x.encode_subgraph(sg)
        probs = glm_x.intent_ffn.predict_probs(emb)
        pred = int(np.argmax(probs))
        logger.info(f"  '{e['head']}' --[{e['relation']}]--> '{e['tail']}'  =>  {INTENT_VOCAB.get(pred, pred)} (conf={probs[pred]:.3f})")


def _make_dummy_subgraph(nodes_set, edges_list):
    from g2p.types import Subgraph
    node_list = sorted(nodes_set)
    sub_edges = []
    strengths, confs = {}, {}
    for e in edges_list:
        key = (e["head_id"], e["tail_id"], e["relation"])
        sub_edges.append(key)
        strengths[key] = 0.8
        confs[key] = 0.8
    return Subgraph(
        nodes=node_list,
        node_activations={n: 0.5 for n in node_list},
        edges=sub_edges,
        edge_strengths=strengths,
        edge_confidences=confs,
        seed_nodes=[0],
        tier_used=1,
        activation_energy=1.0,
        query_embedding=np.random.randn(384).astype(np.float32),
        timestamp=1e9,
    )


if __name__ == "__main__":
    main()
