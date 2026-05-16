import torch
import torch.nn as nn
import numpy as np
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader, TensorDataset

from model_training.models.intent_ffn import IntentFFN, FocalLoss
from g2p.types import Subgraph

logger = logging.getLogger(__name__)

INTENT_NAMES = {
    0: "define", 1: "assert_fact", 2: "explain_cause", 3: "explain_effect",
    4: "contrast", 5: "compare", 6: "list", 7: "example",
    8: "conclude", 9: "question", 10: "uncertain", 11: "clarify",
    12: "summarize", 13: "elaborate", 14: "transition", 15: "emphasize",
}


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    _, pred = torch.topk(logits, k=top_k, dim=-1)
    correct = pred.eq(targets.unsqueeze(-1)).any(dim=-1).float().sum().item()
    return correct / targets.size(0)


def compute_confusion_matrix(
    model: IntentFFN,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    model.eval()
    confusion = np.zeros((16, 16), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(X_test), 32):
            batch_X = torch.from_numpy(X_test[i : i + 32]).float()
            batch_y = y_test[i : i + 32]
            logits = model(batch_X)
            preds = torch.argmax(logits, dim=-1).numpy()
            for p, t in zip(preds, batch_y):
                confusion[t, p] += 1
    return confusion


def per_class_metrics(confusion: np.ndarray) -> Dict[str, Any]:
    n_classes = confusion.shape[0]
    results = {}
    for i in range(n_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[INTENT_NAMES.get(i, f"intent_{i}")] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(confusion[i, :].sum()),
        }
    return results


def run_evaluation(
    model: IntentFFN,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    model.eval()
    criterion = FocalLoss(gamma=2.0)
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )
    test_loader = DataLoader(test_dataset, batch_size=32)

    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_top5 = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            total_top1 += compute_accuracy(logits, batch_y, top_k=1)
            total_top3 += compute_accuracy(logits, batch_y, top_k=3)
            total_top5 += compute_accuracy(logits, batch_y, top_k=5)
            all_preds.extend(torch.argmax(logits, dim=-1).numpy().tolist())
            all_targets.extend(batch_y.numpy().tolist())

    n_batches = len(test_loader)
    confusion = compute_confusion_matrix(model, X_test, y_test)
    class_metrics = per_class_metrics(confusion)

    metrics = {
        "test_loss": round(total_loss / n_batches, 6),
        "test_top1_accuracy": round(total_top1 / n_batches, 6),
        "test_top3_accuracy": round(total_top3 / n_batches, 6),
        "test_top5_accuracy": round(total_top5 / n_batches, 6),
        "confusion_matrix": confusion.tolist(),
        "per_class_metrics": class_metrics,
        "num_test_samples": len(y_test),
    }

    logger.info(
        f"Evaluation: loss={metrics['test_loss']:.4f}, "
        f"top1={metrics['test_top1_accuracy']:.4f}, "
        f"top3={metrics['test_top3_accuracy']:.4f}, "
        f"top5={metrics['test_top5_accuracy']:.4f}"
    )
    return metrics
