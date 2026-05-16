import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader, TensorDataset

from model_training.models.intent_ffn import IntentFFN, FocalLoss
from model_training.models.model_manager import ModelManager

logger = logging.getLogger(__name__)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    _, pred = torch.topk(logits, k=top_k, dim=-1)
    correct = pred.eq(targets.unsqueeze(-1)).any(dim=-1).float().sum().item()
    return correct / targets.size(0)


class TrainingRun:
    def __init__(
        self,
        model: IntentFFN,
        dataset_name: str,
        config: Dict[str, Any],
        output_dir: Path,
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {
            "dataset": dataset_name,
            "config": config,
            "epochs": [],
            "best_val_loss": None,
            "best_epoch": None,
            "train_time_seconds": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _get_loss_fn(self) -> nn.Module:
        loss_type = self.config.get("loss", "focal")
        if loss_type == "focal":
            gamma = self.config.get("focal_gamma", 2.0)
            return FocalLoss(gamma=gamma)
        return nn.CrossEntropyLoss()

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        c = self.config
        batch_size = c.get("batch_size", 32)
        epochs = c.get("epochs", 100)
        lr = c.get("learning_rate", 0.001)
        weight_decay = c.get("weight_decay", 0.0001)
        clip_norm = c.get("gradient_clip_norm", 1.0)
        patience = c.get("early_stopping_patience", 10)

        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, batch_size, shuffle=False)
        test_loader = self._make_loader(X_test, y_test, batch_size, shuffle=False)

        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = self._get_loss_fn()

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_top1 = 0.0
            train_top3 = 0.0
            num_train_batches = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                optimizer.step()

                train_loss += loss.item()
                train_top1 += compute_accuracy(logits, batch_y, top_k=1)
                train_top3 += compute_accuracy(logits, batch_y, top_k=3)
                num_train_batches += 1

            avg_train_loss = train_loss / num_train_batches
            avg_train_top1 = train_top1 / num_train_batches
            avg_train_top3 = train_top3 / num_train_batches

            self.model.eval()
            val_loss = 0.0
            val_top1 = 0.0
            val_top3 = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
                    val_top1 += compute_accuracy(logits, batch_y, top_k=1)
                    val_top3 += compute_accuracy(logits, batch_y, top_k=3)
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            avg_val_top1 = val_top1 / num_val_batches
            avg_val_top3 = val_top3 / num_val_batches
            current_lr = scheduler.get_last_lr()[0]

            self.metrics["epochs"].append({
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "train_top1_accuracy": round(avg_train_top1, 6),
                "val_top1_accuracy": round(avg_val_top1, 6),
                "train_top3_accuracy": round(avg_train_top3, 6),
                "val_top3_accuracy": round(avg_val_top3, 6),
                "learning_rate": round(current_lr, 8),
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = self.model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            scheduler.step()

            if (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | "
                    f"train_top1={avg_train_top1:.4f} val_top1={avg_val_top1:.4f}"
                )

        train_time = time.time() - start_time

        self.model.load_state_dict(best_state)
        self.metrics["best_val_loss"] = round(best_val_loss, 6)
        self.metrics["best_epoch"] = best_epoch
        self.metrics["train_time_seconds"] = round(train_time, 2)
        self.metrics["total_params"] = self.model.get_num_params()

        test_top1, test_top3, test_loss = self._evaluate(
            self.model, test_loader, criterion
        )
        self.metrics["test"] = {
            "loss": round(test_loss, 6),
            "top1_accuracy": round(test_top1, 6),
            "top3_accuracy": round(test_top3, 6),
        }

        logger.info(
            f"Training complete: {epoch+1} epochs, {train_time:.1f}s, "
            f"best_val_loss={best_val_loss:.4f} (epoch {best_epoch}), "
            f"test_top1={test_top1:.4f}"
        )

        return self.metrics

    def _evaluate(self, model: IntentFFN, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float]:
        model.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top3 = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                total_top1 += compute_accuracy(logits, batch_y, top_k=1)
                total_top3 += compute_accuracy(logits, batch_y, top_k=3)
                num_batches += 1
        return (
            total_top1 / num_batches,
            total_top3 / num_batches,
            total_loss / num_batches,
        )

    def save_results(self, unified_metrics_path: Path):
        unified_metrics_path = Path(unified_metrics_path)
        existing = {}
        if unified_metrics_path.exists():
            with open(unified_metrics_path, "r") as f:
                existing = json.load(f)
        existing[self.dataset_name] = self.metrics
        with open(unified_metrics_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info(f"Metrics saved to {unified_metrics_path}")
