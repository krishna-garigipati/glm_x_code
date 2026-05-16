import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class IntentFFN(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        classifier_input_dim: int = 128,
        classifier_hidden_dim: int = 64,
        classifier_num_layers: int = 1,
        output_dim: int = 16,
    ):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "classifier_input_dim": classifier_input_dim,
            "classifier_hidden_dim": classifier_hidden_dim,
            "classifier_num_layers": classifier_num_layers,
            "output_dim": output_dim,
        }
        self._build_body()
        self._build_classifier()
        self._trained_on_datasets: list = []

    def _build_body(self):
        c = self.config
        layers = []
        in_dim = c["input_dim"]
        for _ in range(c["num_layers"]):
            layers.append(nn.Linear(in_dim, c["hidden_dim"]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(c["dropout"]))
            in_dim = c["hidden_dim"]
        layers.append(nn.Linear(c["hidden_dim"], c["classifier_input_dim"]))
        self.body = nn.Sequential(*layers)

    def _build_classifier(self):
        c = self.config
        layers = []
        cin = c["classifier_input_dim"]
        for _ in range(c["classifier_num_layers"]):
            layers.append(nn.Linear(cin, c["classifier_hidden_dim"]))
            layers.append(nn.ReLU())
            cin = c["classifier_hidden_dim"]
        layers.append(nn.Linear(cin, c["output_dim"]))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        logits = self.classifier(h)
        return logits

    def predict_logits(self, embedding: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(embedding).float()
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            return logits.squeeze(0).numpy()

    def predict_probs(self, embedding: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(embedding)
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        return probs

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def add_trained_dataset(self, dataset_name: str):
        if dataset_name not in self._trained_on_datasets:
            self._trained_on_datasets.append(dataset_name)

    def save(self, path: Path, metadata: Optional[Dict[str, Any]] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "trained_on_datasets": self._trained_on_datasets,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "IntentFFN":
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model._trained_on_datasets = checkpoint.get("trained_on_datasets", [])
        logger.info(
            f"Model loaded from {path} "
            f"(trained on: {model._trained_on_datasets})"
        )
        return model

    def expand_for_new_dataset(
        self,
        new_output_classes: int = 16,
        preserve_weights: bool = True,
    ) -> "IntentFFN":
        if new_output_classes == self.config["output_dim"] and preserve_weights:
            logger.info("Output dim unchanged, returning self")
            return self

        old_config = self.config.copy()
        old_config["output_dim"] = new_output_classes

        new_model = IntentFFN(**old_config)
        if preserve_weights:
            old_state = self.state_dict()
            new_state = new_model.state_dict()
            for key in old_state:
                if key in new_state and old_state[key].shape == new_state[key].shape:
                    new_state[key] = old_state[key]
            new_model.load_state_dict(new_state)
            new_model._trained_on_datasets = self._trained_on_datasets.copy()
            logger.info(
                f"Expanded model: output_dim {self.config['output_dim']} -> {new_output_classes}, "
                f"weights preserved where shapes match"
            )
        return new_model
