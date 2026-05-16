import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import FFNConfig


class IntentFFN(nn.Module):
    def __init__(self, config: FFNConfig):
        super().__init__()
        self.config = config
        layers = []
        in_dim = config.input_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(config.hidden_dim, config.classifier_input_dim))
        self.body = nn.Sequential(*layers)

        classifier_layers = []
        cin = config.classifier_input_dim
        for _ in range(config.classifier_num_layers):
            classifier_layers.append(nn.Linear(cin, config.classifier_hidden_dim))
            classifier_layers.append(nn.ReLU())
            cin = config.classifier_hidden_dim
        classifier_layers.append(nn.Linear(cin, config.output_dim))
        self.classifier = nn.Sequential(*classifier_layers)

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
        pass
