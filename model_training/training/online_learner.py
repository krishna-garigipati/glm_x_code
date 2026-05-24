import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DynamicsTracker:
    def __init__(self):
        self.gradient_noise_scale = 1.0
        self.relative_grad_norm = 1.0


class OnlineLearner:
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.dynamics = DynamicsTracker()
        self.converged = False
        self.total_exposures = 0
        self._loss_history = []
        self._gns_history = []

    def absorb(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> dict:
        self.model.train()
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        loader = DataLoader(dataset, batch_size=min(batch_size, len(X)), shuffle=True)

        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0

        for bx, by in loader:
            self.optimizer.zero_grad()
            logits = self.model(bx)
            loss = self.criterion(logits, by)
            loss.backward()
            grad_norm = sum(p.grad.norm().item() ** 2 for p in self.model.parameters()
                           if p.grad is not None) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_batches += 1

        self.total_exposures += len(X)
        avg_loss = total_loss / n_batches
        avg_gns = total_grad_norm / n_batches
        self._loss_history.append(avg_loss)
        self._gns_history.append(avg_gns)

        self.dynamics.gradient_noise_scale = float(avg_gns)
        self.dynamics.relative_grad_norm = float(avg_gns / max(avg_loss, 1e-8))

        lir = 0.0
        if len(self._loss_history) >= 3:
            recent = self._loss_history[-3:]
            lir = (recent[0] - recent[-1]) / max(recent[0], 1e-8)

        if len(self._loss_history) >= 10:
            recent_gns = self._gns_history[-5:]
            if all(g < 0.01 for g in recent_gns):
                self.converged = True

        self.scheduler.step()

        return {
            "loss": round(float(avg_loss), 6),
            "gradient_noise_scale": round(float(avg_gns), 6),
            "relative_grad_norm": round(float(self.dynamics.relative_grad_norm), 6),
            "loss_improvement_rate": round(float(lir), 6),
            "converged": self.converged,
        }
