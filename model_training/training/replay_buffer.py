import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.samples: List[Tuple[np.ndarray, int]] = []

    def add_dataset(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        for i in range(n):
            if len(self.samples) < self.capacity:
                self.samples.append((X[i].copy(), int(y[i])))
            else:
                idx = np.random.randint(0, len(self.samples))
                self.samples[idx] = (X[i].copy(), int(y[i]))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.samples:
            return np.zeros((0, 384)), np.zeros(0, dtype=np.int64)
        k = min(batch_size, len(self.samples))
        indices = np.random.choice(len(self.samples), size=k, replace=False)
        X = np.stack([self.samples[i][0] for i in indices])
        y = np.array([self.samples[i][1] for i in indices], dtype=np.int64)
        return X, y

    def __len__(self) -> int:
        return len(self.samples)

    def clear(self):
        self.samples.clear()
