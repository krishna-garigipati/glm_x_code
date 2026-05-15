import os
import pickle
import logging
import threading
from typing import Dict, List, Optional, Any

import numpy as np

from learner.config import LearningConfig

logger = logging.getLogger(__name__)


class LearningStatePersistence:
    def __init__(self, config: LearningConfig):
        self.cfg = config.state
        self._lock = threading.Lock()
        self._query_counter = 0
        self._ensure_save_path()

    @property
    def query_counter(self) -> int:
        return self._query_counter

    def _ensure_save_path(self) -> None:
        os.makedirs(self.cfg.save_path, exist_ok=True)

    def save(self, name: str, data: Any) -> bool:
        if name not in self.cfg.files:
            logger.warning("Unknown state file: %s (not in config)", name)
            return False
        path = os.path.join(self.cfg.save_path, name)
        try:
            with self._lock:
                with open(path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Saved state: %s (%d bytes)", path, os.path.getsize(path))
            return True
        except Exception as e:
            logger.error("Failed to save state %s: %s", path, e)
            return False

    def load(self, name: str) -> Optional[Any]:
        if name not in self.cfg.files:
            logger.warning("Unknown state file: %s (not in config)", name)
            return None
        path = os.path.join(self.cfg.save_path, name)
        if not os.path.exists(path):
            logger.info("State file not found: %s", path)
            return None
        try:
            with self._lock:
                with open(path, "rb") as f:
                    data = pickle.load(f)
            logger.debug("Loaded state: %s", path)
            return data
        except Exception as e:
            logger.error("Failed to load state %s: %s", path, e)
            return None

    def save_all(self, state_dict: Dict[str, Any]) -> Dict[str, bool]:
        results = {}
        for name, data in state_dict.items():
            results[name] = self.save(name, data)
        return results

    def load_all(self) -> Dict[str, Any]:
        result = {}
        for name in self.cfg.files:
            data = self.load(name)
            if data is not None:
                result[name] = data
        return result

    def list_saved_states(self) -> List[str]:
        available = []
        for name in self.cfg.files:
            path = os.path.join(self.cfg.save_path, name)
            if os.path.exists(path):
                available.append(name)
        return available

    def clear_all(self) -> bool:
        try:
            with self._lock:
                for name in self.cfg.files:
                    path = os.path.join(self.cfg.save_path, name)
                    if os.path.exists(path):
                        os.remove(path)
                        logger.debug("Removed state file: %s", path)
            return True
        except Exception as e:
            logger.error("Failed to clear state files: %s", e)
            return False

    def increment_query_counter(self, n: int = 1) -> bool:
        with self._lock:
            self._query_counter += n
            return self._query_counter >= self.cfg.save_interval_queries

    def reset_query_counter(self) -> None:
        with self._lock:
            self._query_counter = 0
