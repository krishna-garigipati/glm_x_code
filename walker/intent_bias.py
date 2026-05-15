"""Intent-to-edge bias lookup table."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Dict


@dataclass
class IntentBiasTable:
    _biases: Dict[int, Dict[str, float]]
    _lock: RLock

    def __init__(self, biases: Dict[int, Dict[str, float]]):
        self._biases = {int(k): dict(v) for k, v in biases.items()}
        self._lock = RLock()

    def get_bias(self, intent_id: int, relation: str) -> float:
        with self._lock:
            intent_biases = self._biases.get(intent_id, {})
            if relation in intent_biases:
                return float(intent_biases[relation])
            if "default" in intent_biases:
                return float(intent_biases["default"])
            return 1.0

    def update_bias(self, intent_id: int, relation: str, bias: float) -> None:
        with self._lock:
            if intent_id not in self._biases:
                self._biases[intent_id] = {}
            self._biases[intent_id][relation] = float(bias)

    def snapshot(self) -> Dict[int, Dict[str, float]]:
        with self._lock:
            return {intent_id: dict(biases) for intent_id, biases in self._biases.items()}
