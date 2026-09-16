import logging
import threading
from typing import Dict, Tuple

import numpy as np

from learning.types import Edge, Subgraph, WalkResult, GraphStoreInterface
from learning.config import LearningConfig

logger = logging.getLogger(__name__)


class HebbianUpdater:
    def __init__(self, config: LearningConfig):
        self.cfg = config.hebbian
        self.decay_cfg = config.global_decay
        self._lock = threading.Lock()
        self._query_counter = 0

    @property
    def query_counter(self) -> int:
        return self._query_counter

    def update_strength(self, s_old: float, reward: float, eligibility: float) -> float:
        delta = self.cfg.alpha * reward * eligibility
        s_new = s_old + delta
        return max(0.0, min(1.0, s_new))

    def update_confidence(self, c_old: float, reward: float, eligibility: float, walk_confidence: float = 0.5) -> float:
        effective_reward = abs(reward)
        walk_quality_mod = 0.5 + 0.5 * walk_confidence
        c_new = c_old + self.cfg.beta * effective_reward * eligibility * walk_quality_mod
        return max(0.0, min(1.0, c_new))

    def apply_hebbian_updates(
        self,
        graph: GraphStoreInterface,
        eligibility_traces: Dict[Tuple[int, int, str], float],
        reward: float,
        subgraph: Subgraph,
        plan_adherence: float = 1.0,
        model_quality: float = 1.0,
        walk_confidence: float = 0.5,
    ) -> None:
        if not eligibility_traces:
            return

        reward = reward * (0.5 + 0.5 * model_quality)

        updates: Dict[Tuple[int, int, str], Tuple[float, float]] = {}

        for edge_key, eligibility in eligibility_traces.items():
            source, target, relation = edge_key
            existing_edge = graph.get_edge(source, target, relation)
            if existing_edge is None:
                if reward > 0 and eligibility > 0:
                    graph.add_edge(source, target, relation, strength=0.5, confidence=0.5)
                    existing_edge = graph.get_edge(source, target, relation)
                    if existing_edge is None:
                        continue
                else:
                    continue

            s_old = existing_edge.strength
            c_old = existing_edge.confidence
            s_new = self.update_strength(s_old, reward, eligibility)
            c_new = self.update_confidence(c_old, reward, eligibility, walk_confidence)
            updates[edge_key] = (s_new, c_new)
            sign = "+" if s_new >= s_old else "-"
            logger.debug(
                "Hebbian %s: edge(%d,%d,%s) S: %.4f->%.4f C: %.4f->%.4f R=%.4f E_e=%.4f",
                sign, source, target, relation, s_old, s_new, c_old, c_new, reward, eligibility,
            )

        if updates:
            graph.update_edge_weights(updates)

    def apply_global_decay(
        self,
        graph: GraphStoreInterface,
        subgraph: Subgraph,
    ) -> None:
        if not self.decay_cfg.enabled:
            return

        delta = self.decay_cfg.delta_base
        min_s = self.decay_cfg.min_strength

        updates: Dict[Tuple[int, int, str], Tuple[float, float]] = {}

        for edge_key in subgraph.edge_strengths:
            source, target, relation = edge_key
            existing_edge = graph.get_edge(source, target, relation)
            if existing_edge is None:
                continue
            s_old = existing_edge.strength
            freq = existing_edge.frequency
            if self.decay_cfg.frequency_protection:
                protection = 1.0 + min(50.0, freq) / 50.0
                decay_factor = 1.0 - delta / protection
            else:
                decay_factor = 1.0 - delta
            s_new = s_old * decay_factor
            s_new = max(min_s, s_new)
            if s_new < s_old:
                updates[edge_key] = (s_new, existing_edge.confidence)

        if updates:
            graph.update_edge_weights(updates)
            logger.debug("Global decay applied to %d edges (delta=%.6f)", len(updates), delta)

    def increment_query_counter(self, n: int = 1) -> bool:
        with self._lock:
            self._query_counter += n
            return self._query_counter >= self.decay_cfg.interval_queries

    def reset_query_counter(self) -> None:
        with self._lock:
            self._query_counter = 0
