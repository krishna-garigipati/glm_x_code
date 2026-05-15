import logging
import threading
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

import numpy as np

from learner.types import Subgraph, Edge, GraphStoreInterface
from learner.config import LearningConfig

logger = logging.getLogger(__name__)


class PatternCompressor:
    def __init__(self, config: LearningConfig):
        self.cfg = config.compression
        self._lock = threading.Lock()
        self._query_counter = 0
        self._co_activation_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self._co_activation_contexts: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self._co_activation_rewards: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        self._sequence_buffer: List[List[int]] = []
        self._pattern_node_id_counter: int = 2 ** 60

    @property
    def query_counter(self) -> int:
        return self._query_counter

    def record_co_activation(self, node_a: int, node_b: int, context_id: int, reward: float = 0.5) -> None:
        key = (min(node_a, node_b), max(node_a, node_b))
        with self._lock:
            self._co_activation_counts[key] += 1
            self._co_activation_contexts[key].add(context_id)
            self._co_activation_rewards[key].append(reward)

    def record_walk_sequence(self, node_ids: List[int]) -> None:
        with self._lock:
            self._sequence_buffer.append(list(node_ids))
            if len(self._sequence_buffer) > 1000:
                self._sequence_buffer = self._sequence_buffer[-500:]

    def get_frequent_co_activations(self, graph: Optional[GraphStoreInterface] = None) -> List[Tuple[int, int, int, float]]:
        threshold = self.cfg.co_activation_threshold
        min_contexts = self.cfg.diverse_contexts_required
        results = []
        with self._lock:
            for (na, nb), count in self._co_activation_counts.items():
                if count >= threshold:
                    contexts = self._co_activation_contexts[(na, nb)]
                    if len(contexts) >= min_contexts:
                        avg_reward = float(np.mean(self._co_activation_rewards[(na, nb)]))
                        results.append((na, nb, count, avg_reward))
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def find_pattern_sequences(self, graph: Optional[GraphStoreInterface] = None) -> List[List[int]]:
        seq_len = self.cfg.sequence_length
        freq_map: Dict[Tuple[int, ...], int] = defaultdict(int)
        with self._lock:
            for seq in self._sequence_buffer:
                for i in range(len(seq) - seq_len + 1):
                    pattern = tuple(seq[i:i + seq_len])
                    freq_map[pattern] += 1
        thresh = self.cfg.co_activation_threshold
        min_count = max(2, thresh // 2)
        patterns = [list(p) for p, c in freq_map.items() if c >= min_count]
        return patterns[:50]

    def create_pattern_node(
        self,
        graph: GraphStoreInterface,
        node_a: int,
        node_b: int,
    ) -> Optional[int]:
        node_a_data = graph.get_node(node_a)
        node_b_data = graph.get_node(node_b)
        if node_a_data is None or node_b_data is None:
            return None

        pattern_id = self._next_pattern_id()
        pattern_label = f"Pattern({node_a},{node_b})"
        avg_embedding = self._average_embeddings(
            node_a_data.embedding, node_b_data.embedding
        )

        graph.add_node(
            node_id=pattern_id,
            label=pattern_label,
            node_type=self.cfg.pattern_node_type,
            embedding=avg_embedding,
            activation=0.01,
        )

        s = self.cfg.pattern_edge_strength
        c = self.cfg.pattern_edge_confidence
        graph.add_edge(pattern_id, node_a, "associated_with", strength=s, confidence=c)
        graph.add_edge(pattern_id, node_b, "associated_with", strength=s, confidence=c)

        logger.info("Created pattern node %d for (%d, %d)", pattern_id, node_a, node_b)
        return pattern_id

    def create_linguistic_pattern_node(
        self,
        graph: GraphStoreInterface,
        tokens: List[int],
    ) -> Optional[int]:
        if not tokens:
            return None

        pattern_id = self._next_pattern_id()
        token_labels = []
        embeddings = []
        for tid in tokens:
            node = graph.get_node(tid)
            if node is not None:
                token_labels.append(node.label)
                embeddings.append(node.embedding)

        if not embeddings:
            return None

        pattern_label = "LingPattern(" + ",".join(token_labels[:5]) + ")"
        avg_embedding = self._average_embeddings(*embeddings)

        graph.add_node(
            node_id=pattern_id,
            label=pattern_label,
            node_type=self.cfg.linguistic_pattern_node_type,
            embedding=avg_embedding,
            activation=0.01,
        )

        s = self.cfg.pattern_edge_strength
        c = self.cfg.pattern_edge_confidence
        for tid in tokens:
            graph.add_edge(pattern_id, tid, "linguistic_maps", strength=s, confidence=c)

        logger.info("Created linguistic pattern node %d for %d tokens", pattern_id, len(tokens))
        return pattern_id

    def compress_pattern_nodes(
        self,
        graph: GraphStoreInterface,
        min_co_activation: Optional[int] = None,
    ) -> List[int]:
        new_node_ids: List[int] = []
        pairs = self.get_frequent_co_activations(graph)
        for na, nb, _count, _avg_reward in pairs:
            existing = graph.get_edge(na, nb, "associated_with")
            if existing is not None and existing.strength > 0.5:
                continue
            nid = self.create_pattern_node(graph, na, nb)
            if nid is not None:
                new_node_ids.append(nid)

        if self.cfg.linguistic_enabled:
            patterns = self.find_pattern_sequences(graph)
            for seq in patterns[:10]:
                nid = self.create_linguistic_pattern_node(graph, seq)
                if nid is not None:
                    new_node_ids.append(nid)

        if new_node_ids:
            with self._lock:
                self._co_activation_counts.clear()
                self._co_activation_contexts.clear()
                self._co_activation_rewards.clear()

        logger.info("Compression created %d new pattern nodes", len(new_node_ids))
        return new_node_ids

    def increment_query_counter(self, n: int = 1) -> bool:
        with self._lock:
            self._query_counter += n
            return self._query_counter >= self.cfg.interval_queries

    def reset_query_counter(self) -> None:
        with self._lock:
            self._query_counter = 0

    def _next_pattern_id(self) -> int:
        with self._lock:
            self._pattern_node_id_counter += 1
            return self._pattern_node_id_counter

    @staticmethod
    def _average_embeddings(*embeddings: np.ndarray) -> np.ndarray:
        if not embeddings:
            raise ValueError("At least one embedding required")
        float_embs = [emb.astype(np.float32) / 127.0 for emb in embeddings]
        avg_float = np.mean(float_embs, axis=0)
        avg_float = np.clip(avg_float, -1.0, 1.0)
        avg_int8 = np.round(avg_float * 127).astype(np.int8)
        return avg_int8
