from __future__ import annotations

from typing import Dict, List, Set, Tuple
import logging
import threading
import numpy as np

from .config import CoreConfig, AnalogyParameters
from .types import GraphStore

logger = logging.getLogger(__name__)


class AnalogyFinder:
    def __init__(self, core_config: CoreConfig, params: AnalogyParameters, embedding_dim: int = 32):
        self._core = core_config
        self._params = params
        self._embedding_dim = embedding_dim
        self._lock = threading.Lock()
        self._lsh_index: List[Dict[bytes, List[int]]] | None = None
        self._lsh_hyperplanes: List[np.ndarray] | None = None
        self._node_embeddings: Dict[int, np.ndarray] = {}

    def get_analogy_leaps(
        self, target_node: int, graph: GraphStore, top_k: int = 3
    ) -> List[Tuple[int, float]]:
        if top_k < 1:
            return []

        target = graph.get_node(target_node)
        if target is None:
            logger.debug("Target node %s not found in graph", target_node)
            return []

        target_embedding = _to_float_embedding(target.embedding)

        try:
            self._ensure_lsh_index(graph)
            return self._query_lsh(target_node, target_embedding, graph, top_k)
        except (AttributeError, NotImplementedError):
            logger.debug("LSH not available, falling back to brute-force")
            return self._query_brute_force(target_node, target_embedding, graph, top_k)

    def clear_cache(self) -> None:
        with self._lock:
            self._lsh_index = None
            self._lsh_hyperplanes = None
            self._node_embeddings.clear()

    def _ensure_lsh_index(self, graph: GraphStore) -> None:
        with self._lock:
            if self._lsh_index is not None:
                return
            self._build_lsh_index(graph)

    def _build_lsh_index(self, graph: GraphStore) -> None:
        all_nodes = graph.get_all_nodes()
        if not all_nodes:
            logger.warning("No nodes in graph, LSH index will be empty")
            self._lsh_index = [{}]
            self._lsh_hyperplanes = [np.zeros((1, 32), dtype=np.float32)]
            return

        n_bands = max(1, self._params.lsh_bands)
        n_tables = max(1, self._params.lsh_tables)

        rng = np.random.RandomState(42)
        self._lsh_hyperplanes = []
        for _ in range(n_tables):
            planes = rng.normal(size=(n_bands, self._embedding_dim)).astype(np.float32)
            norms = np.linalg.norm(planes, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            self._lsh_hyperplanes.append(planes / norms)

        self._lsh_index = [{} for _ in range(n_tables)]
        for node in all_nodes:
            emb = _to_float_embedding(node.embedding)
            self._node_embeddings[node.id] = emb
            for table_idx in range(n_tables):
                hash_key = self._hash(emb, self._lsh_hyperplanes[table_idx])
                bucket = self._lsh_index[table_idx]
                bucket.setdefault(hash_key, []).append(node.id)

        logger.info("Built LSH index: %d nodes, %d tables, %d bands",
                    len(all_nodes), n_tables, n_bands)

    def _query_lsh(
        self, target_node: int, target_embedding: np.ndarray, graph: GraphStore, top_k: int
    ) -> List[Tuple[int, float]]:
        if not self._lsh_index or not self._lsh_hyperplanes:
            return self._query_brute_force(target_node, target_embedding, graph, top_k)

        candidate_set: Set[int] = set()
        for table_idx in range(len(self._lsh_index)):
            hash_key = self._hash(target_embedding, self._lsh_hyperplanes[table_idx])
            bucket = self._lsh_index[table_idx].get(hash_key, [])
            candidate_set.update(bucket)

        candidates: List[Tuple[int, float]] = []
        for node_id in candidate_set:
            if node_id == target_node:
                continue
            emb = self._node_embeddings.get(node_id)
            if emb is None:
                node = graph.get_node(node_id)
                if node is None:
                    continue
                emb = _to_float_embedding(node.embedding)
                self._node_embeddings[node_id] = emb
            similarity = _cosine_similarity(target_embedding, emb)
            candidates.append((node_id, similarity))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return self._validate_candidates(target_node, candidates, graph, top_k)

    def _query_brute_force(
        self, target_node: int, target_embedding: np.ndarray, graph: GraphStore, top_k: int
    ) -> List[Tuple[int, float]]:
        subgraph = graph.get_subgraph_by_embedding_similarity(target_embedding, top_k=top_k)
        candidates: List[Tuple[int, float]] = []
        for node_id in subgraph.nodes:
            if node_id == target_node:
                continue
            node = graph.get_node(node_id)
            if node is None:
                continue
            embedding = _to_float_embedding(node.embedding)
            similarity = _cosine_similarity(target_embedding, embedding)
            candidates.append((node_id, similarity))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return self._validate_candidates(target_node, candidates, graph, top_k)

    def _validate_candidates(
        self, target_node: int, candidates: List[Tuple[int, float]], graph: GraphStore, top_k: int
    ) -> List[Tuple[int, float]]:
        results: List[Tuple[int, float]] = []
        for node_id, similarity in candidates:
            if self._params.overlap_validation_required:
                overlap = self._jaccard_overlap(target_node, node_id, graph)
                if overlap < self._params.jaccard_overlap_min:
                    continue
            results.append((node_id, similarity))
            if len(results) >= top_k:
                break
        return results

    def _jaccard_overlap(self, node_a: int, node_b: int, graph: GraphStore) -> float:
        neighbors_a = self._neighbor_set(node_a, graph)
        neighbors_b = self._neighbor_set(node_b, graph)
        if not neighbors_a and not neighbors_b:
            return 0.0
        union_size = len(neighbors_a | neighbors_b)
        if union_size == 0:
            return 0.0
        return len(neighbors_a & neighbors_b) / float(union_size)

    def _neighbor_set(self, node_id: int, graph: GraphStore) -> Set[int]:
        try:
            neighbors = graph.get_neighbors(node_id)
        except Exception:
            logger.exception("Failed to get neighbors for node %s", node_id)
            return set()
        filtered: Set[int] = set()
        for neighbor_id, edge in neighbors:
            if edge.confidence >= self._params.edge_confidence_min:
                filtered.add(neighbor_id)
        return filtered

    @staticmethod
    def _hash(embedding: np.ndarray, hyperplanes: np.ndarray) -> bytes:
        return (embedding @ hyperplanes.T >= 0).tobytes()


def _to_float_embedding(embedding: np.ndarray) -> np.ndarray:
    if not isinstance(embedding, np.ndarray):
        raise TypeError(f"Expected ndarray, got {type(embedding)}")
    if embedding.dtype == np.int8:
        return embedding.astype(np.float32) / 127.0
    result = embedding.astype(np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    return result


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    denom = norm_a * norm_b
    if denom < 1e-12:
        return 0.0
    dot = float(np.dot(vec_a, vec_b))
    result = dot / denom
    if not np.isfinite(result):
        return 0.0
    return np.float32(max(-1.0, min(1.0, result)))
