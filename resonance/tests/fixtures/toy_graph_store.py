from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...types import Edge, GraphStore, Node, Subgraph


class ToyGraphStore:
    def __init__(self):
        self._nodes: Dict[int, Node] = {}
        self._neighbors: Dict[int, List[Tuple[int, Edge]]] = {}

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node

    def add_edge(
        self,
        source: int,
        target: int,
        relation_type: str,
        strength: float = 0.5,
        confidence: float = 0.5,
        last_used: Optional[float] = None,
        frequency: int = 1,
    ) -> None:
        edge = Edge(
            source=source,
            target=target,
            relation_type=relation_type,
            strength=float(np.clip(strength, 0.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            last_used=last_used if last_used is not None else time.time(),
            frequency=frequency,
        )
        if source not in self._neighbors:
            self._neighbors[source] = []
        self._neighbors[source].append((target, edge))

        edge_rev = Edge(
            source=target,
            target=source,
            relation_type=relation_type,
            strength=float(np.clip(strength, 0.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            last_used=last_used if last_used is not None else time.time(),
            frequency=frequency,
        )
        if target not in self._neighbors:
            self._neighbors[target] = []
        self._neighbors[target].append((source, edge_rev))

    def remove_node(self, node_id: int) -> None:
        self._nodes.pop(node_id, None)
        self._neighbors.pop(node_id, None)
        for src in list(self._neighbors.keys()):
            self._neighbors[src] = [(nid, e) for nid, e in self._neighbors[src] if nid != node_id]

    def get_node(self, node_id: int) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_neighbors(self, node_id: int) -> List[Tuple[int, Edge]]:
        return self._neighbors.get(node_id, [])

    def get_all_nodes(self) -> List[Node]:
        return list(self._nodes.values())

    def get_subgraph_by_embedding_similarity(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> Subgraph:
        if not self._nodes:
            return Subgraph(
                nodes=[], node_activations={}, edges=[],
                edge_strengths={}, edge_confidences={},
                seed_nodes=[], tier_used=1,
                activation_energy=0.0,
                query_embedding=np.ascontiguousarray(query_embedding.astype(np.float32)),
                timestamp=time.time(),
            )
        scored = []
        for nid, node in self._nodes.items():
            emb = node.embedding.astype(np.float32)
            if emb.ndim == 1 and emb.shape[0] == 32:
                emb_logical = emb / 127.0
            else:
                emb_logical = emb
            sim = _cosine_similarity_32(query_embedding, emb_logical)
            scored.append((nid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [nid for nid, _ in scored[:top_k]]
        node_activations = {nid: self._nodes[nid].activation for nid in top_ids}
        edge_list = []
        edge_strengths = {}
        edge_confidences = {}
        node_set = set(top_ids)
        for nid in top_ids:
            for neighbor_id, edge in self._neighbors.get(nid, []):
                if neighbor_id in node_set:
                    key = (nid, neighbor_id, edge.relation_type)
                    if key not in edge_list:
                        edge_list.append(key)
                        edge_strengths[key] = edge.strength
                        edge_confidences[key] = edge.confidence
        return Subgraph(
            nodes=top_ids,
            node_activations=node_activations,
            edges=edge_list,
            edge_strengths=edge_strengths,
            edge_confidences=edge_confidences,
            seed_nodes=top_ids[:1] if top_ids else [],
            tier_used=1,
            activation_energy=0.0,
            query_embedding=np.ascontiguousarray(query_embedding.astype(np.float32)),
            timestamp=time.time(),
        )

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        total = 0
        for neighbors in self._neighbors.values():
            total += len(neighbors)
        return total // 2

    def clear(self) -> None:
        self._nodes.clear()
        self._neighbors.clear()


def _cosine_similarity_32(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a.ravel().astype(np.float32)
    b = vec_b.ravel().astype(np.float32)
    min_len = min(a.shape[0], b.shape[0])
    a = a[:min_len]
    b = b[:min_len]
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b
    if denom < 1e-12:
        return 0.0
    dot = float(np.dot(a, b))
    result = dot / denom
    if not np.isfinite(result):
        return 0.0
    return np.float32(max(-1.0, min(1.0, result)))
