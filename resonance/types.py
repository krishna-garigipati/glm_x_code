from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable
import numpy as np


@dataclass(frozen=True)
class Node:
    id: int
    label: str
    node_type: str
    embedding: np.ndarray
    activation: float
    use_count: int
    create_time: float
    sense_id: Optional[int] = None


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    relation_type: str
    strength: float
    confidence: float
    last_used: float
    frequency: int


@dataclass(frozen=True)
class Subgraph:
    nodes: List[int]
    node_activations: Dict[int, float]
    edges: List[Tuple[int, int, str]]
    edge_strengths: Dict[Tuple[int, int, str], float]
    edge_confidences: Dict[Tuple[int, int, str], float]
    seed_nodes: List[int]
    tier_used: int
    activation_energy: float
    query_embedding: np.ndarray
    timestamp: float
    node_embeddings: Optional[Dict[int, np.ndarray]] = None


@runtime_checkable
class GraphStore(Protocol):
    def get_node(self, node_id: int) -> Optional[Node]: ...
    def get_neighbors(self, node_id: int) -> List[Tuple[int, Edge]]: ...
    def get_all_nodes(self) -> List[Node]: ...
    def get_subgraph_by_embedding_similarity(self, embedding: np.ndarray, top_k: int) -> Subgraph: ...
