from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...types import Node
from .toy_graph_store import ToyGraphStore

SEED = 42
RNG = np.random.RandomState(SEED)

_NODE_TYPES = ["Concept", "Entity", "TemporalAnchor", "LinguisticToken", "ContextLabel", "Pattern"]

_RELATIONS = [
    "is_a", "has_property", "causes", "caused_by",
    "follows", "precedes", "contradicts", "supports",
    "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident",
    "spatial_near", "linguistic_maps",
]


def _make_embedding() -> np.ndarray:
    return RNG.randint(-128, 128, size=(32,), dtype=np.int8)


def _make_query_embedding() -> np.ndarray:
    emb = RNG.randn(384).astype(np.float32)
    emb = emb / float(np.linalg.norm(emb))
    return emb


def create_node(
    node_id: int,
    label: str = "",
    node_type: str = "Concept",
    embedding: Optional[np.ndarray] = None,
    activation: float = 0.01,
) -> Node:
    return Node(
        id=node_id,
        label=label or f"node_{node_id}",
        node_type=node_type if node_type in _NODE_TYPES else "Concept",
        embedding=embedding if embedding is not None else _make_embedding(),
        activation=float(np.clip(activation, 0.01, 1.0)),
        use_count=0,
        create_time=time.time(),
    )


def build_empty_graph() -> ToyGraphStore:
    return ToyGraphStore()


def build_single_node_graph() -> ToyGraphStore:
    g = ToyGraphStore()
    g.add_node(create_node(1, "lonely_node"))
    return g


def build_two_node_graph() -> ToyGraphStore:
    g = ToyGraphStore()
    g.add_node(create_node(1, "alpha"))
    g.add_node(create_node(2, "beta"))
    g.add_edge(1, 2, "is_a", strength=0.8, confidence=0.9)
    return g


def build_chain_graph(length: int = 5, relation: str = "follows") -> ToyGraphStore:
    g = ToyGraphStore()
    for i in range(length):
        g.add_node(create_node(i, f"chain_{i}"))
    for i in range(length - 1):
        g.add_edge(i, i + 1, relation, strength=0.6 + 0.1 * RNG.rand(), confidence=0.7 + 0.2 * RNG.rand())
    return g


def build_star_graph(center_id: int = 0, leaf_count: int = 6) -> ToyGraphStore:
    g = ToyGraphStore()
    g.add_node(create_node(center_id, "center"))
    for i in range(leaf_count):
        lid = center_id + 1 + i
        g.add_node(create_node(lid, f"leaf_{i}"))
        g.add_edge(center_id, lid, "associated_with", strength=0.7, confidence=0.8)
    return g


def build_cluster_graph(
    num_clusters: int = 3, nodes_per_cluster: int = 5, bridge_edges: int = 2
) -> ToyGraphStore:
    g = ToyGraphStore()
    nid = 0
    cluster_centers = []
    for c in range(num_clusters):
        center = nid
        cluster_centers.append(center)
        g.add_node(create_node(center, f"cluster_{c}_center"))
        nid += 1
        for _ in range(nodes_per_cluster - 1):
            g.add_node(create_node(nid, f"cluster_{c}_member_{nid}"))
            g.add_edge(center, nid, "part_of", strength=0.9, confidence=0.95)
            nid += 1
        for i in range(nodes_per_cluster - 1):
            for j in range(i + 1, nodes_per_cluster - 1):
                g.add_edge(center + 1 + i, center + 1 + j, "associated_with",
                           strength=0.3 + 0.3 * RNG.rand(), confidence=0.5 + 0.3 * RNG.rand())
    for _ in range(bridge_edges):
        if len(cluster_centers) >= 2:
            a = int(RNG.randint(0, len(cluster_centers)))
            b = int(RNG.randint(0, len(cluster_centers)))
            while b == a:
                b = int(RNG.randint(0, len(cluster_centers)))
            g.add_edge(cluster_centers[a], cluster_centers[b], "supports", strength=0.5, confidence=0.6)
    return g


def build_disconnected_graph() -> ToyGraphStore:
    g = ToyGraphStore()
    for i in range(3):
        g.add_node(create_node(i, f"isolated_{i}"))
    return g


def build_dense_graph(node_count: int = 20, edge_density: float = 0.3) -> ToyGraphStore:
    g = ToyGraphStore()
    for i in range(node_count):
        g.add_node(create_node(i, f"dense_{i}"))
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if RNG.rand() < edge_density:
                rel = _RELATIONS[int(RNG.randint(0, len(_RELATIONS)))]
                g.add_edge(i, j, rel, strength=0.3 + 0.7 * RNG.rand(), confidence=0.3 + 0.7 * RNG.rand())
    return g


def build_self_loop_graph() -> ToyGraphStore:
    g = ToyGraphStore()
    g.add_node(create_node(1, "self_looper"))
    g.add_edge(1, 1, "associated_with", strength=1.0, confidence=1.0)
    return g


def build_multi_edge_graph() -> ToyGraphStore:
    g = ToyGraphStore()
    g.add_node(create_node(1, "multi_a"))
    g.add_node(create_node(2, "multi_b"))
    g.add_edge(1, 2, "is_a", strength=0.9, confidence=0.8)
    g.add_edge(1, 2, "causes", strength=0.3, confidence=0.4)
    g.add_edge(1, 2, "contradicts", strength=0.1, confidence=0.2)
    return g


def build_contradiction_graph() -> ToyGraphStore:
    g = ToyGraphStore()
    g.add_node(create_node(1, "premise_a"))
    g.add_node(create_node(2, "premise_b"))
    g.add_node(create_node(3, "conclusion"))
    g.add_edge(1, 3, "causes", strength=0.9, confidence=0.9)
    g.add_edge(2, 3, "contradicts", strength=0.8, confidence=0.8)
    g.add_edge(1, 2, "contradicts", strength=0.7, confidence=0.9)
    return g


def make_query_embedding() -> np.ndarray:
    return _make_query_embedding()
