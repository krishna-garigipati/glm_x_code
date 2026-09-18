"""Toy Dataset for GLM-X Full Pipeline Testing - 30 Nodes with REAL SBERT Embeddings.

Creates a semantically rich knowledge graph with 30 concepts, 384-dim SBERT embeddings,
covering key relations for testing the updated walker (P1/P2: semantic similarity + chain-aware).
"""
from __future__ import annotations

import sys
import os
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.graph_component_implementation.dict_graph_store import DictGraphStore
from resonance.types import GraphStore as ResonanceGraphStore, Node as ResonanceNode, Edge as ResonanceEdge, Subgraph as ResonanceSubgraph


# ============================================================================
# 30-NODE TOY KNOWLEDGE GRAPH - 3 Domains, Real SBERT Embeddings
# ============================================================================

# 30 concepts across 3 focused domains
TOY_CONCEPTS: List[str] = [
    # Domain 1: Temperature (6 concepts)
    "hot", "cold", "warm", "freezing", "boiling", "lukewarm",
    # Domain 2: Animals (6 concepts)
    "dog", "cat", "animal", "mammal", "bird", "fish",
    # Domain 3: Vehicles (6 concepts)
    "car", "vehicle", "automobile", "wheel", "engine", "bicycle",
    # Domain 4: Weather (6 concepts)
    "rain", "cloud", "sun", "storm", "snow", "wind",
    # Domain 5: Food (6 concepts)
    "apple", "fruit", "bread", "vegetable", "meat", "water",
]

# Relations covering the 16 canonical types
TOY_EDGES: List[Tuple[str, str, str]] = [
    # ---- Temperature domain ----
    ("hot", "cold", "antonym"),
    ("hot", "warm", "has_property"),
    ("hot", "boiling", "has_property"),
    ("cold", "freezing", "has_property"),
    ("warm", "lukewarm", "has_property"),
    ("boiling", "hot", "has_property"),
    ("lukewarm", "warm", "has_property"),

    # ---- Animal domain ----
    ("dog", "animal", "is_a"),
    ("cat", "animal", "is_a"),
    ("dog", "mammal", "is_a"),
    ("cat", "mammal", "is_a"),
    ("bird", "animal", "is_a"),
    ("fish", "animal", "is_a"),
    ("dog", "cat", "associated_with"),

    # ---- Vehicle domain ----
    ("car", "vehicle", "is_a"),
    ("automobile", "car", "synonym"),
    ("car", "wheel", "part_of"),
    ("car", "engine", "part_of"),
    ("bicycle", "vehicle", "is_a"),
    ("wheel", "bicycle", "part_of"),

    # ---- Weather domain ----
    ("rain", "cloud", "associated_with"),
    ("storm", "rain", "has_property"),
    ("sun", "cloud", "antonym"),
    ("snow", "cold", "has_property"),
    ("wind", "storm", "associated_with"),

    # ---- Food domain ----
    ("apple", "fruit", "is_a"),
    ("apple", "food", "is_a"),  # food not in concepts, will skip
    ("bread", "food", "is_a"),
    ("vegetable", "food", "is_a"),
    ("meat", "food", "is_a"),
    ("water", "liquid", "has_property"),
    ("water", "drink", "is_a"),  # drink not in concepts

    # ---- Cross-domain ----
    ("hot", "sun", "associated_with"),
    ("cold", "snow", "associated_with"),
    ("rain", "water", "associated_with"),
    ("car", "engine", "part_of"),
]

# 16 canonical relations
ALL_16_RELATIONS = [
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near", "linguistic_maps",
]


class SimpleGraphStore(ResonanceGraphStore):
    """Simple in-memory graph store with 384-dim SBERT embeddings."""

    def __init__(self):
        self._nodes: Dict[int, ResonanceNode] = {}
        self._label_to_id: Dict[str, int] = {}
        self._id_to_label: Dict[int, str] = {}
        self._neighbors: Dict[int, List[Tuple[int, ResonanceEdge]]] = {}
        self._embeddings: Dict[int, np.ndarray] = {}
        self._edges_raw: List = []
        self._relation_set: set = set()
        self._next_id: int = 1

    def add_node(self, label: str, embedding: Optional[np.ndarray] = None, node_type: str = "concept") -> int:
        nid = self._next_id
        self._next_id += 1
        self._id_to_label[nid] = label
        self._label_to_id[label] = nid
        self._label_to_id.setdefault(label.strip().lower(), nid)
        self._nodes[nid] = ResonanceNode(
            id=nid, label=label, node_type=node_type,
            embedding=embedding if embedding is not None else np.zeros(384, dtype=np.float32),
            activation=0.5, use_count=0, create_time=time.time(),
        )
        if embedding is not None:
            self._embeddings[nid] = embedding
        return nid

    def add_edge(self, source: int, target: int, relation: str, strength: float = 0.9, confidence: float = 0.8) -> None:
        self._relation_set.add(relation)
        edge = ResonanceEdge(
            source=source, target=target, relation_type=relation,
            strength=strength, confidence=confidence,
            last_used=time.time(), frequency=1,
        )
        if source not in self._neighbors:
            self._neighbors[source] = []
        self._neighbors[source].append((target, edge))
        if target not in self._neighbors:
            self._neighbors[target] = []
        self._neighbors[target].append((source, edge))
        self._edges_raw.append(edge)

    def get_all_relations(self) -> List[str]:
        return sorted(self._relation_set)

    def get_node_count(self) -> int:
        return len(self._nodes)

    def get_edge_count(self) -> int:
        return len(self._edges_raw)

    def get_node(self, node_id: int) -> Optional[ResonanceNode]:
        return self._nodes.get(node_id)

    def get_neighbors(self, node_id: int) -> List[Tuple[int, ResonanceEdge]]:
        return self._neighbors.get(node_id, [])

    def get_all_nodes(self) -> List[ResonanceNode]:
        return list(self._nodes.values())

    def get_subgraph_by_embedding_similarity(self, query_embedding: np.ndarray, top_k: int = 100) -> ResonanceSubgraph:
        # Handle 2D query embeddings
        q_emb = query_embedding
        if q_emb.ndim > 1:
            q_emb = q_emb.flatten()

        scored = []
        for nid, emb in self._embeddings.items():
            sim = float(np.dot(q_emb, emb))
            scored.append((sim, nid))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        seed_ids = [nid for _, nid in top]

        included = set(seed_ids)
        expansion_cap = max(top_k * 3, 10)
        for e in self._edges_raw:
            if e.source in included or e.target in included:
                included.add(e.source)
                included.add(e.target)
                if len(included) >= expansion_cap:
                    break

        included = sorted(included)
        node_activations = {}
        for nid in included:
            sim = 0.0
            if nid in self._embeddings:
                sim = float(np.dot(q_emb, self._embeddings[nid]))
            node_activations[nid] = max(0.01, min(1.0, sim))

        subgraph_edges: List[Tuple[int, int, str]] = []
        edge_strengths: Dict[Tuple[int, int, str], float] = {}
        edge_confidences: Dict[Tuple[int, int, str], float] = {}
        for e in self._edges_raw:
            if e.source in included and e.target in included:
                key = (e.source, e.target, e.relation_type)
                subgraph_edges.append(key)
                edge_strengths[key] = e.strength
                edge_confidences[key] = e.confidence
                rev_key = (e.target, e.source, e.relation_type)
                subgraph_edges.append(rev_key)
                edge_strengths[rev_key] = e.strength
                edge_confidences[rev_key] = e.confidence

        return ResonanceSubgraph(
            nodes=included,
            node_activations=node_activations,
            edges=subgraph_edges,
            edge_strengths=edge_strengths,
            edge_confidences=edge_confidences,
            seed_nodes=seed_ids,
            tier_used=1,
            activation_energy=0.0,
            query_embedding=q_emb.astype(np.float32),
            timestamp=time.time(),
        )

    def get_label(self, node_id: int) -> str:
        return self._id_to_label.get(node_id, f"node_{node_id}")

    def get_embedding(self, node_id: int) -> Optional[np.ndarray]:
        return self._embeddings.get(node_id)

    def get_all_edges(self) -> List:
        return self._edges_raw


def build_toy_graph() -> Tuple[SimpleGraphStore, None]:
    """Build 30-node graph with REAL SBERT embeddings (no random noise)."""
    from sentence_transformers import SentenceTransformer

    # Load real SBERT model
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Build graph
    store = SimpleGraphStore()

    # Get REAL SBERT embeddings for all concepts
    print("Generating REAL SBERT embeddings for 30 concepts...")
    embeddings = sbert.encode(TOY_CONCEPTS, normalize_embeddings=True, convert_to_numpy=True)
    # embeddings shape: (30, 384)

    # Add nodes with REAL embeddings
    label_to_store_id = {}
    for i, label in enumerate(TOY_CONCEPTS):
        store_id = store.add_node(label, embeddings[i].astype(np.float32), "Concept")
        label_to_store_id[label] = store_id

    # Add edges
    for src_label, tgt_label, rel in TOY_EDGES:
        if src_label in label_to_store_id and tgt_label in label_to_store_id:
            src_id = label_to_store_id[src_label]
            tgt_id = label_to_store_id[tgt_label]
            store.add_edge(src_id, tgt_id, rel, strength=0.85, confidence=0.8)

    print(f"Built toy graph: {store.get_node_count()} nodes, {store.get_edge_count()} edges ({len(ALL_16_RELATIONS)} relation types)")
    return store, None


# ============================================================================
# TEST QUERIES - Matching the 30-node graph
# ============================================================================

TOY_QUERIES = [
    # Temperature domain
    {
        "question": "What is the opposite of hot?",
        "expected_chain": ["antonym"],
        "expected_nodes": ["hot", "cold"],
        "domain": "temperature",
    },
    {
        "question": "What is hot?",
        "expected_chain": ["has_property"],
        "expected_nodes": ["hot", "warm"],
        "domain": "temperature",
    },
    {
        "question": "What is freezing?",
        "expected_chain": ["has_property"],
        "expected_nodes": ["freezing", "cold"],
        "domain": "temperature",
    },

    # Animal domain
    {
        "question": "What is a dog?",
        "expected_chain": ["is_a"],
        "expected_nodes": ["dog", "animal"],
        "domain": "animals",
    },
    {
        "question": "What is a cat?",
        "expected_chain": ["is_a"],
        "expected_nodes": ["cat", "animal"],
        "domain": "animals",
    },
    {
        "question": "What is a mammal?",
        "expected_chain": ["is_a"],
        "expected_nodes": ["dog", "mammal"],
        "domain": "animals",
    },

    # Vehicle domain
    {
        "question": "What is a car?",
        "expected_chain": ["is_a"],
        "expected_nodes": ["car", "vehicle"],
        "domain": "vehicles",
    },
    {
        "question": "What is part of a car?",
        "expected_chain": ["part_of"],
        "expected_nodes": ["wheel", "car"],
        "domain": "vehicles",
    },
    {
        "question": "What is another word for automobile?",
        "expected_chain": ["synonym"],
        "expected_nodes": ["automobile", "car"],
        "domain": "vehicles",
    },

    # Weather domain
    {
        "question": "What is associated with rain?",
        "expected_chain": ["associated_with"],
        "expected_nodes": ["rain", "cloud"],
        "domain": "weather",
    },
    {
        "question": "What is the opposite of sun?",
        "expected_chain": ["antonym"],
        "expected_nodes": ["sun", "cloud"],
        "domain": "weather",
    },

    # Food domain
    {
        "question": "What is an apple?",
        "expected_chain": ["is_a"],
        "expected_nodes": ["apple", "fruit"],
        "domain": "food",
    },
    {
        "question": "What is water?",
        "expected_chain": ["has_property"],
        "expected_nodes": ["water", "liquid"],
        "domain": "food",
    },

    # Cross-domain
    {
        "question": "What is associated with hot?",
        "expected_chain": ["associated_with"],
        "expected_nodes": ["hot", "sun"],
        "domain": "cross",
    },
    {
        "question": "What is associated with cold?",
        "expected_chain": ["associated_with"],
        "expected_nodes": ["cold", "snow"],
        "domain": "cross",
    },

    # Multi-hop
    {
        "question": "What is a dog and what is it part of?",
        "expected_chain": ["is_a", "part_of"],
        "expected_nodes": ["dog", "animal"],
        "domain": "multi",
    },
]

if __name__ == "__main__":
    # Quick test
    print("Building 30-node graph with REAL SBERT embeddings...")
    store, _ = build_toy_graph()
    print(f"Done: {store.get_node_count()} nodes, {store.get_edge_count()} edges")