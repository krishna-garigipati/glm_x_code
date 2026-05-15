from __future__ import annotations

import time

import numpy as np
import pytest

from ...types import Edge, Node, Subgraph


class TestNode:
    def test_minimal_creation(self):
        emb = np.zeros(32, dtype=np.int8)
        n = Node(id=1, label="test", node_type="Concept", embedding=emb,
                  activation=0.5, use_count=0, create_time=time.time())
        assert n.id == 1
        assert n.label == "test"
        assert n.node_type == "Concept"
        assert n.embedding.shape == (32,)

    def test_immutable(self):
        emb = np.zeros(32, dtype=np.int8)
        n = Node(id=1, label="test", node_type="Concept", embedding=emb,
                  activation=0.5, use_count=0, create_time=time.time())
        with pytest.raises((AttributeError, TypeError)):
            n.id = 2

    def test_optional_sense_id_default_none(self):
        emb = np.zeros(32, dtype=np.int8)
        n = Node(id=1, label="test", node_type="Concept", embedding=emb,
                  activation=0.5, use_count=0, create_time=time.time())
        assert n.sense_id is None

    def test_sense_id_set(self):
        emb = np.zeros(32, dtype=np.int8)
        n = Node(id=1, label="test", node_type="Concept", embedding=emb,
                  activation=0.5, use_count=0, create_time=time.time(), sense_id=42)
        assert n.sense_id == 42


class TestEdge:
    def test_minimal_creation(self):
        e = Edge(source=1, target=2, relation_type="is_a", strength=0.8,
                  confidence=0.9, last_used=time.time(), frequency=5)
        assert e.source == 1
        assert e.target == 2
        assert e.relation_type == "is_a"

    def test_immutable(self):
        e = Edge(source=1, target=2, relation_type="is_a", strength=0.8,
                  confidence=0.9, last_used=time.time(), frequency=5)
        with pytest.raises((AttributeError, TypeError)):
            e.source = 99

    def test_strength_range(self):
        e = Edge(source=1, target=2, relation_type="causes", strength=0.5,
                  confidence=0.5, last_used=time.time(), frequency=1)
        assert 0.0 <= e.strength <= 1.0

    def test_confidence_range(self):
        e = Edge(source=1, target=2, relation_type="causes", strength=0.5,
                  confidence=0.5, last_used=time.time(), frequency=1)
        assert 0.0 <= e.confidence <= 1.0


class TestSubgraph:
    def test_minimal_creation(self):
        sg = Subgraph(
            nodes=[1, 2, 3],
            node_activations={1: 0.8, 2: 0.3, 3: 0.1},
            edges=[(1, 2, "is_a"), (2, 3, "causes")],
            edge_strengths={(1, 2, "is_a"): 0.9, (2, 3, "causes"): 0.7},
            edge_confidences={(1, 2, "is_a"): 0.8, (2, 3, "causes"): 0.6},
            seed_nodes=[1],
            tier_used=1,
            activation_energy=1.5,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )
        assert len(sg.nodes) == 3
        assert sg.tier_used == 1

    def test_all_edges_have_strengths(self):
        sg = Subgraph(
            nodes=[1, 2], node_activations={1: 0.5, 2: 0.3},
            edges=[(1, 2, "is_a")],
            edge_strengths={(1, 2, "is_a"): 0.9},
            edge_confidences={(1, 2, "is_a"): 0.8},
            seed_nodes=[1], tier_used=1, activation_energy=0.5,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )
        assert len(sg.edge_strengths) == len(sg.edges)

    def test_immutable(self):
        sg = Subgraph(
            nodes=[1], node_activations={1: 0.5}, edges=[],
            edge_strengths={}, edge_confidences={},
            seed_nodes=[1], tier_used=1, activation_energy=0.0,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )
        with pytest.raises((AttributeError, TypeError)):
            sg.nodes = []

    def test_empty_nodes_allowed(self):
        sg = Subgraph(
            nodes=[], node_activations={}, edges=[],
            edge_strengths={}, edge_confidences={},
            seed_nodes=[], tier_used=1, activation_energy=0.0,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )
        assert sg.nodes == []


def _to_attrdict(d):
    if isinstance(d, dict):
        from ...config_loader import AttrDict
        return AttrDict({k: _to_attrdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_attrdict(item) for item in d]
    return d
