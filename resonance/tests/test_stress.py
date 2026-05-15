from __future__ import annotations

"""Stress tests: Push component beyond normal operating limits."""

import time
from typing import List

import numpy as np
import pytest

from ..engine import ResonanceEngine
from ..es_controller import EvolutionaryController
from ..tier1 import Tier1Resonance
from ..tier2 import Tier2Resonance
from ..types import Subgraph
from .fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_loaded_configs,
    build_minimal_resonance_config,
    get_test_config_dir,
)
from .fixtures.toy_graph_builder import build_dense_graph, make_query_embedding
from .fixtures.toy_graph_store import ToyGraphStore
from ..types import Node, Edge


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


class TestStressLargeGraph:
    def test_large_graph_tier1_does_not_crash(self, engine):
        g = build_dense_graph(node_count=200, edge_density=0.1)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        assert isinstance(result, Subgraph)

    def test_large_graph_tier2_does_not_crash(self, engine):
        g = build_dense_graph(node_count=100, edge_density=0.08)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=2)
        assert isinstance(result, Subgraph)


class TestStressManySeeds:
    def test_many_seeds_tier1(self, engine):
        g = build_dense_graph(node_count=50, edge_density=0.2)
        q = make_query_embedding()
        seeds = list(range(10))
        result = engine.resonate(q, g, seeds, tier=1)
        assert isinstance(result, Subgraph)

    def test_single_large_seed_batch(self, engine):
        g = build_dense_graph(node_count=80, edge_density=0.15)
        q = make_query_embedding()
        seeds = list(range(20))
        result = engine.resonate(q, g, seeds, tier=2)
        assert isinstance(result, Subgraph)


class TestStressStarvedGraph:
    def test_very_low_edge_weights(self, engine):
        g = ToyGraphStore()
        rng = np.random.RandomState(0)
        for i in range(20):
            g.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                             embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                             activation=0.01, use_count=0, create_time=0.0))
        for i in range(19):
            g.add_edge(i, i + 1, "is_a", strength=1e-6, confidence=1e-6, frequency=1)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        assert isinstance(result, Subgraph)

    def test_maximum_edge_weights(self, engine):
        g = ToyGraphStore()
        rng = np.random.RandomState(0)
        for i in range(10):
            g.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                             embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                             activation=0.01, use_count=0, create_time=0.0))
        for i in range(9):
            g.add_edge(i, i + 1, "is_a", strength=1.0, confidence=1.0, frequency=1000)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        assert isinstance(result, Subgraph)
        for v in result.node_activations.values():
            assert 0.01 <= v <= 1.0


class TestStressRapidReinit:
    def test_repeated_engine_init(self):
        for _ in range(5):
            eng = ResonanceEngine(get_test_config_dir())
            assert eng is not None

    def test_rapid_theta_updates(self, engine):
        for _ in range(100):
            theta = engine.get_theta()
            engine.set_theta(theta + np.random.randn(48).astype(np.float32) * 0.001)


class TestStressExtremeIterations:
    def test_many_iterations_tier1(self, engine):
        g = build_dense_graph(node_count=10, edge_density=0.3)
        q = make_query_embedding()
        result = engine._tier1.resonate(q, g, [0], max_iterations=20)
        assert isinstance(result[0], Subgraph)

    def test_many_iterations_tier2(self, engine):
        g = build_dense_graph(node_count=10, edge_density=0.2)
        q = make_query_embedding()
        result, _ = engine._tier2.resonate(q, g, [0])
        assert isinstance(result, Subgraph)


class TestStressRapidProposal:
    def test_many_proposals_no_update(self, engine):
        proposals = []
        for _ in range(1000):
            p = engine.propose_theta_mutation()
            proposals.append(p)
        assert all(np.isfinite(p).all() for p in proposals)


class TestStressTimeLimit:
    def test_resonate_tier1_time_budget(self, engine):
        g = build_dense_graph(node_count=50, edge_density=0.2)
        q = make_query_embedding()
        start = time.perf_counter()
        for i in range(5):
            engine.resonate(q, g, [i], tier=1)
        elapsed = time.perf_counter() - start
        assert elapsed < 30.0, f"5 Tier1 resonances took {elapsed:.1f}s"
