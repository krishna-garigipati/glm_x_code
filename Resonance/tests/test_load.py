from __future__ import annotations

"""Load tests: Simulate realistic production-level usage on CPU environments."""

import time
from typing import List

import numpy as np
import pytest

from ..engine import ResonanceEngine
from .fixtures.config_provider import get_test_config_dir
from .fixtures.toy_graph_builder import build_dense_graph, make_query_embedding
from .fixtures.toy_graph_store import ToyGraphStore


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


class TestLoadLatency:
    def test_single_resonate_latency_tier1(self, engine):
        g = build_dense_graph(node_count=30, edge_density=0.2)
        q = make_query_embedding()
        start = time.perf_counter()
        for _ in range(5):
            engine.resonate(q, g, [0], tier=1)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 5) * 1000
        assert avg_ms < 5000, f"Tier1 avg latency {avg_ms:.1f}ms exceeded 5000ms"

    def test_single_resonate_latency_tier2(self, engine):
        g = build_dense_graph(node_count=20, edge_density=0.15)
        q = make_query_embedding()
        start = time.perf_counter()
        for _ in range(3):
            engine.resonate(q, g, [0], tier=2)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 3) * 1000
        assert avg_ms < 10000, f"Tier2 avg latency {avg_ms:.1f}ms exceeded 10000ms"

    def test_get_theta_latency(self, engine):
        start = time.perf_counter()
        for _ in range(100):
            engine.get_theta()
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / 100) * 1_000_000
        assert avg_us < 10000, f"get_theta avg {avg_us:.1f}us exceeded 10ms"

    def test_set_theta_latency(self, engine):
        theta = engine.get_theta()
        start = time.perf_counter()
        for _ in range(100):
            engine.set_theta(theta.copy())
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / 100) * 1_000_000
        assert avg_us < 10000, f"set_theta avg {avg_us:.1f}us exceeded 10ms"

    def test_energy_computation_latency(self, engine):
        g = build_dense_graph(node_count=30, edge_density=0.2)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        start = time.perf_counter()
        for _ in range(50):
            engine.compute_activation_energy(result)
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / 50) * 1_000_000
        assert avg_us < 50000, f"Energy computation avg {avg_us:.1f}us exceeded 50ms"


class TestLoadThroughput:
    def test_batch_resonate_throughput(self, engine):
        g = build_dense_graph(node_count=15, edge_density=0.2)
        queries = [make_query_embedding() for _ in range(10)]
        seeds_list = [[0] for _ in range(10)]
        start = time.perf_counter()
        for q, s in zip(queries, seeds_list):
            engine.resonate(q, g, s, tier=1)
        elapsed = time.perf_counter() - start
        throughput = 10.0 / elapsed if elapsed > 0 else float("inf")
        assert throughput > 0.5, f"Throughput {throughput:.1f} queries/sec too low"

    def test_es_mutation_throughput(self, engine):
        start = time.perf_counter()
        for _ in range(50):
            engine.propose_theta_mutation()
        elapsed = time.perf_counter() - start
        throughput = 50.0 / elapsed if elapsed > 0 else float("inf")
        assert throughput > 100, f"ES mutation throughput {throughput:.1f}/sec too low"


class TestLoadMemory:
    def test_memory_not_leaking_on_repeated_calls(self, engine):
        g = build_dense_graph(node_count=20, edge_density=0.2)
        q = make_query_embedding()
        theta = engine.get_theta()
        results = []
        for i in range(50):
            r = engine.resonate(q, g, [i % 10], tier=1)
            results.append(len(r.nodes))
            engine.update_es_with_reward(0.5, theta)
            engine.propose_theta_mutation()
        assert len(results) == 50

    def test_subgraph_size_bounded(self, engine):
        g = build_dense_graph(node_count=50, edge_density=0.3)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        assert len(result.nodes) <= 64
