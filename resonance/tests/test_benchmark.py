from __future__ import annotations

"""Benchmark / Evaluation tests: Component performance metrics and reproducibility."""

import statistics
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from ..engine import ResonanceEngine
from ..config import build_default_theta
from ..temporal import compute_temporal_factor
from ..energy import compute_activation_energy
from ..es_controller import EvolutionaryController
from .fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_loaded_configs,
    build_minimal_resonance_config,
    get_test_config_dir,
)
from .fixtures.toy_graph_builder import (
    build_chain_graph,
    build_dense_graph,
    build_star_graph,
    make_query_embedding,
)
from .fixtures.toy_data import build_animal_kingdom_graph


BENCHMARK_ITERATIONS = 10


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


class BenchmarkMetrics:
    def __init__(self):
        self.latencies: List[float] = []
        self.name: str = ""

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def p99_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)] * 1000

    @property
    def min_ms(self) -> float:
        return min(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.latencies) * 1000 if len(self.latencies) > 1 else 0.0

    def add(self, seconds: float) -> None:
        self.latencies.append(seconds)

    def report(self) -> str:
        return (
            f"  Mean:   {self.mean_ms:.2f}ms\n"
            f"  Median: {self.median_ms:.2f}ms\n"
            f"  P99:    {self.p99_ms:.2f}ms\n"
            f"  Min:    {self.min_ms:.2f}ms\n"
            f"  Max:    {self.max_ms:.2f}ms\n"
            f"  Std:    {self.std_ms:.2f}ms\n"
            f"  N:      {len(self.latencies)}"
        )


class TestBenchmarkTier1:
    def test_tier1_small_graph_latency(self, engine):
        metrics = BenchmarkMetrics()
        metrics.name = "Tier1 Small Graph"
        g = build_star_graph(center_id=0, leaf_count=5)
        q = make_query_embedding()
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            engine.resonate(q, g, [0], tier=1)
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 1000

    def test_tier1_medium_graph_latency(self, engine):
        metrics = BenchmarkMetrics()
        metrics.name = "Tier1 Medium Graph"
        g = build_dense_graph(node_count=30, edge_density=0.2)
        q = make_query_embedding()
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            engine.resonate(q, g, [0], tier=1)
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 3000


class TestBenchmarkTier2:
    def test_tier2_small_graph_latency(self, engine):
        metrics = BenchmarkMetrics()
        metrics.name = "Tier2 Small Graph"
        g = build_star_graph(center_id=0, leaf_count=5)
        q = make_query_embedding()
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            engine.resonate(q, g, [0], tier=2)
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 5000


class TestBenchmarkTheta:
    def test_get_theta_latency(self, engine):
        metrics = BenchmarkMetrics()
        metrics.name = "get_theta"
        for _ in range(BENCHMARK_ITERATIONS * 10):
            start = time.perf_counter()
            engine.get_theta()
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 1.0

    def test_set_theta_latency(self, engine):
        metrics = BenchmarkMetrics()
        metrics.name = "set_theta"
        theta = engine.get_theta()
        for _ in range(BENCHMARK_ITERATIONS * 10):
            start = time.perf_counter()
            engine.set_theta(theta.copy())
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 1.0


class TestBenchmarkES:
    def test_propose_mutation_latency(self, engine):
        metrics = BenchmarkMetrics()
        metrics.name = "propose_theta_mutation"
        for _ in range(BENCHMARK_ITERATIONS * 10):
            start = time.perf_counter()
            engine.propose_theta_mutation()
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 5.0

    def test_es_update_latency(self, engine):
        metrics = BenchmarkMetrics()
        metrics.name = "update_es_with_reward"
        theta = engine.get_theta()
        for _ in range(BENCHMARK_ITERATIONS * 5):
            start = time.perf_counter()
            engine.update_es_with_reward(0.5, theta)
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")


class TestBenchmarkEnergy:
    def test_energy_computation_latency(self, engine):
        g = build_dense_graph(node_count=20, edge_density=0.2)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        metrics = BenchmarkMetrics()
        metrics.name = "compute_activation_energy"
        for _ in range(BENCHMARK_ITERATIONS * 10):
            start = time.perf_counter()
            engine.compute_activation_energy(result)
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 5.0


class TestBenchmarkTemporal:
    def test_temporal_computation_latency(self):
        import time as ttime
        metrics = BenchmarkMetrics()
        metrics.name = "compute_temporal_factor"
        now = ttime.time()
        for _ in range(BENCHMARK_ITERATIONS * 100):
            start = time.perf_counter()
            compute_temporal_factor(now, 10, 0.5, 20)
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 0.1


class TestBenchmarkAnalogy:
    def test_analogy_leaps_latency(self, engine):
        g = build_dense_graph(node_count=30, edge_density=0.15)
        metrics = BenchmarkMetrics()
        metrics.name = "get_analogy_leaps"
        for nid in range(min(10, g.node_count())):
            start = time.perf_counter()
            engine.get_analogy_leaps(nid, g, top_k=5)
            metrics.add(time.perf_counter() - start)
        print(f"\n--- {metrics.name} ---\n{metrics.report()}")
        assert metrics.mean_ms < 5000


class TestBenchmarkReproducibility:
    def test_deterministic_tier1_results(self, engine):
        g = build_chain_graph(length=6, relation="follows")
        q = make_query_embedding()
        results = []
        for _ in range(5):
            r = engine.resonate(q, g, [0], tier=1)
            results.append(r)
        for i in range(1, len(results)):
            assert results[i].nodes == results[0].nodes
            for k in results[0].node_activations:
                assert abs(results[i].node_activations[k] - results[0].node_activations[k]) < 5e-4

    def test_deterministic_energy(self, engine):
        g = build_star_graph(center_id=0, leaf_count=4)
        q = make_query_embedding()
        r = engine.resonate(q, g, [0], tier=1)
        energies = [engine.compute_activation_energy(r) for _ in range(5)]
        for i in range(1, len(energies)):
            assert abs(energies[i] - energies[0]) < 1e-6
