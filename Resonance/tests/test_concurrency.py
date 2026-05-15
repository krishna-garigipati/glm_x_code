from __future__ import annotations

"""Concurrency tests: Validate async safety, synchronization, and race-condition safety."""

import threading
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from ..engine import ResonanceEngine
from ..es_controller import EvolutionaryController
from ..tier1 import Tier1Resonance
from ..types import Subgraph
from .fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_loaded_configs,
    build_minimal_resonance_config,
    get_test_config_dir,
)
from .fixtures.toy_graph_builder import build_dense_graph, make_query_embedding
from .fixtures.toy_graph_store import ToyGraphStore


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


class TestConcurrentGetTheta:
    def test_concurrent_get_theta(self, engine):
        results: List[np.ndarray] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker():
            try:
                t = engine.get_theta()
                with lock:
                    results.append(t)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent get_theta errors: {errors}"
        assert len(results) == 20
        for r in results:
            assert r.shape == (48,)


class TestConcurrentSetTheta:
    def test_concurrent_set_theta(self, engine):
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker(val: float):
            try:
                theta = np.full(48, val, dtype=np.float32)
                engine.set_theta(theta)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(float(i),)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent set_theta errors: {errors}"


class TestConcurrentES:
    def test_concurrent_propose_mutation(self, engine):
        proposals: List[np.ndarray] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker():
            try:
                p = engine.propose_theta_mutation()
                with lock:
                    proposals.append(p)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent propose errors: {errors}"
        for p in proposals:
            assert p.shape == (48,)
            assert np.isfinite(p).all()

    def test_concurrent_es_update(self, engine):
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker():
            try:
                theta = engine.get_theta()
                engine.update_es_with_reward(0.5, theta)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestConcurrentResonate:
    def test_concurrent_tier1_resonate(self, engine):
        g = build_dense_graph(node_count=15, edge_density=0.2)
        results: List[Subgraph] = []
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker(seed: int):
            try:
                q = make_query_embedding()
                r = engine.resonate(q, g, [seed], tier=1)
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent resonate errors: {errors}"
        assert len(results) == 8
        for r in results:
            assert isinstance(r, Subgraph)

    def test_concurrent_mixed_operations(self, engine):
        g = build_dense_graph(node_count=10, edge_density=0.15)
        errors: List[Exception] = []
        lock = threading.Lock()

        def resonate_worker():
            try:
                q = make_query_embedding()
                r = engine.resonate(q, g, [0], tier=1)
            except Exception as e:
                with lock:
                    errors.append(e)

        def theta_worker():
            try:
                t = engine.get_theta()
                engine.set_theta(t + 0.01)
                engine.propose_theta_mutation()
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=resonate_worker))
            threads.append(threading.Thread(target=theta_worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Mixed concurrent errors: {errors}"


class TestConcurrentAnalogy:
    def test_concurrent_lsh_build(self):
        from ..analogy import AnalogyFinder
        from ..config import AnalogyParameters
        cc = build_minimal_core_config()
        ap = AnalogyParameters(lsh_bands=8, lsh_tables=4, temp_edge_strength=0.5,
                                overlap_validation_required=False, jaccard_overlap_min=0.3,
                                mini_propagation_steps=2, edge_confidence_min=0.5)
        af = AnalogyFinder(cc, ap)
        graph = ToyGraphStore()
        from ..types import Node
        rng = np.random.RandomState(0)
        for i in range(20):
            graph.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                                 embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                                 activation=0.01, use_count=0, create_time=0.0))
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker(nid: int):
            try:
                af.get_analogy_leaps(nid, graph, top_k=3)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent analogy errors: {errors}"


class TestConcurrentClearCache:
    def test_concurrent_clear_analogy_cache(self, engine):
        from ..analogy import AnalogyFinder
        from ..config import AnalogyParameters
        cc = build_minimal_core_config()
        ap = AnalogyParameters(lsh_bands=4, lsh_tables=2, temp_edge_strength=0.5,
                                overlap_validation_required=False, jaccard_overlap_min=0.3,
                                mini_propagation_steps=2, edge_confidence_min=0.5)
        af = AnalogyFinder(cc, ap)
        errors: List[Exception] = []
        lock = threading.Lock()

        def worker():
            try:
                af.clear_cache()
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
