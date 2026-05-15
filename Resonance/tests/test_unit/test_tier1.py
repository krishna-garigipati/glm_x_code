from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from ...config import AlgorithmConfig, CoreConfig, TemporalConfig, TierConfig
from ...tier1 import Tier1Resonance
from ...types import Edge, Node, Subgraph
from ..fixtures.config_provider import build_minimal_core_config, build_minimal_resonance_config


def make_tier1(
    core: Optional[CoreConfig] = None,
    tier: Optional[TierConfig] = None,
    algorithm: Optional[AlgorithmConfig] = None,
    temporal: Optional[TemporalConfig] = None,
) -> Tier1Resonance:
    rc = build_minimal_resonance_config()
    return Tier1Resonance(
        core_config=core or build_minimal_core_config(),
        algorithm=algorithm or rc.algorithm,
        temporal=temporal or rc.temporal,
        tier_config=tier or rc.tier1,
        log_activation_history=False,
        history_buffer_size=1000,
    )


def _make_graph(edges: Optional[List[Tuple[int, int, str, float, float, int]]] = None):
    from ..fixtures.toy_graph_store import ToyGraphStore
    g = ToyGraphStore()
    rng = np.random.RandomState(0)
    nodes_seen = set()
    if edges:
        for src, tgt, rel, strength, conf, freq in edges:
            if src not in nodes_seen:
                g.add_node(Node(id=src, label=f"n{src}", node_type="Concept",
                                 embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                                 activation=0.01, use_count=0, create_time=0.0))
                nodes_seen.add(src)
            if tgt not in nodes_seen:
                g.add_node(Node(id=tgt, label=f"n{tgt}", node_type="Concept",
                                 embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                                 activation=0.01, use_count=0, create_time=0.0))
                nodes_seen.add(tgt)
            g.add_edge(src, tgt, rel, strength=strength, confidence=conf, frequency=freq)
    if not nodes_seen:
        for i in range(5):
            g.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                             embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                             activation=0.01, use_count=0, create_time=0.0))
    return g


class TestTier1ResonanceInit:
    def test_initializes(self):
        t1 = make_tier1()
        assert t1 is not None
        assert t1._algorithm.propagation_type == "wilson_cowan"
        assert t1._tier.top_k == 64
        assert t1._tier.max_iterations == 4


class TestTier1Resonate:
    def test_resonate_returns_subgraph(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([(0, 1, "is_a", 0.8, 0.9, 5), (1, 2, "causes", 0.7, 0.8, 3)])
        subgraph, history = t1.resonate(query_embedding, graph, [0])
        assert isinstance(subgraph, Subgraph)
        assert len(subgraph.nodes) > 0
        assert subgraph.tier_used == 1

    def test_resonate_with_single_seed_produces_activations(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10), (1, 2, "has_property", 0.8, 0.9, 5)])
        subgraph, _ = t1.resonate(query_embedding, graph, [0])
        assert 0 in subgraph.node_activations
        for v in subgraph.node_activations.values():
            assert 0.01 <= v <= 1.0

    def test_propagation_with_max_iterations_override(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10)])
        subgraph, history = t1.resonate(query_embedding, graph, [0], max_iterations=1)
        assert isinstance(subgraph, Subgraph)

    def test_empty_graph_returns_only_seed(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([])
        subgraph, _ = t1.resonate(query_embedding, graph, [0])
        assert 0 in subgraph.node_activations

    def test_initial_activations_override(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10)])
        initial = {0: 0.5, 1: 0.3}
        subgraph, _ = t1.resonate(query_embedding, graph, [0], initial_activations=initial)
        assert subgraph.node_activations.get(0, 0) >= 0.5

    def test_activation_history_length(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10)])
        _, history = t1.resonate(query_embedding, graph, [0], max_iterations=3)
        assert len(history) <= 3

    def test_invalid_max_iterations_raises(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([])
        with pytest.raises(ValueError, match="max_iterations"):
            t1.resonate(query_embedding, graph, [0], max_iterations=0)

    def test_nan_query_embedding_raises(self):
        t1 = make_tier1()
        graph = _make_graph([])
        emb = np.full(384, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            t1.resonate(emb, graph, [0])

    def test_empty_seeds_raises(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([])
        with pytest.raises(ValueError, match="non-empty"):
            t1.resonate(query_embedding, graph, [])


class TestTier1Propagation:
    def test_wilson_cowan_dynamics(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10)])
        result, _ = t1.resonate(query_embedding, graph, [0], max_iterations=1)
        assert all(0.01 <= v <= 1.0 for v in result.node_activations.values())

    def test_simple_diffusion_dynamics(self):
        alg = AlgorithmConfig(propagation_type="simple_diffusion", normalization="none",
                               gate_type="none", temporal_factor_enabled=False)
        t1 = make_tier1(algorithm=alg)
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10)])
        emb = np.ones(384, dtype=np.float32)
        result, _ = t1.resonate(emb, graph, [0], max_iterations=1)
        assert all(0.01 <= v <= 1.0 for v in result.node_activations.values())

    def test_threshold_dynamics(self):
        alg = AlgorithmConfig(propagation_type="threshold", normalization="none",
                               gate_type="none", temporal_factor_enabled=False)
        t1 = make_tier1(algorithm=alg)
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10)])
        emb = np.ones(384, dtype=np.float32)
        result, _ = t1.resonate(emb, graph, [0], max_iterations=1)
        assert all(0.01 <= v <= 1.0 for v in result.node_activations.values())

    def test_top_k_gating(self):
        alg = AlgorithmConfig(propagation_type="simple_diffusion", normalization="none",
                               gate_type="top_k", temporal_factor_enabled=False)
        tier_cfg = build_minimal_resonance_config().tier1
        import dataclasses
        tier_cfg = dataclasses.replace(tier_cfg, top_k=2)
        t1 = make_tier1(algorithm=alg, tier=tier_cfg)
        graph = _make_graph([
            (0, 1, "is_a", 0.9, 0.95, 10),
            (1, 2, "causes", 0.8, 0.9, 5),
            (2, 3, "follows", 0.7, 0.8, 3),
        ])
        emb = np.ones(384, dtype=np.float32)
        result, _ = t1.resonate(emb, graph, [0], max_iterations=1)
        assert len(result.nodes) <= 2

    def test_threshold_gating(self):
        alg = AlgorithmConfig(propagation_type="simple_diffusion", normalization="none",
                               gate_type="threshold", temporal_factor_enabled=False)
        t1 = make_tier1(algorithm=alg)
        graph = _make_graph([(0, 1, "is_a", 0.01, 0.01, 1)])
        emb = np.ones(384, dtype=np.float32)
        result, _ = t1.resonate(emb, graph, [0], max_iterations=1)
        assert isinstance(result, Subgraph)


class TestTier1Normalization:
    def test_l1_norm(self, query_embedding):
        alg = AlgorithmConfig(propagation_type="simple_diffusion", normalization="l1_norm",
                               gate_type="none", temporal_factor_enabled=False)
        t1 = make_tier1(algorithm=alg)
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10)])
        result, _ = t1.resonate(query_embedding, graph, [0], max_iterations=1)
        total = sum(result.node_activations.values())
        assert abs(total - 1.0) < 0.01 or total <= 1.0

    def test_no_normalization(self, query_embedding):
        alg = AlgorithmConfig(propagation_type="simple_diffusion", normalization="none",
                               gate_type="none", temporal_factor_enabled=False)
        t1 = make_tier1(algorithm=alg)
        graph = _make_graph([(0, 1, "is_a", 0.9, 0.95, 10)])
        result, _ = t1.resonate(query_embedding, graph, [0], max_iterations=1)
        assert len(result.node_activations) > 0


class TestTier1EdgeCases:
    def test_self_loop_does_not_crash(self, query_embedding, self_loop_graph):
        t1 = make_tier1()
        subgraph, _ = t1.resonate(query_embedding, self_loop_graph, [1], max_iterations=2)
        assert isinstance(subgraph, Subgraph)

    def test_multi_edges_between_same_nodes(self, query_embedding, multi_edge_graph):
        t1 = make_tier1()
        subgraph, _ = t1.resonate(query_embedding, multi_edge_graph, [1], max_iterations=2)
        assert isinstance(subgraph, Subgraph)

    def test_disconnected_graph(self, query_embedding, disconnected_graph):
        t1 = make_tier1()
        for seed in [0, 1, 2]:
            subgraph, _ = t1.resonate(query_embedding, disconnected_graph, [seed], max_iterations=2)
            assert seed in subgraph.node_activations

    def test_edge_weight_below_threshold_filtered(self, query_embedding):
        t1 = make_tier1()
        graph = _make_graph([(0, 1, "is_a", 0.001, 0.001, 1)])
        subgraph, _ = t1.resonate(query_embedding, graph, [0], max_iterations=1)
        assert 1 not in subgraph.node_activations or subgraph.node_activations.get(1, 0) < 0.02


class TestTier1CheckFinite:
    def test_finite_values_pass(self):
        assert Tier1Resonance._check_finite(0.5, "test")

    def test_nan_fails(self):
        assert not Tier1Resonance._check_finite(float("nan"), "test")

    def test_inf_fails(self):
        assert not Tier1Resonance._check_finite(float("inf"), "test")


class TestTier1CheckConvergence:
    def test_single_entry_not_converged(self):
        h = [np.array([0.5, 0.3], dtype=np.float32)]
        assert not Tier1Resonance._check_convergence(h, 0.001)

    def test_identical_entries_converged(self):
        h = [np.array([0.5, 0.3], dtype=np.float32),
             np.array([0.5, 0.3], dtype=np.float32)]
        assert Tier1Resonance._check_convergence(h, 0.001)

    def test_different_entries_not_converged(self):
        h = [np.array([0.5, 0.3], dtype=np.float32),
             np.array([0.6, 0.4], dtype=np.float32)]
        assert not Tier1Resonance._check_convergence(h, 0.001)
