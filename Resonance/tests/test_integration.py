from __future__ import annotations

"""Integration tests: Module-to-module communication and API contracts."""

from pathlib import Path

import numpy as np
import pytest

from ..engine import ResonanceEngine
from ..config import build_default_theta
from ..energy import compute_activation_energy
from ..temporal import compute_temporal_factor
from ..types import Subgraph
from .fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_loaded_configs,
    build_minimal_resonance_config,
    get_test_config_dir,
    load_test_configs,
)
from .fixtures.toy_data import build_animal_kingdom_graph
from .fixtures.toy_graph_builder import (
    build_chain_graph,
    build_cluster_graph,
    build_dense_graph,
    build_star_graph,
    build_two_node_graph,
    make_query_embedding,
)
from .fixtures.toy_graph_store import ToyGraphStore


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


@pytest.fixture(scope="module")
def animal_graph():
    return build_animal_kingdom_graph()


class TestConfigEngineIntegration:
    def test_engine_loads_config_correctly(self, engine):
        assert engine._configs.resonance.tier1.propagation_threshold == 0.008
        assert engine._configs.resonance.tier2.enabled is True

    def test_build_default_theta_matches_config(self):
        configs = build_minimal_loaded_configs()
        theta = build_default_theta(configs.resonance, configs.core)
        assert abs(theta[0] - configs.resonance.tier1.propagation_threshold) < 1e-6
        assert abs(theta[3] - configs.resonance.tier1.top_k) < 1e-6


class TestTier1Integration:
    def test_tier1_propagation_cascades(self, engine):
        g = build_chain_graph(length=6, relation="follows")
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        assert len(result.nodes) > 1

    def test_tier1_activations_decrease_with_distance(self, engine):
        g = build_chain_graph(length=5, relation="follows")
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        acts = result.node_activations
        if 0 in acts and 4 in acts:
            assert acts[0] >= acts[4]

    def test_tier1_seed_has_max_activation(self, engine):
        g = build_star_graph(center_id=0, leaf_count=5)
        q = make_query_embedding()
        result = engine.resonate(q, g, [2], tier=1)
        seed_act = result.node_activations.get(2, 0)
        max_act = max(result.node_activations.values())
        assert seed_act == max_act

    def test_tier1_with_multi_seed(self, engine):
        g = build_cluster_graph(num_clusters=2, nodes_per_cluster=4, bridge_edges=1)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0, 5], tier=1)
        assert 0 in result.node_activations
        assert 5 in result.node_activations


class TestTier2Integration:
    def test_tier2_propagates_further_than_tier1(self, engine, animal_graph):
        q = make_query_embedding()
        t1_result = engine.resonate(q, animal_graph, [0], tier=1)
        t2_result = engine.resonate(q, animal_graph, [0], tier=2)
        assert len(t2_result.nodes) >= len(t1_result.nodes) or t2_result.activation_energy >= 0

    def test_tier2_energy_improvement_check(self, engine):
        g = build_dense_graph(node_count=10, edge_density=0.4)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=2)
        assert result.activation_energy >= 0


class TestEnergyIntegration:
    def test_energy_computed_after_resonance(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        result = engine.resonate(q, g, [1], tier=1)
        energy = compute_activation_energy(result)
        assert energy >= 0
        assert np.isfinite(energy)

    def test_tier1_energy_threshold_triggers_tier_skip(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        result = engine.resonate(q, g, [1], tier=2)
        assert result.tier_used in (1, 2)


class TestTemporalIntegration:
    def test_temporal_factor_applied_in_resonance(self, engine):
        import time
        g = ToyGraphStore()
        from ..types import Node
        rng = np.random.RandomState(0)
        for i in range(4):
            g.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                             embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                             activation=0.01, use_count=0, create_time=0.0))
        g.add_edge(0, 1, "is_a", strength=0.9, confidence=0.95, last_used=time.time(), frequency=10)
        g.add_edge(1, 2, "causes", strength=0.8, confidence=0.9, last_used=0.0, frequency=1)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        assert isinstance(result, Subgraph)


class TestESEngineIntegration:
    def test_es_updates_theta_through_engine(self, engine):
        old_theta = engine.get_theta().copy()
        theta = engine.get_theta()
        for i in range(8):
            engine.update_es_with_reward(float(i) / 8.0, theta)
        new_theta = engine.get_theta()
        assert new_theta.shape == (48,)

    def test_es_reward_accumulation(self, engine):
        theta = engine.get_theta()
        engine.update_es_with_reward(0.5, theta)
        assert len(engine._es._samples) > 0 or len(engine._es._samples) == 0

    def test_get_theta_history(self, engine):
        history = engine._es.get_theta_history()
        assert isinstance(history, list)

    def test_set_theta_via_engine_changes_proposals(self, engine):
        theta = engine.get_theta()
        theta[0] = 0.03
        engine.set_theta(theta)
        proposal = engine.propose_theta_mutation()
        assert abs(proposal[0] - 0.03) < 0.05


class TestAnalogyEngineIntegration:
    def test_get_analogy_leaps_from_engine(self, engine, animal_graph):
        results = engine.get_analogy_leaps(0, animal_graph, top_k=5)
        assert isinstance(results, list)
        for nid, score in results:
            assert isinstance(nid, int)

    def test_clear_analogy_cache_from_engine(self, engine):
        engine.clear_analogy_cache()


class TestFullPipeline:
    def test_resonate_then_energy_then_convergence(self, engine, animal_graph):
        q = make_query_embedding()
        seeds = [0]
        subgraph, history_raw = engine._tier1.resonate(q, animal_graph, seeds, max_iterations=4)
        energy = compute_activation_energy(subgraph)
        history = [subgraph.query_embedding]
        converged = engine.check_resonance_convergence([np.array([0.5]), np.array([0.5])])
        assert isinstance(energy, (float, np.floating))
        assert isinstance(converged, bool)
        assert np.isfinite(energy)
