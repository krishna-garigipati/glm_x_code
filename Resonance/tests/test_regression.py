from __future__ import annotations

"""Regression tests: Ensure fixes and changes don't break existing functionality."""

import copy
import time
from typing import Dict, List

import numpy as np
import pytest

from ..engine import ResonanceEngine
from ..config import build_default_theta, load_configs
from ..types import Subgraph
from .fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_loaded_configs,
    build_minimal_resonance_config,
    get_test_config_dir,
)
from .fixtures.toy_graph_builder import (
    build_chain_graph,
    build_dense_graph,
    build_disconnected_graph,
    build_empty_graph,
    build_multi_edge_graph,
    build_self_loop_graph,
    build_single_node_graph,
    build_star_graph,
    build_two_node_graph,
    make_query_embedding,
)
from .fixtures.toy_data import build_animal_kingdom_graph
from .fixtures.toy_graph_store import ToyGraphStore


REGRESSION_SEED = 42
RNG = np.random.RandomState(REGRESSION_SEED)


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


class TestRegressionConfig:
    def test_config_loading_stable(self):
        c1 = load_configs(get_test_config_dir())
        c2 = load_configs(get_test_config_dir())
        assert c1.core.activation.min == c2.core.activation.min
        assert c1.resonance.tier1.top_k == c2.resonance.tier1.top_k

    def test_build_theta_deterministic(self):
        configs = build_minimal_loaded_configs()
        t1 = build_default_theta(configs.resonance, configs.core)
        t2 = build_default_theta(configs.resonance, configs.core)
        assert np.allclose(t1, t2)


class TestRegressionTier1:
    def test_tier1_resonate_deterministic(self, engine):
        g = build_chain_graph(length=5, relation="follows")
        q = make_query_embedding()
        r1 = engine.resonate(q, g, [0], tier=1)
        r2 = engine.resonate(q, g, [0], tier=1)
        assert r1.nodes == r2.nodes
        for k in r1.node_activations:
            assert abs(r1.node_activations[k] - r2.node_activations[k]) < 1e-4

    def test_empty_graph_handled_gracefully(self, engine, empty_graph):
        q = make_query_embedding()
        result = engine.resonate(q, empty_graph, [0], tier=1)
        assert isinstance(result, Subgraph)

    def test_single_node_handled(self, engine, single_node_graph):
        q = make_query_embedding()
        result = engine.resonate(q, single_node_graph, [1], tier=1)
        assert 1 in result.node_activations

    def test_self_loop_stable(self, engine, self_loop_graph):
        q = make_query_embedding()
        result = engine.resonate(q, self_loop_graph, [1], tier=1)
        assert isinstance(result, Subgraph)

    def test_multi_edge_stable(self, engine, multi_edge_graph):
        q = make_query_embedding()
        result = engine.resonate(q, multi_edge_graph, [1], tier=1)
        assert isinstance(result, Subgraph)


class TestRegressionTier2:
    def test_tier2_resonate_stable(self, engine):
        g = build_dense_graph(node_count=8, edge_density=0.3)
        q = make_query_embedding()
        r1 = engine.resonate(q, g, [0], tier=2)
        r2 = engine.resonate(q, g, [0], tier=2)
        assert isinstance(r1, Subgraph)
        assert isinstance(r2, Subgraph)

    def test_tier2_disabled_path(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        result = engine.resonate(q, g, [1], tier=1)
        assert result.tier_used == 1


class TestRegressionES:
    def test_es_deterministic_proposal_after_reinit(self):
        from ..es_controller import EvolutionaryController
        cc = build_minimal_core_config()
        rc = build_minimal_resonance_config()
        theta = np.zeros(48, dtype=np.float32)
        es1 = EvolutionaryController(cc, rc.es_controller, theta, rc.theta_indices, _seed=42)
        es2 = EvolutionaryController(cc, rc.es_controller, theta, rc.theta_indices, _seed=42)
        p1 = es1.propose_theta_mutation()
        p2 = es2.propose_theta_mutation()
        assert p1.shape == p2.shape
        assert np.allclose(p1, p2)

    def test_get_theta_always_returns_copy(self, engine):
        t1 = engine.get_theta()
        t2 = engine.get_theta()
        t1[0] = 999.0
        assert t2[0] != 999.0


class TestRegressionEnergy:
    def test_energy_computation_idempotent(self, engine):
        g = build_star_graph(center_id=0, leaf_count=4)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        e1 = engine.compute_activation_energy(result)
        e2 = engine.compute_activation_energy(result)
        assert abs(e1 - e2) < 1e-6

    def test_energy_with_no_edges(self):
        from ..energy import compute_activation_energy
        sg = Subgraph(
            nodes=[1, 2], node_activations={1: 0.3, 2: 0.7},
            edges=[], edge_strengths={}, edge_confidences={},
            seed_nodes=[1], tier_used=1, activation_energy=0.0,
            query_embedding=np.zeros(384, dtype=np.float32), timestamp=time.time(),
        )
        e = compute_activation_energy(sg)
        assert abs(e - 1.0) < 1e-6


class TestRegressionValidation:
    def test_validation_rejects_nan_consistently(self):
        from ..validation import validate_embedding
        emb = np.full(384, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            validate_embedding(emb)

    def test_validation_rejects_wrong_shape_consistently(self):
        from ..validation import validate_embedding
        emb = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="dim 384"):
            validate_embedding(emb)

    def test_validation_accepts_valid_embedding(self):
        from ..validation import validate_embedding
        emb = np.zeros(384, dtype=np.float32)
        validate_embedding(emb)


class TestRegressionAnalogy:
    def test_analogy_identity_not_in_results(self):
        from ..analogy import AnalogyFinder
        from ..config import AnalogyParameters
        cc = build_minimal_core_config()
        ap = AnalogyParameters(lsh_bands=4, lsh_tables=2, temp_edge_strength=0.5,
                                overlap_validation_required=False, jaccard_overlap_min=0.3,
                                mini_propagation_steps=2, edge_confidence_min=0.5)
        af = AnalogyFinder(cc, ap)
        graph = ToyGraphStore()
        from ..types import Node
        rng = np.random.RandomState(0)
        emb = rng.randn(32).astype(np.int8)
        graph.add_node(Node(id=1, label="self", node_type="Concept", embedding=emb,
                             activation=0.01, use_count=0, create_time=0.0))
        results = af.get_analogy_leaps(1, graph, top_k=3)
        for nid, _ in results:
            assert nid != 1
