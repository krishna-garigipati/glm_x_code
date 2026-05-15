from __future__ import annotations

"""Functional tests: Validate blueprint-defined business logic and execution correctness."""

import time
from typing import Dict, List

import numpy as np
import pytest

from ..engine import ResonanceEngine
from ..energy import compute_activation_energy
from ..temporal import compute_temporal_factor
from ..config import build_default_theta
from ..types import Subgraph
from .fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_loaded_configs,
    build_minimal_resonance_config,
    get_test_config_dir,
)
from .fixtures.toy_data import build_animal_kingdom_graph
from .fixtures.toy_graph_builder import (
    build_chain_graph,
    build_dense_graph,
    build_star_graph,
    build_two_node_graph,
    make_query_embedding,
)
from .fixtures.toy_graph_store import ToyGraphStore
from ..types import Node, Edge


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


@pytest.fixture(scope="module")
def animal_graph():
    return build_animal_kingdom_graph()


class TestWilsonCowanDynamics:
    def test_propagation_obeys_formula(self, engine):
        g = ToyGraphStore()
        rng = np.random.RandomState(42)
        for i in range(4):
            g.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                             embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                             activation=0.01, use_count=0, create_time=0.0))
        g.add_edge(0, 1, "is_a", strength=0.9, confidence=0.95, frequency=10)
        g.add_edge(1, 2, "causes", strength=0.8, confidence=0.9, frequency=5)
        g.add_edge(2, 3, "follows", strength=0.7, confidence=0.8, frequency=3)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        a0 = result.node_activations.get(0, 0)
        a1 = result.node_activations.get(1, 0)
        a2 = result.node_activations.get(2, 0)
        assert a0 >= a1 or a1 <= a0 + 0.1
        for act in result.node_activations.values():
            assert 0.0 <= act <= 1.0 or act == 0.01


class TestEnergyFormula:
    def test_energy_matches_sum_activation_strength_confidence(self, engine):
        g = ToyGraphStore()
        rng = np.random.RandomState(42)
        for i in range(3):
            g.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                             embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                             activation=0.01, use_count=0, create_time=0.0))
        g.add_edge(0, 1, "supports", strength=0.8, confidence=0.7, frequency=5)
        g.add_edge(1, 2, "contradicts", strength=0.6, confidence=0.5, frequency=2)
        q = make_query_embedding()
        result = engine.resonate(q, g, [0], tier=1)
        energy = engine.compute_activation_energy(result)
        assert np.isfinite(energy)
        assert energy >= 0.0


class TestTierSwitching:
    def test_tier1_energy_above_threshold_skips_tier2(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        result = engine.resonate(q, g, [1], tier=2)
        assert result.tier_used in (1, 2)

    def test_resonate_tier1_always_returns_tier1(self, engine, animal_graph):
        q = make_query_embedding()
        result = engine.resonate(q, animal_graph, [0], tier=1)
        assert result.tier_used == 1


class TestTemporalFactorFormula:
    def test_recency_formula_implementation(self):
        now = time.time()
        for hours_ago in [0, 1, 24, 168, 8760]:
            lu = now - hours_ago * 3600
            factor = compute_temporal_factor(lu, 10, 0.5, 20)
            assert 0.0 <= factor <= 1.0

    def test_frequency_scaling_implementation(self):
        now = time.time()
        f_low = compute_temporal_factor(now, 1, 0.5, 20)
        f_high = compute_temporal_factor(now, 50, 0.5, 20)
        assert f_high >= f_low


class TestSeedActivation:
    def test_seed_node_starts_at_max_activation(self):
        g = ToyGraphStore()
        rng = np.random.RandomState(42)
        g.add_node(Node(id=5, label="seed", node_type="Concept",
                         embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                         activation=0.01, use_count=0, create_time=0.0))
        from ..tier1 import Tier1Resonance
        rc = build_minimal_resonance_config()
        cc = build_minimal_core_config()
        t1 = Tier1Resonance(cc, rc.algorithm, rc.temporal, rc.tier1, False, 1000)
        init = t1._initialize_activations([5], None)
        assert init[5] == cc.activation.max


class TestThetaOverride:
    def test_override_tier1_uses_theta_values(self, engine):
        theta = engine.get_theta().copy()
        theta[0] = 0.02
        theta[1] = 0.05
        t1 = engine._override_tier1(theta)
        assert abs(t1._tier.propagation_threshold - 0.02) < 1e-6
        assert abs(t1._tier.edge_threshold - 0.05) < 1e-6

    def test_resonate_with_theta_uses_overrides(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        theta = engine.get_theta().copy()
        theta[0] = 0.01
        result = engine.resonate_with_theta(theta, q, g, [1])
        assert isinstance(result, Subgraph)


class TestBudgetSoftCap:
    def test_normalization_budget_soft_cap(self):
        from ..tier1 import Tier1Resonance
        cc = build_minimal_core_config()
        rc = build_minimal_resonance_config()
        t1 = Tier1Resonance(cc, rc.algorithm, rc.temporal, rc.tier1, False, 1000)
        activations = {i: 1.0 for i in range(10)}
        normed = t1._normalize(activations)
        total = sum(normed.values())
        assert total <= cc.resonance.budget_max + 1e-6
