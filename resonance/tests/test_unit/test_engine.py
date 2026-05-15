from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from ...engine import ResonanceEngine
from ...types import Subgraph
from ..fixtures.config_provider import get_test_config_dir, load_test_configs


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


@pytest.fixture(scope="module")
def animal_graph():
    from ..fixtures.toy_data import build_animal_kingdom_graph
    return build_animal_kingdom_graph()


@pytest.fixture(scope="module")
def animal_seeds():
    return [0]


@pytest.fixture(scope="module")
def animal_query():
    rng = np.random.RandomState(42)
    emb = rng.randn(384).astype(np.float32)
    emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
    return np.ascontiguousarray(emb)


def make_query() -> np.ndarray:
    rng = np.random.RandomState(99)
    emb = rng.randn(384).astype(np.float32)
    emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
    return np.ascontiguousarray(emb)


class TestResonanceEngineInit:
    def test_initializes_with_config_dir(self):
        eng = ResonanceEngine(get_test_config_dir())
        assert eng is not None
        assert eng._theta.shape == (48,)
        assert eng._configs is not None

    def test_invalid_config_dir_raises(self):
        with pytest.raises((FileNotFoundError, NotADirectoryError)):
            ResonanceEngine(Path("/nonexistent/path"))

    def test_loads_real_configs(self, engine):
        assert engine._configs.core.activation.min == 0.01
        assert engine._configs.core.activation.max == 1.0
        assert len(engine._configs.core.relations) == 16

    def test_tier1_initialized(self, engine):
        assert engine._tier1 is not None
        assert engine._tier1._tier.top_k == 64

    def test_tier2_initialized(self, engine):
        assert engine._tier2 is not None
        assert engine._tier2._tier.enabled is True

    def test_es_initialized(self, engine):
        assert engine._es is not None
        assert engine._es._mu.shape == (48,)

    def test_context_manager(self):
        with ResonanceEngine(get_test_config_dir()) as eng:
            assert eng is not None


class TestResonate:
    def test_resonate_tier1_returns_subgraph(self, engine, animal_graph, animal_seeds, animal_query):
        result = engine.resonate(animal_query, animal_graph, animal_seeds, tier=1)
        assert isinstance(result, Subgraph)
        assert result.tier_used == 1
        assert len(result.nodes) > 0

    def test_resonate_tier2_returns_subgraph(self, engine, animal_graph, animal_seeds, animal_query):
        result = engine.resonate(animal_query, animal_graph, animal_seeds, tier=2)
        assert isinstance(result, Subgraph)
        assert result.tier_used in (1, 2)

    def test_resonate_with_valid_tier2_fallback(self, engine):
        graph = _simple_graph()
        q = make_query()
        result = engine.resonate(q, graph, [0], tier=2)
        assert isinstance(result, Subgraph)

    def test_resonate_invalid_tier_raises(self, engine, animal_graph, animal_seeds, animal_query):
        with pytest.raises(ValueError, match="tier must be 1 or 2"):
            engine.resonate(animal_query, animal_graph, animal_seeds, tier=3)

    def test_resonate_empty_seeds_raises(self, engine, animal_graph, animal_query):
        with pytest.raises(ValueError, match="non-empty"):
            engine.resonate(animal_query, animal_graph, [], tier=1)

    def test_resonate_nan_embedding_raises(self, engine, animal_graph, animal_seeds):
        emb = np.full(384, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            engine.resonate(emb, animal_graph, animal_seeds, tier=1)


class TestResonateWithTheta:
    def test_resonate_with_theta_returns_subgraph(self, engine, animal_graph, animal_seeds, animal_query):
        theta = engine.get_theta()
        result = engine.resonate_with_theta(theta, animal_query, animal_graph, animal_seeds)
        assert isinstance(result, Subgraph)

    def test_resonate_with_modified_theta(self, engine, animal_graph, animal_seeds, animal_query):
        theta = engine.get_theta().copy()
        theta[0] = 0.05
        result = engine.resonate_with_theta(theta, animal_query, animal_graph, animal_seeds)
        assert isinstance(result, Subgraph)

    def test_resonate_with_theta_wrong_shape_raises(self, engine, animal_graph, animal_seeds, animal_query):
        bad = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="dim mismatch"):
            engine.resonate_with_theta(bad, animal_query, animal_graph, animal_seeds)


class TestThetaManagement:
    def test_get_theta_shape(self, engine):
        theta = engine.get_theta()
        assert theta.shape == (48,)

    def test_get_theta_float32(self, engine):
        theta = engine.get_theta()
        assert theta.dtype == np.float32

    def test_set_theta_updates(self, engine):
        new = np.ones(48, dtype=np.float32) * 0.5
        engine.set_theta(new)
        current = engine.get_theta()
        assert np.allclose(current, new)

    def test_propose_theta_mutation(self, engine):
        proposal = engine.propose_theta_mutation()
        assert proposal.shape == (48,)
        assert np.isfinite(proposal).all()

    def test_update_es_with_reward(self, engine):
        theta = engine.get_theta()
        engine.update_es_with_reward(1.0, theta)


class TestEnergyAndConvergence:
    def test_compute_activation_energy_returns_float(self, engine, animal_graph, animal_seeds, animal_query):
        result = engine.resonate(animal_query, animal_graph, animal_seeds, tier=1)
        energy = engine.compute_activation_energy(result)
        assert isinstance(energy, (float, np.floating))
        assert np.isfinite(energy)

    def test_check_convergence_short_history(self, engine):
        h = [np.array([0.5], dtype=np.float32)]
        assert not engine.check_resonance_convergence(h)

    def test_check_convergence_converged(self, engine):
        h = [np.array([0.5, 0.3], dtype=np.float32),
             np.array([0.5, 0.3], dtype=np.float32)]
        assert engine.check_resonance_convergence(h)

    def test_check_convergence_not_converged(self, engine):
        h = [np.array([0.5, 0.3], dtype=np.float32),
             np.array([0.6, 0.4], dtype=np.float32)]
        assert not engine.check_resonance_convergence(h)

    def test_check_convergence_custom_epsilon(self, engine):
        h = [np.array([0.5, 0.3], dtype=np.float32),
             np.array([0.5005, 0.3005], dtype=np.float32)]
        assert engine.check_resonance_convergence(h, epsilon=0.001)

    def test_check_convergence_epsilon_reject(self, engine):
        h = [np.array([0.5, 0.3], dtype=np.float32),
             np.array([0.6, 0.4], dtype=np.float32)]
        assert not engine.check_resonance_convergence(h, epsilon=0.001)


class TestAnalogyLeaps:
    def test_get_analogy_leaps_returns_list(self, engine, animal_graph):
        result = engine.get_analogy_leaps(0, animal_graph, top_k=3)
        assert isinstance(result, list)
        if result:
            node_id, score = result[0]
            assert isinstance(node_id, int)
            assert isinstance(score, (float, np.floating))
            assert -1.0 <= score <= 1.0

    def test_clear_analogy_cache(self, engine):
        engine.clear_analogy_cache()


class TestConfigAccess:
    def test_get_configs_returns_loaded_configs(self, engine):
        configs = engine.get_configs()
        assert configs is not None
        assert configs.core.activation.min == 0.01


class TestEngineEdgeCases:
    def test_resonate_tier1_runtime_threshold(self, engine):
        graph = _simple_graph()
        q = make_query()
        result = engine.resonate(q, graph, [0], tier=1)
        assert isinstance(result, Subgraph)


def _simple_graph():
    from ..fixtures.toy_graph_store import ToyGraphStore
    from ...types import Node
    g = ToyGraphStore()
    rng = np.random.RandomState(0)
    for i in range(5):
        g.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                         embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                         activation=0.01, use_count=0, create_time=0.0))
    g.add_edge(0, 1, "is_a", strength=0.9, confidence=0.95, frequency=10)
    g.add_edge(1, 2, "causes", strength=0.8, confidence=0.9, frequency=5)
    g.add_edge(2, 3, "follows", strength=0.7, confidence=0.8, frequency=3)
    return g
