from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pytest

from ...config import AlgorithmConfig, CoreConfig, TemporalConfig, Tier2Config
from ...tier2 import Tier2Resonance
from ...types import Subgraph
from ..fixtures.config_provider import build_minimal_core_config, build_minimal_resonance_config


def make_tier2(
    core: Optional[CoreConfig] = None,
    tier2_cfg: Optional[Tier2Config] = None,
) -> Tier2Resonance:
    rc = build_minimal_resonance_config()
    return Tier2Resonance(
        core_config=core or build_minimal_core_config(),
        algorithm=rc.algorithm,
        temporal=rc.temporal,
        tier_config=tier2_cfg or rc.tier2,
        relation_bias=rc.tier1.relation_bias,
        log_activation_history=False,
        history_buffer_size=1000,
    )


def _make_graph():
    from ..fixtures.toy_graph_store import ToyGraphStore
    from ...types import Node
    g = ToyGraphStore()
    rng = np.random.RandomState(42)
    for i in range(10):
        g.add_node(Node(id=i, label=f"n{i}", node_type="Concept",
                         embedding=rng.randint(-128, 128, size=(32,)).astype(np.int8),
                         activation=0.01, use_count=0, create_time=0.0))
    g.add_edge(0, 1, "is_a", strength=0.9, confidence=0.95, frequency=10)
    g.add_edge(1, 2, "causes", strength=0.8, confidence=0.9, frequency=5)
    g.add_edge(2, 3, "follows", strength=0.7, confidence=0.8, frequency=3)
    g.add_edge(0, 4, "has_property", strength=0.6, confidence=0.7, frequency=2)
    g.add_edge(4, 5, "part_of", strength=0.5, confidence=0.6, frequency=1)
    g.add_edge(0, 6, "associated_with", strength=0.4, confidence=0.5, frequency=1)
    g.add_edge(6, 7, "synonym", strength=0.9, confidence=0.9, frequency=8)
    g.add_edge(7, 8, "antonym", strength=0.7, confidence=0.8, frequency=4)
    g.add_edge(8, 9, "caused_by", strength=0.6, confidence=0.7, frequency=2)
    return g


class TestTier2ResonanceInit:
    def test_initializes(self):
        t2 = make_tier2()
        assert t2 is not None
        assert t2._tier.enabled is True
        assert t2._tier.top_k == 1024


class TestTier2Resonate:
    def test_resonate_returns_subgraph(self, query_embedding):
        t2 = make_tier2()
        graph = _make_graph()
        subgraph, history = t2.resonate(query_embedding, graph, [0])
        assert isinstance(subgraph, Subgraph)
        assert subgraph.tier_used == 1

    def test_resonate_with_analogies(self, query_embedding):
        t2 = make_tier2()
        graph = _make_graph()
        subgraph, _ = t2.resonate(query_embedding, graph, [0])
        assert isinstance(subgraph, Subgraph)
        assert len(subgraph.nodes) > 0

    def test_disabled_tier2_raises(self, query_embedding):
        import dataclasses
        rc = build_minimal_resonance_config()
        t2c = dataclasses.replace(rc.tier2, enabled=False)
        t2 = make_tier2(tier2_cfg=t2c)
        graph = _make_graph()
        with pytest.raises(RuntimeError, match="disabled"):
            t2.resonate(query_embedding, graph, [0])

    def test_larger_graph_propagation(self, query_embedding):
        t2 = make_tier2()
        graph = _make_graph()
        subgraph, _ = t2.resonate(query_embedding, graph, [0])
        assert 0 in subgraph.node_activations

    def test_empty_seeds_raises(self, query_embedding):
        t2 = make_tier2()
        graph = _make_graph()
        with pytest.raises(ValueError, match="non-empty"):
            t2.resonate(query_embedding, graph, [])

    def test_all_activations_in_range(self, query_embedding):
        t2 = make_tier2()
        graph = _make_graph()
        subgraph, _ = t2.resonate(query_embedding, graph, [0])
        for v in subgraph.node_activations.values():
            assert 0.01 <= v <= 1.0, f"Activation {v} out of range"

    def test_seed_node_in_activations(self, query_embedding):
        t2 = make_tier2()
        graph = _make_graph()
        subgraph, _ = t2.resonate(query_embedding, graph, [5])
        assert 5 in subgraph.node_activations


class TestTier2Analogy:
    def test_apply_analogies_returns_dict(self):
        t2 = make_tier2()
        graph = _make_graph()
        result = t2._apply_analogies(graph, [0])
        assert isinstance(result, dict)

    def test_apply_analogies_empty_seeds(self):
        t2 = make_tier2()
        graph = _make_graph()
        result = t2._apply_analogies(graph, [])
        assert result == {}

    def test_analogies_disabled(self, query_embedding):
        import dataclasses
        rc = build_minimal_resonance_config()
        t2c = dataclasses.replace(rc.tier2, analogies_enabled=False)
        t2 = make_tier2(tier2_cfg=t2c)
        graph = _make_graph()
        subgraph, _ = t2.resonate(query_embedding, graph, [0])
        assert isinstance(subgraph, Subgraph)


class TestTier1View:
    def test_creates_valid_tier_config(self):
        from ...tier2 import _tier1_view
        rc = build_minimal_resonance_config()
        result = _tier1_view(rc.tier2, rc.tier1.relation_bias)
        assert result.propagation_threshold == rc.tier2.propagation_threshold
        assert result.edge_threshold == rc.tier2.edge_threshold
        assert result.decay_lambda == rc.tier2.decay_lambda
        assert result.top_k == rc.tier2.top_k
        assert result.max_iterations == rc.tier2.max_iterations
        assert result.relation_bias == rc.tier1.relation_bias
        assert result.energy_threshold_formula is None
        assert result.t_conf_coefficient is None
        assert result.multiplied_at_runtime is None
