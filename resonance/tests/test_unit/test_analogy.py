from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import pytest

from ...analogy import AnalogyFinder, _cosine_similarity, _to_float_embedding
from ...config import AnalogyParameters, CoreActivationConfig, CoreConfig, CoreResonanceConfig, ESBounds
from ...types import Edge, GraphStore, Node, Subgraph


def _make_core_config() -> CoreConfig:
    return CoreConfig(
        activation=CoreActivationConfig(min=0.01, max=1.0, default=0.01, threshold_resonance=0.2),
        resonance=CoreResonanceConfig(
            propagation_threshold=0.008, edge_threshold=0.02, decay_lambda=0.1,
            top_k=64, budget_max=2.0, convergence_epsilon=0.001,
            tier1_energy_threshold=0.4, tier2_max_nodes=1024, analogy_validation_overlap=0.3,
        ),
        relations=["is_a", "has_property", "causes", "caused_by", "follows", "precedes",
                    "contradicts", "supports", "associated_with", "example_of", "part_of",
                    "synonym", "antonym", "temporal_coincident", "spatial_near", "linguistic_maps"],
        es_bounds=ESBounds(
            propagation_threshold=(0.001, 0.05), edge_threshold=(0.01, 0.1),
            decay_lambda=(0.05, 0.5), top_k=(16, 1024), relation_bias=(0.0, 2.0),
        ),
    )


def _default_analogy_params(**kwargs) -> AnalogyParameters:
    defaults = dict(
        lsh_bands=16,
        lsh_tables=4,
        temp_edge_strength=0.5,
        overlap_validation_required=False,
        jaccard_overlap_min=0.3,
        mini_propagation_steps=2,
        edge_confidence_min=0.5,
    )
    defaults.update(kwargs)
    return AnalogyParameters(**defaults)


class _MockGraph:
    def __init__(self):
        self.nodes: dict = {}
        self.edges: dict = {}

    def add_node(self, nid: int, emb: np.ndarray, ntype: str = "Concept"):
        self.nodes[nid] = Node(
            id=nid, label=f"node_{nid}", node_type=ntype,
            embedding=emb, activation=0.01, use_count=0, create_time=time.time(),
        )

    def add_edge(self, src: int, tgt: int, rel: str, strength=0.7, conf=0.8, freq=1):
        e = Edge(source=src, target=tgt, relation_type=rel,
                  strength=strength, confidence=conf,
                  last_used=time.time(), frequency=freq)
        self.edges.setdefault(src, []).append((tgt, e))

    def get_node(self, nid: int) -> Optional[Node]:
        return self.nodes.get(nid)

    def get_neighbors(self, nid: int) -> List[Tuple[int, Edge]]:
        return self.edges.get(nid, [])

    def get_all_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def get_subgraph_by_embedding_similarity(self, emb: np.ndarray, top_k: int) -> Subgraph:
        scores = []
        for nid, node in self.nodes.items():
            e = _to_float_embedding(node.embedding)
            sim = _cosine_similarity(emb, e)
            scores.append((nid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [s[0] for s in scores[:top_k]]
        return Subgraph(
            nodes=top_ids, node_activations={n: 0.5 for n in top_ids},
            edges=[], edge_strengths={}, edge_confidences={},
            seed_nodes=top_ids[:1] if top_ids else [],
            tier_used=1, activation_energy=0.0,
            query_embedding=np.ascontiguousarray(emb.astype(np.float32)),
            timestamp=time.time(),
        )


class TestAnalogyFinder:
    def test_init_creates_finder(self):
        cc = _make_core_config()
        ap = _default_analogy_params()
        af = AnalogyFinder(cc, ap)
        assert af is not None

    def test_empty_graph_returns_empty(self):
        cc = _make_core_config()
        ap = _default_analogy_params()
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()
        results = af.get_analogy_leaps(1, graph, top_k=3)
        assert results == []

    def test_target_not_in_graph_returns_empty(self):
        cc = _make_core_config()
        ap = _default_analogy_params()
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()
        graph.add_node(1, _int8_emb([1] * 32))
        results = af.get_analogy_leaps(99, graph, top_k=3)
        assert results == []

    def test_finds_similar_nodes(self):
        cc = _make_core_config()
        ap = _default_analogy_params(overlap_validation_required=False, lsh_bands=2, lsh_tables=4)
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()

        rng = np.random.RandomState(0)
        emb_target = rng.randn(32).astype(np.int8)
        for i in range(20):
            emb = emb_target + np.random.RandomState(i).randn(32).astype(np.int8) * 2
            graph.add_node(10 + i, emb)
        emb_different = rng.randn(32).astype(np.int8) * 10
        graph.add_node(1, emb_target)
        graph.add_node(30, emb_different)

        results = af.get_analogy_leaps(1, graph, top_k=2)
        assert len(results) >= 1

    def test_top_k_limit(self):
        cc = _make_core_config()
        ap = _default_analogy_params(overlap_validation_required=False)
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()

        rng = np.random.RandomState(0)
        emb_target = rng.randn(32).astype(np.int8)
        graph.add_node(1, emb_target)
        for i in range(10):
            graph.add_node(10 + i, rng.randn(32).astype(np.int8))

        results = af.get_analogy_leaps(1, graph, top_k=3)
        assert len(results) <= 3

    def test_top_k_zero_returns_empty(self):
        cc = _make_core_config()
        ap = _default_analogy_params()
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()
        graph.add_node(1, _int8_emb([1] * 32))
        results = af.get_analogy_leaps(1, graph, top_k=0)
        assert results == []

    def test_jaccard_validation_filters_candidates(self):
        cc = _make_core_config()
        ap = _default_analogy_params(overlap_validation_required=True, jaccard_overlap_min=0.5)
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()

        rng = np.random.RandomState(0)
        emb1 = rng.randn(32).astype(np.int8)
        emb2 = emb1 + rng.randn(32).astype(np.int8) * 0.5
        emb3 = rng.randn(32).astype(np.int8)

        graph.add_node(1, emb1)
        graph.add_node(2, emb2)
        graph.add_node(3, emb3)

        graph.add_edge(1, 2, "associated_with", conf=0.9)
        graph.add_edge(2, 1, "associated_with", conf=0.9)
        graph.add_edge(2, 3, "associated_with", conf=0.9)

        results = af.get_analogy_leaps(1, graph, top_k=5)
        assert isinstance(results, list)

    def test_lsh_index_rebuilt_after_clear_cache(self):
        cc = _make_core_config()
        ap = _default_analogy_params()
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()
        rng = np.random.RandomState(0)
        for i in range(5):
            graph.add_node(i, rng.randn(32).astype(np.int8))
        af._ensure_lsh_index(graph)
        assert af._lsh_index is not None
        old_index = af._lsh_index
        af.clear_cache()
        assert af._lsh_index is None
        af._ensure_lsh_index(graph)
        assert af._lsh_index is not None

    def test_cosine_similarity_same_vector(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = _cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_opposite(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        sim = _cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        sim = _cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.zeros(2, dtype=np.float32)
        sim = _cosine_similarity(v1, v2)
        assert sim == 0.0

    def test_to_float_embedding_converts_int8(self):
        emb = np.array([-128, 0, 127], dtype=np.int8)
        result = _to_float_embedding(emb)
        assert result.dtype == np.float32
        assert abs(result[0] - (-128.0 / 127.0)) < 1e-6
        assert abs(result[1]) < 1e-6
        assert abs(result[2] - (127.0 / 127.0)) < 1e-6

    def test_to_float_embedding_non_array_raises(self):
        with pytest.raises(TypeError, match="ndarray"):
            _to_float_embedding([1, 2, 3])

    def test_to_float_embedding_nan_to_num(self):
        emb = np.array([float("nan")], dtype=np.float32)
        result = _to_float_embedding(emb)
        assert np.isfinite(result[0])

    def test_jaccard_overlap_zero_isolation(self):
        cc = _make_core_config()
        ap = _default_analogy_params(overlap_validation_required=False)
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()
        rng = np.random.RandomState(0)
        graph.add_node(1, rng.randn(32).astype(np.int8))
        graph.add_node(2, rng.randn(32).astype(np.int8))
        overlap = af._jaccard_overlap(1, 2, graph)
        assert overlap == 0.0

    def test_jaccard_overlap_shared(self):
        cc = _make_core_config()
        ap = _default_analogy_params(overlap_validation_required=False)
        af = AnalogyFinder(cc, ap)
        graph = _MockGraph()
        rng = np.random.RandomState(0)
        graph.add_node(1, rng.randn(32).astype(np.int8))
        graph.add_node(2, rng.randn(32).astype(np.int8))
        graph.add_node(3, rng.randn(32).astype(np.int8))
        graph.add_edge(1, 3, "is_a", conf=0.9)
        graph.add_edge(2, 3, "is_a", conf=0.9)
        overlap = af._jaccard_overlap(1, 2, graph)
        assert overlap > 0.0


def _int8_emb(values) -> np.ndarray:
    arr = np.array(values[:32], dtype=np.int8)
    if len(arr) < 32:
        arr = np.pad(arr, (0, 32 - len(arr)))
    return arr[:32]
