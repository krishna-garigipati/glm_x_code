from __future__ import annotations

import time

import numpy as np
import pytest

from ...energy import _is_bad, _safe_sum, compute_activation_energy
from ...types import Subgraph


def _make_subgraph(
    nodes=None, activations=None, edges=None,
    strengths=None, confidences=None, seed_nodes=None,
) -> Subgraph:
    if nodes is None:
        nodes = [1]
    if activations is None:
        activations = {1: 0.5}
    if edges is None:
        edges = []
    if strengths is None:
        strengths = {}
    if confidences is None:
        confidences = {}
    if seed_nodes is None:
        seed_nodes = [1]
    return Subgraph(
        nodes=nodes,
        node_activations=activations,
        edges=edges,
        edge_strengths=strengths,
        edge_confidences=confidences,
        seed_nodes=seed_nodes,
        tier_used=1,
        activation_energy=0.0,
        query_embedding=np.zeros(384, dtype=np.float32),
        timestamp=time.time(),
    )


class TestComputeActivationEnergy:
    def test_no_edges_returns_sum_of_activations(self):
        sg = _make_subgraph(nodes=[1, 2], activations={1: 0.3, 2: 0.7}, edges=[])
        result = compute_activation_energy(sg)
        assert abs(result - 1.0) < 1e-6

    def test_single_edge_computes_correctly(self):
        sg = _make_subgraph(
            nodes=[1, 2],
            activations={1: 0.5, 2: 0.3},
            edges=[(1, 2, "is_a")],
            strengths={(1, 2, "is_a"): 0.8},
            confidences={(1, 2, "is_a"): 0.9},
        )
        result = compute_activation_energy(sg)
        expected = 0.5 * (0.8 * 0.9)
        assert abs(result - expected) < 1e-6

    def test_multiple_edges(self):
        sg = _make_subgraph(
            nodes=[1, 2, 3],
            activations={1: 0.9, 2: 0.5, 3: 0.2},
            edges=[(1, 2, "causes"), (2, 3, "follows")],
            strengths={(1, 2, "causes"): 0.7, (2, 3, "follows"): 0.6},
            confidences={(1, 2, "causes"): 0.8, (2, 3, "follows"): 0.9},
        )
        result = compute_activation_energy(sg)
        e1 = 0.9 * (0.7 * 0.8)
        e2 = 0.5 * (0.6 * 0.9)
        assert abs(result - (e1 + e2)) < 1e-6

    def test_empty_subgraph_returns_zero(self):
        sg = _make_subgraph(nodes=[], activations={})
        result = compute_activation_energy(sg)
        assert result == 0.0

    def test_nan_activations_skipped(self):
        sg = _make_subgraph(
            nodes=[1, 2],
            activations={1: float("nan"), 2: 0.5},
            edges=[(1, 2, "is_a")],
            strengths={(1, 2, "is_a"): 0.8},
            confidences={(1, 2, "is_a"): 0.9},
        )
        result = compute_activation_energy(sg)
        assert np.isfinite(result)

    def test_inf_edge_strength_skipped(self):
        sg = _make_subgraph(
            nodes=[1, 2],
            activations={1: 0.5, 2: 0.3},
            edges=[(1, 2, "is_a")],
            strengths={(1, 2, "is_a"): float("inf")},
            confidences={(1, 2, "is_a"): 0.9},
        )
        result = compute_activation_energy(sg)
        assert np.isfinite(result)

    def test_result_is_finite(self):
        for _ in range(50):
            n = np.random.randint(1, 10)
            nodes = list(range(n))
            acts = {i: np.random.uniform(0.01, 1.0) for i in range(n)}
            edges = []
            strengths = {}
            confidences = {}
            for _ in range(np.random.randint(0, n * 2)):
                s = np.random.randint(0, n)
                t = np.random.randint(0, n)
                if s != t:
                    edges.append((s, t, "test_rel"))
                    strengths[(s, t, "test_rel")] = np.random.uniform(0, 1)
                    confidences[(s, t, "test_rel")] = np.random.uniform(0, 1)
            sg = _make_subgraph(nodes=nodes, activations=acts, edges=edges,
                                strengths=strengths, confidences=confidences)
            result = compute_activation_energy(sg)
            assert np.isfinite(result)


class TestSafeSum:
    def test_normal_values(self):
        result = _safe_sum([1.0, 2.0, 3.0])
        assert abs(result - 6.0) < 1e-6

    def test_empty_iterable(self):
        result = _safe_sum([])
        assert result == 0.0

    def test_skips_nan(self):
        result = _safe_sum([1.0, float("nan"), 3.0])
        assert abs(result - 4.0) < 1e-6

    def test_skips_inf(self):
        result = _safe_sum([1.0, float("inf"), 3.0])
        assert abs(result - 4.0) < 1e-6


class TestIsBad:
    def test_nan_is_bad(self):
        assert _is_bad(float("nan"))

    def test_inf_is_bad(self):
        assert _is_bad(float("inf"))

    def test_neg_inf_is_bad(self):
        assert _is_bad(float("-inf"))

    def test_normal_number_not_bad(self):
        assert not _is_bad(0.5)

    def test_zero_not_bad(self):
        assert not _is_bad(0.0)
