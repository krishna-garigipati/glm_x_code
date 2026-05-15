from __future__ import annotations

import time

import numpy as np
import pytest

from ...types import Subgraph
from ...validation import (
    validate_embedding,
    validate_seeds,
    validate_subgraph,
    validate_theta,
)
from ..fixtures.config_provider import build_minimal_core_config


def _make_subgraph(
    nodes=None, activations=None, edges=None,
    strengths=None, confidences=None, seed_nodes=None,
    tier=1, energy=0.0, embedding=None,
) -> Subgraph:
    if nodes is None:
        nodes = [1, 2]
    if activations is None:
        activations = {1: 0.5, 2: 0.3}
    if edges is None:
        edges = [(1, 2, "is_a")]
    if strengths is None:
        strengths = {(1, 2, "is_a"): 0.8}
    if confidences is None:
        confidences = {(1, 2, "is_a"): 0.9}
    if seed_nodes is None:
        seed_nodes = [1]
    return Subgraph(
        nodes=nodes,
        node_activations=activations,
        edges=edges,
        edge_strengths=strengths,
        edge_confidences=confidences,
        seed_nodes=seed_nodes,
        tier_used=tier,
        activation_energy=energy,
        query_embedding=embedding if embedding is not None else np.zeros(384, dtype=np.float32),
        timestamp=time.time(),
    )


class TestValidateSubgraph:
    def test_valid_subgraph_passes(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph()
        validate_subgraph(sg, cc)

    def test_empty_seeds_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(seed_nodes=[])
        with pytest.raises(ValueError, match="seed_nodes"):
            validate_subgraph(sg, cc)

    def test_invalid_tier_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(tier=3)
        with pytest.raises(ValueError, match="tier_used"):
            validate_subgraph(sg, cc)

    def test_wrong_embedding_shape_raises(self):
        cc = build_minimal_core_config()
        emb = np.zeros(128, dtype=np.float32)
        sg = _make_subgraph(embedding=emb)
        with pytest.raises(ValueError, match="query_embedding shape"):
            validate_subgraph(sg, cc)

    def test_nan_in_embedding_raises(self):
        cc = build_minimal_core_config()
        emb = np.full(384, np.nan, dtype=np.float32)
        sg = _make_subgraph(embedding=emb)
        with pytest.raises(ValueError, match="NaN"):
            validate_subgraph(sg, cc)

    def test_inf_in_embedding_raises(self):
        cc = build_minimal_core_config()
        emb = np.full(384, np.inf, dtype=np.float32)
        sg = _make_subgraph(embedding=emb)
        with pytest.raises(ValueError, match="infinite"):
            validate_subgraph(sg, cc)

    def test_activation_out_of_range_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(activations={1: 5.0})
        with pytest.raises(ValueError, match="activation.*range"):
            validate_subgraph(sg, cc)

    def test_activation_below_min_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(activations={1: -0.5})
        with pytest.raises(ValueError, match="activation.*range"):
            validate_subgraph(sg, cc)

    def test_seeds_not_in_nodes_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(seed_nodes=[99])
        with pytest.raises(ValueError, match="seed_nodes"):
            validate_subgraph(sg, cc)

    def test_edge_references_nonexistent_node_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(edges=[(1, 99, "is_a")])
        with pytest.raises(ValueError, match="edge"):
            validate_subgraph(sg, cc)

    def test_edge_strength_out_of_range_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(strengths={(1, 2, "is_a"): 1.5})
        with pytest.raises(ValueError, match="edge_strength"):
            validate_subgraph(sg, cc)

    def test_edge_confidence_out_of_range_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(confidences={(1, 2, "is_a"): -0.1})
        with pytest.raises(ValueError, match="edge_confidence"):
            validate_subgraph(sg, cc)

    def test_edge_strength_keys_mismatch_raises(self):
        cc = build_minimal_core_config()
        sg = _make_subgraph(
            edges=[(1, 2, "is_a")],
            strengths={(1, 2, "causes"): 0.8},
        )
        with pytest.raises(ValueError, match="edge_strengths keys"):
            validate_subgraph(sg, cc)


class TestValidateEmbedding:
    def test_valid_embedding_passes(self):
        emb = np.zeros(384, dtype=np.float32)
        validate_embedding(emb)

    def test_non_numpy_raises(self):
        with pytest.raises(TypeError, match="numpy array"):
            validate_embedding([0.0] * 384)

    def test_wrong_ndim_raises(self):
        emb = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="1D"):
            validate_embedding(emb)

    def test_wrong_shape_raises(self):
        emb = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValueError, match="dim 384"):
            validate_embedding(emb)

    def test_nan_raises(self):
        emb = np.full(384, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            validate_embedding(emb)

    def test_inf_raises(self):
        emb = np.full(384, np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="infinite"):
            validate_embedding(emb)


class TestValidateSeeds:
    def test_valid_seeds_passes(self):
        validate_seeds([1, 2, 3])

    def test_empty_seeds_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_seeds([])

    def test_duplicate_seeds_raises(self):
        with pytest.raises(ValueError, match="duplicates"):
            validate_seeds([1, 1])

    def test_non_integer_seeds_raises(self):
        with pytest.raises(TypeError, match="integers"):
            validate_seeds([1, "two"])


class TestValidateTheta:
    def test_valid_theta_passes(self):
        theta = np.zeros(48, dtype=np.float32)
        validate_theta(theta, 48)

    def test_wrong_shape_raises(self):
        theta = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="dim mismatch"):
            validate_theta(theta, 48)

    def test_non_numpy_raises(self):
        with pytest.raises(TypeError, match="numpy array"):
            validate_theta([0.0] * 48, 48)

    def test_nan_raises(self):
        theta = np.full(48, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            validate_theta(theta, 48)

    def test_inf_raises(self):
        theta = np.full(48, np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="infinite"):
            validate_theta(theta, 48)
