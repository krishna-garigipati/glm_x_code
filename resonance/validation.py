from __future__ import annotations

from typing import Iterable, List
import logging
import math
import numpy as np

from .types import Subgraph
from .config import CoreConfig

logger = logging.getLogger(__name__)


def validate_subgraph(subgraph: Subgraph, core_config: CoreConfig) -> None:
    _check(not subgraph.seed_nodes, "seed_nodes must be non-empty")
    _check(subgraph.tier_used not in (1, 2), "tier_used must be 1 or 2")

    if subgraph.query_embedding.ndim != 1 or subgraph.query_embedding.shape[0] != 384:
        raise ValueError(
            f"query_embedding shape must be (384,), got {subgraph.query_embedding.shape}"
        )
    _check_nan_inf(subgraph.query_embedding, "query_embedding")

    _validate_range(subgraph.node_activations.values(), core_config.activation.min,
                    core_config.activation.max, "activation")

    _check(not set(subgraph.nodes).issuperset(subgraph.seed_nodes),
           "seed_nodes must be included in nodes list")

    for edge in subgraph.edges:
        _check(edge[0] not in subgraph.nodes or edge[1] not in subgraph.nodes,
               f"edge {edge} references node not in node list")

    edge_keys = set(subgraph.edges)
    _check(set(subgraph.edge_strengths.keys()) != edge_keys,
           "edge_strengths keys must match edges")
    _check(set(subgraph.edge_confidences.keys()) != edge_keys,
           "edge_confidences keys must match edges")

    for v in subgraph.edge_strengths.values():
        _check(not (0.0 <= v <= 1.0), f"edge_strength out of range: {v}")
    for v in subgraph.edge_confidences.values():
        _check(not (0.0 <= v <= 1.0), f"edge_confidence out of range: {v}")


def validate_embedding(embedding: np.ndarray, name: str = "embedding") -> None:
    if not isinstance(embedding, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(embedding)}")
    if embedding.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {embedding.ndim}D")
    if embedding.shape[0] != 384:
        raise ValueError(f"{name} must have dim 384, got {embedding.shape[0]}")
    _check_nan_inf(embedding, name)


def validate_theta(theta: np.ndarray, expected_dim: int) -> None:
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"theta must be numpy array, got {type(theta)}")
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1D, got {theta.ndim}D")
    if theta.shape[0] != expected_dim:
        raise ValueError(f"theta dim mismatch: expected {expected_dim}, got {theta.shape[0]}")
    _check_nan_inf(theta, "theta")


def validate_seeds(seeds: List[int], name: str = "initial_seeds") -> None:
    if not seeds:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(s, int) for s in seeds):
        raise TypeError(f"{name} must contain only integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must not contain duplicates")


def _check(condition: bool, message: str) -> None:
    if condition:
        raise ValueError(message)


def _check_nan_inf(arr: np.ndarray, name: str) -> None:
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains infinite values")


def _validate_range(
    values: Iterable[float], min_value: float, max_value: float, label: str
) -> None:
    eps = 1e-6
    for value in values:
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{label} contains NaN or Inf: {value}")
        if value < min_value - eps or value > max_value + eps:
            raise ValueError(f"{label} value out of range [{min_value}, {max_value}]: {value}")
