from __future__ import annotations

import logging
import numpy as np

from .types import Subgraph

logger = logging.getLogger(__name__)


def compute_activation_energy(subgraph: Subgraph) -> float:
    if not subgraph.edges:
        return _safe_sum(subgraph.node_activations.values())

    energy = 0.0
    for edge in subgraph.edges:
        activation = subgraph.node_activations.get(edge[0], 0.0)
        strength = subgraph.edge_strengths.get(edge, 0.0)
        confidence = subgraph.edge_confidences.get(edge, 0.0)

        if _is_bad(activation) or _is_bad(strength) or _is_bad(confidence):
            logger.warning("Bad value in energy computation: act=%s, str=%s, conf=%s",
                           activation, strength, confidence)
            continue

        energy += activation * (strength * confidence)

    if _is_bad(energy):
        logger.error("Energy computation produced %s, returning 0.0", energy)
        return 0.0

    return energy


def _safe_sum(values) -> float:
    total = 0.0
    for v in values:
        if _is_bad(v):
            logger.warning("Bad value in sum: %s, skipping", v)
            continue
        total += v
    return total


def _is_bad(x: float) -> bool:
    return not np.isfinite(x)
