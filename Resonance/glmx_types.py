from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .types import Node, Edge, Subgraph


@dataclass(frozen=True)
class Plan:
    intent_sequence: List[int]
    plan_confidence: float
    heuristic_fallback_used: bool
    intent_names: Optional[List[str]] = None


@dataclass(frozen=True)
class WalkResult:
    path: List[int]
    path_edges: List[str]
    path_activations: List[float]
    path_confidences: List[float]
    path_embeddings: List[np.ndarray]
    walk_confidence: float
    final_activation: float
    steps_taken: int
    plan_followed: Plan
    timestamp: float
    intent_sequence_used: List[int]


@dataclass(frozen=True)
class Answer:
    text: str
    confidence: float
    intent_used: int
    nodes_mentioned: List[int]
    generation_method: str
    walk_used: WalkResult
    subgraph_used: Subgraph
    timestamp: float
    reasoning_trace: Optional[Dict] = None


__all__ = [
    "Node", "Edge", "Subgraph",
    "Plan", "WalkResult", "Answer",
]
