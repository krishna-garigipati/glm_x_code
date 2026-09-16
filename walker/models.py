"""Dataclass models for Walker I/O contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

try:
    import numpy as np
except ImportError:
    np = None


@dataclass(frozen=True)
class Subgraph:
    nodes: List[int]
    node_activations: Dict[int, float]
    edges: List[Tuple[int, int, str]]
    edge_strengths: Dict[Tuple[int, int, str], float]
    edge_confidences: Dict[Tuple[int, int, str], float]
    seed_nodes: List[int]
    tier_used: int
    activation_energy: float
    query_embedding: "np.ndarray"  # type: ignore[name-defined]
    timestamp: float
    node_embeddings: Optional[Dict[int, "np.ndarray"]] = None

    def validate(self, activation_min: float, activation_max: float) -> None:
        if not self.nodes:
            raise ValueError("Subgraph.nodes must be non-empty")
        if not self.seed_nodes:
            raise ValueError("Subgraph.seed_nodes must be non-empty")
        if self.tier_used not in (1, 2):
            raise ValueError("Subgraph.tier_used must be 1 or 2")
        if len(self.edges) != len(self.edge_strengths) or len(self.edges) != len(self.edge_confidences):
            raise ValueError("Subgraph edge fields must have matching keys")
        for node_id in self.nodes:
            if node_id not in self.node_activations:
                raise ValueError(f"Missing activation for node {node_id}")
            activation = self.node_activations[node_id]
            if not (activation_min <= activation <= activation_max):
                raise ValueError("Node activation out of range")
        edge_set = set(self.edges)
        if edge_set != set(self.edge_strengths.keys()) or edge_set != set(self.edge_confidences.keys()):
            raise ValueError("Edge strengths/confidences keys must match edges list")
        for source, target, _ in self.edges:
            if source not in self.nodes or target not in self.nodes:
                raise ValueError("Edges must reference nodes inside Subgraph.nodes")
        if np is not None:
            if not isinstance(self.query_embedding, np.ndarray):
                raise ValueError("query_embedding must be a numpy array")
            if self.query_embedding.shape != (384,):
                raise ValueError("query_embedding must have shape (384,)")
        if self.node_embeddings is not None:
            for nid in self.node_embeddings:
                if nid not in self.nodes:
                    raise ValueError(f"node_embeddings key {nid} not in Subgraph.nodes")


@dataclass(frozen=True)
class Plan:
    intent_sequence: Optional[List[int]] = None  # DORMANT (DEVIATION 9)
    plan_confidence: float = 0.5
    heuristic_fallback_used: bool = False
    intent_names: Optional[List[str]] = None  # DORMANT (DEVIATION 9)
    relation_chain: Optional[List[str]] = None  # ordered relation chain (source of truth)

    def validate(self) -> None:
        if self.relation_chain is not None:
            if not self.relation_chain:
                raise ValueError("Plan.relation_chain must be non-empty when set")
            if len(self.relation_chain) > 8:
                raise ValueError("Plan.relation_chain length must be <= 8")
            if not all(isinstance(r, str) for r in self.relation_chain):
                raise ValueError("Plan.relation_chain entries must be relation strings")
        if self.intent_sequence is not None:
            if not self.intent_sequence:
                raise ValueError("Plan.intent_sequence must be non-empty when set")
            if len(self.intent_sequence) > 8:
                raise ValueError("Plan.intent_sequence length must be <= 8")
            for intent_id in self.intent_sequence:
                if not (0 <= intent_id <= 15):
                    raise ValueError("Intent IDs must be in [0, 15]")
            if self.intent_names is not None and len(self.intent_names) != len(self.intent_sequence):
                raise ValueError("intent_names length must match intent_sequence length")
        if not (0.0 <= self.plan_confidence <= 1.0):
            raise ValueError("plan_confidence must be in [0.0, 1.0]")
        if self.relation_chain is None and self.intent_sequence is None:
            raise ValueError("Plan must carry at least one of relation_chain or intent_sequence")


@dataclass(frozen=True)
class WalkResult:
    path: List[int]
    path_edges: List[str]
    path_activations: List[float]
    path_confidences: List[float]
    path_embeddings: List["np.ndarray"]  # type: ignore[name-defined]
    walk_confidence: float
    final_activation: float
    steps_taken: int
    plan_followed: Plan
    timestamp: float
    intent_sequence_used: List[int]  # DORMANT (DEVIATION 9)
    relation_chain_used: List[str] = field(default_factory=list)  # chain actually walked

    @classmethod
    def build(
        cls,
        path: List[int],
        path_edges: List[str],
        path_activations: List[float],
        path_confidences: List[float],
        path_embeddings: List["np.ndarray"],  # type: ignore[name-defined]
        plan: Plan,
    ) -> "WalkResult":
        return cls(
            path=path,
            path_edges=path_edges,
            path_activations=path_activations,
            path_confidences=path_confidences,
            path_embeddings=path_embeddings,
            walk_confidence=1.0,
            final_activation=path_activations[-1] if path_activations else 0.0,
            steps_taken=len(path_edges),
            plan_followed=plan,
            timestamp=time.time(),
            intent_sequence_used=list(plan.intent_sequence or []),
            relation_chain_used=list(plan.relation_chain or []),
        )

    def validate(self, activation_min: float, activation_max: float) -> None:
        if not self.path:
            raise ValueError("WalkResult.path must be non-empty")
        if len(self.path_edges) != len(self.path) - 1:
            raise ValueError("path_edges length must be len(path) - 1")
        if self.path_activations:
            if len(self.path_activations) != len(self.path):
                raise ValueError("path_activations length must match path length")
            for activation in self.path_activations:
                if not (activation_min <= activation <= activation_max):
                    raise ValueError("path activation out of range")
        if self.path_confidences:
            if len(self.path_confidences) != len(self.path) - 1:
                raise ValueError("path_confidences length must be len(path) - 1")
            for confidence in self.path_confidences:
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError("path confidence out of range")
        if len(self.path_embeddings) != len(self.path):
            raise ValueError("path_embeddings length must match path length")
        if not (0.0 <= self.walk_confidence <= 1.0):
            raise ValueError("walk_confidence out of range")
        if self.path_activations and not (activation_min <= self.final_activation <= activation_max):
            raise ValueError("final_activation out of range")
        for i in range(1, len(self.path)):
            if self.path[i] == self.path[i - 1]:
                raise ValueError("consecutive duplicate node in path")

    def to_tuples(self) -> List[Tuple[int, str, float, float]]:
        result: List[Tuple[int, str, float, float]] = []
        for i, node_id in enumerate(self.path):
            if i == 0:
                edge_type = ""
                confidence = 0.0
            else:
                edge_type = self.path_edges[i - 1]
                confidence = self.path_confidences[i - 1] if self.path_confidences else 0.0
            activation = self.path_activations[i] if self.path_activations else 0.0
            result.append((node_id, edge_type, activation, confidence))
        return result
