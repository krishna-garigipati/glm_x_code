import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Subgraph:
    nodes: List[int]
    node_activations: Dict[int, float]
    edges: List[Tuple[int, int, str]]
    edge_strengths: Dict[Tuple[int, int, str], float]
    edge_confidences: Dict[Tuple[int, int, str], float]
    seed_nodes: List[int]
    tier_used: int
    activation_energy: float
    query_embedding: np.ndarray
    timestamp: float

    def validate(self):
        for v in self.node_activations.values():
            if not (0.01 <= v <= 1.0):
                raise ValueError(f"activation {v} outside [0.01, 1.0]")
        node_set = set(self.nodes)
        for s, t, _ in self.edges:
            if s not in node_set or t not in node_set:
                raise ValueError(f"edge ({s},{t}) nodes not in nodes list")
        edge_keys = {(s, t, r) for s, t, r in self.edges}
        if set(self.edge_strengths.keys()) != edge_keys:
            raise ValueError("edge_strengths keys must match edges")
        if set(self.edge_confidences.keys()) != edge_keys:
            raise ValueError("edge_confidences keys must match edges")
        if not self.seed_nodes:
            raise ValueError("seed_nodes must be non-empty")
        if self.tier_used not in (1, 2):
            raise ValueError("tier_used must be 1 or 2")
        if self.query_embedding.shape != (384,):
            raise ValueError("query_embedding shape must be (384,)")


@dataclass
class Plan:
    # Deviation 9: intent_sequence/intent_names are DORMANT (superseded by relation_chain).
    # Kept Optional so legacy callers still construct Plan objects; the runtime pipeline
    # reads only relation_chain.
    intent_sequence: Optional[List[int]] = None
    plan_confidence: float = 0.5
    heuristic_fallback_used: bool = False
    intent_names: Optional[List[str]] = None
    relation_chain: Optional[List[str]] = None

    def validate(self):
        if not (0.0 <= self.plan_confidence <= 1.0):
            raise ValueError("plan_confidence must be in [0.0, 1.0]")
        if self.relation_chain is not None:
            if not self.relation_chain:
                raise ValueError("relation_chain must be non-empty when set")
            if len(self.relation_chain) > 8:
                raise ValueError("relation_chain length must be in [1, 8]")
            if not all(isinstance(r, str) for r in self.relation_chain):
                raise ValueError("relation_chain entries must be relation strings")
        if self.intent_sequence is not None:
            for i in self.intent_sequence:
                if not (0 <= i <= 15):
                    raise ValueError(f"intent_id {i} outside [0, 15]")
            if not (1 <= len(self.intent_sequence) <= 8):
                raise ValueError("intent_sequence length must be in [1, 8]")
            if self.intent_names is not None and len(self.intent_names) != len(self.intent_sequence):
                raise ValueError("intent_names length must match intent_sequence length")
        elif self.intent_names is not None:
            raise ValueError("intent_names requires intent_sequence")
