from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class Node:
    id: int
    label: str
    node_type: str
    embedding: np.ndarray
    activation: float = 0.01
    use_count: int = 0
    create_time: float = 0.0
    sense_id: Optional[int] = None

    def __post_init__(self):
        if not (0.01 <= self.activation <= 1.0):
            raise ValueError(f"Node activation must be in [0.01, 1.0], got {self.activation}")


@dataclass
class Edge:
    source: int
    target: int
    relation_type: str
    strength: float = 0.5
    confidence: float = 0.5
    last_used: float = 0.0
    frequency: int = 1

    VALID_RELATIONS = (
        "is_a", "has_property", "causes", "caused_by",
        "follows", "precedes", "contradicts", "supports",
        "associated_with", "example_of", "part_of",
        "synonym", "antonym", "temporal_coincident",
        "spatial_near", "linguistic_maps",
    )

    def __post_init__(self):
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"Edge strength must be in [0.0, 1.0], got {self.strength}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Edge confidence must be in [0.0, 1.0], got {self.confidence}")

    @property
    def key(self) -> Tuple[int, int, str]:
        return (self.source, self.target, self.relation_type)


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

    def __post_init__(self):
        if self.tier_used not in (1, 2):
            raise ValueError(f"tier_used must be 1 or 2, got {self.tier_used}")


@dataclass
class Plan:
    intent_sequence: Optional[List[int]] = None
    plan_confidence: float = 0.5
    heuristic_fallback_used: bool = False
    intent_names: Optional[List[str]] = None
    relation_chain: Optional[List[str]] = None

    def __post_init__(self):
        if self.intent_sequence is None and self.relation_chain is None:
            raise ValueError("Plan must carry at least one of intent_sequence or relation_chain")
        if self.intent_sequence is not None and not self.intent_sequence:
            raise ValueError("intent_sequence must be non-empty")
        if not (0.0 <= self.plan_confidence <= 1.0):
            raise ValueError(f"plan_confidence must be in [0.0, 1.0], got {self.plan_confidence}")


@dataclass
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
    relation_chain_used: Optional[List[str]] = None

    def __post_init__(self):
        if not self.path:
            raise ValueError("path must be non-empty")
        if not (0.0 <= self.walk_confidence <= 1.0):
            raise ValueError(f"walk_confidence must be in [0.0, 1.0], got {self.walk_confidence}")


@dataclass
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

    def __post_init__(self):
        if not self.text:
            raise ValueError("text must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


@dataclass
class InternalRewardComponents:
    goal_alignment: float
    value_alignment: float
    emotional_consistency: float
    weighted_sum: float

    def __post_init__(self):
        if not (0.0 <= self.goal_alignment <= 1.0):
            raise ValueError(f"goal_alignment must be in [0.0, 1.0]")
        if not (0.0 <= self.value_alignment <= 1.0):
            raise ValueError(f"value_alignment must be in [0.0, 1.0]")
        if not (0.0 <= self.emotional_consistency <= 1.0):
            raise ValueError(f"emotional_consistency must be in [0.0, 1.0]")
        expected = 0.4 * self.goal_alignment + 0.3 * self.value_alignment + 0.3 * self.emotional_consistency
        if abs(self.weighted_sum - expected) > 1e-6:
            raise ValueError(f"weighted_sum ({self.weighted_sum}) != 0.4*goal + 0.3*value + 0.3*emotion ({expected})")


@dataclass
class EligibilityTrace:
    edge_key: Tuple[int, int, str]
    eligibility: float
    time_contribution: Optional[List[float]] = None


class GraphStoreInterface(ABC):
    @abstractmethod
    def add_node(self, node_id: int, label: str, node_type: str, embedding: np.ndarray, activation: float = 0.01) -> bool:
        ...

    @abstractmethod
    def get_node(self, node_id: int) -> Optional[Node]:
        ...

    @abstractmethod
    def update_node_embedding(self, node_id: int, embedding: np.ndarray) -> bool:
        ...

    @abstractmethod
    def add_edge(self, source: int, target: int, relation: str, strength: float = 0.5, confidence: float = 0.5) -> bool:
        ...

    @abstractmethod
    def get_edge(self, source: int, target: int, relation: str) -> Optional[Edge]:
        ...

    @abstractmethod
    def update_edge_weights(self, updates: Dict[Tuple[int, int, str], Tuple[float, float]]) -> None:
        ...

    @abstractmethod
    def get_neighbors(self, node_id: int, relation_filter: Optional[List[str]] = None) -> List[Tuple[int, Edge]]:
        ...

    @abstractmethod
    def get_subgraph_activated(self, seed_nodes: List[int], max_nodes: int = 1000) -> Subgraph:
        ...

    @abstractmethod
    def get_subgraph_by_embedding_similarity(self, query_embedding: np.ndarray, top_k: int = 100) -> Subgraph:
        ...

    @abstractmethod
    def prune(self, utility_threshold: float = 0.01) -> int:
        ...

    @abstractmethod
    def save_checkpoint(self, filepath: str) -> bool:
        ...

    @abstractmethod
    def load_checkpoint(self, filepath: str) -> bool:
        ...


class ResonanceEngineInterface(ABC):
    @abstractmethod
    def resonate(self, query_embedding: np.ndarray, graph: GraphStoreInterface, initial_seeds: List[int], tier: int = 1) -> Subgraph:
        ...

    @abstractmethod
    def resonate_with_theta(self, theta: np.ndarray, query_embedding: np.ndarray, graph: GraphStoreInterface, initial_seeds: List[int]) -> Subgraph:
        ...

    @abstractmethod
    def get_theta(self) -> np.ndarray:
        ...

    @abstractmethod
    def set_theta(self, theta: np.ndarray) -> None:
        ...

    @abstractmethod
    def propose_theta_mutation(self) -> np.ndarray:
        ...

    @abstractmethod
    def update_es_with_reward(self, reward: float, theta_used: np.ndarray) -> None:
        ...

    @abstractmethod
    def compute_activation_energy(self, subgraph: Subgraph) -> float:
        ...

    @abstractmethod
    def check_resonance_convergence(self, activation_history: List[np.ndarray], epsilon: float = 0.001) -> bool:
        ...

    @abstractmethod
    def get_analogy_leaps(self, target_node: int, graph: GraphStoreInterface, top_k: int = 3) -> List[Tuple[int, float]]:
        ...


class G2PPlannerInterface(ABC):
    @abstractmethod
    def plan(self, subgraph: Subgraph) -> Plan:
        ...

    @abstractmethod
    def plan_batch(self, subgraphs: List[Subgraph]) -> List[Plan]:
        ...

    @abstractmethod
    def get_plan_confidence(self, subgraph: Subgraph) -> float:
        ...


class GraphWalkerInterface(ABC):
    @abstractmethod
    def walk(self, subgraph: Subgraph, plan: Plan) -> WalkResult:
        ...

    @abstractmethod
    def compute_eligibility_trace(self, walk: WalkResult, subgraph: Subgraph) -> Dict[Tuple[int, int, str], float]:
        ...

    @abstractmethod
    def get_walk_confidence(self, walk: WalkResult) -> float:
        ...


class MicroDecoderInterface(ABC):
    @abstractmethod
    def decode(self, walk: WalkResult, plan: Plan) -> Answer:
        ...

    @abstractmethod
    def get_confidence(self, walk: WalkResult, plan: Plan, generated_text: str) -> float:
        ...
