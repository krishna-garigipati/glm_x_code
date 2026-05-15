from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class Node:
    node_id: int
    label: str
    node_type: str
    embedding: np.ndarray
    activation: float
    use_count: int
    create_time: float


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    relation: str
    strength: float
    confidence: float
    last_used: float
    frequency: int


@dataclass(frozen=True)
class Subgraph:
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
