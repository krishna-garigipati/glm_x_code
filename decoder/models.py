from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from dataclass_schema import Answer, Plan, WalkResult
except Exception:
    @dataclass(frozen=True)
    class WalkResult:
        path: List[int]
        path_edges: List[str]
        path_labels: List[str]
        path_activations: List[float] = field(default_factory=list)
        path_confidences: List[float] = field(default_factory=list)
        path_embeddings: List = field(default_factory=list)
        walk_confidence: float = 1.0
        final_activation: float = 0.0
        steps_taken: int = 0
        plan_followed: Any = None
        timestamp: float = 0.0
        intent_sequence_used: List[int] = field(default_factory=list)

    @dataclass(frozen=True)
    class Plan:
        intent_sequence: List[int]
        plan_confidence: float = 1.0
        heuristic_fallback_used: bool = False
        intent_names: List[str] = field(default_factory=list)

    @dataclass(frozen=True)
    class Answer:
        text: str
        confidence: float
        intent_used: int = 0
        nodes_mentioned: List[int] = field(default_factory=list)
        generation_method: str = "template"
        walk_used: Any = None
        subgraph_used: Any = None
        reasoning_trace: Optional[Dict[str, Any]] = None
        timestamp: float = 0.0
