"""Eligibility trace calculation for Walker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class EligibilityTrace:
    edge_key: Tuple[int, int, str]
    eligibility: float
    time_contribution: List[float] | None = None


def compute_eligibility_trace(
    path: List[int],
    path_edges: List[str],
    path_activations: List[float],
    edge_strengths: Dict[Tuple[int, int, str], float],
    edge_confidences: Dict[Tuple[int, int, str], float],
    gamma: float,
    trace_key_format: str = "{source}:{target}:{relation}",
) -> Dict[str, float]:
    trace: Dict[str, float] = {}
    for step_index, edge_type in enumerate(path_edges):
        source = path[step_index]
        target = path[step_index + 1]
        edge_key = (source, target, edge_type)
        formatted_key = trace_key_format.format(source=source, target=target, relation=edge_type)
        strength = edge_strengths[edge_key]
        confidence = edge_confidences[edge_key]
        activation = path_activations[step_index] if path_activations else 0.0
        temporal_factor = 1.0
        contribution = (gamma ** step_index) * activation * (strength * confidence * temporal_factor)
        trace[formatted_key] = trace.get(formatted_key, 0.0) + contribution
    return trace
