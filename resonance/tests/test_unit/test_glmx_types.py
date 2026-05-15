from __future__ import annotations

import time

import numpy as np
import pytest

from ...glmx_types import Answer, Plan, WalkResult
from ...types import Node, Edge, Subgraph


class TestPlan:
    def test_minimal_creation(self):
        p = Plan(intent_sequence=[0, 1, 2], plan_confidence=0.8, heuristic_fallback_used=False)
        assert p.intent_sequence == [0, 1, 2]
        assert p.plan_confidence == 0.8

    def test_optional_intent_names_default_none(self):
        p = Plan(intent_sequence=[0], plan_confidence=0.5, heuristic_fallback_used=False)
        assert p.intent_names is None

    def test_immutable(self):
        p = Plan(intent_sequence=[0], plan_confidence=0.5, heuristic_fallback_used=False)
        with pytest.raises((AttributeError, TypeError)):
            p.intent_sequence = [1]


class TestWalkResult:
    def test_minimal_creation(self):
        p = Plan(intent_sequence=[0], plan_confidence=0.5, heuristic_fallback_used=False)
        wr = WalkResult(
            path=[1, 2, 3],
            path_edges=["is_a", "causes"],
            path_activations=[0.8, 0.5, 0.3],
            path_confidences=[0.9, 0.7],
            path_embeddings=[np.zeros(32, dtype=np.int8) for _ in range(3)],
            walk_confidence=0.8,
            final_activation=0.3,
            steps_taken=2,
            plan_followed=p,
            timestamp=time.time(),
            intent_sequence_used=[0],
        )
        assert wr.steps_taken == 2
        assert len(wr.path) == 3

    def test_immutable(self):
        p = Plan(intent_sequence=[0], plan_confidence=0.5, heuristic_fallback_used=False)
        wr = WalkResult(
            path=[1], path_edges=[], path_activations=[0.5],
            path_confidences=[], path_embeddings=[np.zeros(32, dtype=np.int8)],
            walk_confidence=0.5, final_activation=0.5, steps_taken=0,
            plan_followed=p, timestamp=time.time(), intent_sequence_used=[0],
        )
        with pytest.raises((AttributeError, TypeError)):
            wr.path = []


class TestAnswer:
    def test_minimal_creation(self):
        p = Plan(intent_sequence=[0], plan_confidence=0.5, heuristic_fallback_used=False)
        wr = WalkResult(
            path=[1], path_edges=[], path_activations=[0.5],
            path_confidences=[], path_embeddings=[np.zeros(32, dtype=np.int8)],
            walk_confidence=0.5, final_activation=0.5, steps_taken=0,
            plan_followed=p, timestamp=time.time(), intent_sequence_used=[0],
        )
        sg = Subgraph(
            nodes=[1], node_activations={1: 0.5}, edges=[],
            edge_strengths={}, edge_confidences={},
            seed_nodes=[1], tier_used=1, activation_energy=0.5,
            query_embedding=np.zeros(384, dtype=np.float32), timestamp=time.time(),
        )
        a = Answer(
            text="Hello", confidence=0.9, intent_used=0,
            nodes_mentioned=[1], generation_method="template",
            walk_used=wr, subgraph_used=sg, timestamp=time.time(),
        )
        assert a.text == "Hello"
        assert a.confidence == 0.9

    def test_optional_reasoning_trace_default_none(self):
        p = Plan(intent_sequence=[0], plan_confidence=0.5, heuristic_fallback_used=False)
        wr = WalkResult(
            path=[1], path_edges=[], path_activations=[0.5],
            path_confidences=[], path_embeddings=[np.zeros(32, dtype=np.int8)],
            walk_confidence=0.5, final_activation=0.5, steps_taken=0,
            plan_followed=p, timestamp=time.time(), intent_sequence_used=[0],
        )
        sg = Subgraph(
            nodes=[1], node_activations={1: 0.5}, edges=[],
            edge_strengths={}, edge_confidences={},
            seed_nodes=[1], tier_used=1, activation_energy=0.5,
            query_embedding=np.zeros(384, dtype=np.float32), timestamp=time.time(),
        )
        a = Answer(
            text="Test", confidence=0.5, intent_used=0,
            nodes_mentioned=[1], generation_method="template",
            walk_used=wr, subgraph_used=sg, timestamp=time.time(),
        )
        assert a.reasoning_trace is None
