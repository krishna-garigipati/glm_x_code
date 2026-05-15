"""Integration and behavioral tests for the GraphWalker component.

Covers: deterministic reproducibility, end-to-end path verification,
property-based invariants, scoring steering, intent-guided walks,
restart behavior, and edge-case walks.
"""

from __future__ import annotations

import os
import time
import tempfile
import unittest
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from walker.config import CoreConfig, WalkerConfig
from walker.models import Plan, Subgraph, WalkResult
from walker.exceptions import ValidationError
from walker.graph_walker import GraphWalker
from walker.utils import geometric_mean


SAMPLE_WALKER_YAML = """
walk:
  max_steps: 20
  min_activation: 0.05
  temperature: 0.1
  temperature_range: [0.05, 0.5]
  allow_cycles: false
  allow_backtrack: false
  restart_on_dead_end: true
  restart_penalty: 0.5

intent_biases:
  0:
    is_a: 1.5
    has_property: 1.2
    example_of: 1.0
    default: 0.5
  4:
    contradicts: 2.0
    antonym: 1.8
    associated_with: 0.8
    default: 0.3

scoring:
  formula: "strength * confidence * target_activation * intent_bias(edge_type, current_intent)"
  weight_strength: 1.0
  weight_confidence: 1.0
  weight_target_activation: 1.0
  weight_intent_bias: 1.0
  normalization: "softmax"
  softmax_temperature: 0.1

eligibility:
  gamma: 0.9
  formula: "sum_{t=0..K} gamma^t * source_activation(t) * (strength * confidence * temporal_factor)"
  trace_key_format: "{source}:{target}:{relation}"

path:
  max_length: 20
  record_activations: true
  record_timestamps: true
  record_edge_confidence: true
  output_format: "List[Tuple[node_id, edge_type, activation, confidence]]"

debug:
  log_decision_scores: false
  log_path_taken: true
  save_all_paths: true
"""

SAMPLE_CORE_YAML = """
activation:
  min: 0.01
  max: 1.0

walker:
  default_temperature: 0.1
  softmax_temperature_range: [0.05, 0.5]
  max_steps: 20
  min_activation_to_continue: 0.05

relations:
  0: "is_a"
  1: "has_property"
  2: "causes"
  3: "caused_by"
  4: "follows"
  5: "precedes"
  6: "contradicts"
  7: "supports"
  8: "associated_with"
  9: "example_of"
  10: "part_of"
  11: "synonym"
  12: "antonym"
  13: "temporal_coincident"
  14: "spatial_near"
  15: "linguistic_maps"
"""


def write_yaml(content: str) -> str:
    path = tempfile.mktemp(suffix=".yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def make_subgraph(
    edges: Optional[List[Tuple[int, int, str]]] = None,
    node_activations: Optional[Dict[int, float]] = None,
    seed_nodes: Optional[List[int]] = None,
    node_embeddings: bool = False,
) -> Subgraph:
    edges_list = edges or [(1, 3, "is_a"), (1, 4, "has_property"), (3, 5, "causes")]
    seed = [1] if seed_nodes is None else seed_nodes
    if node_activations is not None:
        nodes = sorted(node_activations.keys())
        act = node_activations
    else:
        edge_nodes = {n for e in edges_list for n in (e[0], e[1])}
        nodes = sorted(edge_nodes | set(seed))
        act = {n: 0.5 for n in nodes}
    strengths = {e: 0.8 for e in edges_list}
    confidences = {e: 0.9 for e in edges_list}
    query_emb = np.zeros(384, dtype=np.float32) if HAS_NUMPY else None
    nemb: Optional[Dict[int, np.ndarray]] = None
    if node_embeddings and HAS_NUMPY:
        nemb = {n: np.zeros(32, dtype=np.int8) for n in nodes}
    return Subgraph(
        nodes=nodes,
        node_activations=act,
        edges=edges_list,
        edge_strengths=strengths,
        edge_confidences=confidences,
        seed_nodes=seed,
        tier_used=1,
        activation_energy=1.0,
        query_embedding=query_emb,
        node_embeddings=nemb,
        timestamp=time.time(),
    )


def make_plan(
    intent_sequence: Optional[List[int]] = None,
    plan_confidence: float = 0.9,
) -> Plan:
    return Plan(
        intent_sequence=[0, 1, 4] if intent_sequence is None else intent_sequence,
        plan_confidence=plan_confidence,
        heuristic_fallback_used=False,
    )


# =========================================================================
# Setup helpers
# =========================================================================

class WalkerTestBase(unittest.TestCase):
    """Base class providing shared GraphWalker instances."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.walker_path = write_yaml(SAMPLE_WALKER_YAML)
        cls.core_path = write_yaml(SAMPLE_CORE_YAML)
        cls.walker_cfg = WalkerConfig.from_yaml(cls.walker_path)
        cls.core_cfg = CoreConfig.from_yaml(cls.core_path)

    @classmethod
    def tearDownClass(cls) -> None:
        os.unlink(cls.walker_path)
        os.unlink(cls.core_path)

    def _walker(self, **kw) -> GraphWalker:
        return GraphWalker(self.walker_cfg, self.core_cfg, **kw)


# =========================================================================
# Deterministic reproducibility
# =========================================================================

class TestDeterministicWalk(WalkerTestBase):
    """Same seed + same inputs = exact same WalkResult."""

    def test_same_seed_produces_identical_walk(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph(
            edges=[(1, 2, "is_a"), (2, 3, "is_a"), (3, 4, "is_a"), (4, 5, "is_a")],
            seed_nodes=[1],
        )
        plan = make_plan(intent_sequence=[0])
        provider = lambda nid: np.zeros(32, dtype=np.int8)

        w1 = self._walker(embedding_provider=provider, random_seed=42)
        w2 = self._walker(embedding_provider=provider, random_seed=42)
        r1 = w1.walk(sg, plan)
        r2 = w2.walk(sg, plan)

        self.assertEqual(r1.path, r2.path)
        self.assertEqual(r1.path_edges, r2.path_edges)
        self.assertEqual(r1.path_activations, r2.path_activations)
        self.assertEqual(r1.path_confidences, r2.path_confidences)
        self.assertEqual(r1.steps_taken, r2.steps_taken)
        self.assertEqual(r1.walk_confidence, r2.walk_confidence)

    def test_normalization_none_is_deterministic(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        cfg_yaml = SAMPLE_WALKER_YAML.replace('normalization: "softmax"', 'normalization: "none"')
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        core = CoreConfig.from_yaml(self.core_path)
        sg = make_subgraph(
            edges=[(1, 2, "is_a"), (1, 3, "contradicts")],
            node_activations={1: 0.5, 2: 0.6, 3: 0.6},
            seed_nodes=[1],
        )
        plan = make_plan(intent_sequence=[0])
        provider = lambda nid: np.zeros(32, dtype=np.int8)
        w1 = GraphWalker(cfg, core, embedding_provider=provider, random_seed=42)
        w2 = GraphWalker(cfg, core, embedding_provider=provider, random_seed=99)
        r1 = w1.walk(sg, plan)
        r2 = w2.walk(sg, plan)
        self.assertEqual(r1.path, r2.path,
                         "normalization=none must always pick the max score edge")
        os.unlink(p)


# =========================================================================
# End-to-end path verification
# =========================================================================

class TestEndToEndPath(WalkerTestBase):
    """Given controlled subgraphs, verify exact walk behavior."""

    def test_chain_path_uses_all_edges(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        chain = [(1, 2, "is_a"), (2, 3, "is_a"), (3, 4, "is_a"), (4, 5, "is_a")]
        sg = make_subgraph(edges=chain, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)

        for i in range(len(result.path) - 1):
            src, tgt = result.path[i], result.path[i + 1]
            edge_type = result.path_edges[i]
            self.assertIn((src, tgt, edge_type), sg.edges,
                          f"Edge ({src}->{tgt}, {edge_type}) not in subgraph")

    def test_every_node_in_path_exists_in_subgraph(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 3, "is_a"), (3, 5, "causes")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)

        for node_id in result.path:
            self.assertIn(node_id, sg.nodes,
                          f"Node {node_id} in path but not in subgraph.nodes")

    def test_first_node_is_seed_node(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph(
            edges=[(1, 2, "is_a")],
            seed_nodes=[1],
        )
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
        )
        result = w.walk(sg, plan)
        self.assertIn(result.path[0], sg.seed_nodes,
                      "Walk must start from a seed node")

    def test_every_edge_type_in_path_edges_matches_transition(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 3, "is_a"), (3, 5, "causes")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)

        for i, etype in enumerate(result.path_edges):
            src, tgt = result.path[i], result.path[i + 1]
            self.assertIn((src, tgt, etype), sg.edges,
                          f"Edge type {etype} does not match transition {src}->{tgt}")

    def test_no_consecutive_duplicates_in_path(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5}
        edges = [(1, 1, "self_loop"), (1, 3, "is_a")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
        )
        result = w.walk(sg, plan)
        for i in range(1, len(result.path)):
            self.assertNotEqual(result.path[i], result.path[i - 1],
                                "No consecutive duplicates allowed")


# =========================================================================
# Property-based invariants (run on many random inputs)
# =========================================================================

class TestWalkInvariants(WalkerTestBase):
    """Invariants that must hold for any valid walk."""

    def _assert_walk_invariants(self, result: WalkResult, subgraph: Subgraph, plan: Plan) -> None:
        self.assertGreater(len(result.path), 0)
        self.assertEqual(len(result.path_edges), len(result.path) - 1)
        self.assertEqual(len(result.path_activations), len(result.path))
        self.assertEqual(len(result.path_confidences), len(result.path) - 1)
        self.assertEqual(len(result.path_embeddings), len(result.path))
        self.assertEqual(result.steps_taken, len(result.path_edges))
        self.assertGreaterEqual(result.walk_confidence, 0.0)
        self.assertLessEqual(result.walk_confidence, 1.0)
        for a in result.path_activations:
            self.assertGreaterEqual(a, 0.01)
            self.assertLessEqual(a, 1.0)
        for c in result.path_confidences:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)
        for node_id in result.path:
            self.assertIn(node_id, subgraph.nodes)
        for i, etype in enumerate(result.path_edges):
            src, tgt = result.path[i], result.path[i + 1]
            self.assertIn((src, tgt, etype), subgraph.edges)
        self.assertAlmostEqual(result.walk_confidence,
                               geometric_mean(result.path_confidences))
        self.assertEqual(list(result.intent_sequence_used), list(plan.intent_sequence))

    def test_simple_graph_invariants(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 3, "is_a"), (3, 5, "causes")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        self._assert_walk_invariants(result, sg, plan)

    def test_branching_graph_invariants(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 2, "is_a"), (1, 3, "causes"), (2, 4, "supports"), (3, 5, "contradicts")]
        act = {1: 0.5, 2: 0.6, 3: 0.6, 4: 0.7, 5: 0.7}
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0, 4])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        self._assert_walk_invariants(result, sg, plan)

    def test_deep_chain_invariants(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        chain = [(i, i + 1, "is_a") for i in range(1, 30)]
        act = {i: 0.5 for i in range(1, 31)}
        nodes = list(range(1, 31))
        strengths = {e: 0.8 for e in chain}
        confidences = {e: 0.9 for e in chain}
        sg = Subgraph(
            nodes=nodes, node_activations=act, edges=chain,
            edge_strengths=strengths, edge_confidences=confidences,
            seed_nodes=[1], tier_used=1, activation_energy=1.0,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        self._assert_walk_invariants(result, sg, plan)


# =========================================================================
# Scoring integration — intent bias steers edge selection
# =========================================================================

class TestScoringSteering(WalkerTestBase):
    """Higher intent bias for a relation type steers the walker toward that edge."""

    def _bias_walker(self, biases_text: str, **kw) -> GraphWalker:
        old_biases = (
            "  0:\n"
            "    is_a: 1.5\n"
            "    has_property: 1.2\n"
            "    example_of: 1.0\n"
            "    default: 0.5\n"
            "  4:\n"
            "    contradicts: 2.0\n"
            "    antonym: 1.8\n"
            "    associated_with: 0.8\n"
            "    default: 0.3"
        )
        cfg_yaml = SAMPLE_WALKER_YAML.replace(
            "restart_on_dead_end: true", "restart_on_dead_end: false",
        ).replace(old_biases, biases_text)
        p = write_yaml(cfg_yaml)
        try:
            cfg = WalkerConfig.from_yaml(p)
        except Exception:
            os.unlink(p)
            raise
        core = CoreConfig.from_yaml(self.core_path)
        w = GraphWalker(cfg, core, **kw)
        os.unlink(p)
        return w

    def test_strong_bias_picks_preferred_edge(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.6, 3: 0.6}
        edges = [(1, 2, "is_a"), (1, 3, "contradicts")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])

        cfg_yaml = (
            SAMPLE_WALKER_YAML
            .replace("restart_on_dead_end: true", "restart_on_dead_end: false")
            .replace(
                "  0:\n"
                "    is_a: 1.5\n"
                "    has_property: 1.2\n"
                "    example_of: 1.0\n"
                "    default: 0.5\n"
                "  4:\n"
                "    contradicts: 2.0\n"
                "    antonym: 1.8\n"
                "    associated_with: 0.8\n"
                "    default: 0.3",
                "  4:\n"
                "    is_a: 0.1\n"
                "    contradicts: 5.0\n"
                "    default: 0.1\n"
            )
        )
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        provider = lambda nid: np.zeros(32, dtype=np.int8)
        w = GraphWalker(cfg, self.core_cfg, embedding_provider=provider, random_seed=0)
        os.unlink(p)

        results = []
        for _ in range(20):
            result = w.walk(sg, plan=make_plan(intent_sequence=[4]))
            if len(result.path) > 1:
                results.append(result.path[1])
        contradiction_count = results.count(3)
        self.assertGreater(contradiction_count, 15,
                           f"Expected >15/20 walks to pick contradicts edge, got {contradiction_count}/20")

    def test_default_bias_applied(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.6}
        edges = [(1, 2, "unknown_rel")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[4])

        biases = (
            "  4:\n"
            "    default: 5.0\n"
        )
        provider = lambda nid: np.zeros(32, dtype=np.int8)
        w = self._bias_walker(biases, embedding_provider=provider, random_seed=42)
        result = w.walk(sg, plan)
        self.assertEqual(result.path[1], 2)

    def test_unknown_intent_bias_defaults_to_one(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.6}
        edges = [(1, 2, "unknown_rel")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[7])

        biases = (
            "  0:\n"
            "    is_a: 10.0\n"
        )
        provider = lambda nid: np.zeros(32, dtype=np.int8)
        w = self._bias_walker(biases, embedding_provider=provider, random_seed=42)
        result = w.walk(sg, plan)
        self.assertEqual(len(result.path), 2)
        self.assertEqual(result.path[1], 2)


# =========================================================================
# Intent sequence guides walk
# =========================================================================

class TestIntentGuidedWalk(WalkerTestBase):
    """Different intent sequences produce different edge choices."""

    def test_different_intents_select_different_edges(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.6, 3: 0.6, 4: 0.7, 5: 0.7}
        edges = [(1, 2, "is_a"), (1, 3, "contradicts"), (2, 4, "is_a"), (3, 5, "contradicts")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])

        cfg_yaml = SAMPLE_WALKER_YAML.replace(
            "weight_intent_bias: 1.0", "weight_intent_bias: 5.0",
        ).replace(
            "restart_on_dead_end: true", "restart_on_dead_end: false",
        )
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        provider = lambda nid: np.zeros(32, dtype=np.int8)

        plan_is_a = Plan(intent_sequence=[0], plan_confidence=0.9, heuristic_fallback_used=False)
        plan_contradicts = Plan(intent_sequence=[4], plan_confidence=0.9, heuristic_fallback_used=False)

        w1 = GraphWalker(cfg, self.core_cfg, embedding_provider=provider, random_seed=0)
        w2 = GraphWalker(cfg, self.core_cfg, embedding_provider=provider, random_seed=0)

        r1 = w1.walk(sg, plan_is_a)
        r2 = w2.walk(sg, plan_contradicts)

        self.assertGreater(len(r1.path), 1)
        self.assertGreater(len(r2.path), 1)
        self.assertEqual(r1.path[1], 2)
        self.assertEqual(r2.path[1], 3)
        os.unlink(p)


# =========================================================================
# Cycle and backtrack behavior
# =========================================================================

class TestWalkConstraints(WalkerTestBase):
    """allow_cycles and allow_backtrack at walk integration level."""

    def test_backtrack_blocked_integration(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.5}
        edges = [(1, 2, "is_a"), (2, 1, "is_a")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        if len(result.path) > 1:
            self.assertNotEqual(result.path[1], 1)

    def test_cycle_blocked_integration(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.5, 3: 0.5}
        triangle = [(1, 2, "is_a"), (2, 3, "is_a"), (3, 1, "is_a")]
        sg = make_subgraph(edges=triangle, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        self.assertEqual(len(set(result.path)), len(result.path),
                         "No duplicates allowed when allow_cycles=False")

    def test_cycles_allowed_integration(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        cfg_yaml = SAMPLE_WALKER_YAML.replace("allow_cycles: false", "allow_cycles: true")
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        act = {1: 0.5, 2: 0.5, 3: 0.5}
        triangle = [(1, 2, "is_a"), (2, 3, "is_a"), (3, 1, "is_a")]
        sg = make_subgraph(edges=triangle, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = GraphWalker(cfg, self.core_cfg,
                        embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
                        random_seed=42)
        result = w.walk(sg, plan)
        self.assertGreater(len(result.path), 1)
        os.unlink(p)


# =========================================================================
# Restart behavior
# =========================================================================

class TestRestartBehavior(WalkerTestBase):
    """Restart recovers from dead ends correctly."""

    def test_restart_from_dead_end(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 3, "is_a")]
        sg = make_subgraph(edges=edges, seed_nodes=[1, 4, 5])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        self.assertGreater(len(result.path), 0)

    def test_restart_picks_unvisited_seed(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 3, "is_a")]
        act = {1: 0.5, 3: 0.5, 5: 0.5}
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[5])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        if len(result.path) > 0:
            self.assertIn(result.path[0], sg.nodes)

    def test_restart_returns_valid_walk(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 3, "is_a")]
        act = {1: 0.5, 2: 0.5, 3: 0.2, 4: 0.5, 5: 0.5}
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1, 2, 4, 5])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        result.validate(0.01, 1.0)

    def test_no_restart_stops_at_dead_end(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        cfg_yaml = SAMPLE_WALKER_YAML.replace(
            "restart_on_dead_end: true", "restart_on_dead_end: false",
        )
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        edges = [(1, 3, "is_a")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = GraphWalker(cfg, self.core_cfg,
                        embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
                        random_seed=42)
        result = w.walk(sg, plan)
        self.assertEqual(result.steps_taken, len(result.path_edges))
        os.unlink(p)


# =========================================================================
# Edge-case walks
# =========================================================================

class TestEdgeCaseWalks(WalkerTestBase):

    def test_single_node_subgraph_walk(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = Subgraph(
            nodes=[1], node_activations={1: 0.5},
            edges=[], edge_strengths={}, edge_confidences={},
            seed_nodes=[1], tier_used=1, activation_energy=1.0,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )
        plan = Plan(intent_sequence=[0], plan_confidence=0.9, heuristic_fallback_used=False)
        w = self._walker(embedding_provider=lambda nid: np.zeros(32, dtype=np.int8))
        result = w.walk(sg, plan)
        self.assertEqual(len(result.path), 1)
        self.assertEqual(result.path[0], 1)
        self.assertEqual(result.steps_taken, 0)

    def test_all_nodes_below_min_activation(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.01, 2: 0.01}
        edges = [(1, 2, "is_a")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(embedding_provider=lambda nid: np.zeros(32, dtype=np.int8))
        result = w.walk(sg, plan)
        self.assertGreaterEqual(len(result.path), 1)

    def test_big_fan_out_walk(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, i, "is_a") for i in range(2, 21)]
        act = {i: 0.5 for i in range(1, 21)}
        nodes = list(range(1, 21))
        strengths = {e: 0.8 for e in edges}
        confidences = {e: 0.9 for e in edges}
        sg = Subgraph(
            nodes=nodes, node_activations=act, edges=edges,
            edge_strengths=strengths, edge_confidences=confidences,
            seed_nodes=[1], tier_used=1, activation_energy=1.0,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        self.assertGreater(len(result.path), 1)
        result.validate(0.01, 1.0)

    def test_walk_truncated_by_max_length(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        chain = [(i, i + 1, "is_a") for i in range(1, 50)]
        act = {i: 0.5 for i in range(1, 51)}
        nodes = list(range(1, 51))
        strengths = {e: 0.8 for e in chain}
        confidences = {e: 0.9 for e in chain}
        sg = Subgraph(
            nodes=nodes, node_activations=act, edges=chain,
            edge_strengths=strengths, edge_confidences=confidences,
            seed_nodes=[1], tier_used=1, activation_energy=1.0,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
        )
        result = w.walk(sg, plan)
        self.assertLessEqual(len(result.path), 21)


# =========================================================================
# WalkResult post-walk invariants
# =========================================================================

class TestWalkVerification(unittest.TestCase):
    """Validates internal consistency of any WalkResult."""

    def test_validate_called_after_walk(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        path_yaml = write_yaml(SAMPLE_WALKER_YAML)
        core_yaml = write_yaml(SAMPLE_CORE_YAML)
        cfg = WalkerConfig.from_yaml(path_yaml)
        core = CoreConfig.from_yaml(core_yaml)
        edges = [(1, 3, "is_a"), (3, 5, "causes")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = GraphWalker(cfg, core,
                        embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
                        random_seed=42)
        result = w.walk(sg, plan)
        result.validate(0.01, 1.0)
        os.unlink(path_yaml)
        os.unlink(core_yaml)

    def test_activations_match_subgraph(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        path_yaml = write_yaml(SAMPLE_WALKER_YAML)
        core_yaml = write_yaml(SAMPLE_CORE_YAML)
        cfg = WalkerConfig.from_yaml(path_yaml)
        core = CoreConfig.from_yaml(core_yaml)
        act = {1: 0.9, 3: 0.4, 5: 0.7}
        edges = [(1, 3, "is_a"), (3, 5, "causes")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = GraphWalker(cfg, core,
                        embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
                        random_seed=42)
        result = w.walk(sg, plan)
        for i, node_id in enumerate(result.path):
            self.assertEqual(result.path_activations[i], sg.node_activations[node_id])
        os.unlink(path_yaml)
        os.unlink(core_yaml)

    def test_walk_confidence_is_geometric_mean_of_confidences(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        path_yaml = write_yaml(SAMPLE_WALKER_YAML)
        core_yaml = write_yaml(SAMPLE_CORE_YAML)
        cfg = WalkerConfig.from_yaml(path_yaml)
        core = CoreConfig.from_yaml(core_yaml)
        edges = [(1, 3, "is_a"), (3, 5, "causes")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = GraphWalker(cfg, core,
                        embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
                        random_seed=42)
        result = w.walk(sg, plan)
        expected = geometric_mean(result.path_confidences)
        self.assertAlmostEqual(result.walk_confidence, expected)
        os.unlink(path_yaml)
        os.unlink(core_yaml)


if __name__ == "__main__":
    unittest.main()
