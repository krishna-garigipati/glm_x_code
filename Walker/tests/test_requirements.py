"""Requirement-gap tests covering uncovered edge cases across all modules."""

from __future__ import annotations

import os
import threading
import time
import tempfile
import unittest
from typing import Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from Walker.config import CoreConfig, WalkerConfig, load_yaml
from Walker.models import Plan, Subgraph, WalkResult
from Walker.intent_bias import IntentBiasTable
from Walker.path_scorer import PathScorer, ScoredCandidate
from Walker.eligibility import compute_eligibility_trace
from Walker.exceptions import EmbeddingLookupError, ValidationError
from Walker.graph_walker import GraphWalker
from Walker.utils import softmax, geometric_mean


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
  save_all_paths: false
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
    edges: Optional[List] = None,
    node_activations: Optional[Dict[int, float]] = None,
    seed_nodes: Optional[List[int]] = None,
    activation_energy: float = 1.0,
    tier_used: int = 1,
    node_embeddings: bool = False,
) -> Subgraph:
    nodes = [1, 2, 3, 4, 5]
    act = node_activations or {n: 0.5 for n in nodes}
    seed = [1, 2] if seed_nodes is None else seed_nodes
    edges_list = edges or [(1, 3, "is_a"), (1, 4, "has_property"), (3, 5, "causes")]
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
        tier_used=tier_used,
        activation_energy=activation_energy,
        query_embedding=query_emb,
        node_embeddings=nemb,
        timestamp=time.time(),
    )


def make_plan(
    intent_sequence: Optional[List[int]] = None,
    plan_confidence: float = 0.9,
    heuristic_fallback: bool = False,
    intent_names: Optional[List[str]] = None,
) -> Plan:
    return Plan(
        intent_sequence=[0, 1, 4] if intent_sequence is None else intent_sequence,
        plan_confidence=plan_confidence,
        heuristic_fallback_used=heuristic_fallback,
        intent_names=intent_names,
    )


# =========================================================================
# Config - missing coverage
# =========================================================================

class TestConfigExtra(unittest.TestCase):
    def test_load_yaml_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_yaml(tempfile.mktemp(suffix=".yaml"))

    def test_walker_config_missing_field_raises(self) -> None:
        minimal = "walk:\n  max_steps: 10\n"
        p = write_yaml(minimal)
        with self.assertRaises(KeyError):
            WalkerConfig.from_yaml(p)
        os.unlink(p)

    def test_core_config_missing_field_raises(self) -> None:
        minimal = "activation:\n  min: 0.0\n"
        p = write_yaml(minimal)
        with self.assertRaises(KeyError):
            CoreConfig.from_yaml(p)
        os.unlink(p)


# =========================================================================
# Subgraph - additional edge case coverage
# =========================================================================

class TestSubgraphExtra(unittest.TestCase):
    def test_edge_strengths_keys_mismatch(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 3, "is_a")]
        strengths = {(1, 3, "is_a"): 0.8, (99, 100, "extra"): 0.5}
        confidences = {(1, 3, "is_a"): 0.9}
        sg = Subgraph(
            nodes=[1, 3], node_activations={1: 0.5, 3: 0.5},
            edges=edges, edge_strengths=strengths,
            edge_confidences=confidences, seed_nodes=[1],
            tier_used=1, activation_energy=1.0,
            query_embedding=np.zeros(384), timestamp=0.0,
        )
        with self.assertRaises(ValueError):
            sg.validate(0.01, 1.0)

    def test_missing_activation_for_node(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = Subgraph(
            nodes=[1, 99], node_activations={1: 0.5},
            edges=[], edge_strengths={},
            edge_confidences={}, seed_nodes=[1],
            tier_used=1, activation_energy=1.0,
            query_embedding=np.zeros(384), timestamp=0.0,
        )
        with self.assertRaises(ValueError):
            sg.validate(0.01, 1.0)


# =========================================================================
# WalkResult.build - edge cases
# =========================================================================

class TestWalkResultBuild(unittest.TestCase):
    def test_build_basic(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        emb = [np.zeros(32) for _ in range(3)]
        wr = WalkResult.build(
            path=[1, 2, 3],
            path_edges=["is_a", "causes"],
            path_activations=[0.5, 0.6, 0.7],
            path_confidences=[0.9, 0.8],
            path_embeddings=emb,
            plan=make_plan(),
        )
        self.assertEqual(wr.steps_taken, 2)
        self.assertEqual(wr.final_activation, 0.7)
        self.assertEqual(wr.walk_confidence, 1.0)
        self.assertEqual(list(wr.intent_sequence_used), [0, 1, 4])

    def test_build_empty_activations(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        emb = [np.zeros(32) for _ in range(2)]
        wr = WalkResult.build(
            path=[1, 2],
            path_edges=["is_a"],
            path_activations=[],
            path_confidences=[],
            path_embeddings=emb,
            plan=make_plan(),
        )
        self.assertEqual(wr.final_activation, 0.0)

    def test_build_single_node_path(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        emb = [np.zeros(32)]
        wr = WalkResult.build(
            path=[1],
            path_edges=[],
            path_activations=[0.5],
            path_confidences=[],
            path_embeddings=emb,
            plan=make_plan(),
        )
        self.assertEqual(wr.steps_taken, 0)
        self.assertEqual(wr.final_activation, 0.5)


# =========================================================================
# WalkResult.to_tuples - edge cases
# =========================================================================

class TestWalkResultToTuplesExtra(unittest.TestCase):
    def test_single_node(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        emb = [np.zeros(32)]
        wr = WalkResult(
            path=[5], path_edges=[], path_activations=[0.9],
            path_confidences=[], path_embeddings=emb,
            walk_confidence=1.0, final_activation=0.9,
            steps_taken=0, plan_followed=make_plan(),
            timestamp=0.0, intent_sequence_used=[0],
        )
        result = wr.to_tuples()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (5, "", 0.9, 0.0))


# =========================================================================
# Utils - additional edge cases
# =========================================================================

class TestUtilsExtra(unittest.TestCase):
    def test_softmax_uniform(self) -> None:
        result = softmax([2.0, 2.0, 2.0], 0.1)
        self.assertAlmostEqual(sum(result), 1.0)
        for v in result:
            self.assertAlmostEqual(v, 1.0 / 3)

    def test_softmax_single(self) -> None:
        result = softmax([5.0], 0.1)
        self.assertAlmostEqual(result[0], 1.0)

    def test_geometric_mean_single(self) -> None:
        self.assertEqual(geometric_mean([0.85]), 0.85)

    def test_geometric_mean_all_ones(self) -> None:
        self.assertEqual(geometric_mean([1.0, 1.0, 1.0]), 1.0)


# =========================================================================
# PathScorer - additional edge cases
# =========================================================================

class TestPathScorerExtra(unittest.TestCase):
    def test_rank_normalize_empty(self) -> None:
        rscorer = PathScorer(
            weight_strength=1.0, weight_confidence=1.0,
            weight_target_activation=1.0, weight_intent_bias=1.0,
            normalization="rank", softmax_temperature=0.1,
        )
        self.assertEqual(rscorer.normalize([]), [])

    def test_score_negative_values(self) -> None:
        scorer = PathScorer(
            weight_strength=1.0, weight_confidence=1.0,
            weight_target_activation=-1.0, weight_intent_bias=1.0,
            normalization="softmax", softmax_temperature=0.1,
        )
        c = ScoredCandidate(
            node_id=5, edge_type="is_a",
            strength=0.8, confidence=0.9,
            target_activation=0.5, intent_bias=1.5,
            raw_score=0.0,
        )
        score = scorer.score(c)
        self.assertAlmostEqual(score, 1.0 * 0.8 * 1.0 * 0.9 * (-1.0) * 0.5 * 1.0 * 1.5)


# =========================================================================
# IntentBiasTable - additional edge cases
# =========================================================================

class TestIntentBiasTableExtra(unittest.TestCase):
    def test_update_bias_thread_safety_new_intent(self) -> None:
        table = IntentBiasTable({})
        errors: List[Exception] = []
        def mutate() -> None:
            try:
                for i in range(100):
                    table.update_bias(i % 10, "rel", 0.5)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=mutate) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_snapshot_empty(self) -> None:
        table = IntentBiasTable({})
        self.assertEqual(table.snapshot(), {})


# =========================================================================
# EligibilityTrace - additional edge cases
# =========================================================================

class TestEligibilityTraceExtra(unittest.TestCase):
    def test_zero_gamma(self) -> None:
        edges = [(1, 3, "is_a")]
        strengths = {e: 0.8 for e in edges}
        confidences = {e: 0.9 for e in edges}
        trace = compute_eligibility_trace(
            path=[1, 3], path_edges=["is_a"],
            path_activations=[0.5],
            edge_strengths=strengths, edge_confidences=confidences,
            gamma=0.0,
        )
        self.assertAlmostEqual(trace["1:3:is_a"], 0.5 * (0.8 * 0.9))

    def test_gamma_one(self) -> None:
        edges = [(1, 3, "is_a"), (3, 5, "causes")]
        strengths = {e: 0.8 for e in edges}
        confidences = {e: 0.9 for e in edges}
        trace = compute_eligibility_trace(
            path=[1, 3, 5], path_edges=["is_a", "causes"],
            path_activations=[0.5, 0.6],
            edge_strengths=strengths, edge_confidences=confidences,
            gamma=1.0,
        )
        self.assertAlmostEqual(trace["1:3:is_a"], 1.0 * 0.5 * (0.8 * 0.9 * 1.0))
        self.assertAlmostEqual(trace["3:5:causes"], 1.0 * 0.6 * (0.8 * 0.9 * 1.0))


# =========================================================================
# GraphWalker construction - additional edge cases
# =========================================================================

class TestGraphWalkerConstructionExtra(unittest.TestCase):
    def setUp(self) -> None:
        self.walker_path = write_yaml(SAMPLE_WALKER_YAML)
        self.core_path = write_yaml(SAMPLE_CORE_YAML)
        self.walker_cfg = WalkerConfig.from_yaml(self.walker_path)
        self.core_cfg = CoreConfig.from_yaml(self.core_path)

    def tearDown(self) -> None:
        os.unlink(self.walker_path)
        os.unlink(self.core_path)

    def test_scoring_formula_multiple_terms_missing(self) -> None:
        bad = SAMPLE_WALKER_YAML.replace(
            'formula: "strength * confidence * target_activation * intent_bias(edge_type, current_intent)"',
            'formula: "strength * confidence"',
        )
        p = write_yaml(bad)
        with self.assertRaises(ValidationError):
            GraphWalker(WalkerConfig.from_yaml(p), self.core_cfg)
        os.unlink(p)


# =========================================================================
# GraphWalker - internal method coverage
# =========================================================================

class TestGraphWalkerInternals(unittest.TestCase):
    def setUp(self) -> None:
        self.walker_path = write_yaml(SAMPLE_WALKER_YAML)
        self.core_path = write_yaml(SAMPLE_CORE_YAML)
        self.walker_cfg = WalkerConfig.from_yaml(self.walker_path)
        self.core_cfg = CoreConfig.from_yaml(self.core_path)

    def tearDown(self) -> None:
        os.unlink(self.walker_path)
        os.unlink(self.core_path)

    def _walker(self, **kw) -> GraphWalker:
        return GraphWalker(self.walker_cfg, self.core_cfg, **kw)

    def test_apply_temperature_zero_raises(self) -> None:
        w = self._walker()
        w._temperature = 0.0
        with self.assertRaises(ValidationError):
            w._apply_temperature([1.0])

    def test_apply_temperature_empty(self) -> None:
        w = self._walker()
        w._temperature = 0.1
        result = w._apply_temperature([])
        self.assertEqual(result, [])

    def test_select_start_node_no_seeds(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph()
        w = self._walker()
        # Override to make seed_nodes empty
        sg2 = Subgraph(
            nodes=sg.nodes, node_activations=sg.node_activations,
            edges=sg.edges, edge_strengths=sg.edge_strengths,
            edge_confidences=sg.edge_confidences, seed_nodes=[],
            tier_used=sg.tier_used, activation_energy=sg.activation_energy,
            query_embedding=sg.query_embedding,
            node_embeddings=sg.node_embeddings, timestamp=sg.timestamp,
        )
        start = w._select_start_node(sg2, restart=False)
        self.assertIsNotNone(start)
        self.assertIn(start, sg.nodes)

    def test_select_start_node_all_visited(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph(seed_nodes=[1, 2])
        w = self._walker()
        start = w._select_start_node(sg, restart=False, visited={1, 2})
        self.assertIsNotNone(start)
        self.assertIn(start, sg.nodes)

    def test_select_start_node_all_visited_no_candidates(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph(seed_nodes=[1, 2])
        w = self._walker()
        start = w._select_start_node(sg, restart=False, visited=set(sg.nodes))
        self.assertIsNone(start)

    def test_collect_candidates_backtrack_prevented(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        # Path: 1 -> 2 -> 3, current=3, prev=2, edge from 3->2 exists but should be blocked
        edges = [(1, 2, "is_a"), (2, 3, "is_a"), (3, 2, "is_a")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        w = self._walker()
        candidates = w._collect_candidates(3, sg, 0, visited=[1, 2, 3])
        for c in candidates:
            self.assertNotEqual(c.node_id, 2, "backtrack should be blocked")

    def test_collect_candidates_cycle_prevented(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 2, "is_a"), (2, 1, "is_a"), (2, 3, "is_a")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        w = self._walker()
        candidates = w._collect_candidates(2, sg, 0, visited=[1, 2])
        for c in candidates:
            self.assertNotEqual(c.node_id, 1, "cycle should be blocked")

    def test_collect_candidates_min_activation_filter(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.01, 3: 0.5}
        edges = [(1, 2, "is_a"), (1, 3, "causes")]
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        w = self._walker()
        candidates = w._collect_candidates(1, sg, 0)
        for c in candidates:
            self.assertGreaterEqual(c.target_activation, 0.05)

    def test_collect_candidates_allow_cycles_and_backtrack(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        cfg_yaml = SAMPLE_WALKER_YAML.replace("allow_cycles: false", "allow_cycles: true")
        cfg_yaml = cfg_yaml.replace("allow_backtrack: false", "allow_backtrack: true")
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        w = GraphWalker(cfg, self.core_cfg, embedding_provider=lambda nid: np.zeros(32))
        edges = [(1, 2, "is_a"), (2, 1, "is_a")]
        act = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5}
        sg = make_subgraph(edges=edges, node_activations=act, seed_nodes=[1])
        candidates = w._collect_candidates(2, sg, 0, visited=[1, 2])
        node_ids = [c.node_id for c in candidates]
        self.assertIn(1, node_ids, "cycle to 1 should be allowed")
        os.unlink(p)


# =========================================================================
# GraphWalker walk - additional edge cases
# =========================================================================

class TestGraphWalkerWalkExtra(unittest.TestCase):
    def setUp(self) -> None:
        self.walker_path = write_yaml(SAMPLE_WALKER_YAML)
        self.core_path = write_yaml(SAMPLE_CORE_YAML)
        self.walker_cfg = WalkerConfig.from_yaml(self.walker_path)
        self.core_cfg = CoreConfig.from_yaml(self.core_path)

    def tearDown(self) -> None:
        os.unlink(self.walker_path)
        os.unlink(self.core_path)

    def _walker(self, **kw) -> GraphWalker:
        return GraphWalker(self.walker_cfg, self.core_cfg, **kw)

    def test_walk_multiple_intents(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 2, "is_a"), (2, 3, "causes"), (3, 4, "contradicts"), (4, 5, "supports")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0, 4, 6])
        w = self._walker(embedding_provider=lambda nid: np.zeros(32), random_seed=42)
        result = w.walk(sg, plan)
        self.assertGreater(len(result.path), 1)
        result.validate(0.01, 1.0)

    def test_walk_no_restart_at_dead_end(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        cfg_yaml = SAMPLE_WALKER_YAML.replace("restart_on_dead_end: true", "restart_on_dead_end: false")
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        # Single edge, after walking it there's a dead end
        edges = [(1, 3, "is_a")]
        sg = make_subgraph(edges=edges, seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = GraphWalker(cfg, self.core_cfg, embedding_provider=lambda nid: np.zeros(32))
        result = w.walk(sg, plan)
        # Should stop at dead end instead of restarting
        self.assertGreaterEqual(len(result.path), 1)
        os.unlink(p)

    def test_walk_max_length_bound(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        # Chain of 30 edges, but max_length=20
        nodes = list(range(1, 32))
        edges = [(i, i + 1, "is_a") for i in range(1, 31)]
        act = {n: 0.5 for n in nodes}
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
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        result = w.walk(sg, plan)
        self.assertLessEqual(len(result.path), 21)

    def test_walk_empty_subgraph_fails(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        empty_sg = Subgraph(
            nodes=[], node_activations={}, edges=[],
            edge_strengths={}, edge_confidences={},
            seed_nodes=[], tier_used=1, activation_energy=0.0,
            query_embedding=np.zeros(384), timestamp=0.0,
        )
        plan = make_plan()
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        with self.assertRaises(ValidationError):
            w.walk(empty_sg, plan)

    def test_walk_with_timestamps_recorded(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph()
        plan = make_plan()
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        w.walk(sg, plan)
        stamps = w.get_last_step_timestamps()
        self.assertGreater(len(stamps), 0)

    def test_get_saved_paths_empty(self) -> None:
        w = self._walker()
        self.assertEqual(w.get_saved_paths(), [])

    def test_get_last_step_timestamps_empty(self) -> None:
        w = self._walker()
        self.assertEqual(w.get_last_step_timestamps(), [])

    def test_get_walk_confidence_zero_values(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        emb = [np.zeros(32) for _ in range(3)]
        wr = WalkResult(
            path=[1, 2, 3], path_edges=["is_a", "causes"],
            path_activations=[0.5, 0.6, 0.7],
            path_confidences=[0.0, 0.8],
            path_embeddings=emb, walk_confidence=0.85,
            final_activation=0.7, steps_taken=2,
            plan_followed=make_plan(), timestamp=0.0,
            intent_sequence_used=[0],
        )
        w = self._walker()
        self.assertEqual(w.get_walk_confidence(wr), 0.0)


if __name__ == "__main__":
    unittest.main()
