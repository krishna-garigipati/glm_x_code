"""Comprehensive unit tests for the Walker component."""

from __future__ import annotations

import dataclasses
import os
import threading
import time
import unittest
import tempfile
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from walker.config import CoreConfig, WalkerConfig, load_yaml
from walker.models import Plan, Subgraph, WalkResult
from walker.intent_bias import IntentBiasTable
from walker.path_scorer import PathScorer, ScoredCandidate
from walker.eligibility import compute_eligibility_trace
from walker.exceptions import EmbeddingLookupError, ValidationError
from walker.graph_walker import GraphWalker
from walker.utils import softmax, geometric_mean

# =========================================================================
# YAML fixtures matching config_walker.yaml exactly
# =========================================================================

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


# =========================================================================
# Helpers
# =========================================================================

def write_yaml(content: str) -> str:
    path = tempfile.mktemp(suffix=".yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def make_subgraph(
    edges: Optional[List[Tuple[int, int, str]]] = None,
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
    query_emb = None
    if HAS_NUMPY:
        query_emb = np.zeros(384, dtype=np.float32)
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
# Config Loading
# =========================================================================

class TestConfigLoading(unittest.TestCase):
    """Config loading covers: walk (8 fields), scoring (7 fields),
       eligibility (3 fields), path (5 fields), debug (3 fields),
       intent_biases (2 intents x N relations)."""

    def test_load_yaml(self) -> None:
        path = write_yaml("key: value\nnested:\n  inner: 42\n")
        data = load_yaml(path)
        self.assertEqual(data["key"], "value")
        self.assertEqual(data["nested"]["inner"], 42)
        os.unlink(path)

    def test_load_yaml_empty(self) -> None:
        path = write_yaml("")
        data = load_yaml(path)
        self.assertEqual(data, {})

    def test_load_yaml_none(self) -> None:
        path = write_yaml("")
        data = load_yaml(path)
        self.assertEqual(data, {})

    def test_walker_config_all_sections(self) -> None:
        path = write_yaml(SAMPLE_WALKER_YAML)
        cfg = WalkerConfig.from_yaml(path)
        os.unlink(path)

        self.assertEqual(cfg.walk.max_steps, 20)
        self.assertEqual(cfg.walk.min_activation, 0.05)
        self.assertEqual(cfg.walk.temperature, 0.1)
        self.assertEqual(cfg.walk.temperature_range, (0.05, 0.5))
        self.assertFalse(cfg.walk.allow_cycles)
        self.assertFalse(cfg.walk.allow_backtrack)
        self.assertTrue(cfg.walk.restart_on_dead_end)
        self.assertEqual(cfg.walk.restart_penalty, 0.5)

        self.assertIn(0, cfg.intent_biases)
        self.assertIn(4, cfg.intent_biases)
        self.assertEqual(cfg.intent_biases[0]["is_a"], 1.5)
        self.assertEqual(cfg.intent_biases[0]["default"], 0.5)
        self.assertEqual(cfg.intent_biases[4]["contradicts"], 2.0)
        self.assertEqual(cfg.intent_biases[4]["default"], 0.3)

        self.assertEqual(cfg.scoring.formula,
                         "strength * confidence * target_activation * intent_bias(edge_type, current_intent)")
        self.assertEqual(cfg.scoring.weight_strength, 1.0)
        self.assertEqual(cfg.scoring.weight_confidence, 1.0)
        self.assertEqual(cfg.scoring.weight_target_activation, 1.0)
        self.assertEqual(cfg.scoring.weight_intent_bias, 1.0)
        self.assertEqual(cfg.scoring.normalization, "softmax")
        self.assertEqual(cfg.scoring.softmax_temperature, 0.1)

        self.assertEqual(cfg.eligibility.gamma, 0.9)
        self.assertIn("gamma^t", cfg.eligibility.formula)
        self.assertEqual(cfg.eligibility.trace_key_format, "{source}:{target}:{relation}")

        self.assertEqual(cfg.path.max_length, 20)
        self.assertTrue(cfg.path.record_activations)
        self.assertTrue(cfg.path.record_timestamps)
        self.assertTrue(cfg.path.record_edge_confidence)
        self.assertEqual(cfg.path.output_format,
                         "List[Tuple[node_id, edge_type, activation, confidence]]")

        self.assertFalse(cfg.debug.log_decision_scores)
        self.assertTrue(cfg.debug.log_path_taken)
        self.assertFalse(cfg.debug.save_all_paths)

    def test_core_config(self) -> None:
        path = write_yaml(SAMPLE_CORE_YAML)
        cfg = CoreConfig.from_yaml(path)
        os.unlink(path)

        self.assertEqual(cfg.activation.min, 0.01)
        self.assertEqual(cfg.activation.max, 1.0)
        self.assertEqual(cfg.walker.default_temperature, 0.1)
        self.assertEqual(cfg.walker.softmax_temperature_range, (0.05, 0.5))
        self.assertEqual(cfg.walker.max_steps, 20)
        self.assertEqual(cfg.walker.min_activation_to_continue, 0.05)
        self.assertEqual(cfg.relations[0], "is_a")
        self.assertEqual(len(cfg.relations), 16)


# =========================================================================
# Subgraph validation
# =========================================================================

class TestSubgraphValidation(unittest.TestCase):
    def test_valid(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        make_subgraph().validate(0.01, 1.0)

    def test_empty_nodes(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = Subgraph(
            nodes=[], node_activations={}, edges=[], edge_strengths={},
            edge_confidences={}, seed_nodes=[1], tier_used=1,
            activation_energy=0.0, query_embedding=np.zeros(384),
            timestamp=0.0,
        )
        with self.assertRaises(ValueError):
            sg.validate(0.01, 1.0)

    def test_empty_seed_nodes(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        with self.assertRaises(ValueError):
            make_subgraph(seed_nodes=[]).validate(0.01, 1.0)

    def test_tier_invalid(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        with self.assertRaises(ValueError):
            make_subgraph(tier_used=3).validate(0.01, 1.0)

    def test_activation_out_of_range(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {n: 0.5 for n in [1, 2, 3, 4, 5]}
        act[3] = 5.0
        with self.assertRaises(ValueError):
            make_subgraph(node_activations=act).validate(0.01, 1.0)

    def test_edge_nodes_not_in_graph(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        with self.assertRaises(ValueError):
            make_subgraph(edges=[(99, 3, "is_a")]).validate(0.01, 1.0)

    def test_node_embeddings_unknown_key(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph()
        with self.assertRaises(ValueError):
            Subgraph(
                nodes=sg.nodes, node_activations=sg.node_activations,
                edges=sg.edges, edge_strengths=sg.edge_strengths,
                edge_confidences=sg.edge_confidences, seed_nodes=sg.seed_nodes,
                tier_used=sg.tier_used, activation_energy=sg.activation_energy,
                query_embedding=sg.query_embedding,
                node_embeddings={99: np.zeros(32)},
                timestamp=sg.timestamp,
            ).validate(0.01, 1.0)


# =========================================================================
# Plan validation
# =========================================================================

class TestPlanValidation(unittest.TestCase):
    def test_valid(self) -> None:
        make_plan().validate()

    def test_empty_sequence(self) -> None:
        with self.assertRaises(ValueError):
            make_plan(intent_sequence=[]).validate()

    def test_sequence_too_long(self) -> None:
        with self.assertRaises(ValueError):
            make_plan(intent_sequence=list(range(9))).validate()

    def test_intent_below_zero(self) -> None:
        with self.assertRaises(ValueError):
            make_plan(intent_sequence=[-1]).validate()

    def test_intent_above_15(self) -> None:
        with self.assertRaises(ValueError):
            make_plan(intent_sequence=[16]).validate()

    def test_confidence_too_high(self) -> None:
        with self.assertRaises(ValueError):
            make_plan(plan_confidence=1.5).validate()

    def test_confidence_negative(self) -> None:
        with self.assertRaises(ValueError):
            make_plan(plan_confidence=-0.1).validate()

    def test_intent_names_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            make_plan(intent_sequence=[0, 1], intent_names=["define"]).validate()

    def test_intent_names_correct(self) -> None:
        make_plan(intent_sequence=[0, 1], intent_names=["define", "assert_fact"]).validate()


# =========================================================================
# WalkResult validation
# =========================================================================

class TestWalkResultValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.emb = [np.zeros(32) for _ in range(3)] if HAS_NUMPY else []

    def _make(self, **overrides) -> WalkResult:
        defaults = dict(
            path=[1, 3, 5],
            path_edges=["is_a", "causes"],
            path_activations=[0.5, 0.6, 0.7],
            path_confidences=[0.9, 0.8],
            path_embeddings=self.emb,
            walk_confidence=0.85,
            final_activation=0.7,
            steps_taken=2,
            plan_followed=make_plan(),
            timestamp=100.0,
            intent_sequence_used=[0, 1],
        )
        defaults.update(overrides)
        return WalkResult(**defaults)

    def test_valid(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        self._make().validate(0.01, 1.0)

    def test_empty_path(self) -> None:
        with self.assertRaises(ValueError):
            self._make(path=[]).validate(0.01, 1.0)

    def test_edges_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            self._make(path_edges=["is_a"]).validate(0.01, 1.0)

    def test_consecutive_duplicate(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        with self.assertRaises(ValueError):
            self._make(path=[1, 1, 3]).validate(0.01, 1.0)

    def test_activation_out_of_range(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        with self.assertRaises(ValueError):
            self._make(path_activations=[0.5, 5.0, 0.7]).validate(0.01, 1.0)

    def test_confidence_out_of_range(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        with self.assertRaises(ValueError):
            self._make(path_confidences=[1.5, 0.8]).validate(0.01, 1.0)

    def test_empty_activations_allowed(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        self._make(path_activations=[], path_confidences=[]).validate(0.01, 1.0)

    def test_walk_confidence_out_of_range(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        with self.assertRaises(ValueError):
            self._make(walk_confidence=1.5).validate(0.01, 1.0)

    def test_embeddings_wrong_length(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        with self.assertRaises(ValueError):
            self._make(path_embeddings=[np.zeros(32)]).validate(0.01, 1.0)


# =========================================================================
# WalkResult.to_tuples (output_format)
# =========================================================================

class TestWalkResultToTuples(unittest.TestCase):
    def test_full_recording(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        emb = [np.zeros(32) for _ in range(3)]
        wr = WalkResult(
            path=[10, 20, 30], path_edges=["is_a", "causes"],
            path_activations=[0.5, 0.6, 0.7],
            path_confidences=[0.9, 0.8],
            path_embeddings=emb, walk_confidence=0.85,
            final_activation=0.7, steps_taken=2,
            plan_followed=make_plan(), timestamp=0.0,
            intent_sequence_used=[0],
        )
        result = wr.to_tuples()
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], (10, "", 0.5, 0.0))
        self.assertEqual(result[1], (20, "is_a", 0.6, 0.9))
        self.assertEqual(result[2], (30, "causes", 0.7, 0.8))

    def test_no_recording(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        emb = [np.zeros(32) for _ in range(3)]
        wr = WalkResult(
            path=[10, 20, 30], path_edges=["is_a", "causes"],
            path_activations=[], path_confidences=[],
            path_embeddings=emb, walk_confidence=1.0,
            final_activation=0.0, steps_taken=2,
            plan_followed=make_plan(), timestamp=0.0,
            intent_sequence_used=[0],
        )
        result = wr.to_tuples()
        self.assertEqual(result[0], (10, "", 0.0, 0.0))
        self.assertEqual(result[1], (20, "is_a", 0.0, 0.0))


# =========================================================================
# IntentBiasTable
# =========================================================================

class TestIntentBiasTable(unittest.TestCase):
    def setUp(self) -> None:
        self.biases = {
            0: {"is_a": 1.5, "has_property": 1.2, "default": 0.5},
            4: {"contradicts": 2.0, "antonym": 1.8, "default": 0.3},
        }
        self.table = IntentBiasTable(self.biases)

    def test_known_relation(self) -> None:
        self.assertEqual(self.table.get_bias(0, "is_a"), 1.5)
        self.assertEqual(self.table.get_bias(4, "contradicts"), 2.0)

    def test_fallback_to_default(self) -> None:
        self.assertEqual(self.table.get_bias(0, "unknown_rel"), 0.5)
        self.assertEqual(self.table.get_bias(4, "unknown_rel"), 0.3)

    def test_unknown_intent_returns_one(self) -> None:
        self.assertEqual(self.table.get_bias(99, "anything"), 1.0)

    def test_update_existing(self) -> None:
        self.table.update_bias(0, "is_a", 2.5)
        self.assertEqual(self.table.get_bias(0, "is_a"), 2.5)

    def test_update_new_intent(self) -> None:
        self.table.update_bias(7, "example_of", 2.0)
        self.assertEqual(self.table.get_bias(7, "example_of"), 2.0)
        self.assertEqual(self.table.get_bias(7, "unknown"), 1.0)

    def test_snapshot_isolation(self) -> None:
        snap = self.table.snapshot()
        snap[0]["is_a"] = 99.0
        self.assertEqual(self.table.get_bias(0, "is_a"), 1.5)

    def test_thread_safety(self) -> None:
        errors: List[Exception] = []
        def mutate() -> None:
            try:
                for i in range(100):
                    self.table.update_bias(0, "is_a", 1.0 + i / 100.0)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=mutate) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


# =========================================================================
# PathScorer
# =========================================================================

class TestPathScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = PathScorer(
            weight_strength=1.0, weight_confidence=1.0,
            weight_target_activation=1.0, weight_intent_bias=1.0,
            normalization="softmax", softmax_temperature=0.1,
        )

    def test_score_computation(self) -> None:
        c = ScoredCandidate(
            node_id=5, edge_type="is_a",
            strength=0.8, confidence=0.9,
            target_activation=0.5, intent_bias=1.5,
            raw_score=0.0,
        )
        expected = 1.0 * 0.8 * 1.0 * 0.9 * 1.0 * 0.5 * 1.0 * 1.5
        self.assertAlmostEqual(self.scorer.score(c), expected)

    def test_score_with_weights(self) -> None:
        wscorer = PathScorer(
            weight_strength=0.5, weight_confidence=2.0,
            weight_target_activation=1.5, weight_intent_bias=0.8,
            normalization="softmax", softmax_temperature=0.1,
        )
        c = ScoredCandidate(
            node_id=5, edge_type="is_a",
            strength=0.8, confidence=0.9,
            target_activation=0.5, intent_bias=1.5,
            raw_score=0.0,
        )
        expected = 0.5 * 0.8 * 2.0 * 0.9 * 1.5 * 0.5 * 0.8 * 1.5
        self.assertAlmostEqual(wscorer.score(c), expected)

    def test_normalize_softmax(self) -> None:
        normalized = self.scorer.normalize([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(normalized), 1.0)

    def test_normalize_rank(self) -> None:
        rscorer = PathScorer(
            weight_strength=1.0, weight_confidence=1.0,
            weight_target_activation=1.0, weight_intent_bias=1.0,
            normalization="rank", softmax_temperature=0.1,
        )
        scores = [3.0, 1.0, 2.0]
        normalized = rscorer.normalize(scores)
        self.assertAlmostEqual(sum(normalized), 1.0)
        self.assertGreater(normalized[0], normalized[2])
        self.assertGreater(normalized[2], normalized[1])

    def test_normalize_none(self) -> None:
        nscorer = PathScorer(
            weight_strength=1.0, weight_confidence=1.0,
            weight_target_activation=1.0, weight_intent_bias=1.0,
            normalization="none", softmax_temperature=0.1,
        )
        self.assertEqual(nscorer.normalize([1.5, 2.5]), [1.5, 2.5])

    def test_normalize_empty(self) -> None:
        self.assertEqual(self.scorer.normalize([]), [])

    def test_invalid_normalization(self) -> None:
        bscorer = PathScorer(
            weight_strength=1.0, weight_confidence=1.0,
            weight_target_activation=1.0, weight_intent_bias=1.0,
            normalization="bad", softmax_temperature=0.1,
        )
        with self.assertRaises(ValueError):
            bscorer.normalize([1.0])


# =========================================================================
# Eligibility Trace
# =========================================================================

class TestEligibilityTrace(unittest.TestCase):
    def test_basic(self) -> None:
        edges = [(1, 3, "is_a"), (3, 5, "causes")]
        strengths = {e: 0.8 for e in edges}
        confidences = {e: 0.9 for e in edges}
        trace = compute_eligibility_trace(
            path=[1, 3, 5], path_edges=["is_a", "causes"],
            path_activations=[0.5, 0.6],
            edge_strengths=strengths, edge_confidences=confidences,
            gamma=0.9,
        )
        self.assertIn("1:3:is_a", trace)
        self.assertIn("3:5:causes", trace)
        self.assertAlmostEqual(trace["1:3:is_a"],
                               (0.9 ** 0) * 0.5 * (0.8 * 0.9 * 1.0))
        self.assertAlmostEqual(trace["3:5:causes"],
                               (0.9 ** 1) * 0.6 * (0.8 * 0.9 * 1.0))

    def test_trace_key_format(self) -> None:
        edges = [(1, 3, "is_a")]
        strengths = {e: 0.8 for e in edges}
        confidences = {e: 0.9 for e in edges}
        trace = compute_eligibility_trace(
            path=[1, 3], path_edges=["is_a"],
            path_activations=[0.5],
            edge_strengths=strengths, edge_confidences=confidences,
            gamma=0.9, trace_key_format="{source}-{target}-{relation}",
        )
        self.assertIn("1-3-is_a", trace)
        self.assertNotIn("1:3:is_a", trace)

    def test_empty_activations(self) -> None:
        edges = [(1, 3, "is_a")]
        strengths = {e: 0.8 for e in edges}
        confidences = {e: 0.9 for e in edges}
        trace = compute_eligibility_trace(
            path=[1, 3], path_edges=["is_a"], path_activations=[],
            edge_strengths=strengths, edge_confidences=confidences,
            gamma=0.9,
        )
        self.assertAlmostEqual(trace["1:3:is_a"], 0.0)

    def test_accumulated_same_edge(self) -> None:
        edges = [(1, 3, "is_a"), (3, 1, "is_a")]
        strengths = {e: 0.8 for e in edges}
        confidences = {e: 0.9 for e in edges}
        trace = compute_eligibility_trace(
            path=[1, 3, 1], path_edges=["is_a", "is_a"],
            path_activations=[0.5, 0.7],
            edge_strengths=strengths, edge_confidences=confidences,
            gamma=0.9,
        )
        expected_0 = (0.9 ** 0) * 0.5 * (0.8 * 0.9)
        expected_1 = (0.9 ** 1) * 0.7 * (0.8 * 0.9)
        self.assertIn("1:3:is_a", trace)
        self.assertIn("3:1:is_a", trace)
        self.assertAlmostEqual(trace["1:3:is_a"], expected_0)
        self.assertAlmostEqual(trace["3:1:is_a"], expected_1)

    def test_eligibility_trace_dataclass(self) -> None:
        from walker.eligibility import EligibilityTrace
        et = EligibilityTrace(edge_key=(1, 3, "is_a"), eligibility=0.875,
                              time_contribution=[0.5, 0.375])
        self.assertEqual(et.edge_key, (1, 3, "is_a"))
        self.assertEqual(et.eligibility, 0.875)
        self.assertEqual(et.time_contribution, [0.5, 0.375])


# =========================================================================
# Utils
# =========================================================================

class TestUtils(unittest.TestCase):
    def test_softmax_normalizes(self) -> None:
        result = softmax([1.0, 2.0, 3.0], 0.1)
        self.assertAlmostEqual(sum(result), 1.0)

    def test_softmax_empty(self) -> None:
        self.assertEqual(softmax([], 0.1), [])

    def test_softmax_zero_temperature_raises(self) -> None:
        with self.assertRaises(ValueError):
            softmax([1.0], 0.0)

    def test_geometric_mean(self) -> None:
        self.assertAlmostEqual(geometric_mean([0.8, 0.9, 0.7]),
                               (0.8 * 0.9 * 0.7) ** (1.0 / 3))

    def test_geometric_mean_empty(self) -> None:
        self.assertEqual(geometric_mean([]), 1.0)

    def test_geometric_mean_zero(self) -> None:
        self.assertEqual(geometric_mean([0.8, 0.0, 0.9]), 0.0)


# =========================================================================
# GraphWalker - construction
# =========================================================================

class TestGraphWalkerConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.walker_path = write_yaml(SAMPLE_WALKER_YAML)
        self.core_path = write_yaml(SAMPLE_CORE_YAML)
        self.walker_cfg = WalkerConfig.from_yaml(self.walker_path)
        self.core_cfg = CoreConfig.from_yaml(self.core_path)

    def tearDown(self) -> None:
        os.unlink(self.walker_path)
        os.unlink(self.core_path)

    def test_constructs(self) -> None:
        w = GraphWalker(self.walker_cfg, self.core_cfg)
        self.assertIsNotNone(w)

    def test_temperature_out_of_range(self) -> None:
        bad = SAMPLE_WALKER_YAML.replace("temperature: 0.1", "temperature: 1.0")
        p = write_yaml(bad)
        with self.assertRaises(ValidationError):
            GraphWalker(WalkerConfig.from_yaml(p), self.core_cfg)

    def test_cross_config_mismatch_raises(self) -> None:
        bad = SAMPLE_CORE_YAML.replace("default_temperature: 0.1",
                                       "default_temperature: 0.5")
        p = write_yaml(bad)
        with self.assertRaises(ValidationError):
            GraphWalker(self.walker_cfg, CoreConfig.from_yaml(p))

    def test_invalid_formula_raises(self) -> None:
        bad = SAMPLE_WALKER_YAML.replace(
            "formula: \"strength * confidence * target_activation * intent_bias(edge_type, current_intent)\"",
            "formula: \"strength * confidence\"",
        )
        p = write_yaml(bad)
        with self.assertRaises(ValidationError):
            GraphWalker(WalkerConfig.from_yaml(p), self.core_cfg)

    def test_set_temperature_valid(self) -> None:
        w = GraphWalker(self.walker_cfg, self.core_cfg)
        w.set_temperature(0.2)
        self.assertEqual(w._temperature, 0.2)

    def test_set_temperature_out_of_range(self) -> None:
        w = GraphWalker(self.walker_cfg, self.core_cfg)
        with self.assertRaises(ValidationError):
            w.set_temperature(0.6)

    def test_random_seed_reproducibility(self) -> None:
        w1 = GraphWalker(self.walker_cfg, self.core_cfg, random_seed=42)
        w2 = GraphWalker(self.walker_cfg, self.core_cfg, random_seed=42)
        self.assertEqual(w1._rng.getstate(), w2._rng.getstate())

    def test_logger_default(self) -> None:
        w = GraphWalker(self.walker_cfg, self.core_cfg)
        self.assertEqual(w._logger.name, "glmx.walker")


# =========================================================================
# GraphWalker - walk execution
# =========================================================================

class TestGraphWalkerWalk(unittest.TestCase):
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

    def test_basic_walk(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph(
            edges=[(1, 2, "is_a"), (2, 3, "is_a"), (3, 4, "is_a"), (4, 5, "is_a")],
            seed_nodes=[1],
        )
        plan = make_plan(intent_sequence=[0])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        self.assertIsInstance(result, WalkResult)
        self.assertGreater(len(result.path), 1)
        self.assertEqual(result.steps_taken, len(result.path_edges))
        self.assertEqual(len(result.path_activations), len(result.path))
        self.assertEqual(len(result.path_confidences), len(result.path_edges))
        self.assertEqual(len(result.path_embeddings), len(result.path))
        self.assertAlmostEqual(result.walk_confidence,
                               geometric_mean(result.path_confidences))
        result.validate(0.01, 1.0)

    def test_walk_with_node_embeddings(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph(node_embeddings=True)
        plan = make_plan()
        w = self._walker()
        result = w.walk(sg, plan)
        self.assertIsInstance(result, WalkResult)

    def test_walk_no_embedding_raises(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph()
        plan = make_plan()
        w = self._walker()
        with self.assertRaises(EmbeddingLookupError):
            w.walk(sg, plan)

    def test_exact_expected_relation_beats_hotter_alternative(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        # The is_a target is far hotter (activation 0.99 vs 0.5), so a pure
        # score/similarity ranking would follow it. The extractor chain says
        # the step expects "has_property"; the exact-match short-circuit must
        # win regardless of how hot the competing edge is.
        sg = make_subgraph(
            edges=[(1, 2, "is_a"), (1, 3, "has_property")],
            node_activations={1: 0.5, 2: 0.99, 3: 0.5, 4: 0.5, 5: 0.5},
            seed_nodes=[1],
        )
        plan = Plan(relation_chain=["has_property"])
        w = self._walker(
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
            random_seed=42,
        )
        result = w.walk(sg, plan)
        self.assertEqual(result.path_edges[0], "has_property")
        self.assertEqual(result.path[1], 3)

    def test_walk_partial_embeddings_uses_provider(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = dataclasses.replace(
            make_subgraph(),
            node_embeddings={n: np.zeros(32, dtype=np.int8) for n in [2, 3, 4, 5]},
        )
        plan = make_plan()
        w = GraphWalker(
            self.walker_cfg, self.core_cfg,
            embedding_provider=lambda nid: np.zeros(32, dtype=np.int8),
        )
        result = w.walk(sg, plan)
        self.assertEqual(len(result.path_embeddings), len(result.path))

    def test_walk_missing_provider_embedding_raises(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = dataclasses.replace(make_subgraph(), node_embeddings={})
        plan = make_plan()
        w = GraphWalker(self.walker_cfg, self.core_cfg, embedding_provider=lambda nid: None)
        with self.assertRaises(EmbeddingLookupError):
            w.walk(sg, plan)

    def test_walk_no_recording(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        cfg_yaml = SAMPLE_WALKER_YAML
        cfg_yaml = cfg_yaml.replace("record_activations: true",
                                    "record_activations: false")
        cfg_yaml = cfg_yaml.replace("record_edge_confidence: true",
                                    "record_edge_confidence: false")
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        sg = make_subgraph()
        plan = make_plan()
        w = GraphWalker(cfg, self.core_cfg,
                        embedding_provider=lambda nid: np.zeros(32))
        result = w.walk(sg, plan)
        self.assertEqual(result.path_activations, [])
        self.assertEqual(result.path_confidences, [])

    def test_walk_no_timestamps(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        cfg_yaml = SAMPLE_WALKER_YAML.replace("record_timestamps: true",
                                              "record_timestamps: false")
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        sg = make_subgraph()
        plan = make_plan()
        w = GraphWalker(cfg, self.core_cfg,
                        embedding_provider=lambda nid: np.zeros(32))
        w.walk(sg, plan)
        self.assertEqual(w.get_last_step_timestamps(), [])

    def test_intent_bias_api(self) -> None:
        w = self._walker()
        self.assertEqual(w.get_intent_bias(0, "is_a"), 1.5)
        w.update_intent_bias(0, "is_a", 2.5)
        self.assertEqual(w.get_intent_bias(0, "is_a"), 2.5)

    def test_next_possible_nodes(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph()
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        nodes = w.next_possible_nodes(1, sg, 0)
        self.assertGreater(len(nodes), 0)
        for nid, score in nodes:
            self.assertIsInstance(nid, int)
            self.assertIsInstance(score, float)

    def test_public_eligibility_trace(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph()
        plan = make_plan()
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        result = w.walk(sg, plan)
        trace = w.compute_eligibility_trace(result, sg)
        self.assertIsInstance(trace, dict)
        for key, value in trace.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, float)

    def test_walk_confidence_range(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        sg = make_subgraph()
        plan = make_plan()
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        result = w.walk(sg, plan)
        self.assertGreaterEqual(result.walk_confidence, 0.0)
        self.assertLessEqual(result.walk_confidence, 1.0)

    def test_saved_paths_debug(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        cfg_yaml = SAMPLE_WALKER_YAML.replace("save_all_paths: false",
                                              "save_all_paths: true")
        p = write_yaml(cfg_yaml)
        cfg = WalkerConfig.from_yaml(p)
        sg = make_subgraph()
        plan = make_plan()
        w = GraphWalker(cfg, self.core_cfg,
                        embedding_provider=lambda nid: np.zeros(32))
        w.walk(sg, plan)
        self.assertGreater(len(w.get_saved_paths()), 0)

    def test_min_activation_stops_walk(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        act = {1: 0.5, 2: 0.04, 3: 0.5, 4: 0.5, 5: 0.5}
        edges = [(1, 2, "is_a"), (2, 3, "causes")]
        sg = make_subgraph(edges=edges, node_activations=act,
                           seed_nodes=[1])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        result = w.walk(sg, plan)
        self.assertEqual(len(result.path), 1)

    def test_restart_on_dead_end(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        edges = [(1, 2, "is_a"), (1, 3, "causes")]
        sg = make_subgraph(edges=edges, seed_nodes=[1, 4, 5])
        plan = make_plan(intent_sequence=[0])
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        result = w.walk(sg, plan)
        self.assertGreater(len(result.path), 0)

    def test_validates_input(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        w = self._walker(embedding_provider=lambda nid: np.zeros(32))
        sg = make_subgraph(tier_used=3)
        plan = make_plan()
        with self.assertRaises(ValidationError):
            w.walk(sg, plan)

    def test_get_walk_confidence_empty(self) -> None:
        if not HAS_NUMPY:
            self.skipTest("numpy unavailable")
        emb = [np.zeros(32) for _ in range(2)]
        wr = WalkResult.build(
            path=[1, 2], path_edges=["is_a"],
            path_activations=[], path_confidences=[],
            path_embeddings=emb, plan=make_plan(),
        )
        w = self._walker()
        self.assertEqual(w.get_walk_confidence(wr), 1.0)


# =========================================================================
# Exceptions
# =========================================================================

class TestExceptions(unittest.TestCase):
    def test_validation_error(self) -> None:
        self.assertTrue(issubclass(ValidationError, Exception))

    def test_embedding_lookup_error(self) -> None:
        self.assertTrue(issubclass(EmbeddingLookupError, Exception))

    def test_validation_message(self) -> None:
        try:
            raise ValidationError("bad config")
        except ValidationError as e:
            self.assertEqual(str(e), "bad config")

    def test_embedding_message(self) -> None:
        try:
            raise EmbeddingLookupError("not found")
        except EmbeddingLookupError as e:
            self.assertEqual(str(e), "not found")


if __name__ == "__main__":
    unittest.main()
