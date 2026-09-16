"""DEVIATION 9: relation-chain guided walker tests.

Plans now carry an ordered ``relation_chain`` (e.g. ["is_a", "causes"]) and the
walker steers each step with the chain's expected relation instead of legacy
intent ids. These tests exercise the chain path end-to-end with a hermetic
relation-bias table.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from typing import Dict, Optional

import numpy as np

from walker.config import CoreConfig, WalkerConfig
from walker.models import Plan, Subgraph
from walker.graph_walker import GraphWalker

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

RELATION_BIAS_YAML = """
walk:
  max_steps: 20
  min_activation: 0.05
  temperature: 0.1
  temperature_range: [0.05, 0.5]
  allow_cycles: false
  allow_backtrack: false
  restart_on_dead_end: false
  restart_penalty: 0.5

relation_biases:
  is_a:
    is_a: 3.0
    causes: 0.2
    contradicts: 0.1
    default: 0.2
  causes:
    is_a: 0.2
    causes: 3.0
    contradicts: 0.1
    default: 0.2
  contradicts:
    is_a: 0.1
    causes: 0.2
    contradicts: 3.0
    default: 0.2

scoring:
  formula: "strength * confidence * target_activation * intent_bias(edge_type, current_intent)"
  weight_strength: 1.0
  weight_confidence: 1.0
  weight_target_activation: 1.0
  weight_intent_bias: 5.0
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
  log_path_taken: false
  save_all_paths: false
"""

LEGACY_INTENT_YAML = """
walk:
  max_steps: 20
  min_activation: 0.05
  temperature: 0.1
  temperature_range: [0.05, 0.5]
  allow_cycles: false
  allow_backtrack: false
  restart_on_dead_end: false
  restart_penalty: 0.5

intent_biases:
  0:
    is_a: 1.5
    has_property: 1.2
    example_of: 1.0
    default: 0.5

scoring:
  formula: "strength * confidence * target_activation * intent_bias(edge_type, current_intent)"
  weight_strength: 1.0
  weight_confidence: 1.0
  weight_target_activation: 1.0
  weight_intent_bias: 5.0
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
  log_path_taken: false
  save_all_paths: false

""".replace("walk:\n  max_steps: 20", "walk:\n  max_steps: 20")  # keep consistent


def write_yaml(content: str) -> str:
    path = tempfile.mktemp(suffix=".yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def make_subgraph(edges, activations: Dict[int, float], seed: int = 1) -> Subgraph:
    nodes = sorted({n for e in edges for n in (e[0], e[1])} | {seed})
    strengths = {e: 0.8 for e in edges}
    confidences = {e: 0.9 for e in edges}
    emb = np.zeros(32, dtype=np.int8)
    return Subgraph(
        nodes=nodes,
        node_activations={n: activations.get(n, 0.5) for n in nodes},
        edges=edges,
        edge_strengths=strengths,
        edge_confidences=confidences,
        seed_nodes=[seed],
        tier_used=1,
        activation_energy=1.0,
        query_embedding=np.zeros(384, dtype=np.float32),
        node_embeddings={n: emb for n in nodes},
        timestamp=time.time(),
    )


class ChainWalkTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._core_p = write_yaml(SAMPLE_CORE_YAML)
        cls._rel_p = write_yaml(RELATION_BIAS_YAML)
        cls._legacy_p = write_yaml(LEGACY_INTENT_YAML)
        cls.core_cfg = CoreConfig.from_yaml(cls._core_p)

    @classmethod
    def tearDownClass(cls):
        for path in (cls._core_p, cls._rel_p, cls._legacy_p):
            os.unlink(path)

    def _walker(self, yaml_path: str, seed: int = 0) -> GraphWalker:
        cfg = WalkerConfig.from_yaml(yaml_path)
        provider = lambda nid: np.zeros(32, dtype=np.int8)
        return GraphWalker(cfg, self.core_cfg, embedding_provider=provider, random_seed=seed)


class TestRelationChainPreference(ChainWalkTestBase):
    def test_chain_is_a_prefers_is_a_edge(self):
        act = {1: 0.5, 2: 0.6, 3: 0.6}
        sg = make_subgraph([(1, 2, "is_a"), (1, 3, "contradicts")], act, seed=1)
        plan = Plan(relation_chain=["is_a"], plan_confidence=0.9, heuristic_fallback_used=False)
        result = self._walker(self._rel_p).walk(sg, plan)
        self.assertGreaterEqual(len(result.path), 2)
        self.assertEqual(result.path[1], 2)
        self.assertEqual(result.path_edges, ["is_a"])

    def test_chain_contradicts_prefers_contradicts_edge(self):
        act = {1: 0.5, 2: 0.6, 3: 0.6}
        sg = make_subgraph([(1, 2, "is_a"), (1, 3, "contradicts")], act, seed=1)
        plan = Plan(relation_chain=["contradicts"], plan_confidence=0.9, heuristic_fallback_used=False)
        result = self._walker(self._rel_p).walk(sg, plan)
        self.assertEqual(result.path[1], 3)
        self.assertEqual(result.path_edges, ["contradicts"])

    def test_relation_bias_api_mirrors_expected_relation(self):
        walker = self._walker(self._rel_p)
        self.assertAlmostEqual(walker.get_relation_bias("is_a", "is_a"), 3.0)
        self.assertAlmostEqual(walker.get_relation_bias("is_a", "contradicts"), 0.1)
        # Legacy intent shim maps intent 0 -> is_a
        self.assertAlmostEqual(walker.get_intent_bias(0, "is_a"), 3.0)


class TestMultiStepChain(ChainWalkTestBase):
    def test_chain_progresses_step_by_step(self):
        act = {1: 0.5, 2: 0.6, 3: 0.7}
        sg = make_subgraph([(1, 2, "is_a"), (2, 3, "causes")], act, seed=1)
        plan = Plan(relation_chain=["is_a", "causes"], plan_confidence=0.9, heuristic_fallback_used=False)
        result = self._walker(self._rel_p).walk(sg, plan)
        self.assertEqual(result.path_edges, ["is_a", "causes"])
        self.assertEqual(result.path, [1, 2, 3])
        self.assertEqual(result.relation_chain_used, ["is_a", "causes"])
        self.assertEqual(list(result.intent_sequence_used), [])

    def test_last_chain_relation_repeats_at_end(self):
        act = {1: 0.5, 2: 0.6, 3: 0.7, 4: 0.7}
        sg = make_subgraph([(1, 2, "causes"), (2, 3, "causes"), (3, 4, "is_a")], act, seed=1)
        plan = Plan(relation_chain=["causes"], plan_confidence=0.9, heuristic_fallback_used=False)
        result = self._walker(self._rel_p).walk(sg, plan)
        self.assertEqual(result.path_edges[:2], ["causes", "causes"])
        self.assertGreaterEqual(len(result.path_edges), 2)


class TestLegacyIntentFallback(ChainWalkTestBase):
    def test_intent_sequence_maps_to_legacy_relation_chain(self):
        act = {1: 0.5, 2: 0.6, 3: 0.6}
        sg = make_subgraph([(1, 2, "is_a"), (1, 3, "contradicts")], act, seed=1)
        plan = Plan(intent_sequence=[0], plan_confidence=0.9, heuristic_fallback_used=False)
        result = self._walker(self._legacy_p).walk(sg, plan)
        self.assertEqual(result.path[1], 2)
        self.assertEqual(result.path_edges, ["is_a"])

    def test_legacy_intent_biases_populate_relation_table(self):
        cfg = WalkerConfig.from_yaml(self._legacy_p)
        self.assertEqual(cfg.relation_biases["is_a"]["is_a"], 1.5)


if __name__ == "__main__":
    unittest.main()