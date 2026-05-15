import os
import sys
import json
import io
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- Early mock of sentence_transformers before any G2P imports ----
import sys as _sys
_sys.modules['sentence_transformers'] = MagicMock()
_sys.modules['sentence_transformers.SentenceTransformer'] = MagicMock()

patch_sentence = patch('sentence_transformers.SentenceTransformer')
_mock_st_class = MagicMock()
_mock_st_instance = MagicMock()
_mock_st_instance.encode.return_value = np.random.randn(384).astype(np.float32)
_mock_st_class.return_value = _mock_st_instance
_sys.modules['sentence_transformers.SentenceTransformer'] = _mock_st_class

# ---- End mock ----

import torch
import yaml

from g2p.types import Subgraph, Plan
from g2p.config import (
    G2PConfig, SentenceBERTConfig, GraphToTextConfig, FFNConfig,
    DecoderConfig, TrainingConfig, MappingConfig, ValidationConfig,
    SyntheticDataConfig, RuleDefinition
)
from g2p.graph_to_text import GraphToTextEncoder
from g2p.intent_ffn import IntentFFN
from g2p.beam_search import BeamSearchDecoder
from g2p.heuristic_planner import HeuristicPlanner, _RuleEnvironment, _SubgraphProxy
from g2p.g2p_planner import G2PPlanner
from g2p.train import generate_synthetic_data, train_g2p


def make_valid_subgraph(seed: int = 42, extra_nodes: int = 0) -> Subgraph:
    rng = np.random.RandomState(seed)
    n_nodes = 5 + extra_nodes
    node_ids = list(range(n_nodes))
    node_activations = {n: float(rng.uniform(0.01, 1.0)) for n in node_ids}
    edges = []
    edge_strengths = {}
    edge_confidences = {}
    for i in range(n_nodes - 1):
        k = (node_ids[i], node_ids[i + 1], "causes")
        edges.append(k)
        edge_strengths[k] = float(rng.uniform(0.0, 1.0))
        edge_confidences[k] = float(rng.uniform(0.0, 1.0))
    qe = rng.randn(384).astype(np.float32)
    qe = qe / np.linalg.norm(qe)
    return Subgraph(
        nodes=node_ids,
        node_activations=node_activations,
        edges=edges,
        edge_strengths=edge_strengths,
        edge_confidences=edge_confidences,
        seed_nodes=[node_ids[0]],
        tier_used=1,
        activation_energy=float(rng.uniform(0.1, 5.0)),
        query_embedding=qe,
        timestamp=float(rng.uniform(1e9, 1.7e9)),
    )


# ====================================================================
# TYPES
# ====================================================================
class TestSubgraphValidation(unittest.TestCase):
    def test_valid_subgraph_passes(self):
        sg = make_valid_subgraph()
        try:
            sg.validate()
        except ValueError as e:
            self.fail(f"Valid subgraph raised: {e}")

    def test_activation_below_min(self):
        sg = make_valid_subgraph()
        sg.node_activations[0] = 0.0
        with self.assertRaises(ValueError):
            sg.validate()

    def test_activation_above_max(self):
        sg = make_valid_subgraph()
        sg.node_activations[0] = 1.5
        with self.assertRaises(ValueError):
            sg.validate()

    def test_edge_refers_missing_node(self):
        sg = make_valid_subgraph()
        sg.edges.append((999, 0, "causes"))
        sg.edge_strengths[(999, 0, "causes")] = 0.5
        sg.edge_confidences[(999, 0, "causes")] = 0.5
        with self.assertRaises(ValueError):
            sg.validate()

    def test_edge_strengths_keys_mismatch(self):
        sg = make_valid_subgraph()
        sg.edge_strengths[(999, 0, "fake")] = 0.5
        with self.assertRaises(ValueError):
            sg.validate()

    def test_edge_confidences_keys_mismatch(self):
        sg = make_valid_subgraph()
        sg.edge_confidences[(999, 0, "fake")] = 0.5
        with self.assertRaises(ValueError):
            sg.validate()

    def test_empty_seed_nodes(self):
        sg = make_valid_subgraph()
        sg.seed_nodes = []
        with self.assertRaises(ValueError):
            sg.validate()

    def test_invalid_tier(self):
        sg = make_valid_subgraph()
        sg.tier_used = 3
        with self.assertRaises(ValueError):
            sg.validate()

    def test_wrong_query_embedding_shape(self):
        sg = make_valid_subgraph()
        sg.query_embedding = np.zeros((128,))
        with self.assertRaises(ValueError):
            sg.validate()


class TestPlanValidation(unittest.TestCase):
    def test_valid_plan_passes(self):
        p = Plan(intent_sequence=[0, 1, 2], plan_confidence=0.8, heuristic_fallback_used=False)
        try:
            p.validate()
        except ValueError as e:
            self.fail(f"Valid plan raised: {e}")

    def test_intent_id_too_low(self):
        p = Plan(intent_sequence=[-1], plan_confidence=0.5, heuristic_fallback_used=False)
        with self.assertRaises(ValueError):
            p.validate()

    def test_intent_id_too_high(self):
        p = Plan(intent_sequence=[16], plan_confidence=0.5, heuristic_fallback_used=False)
        with self.assertRaises(ValueError):
            p.validate()

    def test_empty_sequence(self):
        p = Plan(intent_sequence=[], plan_confidence=0.5, heuristic_fallback_used=False)
        with self.assertRaises(ValueError):
            p.validate()

    def test_sequence_too_long(self):
        p = Plan(intent_sequence=list(range(9)), plan_confidence=0.5, heuristic_fallback_used=False)
        with self.assertRaises(ValueError):
            p.validate()

    def test_confidence_below_zero(self):
        p = Plan(intent_sequence=[0], plan_confidence=-0.1, heuristic_fallback_used=False)
        with self.assertRaises(ValueError):
            p.validate()

    def test_confidence_above_one(self):
        p = Plan(intent_sequence=[0], plan_confidence=1.5, heuristic_fallback_used=False)
        with self.assertRaises(ValueError):
            p.validate()

    def test_intent_names_length_mismatch(self):
        p = Plan(intent_sequence=[0, 1], plan_confidence=0.5, heuristic_fallback_used=False,
                 intent_names=["define"])
        with self.assertRaises(ValueError):
            p.validate()

    def test_intent_names_none_skips_check(self):
        p = Plan(intent_sequence=[0, 1], plan_confidence=0.5, heuristic_fallback_used=False,
                 intent_names=None)
        try:
            p.validate()
        except ValueError as e:
            self.fail(f"Valid plan with None intent_names raised: {e}")


# ====================================================================
# CONFIG
# ====================================================================
class TestG2PConfigFromYaml(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, '_temp_yaml') and os.path.exists(self._temp_yaml):
            os.remove(self._temp_yaml)

    def _write_yaml(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        f.write(content)
        f.close()
        self._temp_yaml = f.name
        return f.name

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            G2PConfig.from_yaml("nonexistent_file.yaml")

    def test_empty_yaml_uses_defaults(self):
        path = self._write_yaml("{}")
        cfg = G2PConfig.from_yaml(path)
        self.assertEqual(cfg.sentence_bert.model_dim, 384)
        self.assertEqual(cfg.ffn.input_dim, 384)
        self.assertEqual(cfg.decoder.beam_width, 2)
        self.assertEqual(cfg.training.epochs, 50)
        self.assertEqual(cfg.validation.allowed_intents, list(range(16)))

    def test_partial_yaml_fills_defaults(self):
        path = self._write_yaml("ffn:\n  hidden_dim: 256\n")
        cfg = G2PConfig.from_yaml(path)
        self.assertEqual(cfg.ffn.hidden_dim, 256)
        self.assertEqual(cfg.ffn.input_dim, 384)  # default
        self.assertEqual(cfg.ffn.num_layers, 2)    # default

    def test_full_yaml_overrides(self):
        path = self._write_yaml("""
sentence_bert:
  model_name: "test-model"
  model_dim: 128
graph_to_text:
  max_nodes_in_text: 50
  separator: ","
ffn:
  input_dim: 128
  output_dim: 16
decoder:
  beam_width: 5
training:
  epochs: 10
  synthetic_data:
    num_samples: 100
mapping:
  heuristic_rules_enabled: false
  rule_definitions:
    - condition: "len(subgraph.nodes) > 5"
      plan: [1, 2, 3]
validation:
  input_subgraph_max_nodes: 500
  allowed_intents: [0, 1, 2]
intent_embeddings:
  0: "test description"
""")
        cfg = G2PConfig.from_yaml(path)
        self.assertEqual(cfg.sentence_bert.model_name, "test-model")
        self.assertEqual(cfg.sentence_bert.model_dim, 128)
        self.assertEqual(cfg.graph_to_text.max_nodes_in_text, 50)
        self.assertEqual(cfg.graph_to_text.separator, ",")
        self.assertEqual(cfg.ffn.input_dim, 128)
        self.assertEqual(cfg.decoder.beam_width, 5)
        self.assertEqual(cfg.training.epochs, 10)
        self.assertEqual(cfg.training.synthetic_data.num_samples, 100)
        self.assertFalse(cfg.mapping.heuristic_rules_enabled)
        self.assertEqual(len(cfg.mapping.rule_definitions), 1)
        self.assertEqual(cfg.mapping.rule_definitions[0].condition, "len(subgraph.nodes) > 5")
        self.assertEqual(cfg.validation.input_subgraph_max_nodes, 500)
        self.assertEqual(cfg.validation.allowed_intents, [0, 1, 2])
        self.assertEqual(cfg.intent_embeddings[0], "test description")

    def test_rule_definitions_empty_by_default(self):
        path = self._write_yaml("mapping:\n  rule_definitions: []\n")
        cfg = G2PConfig.from_yaml(path)
        self.assertEqual(len(cfg.mapping.rule_definitions), 0)

    def test_rule_definitions_parsed_correctly(self):
        path = self._write_yaml("""
mapping:
  rule_definitions:
    - condition: "a"
      plan: [0]
    - condition: "b"
      plan: [1, 2]
""")
        cfg = G2PConfig.from_yaml(path)
        self.assertEqual(len(cfg.mapping.rule_definitions), 2)
        self.assertEqual(cfg.mapping.rule_definitions[0].condition, "a")
        self.assertEqual(cfg.mapping.rule_definitions[1].plan, [1, 2])


class TestG2PConfigValidate(unittest.TestCase):
    def test_dimension_mismatch_fails(self):
        cfg = G2PConfig()
        cfg.ffn.input_dim = 128
        cfg.sentence_bert.model_dim = 384
        with self.assertRaises(AssertionError):
            cfg.validate()

    def test_output_dim_not_16_fails(self):
        cfg = G2PConfig()
        cfg.ffn.output_dim = 32
        with self.assertRaises(AssertionError):
            cfg.validate()

    def test_decoder_length_exceeds_validation_fails(self):
        cfg = G2PConfig()
        cfg.decoder.max_length = 20
        cfg.validation.output_plan_max_length = 8
        with self.assertRaises(AssertionError):
            cfg.validate()

    def test_valid_config_passes(self):
        cfg = G2PConfig()
        try:
            cfg.validate()
        except AssertionError as e:
            self.fail(f"Valid config raised: {e}")


# ====================================================================
# GRAPH TO TEXT
# ====================================================================
class TestGraphToTextEncoder(unittest.TestCase):
    def setUp(self):
        self.default_config = GraphToTextConfig()
        self.label_map = {0: "zero", 1: "one", 2: "two"}
        self.encoder = GraphToTextEncoder(self.default_config, label_map=self.label_map)

    def _make_subgraph(self, node_ids, edges=None):
        node_activations = {n: float(np.random.uniform(0.01, 1.0)) for n in node_ids}
        if edges is None:
            edges = []
            for i in range(len(node_ids) - 1):
                edges.append((node_ids[i], node_ids[i+1], "causes"))
        edge_strengths = {k: 0.5 for k in edges}
        edge_confidences = {k: 0.5 for k in edges}
        qe = np.random.randn(384).astype(np.float32)
        qe = qe / np.linalg.norm(qe)
        return Subgraph(
            nodes=node_ids,
            node_activations=node_activations,
            edges=edges,
            edge_strengths=edge_strengths,
            edge_confidences=edge_confidences,
            seed_nodes=[node_ids[0]] if node_ids else [],
            tier_used=1,
            activation_energy=1.0,
            query_embedding=qe,
            timestamp=1000.0,
        )

    # _get_label
    def test_get_label_from_map(self):
        self.assertEqual(self.encoder._get_label(0), "zero")

    def test_get_label_fallback(self):
        self.assertEqual(self.encoder._get_label(99), "node_99")

    # _get_relation_between
    def test_get_relation_forward(self):
        edges = [(0, 1, "causes")]
        self.assertEqual(self.encoder._get_relation_between(0, 1, edges), "causes")

    def test_get_relation_reverse(self):
        edges = [(0, 1, "causes")]
        self.assertEqual(self.encoder._get_relation_between(1, 0, edges), "causes")

    def test_get_relation_none(self):
        edges = [(0, 1, "causes")]
        self.assertIsNone(self.encoder._get_relation_between(0, 2, edges))

    # _sort_nodes by activation
    def test_sort_by_activation_descending(self):
        self.encoder.config.sort_by = "activation"
        self.encoder.config.sort_order = "descending"
        nodes = [1, 2, 3]
        activations = {1: 0.1, 2: 0.9, 3: 0.5}
        result = self.encoder._sort_nodes(nodes, activations, {}, [])
        self.assertEqual(result, [2, 3, 1])

    def test_sort_by_activation_ascending(self):
        self.encoder.config.sort_by = "activation"
        self.encoder.config.sort_order = "ascending"
        nodes = [1, 2, 3]
        activations = {1: 0.1, 2: 0.9, 3: 0.5}
        result = self.encoder._sort_nodes(nodes, activations, {}, [])
        self.assertEqual(result, [1, 3, 2])

    def test_sort_by_confidence(self):
        self.encoder.config.sort_by = "confidence"
        self.encoder.config.sort_order = "descending"
        nodes = [0, 1, 2]
        activations = {0: 0.1, 1: 0.5, 2: 0.9}
        edges = [(0, 1, "causes"), (1, 2, "is_a")]
        confs = {(0, 1, "causes"): 0.3, (1, 2, "is_a"): 0.8}
        result = self.encoder._sort_nodes(nodes, activations, confs, edges)
        self.assertEqual(result, [1, 2, 0])

    def test_sort_by_recency_fallback(self):
        self.encoder.config.sort_by = "recency"
        self.encoder.config.sort_order = "descending"
        nodes = [1, 2]
        activations = {1: 0.2, 2: 0.7}
        result = self.encoder._sort_nodes(nodes, activations, {}, [])
        self.assertEqual(result, [2, 1])

    def test_sort_by_unknown_fallback(self):
        self.encoder.config.sort_by = "unknown_field"
        self.encoder.config.sort_order = "descending"
        nodes = [3, 1, 2]
        activations = {1: 0.9, 2: 0.1, 3: 0.5}
        result = self.encoder._sort_nodes(nodes, activations, {}, [])
        self.assertEqual(result, [1, 3, 2])

    # encode
    def test_encode_truncates_when_exceeds_max(self):
        self.encoder.config.max_nodes_in_text = 2
        sg = self._make_subgraph([0, 1, 2, 3])
        result = self.encoder.encode(sg)
        # only 2 nodes among the 4 should appear in output
        labels_found = sum(1 for label in ["zero", "one", "two", "node_3"] if label in result)
        self.assertEqual(labels_found, 2)

    def test_encode_all_nodes_when_within_limit(self):
        self.encoder.config.max_nodes_in_text = 100
        sg = self._make_subgraph([0, 1, 2])
        result = self.encoder.encode(sg)
        self.assertIn("zero", result)
        self.assertIn("one", result)
        self.assertIn("two", result)

    def test_encode_with_activations_above_threshold(self):
        self.encoder.config.include_activations = True
        self.encoder.config.activation_threshold = 0.1
        sg = self._make_subgraph([0, 1])
        sg.node_activations[0] = 0.9
        result = self.encoder.encode(sg)
        self.assertIn("(0.90)", result)

    def test_encode_with_activations_below_threshold(self):
        self.encoder.config.include_activations = True
        self.encoder.config.activation_threshold = 0.5
        sg = self._make_subgraph([0, 1])
        sg.node_activations[0] = 0.1
        result = self.encoder.encode(sg)
        self.assertNotIn("(0.10)", result)  # no activation template used

    def test_encode_without_activations(self):
        self.encoder.config.include_activations = False
        sg = self._make_subgraph([0, 1])
        result = self.encoder.encode(sg)
        self.assertNotIn("(", result)

    def test_encode_with_edge_types_shows_relation(self):
        self.encoder.config.include_edge_types = True
        sg = self._make_subgraph([0, 1], edges=[(0, 1, "causes")])
        result = self.encoder.encode(sg)
        self.assertIn("via causes", result)

    def test_encode_with_edge_types_no_relation_uses_separator(self):
        self.encoder.config.include_edge_types = True
        sg = self._make_subgraph([0, 1], edges=[(0, 2, "causes")])  # no edge between 0-1
        result = self.encoder.encode(sg)
        self.assertIn(self.encoder.config.separator, result)

    def test_encode_without_edge_types(self):
        self.encoder.config.include_edge_types = False
        sg = self._make_subgraph([0, 1], edges=[(0, 1, "causes")])
        result = self.encoder.encode(sg)
        self.assertNotIn("via", result)

    def test_encode_empty_subgraph(self):
        sg = self._make_subgraph([])
        result = self.encoder.encode(sg)
        self.assertEqual(result, "")

    def test_encode_single_node(self):
        self.encoder.config.include_activations = False
        sg = self._make_subgraph([5])
        result = self.encoder.encode(sg)
        self.assertEqual(result, "node_5")

    def test_set_label_map_updates(self):
        self.encoder.set_label_map({5: "five"})
        self.assertEqual(self.encoder._get_label(5), "five")

    def test_encode_uses_label_names(self):
        sg = self._make_subgraph([0, 1])
        result = self.encoder.encode(sg)
        self.assertIn("zero", result)
        self.assertIn("one", result)


# ====================================================================
# INTENT FFN
# ====================================================================
class TestIntentFFN(unittest.TestCase):
    def setUp(self):
        self.config = FFNConfig(input_dim=384, hidden_dim=128, num_layers=2, output_dim=16)
        self.model = IntentFFN(self.config)

    def test_forward_output_shape_batch(self):
        x = torch.randn(4, 384)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 16))

    def test_forward_output_shape_single(self):
        x = torch.randn(1, 384)
        out = self.model(x)
        self.assertEqual(out.shape, (1, 16))

    def test_forward_wrong_input_dim_raises(self):
        x = torch.randn(1, 128)
        with self.assertRaises(Exception):
            self.model(x)

    def test_predict_logits_1d_input(self):
        emb = np.random.randn(384).astype(np.float32)
        logits = self.model.predict_logits(emb)
        self.assertEqual(logits.shape, (16,))
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_predict_logits_2d_batch_input(self):
        emb = np.random.randn(3, 384).astype(np.float32)
        logits = self.model.predict_logits(emb)
        self.assertEqual(logits.shape, (3, 16))

    def test_predict_probs_sum_to_one(self):
        emb = np.random.randn(384).astype(np.float32)
        probs = self.model.predict_probs(emb)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0))

    def test_eval_mode_during_predict(self):
        self.assertTrue(self.model.training)
        _ = self.model.predict_logits(np.random.randn(384).astype(np.float32))
        self.assertFalse(self.model.training)

    def test_model_structure_layers(self):
        # 2 hidden layers: each = Linear+ReLU+Dropout (3 modules), final Linear (1) = 7
        self.assertEqual(len(self.model.body), 7)
        # 1 classifier layer: Linear+ReLU+Linear = 3
        self.assertEqual(len(self.model.classifier), 3)

    def test_different_hidden_dim(self):
        cfg = FFNConfig(input_dim=384, hidden_dim=64, num_layers=1, output_dim=16)
        model = IntentFFN(cfg)
        x = torch.randn(2, 384)
        out = model(x)
        self.assertEqual(out.shape, (2, 16))

    def test_different_classifier_layers(self):
        cfg = FFNConfig(input_dim=384, hidden_dim=128, num_layers=1,
                        classifier_input_dim=128, classifier_hidden_dim=32,
                        classifier_num_layers=2, output_dim=16)
        model = IntentFFN(cfg)
        x = torch.randn(2, 384)
        out = model(x)
        self.assertEqual(out.shape, (2, 16))


# ====================================================================
# BEAM SEARCH
# ====================================================================
class TestBeamSearchDecoder(unittest.TestCase):
    def setUp(self):
        self.config = DecoderConfig(beam_width=2, max_length=8, temperature=1.0, repetition_penalty=1.2)
        self.decoder = BeamSearchDecoder(self.config)

    def test_basic_decode_uniform_logits(self):
        logits = np.ones(16, dtype=np.float32)
        seq, conf = self.decoder.decode(logits)
        self.assertGreaterEqual(len(seq), 1)
        self.assertLessEqual(len(seq), 8)
        self.assertTrue(all(0 <= i <= 15 for i in seq))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_decode_favors_dominant_intent(self):
        logits = np.zeros(16, dtype=np.float32)
        logits[3] = 10.0
        seq, _ = self.decoder.decode(logits)
        self.assertEqual(seq[0], 3)

    def test_beam_width_one_is_greedy(self):
        self.config.beam_width = 1
        decoder = BeamSearchDecoder(self.config)
        logits = np.zeros(16, dtype=np.float32)
        logits[7] = 5.0
        seq, _ = decoder.decode(logits)
        # with max_length=8, all tokens should be intent 7
        self.assertTrue(all(i == 7 for i in seq))

    def test_beam_width_greater_than_one(self):
        self.config.beam_width = 3
        decoder = BeamSearchDecoder(self.config)
        logits = np.random.randn(16).astype(np.float32)
        seq, conf = decoder.decode(logits)
        self.assertTrue(len(seq) > 0)
        self.assertTrue(0 <= conf <= 1)

    def test_repetition_penalty_reduces_repeats(self):
        self.config.beam_width = 1
        self.config.repetition_penalty = 100.0
        self.config.max_length = 5
        decoder = BeamSearchDecoder(self.config)
        logits = np.zeros(16, dtype=np.float32)
        logits[0] = 10.0
        seq, _ = decoder.decode(logits)
        self.assertGreater(len(set(seq)), 1)

    def test_no_repetition_penalty_allows_same_intent(self):
        self.config.beam_width = 1
        self.config.repetition_penalty = 0.0
        self.config.max_length = 5
        decoder = BeamSearchDecoder(self.config)
        logits = np.zeros(16, dtype=np.float32)
        logits[0] = 10.0
        seq, _ = decoder.decode(logits)
        self.assertEqual(len(set(seq)), 1)

    def test_temperature_zero(self):
        self.config.temperature = 1e-10
        decoder = BeamSearchDecoder(self.config)
        logits = np.random.randn(16).astype(np.float32)
        seq, conf = decoder.decode(logits)
        self.assertTrue(all(0 <= i <= 15 for i in seq))
        self.assertTrue(0 <= conf <= 1)

    def test_high_temperature_flattens(self):
        self.config.temperature = 100.0
        decoder = BeamSearchDecoder(self.config)
        logits = np.zeros(16, dtype=np.float32)
        logits[0] = 100.0
        seq, conf = decoder.decode(logits)
        self.assertTrue(0 <= conf <= 1)

    def test_max_length_one(self):
        self.config.max_length = 1
        decoder = BeamSearchDecoder(self.config)
        logits = np.random.randn(16).astype(np.float32)
        seq, _ = decoder.decode(logits)
        self.assertEqual(len(seq), 1)

    def test_confidence_clamped(self):
        logits = np.array([1e10 if i == 0 else -1e10 for i in range(16)], dtype=np.float32)
        seq, conf = self.decoder.decode(logits)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_decode_all_equal_logits_deterministic(self):
        self.config.beam_width = 1
        decoder = BeamSearchDecoder(self.config)
        logits = np.zeros(16, dtype=np.float32)
        seq1, _ = decoder.decode(logits)
        seq2, _ = decoder.decode(logits)
        self.assertEqual(seq1, seq2)

    def test_set_intent_names(self):
        names = {0: "test"}
        self.decoder.set_intent_names(names)
        self.assertEqual(self.decoder.intent_names, names)


# ====================================================================
# HEURISTIC PLANNER
# ====================================================================
class TestHeuristicPlanner(unittest.TestCase):
    def setUp(self):
        self.rules = [
            RuleDefinition(condition="len(subgraph.nodes) <= 3", plan=[0]),
            RuleDefinition(condition="has_edge_type('causes')", plan=[1, 2, 7]),
        ]
        self.config = MappingConfig(heuristic_rules_enabled=True, rule_definitions=self.rules)
        self.planner = HeuristicPlanner(self.config)

    def _small_subgraph(self) -> Subgraph:
        return make_valid_subgraph(seed=1, extra_nodes=0)  # 5 nodes

    def test_rules_disabled_returns_none(self):
        self.planner.config.heuristic_rules_enabled = False
        result = self.planner.evaluate(self._small_subgraph())
        self.assertIsNone(result)

    def test_no_rules_returns_none(self):
        planner = HeuristicPlanner(MappingConfig(heuristic_rules_enabled=True, rule_definitions=[]))
        result = planner.evaluate(self._small_subgraph())
        self.assertIsNone(result)

    def test_rule_matches_returns_plan_and_confidence(self):
        sg = self._small_subgraph()
        sg.nodes = [0, 1]  # len <= 3
        result = self.planner.evaluate(sg)
        self.assertIsNotNone(result)
        seq, conf = result
        self.assertEqual(seq, [0])
        self.assertEqual(conf, 0.5)

    def test_rule_not_matched_tries_next(self):
        sg = make_valid_subgraph(seed=1, extra_nodes=10)  # 15 nodes, len > 3
        result = self.planner.evaluate(sg)
        self.assertIsNotNone(result)

    def test_all_rules_fail_returns_none(self):
        planner = HeuristicPlanner(MappingConfig(
            heuristic_rules_enabled=True,
            rule_definitions=[RuleDefinition(condition="1 == 2", plan=[9])]
        ))
        result = planner.evaluate(self._small_subgraph())
        self.assertIsNone(result)

    def test_has_edge_type_true(self):
        env = _RuleEnvironment(self._small_subgraph())
        self.assertTrue(env._has_edge_type("causes"))

    def test_has_edge_type_false(self):
        env = _RuleEnvironment(self._small_subgraph())
        self.assertFalse(env._has_edge_type("nonexistent"))

    def test_average_confidence_computed(self):
        sg = self._small_subgraph()
        env = _RuleEnvironment(sg)
        expected = float(sum(sg.edge_confidences.values()) / len(sg.edge_confidences))
        self.assertAlmostEqual(env._compute_average_confidence(), expected)

    def test_average_confidence_empty(self):
        sg = self._small_subgraph()
        sg.edge_confidences = {}
        env = _RuleEnvironment(sg)
        self.assertEqual(env._compute_average_confidence(), 0.0)

    def test_rule_condition_evaluates_bool(self):
        env = _RuleEnvironment(self._small_subgraph())
        self.assertTrue(env.evaluate("True"))
        self.assertFalse(env.evaluate("False"))

    def test_malformed_condition_caught(self):
        env = _RuleEnvironment(self._small_subgraph())
        result = env.evaluate("this is !!! invalid syntax ???")
        self.assertFalse(result)

    def test_add_rule(self):
        self.planner.add_rule("len(subgraph.nodes) > 100", [8])
        self.assertEqual(len(self.planner._rules), 3)
        self.assertEqual(self.planner._rules[-1].plan, [8])

    def test_clear_rules(self):
        self.planner.clear_rules()
        self.assertEqual(len(self.planner._rules), 0)

    @patch("builtins.open", new_callable=mock_open, read_data='{"rules": [{"condition": "True", "plan": [5]}]}')
    def test_load_rules_from_json(self, mock_file):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            fname = f.name
            f.write('{"rules": [{"condition": "True", "plan": [5]}]}')
        try:
            self.planner.load_rules(fname)
            self.assertEqual(len(self.planner._rules), 1)
            self.assertEqual(self.planner._rules[0].condition, "True")
            self.assertEqual(self.planner._rules[0].plan, [5])
        finally:
            os.remove(fname)

    def test_subgraph_proxy_exposes_nodes_and_edges(self):
        sg = self._small_subgraph()
        proxy = _SubgraphProxy(sg)
        self.assertEqual(proxy.nodes, sg.nodes)
        self.assertEqual(proxy.edges, sg.edges)


# ====================================================================
# G2P PLANNER
# ====================================================================
class TestG2PPlanner(unittest.TestCase):
    def setUp(self):
        self.config = G2PConfig()
        self.config.sentence_bert.model_name = "dummy"
        self.config.sentence_bert.device = "cpu"
        # speed up tests
        self.config.training.epochs = 2
        self.config.training.synthetic_data.num_samples = 10
        self._patcher = patch('sentence_transformers.SentenceTransformer')
        self.mock_st = self._patcher.start()
        self.mock_instance = MagicMock()
        self.mock_instance.encode.return_value = np.random.randn(384).astype(np.float32)
        self.mock_st.return_value = self.mock_instance

    def tearDown(self):
        self._patcher.stop()

    def test_accepts_missing_sentence_transformers(self):
        with patch('builtins.__import__', side_effect=ImportError("no module")):
            pass  # handled in _get_sentence_model

    def test_initializes_submodules(self):
        planner = G2PPlanner(self.config)
        self.assertIsInstance(planner.graph_to_text_encoder, GraphToTextEncoder)
        self.assertIsInstance(planner.intent_ffn, IntentFFN)
        self.assertIsInstance(planner.beam_search, BeamSearchDecoder)
        self.assertIsInstance(planner.heuristic_planner, HeuristicPlanner)

    def test_initialize_loads_model(self):
        planner = G2PPlanner(self.config)
        planner._get_sentence_model()
        self.mock_st.assert_called_once()

    def test_initialize_idempotent(self):
        with patch.object(G2PPlanner, '_precompute_intent_embeddings') as mock_pre:
            planner = G2PPlanner(self.config)
            planner.initialize()
            planner.initialize()
            mock_pre.assert_called_once()

    def test_encode_subgraph_returns_embedding(self):
        planner = G2PPlanner(self.config)
        sg = make_valid_subgraph(extra_nodes=1)
        emb = planner.encode_subgraph(sg)
        self.assertEqual(emb.shape, (384,))
        self.assertEqual(emb.dtype, np.float32)

    def test_encode_subgraph_raises_on_too_many_nodes(self):
        self.config.validation.input_subgraph_max_nodes = 3
        planner = G2PPlanner(self.config)
        sg = make_valid_subgraph(extra_nodes=10)
        with self.assertRaises(ValueError):
            planner.encode_subgraph(sg)

    def test_plan_heuristic_path(self):
        cfg = G2PConfig()
        cfg.mapping.heuristic_rules_enabled = True
        cfg.mapping.rule_definitions = [RuleDefinition(condition="True", plan=[4, 5])]
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        plan = planner.plan(sg)
        self.assertTrue(plan.heuristic_fallback_used)
        self.assertEqual(plan.intent_sequence, [4, 5])
        self.assertEqual(plan.plan_confidence, 0.5)

    def test_plan_ffn_path(self):
        cfg = G2PConfig()
        cfg.mapping.heuristic_rules_enabled = False
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        plan = planner.plan(sg)
        self.assertFalse(plan.heuristic_fallback_used)
        self.assertTrue(1 <= len(plan.intent_sequence) <= 8)
        self.assertTrue(all(0 <= i <= 15 for i in plan.intent_sequence))

    def test_plan_pads_short_sequence(self):
        cfg = G2PConfig()
        cfg.mapping.heuristic_rules_enabled = False
        cfg.validation.output_plan_min_length = 5
        # make beam width broad and temperature high to get varied sequences
        cfg.decoder.beam_width = 1
        cfg.decoder.temperature = 0.001
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        plan = planner.plan(sg)
        self.assertGreaterEqual(len(plan.intent_sequence), 5)
        for i in range(len(plan.intent_sequence)):
            self.assertTrue(0 <= plan.intent_sequence[i] <= 15)

    def test_plan_pads_with_intent_zero(self):
        cfg = G2PConfig()
        cfg.mapping.heuristic_rules_enabled = False
        cfg.validation.output_plan_min_length = 5
        cfg.decoder.max_length = 2
        cfg.decoder.beam_width = 1
        cfg.decoder.temperature = 0.001
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        plan = planner.plan(sg)
        self.assertGreaterEqual(len(plan.intent_sequence), 5)
        padded = plan.intent_sequence[2:]  # after original 2
        for v in padded:
            self.assertEqual(v, 0)

    def test_plan_respects_validation_max_length(self):
        cfg = G2PConfig()
        cfg.mapping.heuristic_rules_enabled = False
        cfg.decoder.max_length = 8
        cfg.validation.output_plan_max_length = 8
        cfg.decoder.beam_width = 1
        cfg.decoder.temperature = 0.001
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        plan = planner.plan(sg)
        self.assertLessEqual(len(plan.intent_sequence), 8)
        self.assertGreaterEqual(len(plan.intent_sequence), 1)

    def test_plan_masks_disallowed_intents(self):
        cfg = G2PConfig()
        cfg.mapping.heuristic_rules_enabled = False
        cfg.validation.allowed_intents = [0, 1]
        cfg.validation.output_plan_max_length = 10
        cfg.decoder.beam_width = 5
        cfg.decoder.max_length = 10
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        plan = planner.plan(sg)
        for i in plan.intent_sequence:
            self.assertIn(i, [0, 1])

    def test_plan_includes_intent_names(self):
        cfg = G2PConfig()
        cfg.mapping.heuristic_rules_enabled = False
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        plan = planner.plan(sg)
        self.assertIsNotNone(plan.intent_names)
        self.assertEqual(len(plan.intent_names), len(plan.intent_sequence))

    def test_plan_batch(self):
        planner = G2PPlanner(self.config)
        sgs = [make_valid_subgraph(extra_nodes=i) for i in range(3)]
        plans = planner.plan_batch(sgs)
        self.assertEqual(len(plans), 3)
        for p in plans:
            self.assertIsInstance(p, Plan)

    def test_get_plan_confidence_heuristic(self):
        cfg = G2PConfig()
        cfg.mapping.rule_definitions = [RuleDefinition(condition="True", plan=[0])]
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        conf = planner.get_plan_confidence(sg)
        self.assertEqual(conf, 0.5)

    def test_get_plan_confidence_ffn(self):
        cfg = G2PConfig()
        cfg.mapping.heuristic_rules_enabled = False
        planner = G2PPlanner(cfg)
        sg = make_valid_subgraph(extra_nodes=1)
        conf = planner.get_plan_confidence(sg)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_get_intent_name_known(self):
        planner = G2PPlanner(self.config)
        self.assertEqual(planner.get_intent_name(0), "define")
        self.assertEqual(planner.get_intent_name(15), "emphasize")

    def test_get_intent_name_unknown(self):
        planner = G2PPlanner(self.config)
        self.assertEqual(planner.get_intent_name(99), "unknown_99")

    def test_intent_to_embedding_returns_cached(self):
        planner = G2PPlanner(self.config)
        planner._intent_embedding_cache[3] = np.ones(384, dtype=np.float32)
        emb = planner.intent_to_embedding(3)
        self.assertTrue(np.allclose(emb, 1.0))

    def test_intent_to_embedding_with_description(self):
        self.config.intent_embeddings[0] = "test description"
        planner = G2PPlanner(self.config)
        emb = planner.intent_to_embedding(0)
        self.assertEqual(emb.shape, (384,))
        self.assertEqual(emb.dtype, np.float32)

    def test_intent_to_embedding_no_description_zeros(self):
        self.config.intent_embeddings = {}
        planner = G2PPlanner(self.config)
        emb = planner.intent_to_embedding(5)
        self.assertTrue(np.allclose(emb, 0.0))

    def test_load_heuristic_rules_delegates(self):
        planner = G2PPlanner(self.config)
        with patch.object(planner.heuristic_planner, 'load_rules') as mock_lr:
            planner.load_heuristic_rules("fake.json")
            mock_lr.assert_called_once_with("fake.json")

    def test_set_label_map_updates_both(self):
        planner = G2PPlanner(self.config)
        planner.set_label_map({1: "one"})
        self.assertEqual(planner.label_map[1], "one")
        self.assertEqual(planner.graph_to_text_encoder.label_map[1], "one")

    def test_config_validated_on_init_with_bad_dim(self):
        bad_cfg = G2PConfig()
        bad_cfg.ffn.input_dim = 128
        with self.assertRaises(AssertionError):
            G2PPlanner(bad_cfg)

    def test_apply_allowed_intents_mask_all_allowed(self):
        planner = G2PPlanner(self.config)
        logits = np.random.randn(16).astype(np.float32)
        masked = planner._apply_allowed_intents_mask(logits)
        self.assertTrue(np.allclose(masked, logits))

    def test_apply_allowed_intents_mask_some_disallowed(self):
        self.config.validation.allowed_intents = [0, 1, 2]
        planner = G2PPlanner(self.config)
        logits = np.random.randn(16).astype(np.float32)
        masked = planner._apply_allowed_intents_mask(logits)
        for i in [0, 1, 2]:
            self.assertEqual(masked[i], logits[i])
        for i in range(3, 16):
            self.assertEqual(masked[i], -1e9)

    def test_precompute_intent_embeddings_caches_values(self):
        self.config.intent_embeddings = {0: "hello", 1: "world"}
        planner = G2PPlanner(self.config)
        planner._precompute_intent_embeddings()
        self.assertIn(0, planner._intent_embedding_cache)
        self.assertIn(1, planner._intent_embedding_cache)
        self.assertNotIn(2, planner._intent_embedding_cache)  # no description


# ====================================================================
# TRAIN
# ====================================================================
class TestGenerateSyntheticData(unittest.TestCase):
    def test_correct_number_of_samples(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.num_samples = 5
        data = generate_synthetic_data(cfg)
        self.assertEqual(len(data), 5)

    def test_each_sample_is_tuple_of_subgraph_and_list(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.num_samples = 3
        cfg.training.synthetic_data.min_nodes_per_graph = 2
        cfg.training.synthetic_data.max_nodes_per_graph = 10
        data = generate_synthetic_data(cfg)
        for sg, intent_seq in data:
            self.assertIsInstance(sg, Subgraph)
            self.assertIsInstance(intent_seq, list)
            sg.validate()

    def test_activations_in_valid_range(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.num_samples = 1
        cfg.training.synthetic_data.min_nodes_per_graph = 5
        data = generate_synthetic_data(cfg)
        sg, _ = data[0]
        for v in sg.node_activations.values():
            self.assertGreaterEqual(v, 0.01)
            self.assertLessEqual(v, 1.0)

    def test_edges_within_node_range(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.num_samples = 1
        cfg.training.synthetic_data.min_nodes_per_graph = 5
        data = generate_synthetic_data(cfg)
        sg, _ = data[0]
        node_set = set(sg.nodes)
        for s, t, _ in sg.edges:
            self.assertIn(s, node_set)
            self.assertIn(t, node_set)

    def test_reproducible_with_seed(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.seed = 42
        cfg.training.synthetic_data.num_samples = 5
        data1 = generate_synthetic_data(cfg)
        data2 = generate_synthetic_data(cfg)
        for (sg1, seq1), (sg2, seq2) in zip(data1, data2):
            self.assertEqual(sg1.nodes, sg2.nodes)
            self.assertEqual(seq1, seq2)

    def test_different_seed_different_data(self):
        cfg1 = G2PConfig()
        cfg1.training.synthetic_data.seed = 1
        cfg1.training.synthetic_data.num_samples = 3
        cfg2 = G2PConfig()
        cfg2.training.synthetic_data.seed = 999
        cfg2.training.synthetic_data.num_samples = 3
        data1 = generate_synthetic_data(cfg1)
        data2 = generate_synthetic_data(cfg2)
        # verify at least one subgraph differs
        differing = False
        for (sg1, _), (sg2, _) in zip(data1, data2):
            if sg1.nodes != sg2.nodes:
                differing = True
                break
        self.assertTrue(differing)

    def test_node_count_in_range(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.num_samples = 10
        cfg.training.synthetic_data.min_nodes_per_graph = 3
        cfg.training.synthetic_data.max_nodes_per_graph = 7
        data = generate_synthetic_data(cfg)
        for sg, _ in data:
            self.assertGreaterEqual(len(sg.nodes), 3)
            self.assertLessEqual(len(sg.nodes), 7)

    def test_intent_sequence_length_in_range(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.num_samples = 10
        data = generate_synthetic_data(cfg)
        for _, intent_seq in data:
            self.assertGreaterEqual(len(intent_seq), 1)
            self.assertLessEqual(len(intent_seq), 4)

    def test_intent_ids_in_valid_range(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.num_samples = 10
        data = generate_synthetic_data(cfg)
        for _, intent_seq in data:
            for i in intent_seq:
                self.assertGreaterEqual(i, 0)
                self.assertLessEqual(i, 15)

    def test_query_embedding_normalized(self):
        cfg = G2PConfig()
        cfg.training.synthetic_data.num_samples = 1
        data = generate_synthetic_data(cfg)
        sg, _ = data[0]
        norm = np.linalg.norm(sg.query_embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestTrainG2P(unittest.TestCase):
    def setUp(self):
        self._patcher = patch('sentence_transformers.SentenceTransformer')
        self.mock_st = self._patcher.start()
        self.mock_instance = MagicMock()
        self.mock_instance.encode.return_value = np.random.randn(384).astype(np.float32)
        self.mock_st.return_value = self.mock_instance

    def tearDown(self):
        self._patcher.stop()

    def test_train_returns_metrics(self):
        cfg = G2PConfig()
        cfg.training.epochs = 2
        cfg.training.synthetic_data.num_samples = 5
        g2p_cfg = G2PConfig()

        # mock config loading
        with patch.object(G2PConfig, 'from_yaml', return_value=g2p_cfg):
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
                fname = f.name
            try:
                data = generate_synthetic_data(g2p_cfg)
                metrics = train_g2p(fname, training_data=data)
                self.assertIn("train_loss", metrics)
                self.assertIn("val_loss", metrics)
                self.assertIn("epochs_trained", metrics)
                self.assertIn("best_val_loss", metrics)
                self.assertGreater(metrics["epochs_trained"], 0)
            finally:
                os.remove(fname)

    def test_train_auto_generates_data(self):
        g2p_cfg = G2PConfig()
        g2p_cfg.training.epochs = 2
        g2p_cfg.training.synthetic_data.num_samples = 5

        with patch.object(G2PConfig, 'from_yaml', return_value=g2p_cfg):
            with patch('G2P.train.generate_synthetic_data', return_value=[]) as mock_gen:
                with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
                    fname = f.name
                try:
                    with self.assertRaises(Exception):
                        train_g2p(fname)
                    mock_gen.assert_called_once()
                finally:
                    os.remove(fname)

    def test_train_with_validation_data(self):
        g2p_cfg = G2PConfig()
        g2p_cfg.training.epochs = 2
        g2p_cfg.training.synthetic_data.num_samples = 5

        with patch.object(G2PConfig, 'from_yaml', return_value=g2p_cfg):
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
                fname = f.name
            try:
                data = generate_synthetic_data(g2p_cfg)
                metrics = train_g2p(fname, training_data=data, validation_data=data[:2])
                self.assertIn("train_loss", metrics)
                self.assertIn("val_loss", metrics)
            finally:
                os.remove(fname)

    def test_train_early_stopping(self):
        g2p_cfg = G2PConfig()
        g2p_cfg.training.epochs = 20
        g2p_cfg.training.synthetic_data.num_samples = 5
        g2p_cfg.training.early_stopping_patience = 2

        with patch.object(G2PConfig, 'from_yaml', return_value=g2p_cfg):
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
                fname = f.name
            try:
                data = generate_synthetic_data(g2p_cfg)
                metrics = train_g2p(fname, training_data=data)
                self.assertLess(metrics["epochs_trained"], 20)
            finally:
                os.remove(fname)

    def test_train_metrics_format(self):
        g2p_cfg = G2PConfig()
        g2p_cfg.training.epochs = 2
        g2p_cfg.training.synthetic_data.num_samples = 5

        with patch.object(G2PConfig, 'from_yaml', return_value=g2p_cfg):
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
                fname = f.name
            try:
                data = generate_synthetic_data(g2p_cfg)
                metrics = train_g2p(fname, training_data=data)
                self.assertIsInstance(metrics["train_loss"], list)
                self.assertIsInstance(metrics["val_loss"], list)
                self.assertIsInstance(metrics["epochs_trained"], int)
                self.assertIsInstance(metrics["best_val_loss"], float)
                self.assertEqual(len(metrics["train_loss"]), metrics["epochs_trained"])
                self.assertEqual(len(metrics["val_loss"]), metrics["epochs_trained"])
            finally:
                os.remove(fname)

    def test_train_label_map(self):
        g2p_cfg = G2PConfig()
        g2p_cfg.training.epochs = 1
        g2p_cfg.training.synthetic_data.num_samples = 3

        with patch.object(G2PConfig, 'from_yaml', return_value=g2p_cfg):
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
                fname = f.name
            try:
                data = generate_synthetic_data(g2p_cfg)
                metrics = train_g2p(fname, training_data=data, label_map={0: "root"})
                self.assertIn("epochs_trained", metrics)
            finally:
                os.remove(fname)


# ====================================================================
# RUNNER
# ====================================================================
def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTests(loader.loadTestsFromTestCase(TestSubgraphValidation))
    s.addTests(loader.loadTestsFromTestCase(TestPlanValidation))
    s.addTests(loader.loadTestsFromTestCase(TestG2PConfigFromYaml))
    s.addTests(loader.loadTestsFromTestCase(TestG2PConfigValidate))
    s.addTests(loader.loadTestsFromTestCase(TestGraphToTextEncoder))
    s.addTests(loader.loadTestsFromTestCase(TestIntentFFN))
    s.addTests(loader.loadTestsFromTestCase(TestBeamSearchDecoder))
    s.addTests(loader.loadTestsFromTestCase(TestHeuristicPlanner))
    s.addTests(loader.loadTestsFromTestCase(TestG2PPlanner))
    s.addTests(loader.loadTestsFromTestCase(TestGenerateSyntheticData))
    s.addTests(loader.loadTestsFromTestCase(TestTrainG2P))
    return s


def run_and_report():
    runner = unittest.TextTestRunner(verbosity=0, stream=io.StringIO())
    s = suite()
    result = runner.run(s)

    summary = {
        "total_tests": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if result.skipped else 0,
        "failures": [
            {
                "test": str(tc).split()[0],
                "message": str(msg).split('\n')[0] if msg else ""
            }
            for tc, msg in result.failures
        ],
        "errors_detail": [
            {
                "test": str(tc).split()[0],
                "message": str(msg).split('\n')[0] if msg else ""
            }
            for tc, msg in result.errors
        ],
    }
    summary["success"] = summary["failed"] == 0 and summary["errors"] == 0
    return summary


if __name__ == "__main__":
    import sys

    if "--json" in sys.argv:
        summary = run_and_report()
        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary["success"] else 1)
    else:
        unittest.main(verbosity=2)
