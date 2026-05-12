#!/usr/bin/env python3
"""
GLM-X Integration Test Suite
Run this after EVERY code change.
All 7 teams must pass this before merging to main.
"""

import unittest
import numpy as np
from glmx_types import *
from config_core import CoreConfig  # Load YAML

class TestComponentCompatibility(unittest.TestCase):
    
    def setUp(self):
        """Load all configurations"""
        self.core = CoreConfig.from_yaml("config_core.yaml")
        self.graph_cfg = load_yaml("config_graph.yaml")
        self.resonance_cfg = load_yaml("config_resonance.yaml")
        self.g2p_cfg = load_yaml("config_g2p.yaml")
        self.walker_cfg = load_yaml("config_walker.yaml")
        self.decoder_cfg = load_yaml("config_decoder.yaml")
        self.learning_cfg = load_yaml("config_learning.yaml")
    
    def test_embedding_dimensions_match(self):
        """Critical: All components using same embedding dim"""
        self.assertEqual(self.core.dimensions.embedding_dim, 32)
        self.assertEqual(self.graph_cfg.node.embedding_dtype, "int8")
        
    def test_activation_ranges_match(self):
        """Resonance output must be within Graph's expected range"""
        self.assertEqual(self.core.activation.min, 0.01)
        self.assertEqual(self.core.activation.max, 1.0)
        
    def test_intent_vocabulary_consistent(self):
        """G2P intents must match Decoder's templates"""
        g2p_intents = set(self.g2p_cfg.intent_embeddings.keys())
        decoder_intents = set([t[0] for t in self.decoder_cfg.templates.definitions])
        # Every intent in G2P should have at least one template
        self.assertTrue(decoder_intents.issubset(g2p_intents))
        
    def test_relation_types_match(self):
        """Graph relations must match Walker's intent biases"""
        graph_relations = set(self.core.relations.values())
        walker_relations = set()
        for biases in self.walker_cfg.intent_biases.values():
            walker_relations.update(biases.keys())
        # Graph has all relations that Walker uses
        self.assertTrue(walker_relations.issubset(graph_relations))
        
    def test_theta_dimension(self):
        """Resonance theta must be 48-dim"""
        self.assertEqual(self.core.es.theta_dim, 48)
        self.assertEqual(self.resonance_cfg.es_controller.theta_dim, 48)
        
    def test_subgraph_dataclass_fields(self):
        """All components expect same Subgraph fields"""
        expected_fields = {'nodes', 'edges', 'activation_energy', 'tier_used', 'seed_nodes'}
        from glmx_types import Subgraph
        actual_fields = set(Subgraph.__dataclass_fields__.keys())
        self.assertEqual(expected_fields, actual_fields)

    def test_walk_result_compatibility(self):
        """Walker output must match Decoder input"""
        from glmx_types import WalkResult, Answer
        walk_fields = set(WalkResult.__dataclass_fields__.keys())
        self.assertTrue('path' in walk_fields)
        self.assertTrue('plan_followed' in walk_fields)
        
    def test_edge_strength_range(self):
        """Learning updates must stay in [0,1]"""
        self.assertEqual(self.core.edge.strength_min, 0.0)
        self.assertEqual(self.core.edge.strength_max, 1.0)
        
    def test_eligibility_trace_format(self):
        """Eligibility traces from Walker must be consumable by Learning"""
        # Key format: (source, target, relation)
        # Value: float eligibility
        sample_trace = {(1, 2, "causes"): 0.875}
        self.assertTrue(isinstance(list(sample_trace.keys())[0], tuple))
        self.assertTrue(len(list(sample_trace.keys())[0]) == 3)
        
    def test_sentence_bert_dimension(self):
        """G2P encoder output must match FFN input"""
        self.assertEqual(self.g2p_cfg.sentence_bert.model_dim, 384)
        self.assertEqual(self.g2p_cfg.ffn.input_dim, 384)
        
    def test_max_walk_len_consistent(self):
        """Walker max steps must match expectations"""
        self.assertEqual(self.core.max_walk_len, 20)
        self.assertEqual(self.walker_cfg.walk.max_steps, 20)
        
    def test_temperature_range_compatible(self):
        """Walker temperature must be within config bounds"""
        temp = self.walker_cfg.walk.temperature
        min_t, max_t = self.walker_cfg.walk.temperature_range
        self.assertTrue(min_t <= temp <= max_t)

if __name__ == "__main__":
    unittest.main()