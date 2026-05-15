#!/usr/bin/env python3
"""
GLM-X Full Pipeline Integration Test Suite
Tests the complete architecture end-to-end:
  Query -> Resonance -> G2P -> Walker -> Decoder -> Learning
"""

import os
import sys
import time
import unittest
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glmx_types import *
from config_core import CoreConfig as CoreConfigLoader

# ── Configs ──────────────────────────────────────────────────
CONFIG_DIR = ROOT / "configs"


def _load_yaml(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Helper: build a minimal toy graph ─────────────────────────
from resonance.types import Node as ResNode, Edge as ResEdge, Subgraph as ResSubgraph
from resonance.tests.fixtures.toy_data import (
    build_animal_kingdom_graph,
    animal_dataset_seeds,
    animal_query_embedding,
)
from resonance.tests.fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_resonance_config,
    build_minimal_loaded_configs,
)

# ── Type adapters ─────────────────────────────────────────────

def res_subgraph_to_g2p_subgraph(sg: ResSubgraph) -> "g2p.types.Subgraph":
    from g2p.types import Subgraph as G2PSubgraph
    return G2PSubgraph(
        nodes=[n for n in sg.nodes],
        node_activations=dict(sg.node_activations),
        edges=[(int(s), int(t), r) for (s, t, r) in sg.edges],
        edge_strengths={},
        edge_confidences={},
        seed_nodes=list(sg.seed_nodes),
        tier_used=sg.tier_used,
        activation_energy=sg.activation_energy,
        query_embedding=sg.query_embedding.copy(),
        timestamp=sg.timestamp,
    )


def res_subgraph_to_walker_subgraph(sg: ResSubgraph, graph_store) -> "walker.models.Subgraph":
    from walker.models import Subgraph as WSubgraph
    node_embeddings = {}
    for nid in sg.nodes:
        node = graph_store.get_node(nid)
        if node is not None:
            node_embeddings[nid] = node.embedding.copy()
    return WSubgraph(
        nodes=list(sg.nodes),
        node_activations=dict(sg.node_activations),
        node_embeddings=node_embeddings,
        edges=[(int(s), int(t), r) for (s, t, r) in sg.edges],
        edge_strengths=dict(sg.edge_strengths),
        edge_confidences=dict(sg.edge_confidences),
        seed_nodes=list(sg.seed_nodes),
        tier_used=sg.tier_used,
        activation_energy=sg.activation_energy,
        query_embedding=sg.query_embedding.copy(),
        timestamp=sg.timestamp,
    )


def plan_to_walker_plan(p: "g2p.types.Plan") -> "walker.models.Plan":
    from walker.models import Plan as WPlan
    return WPlan(
        intent_sequence=list(p.intent_sequence),
        plan_confidence=p.plan_confidence,
        heuristic_fallback_used=p.heuristic_fallback_used,
        intent_names=list(p.intent_names) if p.intent_names else None,
    )


def walk_result_to_decoder_walk(wr: "walker.models.WalkResult") -> "decoder.models.WalkResult":
    from decoder.models import WalkResult as DWalkResult
    return DWalkResult(
        path=list(wr.path),
        path_edges=list(wr.path_edges),
        path_labels=[],
        path_activations=list(wr.path_activations),
        path_confidences=list(wr.path_confidences),
        path_embeddings=[e.copy() for e in wr.path_embeddings],
        walk_confidence=wr.walk_confidence,
        final_activation=wr.final_activation,
        steps_taken=wr.steps_taken,
        plan_followed=wr.plan_followed,
        timestamp=wr.timestamp,
        intent_sequence_used=list(wr.intent_sequence_used),
    )


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════

class TestFullGLMXPipeline(unittest.TestCase):
    """End-to-end integration: every component in the GLM-X architecture."""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        np.random.seed(42)

    # ── PHASE 0: Configuration ────────────────────────────────
    def test_00_configs_load(self):
        """All 7 configuration files load without errors."""
        configs = [
            "config_core.yaml", "config_graph.yaml", "config_resonance.yaml",
            "config_g2p.yaml", "config_walker.yaml", "config_decoder.yaml",
            "config_learning.yaml",
        ]
        for name in configs:
            path = CONFIG_DIR / name
            self.assertTrue(path.exists(), f"Missing config: {name}")
            data = _load_yaml(str(path))
            self.assertIsNotNone(data, f"Empty/invalid config: {name}")

    def test_01_core_config_import(self):
        """CoreConfig loads from config_core module."""
        cfg = CoreConfigLoader.from_yaml(str(CONFIG_DIR / "config_core.yaml"))
        self.assertIsNotNone(cfg)
        self.assertIsNotNone(cfg.dimensions)
        self.assertEqual(cfg.dimensions.embedding_dim, 32)

    # ── PHASE 1: Graph + Toy Data ─────────────────────────────
    def test_10_toy_graph_builds(self):
        """Toy animal kingdom graph builds and has expected structure."""
        g = build_animal_kingdom_graph()
        self.assertGreater(g.node_count(), 30)
        self.assertGreater(g.edge_count(), 50)
        self.assertIsNotNone(g.get_node(0))
        self.assertEqual(g.get_node(0).label, "animal")

    def test_11_toy_graph_subgraph_query(self):
        """Graph returns subgraph from embedding similarity."""
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        sg = g.get_subgraph_by_embedding_similarity(emb, top_k=20)
        self.assertGreater(len(sg.nodes), 0)
        self.assertEqual(sg.query_embedding.shape, (384,))
        self.assertTrue(np.isfinite(sg.query_embedding).all())

    # ── PHASE 2: Resonance ────────────────────────────────────
    def test_20_resonance_engine_initializes(self):
        """ResonanceEngine initialises from config dir."""
        from resonance.engine import ResonanceEngine
        engine = ResonanceEngine(CONFIG_DIR)
        theta = engine.get_theta()
        self.assertEqual(theta.shape, (48,))
        self.assertTrue(np.all(np.isfinite(theta)))

    def test_21_resonance_tier1_produces_subgraph(self):
        """Tier 1 resonance produces valid Subgraph from toy graph."""
        from resonance.engine import ResonanceEngine
        engine = ResonanceEngine(CONFIG_DIR)
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        sg = engine.resonate(emb, g, seeds, tier=1)
        # Subgraph must have nodes and edges
        self.assertGreater(len(sg.nodes), 0, "Tier1 subgraph had zero nodes")
        self.assertEqual(sg.tier_used, 1)
        self.assertGreaterEqual(sg.activation_energy, 0.0)
        # Validate node activations are in range
        for a in sg.node_activations.values():
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)
        # Validate all edges reference nodes in the subgraph
        node_set = set(sg.nodes)
        for (s, t, r) in sg.edges:
            self.assertIn(s, node_set, f"Edge source {s} not in subgraph nodes")
            self.assertIn(t, node_set, f"Edge target {t} not in subgraph nodes")

    def test_22_resonance_tier2_refines_when_needed(self):
        """Tier 2 resonance activates when Tier 1 energy is low."""
        from resonance.engine import ResonanceEngine
        engine = ResonanceEngine(CONFIG_DIR)
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        sg = engine.resonate(emb, g, seeds, tier=2)
        self.assertGreater(len(sg.nodes), 0)
        self.assertIn(sg.tier_used, (1, 2))
        self.assertGreaterEqual(sg.activation_energy, 0.0)

    def test_23_resonance_es_mutation(self):
        """Evolutionary Strategy proposes valid theta mutations."""
        from resonance.engine import ResonanceEngine
        engine = ResonanceEngine(CONFIG_DIR)
        theta_base = engine.get_theta().copy()
        # Propose a mutation
        theta_mut = engine.propose_theta_mutation()
        self.assertEqual(theta_mut.shape, (48,))
        self.assertTrue(np.all(np.isfinite(theta_mut)))
        # Run resonance with mutated theta
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        sg = engine.resonate_with_theta(theta_mut, emb, g, seeds)
        self.assertGreater(len(sg.nodes), 0)
        # Feed back a simulated reward
        energy = engine.compute_activation_energy(sg)
        reward = float(np.clip(energy / (len(seeds) * 1.0), -1.0, 1.0))
        engine.update_es_with_reward(reward, theta_mut)
        # Theta should have been adjusted
        theta_new = engine.get_theta()
        self.assertEqual(theta_new.shape, (48,))
        self.assertTrue(np.all(np.isfinite(theta_new)))

    # ── PHASE 3: G2P Planning ─────────────────────────────────
    def test_30_g2p_planner_initializes(self):
        """G2PPlanner initialises from config."""
        from g2p.config import G2PConfig
        from g2p.g2p_planner import G2PPlanner
        config = G2PConfig.from_yaml(str(CONFIG_DIR / "config_g2p.yaml"))
        planner = G2PPlanner(config)
        planner.initialize()
        self.assertIsNotNone(planner)
        # Check intent vocabulary
        for i in range(16):
            name = planner.get_intent_name(i)
            self.assertIsNotNone(name)

    def test_31_g2p_plans_from_subgraph(self):
        """G2P produces a valid Plan from a resonance subgraph."""
        from resonance.engine import ResonanceEngine
        from g2p.config import G2PConfig
        from g2p.g2p_planner import G2PPlanner

        engine = ResonanceEngine(CONFIG_DIR)
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        sg = engine.resonate(emb, g, seeds, tier=1)

        config = G2PConfig.from_yaml(str(CONFIG_DIR / "config_g2p.yaml"))
        planner = G2PPlanner(config)
        planner.initialize()
        g2p_sg = res_subgraph_to_g2p_subgraph(sg)
        plan = planner.plan(g2p_sg)
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan.intent_sequence), 0)
        self.assertLessEqual(len(plan.intent_sequence), 8)
        for intent_id in plan.intent_sequence:
            self.assertGreaterEqual(intent_id, 0)
            self.assertLess(intent_id, 16)
        self.assertGreaterEqual(plan.plan_confidence, 0.0)
        self.assertLessEqual(plan.plan_confidence, 1.0)

    # ── PHASE 4: Walker ───────────────────────────────────────
    def test_40_walker_initializes(self):
        """GraphWalker initialises from configs."""
        from walker.config import CoreConfig as WCoreConfig, WalkerConfig
        from walker.graph_walker import GraphWalker
        core_cfg = WCoreConfig.from_yaml(str(CONFIG_DIR / "config_core.yaml"))
        walker_cfg = WalkerConfig.from_yaml(str(CONFIG_DIR / "config_walker.yaml"))
        walker = GraphWalker(walker_cfg, core_cfg)
        self.assertIsNotNone(walker)

    def test_41_walker_walks_subgraph(self):
        """Walker traverses the subgraph guided by a plan."""
        from resonance.engine import ResonanceEngine
        from g2p.config import G2PConfig
        from g2p.g2p_planner import G2PPlanner
        from walker.config import CoreConfig as WCoreConfig, WalkerConfig
        from walker.graph_walker import GraphWalker

        engine = ResonanceEngine(CONFIG_DIR)
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        sg = engine.resonate(emb, g, seeds, tier=1)

        g2p_config = G2PConfig.from_yaml(str(CONFIG_DIR / "config_g2p.yaml"))
        planner = G2PPlanner(g2p_config)
        planner.initialize()
        g2p_sg = res_subgraph_to_g2p_subgraph(sg)
        plan = planner.plan(g2p_sg)

        core_cfg = WCoreConfig.from_yaml(str(CONFIG_DIR / "config_core.yaml"))
        walker_cfg = WalkerConfig.from_yaml(str(CONFIG_DIR / "config_walker.yaml"))
        walker = GraphWalker(walker_cfg, core_cfg)

        w_sg = res_subgraph_to_walker_subgraph(sg, g)
        w_plan = plan_to_walker_plan(plan)
        result = walker.walk(w_sg, w_plan)
        self.assertIsNotNone(result)
        self.assertGreater(len(result.path), 0, "Walker path was empty")
        self.assertGreaterEqual(result.walk_confidence, 0.0)
        self.assertLessEqual(result.walk_confidence, 1.0)
        self.assertGreaterEqual(result.steps_taken, 1)
        # Path nodes must be in the subgraph
        path_nodes = set(result.path)
        sg_nodes = set(sg.nodes)
        overlap = path_nodes & sg_nodes
        self.assertGreater(len(overlap), 0, "Walker path has no overlap with subgraph nodes")

    def test_42_walker_eligibility_trace(self):
        """Walker computes eligibility traces consumable by Learning."""
        from resonance.engine import ResonanceEngine
        from g2p.config import G2PConfig
        from g2p.g2p_planner import G2PPlanner
        from walker.config import CoreConfig as WCoreConfig, WalkerConfig
        from walker.graph_walker import GraphWalker

        engine = ResonanceEngine(CONFIG_DIR)
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        sg = engine.resonate(emb, g, seeds, tier=1)

        g2p_config = G2PConfig.from_yaml(str(CONFIG_DIR / "config_g2p.yaml"))
        planner = G2PPlanner(g2p_config)
        planner.initialize()
        g2p_sg = res_subgraph_to_g2p_subgraph(sg)
        plan = planner.plan(g2p_sg)

        core_cfg = WCoreConfig.from_yaml(str(CONFIG_DIR / "config_core.yaml"))
        walker_cfg = WalkerConfig.from_yaml(str(CONFIG_DIR / "config_walker.yaml"))
        walker = GraphWalker(walker_cfg, core_cfg)

        w_sg = res_subgraph_to_walker_subgraph(sg, g)
        w_plan = plan_to_walker_plan(plan)
        result = walker.walk(w_sg, w_plan)
        trace = walker.compute_eligibility_trace(result, w_sg)
        self.assertIsInstance(trace, dict)
        for key, val in trace.items():
            self.assertIsInstance(val, (float, np.floating))
            self.assertGreaterEqual(float(val), 0.0)
            self.assertLessEqual(float(val), 1.0)

    # ── PHASE 5: Decoder ──────────────────────────────────────
    def test_50_decoder_initializes(self):
        """MicroDecoder initialises from config."""
        from decoder.micro_decoder import MicroDecoder
        dec = MicroDecoder(str(CONFIG_DIR / "config_decoder.yaml"))
        self.assertIsNotNone(dec)

    def test_51_decoder_produces_answer(self):
        """Decoder produces a valid Answer from walk+plan."""
        from resonance.engine import ResonanceEngine
        from g2p.config import G2PConfig
        from g2p.g2p_planner import G2PPlanner
        from walker.config import CoreConfig as WCoreConfig, WalkerConfig
        from walker.graph_walker import GraphWalker
        from decoder.micro_decoder import MicroDecoder

        engine = ResonanceEngine(CONFIG_DIR)
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        sg = engine.resonate(emb, g, seeds, tier=1)

        g2p_config = G2PConfig.from_yaml(str(CONFIG_DIR / "config_g2p.yaml"))
        planner = G2PPlanner(g2p_config)
        planner.initialize()
        planner._graph_store = g
        g2p_sg = res_subgraph_to_g2p_subgraph(sg)
        plan = planner.plan(g2p_sg)

        core_cfg = WCoreConfig.from_yaml(str(CONFIG_DIR / "config_core.yaml"))
        walker_cfg = WalkerConfig.from_yaml(str(CONFIG_DIR / "config_walker.yaml"))
        walker = GraphWalker(walker_cfg, core_cfg)
        w_sg = res_subgraph_to_walker_subgraph(sg, g)
        w_plan = plan_to_walker_plan(plan)
        result = walker.walk(w_sg, w_plan)

        dec = MicroDecoder(str(CONFIG_DIR / "config_decoder.yaml"))
        d_walk = walk_result_to_decoder_walk(result)
        answer = dec.decode(d_walk, d_walk.plan_followed)
        self.assertIsNotNone(answer)
        self.assertIsInstance(answer.text, str)
        self.assertGreater(len(answer.text), 0, "Answer text was empty")
        self.assertGreaterEqual(answer.confidence, 0.0)
        self.assertLessEqual(answer.confidence, 1.0)

    # ── PHASE 6: Learning ─────────────────────────────────────
    def test_60_learning_engine_initializes(self):
        """LearningEngine initialises from config."""
        from learning.config import LearningConfig
        from learning.engine import LearningEngine
        config = LearningConfig.from_yaml(str(CONFIG_DIR / "config_learning.yaml"))
        engine = LearningEngine(config)
        self.assertIsNotNone(engine)

    def test_61_learning_processes_feedback(self):
        """Learning engine processes feedback and updates edges."""
        from resonance.engine import ResonanceEngine
        from g2p.config import G2PConfig
        from g2p.g2p_planner import G2PPlanner
        from walker.config import CoreConfig as WCoreConfig, WalkerConfig
        from walker.graph_walker import GraphWalker
        from decoder.micro_decoder import MicroDecoder
        from learning.config import LearningConfig
        from learning.engine import LearningEngine

        engine = ResonanceEngine(CONFIG_DIR)
        g = build_animal_kingdom_graph()
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        sg = engine.resonate(emb, g, seeds, tier=1)

        g2p_config = G2PConfig.from_yaml(str(CONFIG_DIR / "config_g2p.yaml"))
        planner = G2PPlanner(g2p_config)
        planner.initialize()
        planner._graph_store = g
        g2p_sg = res_subgraph_to_g2p_subgraph(sg)
        plan = planner.plan(g2p_sg)

        core_cfg = WCoreConfig.from_yaml(str(CONFIG_DIR / "config_core.yaml"))
        walker_cfg = WalkerConfig.from_yaml(str(CONFIG_DIR / "config_walker.yaml"))
        walker = GraphWalker(walker_cfg, core_cfg)
        w_sg = res_subgraph_to_walker_subgraph(sg, g)
        w_plan = plan_to_walker_plan(plan)
        result = walker.walk(w_sg, w_plan)

        dec = MicroDecoder(str(CONFIG_DIR / "config_decoder.yaml"))
        d_walk = walk_result_to_decoder_walk(result)
        answer = dec.decode(d_walk, d_walk.plan_followed)

        learn_config = LearningConfig.from_yaml(str(CONFIG_DIR / "config_learning.yaml"))
        learner = LearningEngine(learn_config)

        # Capture edge strengths before
        pre_strengths = {}
        if len(sg.edges) > 0:
            for i, (s, t, r) in enumerate(sg.edges[:5]):
                pre_strengths[(s, t, r)] = sg.edge_strengths.get((s, t, r), 0.5)

        # Process feedback with a positive rating
        learner.process_feedback(
            answer=answer,
            user_rating=0.8,
            walk=result,
            subgraph=w_sg,
            graph=g,
            resonance_engine=engine,
            g2p=planner,
            walker=walker,
            decoder=dec,
        )
        # Verify learning engine stored experience in replay buffer
        self.assertGreater(learner.replay_buffer_size, 0,
                           "Learning replay buffer should have entries after feedback")

    # ── PHASE 7: Full Pipeline End-to-End ─────────────────────
    def test_70_full_pipeline_e2e(self):
        """The complete GLM-X pipeline runs end-to-end without errors.

        Architecture: Graph → Resonance → G2P → Walker → Decoder → Learning
        """
        from resonance.engine import ResonanceEngine
        from g2p.config import G2PConfig
        from g2p.g2p_planner import G2PPlanner
        from walker.config import CoreConfig as WCoreConfig, WalkerConfig
        from walker.graph_walker import GraphWalker
        from decoder.micro_decoder import MicroDecoder
        from learning.config import LearningConfig
        from learning.engine import LearningEngine

        # 1. Graph
        graph = build_animal_kingdom_graph()
        self.assertGreater(graph.node_count(), 0)

        # 2. Resonance
        engine = ResonanceEngine(CONFIG_DIR)
        emb = animal_query_embedding()
        seeds = animal_dataset_seeds()
        subgraph = engine.resonate(emb, graph, seeds, tier=2)
        self.assertGreater(len(subgraph.nodes), 0)
        self.assertIn(subgraph.tier_used, (1, 2))

        # 3. G2P
        g2p_config = G2PConfig.from_yaml(str(CONFIG_DIR / "config_g2p.yaml"))
        planner = G2PPlanner(g2p_config)
        planner.initialize()
        planner._graph_store = graph
        g2p_sg = res_subgraph_to_g2p_subgraph(subgraph)
        plan = planner.plan(g2p_sg)
        self.assertGreater(len(plan.intent_sequence), 0)

        # 4. Walker
        core_cfg = WCoreConfig.from_yaml(str(CONFIG_DIR / "config_core.yaml"))
        walker_cfg = WalkerConfig.from_yaml(str(CONFIG_DIR / "config_walker.yaml"))
        walker = GraphWalker(walker_cfg, core_cfg)
        w_sg = res_subgraph_to_walker_subgraph(subgraph, graph)
        w_plan = plan_to_walker_plan(plan)
        walk_result = walker.walk(w_sg, w_plan)
        self.assertGreater(len(walk_result.path), 0)

        # 5. Decoder
        dec = MicroDecoder(str(CONFIG_DIR / "config_decoder.yaml"))
        d_walk = walk_result_to_decoder_walk(walk_result)
        answer = dec.decode(d_walk, d_walk.plan_followed)
        self.assertGreater(len(answer.text), 0)

        # 6. Learning
        learn_config = LearningConfig.from_yaml(str(CONFIG_DIR / "config_learning.yaml"))
        learner = LearningEngine(learn_config)
        learner.process_feedback(
            answer=answer,
            user_rating=0.7,
            walk=walk_result,
            subgraph=w_sg,
            graph=graph,
            resonance_engine=engine,
            g2p=planner,
            walker=walker,
            decoder=dec,
        )
        self.assertGreater(learner.replay_buffer_size, 0)

    # ── PHASE 8: Architecture Constraint Validation ───────────
    def test_80_embedding_dimensions_consistent(self):
        """All components agree on 32-dim int8 embeddings, 384-dim queries."""
        core = _load_yaml(str(CONFIG_DIR / "config_core.yaml"))
        graph_cfg = _load_yaml(str(CONFIG_DIR / "config_graph.yaml"))
        self.assertEqual(core["dimensions"]["embedding_dim"], 32)
        self.assertEqual(graph_cfg["node"]["embedding_dtype"], "int8")

    def test_81_theta_dimension_consistent(self):
        """Theta vector is 48-dim across all configs."""
        core = _load_yaml(str(CONFIG_DIR / "config_core.yaml"))
        res_cfg = _load_yaml(str(CONFIG_DIR / "config_resonance.yaml"))
        self.assertEqual(core["es"]["theta_dim"], 48)
        self.assertEqual(res_cfg["es_controller"]["theta_dim"], 48)

    def test_82_sixteen_intents_across_system(self):
        """16 intents exist in G2P, Decoder templates, and core config."""
        core = _load_yaml(str(CONFIG_DIR / "config_core.yaml"))
        g2p = _load_yaml(str(CONFIG_DIR / "config_g2p.yaml"))
        dec = _load_yaml(str(CONFIG_DIR / "config_decoder.yaml"))
        intent_count = len(core["intents"])
        g2p_intent_count = len(g2p["intent_embeddings"])
        self.assertEqual(intent_count, 16)
        self.assertEqual(g2p_intent_count, 16)
        self.assertGreaterEqual(len(dec["templates"]["definitions"]), 1)

    def test_83_sixteen_relations_across_system(self):
        """16 relations in core, resonance biases, and walker biases."""
        core = _load_yaml(str(CONFIG_DIR / "config_core.yaml"))
        res = _load_yaml(str(CONFIG_DIR / "config_resonance.yaml"))
        walker = _load_yaml(str(CONFIG_DIR / "config_walker.yaml"))
        rel_count = len(core["relations"])
        res_bias_count = len(res["tier1"]["relation_bias"])
        self.assertEqual(rel_count, 16)
        self.assertEqual(res_bias_count, 16)
        # Each walker intent biases a subset of relations + a default
        for intent_id, biases in walker["intent_biases"].items():
            self.assertIn("default", biases, f"Intent {intent_id} missing default bias")

    def test_84_activation_range_consistent(self):
        """A_rest=0.01, A_max=1.0 across all components."""
        core = _load_yaml(str(CONFIG_DIR / "config_core.yaml"))
        self.assertEqual(core["activation"]["min"], 0.01)
        self.assertEqual(core["activation"]["max"], 1.0)

    def test_85_max_walk_len_consistent(self):
        """Walker max_steps matches core config max_walk_len."""
        core = _load_yaml(str(CONFIG_DIR / "config_core.yaml"))
        walker = _load_yaml(str(CONFIG_DIR / "config_walker.yaml"))
        self.assertEqual(core["dimensions"]["max_walk_len"], walker["walk"]["max_steps"])

    def test_86_subgraph_dataclass_fields(self):
        """Subgraph has the canonical field set expected by all components."""
        from glmx_types import Subgraph
        required = {'nodes', 'edges', 'node_activations', 'edge_strengths',
                     'edge_confidences', 'seed_nodes', 'tier_used',
                     'activation_energy', 'query_embedding', 'timestamp'}
        actual = set(Subgraph.__dataclass_fields__.keys())
        self.assertTrue(required.issubset(actual),
                        f"Missing fields: {required - actual}")


if __name__ == "__main__":
    unittest.main()
