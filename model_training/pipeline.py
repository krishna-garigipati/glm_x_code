"""GLM-X Pipeline: clean production wrapper that imports and orchestrates all 6 components.
No component logic duplicated — pure integration with type adapters."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from resonance.engine import ResonanceEngine
from resonance.types import Subgraph as ResonanceSubgraph
from g2p.g2p_planner import G2PPlanner
from g2p.config import G2PConfig
from g2p.types import Subgraph as G2PSubgraph, Plan as G2PPlan
from walker.graph_walker import GraphWalker
from walker.config import CoreConfig as WalkerCoreConfig, WalkerConfig
from walker.models import Subgraph as WalkerSubgraph, Plan as WalkerPlan, WalkResult
from decoder.template_decoder import TemplateDecoder
from decoder.t5_decoder import T5Decoder
from learning.engine import LearningEngine
from learning.config import LearningConfig
from learning.types import Answer as LearningAnswer, WalkResult as LearningWalkResult
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

logger = logging.getLogger(__name__)

TEST_QUERIES: List[Tuple[str, str]] = [
    ("What did Albert Einstein develop?", "theory of relativity"),
    ("What did Marie Curie discover?", "radium"),
    ("What did Isaac Newton formulate?", "laws of motion"),
    ("Who developed the polio vaccine?", "Jonas Salk"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is Paris in?", "France"),
    ("Where is Tokyo?", "Japan"),
    ("What is London in?", "England"),
    ("What causes lung cancer?", "Smoking"),
    ("What does exercise cause?", "good health"),
    ("What is the CPU part of?", "computer"),
    ("What is the heart part of?", "circulatory system"),
    ("What is a dog?", "animal"),
    ("What is a rose?", "flower"),
    ("What originated in Ethiopia?", "Coffee"),
    ("What originated in China?", "Paper"),
    ("What is the opposite of hot?", "cold"),
    ("What is the opposite of light?", "darkness"),
    ("What did Isaac Newton discover?", "laws of motion"),
    ("Where is the Nile river?", "Egypt"),
    ("What does smoking cause?", "lung cancer"),
    ("What is a diamond?", "gemstone"),
    ("What is in France?", "Paris"),
    ("What is in Japan?", "Tokyo"),
]


class TypeAdapter:
    """Converts between component-specific type definitions.
    Every component defines its own Subgraph/Plan/WalkResult with identical fields.
    These functions copy field-by-field across type boundaries."""

    @staticmethod
    def resonance_to_g2p(sg: ResonanceSubgraph) -> G2PSubgraph:
        return G2PSubgraph(
            nodes=list(sg.nodes),
            node_activations=dict(sg.node_activations),
            edges=list(sg.edges),
            edge_strengths=dict(sg.edge_strengths),
            edge_confidences=dict(sg.edge_confidences),
            seed_nodes=list(sg.seed_nodes),
            tier_used=sg.tier_used,
            activation_energy=sg.activation_energy,
            query_embedding=sg.query_embedding.copy(),
            timestamp=sg.timestamp,
        )

    @staticmethod
    def resonance_to_walker(
        sg: ResonanceSubgraph,
        node_embeddings: Optional[Dict[int, np.ndarray]] = None,
    ) -> WalkerSubgraph:
        return WalkerSubgraph(
            nodes=list(sg.nodes),
            node_activations=dict(sg.node_activations),
            edges=list(sg.edges),
            edge_strengths=dict(sg.edge_strengths),
            edge_confidences=dict(sg.edge_confidences),
            seed_nodes=list(sg.seed_nodes),
            tier_used=sg.tier_used,
            activation_energy=sg.activation_energy,
            query_embedding=sg.query_embedding.copy(),
            timestamp=sg.timestamp,
            node_embeddings=node_embeddings,
        )

    @staticmethod
    def g2p_plan_to_walker(plan: G2PPlan) -> WalkerPlan:
        return WalkerPlan(
            intent_sequence=list(plan.intent_sequence),
            plan_confidence=plan.plan_confidence,
            heuristic_fallback_used=plan.heuristic_fallback_used,
            intent_names=list(plan.intent_names) if plan.intent_names else None,
        )

    @staticmethod
    def walker_to_learning(walk: WalkResult) -> LearningWalkResult:
        return LearningWalkResult(
            path=list(walk.path),
            path_edges=list(walk.path_edges),
            path_activations=list(walk.path_activations),
            path_confidences=list(walk.path_confidences),
            path_embeddings=list(walk.path_embeddings),
            walk_confidence=walk.walk_confidence,
            final_activation=walk.final_activation,
            steps_taken=walk.steps_taken,
            plan_followed=walk.plan_followed,
            timestamp=walk.timestamp,
            intent_sequence_used=list(walk.intent_sequence_used),
        )


class Pipeline:
    """Production-grade GLM-X pipeline wiring all 6 components.
    
    Usage:
        pipeline = Pipeline(config_dir="configs")
        pipeline.initialize()
        pipeline.load_graph("kg/graph.db")
        result = pipeline.answer("What is Paris in?")
    """

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.graph_store: Optional[SQLiteGraphStore] = None
        self.resonance: Optional[ResonanceEngine] = None
        self.planner: Optional[G2PPlanner] = None
        self.walker: Optional[GraphWalker] = None
        self.template_decoder: Optional[TemplateDecoder] = None
        self.t5_decoder: Optional[T5Decoder] = None
        self.learning: Optional[LearningEngine] = None

    def initialize(self) -> None:
        """Initialize all 6 components from config files in config_dir."""
        logger.info("Initializing GLM-X Pipeline from %s", self.config_dir)

        # 1. Resonance Engine
        self.resonance = ResonanceEngine(self.config_dir)
        logger.info("ResonanceEngine initialized")

        # 2. G2P Planner
        g2p_config = G2PConfig.from_yaml(str(self.config_dir / "config_g2p.yaml"))
        self.planner = G2PPlanner(g2p_config)
        self.planner.initialize()
        logger.info("G2PPlanner initialized")

        # 3. Graph Walker
        walker_config = WalkerConfig.from_yaml(str(self.config_dir / "config_walker.yaml"))
        walker_core = WalkerCoreConfig.from_yaml(str(self.config_dir / "config_core.yaml"))
        self.walker = GraphWalker(walker_config, walker_core)
        logger.info("GraphWalker initialized")

        # 4. Decoder — TemplateDecoder + T5Decoder (used independently, not HybridDecoder)
        with open(self.config_dir / "config_decoder.yaml") as f:
            decoder_raw = yaml.safe_load(f)
        self.template_decoder = TemplateDecoder(
            templates=decoder_raw["templates"]["definitions"],
            relation_phrases=decoder_raw["templates"]["relation_phrases"],
            sentence_starters=decoder_raw["templates"]["sentence_starters"],
            fallback_cfg=decoder_raw["fallback"],
            validation_cfg=decoder_raw["validation"],
        )
        t5_cfg = decoder_raw.get("t5", {})
        self.t5_decoder = T5Decoder(
            model_name=t5_cfg.get("model_name", "t5-small"),
            max_input_length=t5_cfg.get("max_input_length", 512),
            max_output_length=t5_cfg.get("max_output_length", 128),
            num_beams=t5_cfg.get("num_beams", 4),
            temperature=t5_cfg.get("temperature", 0.7),
            top_p=t5_cfg.get("top_p", 0.9),
            repetition_penalty=t5_cfg.get("repetition_penalty", 1.2),
            do_sample=t5_cfg.get("do_sample", True),
            fallback_cfg=decoder_raw.get("fallback"),
            validation_cfg=decoder_raw.get("validation"),
        )
        logger.info("TemplateDecoder + T5Decoder initialized")

        # 5. Learning Engine
        self.learning = LearningEngine(LearningConfig())
        logger.info("LearningEngine initialized")

    def load_graph(self, db_path: str) -> None:
        """Load a SQLiteGraphStore and configure dependent components."""
        self.graph_store = SQLiteGraphStore(db_path)
        label_map = {nid: node.label for nid, node in self.graph_store._nodes.items()}
        self.planner.set_label_map(label_map)
        logger.info(
            "Graph loaded: %d nodes, %d edges, %d embeddings",
            self.graph_store.get_node_count(),
            self.graph_store.get_edge_count(),
            len(self.graph_store._embeddings),
        )

    def answer(self, question: str) -> Dict[str, Any]:
        """Full inference pipeline: embed → graph query → resonate → plan → walk → decode."""
        if self.graph_store is None:
            raise RuntimeError("No graph loaded. Call load_graph() first.")

        store = self.graph_store
        t_start = time.time()

        # 1. Embed query
        q_emb = self.sbert.encode(question, normalize_embeddings=True)

        # 2. Query graph for seed nodes
        seed_sg = store.get_subgraph_by_embedding_similarity(q_emb, top_k=20)
        if not seed_sg.seed_nodes:
            return {
                "question": question,
                "answer": "",
                "confidence": 0.0,
                "template_matched": False,
                "time_seconds": round(time.time() - t_start, 3),
            }

        # 3. Resonance → amplify subgraph
        resonated = self.resonance.resonate(q_emb, store, seed_sg.seed_nodes)

        # 4. G2P → plan
        g2p_sg = TypeAdapter.resonance_to_g2p(resonated)
        plan = self.planner.plan(g2p_sg, query_text=question)

        # 5. Walk → path
        node_embs = {nid: store.get_embedding(nid) for nid in resonated.nodes
                     if store.get_embedding(nid) is not None}
        walker_sg = TypeAdapter.resonance_to_walker(resonated, node_embeddings=node_embs)
        walker_plan = TypeAdapter.g2p_plan_to_walker(plan)
        walk_result = self.walker.walk(walker_sg, walker_plan)

        # 6. Decode → natural language (template first, T5 fallback)
        path_labels = [store.get_label(nid) for nid in walk_result.path]
        text, template_ok = self.template_decoder.decode(
            node_labels=path_labels,
            relation_labels=walk_result.path_edges,
            intents=walk_result.intent_sequence_used,
        )
        confidence = 1.0 if template_ok else 0.0
        if not template_ok:
            text, t5_ok = self.t5_decoder.decode(
                node_labels=path_labels,
                relation_labels=walk_result.path_edges,
                intents=walk_result.intent_sequence_used,
            )

        # 7. Apply walker reward (implicit learning)
        self.walker.apply_reward(
            reward=1.0 if template_ok else -0.3,
            path_edges=walk_result.path_edges,
            path_intents=walk_result.intent_sequence_used,
            learning_rate=0.01,
        )

        elapsed = round(time.time() - t_start, 3)
        return {
            "question": question,
            "answer": text,
            "confidence": float(confidence),
            "template_matched": template_ok,
            "plan_confidence": float(plan.plan_confidence),
            "walk_confidence": float(walk_result.walk_confidence),
            "plan_intents": list(plan.intent_sequence),
            "path_labels": path_labels,
            "path_edges": list(walk_result.path_edges),
            "time_seconds": elapsed,
        }

    def evaluate(self, queries: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """Run QA evaluation against a set of query→expected pairs."""
        queries = queries or TEST_QUERIES
        correct, total = 0, 0
        results = []
        for q, exp in queries:
            total += 1
            try:
                r = self.answer(q)
                answer_text = r.get("answer", "")
                path_text = " ".join(r.get("path_labels", []))
                combined = (answer_text + " " + path_text).lower()
                ok = exp.lower() in combined
                if ok:
                    correct += 1
                results.append({"question": q, "answer": answer_text, "expected": exp, "correct": ok})
            except Exception as e:
                logger.warning("QA failed for %r: %s", q, e)
                results.append({"question": q, "answer": "", "expected": exp, "correct": False})
        accuracy = correct / total * 100 if total else 0.0
        logger.info("QA evaluation: %d/%d correct (%.1f%%)", correct, total, accuracy)
        return {
            "qa_correct": correct,
            "qa_total": total,
            "qa_accuracy": accuracy,
            "per_question": results,
        }
