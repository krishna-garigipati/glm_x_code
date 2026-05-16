#!/usr/bin/env python
"""
GLM-X Full Pipeline: resonance -> planner -> walker -> decoder

All components are wired together and every inch is used at inference.

Flow:
  question -> SBERT encode -> GraphStore.get_subgraph() -> seed Subgraph
  -> Tier1Resonance.resonate() -> activated Subgraph (activations propagate)
  -> G2PPlanner.plan() -> intent Plan
  -> GraphWalker.walk() -> WalkResult (ordered path through graph)
  -> TemplateDecoder.decode() -> final answer text
"""

import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("glmx")

from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
import torch

from graph.graph_component_implementation.dict_graph_store import DictGraphStore

from resonance.tier1 import Tier1Resonance
from resonance.config import (
    CoreConfig, CoreActivationConfig, CoreResonanceConfig, ESBounds,
    AlgorithmConfig, TierConfig, TemporalConfig,
)

from g2p.g2p_planner import G2PPlanner
from g2p.types import Subgraph as G2PSubgraph, Plan as G2PPlan
from g2p.config import G2PConfig

from walker.config import CoreConfig as WalkerCoreConfig, WalkerConfig, ActivationConfig, WalkerCoreConfig as WCore, load_yaml as load_walker_yaml
from walker.models import Plan as WalkerPlan, Subgraph as WalkerSubgraph, WalkResult
from walker.graph_walker import GraphWalker

from decoder.template_decoder import TemplateDecoder
from decoder.config_loader import load_config as load_decoder_config

from model_training.dataset_conceptnet.relation_map import CONCEPTNET_RELATION_MAP, INTENT_VOCAB

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "model_training" / "dataset_conceptnet" / "conceptnet" / "data"
CONFIG_PATH = BASE_DIR / "model_training" / "config.yaml"
DECODER_CONFIG_PATH = BASE_DIR / "decoder" / "config_decoder.yaml"
SAVED_MODELS_DIR = BASE_DIR / "model_training" / "saved_models"
WALKER_CONFIG_PATH = BASE_DIR / "configs" / "config_walker.yaml"
CORE_CONFIG_PATH = BASE_DIR / "configs" / "config_core.yaml"

RELATION_SHARDS = {"Synonym": [8], "RelatedTo": [5, 6, 7], "Antonym": [0]}
CONCEPTNET_RELATION_MAP_CFG = {
    "Synonym": "synonym",
    "Antonym": "antonym",
    "RelatedTo": "associated_with",
}
REVERSE_RELATION_MAP = {v: k for k, v in CONCEPTNET_RELATION_MAP_CFG.items()}

INTENT_NAMES = {
    0: "define", 1: "assert_fact", 2: "explain_cause", 3: "explain_effect",
    4: "contrast", 5: "compare", 6: "list", 7: "example",
    8: "conclude", 9: "question", 10: "uncertain", 11: "clarify",
    12: "summarize", 13: "elaborate", 14: "transition", 15: "emphasize",
}


def concept_label(uri: str) -> str:
    parts = uri.strip("/").split("/")
    name_idx = 5 if len(parts) > 5 else 4
    return parts[name_idx].replace("_", " ") if len(parts) > name_idx else uri


def concept_lang(uri: str) -> str:
    parts = uri.strip("/").split("/")
    return parts[4] if len(parts) >= 5 and parts[0] == "http:" else ""


def load_conceptnet(max_edges_per_rel: int = 2000) -> DictGraphStore:
    """Load ConceptNet parquet into DictGraphStore."""
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    index_to_file = {int(f.stem.split("-")[1]): f for f in parquet_files}

    concepts: Dict[str, int] = {}
    edges = []
    next_id = 1
    seen_pairs = set()

    for rel_name, shard_indices in RELATION_SHARDS.items():
        needed = max_edges_per_rel
        for sidx in shard_indices:
            if needed <= 0:
                break
            fpath = index_to_file.get(sidx)
            if fpath is None:
                continue
            pf = pq.ParquetFile(fpath)
            for gi in range(pf.metadata.num_row_groups):
                if needed <= 0:
                    break
                tbl = pf.read_row_groups([gi], columns=["subject", "predicate", "object"])
                for row in tbl.to_pylist():
                    hl = concept_lang(row["subject"])
                    tl = concept_lang(row["object"])
                    if hl != "en" or tl != "en":
                        continue
                    h = concept_label(row["subject"])
                    t = concept_label(row["object"])
                    pair = (h, rel_name, t)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    for c in (h, t):
                        if c not in concepts:
                            concepts[c] = next_id
                            next_id += 1
                    edges.append({
                        "source": concepts[h],
                        "target": concepts[t],
                        "relation": rel_name,
                        "strength": 0.9,
                        "confidence": 0.8,
                    })
                    needed -= 1
                    if needed <= 0:
                        break

    id_to_label = {v: k for k, v in concepts.items()}
    logger.info(f"Loaded {len(concepts)} concepts, {len(edges)} edges")

    store = DictGraphStore()
    store.add_dataset(
        concepts, edges, id_to_label,
        relation_map=CONCEPTNET_RELATION_MAP_CFG,
    )
    logger.info(f"GraphStore ready: {store.get_node_count()} nodes, {store.get_edge_count()} edges")
    return store


def make_resonance_configs() -> Tuple[CoreConfig, AlgorithmConfig, TemporalConfig, TierConfig]:
    """Build resonance configs programmatically for our 3 relation types."""
    core = CoreConfig(
        activation=CoreActivationConfig(min=0.01, max=1.0, default=0.01, threshold_resonance=0.2),
        resonance=CoreResonanceConfig(
            propagation_threshold=0.008, edge_threshold=0.02, decay_lambda=0.1,
            top_k=64, budget_max=2.0, convergence_epsilon=0.001,
            tier1_energy_threshold=0.4, tier2_max_nodes=1024, analogy_validation_overlap=0.3,
        ),
        relations=["synonym", "antonym", "associated_with"],
        es_bounds=ESBounds(
            propagation_threshold=(0.001, 0.05), edge_threshold=(0.01, 0.1),
            decay_lambda=(0.05, 0.5), top_k=(16, 1024), relation_bias=(0.0, 2.0),
        ),
    )

    algorithm = AlgorithmConfig(
        propagation_type="wilson_cowan", normalization="budget_soft_cap",
        gate_type="top_k", temporal_factor_enabled=False,
    )

    temporal = TemporalConfig(gamma=0.5, frequency_threshold=20)

    tier = TierConfig(
        propagation_threshold=0.008, edge_threshold=0.02, decay_lambda=0.1,
        top_k=64, max_iterations=3,
        relation_bias={
            "synonym": 1.0,
            "antonym": 0.4,
            "associated_with": 0.5,
        },
        energy_threshold_formula=None, t_conf_coefficient=None, multiplied_at_runtime=None,
    )

    return core, algorithm, temporal, tier


class GLMXPipeline:
    """Full GLM-X pipeline using every component."""

    def __init__(self):
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")

        self.graph_store: Optional[DictGraphStore] = None
        self.tier1: Optional[Tier1Resonance] = None
        self.planner: Optional[G2PPlanner] = None
        self.walker: Optional[GraphWalker] = None
        self.decoder: Optional[TemplateDecoder] = None

    def load_graph(self, max_edges_per_rel: int = 2000) -> None:
        self.graph_store = load_conceptnet(max_edges_per_rel)

        labels = list(self.graph_store._label_to_id.keys())
        logger.info(f"Computing embeddings for {len(labels)} concepts...")
        embs = self.sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
        for label, emb in zip(labels, embs):
            nid = self.graph_store._label_to_id[label]
            self.graph_store._embeddings[nid] = emb
            node = self.graph_store._nodes[nid]
            self.graph_store._nodes[nid] = type(node)(
                id=node.id, label=node.label, node_type=node.node_type,
                embedding=emb, activation=node.activation,
                use_count=node.use_count, create_time=node.create_time,
            )
        logger.info("Embeddings computed and stored")

    def load_models(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = str(SAVED_MODELS_DIR / "conceptnet" / "intent_ffn_best.pt")

        # ---- Tier1 Resonance ----
        core_cfg, algo_cfg, temporal_cfg, tier_cfg = make_resonance_configs()
        self.tier1 = Tier1Resonance(
            core_config=core_cfg, algorithm=algo_cfg, temporal=temporal_cfg,
            tier_config=tier_cfg, log_activation_history=True, history_buffer_size=10,
        )
        logger.info("Tier1Resonance initialized (wilson_cowan, top_k=64, 3 iterations)")

        # ---- G2P Planner ----
        self.planner = G2PPlanner(G2PConfig.from_yaml(str(CONFIG_PATH)))
        self.planner.initialize()
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            old_state = checkpoint["model_state_dict"]
            new_state = self.planner.intent_ffn.state_dict()
            for k in new_state:
                if k in old_state and old_state[k].shape == new_state[k].shape:
                    new_state[k] = old_state[k]
            self.planner.intent_ffn.load_state_dict(new_state)
            logger.info(f"Loaded IntentFFN from {model_path}")
        else:
            logger.warning(f"No trained model at {model_path}")

        # ---- Graph Walker ----
        wc_raw = load_walker_yaml(str(CORE_CONFIG_PATH))
        ww_raw = load_walker_yaml(str(WALKER_CONFIG_PATH))
        walker_core_cfg = WalkerCoreConfig(
            activation=ActivationConfig(
                min=float(wc_raw.get("activation", {}).get("min", 0.01)),
                max=float(wc_raw.get("activation", {}).get("max", 1.0)),
            ),
            walker=WCore(
                default_temperature=float(wc_raw.get("walker", {}).get("default_temperature", 0.1)),
                softmax_temperature_range=tuple(
                    wc_raw.get("walker", {}).get("softmax_temperature_range", [0.05, 0.5])
                ),
                max_steps=int(wc_raw.get("walker", {}).get("max_steps", 20)),
                min_activation_to_continue=float(
                    wc_raw.get("walker", {}).get("min_activation_to_continue", 0.05)
                ),
            ),
            relations={int(k): v for k, v in wc_raw.get("relations", {}).items()},
        )
        walker_cfg = WalkerConfig.from_yaml(str(WALKER_CONFIG_PATH))
        self.walker = GraphWalker(walker_cfg, walker_core_cfg)
        logger.info("GraphWalker initialized")

        # ---- Template Decoder ----
        decoder_cfg = load_decoder_config(str(DECODER_CONFIG_PATH))
        self.decoder = TemplateDecoder(
            templates=decoder_cfg.get("templates", {}).get("definitions", []),
            relation_phrases=decoder_cfg.get("templates", {}).get("relation_phrases", {}),
            sentence_starters=decoder_cfg.get("templates", {}).get("sentence_starters", []),
            fallback_cfg=decoder_cfg.get("fallback", {}),
            validation_cfg=decoder_cfg.get("validation", {}),
        )
        logger.info("TemplateDecoder loaded")

        logger.info("All components initialized. GLM-X pipeline ready.")

    def subgraph_to_text(self, subgraph) -> str:
        lines = [f"Concepts ({len(subgraph.nodes)}):"]
        for nid in subgraph.nodes:
            label = self.graph_store.get_label(nid)
            act = subgraph.node_activations.get(nid, 0)
            lines.append(f"  [{nid}] {label} (act={act:.4f})")
        lines.append(f"Relations ({len(subgraph.edges)}):")
        for s, t, r in sorted(set(subgraph.edges)):
            sl = self.graph_store.get_label(s)
            tl = self.graph_store.get_label(t)
            orig_r = REVERSE_RELATION_MAP.get(r, r)
            lines.append(f"  {sl} --[{orig_r}]--> {tl}")
        return "\n".join(lines)

    def walk_result_to_text(self, walk: WalkResult) -> str:
        lines = [f"Walk ({len(walk.path)} nodes, {len(walk.path_edges)} steps):"]
        for i, nid in enumerate(walk.path):
            label = self.graph_store.get_label(nid)
            act = walk.path_activations[i] if walk.path_activations else 0
            if i == 0:
                lines.append(f"  START -> [{nid}] {label} (act={act:.4f})")
            else:
                edge = walk.path_edges[i - 1]
                conf = walk.path_confidences[i - 1] if walk.path_confidences else 0
                orig_r = REVERSE_RELATION_MAP.get(edge, edge)
                lines.append(f"  --[{orig_r}] (conf={conf:.2f})-> [{nid}] {label} (act={act:.4f})")
        return "\n".join(lines)

    def ask(self, question: str) -> Dict[str, Any]:
        t0 = time.time()
        steps_log: Dict[str, float] = {}

        # ===== STEP 1: Embed question =====
        ts = time.time()
        q_emb = self.sbert.encode(question, normalize_embeddings=True)
        steps_log["1_encode"] = round(time.time() - ts, 3)

        # ===== STEP 2: Get seed subgraph from graph store =====
        ts = time.time()
        seed_sub = self.graph_store.get_subgraph_by_embedding_similarity(q_emb, top_k=5)
        steps_log["2_subgraph"] = round(time.time() - ts, 3)
        logger.info(f"[2/6] GraphStore returned {len(seed_sub.nodes)} nodes, {len(seed_sub.edges)} edges")

        # ===== STEP 3: Tier1Resonance propagation =====
        ts = time.time()
        resonated, history = self.tier1.resonate(q_emb, self.graph_store, seed_sub.seed_nodes)
        steps_log["3_resonance"] = round(time.time() - ts, 3)
        logger.info(f"[3/6] Tier1 resonance: {len(resonated.nodes)} nodes, "
                    f"energy={resonated.activation_energy:.4f}, "
                    f"tier={resonated.tier_used}, {len(history)} iterations")

        # ===== STEP 4: G2P Planner -> intent sequence =====
        ts = time.time()
        plan = self.planner.plan(resonated, query_text=question)
        steps_log["4_plan"] = round(time.time() - ts, 3)
        logger.info(f"[4/6] Plan: intents={plan.intent_sequence} "
                    f"({', '.join(plan.intent_names)}), "
                    f"heuristic={plan.heuristic_fallback_used}, "
                    f"confidence={plan.plan_confidence:.4f}")

        # ===== STEP 5: Graph Walker =====
        ts = time.time()
        node_embeddings = {}
        for nid in resonated.nodes:
            emb = self.graph_store.get_embedding(nid)
            if emb is not None:
                node_embeddings[nid] = emb

        walker_sub = WalkerSubgraph(
            nodes=resonated.nodes,
            node_activations=resonated.node_activations,
            edges=resonated.edges,
            edge_strengths=resonated.edge_strengths,
            edge_confidences=resonated.edge_confidences,
            seed_nodes=resonated.seed_nodes,
            tier_used=resonated.tier_used,
            activation_energy=resonated.activation_energy,
            query_embedding=resonated.query_embedding,
            timestamp=resonated.timestamp,
            node_embeddings=node_embeddings,
        )
        walker_plan = WalkerPlan(
            intent_sequence=plan.intent_sequence,
            plan_confidence=plan.plan_confidence,
            heuristic_fallback_used=plan.heuristic_fallback_used,
            intent_names=plan.intent_names,
        )

        walk = self.walker.walk(walker_sub, walker_plan)
        steps_log["5_walk"] = round(time.time() - ts, 3)
        logger.info(f"[5/6] Walker: {walk.steps_taken} steps, "
                    f"confidence={walk.walk_confidence:.4f}, "
                    f"path={[self.graph_store.get_label(n) for n in walk.path]}")

        # ===== STEP 6: Template Decoder =====
        ts = time.time()
        node_labels = [self.graph_store.get_label(n) for n in walk.path]
        edge_labels = list(walk.path_edges)

        answer, template_ok = self.decoder.decode(node_labels, edge_labels, plan.intent_sequence)

        if not template_ok:
            try:
                answer = self.decoder.fallback(node_labels, edge_labels, plan.intent_sequence)
            except Exception:
                fallback = " ".join(node_labels[:5])
                starter = self.decoder._select_sentence_starter(plan.intent_sequence)
                answer = f"{starter} {fallback}" if starter else fallback

        steps_log["6_decode"] = round(time.time() - ts, 3)
        logger.info(f"[6/6] Decoder: template_ok={template_ok}, "
                    f"answer_len={len(answer)}, "
                    f"answer_start={answer[:60]!r}")

        elapsed = time.time() - t0
        subgraph_text = self.subgraph_to_text(resonated)
        walk_text = self.walk_result_to_text(walk)

        # Build intent explanation
        intent_details = []
        for i, (intent_id, name) in enumerate(zip(plan.intent_sequence, plan.intent_names)):
            if i < len(walk.path_edges):
                n0 = self.graph_store.get_label(walk.path[i])
                n1 = self.graph_store.get_label(walk.path[i + 1])
                edge_r = REVERSE_RELATION_MAP.get(walk.path_edges[i], walk.path_edges[i])
                intent_details.append(f"  step {i}: intent={name}({intent_id}) "
                                      f"follow {n0} --[{edge_r}]--> {n1}")
            else:
                intent_details.append(f"  step {i}: intent={name}({intent_id}) [no more path edges]")

        return {
            "question": question,
            "answer": answer,
            "plan_intents": plan.intent_sequence,
            "plan_names": plan.intent_names,
            "heuristic_used": plan.heuristic_fallback_used,
            "template_matched": template_ok,
            "confidence": float(plan.plan_confidence),
            "walk_confidence": float(walk.walk_confidence),
            "time_seconds": round(elapsed, 2),
            "steps_timing": steps_log,
            "n_resonated_nodes": len(resonated.nodes),
            "n_resonated_edges": len(resonated.edges),
            "resonance_energy": round(resonated.activation_energy, 4),
            "n_walk_steps": walk.steps_taken,
            "walk_path_labels": [self.graph_store.get_label(n) for n in walk.path],
            "walk_path_edges": [REVERSE_RELATION_MAP.get(e, e) for e in walk.path_edges],
            "walk_path_activations": [round(a, 4) for a in walk.path_activations],
            "intent_details": intent_details,
            "subgraph": subgraph_text,
            "walk_path": walk_text,
        }


    def save_checkpoint(self, path: str) -> None:
        """Save full model checkpoint: graph + IntentFFN + config references."""
        import json, os, shutil
        os.makedirs(path, exist_ok=True)

        graph_dir = os.path.join(path, "graph_store")
        self.graph_store.save_state(graph_dir)

        model_src = str(SAVED_MODELS_DIR / "conceptnet" / "intent_ffn_best.pt")
        if os.path.exists(model_src):
            shutil.copy2(model_src, os.path.join(path, "intent_ffn.pt"))

        checkpoint_manifest = {
            "model": "GLM-X",
            "version": "1.0.0",
            "graph": {
                "nodes": self.graph_store.get_node_count(),
                "edges": self.graph_store.get_edge_count(),
                "relations": self.graph_store.get_all_relations(),
            },
            "configs": {
                "g2p_config": str(CONFIG_PATH),
                "decoder_config": str(DECODER_CONFIG_PATH),
                "walker_config": str(WALKER_CONFIG_PATH),
                "core_config": str(CORE_CONFIG_PATH),
            },
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        }
        with open(os.path.join(path, "checkpoint.json"), "w") as f:
            json.dump(checkpoint_manifest, f, indent=2)

        logger.info("Full checkpoint saved to %s", path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "GLMXPipeline":
        """Load full model checkpoint."""
        import json, os
        pipeline = cls()

        graph_dir = os.path.join(path, "graph_store")
        if os.path.exists(graph_dir):
            pipeline.graph_store = DictGraphStore.load_state(graph_dir)
        else:
            pipeline.load_graph()

        model_path = os.path.join(path, "intent_ffn.pt")
        pipeline.load_models(model_path=model_path if os.path.exists(model_path) else None)

        if pipeline.graph_store is None:
            pipeline.load_graph()

        return pipeline


def main():
    pipeline = GLMXPipeline()
    pipeline.load_graph(max_edges_per_rel=2000)
    pipeline.load_models()
    logger.info("\n" + "=" * 70)
    logger.info("GLM-X PIPELINE READY")
    logger.info("Resonance -> G2P Planner -> Graph Walker -> Template Decoder")
    logger.info("=" * 70 + "\n")

    questions = [
        "What is the opposite of hot?",
        "Tell me something related to water",
        "What is a dog?",
    ]

    for q in questions:
        result = pipeline.ask(q)
        print("\n" + "=" * 70)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"\n  Intent Sequence: {result['plan_intents']} ({', '.join(result['plan_names'])})")
        print(f"  Heuristic: {result['heuristic_used']}")
        print(f"  Template matched: {result['template_matched']}")
        print(f"  Plan confidence: {result['confidence']:.4f}")
        print(f"  Walk confidence: {result['walk_confidence']:.4f}")
        print(f"  Walk steps: {result['n_walk_steps']}")
        print(f"  Walk path: {' -> '.join(result['walk_path_labels'])}")
        print(f"  Walk edges: {result['walk_path_edges']}")
        print(f"  Timing: {result['steps_timing']}")
        print(f"  Total: {result['time_seconds']}s")
        print(f"\n  ---- Intent-guided Walk Details ----")
        for line in result['intent_details']:
            print(f"  {line}")
        print(f"\n  ---- Resonance Subgraph ({result['n_resonated_nodes']} nodes, {result['n_resonated_edges']} edges) ----")
        print(f"  Energy: {result['resonance_energy']}")
        print(result['walk_path'])


if __name__ == "__main__":
    main()
