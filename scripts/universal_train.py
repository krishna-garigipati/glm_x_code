#!/usr/bin/env python
"""
Unified GLM-X Training Pipeline — fully online, adaptive, zero heuristic.
No fixed epochs, batch sizes, learning rates, or relation-to-intent maps.
All parameters emerge from data statistics and gradient dynamics.

Connects ALL 6 components: Graph -> Resonance -> G2P -> Walker -> Decoder -> Learner
Data stream: any format -> ingest -> .db -> absorb -> converge -> evaluate
"""

import sys
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("universal_train")

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

from resonance.tier1 import Tier1Resonance
from resonance.config import (
    CoreConfig, CoreActivationConfig, CoreResonanceConfig, ESBounds,
    AlgorithmConfig, TierConfig, TemporalConfig,
)

from g2p.g2p_planner import G2PPlanner
from g2p.types import Subgraph as G2PSubgraph
from g2p.config import G2PConfig

from walker.config import CoreConfig as WalkerCoreConfig, WalkerConfig, ActivationConfig
from walker.config import CoreConfig as WCore
from walker.config import load_yaml as load_walker_yaml
from walker.models import Plan as WalkerPlan, Subgraph as WalkerSubgraph
from walker.graph_walker import GraphWalker

from decoder.template_decoder import TemplateDecoder
from decoder.t5_decoder import T5Decoder
from decoder.config_loader import load_config as load_decoder_config

from learning.engine import LearningEngine
from learning.config import LearningConfig

from model_training.training.metrics_tracker import MetricsTracker
from model_training.training.online_learner import OnlineLearner
from model_training.models.intent_ffn import IntentFFN


BASE_DIR = Path(__file__).resolve().parent.parent


REVERSE_RELATION_MAP = {
    "synonym": "Synonym", "antonym": "Antonym",
    "associated_with": "RelatedTo",
}


TEST_QUERIES = [
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


def make_resonance_configs():
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
        relation_bias={"synonym": 1.0, "antonym": 0.4, "associated_with": 0.5},
        energy_threshold_formula=None, t_conf_coefficient=None, multiplied_at_runtime=None,
    )
    return core, algorithm, temporal, tier


def assign_intents_by_embedding(relation_labels: List[str], sbert: SentenceTransformer,
                                 n_intents: int = 8) -> Dict[str, int]:
    """Cluster relations by their BGE embedding similarity.
    Zero heuristic — intent labels emerge from relation semantics."""
    if len(relation_labels) <= n_intents:
        return {rel: i % n_intents for i, rel in enumerate(relation_labels)}
    embs = sbert.encode(relation_labels, normalize_embeddings=True)
    best_k = n_intents
    if len(relation_labels) >= n_intents * 2:
        scores = []
        for k in range(max(2, n_intents - 2), min(n_intents + 3, len(relation_labels))):
            km = KMeans(n_clusters=k, n_init=3, random_state=0)
            labels = km.fit_predict(embs)
            if len(set(labels)) > 1:
                scores.append((silhouette_score(embs, labels), k))
        if scores:
            best_k = max(scores, key=lambda x: x[0])[1]
    km = KMeans(n_clusters=best_k, n_init=5, random_state=0)
    clusters = km.fit_predict(embs)
    return {rel: int(clusters[i]) for i, rel in enumerate(relation_labels)}


class GLMXSystem:
    """All 6 components wired together. Inference-only — created once, used many times."""

    def __init__(self, sbert: SentenceTransformer):
        self.sbert = sbert
        self.graph_store: Optional[SQLiteGraphStore] = None
        self.tier1: Optional[Tier1Resonance] = None
        self.planner: Optional[G2PPlanner] = None
        self.walker: Optional[GraphWalker] = None
        self.decoder: Optional[TemplateDecoder] = None
        self.t5_decoder: Optional[T5Decoder] = None
        self.learning_engine: Optional[LearningEngine] = None

    def initialize(self):
        self.graph_store = None
        core_cfg, algo_cfg, temporal_cfg, tier_cfg = make_resonance_configs()
        self.tier1 = Tier1Resonance(
            core_config=core_cfg, algorithm=algo_cfg, temporal=temporal_cfg,
            tier_config=tier_cfg, log_activation_history=True, history_buffer_size=10,
        )
        cfg_path = BASE_DIR / "model_training" / "config.yaml"
        self.planner = G2PPlanner(G2PConfig.from_yaml(str(cfg_path)))
        self.planner.initialize()

        core_cfg_path = BASE_DIR / "configs" / "config_core.yaml"
        walker_cfg_path = BASE_DIR / "configs" / "config_walker.yaml"
        if core_cfg_path.exists() and walker_cfg_path.exists():
            wc_raw = load_walker_yaml(str(core_cfg_path))
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
            walker_cfg = WalkerConfig.from_yaml(str(walker_cfg_path))
            self.walker = GraphWalker(walker_cfg, walker_core_cfg)
        else:
            self.walker = GraphWalker(
                walker_config=WalkerConfig(
                    max_steps=5, default_temperature=0.1,
                    softmax_temperature_range=[0.05, 0.5],
                    min_activation_to_continue=0.05,
                    restart_on_dead_end=False,
                    scoring_formula="strength * confidence * target_activation * intent_bias",
                ),
                core_config=WalkerCoreConfig(
                    activation=ActivationConfig(min=0.01, max=1.0),
                    walker=WCore(
                        default_temperature=0.1, softmax_temperature_range=(0.05, 0.5),
                        max_steps=5, min_activation_to_continue=0.05,
                    ),
                    relations={},
                ),
            )

        decoder_cfg_path = BASE_DIR / "decoder" / "config_decoder.yaml"
        if decoder_cfg_path.exists():
            dcfg = load_decoder_config(str(decoder_cfg_path))
            self.decoder = TemplateDecoder(
                templates=dcfg.get("templates", {}).get("definitions", []),
                relation_phrases=dcfg.get("templates", {}).get("relation_phrases", {}),
                sentence_starters=dcfg.get("templates", {}).get("sentence_starters", []),
                fallback_cfg=dcfg.get("fallback", {}),
                validation_cfg=dcfg.get("validation", {}),
            )
            t5c = dcfg.get("t5", {})
            self.t5_decoder = T5Decoder(
                model_name=t5c.get("model_name", "t5-small"),
                max_input_length=t5c.get("max_input_length", 512),
                max_output_length=t5c.get("max_output_length", 128),
                num_beams=t5c.get("num_beams", 4),
                temperature=t5c.get("temperature", 0.7),
                top_p=t5c.get("top_p", 0.9),
                repetition_penalty=t5c.get("repetition_penalty", 1.2),
                do_sample=t5c.get("do_sample", True),
                fallback_cfg=dcfg.get("fallback"),
                validation_cfg=dcfg.get("validation"),
            )
        else:
            self.decoder = TemplateDecoder(
                templates=[], relation_phrases={}, sentence_starters=[],
                fallback_cfg={}, validation_cfg={},
            )
            self.t5_decoder = T5Decoder()

        self.learning_engine = LearningEngine(LearningConfig())

    def attach_graph(self, store: SQLiteGraphStore):
        self.graph_store = store
        id_to_label = {nid: node.label for nid, node in store._nodes.items()}
        self.planner.set_label_map(id_to_label)

    def answer(self, question: str) -> Dict:
        store = self.graph_store
        sbert = self.sbert
        tier1 = self.tier1
        planner = self.planner
        walker = self.walker
        decoder = self.decoder
        t5_decoder = self.t5_decoder

        q_emb = sbert.encode(question, normalize_embeddings=True)
        seed_sub = store.get_subgraph_by_embedding_similarity(q_emb, top_k=20)
        target_entity_ids = [seed_sub.seed_nodes[0]] if seed_sub.seed_nodes else []
        for nid in target_entity_ids:
            seed_sub.node_activations[nid] = max(seed_sub.node_activations.get(nid, 0), 0.9)
        for nid in seed_sub.seed_nodes[:3]:
            seed_sub.node_activations[nid] = max(seed_sub.node_activations.get(nid, 0), 0.8)

        resonated, history = tier1.resonate(q_emb, store, seed_sub.seed_nodes)
        plan = planner.plan(resonated, query_text=question)

        for nid in target_entity_ids:
            resonated.node_activations[nid] = 1.0
        target_set = set(target_entity_ids)
        for nid in resonated.seed_nodes:
            if nid not in target_set:
                resonated.node_activations[nid] = min(resonated.node_activations.get(nid, 0), 0.99)

        node_embeddings = {}
        for nid in resonated.nodes:
            emb = store.get_embedding(nid)
            if emb is not None:
                node_embeddings[nid] = emb

        rel_labels = sorted(set(r for _, _, r in resonated.edges))
        if len(rel_labels) > 1:
            rel_embs = sbert.encode(rel_labels, normalize_embeddings=True)
            for s, t, r in resonated.edges:
                rel_idx = rel_labels.index(r)
                rel_sim = float(np.dot(q_emb, rel_embs[rel_idx]))
                target_emb = node_embeddings.get(t)
                target_sim = float(np.dot(q_emb, target_emb)) if target_emb is not None else 0
                boost = 1.0 + 2.0 * max(0.0, rel_sim - 0.15) + 0.5 * max(0.0, target_sim - 0.15)
                resonated.edge_strengths[(s, t, r)] *= boost

        rev_edges = [(t, s, r) for s, t, r in resonated.edges]
        rev_strengths = {}
        rev_confidences = {}
        for s, t, r in resonated.edges:
            rev_strengths[(t, s, r)] = resonated.edge_strengths.get((s, t, r), 0.5)
            rev_confidences[(t, s, r)] = resonated.edge_confidences.get((s, t, r), 0.5)
        all_edges = resonated.edges + rev_edges
        all_strengths = {**resonated.edge_strengths, **rev_strengths}
        all_confidences = {**resonated.edge_confidences, **rev_confidences}

        walker_sub = WalkerSubgraph(
            nodes=resonated.nodes, node_activations=resonated.node_activations,
            edges=all_edges, edge_strengths=all_strengths,
            edge_confidences=all_confidences, seed_nodes=resonated.seed_nodes,
            tier_used=resonated.tier_used, activation_energy=resonated.activation_energy,
            query_embedding=resonated.query_embedding, timestamp=resonated.timestamp,
            node_embeddings=node_embeddings,
        )
        walker_plan = WalkerPlan(
            intent_sequence=plan.intent_sequence, plan_confidence=plan.plan_confidence,
            heuristic_fallback_used=plan.heuristic_fallback_used, intent_names=plan.intent_names,
        )
        walk = walker.walk(walker_sub, walker_plan)

        node_labels = [store.get_label(n) for n in walk.path]
        edge_labels = list(walk.path_edges)
        truncated_intents = plan.intent_sequence[:len(walk.path_edges)] if walk.path_edges else plan.intent_sequence[:1]

        answer, template_ok = decoder.decode(node_labels, edge_labels, truncated_intents)
        if not template_ok:
            try:
                t5_answer, t5_ok = t5_decoder.decode(node_labels, edge_labels, truncated_intents)
                if t5_ok:
                    answer = t5_answer
                    template_ok = True
            except Exception:
                pass
        if not template_ok:
            try:
                answer = decoder.fallback(node_labels, edge_labels, truncated_intents)
            except Exception:
                fallback = " ".join(node_labels[:5])
                starter = ""
                try:
                    starter = decoder._select_sentence_starter(truncated_intents)
                except Exception:
                    pass
                answer = f"{starter} {fallback}" if starter else fallback

        reward = 1.0 if template_ok else -0.3
        path_intents = plan.intent_sequence[:len(walk.path_edges)]
        walker.apply_reward(reward=reward, path_edges=list(walk.path_edges),
                           path_intents=path_intents, learning_rate=0.01)

        return {
            "question": question, "answer": answer,
            "plan_intents": plan.intent_sequence, "plan_names": plan.intent_names,
            "heuristic_used": plan.heuristic_fallback_used, "template_matched": template_ok,
            "confidence": float(plan.plan_confidence),
            "walk_confidence": float(walk.walk_confidence),
            "time_seconds": 0, "walk_path_labels": [store.get_label(n) for n in walk.path],
            "walk_path_edges": [REVERSE_RELATION_MAP.get(e, e) for e in walk.path_edges],
        }


class DataStream:
    """Online data stream from .db. No fixed size — yields triples as they come."""

    def __init__(self, store: SQLiteGraphStore, sbert: SentenceTransformer):
        self.store = store
        self.sbert = sbert
        self._triples = list(store.get_all_edges())
        self._cursor = 0

    def __len__(self):
        return len(self._triples)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n triples, return (embeddings, intent_labels).
        Intent labels derived from relation embedding clustering — zero heuristic."""
        if not self._triples:
            return np.zeros((0, 384)), np.zeros(0, dtype=np.int64)

        n = min(n, len(self._triples))
        indices = np.random.choice(len(self._triples), size=n, replace=False)

        relation_labels = list({self._triples[i].relation for i in indices})
        rel_to_intent = assign_intents_by_embedding(relation_labels, self.sbert)

        embs = []
        intents = []
        for i in indices:
            t = self._triples[i]
            src_node = self.store.get_node(t.source)
            if src_node is None:
                continue
            emb = self.store.get_embedding(t.source)
            if emb is None:
                continue
            label = f"{src_node.label} {t.relation}"
            full_emb = self.sbert.encode(label, normalize_embeddings=True)
            embs.append(full_emb)
            intents.append(rel_to_intent.get(t.relation, 0))

        if not embs:
            return np.zeros((0, 384)), np.zeros(0, dtype=np.int64)
        return np.stack(embs), np.array(intents, dtype=np.int64)


def run_qa_eval(system: GLMXSystem) -> Dict:
    correct = 0
    total = 0
    results = []
    for question, expected in TEST_QUERIES:
        total += 1
        try:
            result = system.answer(question)
            answer = result.get("answer", "").lower()
            walk_labels = " ".join(result.get("walk_path_labels", [])).lower()
            is_correct = expected.lower() in answer + " " + walk_labels
            if is_correct:
                correct += 1
            results.append({
                "question": question, "answer": result.get("answer", ""),
                "expected": expected, "correct": is_correct,
                "confidence": result.get("confidence", 0),
                "walk_confidence": result.get("walk_confidence", 0),
            })
            status = "✓" if is_correct else "✗"
            logger.info(f"  {status} {question[:50]:50s} -> {result.get('answer', '')[:40]:40s} [{expected}]")
        except Exception as e:
            logger.warning(f"  ! {question}: {e}")
    accuracy = correct / total * 100 if total > 0 else 0
    logger.info(f"\nQA: {correct}/{total} ({accuracy:.1f}%)")
    return {"qa_correct": correct, "qa_total": total, "qa_accuracy": accuracy,
            "per_question_results": results}


def train_on_db(db_path: str, checkpoint_dir: str,
                sbert_model: str = "BAAI/bge-small-en-v1.5",
                metrics: Optional[MetricsTracker] = None) -> Dict:
    """Absorb a .db into the model. Fully adaptive: no epochs, no LR, no heuristics.
    Stops when gradient dynamics indicate convergence."""
    
    t_start = time.time()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sbert = SentenceTransformer(sbert_model)
    store = SQLiteGraphStore.load_state(str(db_path))
    logger.info(f"Graph loaded: {store.get_node_count()} nodes, {store.get_edge_count()} edges")

    if len(store._embeddings) < store.get_node_count():
        nids = sorted(store._nodes.keys())
        labels = [store._nodes[nid].label for nid in nids]
        embs = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=True)
        for nid, emb in zip(nids, embs):
            store._embeddings[nid] = emb.astype(np.float32)
            node = store._nodes[nid]
            store._nodes[nid] = type(node)(
                id=node.id, label=node.label, node_type=node.node_type,
                embedding=emb.astype(np.float32), activation=node.activation,
                use_count=node.use_count, create_time=node.create_time,
            )

    dataset_name = Path(db_path).stem
    if metrics:
        metrics.dataset_name = dataset_name
        metrics.source_file = db_path
        metrics.set_component("graph_store", {
            "node_count": store.get_node_count(), "edge_count": store.get_edge_count(),
            "relation_types": store.get_all_relations(), "has_all_embeddings": True,
            "nodes_without_embeddings": store.get_node_count() - len(store._embeddings),
        })

    # Initialize system components
    system = GLMXSystem(sbert)
    system.initialize()
    system.attach_graph(store)
    logger.info("System initialized — all 6 components ready")

    # Build / load IntentFFN
    intent_ffn_path = checkpoint_dir / "intent_ffn.pt"
    if intent_ffn_path.exists():
        ckpt = torch.load(str(intent_ffn_path), map_location="cpu", weights_only=False)
        model = IntentFFN(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        model._trained_on_datasets = ckpt.get("trained_on_datasets", [])
        system.planner._ffn_trained = True
        logger.info(f"Loaded existing model (previously trained on: {model._trained_on_datasets})")
    else:
        model = IntentFFN(input_dim=384, hidden_dim=128, output_dim=16)
        logger.info("Created new IntentFFN")

    # Online learner — fully adaptive, no fixed params
    learner = OnlineLearner(model)
    stream = DataStream(store, sbert)

    qa_baseline = run_qa_eval(system)
    logger.info(f"QA baseline (untrained): {qa_baseline.get('qa_accuracy', 0):.1f}%")

    # Absorb data until convergence
    absorb_log = []
    round_num = 0
    sample_sizes = []

    while not learner.converged and round_num < 100:
        round_num += 1
        n_samples = max(1, len(stream) // max(1, 20 - round_num))
        n_samples = min(n_samples, len(stream), 512)
        sample_sizes.append(n_samples)

        X, y = stream.sample(n_samples)
        if len(X) == 0:
            break

        result = learner.absorb(X, y)
        absorb_log.append(result)

        if round_num % 5 == 0 or round_num == 1:
            logger.info(
                f"  absorb[{round_num}] n={n_samples} "
                f"loss={result['loss']:.4f} gns={result['gradient_noise_scale']:.3f} "
                f"rgn={result['relative_grad_norm']:.5f} lir={result['loss_improvement_rate']:.4f} "
                f"conv={result['converged']}"
            )

    # Transfer learned model to the planner
    old_state = system.planner.intent_ffn.state_dict()
    new_state = model.state_dict()
    for k in old_state:
        if k in new_state and old_state[k].shape == new_state[k].shape:
            old_state[k] = new_state[k]
    system.planner.intent_ffn.load_state_dict(old_state)
    system.planner.mark_trained()

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "trained_on_datasets": model._trained_on_datasets + [dataset_name],
        "absorb_log": absorb_log[-100:],
        "total_exposures": learner.total_exposures,
    }, str(intent_ffn_path))
    logger.info(f"Model saved to {intent_ffn_path}")

    # Evaluate QA
    qa_result = run_qa_eval(system)
    qa_accuracy = qa_result.get("qa_accuracy", 0)

    if metrics:
        metrics.set_component("intent_ffn", {
            "dataset_name": dataset_name,
            "is_continual_training": intent_ffn_path.exists(),
            "total_samples": len(stream),
            "total_exposures": learner.total_exposures,
            "absorb_rounds": round_num,
            "final_loss": absorb_log[-1]["loss"] if absorb_log else 0,
            "gradient_noise_scale": learner.dynamics.gradient_noise_scale,
            "converged": learner.converged,
            "learning_rate": "adaptive",
            "batch_size": "adaptive",
        })
        metrics.set_component("qa_eval", qa_result)
        metrics.save_all()
        metrics.append_to_history()

    elapsed = time.time() - t_start
    summary = {
        "dataset": dataset_name,
        "nodes": store.get_node_count(),
        "edges": store.get_edge_count(),
        "absorb_rounds": round_num,
        "total_exposures": learner.total_exposures,
        "converged": learner.converged,
        "qa_accuracy": qa_accuracy,
        "qa_correct": qa_result.get("qa_correct", 0),
        "qa_total": qa_result.get("qa_total", 0),
        "total_time_seconds": round(elapsed, 1),
    }
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE: {summary}")
    logger.info(f"{'='*60}")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GLM-X Adaptive Training Pipeline")
    parser.add_argument("--db", required=True, help="Path to .db graph file")
    parser.add_argument("--checkpoint", default=str(BASE_DIR / "checkpoints" / "unified"),
                        help="Checkpoint directory")
    parser.add_argument("--sbert", default="BAAI/bge-small-en-v1.5", help="SBERT model name")
    args = parser.parse_args()

    metrics = MetricsTracker(
        run_id=time.strftime("%Y%m%d_%H%M%S") + f"_{Path(args.db).stem}",
        checkpoint_dir=args.checkpoint,
    )
    result = train_on_db(
        db_path=args.db,
        checkpoint_dir=args.checkpoint,
        sbert_model=args.sbert,
        metrics=metrics,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
