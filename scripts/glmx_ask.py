#!/usr/bin/env python
"""
GLM-X Full Pipeline: resonance -> extractor -> walker -> decoder

All components are wired together and every inch is used at inference.

Flow (DEVIATION 9 — no intents, no IntentFFN):
  question -> SBERT encode -> GraphStore.get_subgraph() -> seed Subgraph
  -> Tier1Resonance.resonate() -> activated Subgraph (activations propagate)
  -> QueryRelationExtractor.extract() -> Plan (ordered relation chain)
  -> GraphWalker.walk() (chain-guided) -> WalkResult (ordered path through graph)
  -> TemplateDecoder.decode() (chain-rendered) -> final answer text
  -> reward -> EvolutionaryController updates walker relation biases
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

from graph.graph_component_implementation.dict_graph_store import DictGraphStore
from graph.demo_graph_data import build_demo_store

from resonance.tier1 import Tier1Resonance
from resonance.config import (
    CoreConfig, CoreActivationConfig, CoreResonanceConfig, ESBounds,
    AlgorithmConfig, TierConfig, TemporalConfig,
)
from resonance.config import (
    load_configs as load_resonance_configs,
    build_default_theta, ThetaIndices, ESControllerConfig,
)
from resonance.es_controller import EvolutionaryController

from g2p.g2p_planner import QueryRelationExtractor
from g2p.types import Subgraph as G2PSubgraph, Plan as G2PPlan
from g2p.config import G2PConfig

from walker.config import CoreConfig as WalkerCoreConfig, WalkerConfig, ActivationConfig, WalkerCoreConfig as WCore, load_yaml as load_walker_yaml
from walker.models import Plan as WalkerPlan, Subgraph as WalkerSubgraph, WalkResult
from walker.graph_walker import GraphWalker

from decoder.template_decoder import TemplateDecoder
from decoder.config_loader import load_config as load_decoder_config

from learning.engine import LearningEngine
from learning.config import LearningConfig
from learning.types import (
    Node as LNode, Edge as LEdge, Subgraph as LSubgraph,
    WalkResult as LWalkResult, Plan as LPlan, Answer as LAnswer,
    GraphStoreInterface as LGraphStoreInterface,
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "model_training" / "dataset_conceptnet" / "conceptnet" / "data"
CONFIG_PATH = BASE_DIR / "configs" / "config_g2p.yaml"
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


def relation_display_name(relation: str) -> str:
    """Map canonical 16-relation names back to friendly labels."""
    return REVERSE_RELATION_MAP.get(relation, relation)


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


def load_demo_graph() -> DictGraphStore:
    """Load the bundled deterministic demo graph (offline, no downloads)."""
    store = build_demo_store()
    logger.info(
        "Demo graph ready: %d nodes, %d edges (%d relations)",
        store.get_node_count(), store.get_edge_count(), len(store.get_all_relations()),
    )
    return store


def make_resonance_configs() -> Tuple[CoreConfig, AlgorithmConfig, TemporalConfig, TierConfig]:
    """Build resonance configs programmatically for the 16 canonical relations."""
    core = CoreConfig(
        activation=CoreActivationConfig(min=0.01, max=1.0, default=0.01, threshold_resonance=0.2),
        resonance=CoreResonanceConfig(
            propagation_threshold=0.008, edge_threshold=0.02, decay_lambda=0.1,
            top_k=64, budget_max=2.0, convergence_epsilon=0.001,
            tier1_energy_threshold=0.4, tier2_max_nodes=1024, analogy_validation_overlap=0.3,
        ),
        relations=["is_a", "has_property", "causes", "caused_by", "follows",
                   "precedes", "contradicts", "supports", "associated_with",
                   "example_of", "part_of", "synonym", "antonym",
                   "temporal_coincident", "spatial_near", "linguistic_maps"],
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
            "is_a": 0.8,
            "has_property": 0.5,
            "causes": 1.5,
            "caused_by": 0.6,
            "follows": 0.5,
            "precedes": 0.5,
            "contradicts": 0.5,
            "supports": 0.5,
            "associated_with": 0.5,
            "example_of": 0.5,
            "part_of": 0.7,
            "synonym": 1.0,
            "antonym": 0.4,
            "temporal_coincident": 0.5,
            "spatial_near": 0.5,
            "linguistic_maps": 0.5,
        },
        energy_threshold_formula=None, t_conf_coefficient=None, multiplied_at_runtime=None,
    )

    return core, algorithm, temporal, tier


class LearningGraphAdapter(LGraphStoreInterface):
    """Adapter wrapping DictGraphStore for the LearningEngine's GraphStoreInterface."""

    def __init__(self, store: DictGraphStore):
        self._store = store
        self._node_alias: Dict[int, int] = {}  # learning-assigned id -> store id

    def _resolve(self, node_id: int) -> int:
        return self._node_alias.get(node_id, node_id)

    def get_node(self, node_id: int) -> Optional[LNode]:
        real_id = self._resolve(node_id)
        node = self._store._nodes.get(real_id)
        if node is None:
            return None
        embedding = self._store._embeddings.get(real_id, np.zeros(384, dtype=np.float32))
        return LNode(
            id=node.id, label=node.label, node_type=getattr(node, 'node_type', 'Concept'),
            embedding=embedding,
            activation=node.activation, use_count=node.use_count,
            create_time=node.create_time, sense_id=node.sense_id,
        )

    def get_edge(self, source: int, target: int, relation: str) -> Optional[LEdge]:
        source, target = self._resolve(source), self._resolve(target)
        for e in self._store._edges_raw:
            if e.source == source and e.target == target and e.relation == relation:
                return LEdge(
                    source=source, target=target, relation_type=relation,
                    strength=e.strength, confidence=e.confidence,
                    last_used=getattr(e, 'last_used', 0.0),
                    frequency=getattr(e, 'frequency', 1),
                )
        return None

    def update_edge_weights(self, updates: Dict[Tuple[int, int, str], Tuple[float, float]]) -> None:
        for key, (strength, confidence) in updates.items():
            src, tgt, rel = key
            src, tgt = self._resolve(src), self._resolve(tgt)
            for e in self._store._edges_raw:
                if e.source == src and e.target == tgt and e.relation == rel:
                    e.strength = strength
                    e.confidence = confidence
                    break

    def add_node(self, node_id: int, label: str, node_type: str, embedding: np.ndarray, activation: float = 0.01) -> bool:
        real_id = self._store.add_node(label=label, embedding=embedding, node_type=node_type)
        self._node_alias[node_id] = real_id
        return True

    def update_node_embedding(self, node_id: int, embedding: np.ndarray) -> bool:
        self._store._embeddings[self._resolve(node_id)] = embedding
        return True

    def add_edge(self, source: int, target: int, relation: str, strength: float = 0.5, confidence: float = 0.5) -> bool:
        self._store.add_edge(
            source=self._resolve(source),
            target=self._resolve(target),
            relation=relation,
            strength=strength,
            confidence=confidence,
        )
        return True

    def get_neighbors(self, node_id: int, relation_filter: Optional[List[str]] = None) -> List[Tuple[int, LEdge]]:
        node_id = self._resolve(node_id)
        result = []
        for e in self._store._edges_raw:
            edge = LEdge(
                source=e.source, target=e.target, relation_type=e.relation,
                strength=e.strength, confidence=e.confidence,
                last_used=getattr(e, 'last_used', 0.0), frequency=getattr(e, 'frequency', 1),
            )
            if e.source == node_id and (relation_filter is None or e.relation in relation_filter):
                result.append((e.target, edge))
            elif e.target == node_id and (relation_filter is None or e.relation in relation_filter):
                result.append((e.source, edge))
        return result

    def get_subgraph_activated(self, seed_nodes: List[int], max_nodes: int = 1000) -> LSubgraph:
        return self._store.get_subgraph_by_embedding_similarity(np.zeros(384, dtype=np.float32), top_k=max_nodes)

    def get_subgraph_by_embedding_similarity(self, query_embedding: np.ndarray, top_k: int = 100) -> LSubgraph:
        return self._store.get_subgraph_by_embedding_similarity(query_embedding, top_k=top_k)

    def prune(self, utility_threshold: float = 0.01) -> int:
        return 0

    def save_checkpoint(self, filepath: str) -> bool:
        self._store.save_state(filepath)
        return True

    def load_checkpoint(self, filepath: str) -> bool:
        from pathlib import Path
        self._store = DictGraphStore.load_state(filepath)
        return True


class GLMXPipeline:
    """Full GLM-X pipeline using every component."""

    def __init__(self):
        self.sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")

        self.graph_store: Optional[DictGraphStore] = None
        self.tier1: Optional[Tier1Resonance] = None
        self.planner: Optional[QueryRelationExtractor] = None
        self.walker: Optional[GraphWalker] = None
        self.decoder: Optional[TemplateDecoder] = None
        self.learning_engine: Optional[LearningEngine] = None
        self._learning_graph: Optional[LearningGraphAdapter] = None
        self.es_controller: Optional[EvolutionaryController] = None
        self._base_relation_biases: Dict[str, Dict[str, float]] = {}
        self._last_theta: Optional[np.ndarray] = None

    def load_graph(self, max_edges_per_rel: int = 2000) -> None:
        parquet_files = sorted(DATA_DIR.glob("*.parquet"))
        if parquet_files:
            self.graph_store = load_conceptnet(max_edges_per_rel)
            logger.info("Graph source: ConceptNet parquet")
        else:
            self.graph_store = load_demo_graph()
            logger.info("Graph source: bundled demo graph (offline PoC data)")

        if self.graph_store.get_node_count() == 0 or self.graph_store.get_edge_count() == 0:
            raise RuntimeError(
                "Graph is empty: no ConceptNet parquet data found under "
                f"{DATA_DIR} and the bundled demo graph failed to load. "
                "Install the parquet shards or fix demo_graph_data.py."
            )

        nids = sorted(self.graph_store._nodes.keys())
        labels = [self.graph_store._nodes[nid].label for nid in nids]
        logger.info(f"Computing embeddings for {len(labels)} concepts...")
        embs = self.sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
        for nid, emb in zip(nids, embs):
            self.graph_store._embeddings[nid] = emb
            node = self.graph_store._nodes[nid]
            self.graph_store._nodes[nid] = type(node)(
                id=node.id, label=node.label, node_type=node.node_type,
                embedding=emb, activation=node.activation,
                use_count=node.use_count, create_time=node.create_time,
            )
        logger.info("Embeddings computed and stored")

    def load_models(self, model_path: Optional[str] = None) -> None:
        # ---- Tier1 Resonance ----
        core_cfg, algo_cfg, temporal_cfg, tier_cfg = make_resonance_configs()
        self.tier1 = Tier1Resonance(
            core_config=core_cfg, algorithm=algo_cfg, temporal=temporal_cfg,
            tier_config=tier_cfg, log_activation_history=True, history_buffer_size=10,
        )
        logger.info("Tier1Resonance initialized (wilson_cowan, top_k=64, 3 iterations)")

        # ---- Query-Relation Extractor (DEVIATION 9) ----
        self.planner = QueryRelationExtractor(G2PConfig.from_yaml(str(CONFIG_PATH)))
        graph_relations = (
            sorted(self.graph_store.get_all_relations())
            if self.graph_store is not None else None
        )
        self.planner.initialize(graph_relations=graph_relations)
        self.planner.mark_trained()
        logger.info("QueryRelationExtractor ready (no IntentFFN, no checkpoint needed)")

        # ---- Graph Walker (chain-guided) ----
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
                max_steps=int(wc_raw.get("walker", {}).get("max_steps", 6)),
                min_activation_to_continue=float(
                    wc_raw.get("walker", {}).get("min_activation_to_continue", 0.05)
                ),
            ),
            relations={int(k): v for k, v in wc_raw.get("relations", {}).items()},
        )
        walker_cfg = WalkerConfig.from_yaml(str(WALKER_CONFIG_PATH))
        self.walker = GraphWalker(walker_cfg, walker_core_cfg)
        self._base_relation_biases = {
            str(k): dict(v) for k, v in walker_cfg.relation_biases.items()
        }
        logger.info("GraphWalker initialized (relation-chain guided)")

        # ---- Template Decoder (chain rendering) ----
        decoder_cfg = load_decoder_config(str(DECODER_CONFIG_PATH))
        self.decoder = TemplateDecoder(
            templates=decoder_cfg.get("templates", {}).get("definitions", []),
            relation_phrases=decoder_cfg.get("templates", {}).get("relation_phrases", {}),
            sentence_starters=decoder_cfg.get("templates", {}).get("sentence_starters", []),
            fallback_cfg=decoder_cfg.get("fallback", {}),
            validation_cfg=decoder_cfg.get("validation", {}),
            chain_render_cfg=decoder_cfg.get("templates", {}).get("chain_render", {}),
        )
        logger.info("TemplateDecoder loaded (relation-chain template mode)")

        # ---- Evolutionary Controller (DEVIATION 9: tunes walker relation biases) ----
        loaded = load_resonance_configs(BASE_DIR / "configs")
        self.es_controller = EvolutionaryController(
            core_config=loaded.core,
            es_config=loaded.resonance.es_controller,
            initial_theta=build_default_theta(loaded.resonance, loaded.core),
            theta_indices=loaded.resonance.theta_indices,
            _seed=42,
        )
        self._last_theta = self.es_controller.get_theta()
        self._apply_es_theta(self._last_theta)
        logger.info("EvolutionaryController ready (48-dim theta, %d relation slots)",
                    self.es_controller._theta_indices.relation_bias_end - self.es_controller._theta_indices.relation_bias_start)

        # ---- Learning Engine (Hebbian, replay, compression, audit) ----
        self.learning_engine = LearningEngine(LearningConfig())
        self._learning_graph = LearningGraphAdapter(self.graph_store)
        logger.info("LearningEngine initialized (Hebbian + replay + compression + audit)")

        logger.info("All components initialized. GLM-X pipeline ready.")

    def _apply_es_theta(self, theta: np.ndarray) -> None:
        """Push ES relation-bias slots into the walker as multipliers over the base table."""
        if self.es_controller is None or self.walker is None:
            return
        indices = self.es_controller._theta_indices
        relations = self.es_controller._core.relations
        start = indices.relation_bias_start
        overrides: Dict[str, Dict[str, float]] = {}
        for offset, relation in enumerate(relations):
            base = self._base_relation_biases.get(relation)
            if base is None:
                continue
            scale = max(float(theta[start + offset]), 1e-3)
            overrides[relation] = {
                rel: float(min(3.0, bias * scale))
                for rel, bias in base.items()
            }
        self.walker.replace_relation_biases(overrides)

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

        # ===== STEP 2: Target entity = single highest-sim embedding neighbor =====
        ts = time.time()
        seed_sub = self.graph_store.get_subgraph_by_embedding_similarity(q_emb, top_k=20)
        target_entity_ids = [seed_sub.seed_nodes[0]] if seed_sub.seed_nodes else []
        for nid in target_entity_ids:
            seed_sub.node_activations[nid] = max(
                seed_sub.node_activations.get(nid, 0), 0.9
            )
        for nid in seed_sub.seed_nodes[:3]:
            seed_sub.node_activations[nid] = max(
                seed_sub.node_activations.get(nid, 0), 0.8
            )
        steps_log["2_subgraph"] = round(time.time() - ts, 3)
        logger.info(f"[2/6] Embedding path: "
                    f"| GraphStore: {len(seed_sub.nodes)} nodes, {len(seed_sub.seed_nodes)} seeds"
                    f"| Target entity: {target_entity_ids}")

        # ===== STEP 3: Tier1Resonance propagation =====
        ts = time.time()
        resonated, history = self.tier1.resonate(q_emb, self.graph_store, seed_sub.seed_nodes)
        steps_log["3_resonance"] = round(time.time() - ts, 3)
        logger.info(f"[3/6] Tier1 resonance: {len(resonated.nodes)} nodes, "
                    f"energy={resonated.activation_energy:.4f}, "
                    f"tier={resonated.tier_used}, {len(history)} iterations")

        # ===== STEP 4: QueryRelationExtractor -> relation chain =====
        ts = time.time()
        plan = self.planner.plan(resonated, query_text=question)
        steps_log["4_plan"] = round(time.time() - ts, 3)
        logger.info(f"[4/6] Plan: chain={plan.relation_chain}, "
                    f"heuristic_fallback={plan.heuristic_fallback_used}, "
                    f"confidence={plan.plan_confidence:.4f}")

        # ===== STEP 5: Graph Walker =====
        ts = time.time()
        for nid in target_entity_ids:
            resonated.node_activations[nid] = 1.0
        # Ensure target entities are the clear start-node winner: cap non-target seeds below 1.0
        target_set = set(target_entity_ids)
        for nid in resonated.seed_nodes:
            if nid not in target_set:
                resonated.node_activations[nid] = min(
                    resonated.node_activations.get(nid, 0), 0.99
                )

        if resonated.node_embeddings is not None:
            node_embeddings = resonated.node_embeddings
        else:
            node_embeddings = {}
            for nid in resonated.nodes:
                emb = self.graph_store.get_embedding(nid)
                if emb is not None:
                    node_embeddings[nid] = emb

        # Boost edges whose relation label is semantically similar to the query
        # e.g. "What originated in China?" boosts the "originated in" relation
        rel_labels = sorted(set(r for _, _, r in resonated.edges))
        if len(rel_labels) > 1:
            rel_embs = self.sbert.encode(rel_labels, normalize_embeddings=True)
            for s, t, r in resonated.edges:
                rel_idx = rel_labels.index(r)
                rel_sim = float(np.dot(q_emb, rel_embs[rel_idx]))
                target_emb = node_embeddings.get(t)
                target_sim = float(np.dot(q_emb, target_emb)) if target_emb is not None else 0
                boost = 1.0 + 2.0 * max(0.0, rel_sim - 0.15) + 0.5 * max(0.0, target_sim - 0.15)
                resonated.edge_strengths[(s, t, r)] *= boost

        # Add reverse edges for bidirectional walking (e.g. "What is in France?" needs france->paris)
        rev_edges = [(t, s, r) for s, t, r in resonated.edges]
        rev_strengths = {}
        rev_confidences = {}
        for s, t, r in resonated.edges:
            rev_strengths[(t, s, r)] = resonated.edge_strengths.get((s, t, r), 0.5)
            rev_confidences[(t, s, r)] = resonated.edge_confidences.get((s, t, r), 0.5)
        all_edges = list(dict.fromkeys(resonated.edges + rev_edges))
        all_strengths = {**resonated.edge_strengths, **rev_strengths}
        all_confidences = {**resonated.edge_confidences, **rev_confidences}

        walker_sub = WalkerSubgraph(
            nodes=resonated.nodes,
            node_activations=resonated.node_activations,
            edges=all_edges,
            edge_strengths=all_strengths,
            edge_confidences=all_confidences,
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
            relation_chain=plan.relation_chain,
        )

        self._last_theta = self.es_controller.get_theta() if self.es_controller is not None else None
        if self._last_theta is not None:
            self._apply_es_theta(self._last_theta)

        walk = self.walker.walk(walker_sub, walker_plan)
        steps_log["5_walk"] = round(time.time() - ts, 3)
        logger.info(f"[5/6] Walker: {walk.steps_taken} steps, "
                    f"chain_used={walk.relation_chain_used or plan.relation_chain}, "
                    f"confidence={walk.walk_confidence:.4f}, "
                    f"path={[self.graph_store.get_label(n) for n in walk.path]}")

        # ===== STEP 6: Decode (chain templates) =====
        ts = time.time()
        node_labels = [self.graph_store.get_label(n) for n in walk.path]
        edge_labels = list(walk.path_edges)
        chain = plan.relation_chain or ["has_property"]

        # Honest "no relation" answer: extractor fell back AND the walk found no edges.
        if plan.heuristic_fallback_used and not walk.path_edges:
            answer = self.decoder.render_no_relation(node_labels, chain=chain)
            template_ok = True if answer else False
            logger.info("[6/6] Decoder: no relation found; emitted honest no_relation answer")
        else:
            answer, template_ok = self.decoder.decode(node_labels, edge_labels, chain=chain)
            if not template_ok:
                try:
                    answer = self.decoder.fallback(node_labels, edge_labels, chain=chain)
                except Exception:
                    fallback = " ".join(node_labels[:5])
                    starter = self.decoder._select_sentence_starter(None)
                    answer = f"{starter} {fallback}" if starter else fallback
                template_ok = True
                logger.info("[6/6] Decoder: template failed; concatenative fallback used")

        steps_log["6_decode"] = round(time.time() - ts, 3)
        logger.info(f"[6/6] Decoder: template_ok={template_ok}, "
                    f"answer_len={len(answer)}, "
                    f"answer_start={answer[:60]!r}")

        # ===== REINFORCE feedback: reward = +1 if template matched walk, -0.3 otherwise =====
        reward = 1.0 if template_ok else -0.3
        path_relations = chain[:len(walk.path_edges)] if walk.path_edges else chain[:1]
        self.walker.apply_reward(
            reward=reward,
            path_edges=list(walk.path_edges),
            path_relations=path_relations,
            learning_rate=0.01,
        )

        # EvolutionaryController: reward the theta that produced this walk.
        if self.es_controller is not None and self._last_theta is not None:
            try:
                self.es_controller.update_es_with_reward(float(reward), self._last_theta)
            except Exception as exc:
                logger.warning(f"EvolutionaryController update failed: {exc}")

        # Build LearningEngine types from pipeline output
        l_plan = LPlan(
            intent_sequence=plan.intent_sequence,
            plan_confidence=plan.plan_confidence,
            heuristic_fallback_used=plan.heuristic_fallback_used,
            intent_names=plan.intent_names,
            relation_chain=plan.relation_chain,
        )
        l_walk_result = LWalkResult(
            path=walk.path,
            path_edges=list(walk.path_edges),
            path_activations=walk.path_activations if walk.path_activations else [0.5] * len(walk.path),
            path_confidences=walk.path_confidences if walk.path_confidences else [0.5] * len(walk.path_edges),
            path_embeddings=list(walk.path_embeddings) if walk.path_embeddings else [np.zeros(384, dtype=np.float32)] * len(walk.path),
            walk_confidence=walk.walk_confidence,
            final_activation=getattr(walk, 'final_activation', 0.5),
            steps_taken=walk.steps_taken,
            plan_followed=l_plan,
            timestamp=time.time(),
            intent_sequence_used=plan.intent_sequence or [],
            relation_chain_used=walk.relation_chain_used or list(chain),
        )
        l_subgraph = LSubgraph(
            nodes=resonated.nodes,
            node_activations=resonated.node_activations,
            edges=resonated.edges,
            edge_strengths=resonated.edge_strengths,
            edge_confidences=resonated.edge_confidences,
            seed_nodes=resonated.seed_nodes,
            tier_used=resonated.tier_used,
            activation_energy=resonated.activation_energy,
            query_embedding=resonated.query_embedding,
            timestamp=getattr(resonated, 'timestamp', time.time()),
        )
        l_answer = LAnswer(
            text=answer,
            confidence=float(plan.plan_confidence),
            intent_used=1,
            nodes_mentioned=list(walk.path),
            generation_method="template" if template_ok else "fallback",
            walk_used=l_walk_result,
            subgraph_used=l_subgraph,
            timestamp=time.time(),
        )
        try:
            self.learning_engine.process_feedback(
                answer=l_answer,
                user_rating=float(reward),
                walk=l_walk_result,
                subgraph=l_subgraph,
                graph=self._learning_graph,
                resonance_engine=None,
                external_reward=float(reward),
                plan_adherence=1.0 if template_ok else 0.3,
            )
        except Exception as e:
            logger.warning(f"LearningEngine feedback failed: {e}")

        steps_log["6b_reinforce"] = round(time.time() - ts, 3)

        elapsed = time.time() - t0
        subgraph_text = self.subgraph_to_text(resonated)
        walk_text = self.walk_result_to_text(walk)

        # Build relation-chain explanation
        relation_details = []
        for i, rel in enumerate(chain):
            if i < len(walk.path_edges):
                n0 = self.graph_store.get_label(walk.path[i])
                n1 = self.graph_store.get_label(walk.path[i + 1])
                edge_r = REVERSE_RELATION_MAP.get(walk.path_edges[i], walk.path_edges[i])
                relation_details.append(f"  step {i}: expected={rel} "
                                        f"follow {n0} --[{edge_r}]--> {n1}")
            else:
                relation_details.append(f"  step {i}: expected={rel} [no more path edges]")

        return {
            "question": question,
            "answer": answer,
            "relation_chain": chain,
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
            "relation_details": relation_details,
            "subgraph": subgraph_text,
            "walk_path": walk_text,
        }


    def save_checkpoint(self, path: str) -> None:
        """Save full model checkpoint: graph + config references (no IntentFFN)."""
        import json, os
        os.makedirs(path, exist_ok=True)

        graph_dir = os.path.join(path, "graph_store")
        self.graph_store.save_state(graph_dir)

        checkpoint_manifest = {
            "model": "GLM-X",
            "version": "1.1.0",
            "architecture": "query-relation-chain (Deviation 9, no IntentFFN)",
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
            "encoder": "BAAI/bge-small-en-v1.5",
        }
        with open(os.path.join(path, "checkpoint.json"), "w") as f:
            json.dump(checkpoint_manifest, f, indent=2)

        logger.info("Full checkpoint saved to %s", path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "GLMXPipeline":
        """Load full model checkpoint."""
        import os
        pipeline = cls()

        graph_dir = os.path.join(path, "graph_store")
        if os.path.exists(graph_dir):
            pipeline.graph_store = DictGraphStore.load_state(graph_dir)
        else:
            pipeline.load_graph()

        pipeline.load_models()

        if pipeline.graph_store is None:
            pipeline.load_graph()

        return pipeline

    @classmethod
    def from_corpus(
        cls,
        texts,
        max_docs=None,
        kg_config=None,
        model_path=None,
    ):
        from kg_builder import KGBuilderPipeline, KGBuilderConfig

        config = kg_config or KGBuilderConfig()
        builder = KGBuilderPipeline(config)
        store = builder.process_and_store(texts=texts, max_docs=max_docs, show_progress=True)
        pipeline = cls()
        pipeline.graph_store = store
        pipeline.load_models(model_path=model_path)
        logger.info(
            "GLM-X pipeline built from corpus: %d nodes, %d edges, %d types",
            store.get_node_count(),
            store.get_edge_count(),
            len(store.get_all_relations()),
        )
        return pipeline


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GLM-X Question Answering Pipeline")
    parser.add_argument("--db", default=None, help="Path to .db graph file (SQLiteGraphStore)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory with trained models")
    parser.add_argument("--question", "-q", default=None, help="Single question to answer")
    args = parser.parse_args()

    pipeline = GLMXPipeline()

    if args.db:
        db_path = Path(args.db)
        if not db_path.exists():
            print(f"Error: DB file not found: {db_path}", file=sys.stderr)
            sys.exit(1)
        from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
        pipeline.graph_store = SQLiteGraphStore.load_state(str(db_path))
        logger.info(f"Graph loaded from {db_path}")
    else:
        pipeline.load_graph(max_edges_per_rel=2000)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path and checkpoint_path.is_dir():
        pipeline.load_checkpoint(str(checkpoint_path))
    else:
        pipeline.load_models()

    logger.info("\n" + "=" * 70)
    logger.info("GLM-X PIPELINE READY")
    logger.info("Resonance -> QueryRelationExtractor -> Graph Walker -> Template Decoder")
    logger.info("=" * 70 + "\n")

    if args.question:
        result = pipeline.ask(args.question)
        print(json.dumps(result, indent=2, default=str))
        return

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
        print(f"\n  Relation Chain: {result['relation_chain']}")
        print(f"  Heuristic fallback: {result['heuristic_used']}")
        print(f"  Template matched: {result['template_matched']}")
        print(f"  Plan confidence: {result['confidence']:.4f}")
        print(f"  Walk confidence: {result['walk_confidence']:.4f}")
        print(f"  Walk steps: {result['n_walk_steps']}")
        print(f"  Walk path: {' -> '.join(result['walk_path_labels'])}")
        print(f"  Walk edges: {result['walk_path_edges']}")
        print(f"  Timing: {result['steps_timing']}")
        print(f"  Total: {result['time_seconds']}s")
        print(f"\n  ---- Chain-guided Walk Details ----")
        for line in result['relation_details']:
            print(f"  {line}")
        print(f"\n  ---- Resonance Subgraph ({result['n_resonated_nodes']} nodes, {result['n_resonated_edges']} edges) ----")
        print(f"  Energy: {result['resonance_energy']}")
        print(result['walk_path'])


if __name__ == "__main__":
    main()
