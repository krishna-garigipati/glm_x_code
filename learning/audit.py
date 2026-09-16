import logging
import threading
import time
from typing import List, Dict, Tuple, Optional, Set
from collections import deque

import numpy as np

from learning.types import (
    Subgraph, Edge, Plan, WalkResult, Answer,
    GraphStoreInterface, ResonanceEngineInterface,
    G2PPlannerInterface, GraphWalkerInterface, MicroDecoderInterface,
)
from learning.config import LearningConfig

logger = logging.getLogger(__name__)


class SelfAuditor:
    def __init__(self, config: LearningConfig):
        self.cfg = config.audit
        self._lock = threading.Lock()
        self._query_counter = 0
        self._last_audit_time: float = 0.0
        self._audit_reports: List[Dict] = []

    @property
    def query_counter(self) -> int:
        return self._query_counter

    def run_self_audit(
        self,
        graph: GraphStoreInterface,
        resonance_engine: ResonanceEngineInterface,
        g2p: G2PPlannerInterface,
        walker: GraphWalkerInterface,
        decoder: MicroDecoderInterface,
    ) -> Dict:
        report: Dict = {
            "timestamp": time.time(),
            "contradictions": [],
            "low_confidence_issues": [],
            "uncertain_intents": [],
            "audit_success": True,
        }

        logger.info("Starting self-audit cycle")

        try:
            contradictions = self._detect_contradictions_bfs(graph)
            report["contradictions"] = contradictions
        except Exception as e:
            logger.error("Contradiction detection failed: %s", e)
            report["contradictions"] = []
            report["audit_success"] = False

        try:
            low_conf = self._detect_low_confidence_edges(graph)
            report["low_confidence_issues"] = low_conf
        except Exception as e:
            logger.error("Low confidence detection failed: %s", e)
            report["low_confidence_issues"] = []
            report["audit_success"] = False

        try:
            seed_nodes = self._get_sample_seeds(graph)
            if seed_nodes:
                uncertain_results = self._check_uncertain_intents(
                    graph, resonance_engine, g2p, walker, decoder, seed_nodes
                )
                report["uncertain_intents"] = uncertain_results
        except Exception as e:
            logger.error("Uncertain intent check failed: %s", e)
            report["uncertain_intents"] = []
            report["audit_success"] = False

        with self._lock:
            self._audit_reports.append(report)
            if len(self._audit_reports) > 100:
                self._audit_reports = self._audit_reports[-50:]

        logger.info(
            "Self-audit complete: %d contradictions, %d low-confidence, %d uncertain",
            len(report["contradictions"]),
            len(report["low_confidence_issues"]),
            len(report["uncertain_intents"]),
        )
        return report

    def detect_contradictions(self, subgraph: Subgraph) -> List[Tuple[Edge, Edge]]:
        result: List[Tuple[Edge, Edge]] = []
        contradicts_pairs: Set[Tuple[int, int]] = set()
        threshold = self.cfg.contradiction_threshold

        for src, tgt, rel in subgraph.edges:
            if rel in ("contradicts", "antonym"):
                contradicts_pairs.add((src, tgt))

        for src, tgt in contradicts_pairs:
            node_a_support = sum(
                subgraph.edge_strengths.get((src, n, r), 0.0) *
                subgraph.edge_confidences.get((src, n, r), 0.0)
                for n in subgraph.nodes
                for r in [e[2] for e in subgraph.edges if e[0] == src and e[1] == n]
            )
            node_b_support = sum(
                subgraph.edge_strengths.get((tgt, n, r), 0.0) *
                subgraph.edge_confidences.get((tgt, n, r), 0.0)
                for n in subgraph.nodes
                for r in [e[2] for e in subgraph.edges if e[0] == tgt and e[1] == n]
            )

            score = abs(node_a_support - node_b_support) / max(node_a_support + node_b_support, 1e-8)
            if score >= threshold:
                edge_a = Edge(
                    source=src, target=tgt, relation_type="contradicts",
                    strength=1.0, confidence=1.0, last_used=time.time(), frequency=1,
                )
                edge_b = Edge(
                    source=tgt, target=src, relation_type="contradicts",
                    strength=1.0, confidence=1.0, last_used=time.time(), frequency=1,
                )
                result.append((edge_a, edge_b))

        return result

    def _detect_contradictions_bfs(self, graph: GraphStoreInterface) -> List[Dict]:
        contradictions: List[Dict] = []
        seen_pairs: Set[Tuple[int, int]] = set()

        sample_nodes = self._get_high_degree_sample_seeds(graph, max_samples=50)
        for seed in sample_nodes:
            for neighbor_id, edge in graph.get_neighbors(seed):
                if edge.relation_type not in ("contradicts", "antonym"):
                    continue
                pair = (min(seed, neighbor_id), max(seed, neighbor_id))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                orig_strength = edge.strength
                orig_confidence = edge.confidence
                try:
                    subgraph = graph.get_subgraph_activated([seed, neighbor_id], max_nodes=50)
                    pairs = self.detect_contradictions(subgraph)
                    for edge_a, edge_b in pairs:
                        contradictions.append({
                            "source": edge_a.source,
                            "target": edge_a.target,
                            "relation": edge_a.relation_type,
                            "score": orig_strength,
                            "edge_strength": orig_strength,
                            "edge_confidence": orig_confidence,
                            "method": "bfs",
                        })
                except Exception as e:
                    logger.debug("Contradiction check failed for %d -> %d: %s", seed, neighbor_id, e)
                    continue

        contradictions.sort(key=lambda x: x["score"], reverse=True)
        return contradictions

    def _detect_low_confidence_edges(self, graph: GraphStoreInterface) -> List[Dict]:
        issues: List[Dict] = []
        threshold = self.cfg.low_confidence_threshold
        sample_nodes = self._get_high_degree_sample_seeds(graph, max_samples=50)

        for node_id in sample_nodes:
            neighbors = graph.get_neighbors(node_id)
            for neighbor_id, edge in neighbors:
                if edge.confidence < threshold:
                    issues.append({
                        "source": node_id,
                        "target": neighbor_id,
                        "relation": edge.relation_type,
                        "confidence": edge.confidence,
                        "strength": edge.strength,
                        "below_threshold_by": threshold - edge.confidence,
                    })

        issues.sort(key=lambda x: x["confidence"])
        return issues

    def _check_uncertain_intents(
        self,
        graph: GraphStoreInterface,
        resonance_engine: ResonanceEngineInterface,
        g2p: G2PPlannerInterface,
        walker: GraphWalkerInterface,
        decoder: MicroDecoderInterface,
        seed_nodes: List[int],
    ) -> List[Dict]:
        uncertain_results: List[Dict] = []
        uncertain_ids = self.cfg.uncertain_intent_ids
        num_queries = self.cfg.synthetic_queries_per_audit

        for i in range(min(num_queries, len(seed_nodes))):
            seed = seed_nodes[i % len(seed_nodes)]
            node = graph.get_node(seed)
            if node is None:
                continue

            try:
                query_emb = np.zeros(384, dtype=np.float32)
                subgraph = resonance_engine.resonate(query_emb, graph, [seed], tier=1)
                plan = g2p.plan(subgraph)

                if plan.plan_confidence < self.cfg.low_confidence_threshold:
                    if not plan.heuristic_fallback_used:
                        uncertain_results.append({
                            "seed_node": seed,
                            "plan_confidence": plan.plan_confidence,
                            "intent_sequence": plan.intent_sequence,
                            "issue": "low_plan_confidence",
                        })

                for intent_id in (plan.intent_sequence or []):
                    if intent_id in uncertain_ids:
                        uncertain_results.append({
                            "seed_node": seed,
                            "plan_confidence": plan.plan_confidence,
                            "intent_sequence": plan.intent_sequence,
                            "issue": "uncertain_intent",
                            "uncertain_intent_id": intent_id,
                            "uncertain_intent_name": self._get_intent_name(intent_id),
                        })
            except Exception as e:
                logger.debug("Uncertain intent check failed for seed %d: %s", seed, e)
                continue

        return uncertain_results

    def _get_sample_seeds(
        self,
        graph: GraphStoreInterface,
        max_samples: int = 50,
    ) -> List[int]:
        node_ids = list(getattr(graph, 'nodes', {}).keys())
        if not node_ids:
            return [0]
        return node_ids[:max_samples]

    def _get_high_degree_sample_seeds(
        self,
        graph: GraphStoreInterface,
        max_samples: int = 50,
    ) -> List[int]:
        node_ids = list(getattr(graph, 'nodes', {}).keys())
        if not node_ids:
            return [0]
        node_degrees = []
        for nid in node_ids[:max_samples * 3]:
            neighbors = graph.get_neighbors(nid)
            node_degrees.append((nid, len(neighbors)))
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _deg in node_degrees[:max_samples]]

    def _get_intent_name(self, intent_id: int) -> str:
        names = {
            0: "define", 1: "assert_fact", 2: "explain_cause",
            3: "explain_effect", 4: "contrast", 5: "compare",
            6: "list", 7: "example", 8: "conclude", 9: "question",
            10: "uncertain", 11: "clarify", 12: "summarize",
            13: "elaborate", 14: "transition", 15: "emphasize",
        }
        return names.get(intent_id, "unknown")

    def increment_query_counter(self, n: int = 1) -> bool:
        with self._lock:
            self._query_counter += n
            return self._query_counter >= self.cfg.interval_queries

    def reset_query_counter(self) -> None:
        with self._lock:
            self._query_counter = 0

    def get_audit_reports(self, last_n: int = 5) -> List[Dict]:
        with self._lock:
            return self._audit_reports[-last_n:]
