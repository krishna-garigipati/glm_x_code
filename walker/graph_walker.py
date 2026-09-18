"""Graph Walker implementation for Team D."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
import time
from threading import RLock
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .config import CoreConfig, WalkerConfig
from .eligibility import compute_eligibility_trace
from .exceptions import EmbeddingLookupError, ValidationError
from .relation_bias import LEGACY_INTENT_TO_RELATION, RelationBiasTable
from .models import Plan, Subgraph, WalkResult
from .path_scorer import PathScorer, ScoredCandidate
from .utils import geometric_mean


@dataclass(frozen=True)
class WalkDecision:
    current_node: int
    expected_relation: str
    candidates: List[ScoredCandidate]
    probabilities: List[float]


class GraphWalker:
    def __init__(
        self,
        walker_config: WalkerConfig,
        core_config: CoreConfig,
        embedding_provider: Optional[Callable[[int], "object"]] = None,
        random_seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._walker_config = walker_config
        self._core_config = core_config
        self._validate_config_consistency()
        self._validate_scoring_formula()
        self._relation_bias_table = RelationBiasTable(walker_config.relation_biases)
        self._scorer = PathScorer(
            weight_strength=walker_config.scoring.weight_strength,
            weight_confidence=walker_config.scoring.weight_confidence,
            weight_target_activation=walker_config.scoring.weight_target_activation,
            weight_intent_bias=walker_config.scoring.weight_intent_bias,
            weight_target_similarity=walker_config.scoring.weight_target_similarity,
            normalization=walker_config.scoring.normalization,
            softmax_temperature=walker_config.scoring.softmax_temperature,
        )
        min_t, max_t = walker_config.walk.temperature_range
        if not (min_t <= walker_config.walk.temperature <= max_t):
            raise ValidationError("Configured temperature is outside range")
        self._temperature = walker_config.walk.temperature
        self._lock = RLock()
        self._rng = random.Random(random_seed)
        self._embedding_provider = embedding_provider
        self._logger = logger or logging.getLogger("glmx.walker")
        self._saved_paths: List[List[int]] = []
        self._last_step_timestamps: List[float] = []

    def _validate_config_consistency(self) -> None:
        core_walker = self._core_config.walker
        walk_cfg = self._walker_config.walk
        mismatches = []
        if core_walker.default_temperature != walk_cfg.temperature:
            mismatches.append(
                f"temperature: core={core_walker.default_temperature} walker={walk_cfg.temperature}"
            )
        if core_walker.max_steps != walk_cfg.max_steps:
            mismatches.append(
                f"max_steps: core={core_walker.max_steps} walker={walk_cfg.max_steps}"
            )
        if core_walker.min_activation_to_continue != walk_cfg.min_activation:
            mismatches.append(
                f"min_activation: core={core_walker.min_activation_to_continue} walker={walk_cfg.min_activation}"
            )
        if tuple(core_walker.softmax_temperature_range) != tuple(walk_cfg.temperature_range):
            mismatches.append(
                f"temperature_range: core={core_walker.softmax_temperature_range} walker={walk_cfg.temperature_range}"
            )
        if mismatches:
            raise ValidationError(
                "CoreConfig/WalkerConfig mismatch: " + "; ".join(mismatches)
            )

    def _validate_scoring_formula(self) -> None:
        formula = self._walker_config.scoring.formula
        required_terms = ["strength", "confidence", "target_activation", "intent_bias"]
        for term in required_terms:
            if term not in formula:
                raise ValidationError(
                    f"Scoring formula does not contain required term '{term}': {formula}"
                )

    def set_temperature(self, temperature: float) -> None:
        min_t, max_t = self._walker_config.walk.temperature_range
        if not (min_t <= temperature <= max_t):
            raise ValidationError("Temperature outside configured range")
        with self._lock:
            self._temperature = float(temperature)

    def get_relation_bias(self, expected_relation: str, relation: str) -> float:
        return self._relation_bias_table.get_bias(expected_relation, relation)

    def update_relation_bias(self, expected_relation: str, relation: str, bias: float) -> None:
        self._relation_bias_table.update_bias(expected_relation, relation, bias)

    def replace_relation_biases(self, overrides: Dict[str, Dict[str, float]]) -> None:
        """Push EvolutionaryController theta into the walker (DEVIATION 9)."""
        self._relation_bias_table.replace_biases(overrides)

    def relation_bias_snapshot(self) -> Dict[str, Dict[str, float]]:
        return self._relation_bias_table.snapshot()

    def get_intent_bias(self, intent_id: int, relation: str) -> float:  # legacy shim
        return self._relation_bias_table.get_bias_for_intent(intent_id, relation)

    def update_intent_bias(self, intent_id: int, relation: str, bias: float) -> None:  # legacy shim
        expected = LEGACY_INTENT_TO_RELATION.get(int(intent_id), "associated_with")
        self._relation_bias_table.update_bias(expected, relation, bias)

    def apply_reward(
        self,
        reward: float,
        path_edges: List[str],
        path_relations: Optional[List[str]] = None,
        path_intents: Optional[List[int]] = None,  # legacy alias
        learning_rate: float = 0.01,
    ) -> None:
        if not path_edges:
            return
        if path_relations is None:
            path_relations = [LEGACY_INTENT_TO_RELATION.get(i, "associated_with") for i in (path_intents or [])]
        if not path_relations:
            return
        for step_idx, edge_type in enumerate(path_edges):
            expected = path_relations[min(step_idx, len(path_relations) - 1)]
            old_bias = self._relation_bias_table.get_bias(expected, edge_type)
            if old_bias <= 0.0:
                old_bias = 1.0
            delta = learning_rate * reward * (1.0 - old_bias / 3.0)
            new_bias = max(0.1, min(3.0, old_bias + delta))
            self._relation_bias_table.update_bias(expected, edge_type, new_bias)

    def next_possible_nodes(
        self,
        current_node: int,
        subgraph: Subgraph,
        expected_relation: str,
        visited: Optional[Iterable[int]] = None,
    ) -> List[Tuple[int, float]]:
        visited_list = list(visited or [])
        candidates = self._collect_candidates(current_node, subgraph, expected_relation, visited_list)
        scored = [self._scorer.score(candidate) for candidate in candidates]
        return [(candidate.node_id, score) for candidate, score in zip(candidates, scored)]

    def walk(self, subgraph: Subgraph, plan: Plan) -> WalkResult:
        self._validate_inputs(subgraph, plan)
        max_steps = min(self._walker_config.walk.max_steps, self._walker_config.path.max_length)
        chain = self._plan_chain(plan)
        if chain:
            max_steps = min(max_steps, len(chain))
        start_node = self._select_start_node(subgraph, restart=False)
        path = [start_node]
        path_edges: List[str] = []
        path_confidences: List[float] = []
        path_embeddings = [self._resolve_embedding(start_node, subgraph)]
        step_timestamps: List[float] = []

        record_act = self._walker_config.path.record_activations
        record_conf = self._walker_config.path.record_edge_confidence
        path_activations: List[float] = []
        if record_act:
            path_activations = [subgraph.node_activations[start_node]]
        if self._walker_config.path.record_timestamps:
            step_timestamps = [time.time()]

        restarts = 0
        max_restarts = len(subgraph.nodes)

        while len(path_edges) < max_steps:
            current_node = path[-1]
            current_activation = subgraph.node_activations[current_node]
            if current_activation < self._walker_config.walk.min_activation:
                break
            expected_relation = self._expected_relation_for_step(plan, len(path_edges))
            candidates = self._collect_candidates(
                current_node,
                subgraph,
                expected_relation,
                visited=path,
            )
            if not candidates:
                if not self._walker_config.walk.restart_on_dead_end:
                    break
                restarts += 1
                if restarts > max_restarts:
                    break
                if self._walker_config.debug.save_all_paths:
                    with self._lock:
                        self._saved_paths.append(list(path))
                start_node = self._select_start_node(subgraph, restart=True, visited=set(path))
                if start_node is None:
                    break
                path = [start_node]
                path_edges = []
                path_confidences = []
                path_embeddings = [self._resolve_embedding(start_node, subgraph)]
                path_activations = []
                if record_act:
                    path_activations = [subgraph.node_activations[start_node]]
                if self._walker_config.path.record_timestamps:
                    step_timestamps = [time.time()]
                continue

            raw_scores = [self._scorer.score(candidate) for candidate in candidates]
            adjusted_scores = self._apply_temperature(raw_scores)
            probabilities = self._scorer.normalize(adjusted_scores)
            if self._walker_config.debug.log_decision_scores:
                decision = WalkDecision(
                    current_node=current_node,
                    expected_relation=expected_relation,
                    candidates=candidates,
                    probabilities=probabilities,
                )
                self._logger.debug("Decision: %s", decision)

            next_index = self._select_index(probabilities, adjusted_scores)
            chosen = candidates[next_index]
            path.append(chosen.node_id)
            path_edges.append(chosen.edge_type)
            if record_act:
                path_activations.append(chosen.target_activation)
            if record_conf:
                path_confidences.append(chosen.confidence)
            path_embeddings.append(self._resolve_embedding(chosen.node_id, subgraph))
            if self._walker_config.path.record_timestamps:
                step_timestamps.append(time.time())

            if self._walker_config.debug.log_path_taken:
                self._logger.info(
                    "Walk step %d -> %d (%s)",
                    current_node,
                    chosen.node_id,
                    chosen.edge_type,
                )

        walk = WalkResult.build(
            path=path,
            path_edges=path_edges,
            path_activations=path_activations,
            path_confidences=path_confidences,
            path_embeddings=path_embeddings,
            plan=plan,
        )
        if self._walker_config.debug.save_all_paths:
            with self._lock:
                self._saved_paths.append(list(path))
        if self._walker_config.path.record_timestamps:
            with self._lock:
                self._last_step_timestamps = list(step_timestamps)
        walk_confidence = self.get_walk_confidence(walk)
        walk = WalkResult(
            **{**walk.__dict__, "walk_confidence": walk_confidence, "timestamp": time.time()}
        )
        walk.validate(self._core_config.activation.min, self._core_config.activation.max)
        return walk

    def get_statistics(self) -> Dict[str, float]:
        with self._lock:
            saved = list(self._saved_paths)
        total_paths = len(saved)
        total_steps = sum(len(p) for p in saved)
        avg_len = total_steps / max(total_paths, 1)
        return {
            "walk_count": total_paths,
            "total_steps": total_steps,
            "avg_path_length": avg_len,
        }

    def get_saved_paths(self) -> List[List[int]]:
        with self._lock:
            return list(self._saved_paths)

    def get_last_step_timestamps(self) -> List[float]:
        with self._lock:
            return list(self._last_step_timestamps)

    def compute_eligibility_trace(self, walk: WalkResult, subgraph: Subgraph) -> Dict[str, float]:
        return compute_eligibility_trace(
            path=walk.path,
            path_edges=walk.path_edges,
            path_activations=walk.path_activations,
            edge_strengths=subgraph.edge_strengths,
            edge_confidences=subgraph.edge_confidences,
            gamma=self._walker_config.eligibility.gamma,
            trace_key_format=self._walker_config.eligibility.trace_key_format,
        )

    def get_walk_confidence(self, walk: WalkResult) -> float:
        return geometric_mean(walk.path_confidences)

    def _validate_inputs(self, subgraph: Subgraph, plan: Plan) -> None:
        try:
            subgraph.validate(self._core_config.activation.min, self._core_config.activation.max)
            plan.validate()
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def _apply_temperature(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        if self._temperature <= 0.0:
            raise ValidationError("Temperature must be positive")
        return [score / self._temperature for score in scores]

    def _select_index(self, probabilities: List[float], scores: List[float]) -> int:
        if not probabilities:
            raise ValidationError("No probabilities to sample")
        if self._walker_config.scoring.normalization == "none":
            max_score = max(scores)
            return scores.index(max_score)
        threshold = self._rng.random()
        cumulative = 0.0
        for idx, probability in enumerate(probabilities):
            cumulative += probability
            if cumulative >= threshold:
                return idx
        return len(probabilities) - 1

    def _select_start_node(
        self,
        subgraph: Subgraph,
        restart: bool,
        visited: Optional[Iterable[int]] = None,
    ) -> Optional[int]:
        visited_set = set(visited or [])
        candidates = [node_id for node_id in subgraph.seed_nodes if node_id not in visited_set]
        if not candidates:
            candidates = [node_id for node_id in subgraph.nodes if node_id not in visited_set]
        if not candidates:
            return None
        penalty = self._walker_config.walk.restart_penalty if restart else 1.0
        return max(candidates, key=lambda node_id: subgraph.node_activations[node_id] * penalty)

    def _plan_chain(self, plan: Plan) -> List[str]:
        """Relation sequence the walk follows. Uses plan.relation_chain (DEVIATION 9);
        falls back to a legacy intent-derived chain only when the plan is intent-based."""
        if plan.relation_chain:
            return list(plan.relation_chain)
        if plan.intent_sequence:
            return [LEGACY_INTENT_TO_RELATION.get(i, "associated_with") for i in plan.intent_sequence]
        return []

    def _expected_relation_for_step(self, plan: Plan, step: int) -> str:
        """Relation the current walk step expects. Uses plan.relation_chain (DEVIATION 9);
        falls back to a legacy intent-derived chain only when the plan is intent-based."""
        chain = self._plan_chain(plan) or ["has_property"]
        return chain[min(step, len(chain) - 1)]

    def _collect_candidates(
        self,
        current_node: int,
        subgraph: Subgraph,
        expected_relation: str,
        visited: Optional[Iterable[int]] = None,
    ) -> List[ScoredCandidate]:
        visited_list = list(visited or [])
        visited_set = set(visited_list)
        candidates: List[ScoredCandidate] = []
        previous_node = None
        if len(visited_list) >= 2:
            previous_node = visited_list[-2]
        for source, target, relation in subgraph.edges:
            if source != current_node:
                continue
            if not self._walker_config.walk.allow_cycles and target in visited_set:
                continue
            if not self._walker_config.walk.allow_backtrack and previous_node is not None and target == previous_node:
                continue
            target_activation = subgraph.node_activations[target]
            if target_activation < self._walker_config.walk.min_activation:
                continue
            edge_key = (source, target, relation)
            strength = subgraph.edge_strengths[edge_key]
            confidence = subgraph.edge_confidences[edge_key]
            relation_bias = self._relation_bias_table.get_bias(expected_relation, relation)
            candidates.append(
                ScoredCandidate(
                    node_id=target,
                    edge_type=relation,
                    strength=strength,
                    confidence=confidence,
                    target_activation=target_activation,
                    intent_bias=relation_bias,
                    target_similarity=self._target_similarity(subgraph, target),
                    raw_score=0.0,
                )
            )
        return candidates

    def _target_similarity(self, subgraph: Subgraph, target: int) -> float:
        """Cosine similarity between the query embedding and the candidate node
        embedding (P1). Neutral 1.0 when either embedding is unavailable."""
        query_emb = subgraph.query_embedding
        if query_emb is None:
            return 1.0
        target_emb = None
        if subgraph.node_embeddings is not None and target in subgraph.node_embeddings:
            target_emb = subgraph.node_embeddings[target]
        else:
            try:
                target_emb = self._resolve_embedding(target, subgraph)
            except EmbeddingLookupError:
                target_emb = None
        if target_emb is None:
            return 1.0
        try:
            import numpy as np
            query_vec = np.asarray(query_emb, dtype=np.float32).reshape(-1)
            target_vec = np.asarray(target_emb, dtype=np.float32).reshape(-1)
            if query_vec.shape != target_vec.shape or query_vec.size == 0:
                return 1.0
            denom = float(np.linalg.norm(query_vec) * np.linalg.norm(target_vec))
            if denom <= 0.0:
                return 1.0
            return float(np.dot(query_vec, target_vec) / denom)
        except Exception:
            return 1.0

    def _resolve_embedding(self, node_id: int, subgraph: Subgraph) -> "object":
        if subgraph.node_embeddings is not None and node_id in subgraph.node_embeddings:
            return subgraph.node_embeddings[node_id]
        if self._embedding_provider is None:
            raise EmbeddingLookupError("No embedding provider configured")
        embedding = self._embedding_provider(node_id)
        if embedding is None:
            raise EmbeddingLookupError(f"Embedding not found for node {node_id}")
        return embedding
