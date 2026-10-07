"""Graph Walker implementation for Team D."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
import time
from threading import RLock
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import CoreConfig, WalkerConfig
from .eligibility import compute_eligibility_trace
from .exceptions import EmbeddingLookupError, ValidationError
from .relation_bias import LEGACY_INTENT_TO_RELATION, RelationBiasTable
from .models import Plan, Subgraph, WalkResult
from .path_scorer import PathScorer, ScoredCandidate
from .utils import geometric_mean

# Canonical inverse relation labels. The reverse-edge mirror pass (glmx_ask.ask)
# emits each stored edge in both directions, labelling the mirrored copy with the
# inverse relation so a reversed traversal never masquerades as the forward one.
# Canonical single source of truth: glmx_ask imports this map rather than
# redefining it, so the mirror pass and the walker's candidate matching cannot
# drift apart.
INVERSE_RELATION_LABELS = {
    "causes": "caused_by",
    "caused_by": "causes",
    "precedes": "follows",
    "follows": "precedes",
    "part_of": "has_part",
    "has_part": "part_of",
    "is_a": "is_a",
}


def inverse_relations(relation: str) -> set:
    """Relations that denote the same directed edge as `relation`: the relation
    itself plus its mirrored counterpart. A walk that starts from the subject of
    "What is part of a storm?" traverses the `has_part` mirror of the stored
    `X part_of storm` edge, so both labels must count as the asked relation."""
    labels = {relation}
    inverse = INVERSE_RELATION_LABELS.get(relation)
    if inverse:
        labels.add(inverse)
    return labels


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
        force_argmax: bool = False,
    ) -> None:
        self._walker_config = walker_config
        self._core_config = core_config
        self._validate_config_consistency()
        self._validate_scoring_formula()
        self._relation_bias_table = RelationBiasTable(walker_config.relation_biases)
        self._scorer = PathScorer(
            normalization=walker_config.scoring.normalization,
            softmax_temperature=walker_config.scoring.softmax_temperature,
        )
        min_t, max_t = walker_config.walk.temperature_range
        if not (min_t <= walker_config.scoring.softmax_temperature <= max_t):
            raise ValidationError("Configured softmax_temperature is outside range")
        self.force_argmax = bool(force_argmax)
        if force_argmax:
            # Deterministic mode: pick the max-score candidate (first on ties)
            # instead of sampling from the softmax distribution, so seed-to-seed
            # runs are byte-identical. Walk contract/gating is untouched.
            self._scorer.normalization = "none"
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
        if core_walker.default_temperature != self._walker_config.scoring.softmax_temperature:
            mismatches.append(
                f"softmax_temperature: core.default_temperature={core_walker.default_temperature} walker.scoring.softmax_temperature={self._walker_config.scoring.softmax_temperature}"
            )
        if mismatches:
            raise ValidationError(
                "CoreConfig/WalkerConfig mismatch: " + "; ".join(mismatches)
            )

    def _validate_scoring_formula(self) -> None:
        """Contract v3.3.2 section 9 fixes the formula to exactly four terms:

            score = strength * confidence * target_activation * relation_bias

        Reject any declared formula that is missing one of them, and reject any
        formula that reintroduces cosine/target_similarity (section 9:
        "cosine_similarity: REMOVED from the main scoring formula", and section
        17 forbidden list: "Putting strong cosine similarity back into Walker
        scoring").
        """
        formula = self._walker_config.scoring.formula
        required_terms = ["strength", "confidence", "target_activation", "relation_bias"]
        for term in required_terms:
            if term not in formula:
                raise ValidationError(
                    f"Scoring formula does not contain required term '{term}': {formula}"
                )
        forbidden_terms = ["target_similarity", "cosine", "intent_bias"]
        for term in forbidden_terms:
            if term in formula:
                raise ValidationError(
                    f"Scoring formula must not contain '{term}' "
                    f"(contract v3.3.2 section 9 removes cosine from Walker scoring): {formula}"
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
        """LEGACY. Maps a legacy intent id to its relation and updates that
        relation's bias. Offline-only; section 17 forbids intent_sequence as an
        inference-time driver."""
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
            # No relation chain supplied: nothing authoritative to reinforce
            # against (section 6). Legacy intent ids are deliberately NOT used
            # to reconstruct the expected relation.
            return
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
            probabilities = self._scorer.normalize(raw_scores)
            if self._walker_config.debug.log_decision_scores:
                decision = WalkDecision(
                    current_node=current_node,
                    expected_relation=expected_relation,
                    candidates=candidates,
                    probabilities=probabilities,
                )
                self._logger.debug("Decision: %s", decision)

            next_index = self._select_index(
                probabilities, raw_scores, candidates, expected_relation
            )
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
        """Delegating wrapper around PathScorer.apply_temperature.

        Temperature scaling is scorer arithmetic. This method exists so the
        walk loop and its callers share one implementation instead of
        re-deriving the division here.
        """
        return self._scorer.apply_temperature(scores, self._temperature)

    def _select_index(
        self,
        probabilities: List[float],
        scores: List[float],
        candidates: Optional[List[ScoredCandidate]] = None,
        expected_relation: Optional[str] = None,
    ) -> int:
        if not probabilities:
            raise ValidationError("No probabilities to sample")
        # DEVIATION 9: the extractor chain is the source of truth. When a
        # candidate edge matches the relation the current walk step expects,
        # prefer it over merely-similar edges whose higher target activation
        # or similarity would otherwise dominate (e.g. `lemon -> sour
        # has_property` losing to the hotter `lemon -> fruit is_a`). Fall back
        # to the normal score-based selection only when no exact match exists.
        if candidates is not None and expected_relation is not None:
            exact = [
                idx
                for idx, candidate in enumerate(candidates)
                if candidate.edge_type == expected_relation
            ]
            if exact:
                return max(exact, key=lambda idx: scores[idx])
        if self._scorer.normalization == "none":
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
        """Relation sequence the walk follows (section 6: relation_chain only).

        intent_sequence is DORMANT (section 17 forbids it as a driving signal),
        so there is deliberately no intent-derived fallback here. A plan without
        a relation_chain yields an empty chain and no walk, which the pipeline
        surfaces through the honesty gate rather than by guessing a relation.
        """
        if plan.relation_chain:
            return list(plan.relation_chain)
        return []

    def _expected_relation_for_step(self, plan: Plan, step: int) -> str:
        """Relation the current walk step expects, from plan.relation_chain."""
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
                    relation_bias=relation_bias,
                    raw_score=0.0,
                )
            )
        # Contract v3.3.2 section 9 (candidate_selection.hard_filter): "Prefer
        # (or keep only) edges whose relation matches the expected relation from
        # the chain or its mirror." The filter is HARD: if no candidate carries
        # the expected relation (or its mirror) we return an empty candidate set
        # so the walk stops and the no_valid_path honesty gate can fire. An
        # earlier fall-through returned the full candidate set here, which let a
        # semantically similar but relationally wrong node win and the decoder
        # then assert a relation that was never walked.
        if expected_relation:
            asked = inverse_relations(expected_relation)
            matching = [c for c in candidates if c.edge_type in asked]
            return matching
        return candidates

    def _resolve_embedding(self, node_id: int, subgraph: Subgraph) -> "object":
        if subgraph.node_embeddings is not None and node_id in subgraph.node_embeddings:
            return subgraph.node_embeddings[node_id]
        if self._embedding_provider is None:
            raise EmbeddingLookupError("No embedding provider configured")
        embedding = self._embedding_provider(node_id)
        if embedding is None:
            raise EmbeddingLookupError(f"Embedding not found for node {node_id}")
        return embedding
