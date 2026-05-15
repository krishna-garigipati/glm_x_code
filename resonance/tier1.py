from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging
import threading
import time
import numpy as np

from .config import AlgorithmConfig, CoreConfig, TierConfig, TemporalConfig
from .energy import compute_activation_energy
from .temporal import compute_temporal_factor
from .types import GraphStore, Subgraph
from .validation import validate_embedding, validate_seeds, validate_subgraph

logger = logging.getLogger(__name__)


class Tier1Resonance:
    def __init__(
        self,
        core_config: CoreConfig,
        algorithm: AlgorithmConfig,
        temporal: TemporalConfig,
        tier_config: TierConfig,
        log_activation_history: bool,
        history_buffer_size: int,
    ):
        self._core = core_config
        self._algorithm = algorithm
        self._temporal = temporal
        self._tier = tier_config
        self._log_activation_history = log_activation_history
        self._history_buffer_size = max(1, history_buffer_size)
        self._lock = threading.Lock()

    def resonate(
        self,
        query_embedding: np.ndarray,
        graph: GraphStore,
        initial_seeds: List[int],
        max_iterations: Optional[int] = None,
        initial_activations: Optional[Dict[int, float]] = None,
    ) -> Tuple[Subgraph, List[np.ndarray]]:
        validate_embedding(query_embedding, "query_embedding")
        validate_seeds(initial_seeds, "initial_seeds")

        if max_iterations is not None and max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

        activation_history: List[np.ndarray] = []
        activations = self._initialize_activations(initial_seeds, initial_activations)
        edges: Dict[Tuple[int, int, str], Tuple[float, float]] = {}

        iterations = max_iterations if max_iterations is not None else self._tier.max_iterations
        for step in range(iterations):
            try:
                activations = self._propagate(graph, activations, edges)
            except Exception:
                logger.exception("Propagation failed at step %d/%d", step + 1, iterations)
                raise

            if self._log_activation_history:
                activation_history.append(self._activation_vector(activations))
                if len(activation_history) > self._history_buffer_size:
                    activation_history.pop(0)

            if self._check_convergence(activation_history, self._core.resonance.convergence_epsilon):
                logger.debug("Tier1 converged at step %d/%d", step + 1, iterations)
                break

        subgraph = self._build_subgraph(
            query_embedding=query_embedding,
            activations=activations,
            edges=edges,
            seed_nodes=initial_seeds,
            tier_used=1,
        )

        try:
            validate_subgraph(subgraph, self._core)
        except ValueError as e:
            logger.error("Subgraph validation failed: %s", e)
            raise

        return subgraph, activation_history

    def _initialize_activations(
        self, seed_nodes: List[int], initial_activations: Optional[Dict[int, float]]
    ) -> Dict[int, float]:
        activations = {}
        if initial_activations:
            for k, v in initial_activations.items():
                activations[k] = float(np.clip(v, self._core.activation.min, self._core.activation.max))

        for node_id in seed_nodes:
            current = activations.get(node_id, self._core.activation.max)
            activations[node_id] = float(max(current, self._core.activation.max))

        return activations

    def _propagate(
        self,
        graph: GraphStore,
        activations: Dict[int, float],
        edges: Dict[Tuple[int, int, str], Tuple[float, float]],
    ) -> Dict[int, float]:
        inputs: Dict[int, float] = {}

        for node_id, activation in activations.items():
            try:
                neighbors = graph.get_neighbors(node_id)
            except Exception:
                logger.warning("Failed to get neighbors for node %s, skipping", node_id)
                continue

            if not self._check_finite(activation, "activation"):
                continue

            for neighbor_id, edge in neighbors:
                edge_weight = edge.strength * edge.confidence
                if not self._check_finite(edge_weight, "edge_weight"):
                    continue
                if edge_weight < self._tier.edge_threshold:
                    continue

                relation_bias = self._tier.relation_bias.get(edge.relation_type, 1.0)
                temporal_factor = 1.0
                if self._algorithm.temporal_factor_enabled:
                    temporal_factor = compute_temporal_factor(
                        edge.last_used, edge.frequency,
                        self._temporal.gamma, self._temporal.frequency_threshold,
                    )

                input_value = activation * edge_weight * relation_bias * temporal_factor
                if not self._check_finite(input_value, "input_value"):
                    continue

                inputs[neighbor_id] = inputs.get(neighbor_id, 0.0) + input_value
                edges[(node_id, neighbor_id, edge.relation_type)] = (edge.strength, edge.confidence)

        all_ids = set(activations.keys()) | set(inputs.keys())
        updated: Dict[int, float] = {}
        for node_id in all_ids:
            current = activations.get(node_id, self._core.activation.default)
            input_value = inputs.get(node_id, 0.0)
            updated[node_id] = self._apply_dynamics(current, input_value)

        updated = self._normalize(updated)
        updated = self._gate(updated)
        return updated

    def _apply_dynamics(self, current: float, input_value: float) -> float:
        if not self._check_finite(current, "current") or not self._check_finite(input_value, "input_value"):
            return self._core.activation.default

        prop_type = self._algorithm.propagation_type
        try:
            if prop_type == "wilson_cowan":
                delta = (1.0 - current) * input_value - self._tier.decay_lambda * current
            elif prop_type == "simple_diffusion":
                delta = input_value - self._tier.decay_lambda * current
            elif prop_type == "threshold":
                if input_value < self._tier.propagation_threshold:
                    delta = -self._tier.decay_lambda * current
                else:
                    delta = input_value - self._tier.decay_lambda * current
            else:
                raise ValueError(f"Unsupported propagation_type: {prop_type}")

            if not self._check_finite(delta, "delta"):
                return self._core.activation.default

            updated = current + delta
            return np.float32(np.clip(updated, self._core.activation.min, self._core.activation.max))
        except Exception:
            logger.exception("Dynamics computation failed")
            return np.float32(np.clip(current, self._core.activation.min, self._core.activation.max))

    def _normalize(self, activations: Dict[int, float]) -> Dict[int, float]:
        if self._algorithm.normalization == "none":
            return activations

        total = sum(activations.values())
        if total <= 0.0 or not np.isfinite(total):
            return activations

        if self._algorithm.normalization == "l1_norm":
            return {nid: np.float32(np.clip(v / total, self._core.activation.min, self._core.activation.max)) for nid, v in activations.items()}

        if self._algorithm.normalization == "budget_soft_cap":
            if total <= self._core.resonance.budget_max:
                return activations
            scale = self._core.resonance.budget_max / total
            return {nid: np.float32(np.clip(v * scale, self._core.activation.min, self._core.activation.max)) for nid, v in activations.items()}

        raise ValueError(f"Unsupported normalization: {self._algorithm.normalization}")

    def _gate(self, activations: Dict[int, float]) -> Dict[int, float]:
        if self._algorithm.gate_type == "none":
            return activations

        if self._algorithm.gate_type == "threshold":
            return {
                nid: val for nid, val in activations.items()
                if val >= self._tier.propagation_threshold
            }

        if self._algorithm.gate_type == "top_k":
            sorted_nodes = sorted(activations.items(), key=lambda item: item[1], reverse=True)
            return dict(sorted_nodes[: self._tier.top_k])

        raise ValueError(f"Unsupported gate_type: {self._algorithm.gate_type}")

    def _build_subgraph(
        self,
        query_embedding: np.ndarray,
        activations: Dict[int, float],
        edges: Dict[Tuple[int, int, str], Tuple[float, float]],
        seed_nodes: List[int],
        tier_used: int,
    ) -> Subgraph:
        nodes = list(activations.keys())
        edge_list: List[Tuple[int, int, str]] = []
        edge_strengths: Dict[Tuple[int, int, str], float] = {}
        edge_confidences: Dict[Tuple[int, int, str], float] = {}

        for edge_key, (strength, confidence) in edges.items():
            if edge_key[0] in activations and edge_key[1] in activations:
                edge_list.append(edge_key)
                edge_strengths[edge_key] = strength
                edge_confidences[edge_key] = confidence

        emb = np.ascontiguousarray(query_embedding.astype(np.float32))
        subgraph = Subgraph(
            nodes=nodes,
            node_activations=activations,
            edges=edge_list,
            edge_strengths=edge_strengths,
            edge_confidences=edge_confidences,
            seed_nodes=seed_nodes,
            tier_used=tier_used,
            activation_energy=0.0,
            query_embedding=emb,
            timestamp=time.time(),
        )
        energy = compute_activation_energy(subgraph)
        return Subgraph(
            nodes=subgraph.nodes,
            node_activations=subgraph.node_activations,
            edges=subgraph.edges,
            edge_strengths=subgraph.edge_strengths,
            edge_confidences=subgraph.edge_confidences,
            seed_nodes=subgraph.seed_nodes,
            tier_used=subgraph.tier_used,
            activation_energy=energy,
            query_embedding=subgraph.query_embedding,
            timestamp=subgraph.timestamp,
        )

    def _activation_vector(self, activations: Dict[int, float]) -> np.ndarray:
        if not activations:
            return np.array([], dtype=np.float32)
        ordered = [v for _, v in sorted(activations.items(), key=lambda item: item[0])]
        return np.array(ordered, dtype=np.float32)

    @staticmethod
    def _check_convergence(history: List[np.ndarray], epsilon: float) -> bool:
        if len(history) < 2:
            return False
        if history[-1].shape != history[-2].shape:
            return False
        diff = np.abs(history[-1] - history[-2])
        return bool(np.max(diff) < epsilon)

    @staticmethod
    def _check_finite(val: float, name: str) -> bool:
        if not np.isfinite(val):
            logger.warning("Non-finite %s: %s", name, val)
            return False
        return True
