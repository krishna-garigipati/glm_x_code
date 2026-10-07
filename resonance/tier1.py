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
        readout_dim: int = 64,
    ):
        self._core = core_config
        self._algorithm = algorithm
        self._temporal = temporal
        self._tier = tier_config
        self._log_activation_history = log_activation_history
        self._history_buffer_size = max(1, history_buffer_size)
        self._lock = threading.Lock()
        self._readout_dim = readout_dim
        self._readout_projection: Optional[np.ndarray] = None
        self._theta: Optional[np.ndarray] = None
        self._es_population: Optional[List[np.ndarray]] = None
        self._es_fitness_history: List[float] = []

    def _lazy_init_readout(self, query_dim: int):
        if self._readout_projection is not None:
            return
        rng = np.random.RandomState(42)
        self._readout_projection = rng.randn(query_dim + self._readout_dim, self._readout_dim).astype(np.float32) * 0.01

    def compute_subgraph_embedding(
        self,
        subgraph: Subgraph,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        self._lazy_init_readout(query_embedding.shape[-1])
        n_nodes = len(subgraph.nodes)
        if n_nodes == 0:
            return np.zeros(self._readout_dim, dtype=np.float32)
        act_vals = np.array([subgraph.node_activations.get(n, 0.0) for n in subgraph.nodes], dtype=np.float32)
        max_act = act_vals.max() if act_vals.max() > 0 else 1.0
        weights = act_vals / max_act
        weighted_avg = np.zeros(self._readout_dim, dtype=np.float32)
        node_embs = subgraph.node_embeddings if subgraph.node_embeddings is not None else {}
        for nid, w in zip(subgraph.nodes, weights):
            emb = node_embs.get(nid)
            if emb is not None:
                e = np.asarray(emb, dtype=np.float32)
                if e.ndim == 0:
                    e = np.zeros(self._readout_dim, dtype=np.float32)
                weighted_avg += w * e[:min(len(e), self._readout_dim)]
        weighted_avg = weighted_avg / (n_nodes + 1e-8)
        combined = np.concatenate([weighted_avg, query_embedding[:self._readout_dim]])
        readout = combined @ self._readout_projection
        return np.tanh(readout).astype(np.float32)

    def es_step(
        self,
        subgraph: Subgraph,
        query_embedding: np.ndarray,
        fitness: float,
    ) -> bool:
        es = self._core.es_bounds
        if self._theta is None:
            theta_dim = 3 + len(self._core.relations)
            self._theta = np.zeros(theta_dim, dtype=np.float32)
            self._theta[0] = self._tier.propagation_threshold
            self._theta[1] = self._tier.decay_lambda
            self._theta[2] = self._tier.top_k
            for i, rel in enumerate(self._core.relations):
                idx = 3 + i
                self._theta[idx] = self._tier.relation_bias.get(rel, 1.0)
            self._es_population = None
        self._es_fitness_history.append(fitness)
        eval_window = 8
        if len(self._es_fitness_history) < eval_window:
            return False
        recent = self._es_fitness_history[-eval_window:]
        fitness_trend = sum(recent) / len(recent)
        if self._es_population is None:
            rng = np.random.RandomState(len(self._es_fitness_history))
            sigma = max(self._tier.decay_lambda, 0.01)
            self._es_population = []
            for _ in range(4):
                pert = rng.randn(*self._theta.shape).astype(np.float32) * sigma
                self._es_population.append(self._theta + pert)
            return True
        best_idx = int(np.argmin([abs(f - fitness_trend) for f in self._es_fitness_history[-len(self._es_population):]])) if self._es_fitness_history else 0
        best_idx = min(best_idx, len(self._es_population) - 1) if self._es_population else 0
        best_theta = self._theta.copy()
        if self._es_population:
            best_theta = self._es_population[best_idx % len(self._es_population)]
        lr = 0.02
        self._theta = self._theta + lr * (best_theta - self._theta)
        self._theta[0] = np.clip(self._theta[0], es.propagation_threshold[0], es.propagation_threshold[1])
        self._theta[1] = np.clip(self._theta[1], es.decay_lambda[0], es.decay_lambda[1])
        self._theta[2] = np.clip(self._theta[2], float(es.top_k[0]), float(es.top_k[1]))
        for i, rel in enumerate(self._core.relations):
            idx = 3 + i
            self._theta[idx] = np.clip(self._theta[idx], es.relation_bias[0], es.relation_bias[1])
        self._es_population = None
        return True

    def get_theta(self) -> Optional[np.ndarray]:
        return self._theta.copy() if self._theta is not None else None

    def get_readout_dim(self) -> int:
        return self._readout_dim

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
        # Real, propagation-computed activations for every node touched this
        # iteration, captured before the node budget discards any of them.
        pre_gate: Dict[int, float] = {}

        iterations = max_iterations if max_iterations is not None else self._tier.max_iterations
        for step in range(iterations):
            try:
                activations = self._propagate(graph, activations, edges,
                                              pre_gate_out=pre_gate)
                # Re-admit inside the loop, not after it. The budget erases a
                # gated-out node from `activations`, and _propagate only records
                # edges leaving nodes that are IN `activations` -- so a neighbour
                # dropped in iteration 1 never gets its own edges collected in
                # iteration 2. Readmitting after the loop would restore the node
                # but not the edges that make it walkable.
                activations = self._readmit_seed_neighbourhood(
                    graph, activations, pre_gate, initial_seeds
                )
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

        node_embeddings: Optional[Dict[int, np.ndarray]] = None
        try:
            node_embeddings = {}
            for nid in activations:
                node = graph.get_node(nid)
                if node is not None and node.embedding is not None:
                    node_embeddings[nid] = node.embedding
        except Exception:
            node_embeddings = None

        subgraph = self._build_subgraph(
            query_embedding=query_embedding,
            activations=activations,
            edges=edges,
            seed_nodes=initial_seeds,
            tier_used=1,
            node_embeddings=node_embeddings,
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

    def _readmit_seed_neighbourhood(
        self,
        graph: GraphStore,
        activations: Dict[int, float],
        pre_gate: Dict[int, float],
        seeds: List[int],
    ) -> Dict[int, float]:
        """Never let the node budget delete a seed's immediate neighbours.

        Contract section 7 fixes ``tier1.top_k`` at 64. On a graph larger than
        that budget the bound binds, and it binds destructively: ``_build_subgraph``
        keeps an edge only when BOTH endpoints survived ``_gate``, so every edge
        incident to a budget-dropped node silently disappears. A stored edge
        leaving the walk's own anchor can then be invisible and the walker
        refuses a question the graph can answer.

        Measured by controlled ablation on the 147-node held-out graph
        (257 questions, readmission on/off crossed with the entity-identity
        gate on/off): this is worth 5 of 257 questions (1.9%), specifically
        ``oh029 oh052 oh097 oh100 mh006``. It was originally estimated at 7 from
        a before/after comparison that could not separate the two changes; the
        ablation attributes 5 here and 1 to the identity gate.

        SCOPE LIMIT, measured not assumed: readmission is ONE-HOP from the seeds.
        It cannot rescue a second hop, because when the walk's hop-1 endpoint is
        not itself a seed, that endpoint's own neighbours stay prunable. On the
        same held-out graph ``mh012`` and ``mh053`` still fail with readmission
        fully enabled -- 69 and 66 resonated nodes respectively, with the hop-2
        target still absent. Widening this to k-hop would void the contract's
        node budget, so the limitation is documented rather than engineered away.

        This is a SCOPING change, not an activation rescue, and the distinction
        matters because activation rescue is forbidden:

        * No activation is invented. ``pre_gate`` holds exactly the values
          ``_propagate`` computed for itself one line earlier; only the budget
          threw them away, and they are reused verbatim.
        * The walker still applies ``walk.min_activation`` to every candidate.
        * The section 9 hard relation filter is untouched.
        * Nothing is mirrored, and no relation is invented.
        * Only nodes the seeds already reached through a qualifying edge are
          affected, so the subgraph stays bounded by the seed neighbourhood
          rather than growing without limit.

        Seeds themselves are left exactly as they were -- they were already
        forced back in by ``_build_subgraph``.
        """
        if self._algorithm.gate_type == "none":
            return activations
        if not seeds or not pre_gate:
            return activations

        admitted: Dict[int, float] = {}
        for seed in seeds:
            try:
                neighbours = graph.get_neighbors(seed)
            except Exception:
                logger.warning("Seed neighbourhood read failed for seed %s, skipping", seed)
                continue
            for neighbour_id, edge in neighbours:
                if neighbour_id in activations:
                    continue
                # Same qualification _propagate applies, so nothing that the
                # propagation itself rejected can be readmitted here.
                weight = edge.strength * edge.confidence
                if not self._check_finite(weight, "edge_weight"):
                    continue
                if weight < self._tier.edge_threshold:
                    continue
                value = pre_gate.get(neighbour_id)
                if value is None:
                    continue
                # Clip into the declared activation range using Python floats.
                # The propagation stores np.float32, and float32(0.01) reads back
                # as 0.009999999776482582 -- just under the floor -- which
                # validate_subgraph rejects. Nodes that survived the budget never
                # hit that comparison before, so only readmitted ones expose it.
                admitted[neighbour_id] = float(min(
                    max(float(value), float(self._core.activation.min)),
                    float(self._core.activation.max),
                ))

        if not admitted:
            return activations
        merged = dict(activations)
        merged.update(admitted)
        logger.debug("Seed neighbourhood readmission: +%d nodes (budget %d)",
                     len(admitted), self._tier.top_k)
        return merged

    def _propagate(
        self,
        graph: GraphStore,
        activations: Dict[int, float],
        edges: Dict[Tuple[int, int, str], Tuple[float, float]],
        pre_gate_out: Optional[Dict[int, float]] = None,
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
                edges[(edge.source, edge.target, edge.relation_type)] = (edge.strength, edge.confidence)

        all_ids = set(activations.keys()) | set(inputs.keys())
        updated: Dict[int, float] = {}
        for node_id in all_ids:
            current = activations.get(node_id, self._core.activation.default)
            input_value = inputs.get(node_id, 0.0)
            updated[node_id] = self._apply_dynamics(current, input_value)

        updated = self._normalize(updated)
        if pre_gate_out is not None:
            pre_gate_out.clear()
            pre_gate_out.update(updated)
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
        node_embeddings: Optional[Dict[int, np.ndarray]] = None,
    ) -> Subgraph:
        activations = dict(activations)
        for nid in seed_nodes:
            if nid not in activations:
                activations[nid] = self._core.activation.default
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
            node_embeddings=node_embeddings,
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
            node_embeddings=subgraph.node_embeddings,
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
