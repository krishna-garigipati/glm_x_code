from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging
import numpy as np

from .config import LoadedConfigs, build_default_theta, load_configs
from .energy import compute_activation_energy
from .es_controller import EvolutionaryController
from .tier1 import Tier1Resonance
from .tier2 import Tier2Resonance
from .types import GraphStore, Subgraph
from .validation import validate_embedding, validate_seeds, validate_theta

logger = logging.getLogger(__name__)


class ResonanceEngine:
    def __init__(self, config_dir: Path | str):
        config_dir = Path(config_dir)
        if not config_dir.is_dir():
            raise NotADirectoryError(f"Config directory not found: {config_dir}")

        logger.info("Loading configuration from %s", config_dir.resolve())
        self._configs: LoadedConfigs = load_configs(config_dir)
        self._theta = build_default_theta(self._configs.resonance, self._configs.core)

        self._tier1 = Tier1Resonance(
            core_config=self._configs.core,
            algorithm=self._configs.resonance.algorithm,
            temporal=self._configs.resonance.temporal,
            tier_config=self._configs.resonance.tier1,
            log_activation_history=self._configs.resonance.diagnostics.log_activation_history,
            history_buffer_size=self._configs.resonance.diagnostics.history_buffer_size,
        )
        self._tier2 = Tier2Resonance(
            core_config=self._configs.core,
            algorithm=self._configs.resonance.algorithm,
            temporal=self._configs.resonance.temporal,
            tier_config=self._configs.resonance.tier2,
            relation_bias=self._configs.resonance.tier1.relation_bias,
            log_activation_history=self._configs.resonance.diagnostics.log_activation_history,
            history_buffer_size=self._configs.resonance.diagnostics.history_buffer_size,
        )

        self._es = EvolutionaryController(
            core_config=self._configs.core,
            es_config=self._configs.resonance.es_controller,
            initial_theta=self._theta,
            theta_indices=self._configs.resonance.theta_indices,
            log_theta_history=self._configs.resonance.diagnostics.log_theta_history,
            history_buffer_size=self._configs.resonance.diagnostics.history_buffer_size,
        )

        logger.info("ResonanceEngine initialized (theta_dim=%d, tier2=%s)",
                     self._configs.resonance.es_controller.theta_dim,
                     self._configs.resonance.tier2.enabled)

    def resonate(
        self,
        query_embedding: np.ndarray,
        graph: GraphStore,
        initial_seeds: List[int],
        tier: int = 1,
    ) -> Subgraph:
        if tier not in (1, 2):
            raise ValueError(f"tier must be 1 or 2, got {tier}")
        validate_embedding(query_embedding, "query_embedding")
        validate_seeds(initial_seeds, "initial_seeds")

        logger.debug("Resonate tier=%d, seeds=%s", tier, initial_seeds)

        try:
            tier1_subgraph, _ = self._tier1.resonate(query_embedding, graph, initial_seeds)
        except Exception:
            logger.exception("Tier1 resonance failed")
            raise

        if tier == 1 or not self._configs.resonance.tier2.enabled:
            return tier1_subgraph

        tier1_energy = tier1_subgraph.activation_energy
        energy_threshold = self._configs.resonance.energy.tier1_threshold
        if self._configs.resonance.tier1.multiplied_at_runtime and self._configs.resonance.tier1.t_conf_coefficient:
            energy_threshold = (
                self._configs.resonance.tier1.t_conf_coefficient
                * len(initial_seeds)
                * self._configs.core.activation.max
            )

        if tier1_energy >= energy_threshold:
            logger.debug("Tier1 energy %.4f >= threshold %.4f, skipping Tier2", tier1_energy, energy_threshold)
            return tier1_subgraph

        try:
            tier2_subgraph, _ = self._tier2.resonate(query_embedding, graph, initial_seeds)
        except Exception:
            logger.exception("Tier2 resonance failed, falling back to Tier1")
            return tier1_subgraph

        if tier2_subgraph.activation_energy >= tier1_energy + self._configs.resonance.energy.tier2_min_improvement:
            logger.debug("Tier2 improvement sufficient, using Tier2 result")
            return tier2_subgraph

        return tier1_subgraph

    def resonate_with_theta(
        self,
        theta: np.ndarray,
        query_embedding: np.ndarray,
        graph: GraphStore,
        initial_seeds: List[int],
    ) -> Subgraph:
        validate_theta(theta, self._configs.resonance.es_controller.theta_dim)
        validate_embedding(query_embedding, "query_embedding")
        validate_seeds(initial_seeds, "initial_seeds")

        tier1 = self._override_tier1(theta)
        subgraph, _ = tier1.resonate(query_embedding, graph, initial_seeds)
        return subgraph

    def get_theta(self) -> np.ndarray:
        return self._es.get_theta()

    def set_theta(self, theta: np.ndarray) -> None:
        self._es.set_theta(theta)

    def propose_theta_mutation(self) -> np.ndarray:
        return self._es.propose_theta_mutation()

    def update_es_with_reward(self, reward: float, theta_used: np.ndarray) -> None:
        self._es.update_es_with_reward(reward, theta_used)

    def compute_activation_energy(self, subgraph: Subgraph) -> float:
        return compute_activation_energy(subgraph)

    def check_resonance_convergence(
        self, activation_history: List[np.ndarray], epsilon: float = 0.001
    ) -> bool:
        if len(activation_history) < 2:
            return False
        diff = np.abs(activation_history[-1] - activation_history[-2])
        return bool(np.max(diff) < epsilon)

    def get_analogy_leaps(self, target_node: int, graph: GraphStore, top_k: int = 3):
        return self._tier2._analogy_finder.get_analogy_leaps(target_node, graph, top_k=top_k)

    def clear_analogy_cache(self) -> None:
        self._tier2._analogy_finder.clear_cache()

    def get_configs(self) -> LoadedConfigs:
        return self._configs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug("ResonanceEngine context exit")
        return False

    def _override_tier1(self, theta: np.ndarray) -> Tier1Resonance:
        idx = self._configs.resonance.theta_indices
        tier1 = self._configs.resonance.tier1

        relation_bias = dict(tier1.relation_bias)
        for offset, relation in enumerate(self._configs.core.relations):
            relation_bias[relation] = float(theta[idx.relation_bias_start + offset])

        override = type(tier1)(
            propagation_threshold=float(theta[idx.propagation_threshold]),
            edge_threshold=float(theta[idx.edge_threshold]),
            decay_lambda=float(theta[idx.decay_lambda]),
            top_k=int(theta[idx.top_k]),
            max_iterations=tier1.max_iterations,
            relation_bias=relation_bias,
            energy_threshold_formula=tier1.energy_threshold_formula,
            t_conf_coefficient=tier1.t_conf_coefficient,
            multiplied_at_runtime=tier1.multiplied_at_runtime,
        )

        return Tier1Resonance(
            core_config=self._configs.core,
            algorithm=self._configs.resonance.algorithm,
            temporal=self._configs.resonance.temporal,
            tier_config=override,
            log_activation_history=self._configs.resonance.diagnostics.log_activation_history,
            history_buffer_size=self._configs.resonance.diagnostics.history_buffer_size,
        )
