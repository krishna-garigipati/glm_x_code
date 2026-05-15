from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from .analogy import AnalogyFinder
from .config import AlgorithmConfig, CoreConfig, Tier2Config, TemporalConfig
from .tier1 import Tier1Resonance
from .types import GraphStore, Subgraph
from .validation import validate_seeds

logger = logging.getLogger(__name__)


class Tier2Resonance:
    def __init__(
        self,
        core_config: CoreConfig,
        algorithm: AlgorithmConfig,
        temporal: TemporalConfig,
        tier_config: Tier2Config,
        relation_bias: Dict[str, float],
        log_activation_history: bool,
        history_buffer_size: int,
    ):
        self._core = core_config
        self._tier = tier_config
        self._propagator = Tier1Resonance(
            core_config=core_config,
            algorithm=algorithm,
            temporal=temporal,
            tier_config=_tier1_view(tier_config, relation_bias),
            log_activation_history=log_activation_history,
            history_buffer_size=history_buffer_size,
        )
        self._analogy_finder = AnalogyFinder(core_config, tier_config.analogy_parameters)

    def resonate(
        self,
        query_embedding: np.ndarray,
        graph: GraphStore,
        initial_seeds: List[int],
    ) -> Tuple[Subgraph, List[np.ndarray]]:
        if not self._tier.enabled:
            raise RuntimeError("Tier2 resonance is disabled")

        validate_seeds(initial_seeds, "initial_seeds")

        initial_activations: Optional[Dict[int, float]] = None
        if self._tier.analogies_enabled:
            try:
                initial_activations = self._apply_analogies(graph, initial_seeds)
            except Exception:
                logger.exception("Analogy pre-activation failed, proceeding without it")
                initial_activations = None

        return self._propagator.resonate(
            query_embedding=query_embedding,
            graph=graph,
            initial_seeds=initial_seeds,
            max_iterations=self._tier.max_iterations,
            initial_activations=initial_activations,
        )

    def _apply_analogies(self, graph: GraphStore, seed_nodes: List[int]) -> Dict[int, float]:
        if not seed_nodes:
            return {}

        analog_activations: Dict[int, float] = {}
        for seed in seed_nodes:
            try:
                analogs = self._analogy_finder.get_analogy_leaps(seed, graph, top_k=3)
            except Exception:
                logger.debug("Analogy search failed for seed %s, skipping", seed)
                continue

            temp_strength = self._tier.analogy_parameters.temp_edge_strength
            max_activation = self._core.activation.max
            for node_id, similarity in analogs:
                activation = max_activation * temp_strength
                if node_id in analog_activations:
                    analog_activations[node_id] = max(analog_activations[node_id], activation * similarity)
                else:
                    analog_activations[node_id] = activation * similarity

        if not analog_activations:
            return {}

        try:
            subgraph, _ = self._propagator.resonate(
                query_embedding=np.zeros((384,), dtype=np.float32),
                graph=graph,
                initial_seeds=list(analog_activations.keys()),
                max_iterations=self._tier.analogy_parameters.mini_propagation_steps,
                initial_activations=analog_activations,
            )
        except Exception:
            logger.exception("Mini-propagation for analogies failed, using raw analog activations")
            return analog_activations

        return subgraph.node_activations


def _tier1_view(tier2: Tier2Config, relation_bias: Dict[str, float]):
    from .config import TierConfig

    return TierConfig(
        propagation_threshold=tier2.propagation_threshold,
        edge_threshold=tier2.edge_threshold,
        decay_lambda=tier2.decay_lambda,
        top_k=tier2.top_k,
        max_iterations=tier2.max_iterations,
        relation_bias=relation_bias,
        energy_threshold_formula=None,
        t_conf_coefficient=None,
        multiplied_at_runtime=None,
    )
