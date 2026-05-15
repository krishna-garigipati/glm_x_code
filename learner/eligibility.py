import logging
from typing import Dict, Tuple, List

import numpy as np

from learner.types import Subgraph, WalkResult, EligibilityTrace
from learner.config import LearningConfig

logger = logging.getLogger(__name__)


class EligibilityTracer:
    def __init__(self, config: LearningConfig):
        self.gamma = config.hebbian.eligibility_gamma
        self.min_threshold = config.eligibility.min_eligibility_threshold
        self.trace_normalization = config.eligibility.trace_normalization
        self.walk_length_penalty = config.eligibility.walk_length_penalty
        self.temporal_discount_strength = config.eligibility.temporal_discount_strength

    def compute_trace(
        self,
        walk: WalkResult,
        subgraph: Subgraph,
    ) -> Dict[Tuple[int, int, str], float]:
        traces: Dict[Tuple[int, int, str], float] = {}
        path = walk.path
        path_edges = walk.path_edges
        path_activations = walk.path_activations
        walk_length = len(path_edges)

        length_penalty = 1.0 / (1.0 + self.walk_length_penalty * walk_length)

        for t in range(len(path_edges)):
            source = path[t]
            target = path[t + 1]
            relation = path_edges[t]
            edge_key = (source, target, relation)
            src_activation = path_activations[t]

            edge_strength = subgraph.edge_strengths.get(edge_key, 0.5)
            edge_confidence = subgraph.edge_confidences.get(edge_key, 0.5)
            temporal_factor = self._compute_temporal_factor(walk.timestamp, subgraph.timestamp)

            gamma_discount = self.gamma ** (t * (1.0 + self.temporal_discount_strength * t / max(walk_length, 1)))

            step_contribution = (
                gamma_discount
                * src_activation
                * edge_strength
                * edge_confidence
                * temporal_factor
                * length_penalty
            )

            if edge_key in traces:
                traces[edge_key] += step_contribution
            else:
                traces[edge_key] = step_contribution

            logger.debug(
                "Eligibility step %d: edge(%d,%d,%s) contrib=%.6f (γ=%.4f, A=%.4f, S*C=%.4f, temp=%.4f, len_pen=%.4f)",
                t, source, target, relation, step_contribution,
                gamma_discount, src_activation,
                edge_strength * edge_confidence, temporal_factor, length_penalty,
            )

        if self.trace_normalization and traces:
            total = sum(traces.values())
            if total > 0:
                for key in traces:
                    traces[key] /= total

        filtered = {}
        for key, val in traces.items():
            if val >= self.min_threshold:
                filtered[key] = val

        return filtered

    def compute_trace_with_details(
        self,
        walk: WalkResult,
        subgraph: Subgraph,
    ) -> List[EligibilityTrace]:
        trace_map: Dict[Tuple[int, int, str], List[float]] = {}
        path = walk.path
        path_edges = walk.path_edges
        path_activations = walk.path_activations
        walk_length = len(path_edges)
        length_penalty = 1.0 / (1.0 + self.walk_length_penalty * walk_length)

        for t in range(len(path_edges)):
            source = path[t]
            target = path[t + 1]
            relation = path_edges[t]
            edge_key = (source, target, relation)
            src_activation = path_activations[t]
            edge_strength = subgraph.edge_strengths.get(edge_key, 0.5)
            edge_confidence = subgraph.edge_confidences.get(edge_key, 0.5)
            temporal_factor = self._compute_temporal_factor(walk.timestamp, subgraph.timestamp)
            gamma_discount = self.gamma ** (t * (1.0 + self.temporal_discount_strength * t / max(walk_length, 1)))
            contribution = (
                gamma_discount
                * src_activation
                * edge_strength
                * edge_confidence
                * temporal_factor
                * length_penalty
            )
            if edge_key in trace_map:
                trace_map[edge_key].append(contribution)
            else:
                trace_map[edge_key] = [contribution]

        results = []
        for edge_key, contributions in trace_map.items():
            total = sum(contributions)
            if total >= self.min_threshold:
                results.append(EligibilityTrace(
                    edge_key=edge_key,
                    eligibility=total,
                    time_contribution=contributions,
                ))

        results.sort(key=lambda x: x.eligibility, reverse=True)
        return results

    def _compute_temporal_factor(self, walk_time: float, subgraph_time: float) -> float:
        dt = abs(walk_time - subgraph_time)
        if dt < 1e-6:
            return 1.0
        return 1.0 / (1.0 + self.temporal_discount_strength * np.log1p(dt))
