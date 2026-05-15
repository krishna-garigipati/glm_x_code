"""Edge scoring and normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .utils import softmax


@dataclass(frozen=True)
class ScoredCandidate:
    node_id: int
    edge_type: str
    strength: float
    confidence: float
    target_activation: float
    intent_bias: float
    raw_score: float


class PathScorer:
    def __init__(
        self,
        weight_strength: float,
        weight_confidence: float,
        weight_target_activation: float,
        weight_intent_bias: float,
        normalization: str,
        softmax_temperature: float,
    ) -> None:
        self.weight_strength = weight_strength
        self.weight_confidence = weight_confidence
        self.weight_target_activation = weight_target_activation
        self.weight_intent_bias = weight_intent_bias
        self.normalization = normalization
        self.softmax_temperature = softmax_temperature

    def score(self, candidate: ScoredCandidate) -> float:
        score = (
            self.weight_strength * candidate.strength
            * self.weight_confidence * candidate.confidence
            * self.weight_target_activation * candidate.target_activation
            * self.weight_intent_bias * candidate.intent_bias
        )
        return float(score)

    def normalize(self, scores: List[float]) -> List[float]:
        if self.normalization == "none":
            return list(scores)
        if self.normalization == "softmax":
            return softmax(scores, self.softmax_temperature)
        if self.normalization == "rank":
            if not scores:
                return []
            sorted_scores = sorted(((score, idx) for idx, score in enumerate(scores)), reverse=True)
            ranks = [0] * len(scores)
            for rank, (_, idx) in enumerate(sorted_scores, start=1):
                ranks[idx] = rank
            weights = [1.0 / rank for rank in ranks]
            total = sum(weights)
            if total == 0.0:
                return [1.0 / len(weights) for _ in weights]
            return [value / total for value in weights]
        raise ValueError(f"Unsupported normalization: {self.normalization}")
