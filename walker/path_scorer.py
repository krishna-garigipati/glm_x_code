"""Edge scoring and normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .exceptions import ValidationError
from .utils import softmax


@dataclass(frozen=True)
class ScoredCandidate:
    node_id: int
    edge_type: str
    strength: float
    confidence: float
    target_activation: float
    relation_bias: float
    raw_score: float = 0.0


class PathScorer:
    """Walker edge scorer.

    Contract v3.3.2 section 9 (scoring) fixes the formula:

        score = strength * confidence * target_activation * relation_bias(expected, actual)

    with ``dominant_signal: relation_bias`` and
    ``cosine_similarity: REMOVED from the main scoring formula``.

    Cosine similarity is therefore NOT a term here. The per-candidate
    ``target_similarity`` computation and its weight are removed from the scoring
    path; the query embedding is still used for seed selection and resonance, per
    section 6 ``allowed_usage``.
    """

    def __init__(
        self,
        normalization: str = "softmax",
        softmax_temperature: float = 0.1,
    ) -> None:
        # The contract formula has no weight terms, so none are configurable.
        # An earlier revision exposed weight_* keys; they were removed because
        # section 9 defines the score exactly and section 17 forbids re-adding
        # tunable signal that could overpower relation_bias.
        self.normalization = normalization
        self.softmax_temperature = softmax_temperature

    def score(self, candidate: ScoredCandidate) -> float:
        score = (
            candidate.strength
            * candidate.confidence
            * candidate.target_activation
            * candidate.relation_bias
        )
        return float(score)

    def apply_temperature(self, scores: List[float], temperature: float) -> List[float]:
        """Scale raw candidate scores by the decision temperature.

        The arithmetic lives here rather than in the walker so the walker holds
        no scoring logic of its own.
        """
        if not scores:
            return []
        if temperature <= 0.0:
            raise ValidationError("Temperature must be positive")
        return [score / temperature for score in scores]

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
