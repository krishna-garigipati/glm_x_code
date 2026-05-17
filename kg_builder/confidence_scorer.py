import logging
import numpy as np
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    def __init__(
        self,
        pattern_weight: float = 0.5,
        resolution_weight: float = 0.3,
        frequency_weight: float = 0.2,
        min_confidence: float = 0.25,
        adaptive: bool = True,
    ):
        self._pattern_weight = pattern_weight
        self._resolution_weight = resolution_weight
        self._frequency_weight = frequency_weight
        self._min_confidence = min_confidence
        self._initial_confidence = min_confidence
        self._pattern_scores = {1: 0.95, 2: 0.75, 3: 0.45}
        self._resolution_scores = {
            "exact": 1.0, "surface_form": 0.95,
            "embedding": 0.80, "graph": 0.70, "new": 0.60,
        }
        self._all_scores: List[float] = []
        self._adaptive = adaptive
        self._update_counter = 0

    def score(
        self,
        extraction_level: int = 1,
        resolution_method: str = "new",
        frequency: int = 1,
        relation_confidence: float = 0.8,
    ) -> float:
        p_conf = self._pattern_scores.get(extraction_level, 0.5)
        r_conf = self._resolution_scores.get(resolution_method, 0.5)
        f_conf = 1.0 - (1.0 / (frequency + 1))
        final = (
            self._pattern_weight * p_conf
            + self._resolution_weight * r_conf
            + self._frequency_weight * f_conf
        ) * relation_confidence
        final = round(float(final), 4)
        if self._adaptive:
            self._all_scores.append(final)
            self._update_counter += 1
            if self._update_counter % 50 == 0:
                self._adapt_threshold()
        return final

    def _adapt_threshold(self):
        if len(self._all_scores) < 20:
            return
        arr = np.array(self._all_scores)
        p25, p50, p75 = np.percentile(arr, [25, 50, 75])
        kept = sum(1 for s in self._all_scores if s >= self._initial_confidence)
        kept_frac = kept / len(self._all_scores)
        if kept_frac < 0.6:
            new_threshold = float(p25)
        elif kept_frac > 0.85:
            new_threshold = float(p50)
        else:
            new_threshold = self._initial_confidence
        new_threshold = max(0.15, min(0.40, new_threshold))
        logger.debug(
            "Adaptive confidence: kept=%.0f%% p25=%.3f p50=%.3f threshold=%.3f",
            kept_frac * 100, p25, p50, new_threshold
        )

    def is_acceptable(self, confidence: float) -> bool:
        return confidence >= self._min_confidence

    def set_min_confidence(self, threshold: float):
        self._min_confidence = threshold

    def get_threshold(self) -> float:
        return self._min_confidence

    def get_score_stats(self) -> Dict[str, float]:
        if not self._all_scores:
            return {"count": 0, "mean": 0.0}
        arr = np.array(self._all_scores)
        return {
            "count": len(self._all_scores),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
