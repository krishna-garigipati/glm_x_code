"""Utility helpers for scoring and math."""

from __future__ import annotations

import math
from typing import Iterable, List


def softmax(scores: List[float], temperature: float) -> List[float]:
    if not scores:
        return []
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    max_score = max(scores)
    scaled = [(s - max_score) / temperature for s in scores]
    exps = [math.exp(s) for s in scaled]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in exps]


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 1.0
    product = 1.0
    for value in values:
        if value <= 0.0:
            return 0.0
        product *= value
    return product ** (1.0 / len(values))
