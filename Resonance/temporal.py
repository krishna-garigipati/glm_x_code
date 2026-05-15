from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)


def compute_temporal_factor(
    last_used: float, frequency: int, gamma: float, frequency_threshold: int
) -> float:
    if gamma < 0.0:
        logger.warning("Negative gamma=%s, clamping to 0", gamma)
        gamma = 0.0
    if frequency_threshold < 1:
        logger.warning("frequency_threshold=%s, clamping to 1", frequency_threshold)
        frequency_threshold = 1
    if frequency < 0:
        logger.warning("Negative frequency=%s, clamping to 0", frequency)
        frequency = 0

    if not math.isfinite(last_used):
        logger.warning("Non-finite last_used=%s, returning 0.0", last_used)
        return 0.0

    now = time.time()
    delta_t = max(0.0, now - last_used)

    recency = 1.0 / (1.0 + gamma * math.log(1.0 + delta_t)) if gamma > 0 else 1.0
    frequency_scale = min(1.0, frequency / float(frequency_threshold))

    result = recency * frequency_scale

    if not math.isfinite(result):
        logger.warning("Non-finite temporal factor=%s, returning 0.0", result)
        return 0.0

    return float(max(0.0, min(1.0, result)))
