from __future__ import annotations

import math
import time

import numpy as np
import pytest

from ...temporal import compute_temporal_factor


class TestComputeTemporalFactor:
    def test_recent_and_frequent_returns_high(self):
        result = compute_temporal_factor(
            last_used=time.time(), frequency=100, gamma=0.5, frequency_threshold=20
        )
        assert 0.0 <= result <= 1.0
        assert result > 0.8

    def test_old_and_rare_returns_low(self):
        result = compute_temporal_factor(
            last_used=time.time() - 86400 * 365, frequency=1, gamma=0.5, frequency_threshold=20
        )
        assert 0.0 <= result <= 1.0
        assert result < 0.3

    def test_frequency_at_threshold(self):
        result = compute_temporal_factor(
            last_used=time.time(), frequency=20, gamma=0.5, frequency_threshold=20
        )
        assert 0.0 <= result <= 1.0

    def test_frequency_above_threshold_capped(self):
        result_high = compute_temporal_factor(
            last_used=time.time(), frequency=1000, gamma=0.5, frequency_threshold=20
        )
        result_at = compute_temporal_factor(
            last_used=time.time(), frequency=20, gamma=0.5, frequency_threshold=20
        )
        assert abs(result_high - result_at) < 0.01

    def test_zero_delta_t(self):
        now = time.time()
        result = compute_temporal_factor(
            last_used=now, frequency=20, gamma=0.5, frequency_threshold=20
        )
        expected_recency = 1.0 / (1.0 + 0.5 * math.log(1.0 + 0.0))
        assert abs(result - expected_recency) < 0.01

    def test_large_delta_t(self):
        result = compute_temporal_factor(
            last_used=0.0, frequency=20, gamma=0.5, frequency_threshold=20
        )
        assert result >= 0.0
        assert result < 0.2

    def test_gamma_zero_disables_recency(self):
        old = compute_temporal_factor(
            last_used=0.0, frequency=20, gamma=0.0, frequency_threshold=20
        )
        recent = compute_temporal_factor(
            last_used=time.time(), frequency=20, gamma=0.0, frequency_threshold=20
        )
        assert abs(old - recent) < 0.01

    def test_zero_frequency(self):
        result = compute_temporal_factor(
            last_used=time.time(), frequency=0, gamma=0.5, frequency_threshold=20
        )
        assert result == 0.0

    def test_negative_gamma_clamped(self):
        result = compute_temporal_factor(
            last_used=time.time(), frequency=20, gamma=-1.0, frequency_threshold=20
        )
        assert 0.0 <= result <= 1.0

    def test_negative_frequency_clamped(self):
        result = compute_temporal_factor(
            last_used=time.time(), frequency=-5, gamma=0.5, frequency_threshold=20
        )
        assert result == 0.0

    def test_frequency_threshold_below_one_clamped(self):
        result = compute_temporal_factor(
            last_used=time.time(), frequency=5, gamma=0.5, frequency_threshold=0
        )
        assert 0.0 <= result <= 1.0

    def test_result_always_in_range(self):
        for _ in range(100):
            lu = time.time() - np.random.uniform(0, 86400 * 365)
            freq = int(np.random.uniform(0, 100))
            gamma = np.random.uniform(0, 2.0)
            ft = int(np.random.uniform(1, 100))
            result = compute_temporal_factor(lu, freq, gamma, ft)
            assert 0.0 <= result <= 1.0, f"Out of range: {result} for lu={lu}, freq={freq}"

    def test_non_finite_delta_t_returns_zero(self):
        result = compute_temporal_factor(
            last_used=float("nan"), frequency=20, gamma=0.5, frequency_threshold=20
        )
        assert result == 0.0

    def test_future_last_used(self):
        result = compute_temporal_factor(
            last_used=time.time() + 3600, frequency=20, gamma=0.5, frequency_threshold=20
        )
        assert 0.0 <= result <= 1.0
