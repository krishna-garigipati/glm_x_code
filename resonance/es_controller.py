from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import threading
import numpy as np

from .config import CoreConfig, ESControllerConfig, ThetaIndices
from .validation import validate_theta

logger = logging.getLogger(__name__)


@dataclass
class _RewardSample:
    reward: float
    theta: np.ndarray


class EvolutionaryController:
    def __init__(
        self,
        core_config: CoreConfig,
        es_config: ESControllerConfig,
        initial_theta: np.ndarray,
        theta_indices: ThetaIndices,
        log_theta_history: bool = True,
        history_buffer_size: int = 1000,
        _seed: Optional[int] = None,
    ):
        validate_theta(initial_theta, es_config.theta_dim)

        self._core = core_config
        self._config = es_config
        self._theta_indices = theta_indices
        self._mu = initial_theta.astype(np.float32).copy()
        self._initial_mu = initial_theta.astype(np.float32).copy()
        self._sigma = max(float(es_config.sigma_initial), 1e-12)
        self._lock = threading.Lock()
        self._samples: List[_RewardSample] = []
        self._theta_history: List[np.ndarray] = []
        self._log_theta_history = log_theta_history
        self._history_buffer_size = max(1, history_buffer_size)
        self._proposal_queue: List[np.ndarray] = []
        self._rng = np.random.RandomState(_seed)

        if self._log_theta_history:
            self._theta_history.append(self._mu.copy())
            logger.info("Theta history logging enabled (buffer=%d)", self._history_buffer_size)

    def get_theta(self) -> np.ndarray:
        with self._lock:
            return self._mu.copy()

    def set_theta(self, theta: np.ndarray) -> None:
        with self._lock:
            validate_theta(theta, self._mu.shape[0])
            self._mu = theta.astype(np.float32).copy()
            self._record_theta_locked()

    def propose_theta_mutation(self) -> np.ndarray:
        with self._lock:
            if not self._proposal_queue:
                self._proposal_queue = self._generate_population()
            return self._proposal_queue.pop(0)

    def _generate_population(self) -> List[np.ndarray]:
        population: List[np.ndarray] = []
        pop_size = max(1, self._config.population_size)
        for _ in range(pop_size):
            noise = self._rng.normal(0.0, self._sigma, size=self._mu.shape).astype(np.float32)
            proposal = self._mu + noise
            population.append(self._apply_bounds(proposal))
        return population

    def update_es_with_reward(self, reward: float, theta_used: np.ndarray) -> None:
        if not np.isfinite(reward):
            logger.warning("Non-finite reward %s, clamping to 0", reward)
            reward = 0.0

        with self._lock:
            if theta_used.shape != self._mu.shape:
                raise ValueError(
                    f"theta_used shape {theta_used.shape} != expected {self._mu.shape}"
                )
            self._samples.append(
                _RewardSample(reward=reward, theta=np.nan_to_num(theta_used.astype(np.float32)))
            )
            if len(self._samples) < self._config.evaluation_window:
                return

            rewards = np.array([s.reward for s in self._samples], dtype=np.float32)
            thetas = np.stack([s.theta for s in self._samples])

            rewards = np.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)

            reward_mean = float(rewards.mean())
            reward_std = float(rewards.std()) if rewards.std() > 1e-8 else 1.0
            normalized = (rewards - reward_mean) / reward_std

            weighted_delta = (normalized[:, None] * (thetas - self._mu)).mean(axis=0)
            weighted_delta = np.nan_to_num(weighted_delta, nan=0.0, posinf=1.0, neginf=-1.0)

            self._mu = self._mu + self._config.learning_rate * weighted_delta
            self._mu = np.nan_to_num(self._mu, nan=0.0, posinf=1.0, neginf=-1.0)

            if self._config.anchor_enabled:
                self._mu = self._mu - self._config.anchor_lambda * (self._mu - self._initial_mu)

            self._mu = self._apply_bounds(self._mu)
            self._mu = np.nan_to_num(self._mu, nan=0.0, posinf=1.0, neginf=-1.0)
            self._sigma = max(1e-12, self._sigma * (1.0 - self._config.sigma_decay_beta))

            self._samples.clear()
            self._proposal_queue.clear()
            self._record_theta_locked()

            logger.debug("ES update: mu_updated, sigma=%.6f, samples=%d", self._sigma, len(self._samples))

    def _record_theta_locked(self) -> None:
        if not self._log_theta_history:
            return
        self._theta_history.append(self._mu.copy())
        if len(self._theta_history) > self._history_buffer_size:
            self._theta_history.pop(0)

    def get_theta_history(self) -> List[np.ndarray]:
        with self._lock:
            return [h.copy() for h in self._theta_history]

    def clear_theta_history(self) -> None:
        with self._lock:
            self._theta_history.clear()

    def _apply_bounds(self, theta: np.ndarray) -> np.ndarray:
        bounds = self._core.es_bounds
        idx = self._theta_indices
        theta = theta.copy()
        theta[idx.propagation_threshold] = np.clip(
            theta[idx.propagation_threshold], *bounds.propagation_threshold
        )
        theta[idx.edge_threshold] = np.clip(
            theta[idx.edge_threshold], *bounds.edge_threshold
        )
        theta[idx.decay_lambda] = np.clip(
            theta[idx.decay_lambda], *bounds.decay_lambda
        )
        theta[idx.top_k] = np.clip(
            theta[idx.top_k], *bounds.top_k
        )
        theta[idx.relation_bias_start:idx.relation_bias_end] = np.clip(
            theta[idx.relation_bias_start:idx.relation_bias_end], *bounds.relation_bias
        )
        return theta.astype(np.float32)
