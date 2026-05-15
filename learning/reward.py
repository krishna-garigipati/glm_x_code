import logging
import threading
from typing import Tuple, Dict, List, Optional

import numpy as np

from learner.types import InternalRewardComponents
from learner.config import LearningConfig

logger = logging.getLogger(__name__)


class SimpleNeuralNetwork:
    def __init__(self, hidden_dims: List[int], activation: str = "relu", output: str = "sigmoid"):
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.activation = activation
        self.output = output
        self.hidden_dims = hidden_dims

    def build(self, input_dim: int) -> None:
        dims = [input_dim] + self.hidden_dims + [1]
        rng = np.random.RandomState(42)
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(rng.randn(dims[i], dims[i + 1]).astype(np.float32) * scale)
            self.biases.append(np.zeros(dims[i + 1], dtype=np.float32))

    def forward(self, x: np.ndarray) -> float:
        h = x.astype(np.float32)
        if h.ndim == 1:
            h = h.reshape(1, -1)
        for i in range(len(self.weights) - 1):
            h = h.dot(self.weights[i]) + self.biases[i]
            if self.activation == "relu":
                h = np.maximum(0, h)
            elif self.activation == "tanh":
                h = np.tanh(h)
        h = h.dot(self.weights[-1]) + self.biases[-1]
        if self.output == "sigmoid":
            h = 1.0 / (1.0 + np.exp(-np.clip(h, -15, 15)))
        elif self.output == "tanh":
            h = np.tanh(h)
        return float(h[0, 0])

    def get_params(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.weights, self.biases

    def set_params(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]


class InternalRewardModel:
    def __init__(self, config: LearningConfig):
        self.cfg = config.internal_reward
        self.goal_weight = config.internal_reward.goal_alignment_weight
        self.value_weight = config.internal_reward.value_alignment_weight
        self.emotion_weight = config.internal_reward.emotional_consistency_weight
        self._lock = threading.Lock()

        self.goal_nn = SimpleNeuralNetwork(
            hidden_dims=config.internal_reward.goal_hidden_dims,
            activation=config.internal_reward.activation,
            output=config.internal_reward.output,
        )
        self.value_nn = SimpleNeuralNetwork(
            hidden_dims=config.internal_reward.value_hidden_dims,
            activation=config.internal_reward.activation,
            output=config.internal_reward.output,
        )
        self.emotion_nn = SimpleNeuralNetwork(
            hidden_dims=config.internal_reward.emotion_hidden_dims,
            activation=config.internal_reward.activation,
            output=config.internal_reward.output,
        )

    def compute_internal_reward(
        self,
        action_vec: np.ndarray,
        outcome_vec: np.ndarray,
        emotional_context_vec: np.ndarray,
        current_goals_vec: np.ndarray,
        context_embedding: np.ndarray,
        vdna_vector: np.ndarray,
    ) -> Tuple[float, InternalRewardComponents]:
        with self._lock:
            goal_input = np.concatenate([
                action_vec.ravel(),
                current_goals_vec.ravel(),
                context_embedding.ravel(),
            ])
            value_input = np.concatenate([
                action_vec.ravel(),
                outcome_vec.ravel(),
                vdna_vector.ravel() if vdna_vector.size > 0 else np.zeros(1),
            ])
            emotion_input = np.concatenate([
                emotional_context_vec.ravel(),
                outcome_vec.ravel(),
                context_embedding.ravel(),
            ])

            if not hasattr(self.goal_nn, "weights") or len(self.goal_nn.weights) == 0:
                self.goal_nn.build(goal_input.shape[0])
            if not hasattr(self.value_nn, "weights") or len(self.value_nn.weights) == 0:
                self.value_nn.build(value_input.shape[0])
            if not hasattr(self.emotion_nn, "weights") or len(self.emotion_nn.weights) == 0:
                self.emotion_nn.build(emotion_input.shape[0])

            goal_score = self.goal_nn.forward(goal_input)
            value_score = self.value_nn.forward(value_input)
            emotion_score = self.emotion_nn.forward(emotion_input)

        weighted = (
            self.goal_weight * goal_score
            + self.value_weight * value_score
            + self.emotion_weight * emotion_score
        )

        components = InternalRewardComponents(
            goal_alignment=float(goal_score),
            value_alignment=float(value_score),
            emotional_consistency=float(emotion_score),
            weighted_sum=float(weighted),
        )

        logger.debug(
            "Internal reward: goal=%.4f value=%.4f emotion=%.4f weighted=%.4f",
            goal_score, value_score, emotion_score, weighted,
        )

        return float(weighted), components

    @staticmethod
    def compute_total_reward(
        external_reward: float,
        human_feedback: float,
        internal_reward: float,
        w_ext: float = 0.4,
        w_human: float = 0.3,
        w_int: float = 0.3,
    ) -> float:
        total = w_ext * external_reward + w_human * human_feedback + w_int * internal_reward
        return total

    def get_nn_params(self) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
        with self._lock:
            return {
                "goal_nn": self.goal_nn.get_params(),
                "value_nn": self.value_nn.get_params(),
                "emotion_nn": self.emotion_nn.get_params(),
            }

    def set_nn_params(self, params: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]) -> None:
        with self._lock:
            if "goal_nn" in params:
                self.goal_nn.set_params(*params["goal_nn"])
            if "value_nn" in params:
                self.value_nn.set_params(*params["value_nn"])
            if "emotion_nn" in params:
                self.emotion_nn.set_params(*params["emotion_nn"])
