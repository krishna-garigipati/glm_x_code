import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

import numpy as np

from learning.types import (
    Answer, Subgraph, WalkResult, Plan, Edge, Node,
    GraphStoreInterface, ResonanceEngineInterface,
    G2PPlannerInterface, GraphWalkerInterface, MicroDecoderInterface,
    InternalRewardComponents,
)
from learning.config import LearningConfig
from learning.hebbian import HebbianUpdater
from learning.eligibility import EligibilityTracer
from learning.compression import PatternCompressor
from learning.audit import SelfAuditor
from learning.reward import InternalRewardModel
from learning.persistence import LearningStatePersistence

logger = logging.getLogger(__name__)


class LearningEngine:
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self._lock = threading.Lock()
        self._total_queries: int = 0

        self.hebbian_updater = HebbianUpdater(self.config)
        self.eligibility_tracer = EligibilityTracer(self.config)
        self.pattern_compressor = PatternCompressor(self.config)
        self.self_auditor = SelfAuditor(self.config)
        self.internal_reward_model = InternalRewardModel(self.config)
        self.persistence = LearningStatePersistence(self.config)

        self._reward_history: List[float] = []
        self._replay_buffer: deque = deque(maxlen=self.config.forgetting.replay_buffer_size)
        self._ema_mean: float = 0.0
        self._ema_var: float = 1.0
        self._ema_initialized: bool = False

        self._es_mu_sigma: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self._load_initial_state()

    def process_feedback(
        self,
        answer: Answer,
        user_rating: float,
        walk: WalkResult,
        subgraph: Subgraph,
        external_reward: float = 0.0,
        graph: Optional[GraphStoreInterface] = None,
        resonance_engine: Optional[ResonanceEngineInterface] = None,
        g2p: Optional[G2PPlannerInterface] = None,
        walker: Optional[GraphWalkerInterface] = None,
        decoder: Optional[MicroDecoderInterface] = None,
        plan_adherence: float = 1.0,
        model_quality: float = 1.0,
    ) -> None:
        if graph is None:
            logger.warning("process_feedback: graph is None, skipping")
            return

        logger.debug(
            "process_feedback: answer.confidence=%.4f, user_rating=%.4f, walk.steps=%d",
            answer.confidence, user_rating, walk.steps_taken,
        )

        user_rating = max(-1.0, min(1.0, user_rating))

        walk_confidence = walk.walk_confidence if hasattr(walk, 'walk_confidence') else 0.5

        eligibility = self.get_eligibility_trace(walk, subgraph)

        effective_reward = self._compute_effective_reward(
            external_reward, user_rating, walk, subgraph, plan_adherence
        )

        self.hebbian_updater.apply_hebbian_updates(
            graph, eligibility, effective_reward, subgraph,
            plan_adherence=plan_adherence,
            model_quality=model_quality,
            walk_confidence=walk_confidence,
        )

        if resonance_engine is not None:
            theta_used = resonance_engine.get_theta()
            self.update_es_controller(effective_reward, theta_used, resonance_engine)

        self._store_in_replay_buffer(answer, user_rating, walk, subgraph, external_reward, effective_reward, plan_adherence)

        with self._lock:
            self._total_queries += 1

        self._run_periodic_tasks(
            graph, resonance_engine, g2p, walker, decoder
        )

    def get_eligibility_trace(
        self,
        walk: WalkResult,
        subgraph: Subgraph,
    ) -> Dict[Tuple[int, int, str], float]:
        return self.eligibility_tracer.compute_trace(walk, subgraph)

    def compute_internal_reward(
        self,
        action_vec: np.ndarray,
        outcome_vec: np.ndarray,
        emotional_context_vec: np.ndarray,
        current_goals_vec: np.ndarray,
        context_embedding: np.ndarray,
        vdna_vector: np.ndarray,
    ) -> Tuple[float, Dict]:
        reward, components = self.internal_reward_model.compute_internal_reward(
            action_vec, outcome_vec, emotional_context_vec,
            current_goals_vec, context_embedding, vdna_vector,
        )
        return reward, {
            "goal_alignment": components.goal_alignment,
            "value_alignment": components.value_alignment,
            "emotional_consistency": components.emotional_consistency,
            "weighted_sum": components.weighted_sum,
        }

    def compute_total_reward(
        self,
        external_reward: float,
        human_feedback: float,
        internal_reward: float,
    ) -> float:
        return InternalRewardModel.compute_total_reward(
            external_reward=external_reward,
            human_feedback=human_feedback,
            internal_reward=internal_reward,
            w_ext=self.config.reward_weights.external,
            w_human=self.config.reward_weights.human_feedback,
            w_int=self.config.reward_weights.internal,
        )

    def update_es_controller(
        self,
        reward: float,
        theta_used: np.ndarray,
        resonance_engine: ResonanceEngineInterface,
    ) -> None:
        normalized_reward = self._normalize_reward(reward)
        resonance_engine.update_es_with_reward(normalized_reward, theta_used)
        logger.debug(
            "ES update: raw_reward=%.4f, normalized=%.4f, theta_norm=%.4f",
            reward, normalized_reward, np.linalg.norm(theta_used),
        )

    def compress_pattern_nodes(
        self,
        graph: GraphStoreInterface,
        min_co_activation: Optional[int] = None,
    ) -> List[int]:
        return self.pattern_compressor.compress_pattern_nodes(graph, min_co_activation)

    def run_self_audit(
        self,
        graph: GraphStoreInterface,
        resonance_engine: ResonanceEngineInterface,
        g2p: G2PPlannerInterface,
        walker: GraphWalkerInterface,
        decoder: MicroDecoderInterface,
    ) -> Dict:
        return self.self_auditor.run_self_audit(
            graph, resonance_engine, g2p, walker, decoder
        )

    def detect_contradictions(self, subgraph: Subgraph) -> List[Tuple[Edge, Edge]]:
        return self.self_auditor.detect_contradictions(subgraph)

    def replay_experiences(
        self,
        graph: GraphStoreInterface,
        num_experiences: Optional[int] = None,
    ) -> None:
        if num_experiences is None:
            num_experiences = self.config.forgetting.replay_batch_size

        buffer = list(self._replay_buffer)
        if not buffer:
            logger.debug("Replay buffer empty, skipping replay")
            return

        batch_size = min(num_experiences, len(buffer))
        indices = np.random.choice(len(buffer), size=batch_size, replace=False).tolist()

        for idx in indices:
            experience = buffer[idx]
            exp_walk = experience.get("walk")
            exp_subgraph = experience.get("subgraph")
            exp_rating = experience.get("user_rating", 0.0)
            if exp_walk is None or exp_subgraph is None:
                continue

            eligibility = self.eligibility_tracer.compute_trace(exp_walk, exp_subgraph)
            reward = experience.get("effective_reward")
            if reward is None:
                reward = self._compute_effective_reward(
                    experience.get("external_reward", 0.0),
                    exp_rating,
                    exp_walk,
                    exp_subgraph,
                    experience.get("plan_adherence", 1.0),
                )
            self.hebbian_updater.apply_hebbian_updates(
                graph, eligibility, reward, exp_subgraph,
                plan_adherence=experience.get("plan_adherence", 1.0),
                model_quality=0.5,
                walk_confidence=0.5,
            )

        logger.debug("Replayed %d experiences from buffer", len(indices))

    def save_state(self) -> Dict[str, bool]:
        state_dict = {
            "reward_history.pkl": list(self._reward_history[-1000:]),
            "es_mu_sigma.pkl": self._es_mu_sigma,
            "replay_buffer.pkl": list(self._replay_buffer),
        }
        pattern_nodes = getattr(self.pattern_compressor, "_pattern_node_id_counter", 0)
        state_dict["pattern_nodes.pkl"] = {"counter": pattern_nodes}
        return self.persistence.save_all(state_dict)

    def load_state(self) -> bool:
        state = self.persistence.load_all()
        if "reward_history.pkl" in state and isinstance(state["reward_history.pkl"], list):
            self._reward_history = state["reward_history.pkl"]
        if "es_mu_sigma.pkl" in state:
            self._es_mu_sigma = state["es_mu_sigma.pkl"]
        if "replay_buffer.pkl" in state and isinstance(state["replay_buffer.pkl"], list):
            self._replay_buffer = deque(state["replay_buffer.pkl"], maxlen=self.config.forgetting.replay_buffer_size)
        logger.info("Loaded %d state files", len(state))
        return len(state) > 0

    @property
    def total_queries(self) -> int:
        return self._total_queries

    @property
    def reward_history(self) -> List[float]:
        return list(self._reward_history)

    @property
    def replay_buffer_size(self) -> int:
        return len(self._replay_buffer)

    def _compute_effective_reward(
        self,
        external_reward: float,
        user_rating: float,
        walk: WalkResult,
        subgraph: Subgraph,
        plan_adherence: float = 1.0,
    ) -> float:
        human_feedback = user_rating
        external = external_reward * self.config.reward_weights.external
        human = human_feedback * self.config.reward_weights.human_feedback
        internal_val = self._estimate_internal_reward(walk, subgraph)
        internal = internal_val * self.config.reward_weights.internal
        total = external + human + internal
        with self._lock:
            self._reward_history.append(total)
            if len(self._reward_history) > 10000:
                self._reward_history = self._reward_history[-5000:]
        return total

    def _estimate_internal_reward(self, walk: WalkResult, subgraph: Subgraph) -> float:
        if not walk.path_activations:
            return 0.0
        avg_activation = float(np.mean(walk.path_activations))
        avg_confidence = float(np.mean(walk.path_confidences)) if walk.path_confidences else 0.0
        energy = subgraph.activation_energy
        internal_score = 0.4 * avg_activation + 0.3 * avg_confidence + 0.3 * min(energy / 10.0, 1.0)
        return max(0.0, min(1.0, internal_score))

    def _normalize_reward(self, reward: float) -> float:
        norm_type = self.config.es_controller.reward_normalization
        alpha = self.config.es_controller.reward_ema_alpha

        if norm_type == "none":
            return reward
        elif norm_type == "z_score":
            if not self._ema_initialized:
                self._ema_mean = reward
                self._ema_var = 1.0
                self._ema_initialized = True
            else:
                self._ema_mean = (1 - alpha) * self._ema_mean + alpha * reward
                self._ema_var = (1 - alpha) * self._ema_var + alpha * (reward - self._ema_mean) ** 2
            std = max(np.sqrt(self._ema_var), 1e-8)
            return (reward - self._ema_mean) / std
        elif norm_type == "min_max":
            if self._reward_history:
                r_min = min(self._reward_history[-100:])
                r_max = max(self._reward_history[-100:])
                if r_max > r_min:
                    return (reward - r_min) / (r_max - r_min) * 2.0 - 1.0
            return reward
        return reward

    def _store_in_replay_buffer(
        self,
        answer: Answer,
        user_rating: float,
        walk: WalkResult,
        subgraph: Subgraph,
        external_reward: float,
        effective_reward: float = 0.0,
        plan_adherence: float = 1.0,
    ) -> None:
        experience = {
            "answer": answer,
            "user_rating": user_rating,
            "walk": walk,
            "subgraph": subgraph,
            "external_reward": external_reward,
            "effective_reward": effective_reward,
            "plan_adherence": plan_adherence,
            "timestamp": time.time(),
        }
        with self._lock:
            self._replay_buffer.append(experience)

    def _run_periodic_tasks(
        self,
        graph: Optional[GraphStoreInterface],
        resonance_engine: Optional[ResonanceEngineInterface],
        g2p: Optional[G2PPlannerInterface],
        walker: Optional[GraphWalkerInterface],
        decoder: Optional[MicroDecoderInterface],
    ) -> None:
        if self.hebbian_updater.increment_query_counter():
            if graph is not None:
                try:
                    dummy_emb = np.zeros(384, dtype=np.float32)
                    decay_subgraph = graph.get_subgraph_by_embedding_similarity(dummy_emb, top_k=100)
                    self.hebbian_updater.apply_global_decay(graph, decay_subgraph)
                except Exception as e:
                    logger.warning("Global decay failed: %s", e)
            self.hebbian_updater.reset_query_counter()

        if self.pattern_compressor.increment_query_counter():
            if graph is not None:
                try:
                    self.pattern_compressor.compress_pattern_nodes(graph)
                except Exception as e:
                    logger.warning("Pattern compression failed: %s", e)
            self.pattern_compressor.reset_query_counter()

        if self.config.forgetting.replay_interval_queries > 0:
            if self._total_queries % self.config.forgetting.replay_interval_queries == 0:
                if graph is not None and self._replay_buffer:
                    try:
                        self.replay_experiences(graph)
                    except Exception as e:
                        logger.warning("Experience replay failed: %s", e)

        if self.self_auditor.increment_query_counter():
            if all(x is not None for x in [graph, resonance_engine, g2p, walker, decoder]):
                try:
                    self.self_auditor.run_self_audit(
                        graph, resonance_engine, g2p, walker, decoder
                    )
                except Exception as e:
                    logger.warning("Self-audit failed: %s", e)
            self.self_auditor.reset_query_counter()

        if self.persistence.increment_query_counter():
            try:
                self.save_state()
            except Exception as e:
                logger.warning("State save failed: %s", e)
            self.persistence.reset_query_counter()

    def _load_initial_state(self) -> None:
        try:
            self.load_state()
        except Exception as e:
            logger.info("No prior state to load: %s", e)
