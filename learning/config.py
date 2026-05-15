import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class HebbianConfig:
    alpha: float = 0.05
    beta: float = 0.02
    eligibility_gamma: float = 0.9
    user_feedback_range: Tuple[float, float] = (-1.0, 1.0)
    neutral_threshold: float = 0.1


@dataclass
class EligibilityConfig:
    min_eligibility_threshold: float = 0.0
    trace_normalization: bool = False
    walk_length_penalty: float = 0.0
    temporal_discount_strength: float = 0.5


@dataclass
class GlobalDecayConfig:
    enabled: bool = True
    interval_queries: int = 1000
    delta_base: float = 0.001
    frequency_protection: bool = True
    min_strength: float = 0.01


@dataclass
class CompressionConfig:
    enabled: bool = True
    interval_queries: int = 5000
    co_activation_threshold: int = 5
    pattern_node_type: str = "Pattern"
    pattern_edge_strength: float = 0.9
    pattern_edge_confidence: float = 0.9
    sequence_length: int = 3
    diverse_contexts_required: int = 5
    linguistic_enabled: bool = True
    linguistic_pattern_node_type: str = "LinguisticToken"


@dataclass
class AuditConfig:
    enabled: bool = True
    interval_queries: int = 10000
    synthetic_queries_per_audit: int = 100
    contradiction_method: str = "bfs"
    bfs_depth: int = 3
    contradiction_threshold: float = 0.3
    low_confidence_threshold: float = 0.4
    uncertain_intent_ids: List[int] = field(default_factory=lambda: [10, 11])
    external_fetch_enabled: bool = False
    external_fetch_timeout_ms: int = 5000
    external_fetch_confidence_increase: float = 0.15
    ask_user_on_uncertain: bool = True


@dataclass
class RewardWeightsConfig:
    external: float = 0.4
    human_feedback: float = 0.3
    internal: float = 0.3


@dataclass
class InternalRewardConfig:
    goal_alignment_weight: float = 0.4
    value_alignment_weight: float = 0.3
    emotional_consistency_weight: float = 0.3
    goal_hidden_dims: List[int] = field(default_factory=lambda: [32])
    value_hidden_dims: List[int] = field(default_factory=lambda: [64])
    emotion_hidden_dims: List[int] = field(default_factory=lambda: [64])
    activation: str = "relu"
    output: str = "sigmoid"


@dataclass
class HubDampingConfig:
    hub_degree_threshold: int = 999999
    damping_factor: float = 0.0
    cluster_budget_enabled: bool = False
    max_cluster_activation: float = 1.0


@dataclass
class DriftAdaptationConfig:
    detection_window: int = 100
    adaptation_rate: float = 0.0
    embedding_drift_threshold: float = 1.0
    strength_damping_factor: float = 0.0


@dataclass
class AntiCollapseConfig:
    embedding_refresh_rate_queries: int = 9999999
    activation_reset_threshold: float = 1.0
    strength_saturation_penalty: float = 0.0
    saturation_window: int = 100


@dataclass
class ESControllerConfig:
    evaluation_window: int = 8
    learning_rate: float = 0.02
    sigma_decay_beta: float = 0.1
    reward_normalization: str = "z_score"
    reward_ema_alpha: float = 0.1


@dataclass
class ForgettingMitigationConfig:
    replay_buffer_size: int = 10000
    replay_interval_queries: int = 100
    replay_batch_size: int = 32
    ewc_enabled: bool = False
    ewc_lambda: float = 5000
    si_enabled: bool = False
    si_c: float = 1.0


@dataclass
class StatePersistenceConfig:
    save_interval_queries: int = 5000
    save_path: str = "./learning_state/"
    files: List[str] = field(default_factory=lambda: [
        "reward_history.pkl",
        "es_mu_sigma.pkl",
        "pattern_nodes.pkl",
        "replay_buffer.pkl",
    ])


@dataclass
class HealthLogConfig:
    log_interval_queries: int = 999999
    metrics_window: int = 1000


@dataclass
class LearningConfig:
    hebbian: HebbianConfig = field(default_factory=HebbianConfig)
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    global_decay: GlobalDecayConfig = field(default_factory=GlobalDecayConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    internal_reward: InternalRewardConfig = field(default_factory=InternalRewardConfig)
    hub_damping: HubDampingConfig = field(default_factory=HubDampingConfig)
    drift: DriftAdaptationConfig = field(default_factory=DriftAdaptationConfig)
    anti_collapse: AntiCollapseConfig = field(default_factory=AntiCollapseConfig)
    es_controller: ESControllerConfig = field(default_factory=ESControllerConfig)
    forgetting: ForgettingMitigationConfig = field(default_factory=ForgettingMitigationConfig)
    state: StatePersistenceConfig = field(default_factory=StatePersistenceConfig)
    health_log: HealthLogConfig = field(default_factory=HealthLogConfig)

    @classmethod
    def from_yaml(cls, path: str = None) -> "LearningConfig":
        if path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, "configs", "config_learning.yaml")
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except (FileNotFoundError, ImportError):
            return cls()

        cfg = cls()

        h = data.get("hebbian", {})
        cfg.hebbian.alpha = h.get("alpha", cfg.hebbian.alpha)
        cfg.hebbian.beta = h.get("beta", cfg.hebbian.beta)
        cfg.hebbian.eligibility_gamma = h.get("eligibility_gamma", cfg.hebbian.eligibility_gamma)
        frange = h.get("user_feedback_range", cfg.hebbian.user_feedback_range)
        cfg.hebbian.user_feedback_range = tuple(frange) if isinstance(frange, list) else frange
        cfg.hebbian.neutral_threshold = h.get("neutral_threshold", cfg.hebbian.neutral_threshold)

        gd = data.get("global_decay", {})
        cfg.global_decay.enabled = gd.get("enabled", cfg.global_decay.enabled)
        cfg.global_decay.interval_queries = gd.get("interval_queries", cfg.global_decay.interval_queries)
        cfg.global_decay.delta_base = gd.get("delta_base", cfg.global_decay.delta_base)
        cfg.global_decay.frequency_protection = gd.get("frequency_protection", cfg.global_decay.frequency_protection)
        cfg.global_decay.min_strength = gd.get("min_strength", cfg.global_decay.min_strength)

        cp = data.get("compression", {})
        cfg.compression.enabled = cp.get("enabled", cfg.compression.enabled)
        cfg.compression.interval_queries = cp.get("interval_queries", cfg.compression.interval_queries)
        cfg.compression.co_activation_threshold = cp.get("co_activation_threshold", cfg.compression.co_activation_threshold)
        cfg.compression.pattern_node_type = cp.get("pattern_node_type", cfg.compression.pattern_node_type)
        cfg.compression.pattern_edge_strength = cp.get("pattern_edge_strength", cfg.compression.pattern_edge_strength)
        cfg.compression.pattern_edge_confidence = cp.get("pattern_edge_confidence", cfg.compression.pattern_edge_confidence)
        cfg.compression.sequence_length = cp.get("sequence_length", cfg.compression.sequence_length)
        cfg.compression.diverse_contexts_required = cp.get("diverse_contexts_required", cfg.compression.diverse_contexts_required)
        cfg.compression.linguistic_enabled = cp.get("linguistic_enabled", cfg.compression.linguistic_enabled)
        cfg.compression.linguistic_pattern_node_type = cp.get("linguistic_pattern_node_type", cfg.compression.linguistic_pattern_node_type)

        au = data.get("audit", {})
        cfg.audit.enabled = au.get("enabled", cfg.audit.enabled)
        cfg.audit.interval_queries = au.get("interval_queries", cfg.audit.interval_queries)
        cfg.audit.synthetic_queries_per_audit = au.get("synthetic_queries_per_audit", cfg.audit.synthetic_queries_per_audit)
        cfg.audit.contradiction_method = au.get("contradiction_method", cfg.audit.contradiction_method)
        cfg.audit.bfs_depth = au.get("bfs_depth", cfg.audit.bfs_depth)
        cfg.audit.contradiction_threshold = au.get("contradiction_threshold", cfg.audit.contradiction_threshold)
        cfg.audit.low_confidence_threshold = au.get("low_confidence_threshold", cfg.audit.low_confidence_threshold)
        cfg.audit.uncertain_intent_ids = au.get("uncertain_intent_ids", cfg.audit.uncertain_intent_ids)
        cfg.audit.external_fetch_enabled = au.get("external_fetch_enabled", cfg.audit.external_fetch_enabled)
        cfg.audit.external_fetch_timeout_ms = au.get("external_fetch_timeout_ms", cfg.audit.external_fetch_timeout_ms)
        cfg.audit.external_fetch_confidence_increase = au.get("external_fetch_confidence_increase", cfg.audit.external_fetch_confidence_increase)
        cfg.audit.ask_user_on_uncertain = au.get("ask_user_on_uncertain", cfg.audit.ask_user_on_uncertain)

        rw = data.get("reward_weights", {})
        cfg.reward_weights.external = rw.get("external", cfg.reward_weights.external)
        cfg.reward_weights.human_feedback = rw.get("human_feedback", cfg.reward_weights.human_feedback)
        cfg.reward_weights.internal = rw.get("internal", cfg.reward_weights.internal)

        ir = data.get("internal_reward", {})
        cfg.internal_reward.goal_alignment_weight = ir.get("goal_alignment_weight", cfg.internal_reward.goal_alignment_weight)
        cfg.internal_reward.value_alignment_weight = ir.get("value_alignment_weight", cfg.internal_reward.value_alignment_weight)
        cfg.internal_reward.emotional_consistency_weight = ir.get("emotional_consistency_weight", cfg.internal_reward.emotional_consistency_weight)
        gn = ir.get("goal_nn", {})
        cfg.internal_reward.goal_hidden_dims = gn.get("hidden_dims", cfg.internal_reward.goal_hidden_dims)
        vn = ir.get("value_nn", {})
        cfg.internal_reward.value_hidden_dims = vn.get("hidden_dims", cfg.internal_reward.value_hidden_dims)
        en = ir.get("emotion_nn", {})
        cfg.internal_reward.emotion_hidden_dims = en.get("hidden_dims", cfg.internal_reward.emotion_hidden_dims)

        es = data.get("es", {})
        cfg.es_controller.evaluation_window = es.get("evaluation_window", cfg.es_controller.evaluation_window)
        cfg.es_controller.learning_rate = es.get("learning_rate", cfg.es_controller.learning_rate)
        cfg.es_controller.sigma_decay_beta = es.get("sigma_decay_beta", cfg.es_controller.sigma_decay_beta)
        cfg.es_controller.reward_normalization = es.get("reward_normalization", cfg.es_controller.reward_normalization)
        cfg.es_controller.reward_ema_alpha = es.get("reward_ema_alpha", cfg.es_controller.reward_ema_alpha)

        fm = data.get("forgetting_mitigation", {})
        cfg.forgetting.replay_buffer_size = fm.get("replay_buffer_size", cfg.forgetting.replay_buffer_size)
        cfg.forgetting.replay_interval_queries = fm.get("replay_interval_queries", cfg.forgetting.replay_interval_queries)
        cfg.forgetting.replay_batch_size = fm.get("replay_batch_size", cfg.forgetting.replay_batch_size)
        cfg.forgetting.ewc_enabled = fm.get("ewc_enabled", cfg.forgetting.ewc_enabled)
        cfg.forgetting.ewc_lambda = fm.get("ewc_lambda", cfg.forgetting.ewc_lambda)
        cfg.forgetting.si_enabled = fm.get("si_enabled", cfg.forgetting.si_enabled)
        cfg.forgetting.si_c = fm.get("si_c", cfg.forgetting.si_c)

        st = data.get("state", {})
        cfg.state.save_interval_queries = st.get("save_interval_queries", cfg.state.save_interval_queries)
        cfg.state.save_path = st.get("save_path", cfg.state.save_path)
        cfg.state.files = st.get("files", cfg.state.files)

        return cfg
