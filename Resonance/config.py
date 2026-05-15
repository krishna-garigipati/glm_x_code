from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlgorithmConfig:
    propagation_type: str
    normalization: str
    gate_type: str
    temporal_factor_enabled: bool


@dataclass(frozen=True)
class TierConfig:
    propagation_threshold: float
    edge_threshold: float
    decay_lambda: float
    top_k: int
    max_iterations: int
    relation_bias: Dict[str, float]
    energy_threshold_formula: str | None
    t_conf_coefficient: float | None
    multiplied_at_runtime: bool | None


@dataclass(frozen=True)
class AnalogyParameters:
    lsh_bands: int
    lsh_tables: int
    temp_edge_strength: float
    overlap_validation_required: bool
    jaccard_overlap_min: float
    mini_propagation_steps: int
    edge_confidence_min: float


@dataclass(frozen=True)
class Tier2Config:
    enabled: bool
    propagation_threshold: float
    edge_threshold: float
    decay_lambda: float
    top_k: int
    max_iterations: int
    analogies_enabled: bool
    analogy_parameters: AnalogyParameters


@dataclass(frozen=True)
class EnergyConfig:
    formula: str
    tier1_threshold: float
    tier2_min_improvement: float


@dataclass(frozen=True)
class TemporalConfig:
    gamma: float
    frequency_threshold: int


@dataclass(frozen=True)
class ESControllerConfig:
    enabled: bool
    theta_dim: int
    population_size: int
    mu_initial: Dict[str, float]
    sigma_initial: float
    learning_rate: float
    sigma_decay_beta: float
    evaluation_window: int
    anchor_enabled: bool
    anchor_lambda: float


@dataclass(frozen=True)
class ThetaIndices:
    propagation_threshold: int
    edge_threshold: int
    decay_lambda: int
    top_k: int
    relation_bias_start: int
    relation_bias_end: int
    reserved_start: int
    reserved_end: int
    relation_bias_mapping: Dict[str, str]


@dataclass(frozen=True)
class DiagnosticsConfig:
    log_activation_history: bool
    log_theta_history: bool
    history_buffer_size: int


@dataclass(frozen=True)
class CoreActivationConfig:
    min: float
    max: float
    default: float
    threshold_resonance: float


@dataclass(frozen=True)
class CoreResonanceConfig:
    propagation_threshold: float
    edge_threshold: float
    decay_lambda: float
    top_k: int
    budget_max: float
    convergence_epsilon: float
    tier1_energy_threshold: float
    tier2_max_nodes: int
    analogy_validation_overlap: float


@dataclass(frozen=True)
class ESBounds:
    propagation_threshold: Tuple[float, float]
    edge_threshold: Tuple[float, float]
    decay_lambda: Tuple[float, float]
    top_k: Tuple[int, int]
    relation_bias: Tuple[float, float]


@dataclass(frozen=True)
class CoreConfig:
    activation: CoreActivationConfig
    resonance: CoreResonanceConfig
    relations: List[str]
    es_bounds: ESBounds


@dataclass(frozen=True)
class ResonanceConfig:
    algorithm: AlgorithmConfig
    tier1: TierConfig
    tier2: Tier2Config
    energy: EnergyConfig
    temporal: TemporalConfig
    es_controller: ESControllerConfig
    theta_indices: ThetaIndices
    diagnostics: DiagnosticsConfig


@dataclass(frozen=True)
class LoadedConfigs:
    core: CoreConfig
    resonance: ResonanceConfig


_VALID_PROPAGATION_TYPES = frozenset({"wilson_cowan", "simple_diffusion", "threshold"})
_VALID_NORMALIZATIONS = frozenset({"budget_soft_cap", "l1_norm", "none"})
_VALID_GATE_TYPES = frozenset({"top_k", "threshold", "none"})
_RELATION_NAMES = frozenset({
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near", "linguistic_maps",
})


def _read_yaml(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file is empty or malformed: {path}")
    return data


def _validate_range(value: float, lo: float, hi: float, name: str) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


def _validate_int_range(value: int, lo: int, hi: int, name: str) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


def _validate_relation_bias(name: str, value: float) -> None:
    if name not in _RELATION_NAMES:
        raise ValueError(f"Unknown relation type: {name}")
    _validate_range(value, 0.0, 2.0, f"relation_bias.{name}")


def load_configs(config_dir: Path) -> LoadedConfigs:
    if isinstance(config_dir, str):
        config_dir = Path(config_dir)
    config_dir = Path(config_dir)

    core_path = config_dir / "config_core.yaml"
    resonance_path = config_dir / "config_resonance.yaml"

    core_data = _read_yaml(core_path)
    resonance_data = _read_yaml(resonance_path)

    # ---- Core config ----
    act_raw = core_data.get("activation", {})
    core_activation = CoreActivationConfig(
        min=_get_float(act_raw, "min", 0.01, lo=0.0, hi=1.0),
        max=_get_float(act_raw, "max", 1.0, lo=0.0, hi=1.0),
        default=_get_float(act_raw, "default", 0.01, lo=0.0, hi=1.0),
        threshold_resonance=_get_float(act_raw, "threshold_resonance", 0.2, lo=0.0, hi=1.0),
    )
    if core_activation.default < core_activation.min:
        raise ValueError("activation.default must be >= activation.min")
    if core_activation.default > core_activation.max:
        raise ValueError("activation.default must be <= activation.max")
    if core_activation.max <= core_activation.min:
        raise ValueError("activation.max must be > activation.min")

    res_raw = core_data.get("resonance", {})
    resonance_core = CoreResonanceConfig(
        propagation_threshold=_get_float(res_raw, "propagation_threshold", 0.008, lo=0.0, hi=1.0),
        edge_threshold=_get_float(res_raw, "edge_threshold", 0.02, lo=0.0, hi=1.0),
        decay_lambda=_get_float(res_raw, "decay_lambda", 0.1, lo=0.0, hi=1.0),
        top_k=_get_int(res_raw, "top_k", 64, lo=1, hi=100000),
        budget_max=_get_float(res_raw, "budget_max", 2.0, lo=0.0, hi=1e6),
        convergence_epsilon=_get_float(res_raw, "convergence_epsilon", 0.001, lo=1e-12, hi=1.0),
        tier1_energy_threshold=_get_float(res_raw, "tier1_energy_threshold", 0.4, lo=0.0, hi=1e6),
        tier2_max_nodes=_get_int(res_raw, "tier2_max_nodes", 1024, lo=1, hi=1000000),
        analogy_validation_overlap=_get_float(res_raw, "analogy_validation_overlap", 0.3, lo=0.0, hi=1.0),
    )

    relations_raw = core_data.get("relations", {})
    relations = []
    seen_keys = set()
    for k in sorted(relations_raw.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) else 0):
        val = relations_raw[k]
        if val in seen_keys:
            continue
        seen_keys.add(val)
        relations.append(val)
    for rel in relations:
        if rel not in _RELATION_NAMES:
            logger.warning("Unknown relation type in core config: %s", rel)

    es_raw = core_data.get("es", {})
    bounds_raw = es_raw.get("bounds", {})
    es_bounds = ESBounds(
        propagation_threshold=_get_float_tuple(bounds_raw, "propagation_threshold", (0.001, 0.05)),
        edge_threshold=_get_float_tuple(bounds_raw, "edge_threshold", (0.01, 0.1)),
        decay_lambda=_get_float_tuple(bounds_raw, "decay_lambda", (0.05, 0.5)),
        top_k=_get_int_tuple(bounds_raw, "top_k", (16, 1024)),
        relation_bias=_get_float_tuple(bounds_raw, "relation_bias", (0.0, 2.0)),
    )

    core_config = CoreConfig(
        activation=core_activation,
        resonance=resonance_core,
        relations=relations,
        es_bounds=es_bounds,
    )

    # ---- Resonance config ----
    algorithm_raw = resonance_data.get("algorithm", {})
    propagation_type = str(algorithm_raw.get("propagation_type", "wilson_cowan"))
    if propagation_type not in _VALID_PROPAGATION_TYPES:
        raise ValueError(f"Invalid propagation_type: {propagation_type}")
    normalization = str(algorithm_raw.get("normalization", "budget_soft_cap"))
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(f"Invalid normalization: {normalization}")
    gate_type = str(algorithm_raw.get("gate_type", "top_k"))
    if gate_type not in _VALID_GATE_TYPES:
        raise ValueError(f"Invalid gate_type: {gate_type}")

    algorithm = AlgorithmConfig(
        propagation_type=propagation_type,
        normalization=normalization,
        gate_type=gate_type,
        temporal_factor_enabled=bool(algorithm_raw.get("temporal_factor_enabled", True)),
    )

    tier1_raw = resonance_data.get("tier1", {})
    relation_bias_raw = tier1_raw.get("relation_bias", {})
    relation_bias = {}
    for rel, val in relation_bias_raw.items():
        v = float(val)
        _validate_relation_bias(rel, v)
        relation_bias[rel] = v

    tier1 = TierConfig(
        propagation_threshold=_get_float(tier1_raw, "propagation_threshold", 0.008, lo=0.0, hi=1.0),
        edge_threshold=_get_float(tier1_raw, "edge_threshold", 0.02, lo=0.0, hi=1.0),
        decay_lambda=_get_float(tier1_raw, "decay_lambda", 0.1, lo=0.0, hi=1.0),
        top_k=_get_int(tier1_raw, "top_k", 64, lo=1, hi=100000),
        max_iterations=_get_int(tier1_raw, "max_iterations", 4, lo=1, hi=1000),
        relation_bias=relation_bias,
        energy_threshold_formula=tier1_raw.get("energy_threshold_formula"),
        t_conf_coefficient=_get_float_opt(tier1_raw, "T_conf_coefficient"),
        multiplied_at_runtime=_get_bool_opt(tier1_raw, "multiplied_at_runtime"),
    )

    analogy_raw = resonance_data.get("tier2", {}).get("analogy_parameters", {})
    analogy_params = AnalogyParameters(
        lsh_bands=_get_int(analogy_raw, "lsh_bands", 16, lo=1, hi=256),
        lsh_tables=_get_int(analogy_raw, "lsh_tables", 8, lo=1, hi=256),
        temp_edge_strength=_get_float(analogy_raw, "temp_edge_strength", 0.5, lo=0.0, hi=1.0),
        overlap_validation_required=bool(analogy_raw.get("overlap_validation_required", True)),
        jaccard_overlap_min=_get_float(analogy_raw, "jaccard_overlap_min", 0.3, lo=0.0, hi=1.0),
        mini_propagation_steps=_get_int(analogy_raw, "mini_propagation_steps", 2, lo=1, hi=100),
        edge_confidence_min=_get_float(analogy_raw, "edge_confidence_min", 0.5, lo=0.0, hi=1.0),
    )

    tier2_raw = resonance_data.get("tier2", {})

    lsh_bands_dup = tier2_raw.get("analogy_lsh_bands")
    lsh_tables_dup = tier2_raw.get("analogy_lsh_tables")
    jaccard_dup = tier2_raw.get("analogy_jaccard_overlap_threshold")
    validated_threshold = tier2_raw.get("analogy_validated_threshold")

    if lsh_bands_dup is not None and int(lsh_bands_dup) != analogy_params.lsh_bands:
        logger.warning("tier2.analogy_lsh_bands (%s) differs from analogy_parameters.lsh_bands (%s)", lsh_bands_dup, analogy_params.lsh_bands)
    if lsh_tables_dup is not None and int(lsh_tables_dup) != analogy_params.lsh_tables:
        logger.warning("tier2.analogy_lsh_tables (%s) differs from analogy_parameters.lsh_tables (%s)", lsh_tables_dup, analogy_params.lsh_tables)
    if jaccard_dup is not None and float(jaccard_dup) != analogy_params.jaccard_overlap_min:
        logger.warning("tier2.analogy_jaccard_overlap_threshold (%s) differs from analogy_parameters.jaccard_overlap_min (%s)", jaccard_dup, analogy_params.jaccard_overlap_min)
    if validated_threshold is not None:
        logger.info("tier2.analogy_validated_threshold=%s (not yet implemented, stored for reference)", validated_threshold)

    tier2 = Tier2Config(
        enabled=bool(tier2_raw.get("enabled", True)),
        propagation_threshold=_get_float(tier2_raw, "propagation_threshold", 0.04, lo=0.0, hi=1.0),
        edge_threshold=_get_float(tier2_raw, "edge_threshold", 0.08, lo=0.0, hi=1.0),
        decay_lambda=_get_float(tier2_raw, "decay_lambda", 0.1, lo=0.0, hi=1.0),
        top_k=_get_int(tier2_raw, "top_k", 1024, lo=1, hi=100000),
        max_iterations=_get_int(tier2_raw, "max_iterations", 8, lo=1, hi=1000),
        analogies_enabled=bool(tier2_raw.get("analogies_enabled", True)),
        analogy_parameters=analogy_params,
    )

    energy_raw = resonance_data.get("energy", {})
    energy = EnergyConfig(
        formula=str(energy_raw.get("formula", "sum(activation * (strength * confidence))")),
        tier1_threshold=_get_float(energy_raw, "tier1_threshold", 0.4, lo=0.0, hi=1e6),
        tier2_min_improvement=_get_float(energy_raw, "tier2_min_improvement", 0.2, lo=0.0, hi=1e6),
    )

    temporal_raw = resonance_data.get("temporal", {})
    temporal = TemporalConfig(
        gamma=_get_float(temporal_raw, "gamma", 0.5, lo=0.0, hi=10.0),
        frequency_threshold=_get_int(temporal_raw, "frequency_threshold", 20, lo=1, hi=1000000),
    )

    es_raw = resonance_data.get("es_controller", {})
    mu_initial_raw = es_raw.get("mu_initial", {})
    mu_initial = {k: float(v) for k, v in mu_initial_raw.items()}
    es_controller = ESControllerConfig(
        enabled=bool(es_raw.get("enabled", True)),
        theta_dim=_get_int(es_raw, "theta_dim", 48, lo=1, hi=10000),
        population_size=_get_int(es_raw, "population_size", 4, lo=1, hi=10000),
        mu_initial=mu_initial,
        sigma_initial=_get_float(es_raw, "sigma_initial", 0.01, lo=1e-12, hi=1.0),
        learning_rate=_get_float(es_raw, "learning_rate", 0.02, lo=0.0, hi=1.0),
        sigma_decay_beta=_get_float(es_raw, "sigma_decay_beta", 0.1, lo=0.0, hi=1.0),
        evaluation_window=_get_int(es_raw, "evaluation_window", 8, lo=1, hi=100000),
        anchor_enabled=bool(es_raw.get("anchor_enabled", True)),
        anchor_lambda=_get_float(es_raw, "anchor_lambda", 0.001, lo=0.0, hi=1.0),
    )

    theta_raw = resonance_data.get("theta_indices", {})
    theta_indices = ThetaIndices(
        propagation_threshold=_get_int(theta_raw, "propagation_threshold", 0, lo=0, hi=1000),
        edge_threshold=_get_int(theta_raw, "edge_threshold", 1, lo=0, hi=1000),
        decay_lambda=_get_int(theta_raw, "decay_lambda", 2, lo=0, hi=1000),
        top_k=_get_int(theta_raw, "top_k", 3, lo=0, hi=1000),
        relation_bias_start=_get_int(theta_raw, "relation_bias_start", 4, lo=0, hi=1000),
        relation_bias_end=_get_int(theta_raw, "relation_bias_end", 20, lo=0, hi=1000),
        reserved_start=_get_int(theta_raw, "reserved_start", 20, lo=0, hi=1000),
        reserved_end=_get_int(theta_raw, "reserved_end", 48, lo=0, hi=1000),
        relation_bias_mapping={k: v for k, v in theta_raw.get("relation_bias_mapping", {}).items()},
    )

    # Validate relation bias mapping matches core relations
    expected_mapping_keys = [f"index_{i}" for i in range(theta_indices.relation_bias_start, theta_indices.relation_bias_end)]
    relation_bias_order = [theta_indices.relation_bias_mapping.get(k) for k in expected_mapping_keys]
    if None in relation_bias_order:
        missing = [k for k, v in zip(expected_mapping_keys, relation_bias_order) if v is None]
        raise KeyError(f"Missing relation bias mapping entries: {missing}")
    if relation_bias_order != relations:
        raise ValueError(
            "Relation bias mapping must match core relation order: "
            f"{relations} vs {relation_bias_order}"
        )

    diagnostics_raw = resonance_data.get("diagnostics", {})
    diagnostics = DiagnosticsConfig(
        log_activation_history=bool(diagnostics_raw.get("log_activation_history", False)),
        log_theta_history=bool(diagnostics_raw.get("log_theta_history", True)),
        history_buffer_size=_get_int(diagnostics_raw, "history_buffer_size", 1000, lo=1, hi=1000000),
    )

    # Validate theta dimension against reserved_end
    if theta_indices.reserved_end != es_controller.theta_dim:
        raise ValueError(
            f"theta_indices.reserved_end ({theta_indices.reserved_end}) must equal "
            f"es_controller.theta_dim ({es_controller.theta_dim})"
        )
    if theta_indices.relation_bias_end - theta_indices.relation_bias_start != len(relations):
        raise ValueError(
            f"Expected {len(relations)} relation bias slots "
            f"(indices {theta_indices.relation_bias_start}:{theta_indices.relation_bias_end}), "
            f"got {theta_indices.relation_bias_end - theta_indices.relation_bias_start}"
        )

    resonance_config = ResonanceConfig(
        algorithm=algorithm,
        tier1=tier1,
        tier2=tier2,
        energy=energy,
        temporal=temporal,
        es_controller=es_controller,
        theta_indices=theta_indices,
        diagnostics=diagnostics,
    )

    return LoadedConfigs(core=core_config, resonance=resonance_config)


def build_default_theta(resonance_config: ResonanceConfig, core_config: CoreConfig) -> np.ndarray:
    theta = np.zeros((resonance_config.es_controller.theta_dim,), dtype=np.float32)

    idx = resonance_config.theta_indices
    theta[idx.propagation_threshold] = resonance_config.tier1.propagation_threshold
    theta[idx.edge_threshold] = resonance_config.tier1.edge_threshold
    theta[idx.decay_lambda] = resonance_config.tier1.decay_lambda
    theta[idx.top_k] = resonance_config.tier1.top_k

    relation_biases = resonance_config.tier1.relation_bias
    for offset, relation in enumerate(core_config.relations):
        theta[idx.relation_bias_start + offset] = relation_biases.get(
            relation,
            resonance_config.es_controller.mu_initial.get("relation_bias_default", 1.0),
        )

    return theta


# ---- Internal helpers ----

def _get_float(d: Dict, key: str, default: float, *, lo: float, hi: float) -> float:
    raw = d.get(key, default)
    val = float(raw)
    if not (lo <= val <= hi):
        raise ValueError(f"{key} must be in [{lo}, {hi}], got {val}")
    return val


def _get_float_opt(d: Dict, key: str) -> float | None:
    if key not in d:
        return None
    return float(d[key])


def _get_bool_opt(d: Dict, key: str) -> bool | None:
    if key not in d:
        return None
    return bool(d[key])


def _get_int(d: Dict, key: str, default: int, *, lo: int, hi: int) -> int:
    raw = d.get(key, default)
    val = int(raw)
    if not (lo <= val <= hi):
        raise ValueError(f"{key} must be in [{lo}, {hi}], got {val}")
    return val


def _get_float_tuple(d: Dict, key: str, default: Tuple[float, float]) -> Tuple[float, float]:
    raw = d.get(key, default)
    return (float(raw[0]), float(raw[1]))


def _get_int_tuple(d: Dict, key: str, default: Tuple[int, int]) -> Tuple[int, int]:
    raw = d.get(key, default)
    return (int(raw[0]), int(raw[1]))
