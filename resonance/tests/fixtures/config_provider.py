from __future__ import annotations

from pathlib import Path

import numpy as np

from ...config import (
    AlgorithmConfig,
    AnalogyParameters,
    CoreActivationConfig,
    CoreConfig,
    CoreResonanceConfig,
    DiagnosticsConfig,
    ESBounds,
    ESControllerConfig,
    EnergyConfig,
    LoadedConfigs,
    ResonanceConfig,
    TemporalConfig,
    ThetaIndices,
    Tier2Config,
    TierConfig,
    load_configs,
)

TEST_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def get_test_config_dir() -> Path:
    return TEST_CONFIG_DIR


def load_test_configs() -> LoadedConfigs:
    return load_configs(TEST_CONFIG_DIR)


def build_minimal_core_config() -> CoreConfig:
    return CoreConfig(
        activation=CoreActivationConfig(min=0.01, max=1.0, default=0.01, threshold_resonance=0.2),
        resonance=CoreResonanceConfig(
            propagation_threshold=0.008,
            edge_threshold=0.02,
            decay_lambda=0.1,
            top_k=64,
            budget_max=2.0,
            convergence_epsilon=0.001,
            tier1_energy_threshold=0.4,
            tier2_max_nodes=1024,
            analogy_validation_overlap=0.3,
        ),
        relations=[
            "is_a", "has_property", "causes", "caused_by",
            "follows", "precedes", "contradicts", "supports",
            "associated_with", "example_of", "part_of",
            "synonym", "antonym", "temporal_coincident",
            "spatial_near", "linguistic_maps",
        ],
        es_bounds=ESBounds(
            propagation_threshold=(0.001, 0.05),
            edge_threshold=(0.01, 0.1),
            decay_lambda=(0.05, 0.5),
            top_k=(16, 1024),
            relation_bias=(0.0, 2.0),
        ),
    )


def build_minimal_resonance_config() -> ResonanceConfig:
    relation_bias = {
        "is_a": 1.0, "has_property": 0.8, "causes": 1.2, "caused_by": 1.1,
        "follows": 0.7, "precedes": 0.7, "contradicts": 0.3, "supports": 1.0,
        "associated_with": 0.5, "example_of": 0.9, "part_of": 0.8,
        "synonym": 1.0, "antonym": 0.4, "temporal_coincident": 0.6,
        "spatial_near": 0.5, "linguistic_maps": 0.7,
    }
    return ResonanceConfig(
        algorithm=AlgorithmConfig(
            propagation_type="wilson_cowan",
            normalization="budget_soft_cap",
            gate_type="top_k",
            temporal_factor_enabled=True,
        ),
        tier1=TierConfig(
            propagation_threshold=0.008,
            edge_threshold=0.02,
            decay_lambda=0.1,
            top_k=64,
            max_iterations=4,
            relation_bias=relation_bias,
            energy_threshold_formula="E_total > 0.4 * |N_seed| * 1.0",
            t_conf_coefficient=0.4,
            multiplied_at_runtime=True,
        ),
        tier2=Tier2Config(
            enabled=True,
            propagation_threshold=0.04,
            edge_threshold=0.08,
            decay_lambda=0.1,
            top_k=1024,
            max_iterations=8,
            analogies_enabled=True,
            analogy_parameters=AnalogyParameters(
                lsh_bands=16,
                lsh_tables=8,
                temp_edge_strength=0.5,
                overlap_validation_required=True,
                jaccard_overlap_min=0.3,
                mini_propagation_steps=2,
                edge_confidence_min=0.5,
            ),
        ),
        energy=EnergyConfig(
            formula="sum(activation * (strength * confidence))",
            tier1_threshold=0.4,
            tier2_min_improvement=0.2,
        ),
        temporal=TemporalConfig(gamma=0.5, frequency_threshold=20),
        es_controller=ESControllerConfig(
            enabled=True,
            theta_dim=48,
            population_size=4,
            mu_initial={
                "propagation_threshold": 0.008,
                "edge_threshold": 0.02,
                "decay_lambda": 0.1,
                "top_k": 64,
                "relation_bias_default": 1.0,
            },
            sigma_initial=0.01,
            learning_rate=0.02,
            sigma_decay_beta=0.1,
            evaluation_window=8,
            anchor_enabled=True,
            anchor_lambda=0.001,
        ),
        theta_indices=ThetaIndices(
            propagation_threshold=0,
            edge_threshold=1,
            decay_lambda=2,
            top_k=3,
            relation_bias_start=4,
            relation_bias_end=20,
            reserved_start=20,
            reserved_end=48,
            relation_bias_mapping={
                "index_4": "is_a", "index_5": "has_property",
                "index_6": "causes", "index_7": "caused_by",
                "index_8": "follows", "index_9": "precedes",
                "index_10": "contradicts", "index_11": "supports",
                "index_12": "associated_with", "index_13": "example_of",
                "index_14": "part_of", "index_15": "synonym",
                "index_16": "antonym", "index_17": "temporal_coincident",
                "index_18": "spatial_near", "index_19": "linguistic_maps",
            },
        ),
        diagnostics=DiagnosticsConfig(
            log_activation_history=False,
            log_theta_history=True,
            history_buffer_size=1000,
        ),
    )


def build_minimal_loaded_configs() -> LoadedConfigs:
    return LoadedConfigs(
        core=build_minimal_core_config(),
        resonance=build_minimal_resonance_config(),
    )
