from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from ...config import (
    _VALID_GATE_TYPES,
    _VALID_NORMALIZATIONS,
    _VALID_PROPAGATION_TYPES,
    AlgorithmConfig,
    AnalogyParameters,
    CoreActivationConfig,
    CoreConfig,
    CoreResonanceConfig,
    DiagnosticsConfig,
    ESBounds,
    ESControllerConfig,
    EnergyConfig,
    ResonanceConfig,
    TemporalConfig,
    ThetaIndices,
    Tier2Config,
    TierConfig,
    build_default_theta,
    load_configs,
)

TEST_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class TestCoreConfigParsing:
    def test_loads_from_real_files(self):
        configs = load_configs(TEST_CONFIG_DIR)
        assert configs.core.activation.min == 0.01
        assert configs.core.activation.max == 1.0
        assert configs.core.activation.default == 0.01
        assert len(configs.core.relations) == 16

    def test_relation_count_matches_expected(self, loaded_configs):
        assert len(loaded_configs.core.relations) == 16

    def test_relation_order_preserved(self, loaded_configs):
        expected_first = "is_a"
        assert loaded_configs.core.relations[0] == expected_first

    def test_es_bounds_have_correct_structure(self, loaded_configs):
        b = loaded_configs.core.es_bounds
        assert b.propagation_threshold[0] < b.propagation_threshold[1]
        assert b.edge_threshold[0] < b.edge_threshold[1]
        assert b.decay_lambda[0] < b.decay_lambda[1]
        assert b.top_k[0] < b.top_k[1]
        assert b.relation_bias[0] < b.relation_bias[1]

    def test_config_dir_not_found_raises(self):
        with pytest.raises((FileNotFoundError, NotADirectoryError)):
            load_configs(Path("/nonexistent/path"))


class TestResonanceConfigParsing:
    def test_tier1_params_loaded(self, loaded_configs):
        t1 = loaded_configs.resonance.tier1
        assert t1.propagation_threshold == 0.008
        assert t1.edge_threshold == 0.02
        assert t1.decay_lambda == 0.1
        assert t1.top_k == 64
        assert t1.max_iterations == 4

    def test_all_relation_biases_present(self, loaded_configs):
        rb = loaded_configs.resonance.tier1.relation_bias
        expected = ["is_a", "has_property", "causes", "caused_by", "follows", "precedes",
                     "contradicts", "supports", "associated_with", "example_of", "part_of",
                     "synonym", "antonym", "temporal_coincident", "spatial_near", "linguistic_maps"]
        for rel in expected:
            assert rel in rb, f"Missing relation bias: {rel}"
        assert len(rb) == 16

    def test_algorithm_defaults(self, loaded_configs):
        alg = loaded_configs.resonance.algorithm
        assert alg.propagation_type in _VALID_PROPAGATION_TYPES
        assert alg.normalization in _VALID_NORMALIZATIONS
        assert alg.gate_type in _VALID_GATE_TYPES
        assert isinstance(alg.temporal_factor_enabled, bool)

    def test_tier2_config(self, loaded_configs):
        t2 = loaded_configs.resonance.tier2
        assert isinstance(t2.enabled, bool)
        assert t2.propagation_threshold == 0.04
        assert t2.top_k == 1024
        assert t2.max_iterations == 8

    def test_analogy_params_structure(self, loaded_configs):
        ap = loaded_configs.resonance.tier2.analogy_parameters
        assert 1 <= ap.lsh_bands <= 256
        assert 1 <= ap.lsh_tables <= 256
        assert 0.0 <= ap.temp_edge_strength <= 1.0
        assert isinstance(ap.overlap_validation_required, bool)
        assert 0.0 <= ap.jaccard_overlap_min <= 1.0
        assert 1 <= ap.mini_propagation_steps <= 100
        assert 0.0 <= ap.edge_confidence_min <= 1.0

    def test_energy_config(self, loaded_configs):
        e = loaded_configs.resonance.energy
        assert e.formula == "sum(activation * (strength * confidence))"
        assert e.tier1_threshold == 0.4
        assert e.tier2_min_improvement == 0.2

    def test_temporal_config(self, loaded_configs):
        t = loaded_configs.resonance.temporal
        assert t.gamma == 0.5
        assert t.frequency_threshold == 20

    def test_es_controller_config(self, loaded_configs):
        es = loaded_configs.resonance.es_controller
        assert es.theta_dim == 48
        assert 1 <= es.population_size <= 10000
        assert 0.0 <= es.learning_rate <= 1.0
        assert es.evaluation_window >= 1
        assert isinstance(es.anchor_enabled, bool)

    def test_theta_indices(self, loaded_configs):
        ti = loaded_configs.resonance.theta_indices
        assert ti.propagation_threshold == 0
        assert ti.edge_threshold == 1
        assert ti.decay_lambda == 2
        assert ti.top_k == 3
        assert ti.relation_bias_start == 4
        assert ti.relation_bias_end == 20
        assert ti.reserved_start == 20
        assert ti.reserved_end == 48
        assert len(ti.relation_bias_mapping) == 16

    def test_theta_dim_matches_reserved_end(self, loaded_configs):
        ti = loaded_configs.resonance.theta_indices
        es = loaded_configs.resonance.es_controller
        assert ti.reserved_end == es.theta_dim

    def test_diagnostics_config(self, loaded_configs):
        d = loaded_configs.resonance.diagnostics
        assert isinstance(d.log_activation_history, bool)
        assert isinstance(d.log_theta_history, bool)
        assert d.history_buffer_size >= 1

    def test_invalid_propagation_type_raises(self, tmp_path):
        cfg = _write_temp_config(tmp_path, {"algorithm": {"propagation_type": "invalid_mode"}})
        with pytest.raises(ValueError, match="Invalid propagation_type"):
            load_configs(tmp_path)

    def test_invalid_normalization_raises(self, tmp_path):
        cfg = _write_temp_config(tmp_path, {"algorithm": {"normalization": "magic_norm"}})
        with pytest.raises(ValueError, match="Invalid normalization"):
            load_configs(tmp_path)

    def test_invalid_gate_type_raises(self, tmp_path):
        cfg = _write_temp_config(tmp_path, {"algorithm": {"gate_type": "magic_gate"}})
        with pytest.raises(ValueError, match="Invalid gate_type"):
            load_configs(tmp_path)

    def test_relation_bias_out_of_range_raises(self, tmp_path, loaded_configs):
        import copy
        res = _read_yaml_safe(TEST_CONFIG_DIR / "config_resonance.yaml")
        res["tier1"]["relation_bias"]["is_a"] = 5.0
        cfg = _write_raw_configs(tmp_path, _read_yaml_safe(TEST_CONFIG_DIR / "config_core.yaml"), res)
        with pytest.raises(ValueError, match="relation_bias"):
            load_configs(tmp_path)

    def test_theta_reserved_end_mismatch_raises(self, tmp_path, loaded_configs):
        res = _read_yaml_safe(TEST_CONFIG_DIR / "config_resonance.yaml")
        res["es_controller"]["theta_dim"] = 50
        cfg = _write_raw_configs(tmp_path, _read_yaml_safe(TEST_CONFIG_DIR / "config_core.yaml"), res)
        with pytest.raises(ValueError, match="reserved_end"):
            load_configs(tmp_path)

    def test_relation_count_mismatch_raises(self, tmp_path, loaded_configs):
        res = _read_yaml_safe(TEST_CONFIG_DIR / "config_resonance.yaml")
        res["theta_indices"]["relation_bias_end"] = 18
        cfg = _write_raw_configs(tmp_path, _read_yaml_safe(TEST_CONFIG_DIR / "config_core.yaml"), res)
        with pytest.raises(ValueError):
            load_configs(tmp_path)

    def test_activation_min_greater_than_max_raises(self, tmp_path):
        core = _read_yaml_safe(TEST_CONFIG_DIR / "config_core.yaml")
        core["activation"]["min"] = 0.3
        core["activation"]["max"] = 0.3
        core["activation"]["default"] = 0.3
        _write_raw_configs(tmp_path, core, _read_yaml_safe(TEST_CONFIG_DIR / "config_resonance.yaml"))
        with pytest.raises(ValueError, match="activation.max must be > activation.min"):
            load_configs(tmp_path)


class TestBuildDefaultTheta:
    def test_returns_correct_shape(self, minimal_configs):
        theta = build_default_theta(minimal_configs.resonance, minimal_configs.core)
        assert theta.shape == (48,)
        assert theta.dtype == np.float32

    def test_core_params_set_correctly(self, minimal_configs):
        theta = build_default_theta(minimal_configs.resonance, minimal_configs.core)
        idx = minimal_configs.resonance.theta_indices
        assert theta[idx.propagation_threshold] == minimal_configs.resonance.tier1.propagation_threshold
        assert theta[idx.edge_threshold] == minimal_configs.resonance.tier1.edge_threshold
        assert theta[idx.decay_lambda] == minimal_configs.resonance.tier1.decay_lambda
        assert theta[idx.top_k] == minimal_configs.resonance.tier1.top_k

    def test_relation_biases_set_correctly(self, minimal_configs):
        theta = build_default_theta(minimal_configs.resonance, minimal_configs.core)
        idx = minimal_configs.resonance.theta_indices
        for offset, rel in enumerate(minimal_configs.core.relations):
            expected = minimal_configs.resonance.tier1.relation_bias[rel]
            assert abs(float(theta[idx.relation_bias_start + offset]) - expected) < 1e-6

    def test_reserved_dims_are_zero(self, minimal_configs):
        theta = build_default_theta(minimal_configs.resonance, minimal_configs.core)
        idx = minimal_configs.resonance.theta_indices
        reserved = theta[idx.reserved_start:idx.reserved_end]
        assert np.all(reserved == 0.0)

    def test_all_finite(self, minimal_configs):
        theta = build_default_theta(minimal_configs.resonance, minimal_configs.core)
        assert np.all(np.isfinite(theta))


class TestConfigDataclasses:
    def test_algorithm_config_immutable(self):
        ac = AlgorithmConfig(propagation_type="wilson_cowan", normalization="budget_soft_cap",
                             gate_type="top_k", temporal_factor_enabled=True)
        with pytest.raises(AttributeError):
            ac.propagation_type = "diffusion"

    def test_tier_config_valid(self):
        tc = TierConfig(
            propagation_threshold=0.008, edge_threshold=0.02, decay_lambda=0.1,
            top_k=64, max_iterations=4, relation_bias={"is_a": 1.0},
            energy_threshold_formula=None, t_conf_coefficient=None, multiplied_at_runtime=None,
        )
        assert tc.top_k == 64

    def test_core_config_valid(self):
        cc = CoreConfig(
            activation=CoreActivationConfig(min=0.01, max=1.0, default=0.01, threshold_resonance=0.2),
            resonance=CoreResonanceConfig(
                propagation_threshold=0.008, edge_threshold=0.02, decay_lambda=0.1,
                top_k=64, budget_max=2.0, convergence_epsilon=0.001,
                tier1_energy_threshold=0.4, tier2_max_nodes=1024, analogy_validation_overlap=0.3,
            ),
            relations=["is_a", "has_property"],
            es_bounds=ESBounds(
                propagation_threshold=(0.001, 0.05), edge_threshold=(0.01, 0.1),
                decay_lambda=(0.05, 0.5), top_k=(16, 1024), relation_bias=(0.0, 2.0),
            ),
        )
        assert cc.activation.min == 0.01
        assert len(cc.relations) == 2


def _read_yaml_safe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_temp_config(tmp_path: Path, overrides: dict) -> Path:
    core = _read_yaml_safe(TEST_CONFIG_DIR / "config_core.yaml")
    res = _read_yaml_safe(TEST_CONFIG_DIR / "config_resonance.yaml")
    for section, values in overrides.items():
        if section in res:
            res[section].update(values)
        else:
            res[section] = values
    _write_raw_configs(tmp_path, core, res)
    return tmp_path


def _write_raw_configs(tmp_path: Path, core: dict, resonance: dict) -> None:
    with (tmp_path / "config_core.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(core, f, default_flow_style=False)
    with (tmp_path / "config_resonance.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(resonance, f, default_flow_style=False)
