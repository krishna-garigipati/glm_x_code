from __future__ import annotations

"""Smoke tests: Verify component boots and initializes correctly."""

from pathlib import Path

import numpy as np
import pytest

from ..engine import ResonanceEngine
from ..config import load_configs
from ..tier1 import Tier1Resonance
from ..tier2 import Tier2Resonance
from ..es_controller import EvolutionaryController
from ..analogy import AnalogyFinder
from ..temporal import compute_temporal_factor
from ..energy import compute_activation_energy
from ..validation import validate_embedding, validate_seeds, validate_theta
from .fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_loaded_configs,
    build_minimal_resonance_config,
    get_test_config_dir,
    load_test_configs,
)
from .fixtures.toy_data import build_animal_kingdom_graph


class TestSmokeConfigLoading:
    def test_config_files_exist(self):
        cfg_dir = get_test_config_dir()
        assert (cfg_dir / "config_core.yaml").is_file()
        assert (cfg_dir / "config_resonance.yaml").is_file()

    def test_yaml_files_parse(self):
        cfg_dir = get_test_config_dir()
        import yaml
        for fname in ["config_core.yaml", "config_resonance.yaml"]:
            with (cfg_dir / fname).open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{fname} did not parse to dict"

    def test_load_configs_succeeds(self):
        configs = load_test_configs()
        assert configs is not None

    def test_minimal_configs_construct(self):
        configs = build_minimal_loaded_configs()
        assert configs.core.activation.min == 0.01
        assert configs.core.activation.max == 1.0


class TestSmokeComponentInit:
    def test_resonance_engine_creates(self):
        eng = ResonanceEngine(get_test_config_dir())
        assert eng is not None
        assert eng._configs is not None
        assert eng._tier1 is not None
        assert eng._tier2 is not None
        assert eng._es is not None
        assert eng._theta.shape == (48,)

    def test_tier1_creates(self):
        rc = build_minimal_resonance_config()
        cc = build_minimal_core_config()
        t1 = Tier1Resonance(
            core_config=cc, algorithm=rc.algorithm, temporal=rc.temporal,
            tier_config=rc.tier1, log_activation_history=False, history_buffer_size=1000,
        )
        assert t1 is not None
        assert t1._tier.top_k == 64

    def test_tier2_creates(self):
        rc = build_minimal_resonance_config()
        cc = build_minimal_core_config()
        t2 = Tier2Resonance(
            core_config=cc, algorithm=rc.algorithm, temporal=rc.temporal,
            tier_config=rc.tier2, relation_bias=rc.tier1.relation_bias,
            log_activation_history=False, history_buffer_size=1000,
        )
        assert t2 is not None

    def test_es_controller_creates(self):
        cc = build_minimal_core_config()
        rc = build_minimal_resonance_config()
        theta = np.zeros(48, dtype=np.float32)
        es = EvolutionaryController(
            core_config=cc, es_config=rc.es_controller, initial_theta=theta,
            theta_indices=rc.theta_indices,
        )
        assert es is not None
        assert es._mu.shape == (48,)
        assert es._sigma == 0.01

    def test_analogy_finder_creates(self):
        cc = build_minimal_core_config()
        rc = build_minimal_resonance_config()
        af = AnalogyFinder(cc, rc.tier2.analogy_parameters)
        assert af is not None


class TestSmokeCoreFunctions:
    def test_compute_temporal_factor_runs(self):
        import time
        result = compute_temporal_factor(time.time(), 10, 0.5, 20)
        assert 0.0 <= result <= 1.0

    def test_validate_embedding_runs(self):
        emb = np.zeros(384, dtype=np.float32)
        validate_embedding(emb)

    def test_validate_seeds_runs(self):
        validate_seeds([1, 2, 3])

    def test_validate_theta_runs(self):
        theta = np.zeros(48, dtype=np.float32)
        validate_theta(theta, 48)


class TestSmokeEngineHealth:
    def test_engine_has_expected_methods(self):
        eng = ResonanceEngine(get_test_config_dir())
        methods = [
            "resonate", "resonate_with_theta", "get_theta", "set_theta",
            "propose_theta_mutation", "update_es_with_reward",
            "compute_activation_energy", "check_resonance_convergence",
            "get_analogy_leaps", "clear_analogy_cache", "get_configs",
        ]
        for m in methods:
            assert hasattr(eng, m), f"Missing method: {m}"

    def test_engine_context_manager(self):
        with ResonanceEngine(get_test_config_dir()) as eng:
            assert eng is not None

    def test_toy_graph_creates(self):
        g = build_animal_kingdom_graph()
        assert g.node_count() > 0
        assert g.edge_count() > 0
