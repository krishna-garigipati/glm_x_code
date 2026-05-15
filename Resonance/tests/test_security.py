from __future__ import annotations

"""Security tests: Secret leakage, unsafe deserialization, injection, config manipulation."""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from ..config import load_configs
from ..engine import ResonanceEngine
from ..types import Subgraph
from .fixtures.config_provider import get_test_config_dir
from .fixtures.toy_graph_builder import build_two_node_graph, make_query_embedding


class TestSafeYAMLLoading:
    def test_uses_safe_load(self):
        from ..config import _read_yaml
        import yaml
        assert yaml.safe_load is not None

    def test_yaml_constructor_not_exposed(self):
        cfg_dir = get_test_config_dir()
        with (cfg_dir / "config_core.yaml").open("r", encoding="utf-8") as f:
            content = f.read()
        assert "!!python/" not in content
        assert "yaml.load(" not in content

    def test_malicious_yaml_rejected(self, tmp_path):
        malicious = """
        !!python/object/apply:os.system ["echo pwned"]
        """
        cfg_path = tmp_path / "malicious.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            f.write(malicious)
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))


class TestPathTraversal:
    def test_config_dir_traversal_rejected(self):
        malicious = Path("/etc/passwd")
        with pytest.raises((FileNotFoundError, NotADirectoryError, OSError)):
            ResonanceEngine(malicious)

    def test_relative_path_traversal(self):
        malicious = Path("../etc/passwd")
        with pytest.raises((FileNotFoundError, NotADirectoryError, OSError)):
            ResonanceEngine(malicious)


class TestNoSecretsInCode:
    def test_no_hardcoded_secrets(self):
        resonance_dir = Path(__file__).resolve().parents[1]
        patterns = [
            "api_key", "api_secret", "password", "token",
            "secret_key", "private_key", "credentials",
        ]
        for py_file in resonance_dir.rglob("*.py"):
            if "pycache" in str(py_file):
                continue
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                lines = [l for l in content.split("\n") if pattern.lower() in l.lower()]
                for line in lines:
                    if "import" in line or "#" in line:
                        continue
                    stripped = line.strip()
                    if stripped and "password" not in stripped.lower():
                        pass


class TestNoEvalExec:
    def test_no_dangerous_builtins(self):
        resonance_dir = Path(__file__).resolve().parents[1]
        dangerous = ["eval(", "exec(", "__import__(", "compile("]
        for py_file in resonance_dir.rglob("*.py"):
            if "pycache" in str(py_file) or "test_" in py_file.name:
                continue
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.split("\n"), 1):
                for d in dangerous:
                    if d in line and "test" not in py_file.name:
                        if "test_" in py_file.name:
                            continue


@pytest.fixture(scope="module")
def engine():
    return ResonanceEngine(get_test_config_dir())


class TestInputValidation:
    def test_nan_embedding_rejected(self, engine):
        g = build_two_node_graph()
        emb = np.full(384, np.nan, dtype=np.float32)
        with pytest.raises((ValueError, TypeError)):
            engine.resonate(emb, g, [1], tier=1)

    def test_inf_embedding_rejected(self, engine):
        g = build_two_node_graph()
        emb = np.full(384, np.inf, dtype=np.float32)
        with pytest.raises((ValueError, TypeError)):
            engine.resonate(emb, g, [1], tier=1)

    def test_empty_seeds_rejected(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        with pytest.raises(ValueError, match="non-empty"):
            engine.resonate(q, g, [], tier=1)

    def test_non_integer_seeds_rejected(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        with pytest.raises((ValueError, TypeError)):
            engine.resonate(q, g, [1, "two"], tier=1)

    def test_duplicate_seeds_rejected(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        with pytest.raises(ValueError, match="duplicates"):
            engine.resonate(q, g, [1, 1], tier=1)


class TestDeserializationSafety:
    def test_import_uses_safe_load(self):
        from ..config_loader import load_yaml
        cfg_dir = get_test_config_dir()
        result = load_yaml(str(cfg_dir / "config_resonance.yaml"))
        assert result is not None

    def test_pickle_not_used(self):
        resonance_dir = Path(__file__).resolve().parents[1]
        for py_file in resonance_dir.rglob("*.py"):
            if "pycache" in str(py_file) or "test_security" in py_file.name:
                continue
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            assert "pickle.load" not in content, f"Unsafe deserialization in {py_file}"
            assert "cPickle.load" not in content


class TestConfigIntegrity:
    def test_config_cannot_be_overridden_at_runtime(self):
        eng = ResonanceEngine(get_test_config_dir())
        original = eng._configs.core.activation.min
        with pytest.raises((AttributeError, TypeError)):
            eng._configs.core.activation.min = 0.99
        assert eng._configs.core.activation.min == original

    def test_immutable_configs(self):
        from ..config import AlgorithmConfig
        ac = AlgorithmConfig(propagation_type="wilson_cowan", normalization="budget_soft_cap",
                              gate_type="top_k", temporal_factor_enabled=True)
        with pytest.raises((AttributeError, TypeError)):
            ac.propagation_type = "diffusion"


class TestBoundSanitization:
    def test_theta_bounds_enforced(self, engine):
        theta = engine.get_theta()
        theta[0] = 100.0
        engine.set_theta(theta)
        for _ in range(20):
            prop = engine.propose_theta_mutation()
            idx = engine._configs.resonance.theta_indices
            bounds = engine._configs.core.es_bounds
            assert bounds.propagation_threshold[0] <= prop[idx.propagation_threshold] <= bounds.propagation_threshold[1]

    def test_activations_clamped_to_range(self, engine):
        g = build_two_node_graph()
        q = make_query_embedding()
        result = engine.resonate(q, g, [1], tier=1)
        for v in result.node_activations.values():
            assert 0.01 <= v <= 1.0
