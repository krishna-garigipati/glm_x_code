from __future__ import annotations

from pathlib import Path

import pytest

from ...config_loader import AttrDict, CoreConfig, load_yaml

TEST_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class TestAttrDict:
    def test_getattr_existing_key(self):
        d = _to_attrdict({"a": 1, "b": {"c": 2}})
        assert d.a == 1
        assert d.b.c == 2

    def test_getattr_missing_key_raises(self):
        d = AttrDict({"a": 1})
        with pytest.raises(AttributeError, match="has no key"):
            _ = d.nonexistent

    def test_setattr(self):
        d = AttrDict()
        d.foo = "bar"
        assert d["foo"] == "bar"

    def test_delattr(self):
        d = AttrDict({"x": 1})
        del d.x
        assert "x" not in d

    def test_delattr_missing_raises(self):
        d = AttrDict()
        with pytest.raises(AttributeError, match="has no key"):
            del d.nope

    def test_nested_attrdict(self):
        d = _to_attrdict({"a": {"b": {"c": 42}}})
        assert d.a.b.c == 42

    def test_list_of_dicts(self):
        d = _to_attrdict({"entries": [{"x": 1}, {"y": 2}]})
        assert d.entries[0].x == 1
        assert d.entries[1].y == 2


class TestLoadYaml:
    def test_loads_real_config(self):
        cfg = load_yaml(str(TEST_CONFIG_DIR / "config_resonance.yaml"))
        assert cfg.algorithm.propagation_type == "wilson_cowan"
        assert cfg.tier1.propagation_threshold == 0.008
        assert cfg.es_controller.theta_dim == 48

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/file.yaml")

    def test_core_config_from_yaml(self):
        cfg = CoreConfig.from_yaml(str(TEST_CONFIG_DIR / "config_core.yaml"))
        assert cfg.activation.min == 0.01
        assert cfg.activation.max == 1.0
        assert cfg.dimensions.sentence_bert_dim == 384


def _to_attrdict(d):
    if isinstance(d, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_attrdict(item) for item in d]
    return d
