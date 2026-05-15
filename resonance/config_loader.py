from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"AttrDict has no key {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"AttrDict has no key {name!r}")


def _to_attrdict(d: Any) -> Any:
    if isinstance(d, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_attrdict(item) for item in d]
    return d


def load_yaml(path: str) -> AttrDict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _to_attrdict(data)


class CoreConfig(AttrDict):
    @classmethod
    def from_yaml(cls, path: str) -> CoreConfig:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return _to_attrdict(data)
