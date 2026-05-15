from typing import Any, Dict

from .errors import GraphStoreError


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise GraphStoreError("E005", "PyYAML is required to load configuration") from exc
    with open(path, "r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)
    if not isinstance(data, dict):
        raise GraphStoreError("E005", "Invalid configuration format")
    return data
