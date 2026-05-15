from typing import Any, Dict

from .errors import ConfigurationError


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise ConfigurationError("PyYAML is required to load configuration") from exc
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            data = yaml.safe_load(file_handle)
    except Exception as exc:
        raise ConfigurationError(f"Failed to load config from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("Invalid configuration format")
    return data
