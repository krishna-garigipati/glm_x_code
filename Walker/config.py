"""Configuration loading and typed configs for the Walker component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
        if data is None:
            return {}
        return data


@dataclass(frozen=True)
class ActivationConfig:
    min: float
    max: float


@dataclass(frozen=True)
class WalkerCoreConfig:
    default_temperature: float
    softmax_temperature_range: Tuple[float, float]
    max_steps: int
    min_activation_to_continue: float


@dataclass(frozen=True)
class CoreConfig:
    activation: ActivationConfig
    walker: WalkerCoreConfig
    relations: Dict[int, str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CoreConfig":
        data = load_yaml(path)
        activation = data.get("activation", {})
        walker = data.get("walker", {})
        return cls(
            activation=ActivationConfig(
                min=float(activation["min"]),
                max=float(activation["max"]),
            ),
            walker=WalkerCoreConfig(
                default_temperature=float(walker["default_temperature"]),
                softmax_temperature_range=tuple(walker["softmax_temperature_range"]),
                max_steps=int(walker["max_steps"]),
                min_activation_to_continue=float(walker["min_activation_to_continue"]),
            ),
            relations={int(k): v for k, v in data.get("relations", {}).items()},
        )


@dataclass(frozen=True)
class WalkConfig:
    max_steps: int
    min_activation: float
    temperature: float
    temperature_range: Tuple[float, float]
    allow_cycles: bool
    allow_backtrack: bool
    restart_on_dead_end: bool
    restart_penalty: float


@dataclass(frozen=True)
class ScoringConfig:
    formula: str
    weight_strength: float
    weight_confidence: float
    weight_target_activation: float
    weight_intent_bias: float
    normalization: str
    softmax_temperature: float


@dataclass(frozen=True)
class EligibilityConfig:
    gamma: float
    formula: str
    trace_key_format: str


@dataclass(frozen=True)
class PathConfig:
    max_length: int
    record_activations: bool
    record_timestamps: bool
    record_edge_confidence: bool
    output_format: str


@dataclass(frozen=True)
class DebugConfig:
    log_decision_scores: bool
    log_path_taken: bool
    save_all_paths: bool


@dataclass(frozen=True)
class WalkerConfig:
    walk: WalkConfig
    intent_biases: Dict[int, Dict[str, float]]
    scoring: ScoringConfig
    eligibility: EligibilityConfig
    path: PathConfig
    debug: DebugConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WalkerConfig":
        data = load_yaml(path)
        walk = data.get("walk", {})
        scoring = data.get("scoring", {})
        eligibility = data.get("eligibility", {})
        path_cfg = data.get("path", {})
        debug = data.get("debug", {})
        return cls(
            walk=WalkConfig(
                max_steps=int(walk["max_steps"]),
                min_activation=float(walk["min_activation"]),
                temperature=float(walk["temperature"]),
                temperature_range=tuple(walk["temperature_range"]),
                allow_cycles=bool(walk["allow_cycles"]),
                allow_backtrack=bool(walk["allow_backtrack"]),
                restart_on_dead_end=bool(walk["restart_on_dead_end"]),
                restart_penalty=float(walk["restart_penalty"]),
            ),
            intent_biases={int(k): dict(v) for k, v in data.get("intent_biases", {}).items()},
            scoring=ScoringConfig(
                formula=str(scoring.get("formula", "")),
                weight_strength=float(scoring["weight_strength"]),
                weight_confidence=float(scoring["weight_confidence"]),
                weight_target_activation=float(scoring["weight_target_activation"]),
                weight_intent_bias=float(scoring["weight_intent_bias"]),
                normalization=str(scoring["normalization"]),
                softmax_temperature=float(scoring["softmax_temperature"]),
            ),
            eligibility=EligibilityConfig(
                gamma=float(eligibility["gamma"]),
                formula=str(eligibility.get("formula", "")),
                trace_key_format=str(eligibility["trace_key_format"]),
            ),
            path=PathConfig(
                max_length=int(path_cfg["max_length"]),
                record_activations=bool(path_cfg["record_activations"]),
                record_timestamps=bool(path_cfg["record_timestamps"]),
                record_edge_confidence=bool(path_cfg["record_edge_confidence"]),
                output_format=str(path_cfg["output_format"]),
            ),
            debug=DebugConfig(
                log_decision_scores=bool(debug["log_decision_scores"]),
                log_path_taken=bool(debug["log_path_taken"]),
                save_all_paths=bool(debug["save_all_paths"]),
            ),
        )
