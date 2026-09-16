"""Expected-relation to edge bias lookup table (DEVIATION 9).

Replaces the intent-to-edge IntentBiasTable. The GraphWalker is now driven by an
ordered relation chain from the query-relation extractor; given the relation a
step *expects* (e.g. "causes"), this table returns a bias multiplier for each
candidate edge relation so the walker prefers edges matching the expected chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Dict, Iterable, List, Optional


# Legacy intent vocabulary (DEVIATION 9: dormant). Used only by the backward-compat
# shim get_bias_for_intent() so legacy training code that still calls
# apply_reward(path_intents=...) keeps working. The runtime pipeline never reads intents.
LEGACY_INTENT_TO_RELATION: Dict[int, str] = {
    0: "is_a",              # define
    1: "supports",          # assert_fact
    2: "causes",            # explain_cause
    3: "causes",            # explain_effect
    4: "contradicts",       # contrast
    5: "synonym",           # compare
    6: "part_of",           # list
    7: "example_of",        # example
    8: "supports",          # conclude
    9: "associated_with",   # question
    10: "contradicts",      # uncertain
    11: "associated_with",  # clarify
    12: "part_of",          # summarize
    13: "has_property",     # elaborate
    14: "follows",          # transition
    15: "supports",         # emphasize
}


@dataclass
class RelationBiasTable:
    _biases: Dict[str, Dict[str, float]]
    _lock: RLock

    def __init__(self, biases: Dict[str, Dict[str, float]] | None = None):
        self._biases = {
            str(k): {str(rel): float(bias) for rel, bias in bias_map.items()}
            for k, bias_map in (biases or {}).items()
        }
        self._lock = RLock()

    def get_bias(self, expected_relation: str, candidate_relation: str) -> float:
        if expected_relation is None:
            return 1.0
        with self._lock:
            bias_map = self._biases.get(str(expected_relation), {})
            if candidate_relation in bias_map:
                return float(bias_map[candidate_relation])
            if "default" in bias_map:
                return float(bias_map["default"])
            return 1.0

    def get_bias_for_intent(self, intent_id: int, candidate_relation: str) -> float:
        """Legacy shim: maps an intent id to its expected relation, then biases."""
        expected_relation = LEGACY_INTENT_TO_RELATION.get(int(intent_id), "associated_with")
        return self.get_bias(expected_relation, candidate_relation)

    def update_bias(self, expected_relation: str, candidate_relation: str, bias: float) -> None:
        with self._lock:
            key = str(expected_relation)
            if key not in self._biases:
                self._biases[key] = {}
            self._biases[key][candidate_relation] = float(bias)

    def replace_biases(self, overrides: Dict[str, Dict[str, float]]) -> None:
        """Replace specific (expected_relation, candidate_relation) entries. Used by the
        EvolutionaryController to push learned theta into the walker."""
        with self._lock:
            for expected, bias_map in overrides.items():
                key = str(expected)
                if key not in self._biases:
                    self._biases[key] = {}
                for candidate, bias in bias_map.items():
                    self._biases[key][candidate] = float(bias)

    def expected_relations(self) -> List[str]:
        with self._lock:
            return sorted(str(k) for k in self._biases.keys())

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {k: dict(v) for k, v in self._biases.items()}

    def strip_relations(self, keep: Iterable[str]) -> None:
        """Drop bias entries for candidate relations that vanished from the KG."""
        keep_set = set(keep)
        with self._lock:
            for bias_map in self._biases.values():
                for relation in [r for r in bias_map if r not in keep_set]:
                    del bias_map[relation]