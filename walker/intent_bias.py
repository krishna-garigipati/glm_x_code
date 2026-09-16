"""DORMANT LEGACY COMPAT SHIM (DEVIATION 9).

IntentBiasTable is superseded by RelationBiasTable (walker/relation_bias.py).
This module is kept so legacy training code and older tests that still import
`from walker.intent_bias import IntentBiasTable` do not crash. Do not use in new
code; the runtime walker reads only RelationBiasTable.

The shim reproduces the *legacy* intent->edge bias semantics exactly (per-intent
relation maps with a "default" fallback and 1.0 for unknown intents) so dormant
tests/trainers behave as before, while mirroring writes into the relation table
for values that share an expected relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .relation_bias import LEGACY_INTENT_TO_RELATION, RelationBiasTable


@dataclass
class IntentBiasTable:
    _biases: Dict[int, Dict[str, float]]

    def __init__(self, biases: Dict[int, Dict[str, float]]):
        self._biases = {int(k): dict(v) for k, v in biases.items()}
        self._table = RelationBiasTable(
            {
                LEGACY_INTENT_TO_RELATION.get(int(k), "associated_with"): dict(v)
                for k, v in biases.items()
            }
        )

    def get_bias(self, intent_id: int, relation: str) -> float:
        row = self._biases.get(int(intent_id))
        if row is None:
            return 1.0
        if relation in row:
            return float(row[relation])
        if "default" in row:
            return float(row["default"])
        return 1.0

    def update_bias(self, intent_id: int, relation: str, bias: float) -> None:
        key = int(intent_id)
        row = self._biases.setdefault(key, {})
        row[relation] = float(bias)
        expected = LEGACY_INTENT_TO_RELATION.get(key, "associated_with")
        self._table.update_bias(expected, relation, float(bias))

    def snapshot(self) -> Dict[int, Dict[str, float]]:
        return {int(k): dict(v) for k, v in self._biases.items()}