from typing import List, Optional, Dict, Tuple
from .types import Subgraph
from .config import RuleDefinition, MappingConfig


class HeuristicPlanner:
    def __init__(self, config: MappingConfig):
        self.config = config
        self._rules: List[RuleDefinition] = list(config.rule_definitions)

    def load_rules(self, filepath: str):
        import json
        with open(filepath, "r") as f:
            data = json.load(f)
        self._rules = [RuleDefinition(**r) for r in data.get("rules", [])]

    def add_rule(self, condition: str, plan: List[int]):
        self._rules.append(RuleDefinition(condition=condition, plan=plan))

    def clear_rules(self):
        self._rules.clear()

    def evaluate(self, subgraph: Subgraph, query_text: str = "") -> Optional[Tuple[List[int], float]]:
        if not self.config.heuristic_rules_enabled:
            return None

        env = _RuleEnvironment(subgraph, query_text)

        for rule in self._rules:
            try:
                result = env.evaluate(rule.condition)
                if result:
                    return rule.plan, 0.5
            except Exception:
                continue

        return None


class _RuleEnvironment:
    def __init__(self, subgraph: Subgraph, query_text: str = ""):
        self._subgraph = subgraph
        self._query_text = query_text
        self._avg_conf = self._compute_average_confidence()

    def evaluate(self, condition: str) -> bool:
        safe_globals = {
            "len": len,
            "has_edge_type": self._has_edge_type,
            "query_contains": self._query_contains,
        }
        safe_locals = {
            "subgraph": _SubgraphProxy(self._subgraph),
            "average_confidence": self._avg_conf,
        }
        try:
            result = eval(condition, {"__builtins__": {}}, {**safe_globals, **safe_locals})
            return bool(result)
        except Exception:
            return False

    def _has_edge_type(self, rel_type: str) -> bool:
        for _, _, r in self._subgraph.edges:
            if r == rel_type:
                return True
        return False

    def _query_contains(self, word: str) -> bool:
        return word.lower() in self._query_text.lower()

    def _compute_average_confidence(self) -> float:
        if not self._subgraph.edge_confidences:
            return 0.0
        values = list(self._subgraph.edge_confidences.values())
        return float(sum(values) / len(values))


class _SubgraphProxy:
    def __init__(self, subgraph: Subgraph):
        self.nodes = subgraph.nodes
        self.edges = subgraph.edges
