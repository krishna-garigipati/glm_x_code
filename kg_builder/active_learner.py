import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ActiveLearner:
    def __init__(self):
        self._weak_regions: Dict[str, int] = defaultdict(int)
        self._total_queries: int = 0
        self._successful_queries: int = 0

    def observe_failure(
        self,
        query_text: str,
        template_matched: bool,
        walk_confidence: float,
        path_labels: Optional[List[str]] = None,
        plan_confidence: float = 0.0,
    ) -> None:
        self._total_queries += 1
        if template_matched:
            self._successful_queries += 1
        if not template_matched and path_labels:
            for label in path_labels:
                if label:
                    self._weak_regions[label.lower()] += 1
        elif walk_confidence < 0.3 and path_labels:
            for label in path_labels:
                if label:
                    self._weak_regions[label.lower()] += 1
        elif plan_confidence < 0.3 and path_labels:
            for label in path_labels:
                if label:
                    self._weak_regions[label.lower()] += 1

    def get_weak_entities(self, top_k: int = 10) -> List[str]:
        sorted_regions = sorted(self._weak_regions.items(), key=lambda x: -x[1])
        return [entity for entity, _ in sorted_regions[:top_k]]

    def get_success_rate(self) -> float:
        if self._total_queries == 0:
            return 0.0
        return self._successful_queries / self._total_queries

    def get_weak_entity_count(self) -> int:
        return len(self._weak_regions)

    def reset(self):
        self._weak_regions.clear()
        self._total_queries = 0
        self._successful_queries = 0

    def get_report(self) -> Dict[str, Any]:
        return {
            "total_queries": self._total_queries,
            "successful_queries": self._successful_queries,
            "success_rate": self.get_success_rate(),
            "weak_entity_count": self.get_weak_entity_count(),
            "top_weak_entities": self.get_weak_entities(5),
        }
