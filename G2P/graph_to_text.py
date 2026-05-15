import numpy as np
from typing import Dict, List, Tuple, Optional
from .types import Subgraph
from .config import GraphToTextConfig


class GraphToTextEncoder:
    def __init__(self, config: GraphToTextConfig, label_map: Optional[Dict[int, str]] = None):
        self.config = config
        self.label_map = label_map if label_map is not None else {}

    def set_label_map(self, label_map: Dict[int, str]):
        self.label_map = label_map

    def _get_label(self, node_id: int) -> str:
        return self.label_map.get(node_id, f"node_{node_id}")

    def _get_relation_between(self, source: int, target: int, edges: List[Tuple[int, int, str]]) -> Optional[str]:
        for s, t, r in edges:
            if s == source and t == target:
                return r
            if s == target and t == source:
                return r
        return None

    def _sort_nodes(self, nodes: List[int], node_activations: Dict[int, float],
                    edge_confidences: Dict[Tuple[int, int, str], float],
                    edges: List[Tuple[int, int, str]]) -> List[int]:
        if self.config.sort_by == "activation":
            key_fn = lambda n: node_activations.get(n, 0.0)
        elif self.config.sort_by == "confidence":
            def confidence_for_node(n):
                confs = []
                for (s, t, r), c in edge_confidences.items():
                    if s == n or t == n:
                        confs.append(c)
                return max(confs) if confs else 0.0
            key_fn = confidence_for_node
        elif self.config.sort_by == "recency":
            key_fn = lambda n: node_activations.get(n, 0.0)
        else:
            key_fn = lambda n: node_activations.get(n, 0.0)

        reverse = self.config.sort_order == "descending"
        return sorted(nodes, key=key_fn, reverse=reverse)

    def encode(self, subgraph: Subgraph) -> str:
        if len(subgraph.nodes) > self.config.max_nodes_in_text:
            nodes = self._truncate_nodes(subgraph)
        else:
            nodes = self._sort_nodes(
                subgraph.nodes, subgraph.node_activations,
                subgraph.edge_confidences, subgraph.edges
            )

        result = []

        for i, node_id in enumerate(nodes):
            activation = subgraph.node_activations.get(node_id, 0.0)
            label = self._get_label(node_id)

            if self.config.include_activations and activation > self.config.activation_threshold:
                node_str = self.config.node_template_active.format(
                    label=label, activation=activation
                )
            else:
                node_str = self.config.node_template_simple.format(label=label)

            if i > 0:
                if self.config.include_edge_types:
                    prev_id = nodes[i - 1]
                    rel = self._get_relation_between(prev_id, node_id, subgraph.edges)
                    if rel:
                        result.append(self.config.edge_format.format(relation=rel))
                    else:
                        result.append(self.config.separator)
                else:
                    result.append(self.config.separator)

            result.append(node_str)

        return "".join(result)

    def _truncate_nodes(self, subgraph: Subgraph) -> List[int]:
        sorted_nodes = self._sort_nodes(
            subgraph.nodes, subgraph.node_activations,
            subgraph.edge_confidences, subgraph.edges
        )
        return sorted_nodes[:self.config.max_nodes_in_text]
