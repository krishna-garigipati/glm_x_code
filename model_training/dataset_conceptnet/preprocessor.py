import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

from g2p.types import Subgraph
from .relation_map import CONCEPTNET_RELATION_MAP

logger = logging.getLogger(__name__)


def build_ego_graph(
    seed_concept_id: int,
    edges: List[dict],
    max_hops: int = 2,
    max_nodes: int = 30,
) -> Tuple[Set[int], List[dict], List[int]]:
    adjacency = defaultdict(list)
    for e in edges:
        adjacency[e["head_id"]].append((e["tail_id"], e["relation"]))
        adjacency[e["tail_id"]].append((e["head_id"], e["relation"]))

    visited_nodes = set()
    visited_edges = []
    intent_sequence = []
    queue = [(seed_concept_id, 0)]
    while queue and len(visited_nodes) < max_nodes:
        node, depth = queue.pop(0)
        if node in visited_nodes or depth > max_hops:
            continue
        visited_nodes.add(node)
        for neighbor, relation in adjacency.get(node, []):
            if neighbor not in visited_nodes:
                intent_id = CONCEPTNET_RELATION_MAP.get(relation)
                if intent_id is not None:
                    visited_edges.append({
                        "source": node,
                        "target": neighbor,
                        "relation": relation,
                        "intent_id": intent_id,
                    })
                    intent_sequence.append(intent_id)
                if len(visited_nodes) < max_nodes:
                    queue.append((neighbor, depth + 1))
    return visited_nodes, visited_edges, intent_sequence


def subgraph_from_ego(
    seed_concept_id: int,
    nodes_set: Set[int],
    edges_list: List[dict],
    concept_to_id: Dict[str, int],
    id_to_concept: Dict[int, str],
    seed_concept_name: str,
) -> Subgraph:
    node_list = sorted(nodes_set)
    node_activations = {n: 0.5 for n in node_list}
    subgraph_edges = []
    edge_strengths = {}
    edge_confidences = {}
    for e in edges_list:
        key = (e["source"], e["target"], e["relation"])
        subgraph_edges.append(key)
        edge_strengths[key] = 0.8
        edge_confidences[key] = 0.7

    query_embedding = np.random.randn(384).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    return Subgraph(
        nodes=node_list,
        node_activations=node_activations,
        edges=subgraph_edges,
        edge_strengths=edge_strengths,
        edge_confidences=edge_confidences,
        seed_nodes=[seed_concept_id],
        tier_used=1,
        activation_energy=1.0,
        query_embedding=query_embedding,
        timestamp=float(np.random.uniform(1e9, 1.7e9)),
    )


def generate_training_data(
    edges: List[dict],
    concept_to_id: Dict[str, int],
    id_to_concept: Dict[int, str],
    seed_limit: Optional[int] = None,
    max_hops: int = 2,
    min_nodes: int = 2,
) -> List[Tuple[Subgraph, List[int], str]]:
    seed_ids = list(concept_to_id.values())
    if seed_limit is not None and seed_limit < len(seed_ids):
        rng = np.random.RandomState(42)
        seed_ids = list(rng.choice(seed_ids, size=seed_limit, replace=False))

    samples = []
    rejected = 0
    for sid in seed_ids:
        seed_name = id_to_concept[sid]
        nodes_set, edges_list, intent_seq = build_ego_graph(
            sid, edges, max_hops=max_hops
        )
        if len(nodes_set) < min_nodes or not intent_seq:
            rejected += 1
            continue
        subgraph = subgraph_from_ego(
            sid, nodes_set, edges_list,
            concept_to_id, id_to_concept, seed_name,
        )
        samples.append((subgraph, intent_seq, seed_name))
    logger.info(
        f"Generated {len(samples)} training samples "
        f"({rejected} rejected, min {min_nodes} nodes + non-empty intents)"
    )
    return samples
