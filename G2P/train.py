import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from .types import Subgraph
from .g2p_planner import G2PPlanner
from .config import G2PConfig

logger = logging.getLogger(__name__)


def generate_synthetic_data(config: G2PConfig) -> List[Tuple[Subgraph, List[int]]]:
    rng = np.random.RandomState(config.training.synthetic_data.seed)
    data: List[Tuple[Subgraph, List[int]]] = []

    for _ in range(config.training.synthetic_data.num_samples):
        num_nodes = rng.randint(
            config.training.synthetic_data.min_nodes_per_graph,
            config.training.synthetic_data.max_nodes_per_graph + 1,
        )

        node_ids = list(range(num_nodes))
        node_activations = {n: float(rng.uniform(0.01, 1.0)) for n in node_ids}

        edges = []
        edge_strengths = {}
        edge_confidences = {}

        for i in range(num_nodes - 1):
            src = node_ids[i]
            tgt = node_ids[i + 1]
            rel_idx = int(rng.randint(0, 16))
            rel_types = [
                "is_a", "has_property", "causes", "caused_by", "follows",
                "precedes", "contradicts", "supports", "associated_with",
                "example_of", "part_of", "synonym", "antonym",
                "temporal_coincident", "spatial_near", "linguistic_maps",
            ]
            rel = rel_types[rel_idx]
            edge_key = (src, tgt, rel)
            edges.append(edge_key)
            edge_strengths[edge_key] = float(rng.uniform(0.0, 1.0))
            edge_confidences[edge_key] = float(rng.uniform(0.0, 1.0))

        seed_nodes = [node_ids[0]]
        tier_used = int(rng.choice([1, 2]))
        activation_energy = float(rng.uniform(0.1, 5.0))
        query_embedding = rng.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        timestamp = float(rng.uniform(1e9, 1.7e9))

        subgraph = Subgraph(
            nodes=node_ids,
            node_activations=node_activations,
            edges=edges,
            edge_strengths=edge_strengths,
            edge_confidences=edge_confidences,
            seed_nodes=seed_nodes,
            tier_used=tier_used,
            activation_energy=activation_energy,
            query_embedding=query_embedding,
            timestamp=timestamp,
        )

        num_intents = int(rng.randint(1, 5))
        intent_seq = [int(rng.randint(0, 15)) for _ in range(num_intents)]

        data.append((subgraph, intent_seq))

    logger.info(f"Generated {len(data)} synthetic training samples")
    return data


def train_g2p(
    config_path: str,
    training_data: Optional[List[Tuple[Subgraph, List[int]]]] = None,
    validation_data: Optional[List[Tuple[Subgraph, List[int]]]] = None,
    label_map: Optional[Dict[int, str]] = None,
) -> Dict:
    config = G2PConfig.from_yaml(config_path)
    planner = G2PPlanner(config, label_map=label_map)
    planner.initialize()

    if training_data is None:
        training_data = generate_synthetic_data(config)

    return planner.train(training_data, validation_data)
