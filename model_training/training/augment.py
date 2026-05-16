import numpy as np
import logging
from typing import List, Tuple
from g2p.types import Subgraph

logger = logging.getLogger(__name__)


def augment_subgraph(sg: Subgraph, intent_seq: List[int]) -> List[Tuple[Subgraph, List[int]]]:
    variants = []
    rng = np.random.RandomState(42)

    edges_kept = [
        (s, t, r) for (s, t, r) in sg.edges
        if sg.edge_confidences.get((s, t, r), 0) > 0.3
    ]
    if edges_kept and len(edges_kept) >= 1:
        kept_edges = edges_kept
        kept_strengths = {k: sg.edge_strengths[k] for k in kept_edges if k in sg.edge_strengths}
        kept_confidences = {k: sg.edge_confidences[k] for k in kept_edges if k in sg.edge_confidences}
        var = Subgraph(
            nodes=sg.nodes,
            node_activations=sg.node_activations,
            edges=kept_edges,
            edge_strengths=kept_strengths,
            edge_confidences=kept_confidences,
            seed_nodes=sg.seed_nodes,
            tier_used=sg.tier_used,
            activation_energy=sg.activation_energy,
            query_embedding=sg.query_embedding.copy(),
            timestamp=sg.timestamp,
        )
        variants.append((var, intent_seq))

    noisy_activations = {
        n: float(np.clip(v * rng.uniform(0.9, 1.1), 0.01, 1.0))
        for n, v in sg.node_activations.items()
    }
    var2 = Subgraph(
        nodes=sg.nodes,
        node_activations=noisy_activations,
        edges=sg.edges,
        edge_strengths=sg.edge_strengths,
        edge_confidences=sg.edge_confidences,
        seed_nodes=sg.seed_nodes,
        tier_used=sg.tier_used,
        activation_energy=sg.activation_energy,
        query_embedding=sg.query_embedding.copy(),
        timestamp=sg.timestamp,
    )
    variants.append((var2, intent_seq))

    return variants


def augment_dataset(
    data: List[Tuple[Subgraph, List[int], str]],
) -> List[Tuple[Subgraph, List[int], str]]:
    augmented = []
    for sg, intent_seq, seed_name in data:
        augmented.append((sg, intent_seq, seed_name))
        variants = augment_subgraph(sg, intent_seq)
        for var_sg, var_seq in variants:
            augmented.append((var_sg, var_seq, seed_name))
    logger.info(f"Augmented {len(data)} -> {len(augmented)} samples")
    return augmented
