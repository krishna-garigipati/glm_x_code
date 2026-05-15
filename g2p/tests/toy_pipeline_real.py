"""
Toy dataset pipeline runner using REAL Sentence-BERT model.
"""
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from G2P.types import Subgraph, Plan
from G2P.config import G2PConfig, RuleDefinition
from G2P.g2p_planner import G2PPlanner


def make_subgraph(
    node_ids: List[int],
    edges: List[Tuple[int, int, str]],
    seed_nodes: Optional[List[int]] = None,
    tier: int = 1,
    activation_energy: float = 1.0,
    node_activations: Optional[Dict[int, float]] = None,
) -> Subgraph:
    rng = np.random.RandomState(42)
    if node_activations is None:
        node_activations = {n: round(float(rng.uniform(0.2, 1.0)), 4) for n in node_ids}
    edge_strengths = {k: round(float(rng.uniform(0.3, 1.0)), 4) for k in edges}
    edge_confidences = {k: round(float(rng.uniform(0.3, 1.0)), 4) for k in edges}
    qe = rng.randn(384).astype(np.float32)
    qe = qe / np.linalg.norm(qe)
    if seed_nodes is None:
        seed_nodes = [node_ids[0]] if node_ids else []
    return Subgraph(
        nodes=node_ids,
        node_activations=node_activations,
        edges=edges,
        edge_strengths=edge_strengths,
        edge_confidences=edge_confidences,
        seed_nodes=seed_nodes,
        tier_used=tier,
        activation_energy=activation_energy,
        query_embedding=qe,
        timestamp=1000000.0,
    )


toy_dataset = [
    {
        "name": "tiny_definition",
        "description": "2 nodes, len<=3 triggers heuristic: define only",
        "label_map": {0: "neuron", 1: "brain_cell"},
        "subgraph": make_subgraph(node_ids=[0, 1], edges=[(0, 1, "is_a")]),
        "expected_heuristic": True,
    },
    {
        "name": "causal_chain",
        "description": "has_edge_type('causes') triggers heuristic",
        "label_map": {0: "smoking", 1: "cancer", 2: "death"},
        "subgraph": make_subgraph(node_ids=[0, 1, 2], edges=[(0, 1, "causes"), (1, 2, "causes")]),
        "expected_heuristic": True,
    },
    {
        "name": "large_graph",
        "description": "13 nodes >= 10 triggers heuristic: list, summarize, conclude",
        "label_map": {i: f"step_{i}" for i in range(13)},
        "subgraph": make_subgraph(node_ids=list(range(13)), edges=[(i, i+1, "follows") for i in range(12)]),
        "expected_heuristic": True,
    },
    {
        "name": "diverse_relations_no_causes",
        "description": "4 nodes with no triggering relation type -> FFN path",
        "label_map": {0: "apple", 1: "fruit", 2: "red", 3: "sweet"},
        "subgraph": make_subgraph(
            node_ids=[0, 1, 2, 3],
            edges=[(0, 1, "is_a"), (1, 2, "has_property"), (2, 3, "example_of")],
        ),
        "expected_heuristic": False,
    },
    {
        "name": "no_edges_single_node",
        "description": "1 node only, len<=3 triggers define",
        "label_map": {0: "singularity"},
        "subgraph": make_subgraph(node_ids=[0], edges=[]),
        "expected_heuristic": True,
    },
]


def run_pipeline(dataset: list) -> list:
    cfg = G2PConfig()
    cfg.mapping.heuristic_rules_enabled = True
    cfg.mapping.rule_definitions = [
        RuleDefinition(condition="has_edge_type('causes')", plan=[1, 2, 7]),
        RuleDefinition(condition="has_edge_type('contradicts')", plan=[4, 1, 7]),
        RuleDefinition(condition="len(subgraph.nodes) >= 10", plan=[6, 12, 8]),
        RuleDefinition(condition="average_confidence < 0.5", plan=[10, 11]),
        RuleDefinition(condition="len(subgraph.nodes) <= 3", plan=[0]),
    ]

    results = []
    for sample in dataset:
        name = sample["name"]
        subgraph = sample["subgraph"]
        label_map = sample.get("label_map", {})

        planner = G2PPlanner(cfg, label_map=label_map)
        planner.initialize()
        plan = planner.plan(subgraph)
        text = planner.graph_to_text_encoder.encode(subgraph)

        results.append({
            "test_name": name,
            "description": sample["description"],
            "input": {
                "num_nodes": len(subgraph.nodes),
                "num_edges": len(subgraph.edges),
                "relation_types": sorted(set(r for _, _, r in subgraph.edges)),
                "graph_text": text,
                "label_map": {str(k): v for k, v in label_map.items()},
            },
            "output": {
                "intent_sequence": plan.intent_sequence,
                "intent_names": plan.intent_names,
                "plan_confidence": round(plan.plan_confidence, 4),
                "heuristic_fallback_used": plan.heuristic_fallback_used,
            },
            "assertions": {
                "expected_heuristic": sample["expected_heuristic"],
                "heuristic_matches_expectation": plan.heuristic_fallback_used == sample["expected_heuristic"],
            },
        })
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("G2P Planner - Toy Pipeline (REAL Sentence-BERT)")
    print("=" * 70)

    results = run_pipeline(toy_dataset)

    print(f"\n{'Test':<35} {'Heuristic?':<12} {'Intents':<40} {'Match?':<8}")
    print("-" * 95)
    for r in results:
        heur = "YES" if r["output"]["heuristic_fallback_used"] else "no"
        names = r["output"]["intent_names"] or []
        intent_str = ", ".join(names[:6])
        if len(names) > 6:
            intent_str += "..."
        match = "[OK]" if r["assertions"]["heuristic_matches_expectation"] else "[FAIL]"
        print(f"{r['test_name']:<35} {heur:<12} {intent_str:<40} {match:<8}")

    output_path = Path(__file__).resolve().parent / "toy_pipeline_output_real.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    total = len(results)
    passed = sum(1 for r in results if r["assertions"]["heuristic_matches_expectation"])
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} assertions passed")
    print(f"JSON: {output_path}")
    print(f"{'=' * 70}")
