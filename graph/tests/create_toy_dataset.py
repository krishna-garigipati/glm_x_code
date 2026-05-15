import sys, os, yaml, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from graph.graph_component_implementation.graph_store import GraphStore


def create_toy_dataset(graph, populate=True):
    if not populate:
        return
    print("Creating Toy Dataset...")

    nodes_data = [
        (1, "Artificial Intelligence", "Concept", 0.95, 42),
        (2, "Machine Learning", "Concept", 0.88, 38),
        (3, "Neural Network", "Concept", 0.82, 35),
        (4, "Transformer", "Concept", 0.91, 29),
        (10, "Elon Musk", "Entity", 0.75, 15),
        (11, "xAI", "Entity", 0.85, 22),
        (12, "Grok", "Entity", 0.92, 31),
        (20, "2025-12-01", "TemporalAnchor", 0.60, 8),
        (21, "Morning", "TemporalAnchor", 0.45, 12),
        (30, "quantum", "LinguisticToken", 0.55, 18),
        (31, "resonance", "LinguisticToken", 0.68, 14),
        (32, "spreading activation", "LinguisticToken", 0.72, 9),
        (40, "High Energy Physics", "ContextLabel", 0.65, 7),
        (41, "Cognitive Architecture", "ContextLabel", 0.80, 19),
        (50, "Pattern: Analogy Making", "Pattern", 0.78, 25),
        (51, "Pattern: Causal Reasoning", "Pattern", 0.71, 16),
    ]

    for node_id, label, node_type, activation, use_count in nodes_data:
        np.random.seed(node_id)
        emb = np.random.uniform(-0.9, 0.9, 32).astype(np.float32)
        if "AI" in label or "Neural" in label or "Transformer" in label:
            emb[0] = 0.85; emb[1] = 0.78
        if "Grok" in label or "xAI" in label:
            emb[0] = 0.92; emb[2] = 0.88
        success = graph.add_node(node_id=node_id, label=label, node_type=node_type,
                                 embedding=emb, activation=activation)
        if not success:
            print(f"Failed to add node {node_id}")

    edges_data = [
        (1, 2, "is_a", 0.95, 0.98), (2, 3, "is_a", 0.88, 0.92),
        (3, 4, "uses", 0.85, 0.90), (1, 4, "enables", 0.82, 0.85),
        (11, 12, "created", 0.97, 0.99), (10, 11, "founded", 0.93, 0.95),
        (10, 12, "inspired", 0.75, 0.80), (1, 50, "exhibits", 0.78, 0.82),
        (4, 51, "enables", 0.81, 0.87), (30, 1, "describes", 0.65, 0.70),
        (31, 1, "related_to", 0.72, 0.75), (20, 12, "launched_at", 0.60, 0.65),
        (40, 1, "contains", 0.55, 0.60), (50, 50, "self_reinforces", 0.40, 0.50),
    ]

    for src, tgt, rel, strength, conf in edges_data:
        graph.add_edge(source=src, target=tgt, relation=rel,
                       strength=strength, confidence=conf)
    print(f"Toy dataset created with {len(nodes_data)} nodes and {len(edges_data)} edges.")


def build_toy_store():
    tmp = tempfile.mkdtemp()
    cfg_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "graph_component_implementation", "config_graph.yaml")
    with open(cfg_src, "r") as f:
        cfg_data = yaml.safe_load(f)
    cfg_data["storage"]["base_path"] = tmp
    cfg_data["backup"]["enabled"] = False
    cfg_path = os.path.join(tmp, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_data, f)
    gs = GraphStore(config_path=cfg_path)
    create_toy_dataset(gs)
    return gs, tmp


def main():
    graph, tmp = build_toy_store()
    print(f"\n--- Validation ---")
    sub = graph.get_subgraph_activated([1], 100)
    print(f"Total nodes: {len(sub.nodes)}")
    print(f"Neighbors of AI (1): {len(graph.get_neighbors(1))}")
    print(f"Grok node exists: {graph.get_node(12) is not None}")
    graph.close()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
