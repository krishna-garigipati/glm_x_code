import inspect, yaml, os, tempfile, shutil
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge_graph.graph_component_implementation.graph_store import GraphStore
from knowledge_graph.graph_component_implementation.models import Node, Edge, Subgraph

CONFIG_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "graph_component_implementation", "config_graph.yaml")


def _make_gs():
    tmp = tempfile.mkdtemp()
    with open(CONFIG_SRC, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["storage"]["base_path"] = tmp
    cfg["backup"]["enabled"] = False
    cfg_path = os.path.join(tmp, "cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    gs = GraphStore(config_path=cfg_path)
    return gs, tmp


def test_class_names():
    gs, tmp = _make_gs()
    assert gs.__class__.__name__ == "GraphStore"
    gs.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_required_methods_exist():
    required = [
        "add_node", "get_node", "update_node_embedding",
        "add_edge", "get_edge", "update_edge_weights",
        "get_neighbors", "get_subgraph_activated",
        "get_subgraph_by_embedding_similarity", "prune",
        "save_checkpoint", "load_checkpoint",
    ]
    gs, tmp = _make_gs()
    for method in required:
        assert hasattr(gs, method), f"Missing method: {method}"
        assert callable(getattr(gs, method))
    gs.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_method_signatures():
    gs, tmp = _make_gs()
    signatures = {
        "add_node": ["node_id", "label", "node_type", "embedding", "activation"],
        "get_node": ["node_id"],
        "update_node_embedding": ["node_id", "embedding"],
        "add_edge": ["source", "target", "relation", "strength", "confidence"],
        "get_edge": ["source", "target", "relation"],
        "update_edge_weights": ["updates"],
        "get_neighbors": ["node_id", "relation_filter"],
        "get_subgraph_activated": ["seed_nodes", "max_nodes"],
        "get_subgraph_by_embedding_similarity": ["query_embedding", "top_k"],
        "prune": ["utility_threshold"],
        "save_checkpoint": ["filepath"],
        "load_checkpoint": ["filepath"],
    }
    for method_name, expected_params in signatures.items():
        sig = inspect.signature(getattr(gs, method_name))
        actual_params = list(sig.parameters.keys())
        for param in expected_params:
            assert param in actual_params, f"{method_name} missing parameter: {param}"
    gs.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_dataclass_models_exist():
    assert Node is not None
    assert Edge is not None
    assert Subgraph is not None


if __name__ == "__main__":
    test_class_names()
    test_required_methods_exist()
    test_method_signatures()
    test_dataclass_models_exist()
    print("All contract tests passed.")
