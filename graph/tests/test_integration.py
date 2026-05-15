import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from graph.graph_component_implementation.models import Subgraph
from graph.tests.conftest import graph


def test_full_workflow(graph):
    for i in range(30):
        emb = np.random.uniform(-1, 1, 32).astype(np.float32)
        graph.add_node(i, f"Node{i}", "Concept", emb, activation=0.8 if i < 15 else 0.1)
    for i in range(25):
        graph.add_edge(i, (i + 1) % 30, "connected", strength=0.7)

    subgraph = graph.get_subgraph_activated(seed_nodes=[0], max_nodes=12)
    assert len(subgraph.nodes) >= 5
    assert 0 in subgraph.nodes

    query = np.zeros(32, dtype=np.float32)
    query[5] = 0.9
    sim_subgraph = graph.get_subgraph_by_embedding_similarity(query, top_k=8)
    assert len(sim_subgraph.nodes) == 8


def test_neighbors_and_edges_integration(graph):
    for i in range(5):
        graph.add_node(i, f"N{i}", "Concept", np.random.rand(32).astype(np.float32))
    graph.add_edge(0, 1, "causes")
    graph.add_edge(0, 2, "related")
    graph.add_edge(0, 3, "causes")

    neighbors = graph.get_neighbors(0)
    assert len(neighbors) == 3

    causes_only = graph.get_neighbors(0, relation_filter=["causes"])
    assert len(causes_only) == 2


def test_checkpoint_roundtrip(graph, tmp_path):
    import yaml, tempfile, shutil
    for i in range(10):
        emb = np.random.rand(32).astype(np.float32)
        graph.add_node(i, f"N{i}", "Concept", emb)
    graph.add_edge(0, 1, "test_rel")

    cp = os.path.join(tempfile.mkdtemp(), "cp.bin")
    assert graph.save_checkpoint(cp) is True
    assert os.path.exists(cp)

    cfg_src = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "graph_component_implementation", "config_graph.yaml")
    with open(cfg_src) as f:
        cfg = yaml.safe_load(f)
    tmp2 = tempfile.mkdtemp()
    cfg["storage"]["base_path"] = tmp2
    cfg["backup"]["enabled"] = False
    cp_cfg = os.path.join(tmp2, "cfg.yaml")
    with open(cp_cfg, "w") as f:
        yaml.dump(cfg, f)
    from graph.graph_component_implementation.graph_store import GraphStore
    gs2 = GraphStore(config_path=cp_cfg)
    assert gs2.load_checkpoint(cp) is True
    assert gs2.get_node(0) is not None
    assert gs2.get_edge(0, 1, "test_rel") is not None
    gs2.close()
    shutil.rmtree(tmp2, ignore_errors=True)


if __name__ == "__main__":
    import tempfile, shutil, yaml
    from graph.graph_component_implementation.graph_store import GraphStore
    tmp = tempfile.mkdtemp()
    cfg_src = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "graph_component_implementation", "config_graph.yaml")
    with open(cfg_src) as f:
        cfg = yaml.safe_load(f)
    cfg["storage"]["base_path"] = tmp
    cfg["backup"]["enabled"] = False
    cfg_path = os.path.join(tmp, "cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    g = GraphStore(config_path=cfg_path)
    test_full_workflow(g)
    test_neighbors_and_edges_integration(g)
    test_checkpoint_roundtrip(g, None)
    g.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("All integration tests passed.")
