import sys, os, time, tempfile, shutil, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from graph.graph_component_implementation.graph_store import GraphStore


def _make_gs():
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
    gs = GraphStore(config_path=cfg_path)
    return gs, tmp


def test_scale_insertion():
    """Insert 5000 nodes + 50 edges, measure time."""
    gs, tmp = _make_gs()
    n_nodes = 5_000
    start = time.perf_counter()
    for i in range(n_nodes):
        emb = np.random.uniform(-1, 1, 32).astype(np.float32)
        gs.add_node(i, f"Node{i}", "Concept", emb)
        if i % 100 == 0 and i > 0:
            gs.add_edge(i - 1, i, "related", 0.6)
    duration = time.perf_counter() - start
    rate = n_nodes / duration
    print(f"[METRIC] ScaleInsertTime: {duration:.2f}s ({rate:.0f} nodes/sec)")
    gs.close()
    shutil.rmtree(tmp, ignore_errors=True)
    assert duration < 120


def test_get_neighbors_speed():
    """Measure get_neighbors() latency over 100 calls."""
    gs, tmp = _make_gs()
    for i in range(100):
        emb = np.random.uniform(-1, 1, 32).astype(np.float32)
        gs.add_node(i, f"N{i}", "Concept", emb)
    for i in range(99):
        gs.add_edge(i, i + 1, "related", 0.6)

    start = time.perf_counter()
    for _ in range(100):
        gs.get_neighbors(50)
    elapsed = (time.perf_counter() - start) / 100
    print(f"[METRIC] GetNeighborsTime: {elapsed*1000:.2f}ms")
    gs.close()
    shutil.rmtree(tmp, ignore_errors=True)
    assert elapsed < 0.05


def test_similarity_search_speed():
    """Measure similarity search latency over existing nodes."""
    gs, tmp = _make_gs()
    for i in range(200):
        emb = np.random.uniform(-1, 1, 32).astype(np.float32)
        gs.add_node(i, f"N{i}", "Concept", emb, activation=0.5)
    query = np.random.uniform(-1, 1, 32).astype(np.float32)

    start = time.perf_counter()
    for _ in range(3):
        gs.get_subgraph_by_embedding_similarity(query, top_k=50)
    elapsed = (time.perf_counter() - start) / 3
    print(f"[METRIC] SimilaritySearchTime: {elapsed*1000:.2f}ms")
    gs.close()
    shutil.rmtree(tmp, ignore_errors=True)
    assert elapsed < 10


def test_memory_usage():
    """Measure peak memory during 5000-node insertion."""
    gs, tmp = _make_gs()
    import tracemalloc
    tracemalloc.start()
    for i in range(5_000):
        gs.add_node(i, f"MemNode{i}", "Concept", np.random.rand(32).astype(np.float32))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gs.close()
    print(f"[METRIC] PeakMemoryMB: {peak / 1024 / 1024:.2f} MB")
    assert peak < 300 * 1024 * 1024
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_scale_insertion()
    test_get_neighbors_speed()
    test_similarity_search_speed()
    test_memory_usage()
    print("All performance tests done.")
