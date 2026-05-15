import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from graph.graph_component_implementation.models import Node, Edge
from graph.tests.conftest import graph


def test_add_get_node(graph):
    emb = np.random.uniform(-1.0, 1.0, 32).astype(np.float32)
    assert graph.add_node(1, "TestNode", "Concept", emb, 0.75) is True

    node = graph.get_node(1)
    assert node is not None
    assert node.node_id == 1
    assert node.label == "TestNode"
    assert node.node_type == "Concept"
    assert node.embedding.shape == (32,)


def test_embedding_quantization(graph):
    original = np.array([0.95, -0.85, 0.0, 0.5], dtype=np.float32)
    emb = np.zeros(32, dtype=np.float32)
    emb[:4] = original
    graph.add_node(999, "Quant", "Concept", emb)

    retrieved = graph.get_node(999)
    restored = retrieved.embedding.astype(np.float32) / 127.0
    assert np.allclose(original, restored[:4], atol=0.02)


def test_edge_operations(graph):
    graph.add_node(10, "A", "Concept", np.zeros(32, dtype=np.float32))
    graph.add_node(20, "B", "Concept", np.zeros(32, dtype=np.float32))

    assert graph.add_edge(10, 20, "related", 0.85, 0.9) is True
    edge = graph.get_edge(10, 20, "related")
    assert edge is not None
    assert abs(edge.strength - 0.85) < 1e-5


def test_duplicate_prevention(graph):
    emb = np.zeros(32, dtype=np.float32)
    assert graph.add_node(500, "Dup", "Entity", emb) is True
    assert graph.add_node(500, "Dup2", "Entity", emb) is False


def test_invalid_embedding_dimension(graph):
    import pytest
    from graph.graph_component_implementation.errors import InvalidEmbeddingDimensionError
    with pytest.raises(InvalidEmbeddingDimensionError):
        graph.add_node(99, "bad", "Concept", np.zeros(31))


def test_errors_importable():
    from graph.graph_component_implementation.errors import (
        GraphStoreError, NodeNotFoundError, EdgeNotFoundError,
        DuplicateNodeError, InvalidEmbeddingDimensionError,
        SerializationFailedError, ShardCorruptedError,
    )
    assert GraphStoreError is not None
    assert NodeNotFoundError is not None


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
    test_add_get_node(g)
    test_embedding_quantization(g)
    test_edge_operations(g)
    test_duplicate_prevention(g)
    test_invalid_embedding_dimension(g)
    test_errors_importable()
    g.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("All unit tests passed.")
