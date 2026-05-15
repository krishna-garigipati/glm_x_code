import sys, os, tempfile, shutil, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from graph.tests.conftest import graph

try:
    from hypothesis import given, strategies as st, settings, HealthCheck
    import hypothesis.extra.numpy as npst
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


def _fresh_graph():
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
    from graph.graph_component_implementation.graph_store import GraphStore
    gs = GraphStore(config_path=cfg_path)
    return gs, tmp


def test_node_roundtrip_property_manual(graph):
    for nid, label, act in [(1, "Alpha", 0.95), (2, "Beta", 0.5), (3, "Gamma", 0.01)]:
        emb = np.random.uniform(-1, 1, 32).astype(np.float32)
        assert graph.add_node(nid, label, "Concept", emb, act) is True
        n = graph.get_node(nid)
        assert n is not None
        assert n.node_id == nid
        assert n.label == label
        assert abs(n.activation - act) < 0.001


def test_edge_property_manual(graph):
    for src, tgt, s, c in [(1, 2, 0.9, 0.8), (2, 3, 0.5, 0.5), (1, 3, 0.1, 0.99)]:
        emb = np.zeros(32, dtype=np.float32)
        graph.add_node(src, "Src", "Concept", emb)
        graph.add_node(tgt, "Tgt", "Concept", emb)
        graph.add_edge(src, tgt, "test_rel", s, c)
        e = graph.get_edge(src, tgt, "test_rel")
        assert e is not None
        assert abs(e.strength - s) < 1e-5


if HAS_HYPOTHESIS:

    @given(
        label=st.text(min_size=1, max_size=50),
        activation=st.floats(0.01, 1.0),
        emb=npst.arrays(
            np.float32, 32,
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False)
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_node_roundtrip_hypothesis(label, activation, emb):
        gs, tmp = _fresh_graph()
        node_id = 1001
        success = gs.add_node(node_id, label, "Concept", emb, activation)
        assert success is True
        retrieved = gs.get_node(node_id)
        assert retrieved is not None
        assert retrieved.node_id == node_id
        assert retrieved.label == label
        assert abs(retrieved.activation - activation) < 0.001
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    @given(
        strength=st.floats(0.0, 1.0),
        confidence=st.floats(0.0, 1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_edge_hypothesis(strength, confidence):
        gs, tmp = _fresh_graph()
        emb = np.zeros(32, dtype=np.float32)
        gs.add_node(1, "Src", "Concept", emb)
        gs.add_node(2, "Tgt", "Concept", emb)
        gs.add_edge(1, 2, "test_rel", strength, confidence)
        edge = gs.get_edge(1, 2, "test_rel")
        assert edge is not None
        assert abs(edge.strength - strength) < 1e-5
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
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
    from graph.graph_component_implementation.graph_store import GraphStore
    g = GraphStore(config_path=cfg_path)
    test_node_roundtrip_property_manual(g)
    test_edge_property_manual(g)
    if HAS_HYPOTHESIS:
        test_node_roundtrip_hypothesis("PropTest", 0.75, np.random.uniform(-1, 1, 32).astype(np.float32))
        test_edge_hypothesis(0.85, 0.9)
    g.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("All property tests passed.")
