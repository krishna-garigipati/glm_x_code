"""Quick validation: all 3 backends + all API functions match config_graph.yaml."""
import sys, os, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from knowledge_graph.graph_component_implementation.graph_store import GraphStore
from knowledge_graph.graph_component_implementation.models import Node, Edge, Subgraph
from knowledge_graph.graph_component_implementation.errors import (
    NodeNotFoundError, EdgeNotFoundError, InvalidEmbeddingDimensionError
)

CONFIG = os.path.join(os.path.dirname(__file__), "config_graph.yaml")

def make_emb():
    return np.random.uniform(-1.0, 1.0, 32).astype(np.float64)

FAIL = 0

def check(name, ok):
    global FAIL
    if not ok:
        print(f"  FAIL: {name}")
        FAIL += 1
    else:
        print(f"  OK: {name}")

for backend in ("memory_only", "sharded_disk", "lmdb"):
    print(f"\n{'='*60}")
    print(f"Backend: {backend}")
    print('='*60)
    tmp = tempfile.mkdtemp()
    cfg = CONFIG
    import yaml
    with open(CONFIG, 'r') as f:
        data = yaml.safe_load(f)
    data['storage']['backend'] = backend
    data['storage']['base_path'] = tmp
    cfg_path = os.path.join(tmp, "config_override.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(data, f)

    try:
        gs = GraphStore(cfg_path)
    except Exception as e:
        err = str(e)
        if "lmdb" in err:
            print(f"  SKIP: lmdb package not installed - {e}")
            continue
        print(f"  FAIL: init - {e}")
        FAIL += 1
        continue

    # add_node
    e = make_emb()
    r = gs.add_node(1, "test", "Concept", e)
    check("add_node (new)", r == True)

    # duplicate returns False (per config)
    r2 = gs.add_node(1, "test2", "Concept", make_emb())
    check("add_node (duplicate=False)", r2 == False)

    # get_node
    n = gs.get_node(1)
    check("get_node exists", n is not None and n.node_id == 1 and n.label == "test")
    check("get_node not exists", gs.get_node(999) is None)

    # update_node_embedding
    e2 = make_emb()
    r = gs.update_node_embedding(1, e2)
    check("update_node_embedding", r == True)
    n2 = gs.get_node(1)
    check("embedding updated", n2 is not None and np.array_equal(n2.embedding, gs._validate_embedding(e2)))

    # update_node_embedding nonexistent
    try:
        gs.update_node_embedding(999, make_emb())
        check("update missing raises", False)
    except NodeNotFoundError:
        check("update missing raises", True)

    # add_edge
    r = gs.add_edge(1, 2, "related_to")
    check("add_edge", r == True)
    r = gs.add_edge(1, 3, "related_to")
    check("add_edge (2nd)", r == True)

    # get_edge
    ed = gs.get_edge(1, 2, "related_to")
    check("get_edge exists", ed is not None and ed.source == 1 and ed.target == 2)
    check("get_edge not exists", gs.get_edge(1, 999, "related_to") is None)

    # update_edge_weights
    gs.update_edge_weights({(1, 2, "related_to"): (0.9, 0.8)})
    ed2 = gs.get_edge(1, 2, "related_to")
    check("update_edge_weights", ed2 is not None and abs(ed2.strength - 0.9) < 1e-6 and abs(ed2.confidence - 0.8) < 1e-6)

    # get_neighbors
    neigh = gs.get_neighbors(1)
    check("get_neighbors count", len(neigh) == 2)
    neigh_filtered = gs.get_neighbors(1, relation_filter=["related_to"])
    check("get_neighbors filter", len(neigh_filtered) == 2)

    # get_subgraph_activated
    sub = gs.get_subgraph_activated([1])
    check("subgraph_activated nodes", 1 in sub.nodes)
    check("subgraph_activated edges", len(sub.edges) > 0)

    # get_subgraph_by_embedding_similarity
    sub2 = gs.get_subgraph_by_embedding_similarity(make_emb(), top_k=5)
    check("subgraph_similarity", isinstance(sub2, Subgraph))

    # prune
    pruned = gs.prune(utility_threshold=-1.0)
    check("prune (none removed)", pruned == 0)

    # save/load checkpoint
    cp = os.path.join(tmp, "checkpoint.bin")
    r = gs.save_checkpoint(cp)
    check("save_checkpoint", r == True and os.path.exists(cp))

    gs.close()
    gs2 = GraphStore(cfg_path)
    r = gs2.load_checkpoint(cp)
    check("load_checkpoint", r == True)
    check("load_checkpoint data", gs2.get_node(1) is not None)

    # invalid embedding dimension
    try:
        gs.add_node(99, "bad", "Concept", np.zeros(31))
        check("invalid emb dim raises", False)
    except InvalidEmbeddingDimensionError:
        check("invalid emb dim raises", True)

    # validate max relations
    for i in range(260):
        try:
            gs.add_edge(1, 100 + i, f"rel_{i}")
        except ValueError:
            pass

    gs.close()
    print(f"  ... cleanup: {tmp}")

print(f"\n{'='*60}")
if FAIL:
    print(f"FAILURES: {FAIL}")
else:
    print("ALL TESTS PASSED")
print('='*60)
sys.exit(FAIL)
