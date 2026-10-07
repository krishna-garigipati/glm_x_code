"""oh18: why does 'What is the tail a part of?' walk tail -> animal?

Uses get_all_edges() (57 rows) rather than get_neighbors(), which is far
slower on this store.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

store = SQLiteGraphStore.load_state(str(Path(__file__).resolve().parent / "toy_graph.db"))
nodes = {n.id: n.label for n in store.get_all_nodes()}
print("nodes", len(nodes), "edges", store.get_edge_count())
print()

edges = store.get_all_edges()

for label in ["tail", "dog", "mammal", "animal", "tree", "plant", "oak"]:
    nid = next((i for i, l in nodes.items() if l == label), None)
    print("{} (id {})".format(label, nid))
    if nid is None:
        continue
    for e in edges:
        if e.source == nid:
            print("   OUT  {:>12} -{:>18}-> {}".format(
                nodes.get(e.source), e.relation, nodes.get(e.target)))
        elif e.target == nid:
            print("   IN   {:>12} -{:>18}-> {:<12} (via mirrored copy)".format(
                nodes.get(e.source), e.relation, nodes.get(e.target)))
    print()

print("=== every edge touching tail or dog ===")
tid = next(i for i, l in nodes.items() if l == "tail")
did = next(i for i, l in nodes.items() if l == "dog")
for e in edges:
    if tid in (e.source, e.target) or did in (e.source, e.target):
        print("   {} -{}-> {}".format(nodes.get(e.source), e.relation, nodes.get(e.target)))