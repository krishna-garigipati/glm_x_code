"""What seasonal edges does the nature graph actually store?

Decides whether the nws03 "summer precedes spring" answer is a direction bug
in the walker or bad data in the dataset.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

cands = [
    ROOT / "test_results" / "tester-b" / "datasets" / "nature_weather_small.db",
    ROOT / "test_results" / "tester-b" / "datasets" / "nature_weather.db",
]
db = next((p for p in cands if p.exists()), None)
if db is None:
    hits = list(ROOT.rglob("nature_weather*.db"))
    db = hits[0] if hits else None
print("db:", db)
if db is None:
    raise SystemExit("nature graph not found")

store = SQLiteGraphStore.load_state(str(db))
nodes = {n.id: n.label for n in store.get_all_nodes()}
edges = store.get_all_edges()
print("nodes", len(nodes), "edges", len(edges))

seasons = {"spring", "summer", "autumn", "winter", "fall", "monsoon"}
print("\nseason-related edges:")
for e in edges:
    s = nodes.get(e.source, "?")
    t = nodes.get(e.target, "?")
    if e.relation in ("precedes", "follows") or s in seasons or t in seasons:
        print("   {:<12} -{:<20}-> {:<12}".format(s, e.relation, t))

print("\ntemporal edges anywhere:")
for e in edges:
    if e.relation in ("precedes", "follows"):
        print("   {:<12} -{:<10}-> {:<12}".format(
            nodes.get(e.source, "?"), e.relation, nodes.get(e.target, "?")))