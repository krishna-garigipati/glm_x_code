"""Render the complete tester-b-real-graph as a picture (PNG + SVG), from the DB.

Loads tester_b_real_graph.db's stored `nodes` and `edges` tables via sqlite3,
draws EVERY node and EVERY edge in a single layout, colours every edge by its
canonical relation, labels all nodes with their stored labels, and saves
tester_b_real_graph.png (bitmap) + tester_b_real_graph.svg (vector).
"""
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import networkx as nx

OUT = Path(__file__).resolve().parent
DB = OUT / "tester_b_real_graph.db"
PNG = OUT / "tester_b_real_graph.png"
SVG = OUT / "tester_b_real_graph.svg"

REL_COLORS = {
    "is_a": "#1f77b4", "has_property": "#aec7e8", "causes": "#ff7f0e",
    "caused_by": "#ffbb78", "follows": "#2ca02c", "precedes": "#98df8a",
    "contradicts": "#d62728", "supports": "#ff9896", "associated_with": "#9467bd",
    "example_of": "#c5b0d5", "part_of": "#8c564b", "synonym": "#c49c94",
    "antonym": "#e377c2", "temporal_coincident": "#f7b6d2", "spatial_near": "#7f7f7f",
    "linguistic_maps": "#c7c7c7",
}

conn = sqlite3.connect(DB)
nodes = dict(conn.execute("SELECT id, label FROM nodes ORDER BY id"))
edges = list(conn.execute(
    "SELECT source_id, target_id, relation, strength, confidence "
    "FROM edges ORDER BY relation, source_id"
))
meta = dict(conn.execute("SELECT key, value FROM metadata"))
conn.close()

G = nx.DiGraph()
for nid, lbl in nodes.items():
    G.add_node(nid, label=lbl)
for s, t, r, st, cf in edges:
    G.add_edge(s, t, relation=r, strength=st, confidence=cf)

pos = nx.spring_layout(G, seed=42, k=0.42, iterations=500)

fig, ax = plt.subplots(figsize=(30, 22))
ax.set_aspect("equal")
ax.axis("off")

nx.draw_networkx_edges(
    G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=16,
    connectionstyle="arc3,rad=0.16", width=2.4, alpha=0.95,
    edge_color=[REL_COLORS[G[u][v]["relation"]] for u, v in G.edges()],
)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#eaf4ff",
                       edgecolors="#33475b", linewidths=1.2, node_size=900)
nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, "label"),
                       font_size=8, font_color="#111111")

legend = [
    Line2D([0], [0], color=c, lw=3, label=r)
    for r, c in sorted(REL_COLORS.items(), key=lambda kv: kv[0])
]
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.005, 1.0),
          frameon=True, fontsize=11, title="relations")

ax.set_title(
    "tester_b_real_graph.db — COMPLETE GRAPH read back from SQLite (nodes/edges tables)\n"
    f"{G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges, "
    f"{len(set(G[u][v]['relation'] for u, v in G.edges()))} relations | "
    f"node_count={meta.get('node_count')} edge_count={meta.get('edge_count')}",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(PNG, dpi=150, bbox_inches="tight")
fig.savefig(SVG, bbox_inches="tight")
print(f"saved {PNG}")
print(f"saved {SVG}")