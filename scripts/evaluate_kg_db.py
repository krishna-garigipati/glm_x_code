#!/usr/bin/env python
"""Quick evaluation of the built KG DB — stats, queries, graph walks."""
import sys, os
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn" / "cnn_dailymail_dataset_data.db"
if not DB.exists():
    print(f"ERROR: DB not found: {DB}")
    sys.exit(1)

db_size = os.path.getsize(DB) / (1024 * 1024)
print("=" * 70)
print(f"KG EVALUATION — {DB.name} ({db_size:.2f} MB)")
print("=" * 70)

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
store = SQLiteGraphStore(db_path=str(DB))

nodes = len(store._nodes)
edges = len(store._edges_raw)
embs = len(store._embeddings)

print(f"\n[1] Stats:")
print(f"  Nodes: {nodes:,}  Edges: {edges:,}  Embs: {embs:,}  ({embs/nodes*100:.1f}%)  Avg deg: {2*edges/nodes:.2f}")

# ── Degree distribution ────────────────────────────────────────────────
print(f"\n[2] Degree Distribution:")
degree = Counter()
for e in store._edges_raw:
    degree[e.source] += 1
    degree[e.target] += 1
if degree:
    degs = list(degree.values())
    print(f"  Max: {max(degs)}  Mean: {sum(degs)/len(degs):.2f}  Median: {sorted(degs)[len(degs)//2]}")
    print(f"  Deg 1: {sum(1 for d in degs if d == 1):,}  Deg 2: {sum(1 for d in degs if d == 2):,}  Deg 3+: {sum(1 for d in degs if d >= 3):,}")
    print(f"  Hubs:")
    for nid, d in sorted(degree.items(), key=lambda x: -x[1])[:8]:
        print(f"    [{d:3d}] {store._id_to_label.get(nid, '?')}")

# ── Relations ──────────────────────────────────────────────────────────
print(f"\n[3] Relations:")
rel_counts = Counter(e.relation for e in store._edges_raw)
sorted_rels = sorted(rel_counts.items(), key=lambda x: -x[1])
print(f"  Types: {len(rel_counts):,}")
for rel, cnt in sorted_rels[:20]:
    print(f"    {cnt:5d}x  {rel}")

# ── Query entities ─────────────────────────────────────────────────────
print(f"\n[4] Entity Queries:")
for name in ["obama", "bush", "he", "police", "cnn", "congress", "court", "president", "september 11", "british"]:
    found = []
    for nid, node in store._nodes.items():
        if name.lower() in node.label.lower():
            for e in store._edges_raw:
                if e.source == nid:
                    tgt = store._id_to_label.get(e.target, "?")
                    if tgt != "?":
                        found.append((node.label, e.relation, tgt))
                elif e.target == nid:
                    src = store._id_to_label.get(e.source, "?")
                    if src != "?":
                        found.append((src, e.relation, node.label))
            if found:
                break
    if found:
        print(f"  '{name}' ({len(found)} facts):")
        for s, r, t in found[:3]:
            print(f"    {s} --[{r}]--> {t}")
    else:
        print(f"  '{name}': not found")

# ── Graph walks (fast: build adjacency once) ───────────────────────────
print(f"\n[5] Graph Walks (2-hop):")
adj_out = {nid: [] for nid in store._nodes}
for e in store._edges_raw:
    adj_out.setdefault(e.source, []).append(e)
hubs = sorted(degree.items(), key=lambda x: -x[1])[:5]
cnt = 0
for nid, _ in hubs:
    label = store._id_to_label.get(nid, "?")
    for e1 in adj_out.get(nid, [])[:3]:
        mid_lbl = store._id_to_label.get(e1.target, "?")
        for e2 in adj_out.get(e1.target, [])[:3]:
            tgt_lbl = store._id_to_label.get(e2.target, "?")
            print(f"    {label} --[{e1.relation}]--> {mid_lbl} --[{e2.relation}]--> {tgt_lbl}")
            cnt += 1
            if cnt >= 12:
                break
        if cnt >= 12:
            break
    if cnt >= 12:
        break

# ── Relation quality breakdown ─────────────────────────────────────────
print(f"\n[6] Quality:")
total = len(rel_counts)
common = sum(1 for _, c in rel_counts.items() if c >= 10)
medium = sum(1 for _, c in rel_counts.items() if 2 <= c <= 9)
rare = sum(1 for _, c in rel_counts.items() if c == 1)
print(f"  Common (>=10): {common} ({common/total*100:.1f}%)  Medium (2-9): {medium} ({medium/total*100:.1f}%)  Rare (1): {rare} ({rare/total*100:.1f}%)")
print(f"  Verdict: {'GOOD' if common/total > 0.2 else 'FAIR'} — {common} well-used relation types out of {total}" if total > 0 else "")

self_loops = sum(1 for e in store._edges_raw if e.source == e.target)
print(f"  Self-loops: {self_loops}")
print(f"  Edges/nodes: {edges/nodes:.2f}")
print()
