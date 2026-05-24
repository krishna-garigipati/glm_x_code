#!/usr/bin/env python
"""Quality-check a built KG SQLite DB."""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn" / "cnn_dailymail_dataset_data.db"
if not DB.exists():
    print(f"ERROR: DB not found: {DB}")
    sys.exit(1)

db_size = os.path.getsize(DB) / (1024 * 1024)
print("=" * 70)
print(f"KG QUALITY CHECK — {DB.name} ({db_size:.2f} MB)")
print("=" * 70)

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
store = SQLiteGraphStore(db_path=str(DB))

nodes = len(store._nodes)
edges = len(store._edges_raw)
embs = len(store._embeddings)

print(f"\n[1] Basic Stats:")
print(f"  Nodes:       {nodes}")
print(f"  Edges:       {edges}")
print(f"  Embeddings:  {embs}")
print(f"  Embed cov:   {embs/nodes*100:.1f}%")

# ── Relation type distribution ──────────────────────────────────────────
print(f"\n[2] Relation Type Distribution:")
rel_counts = {}
for e in store._edges_raw:
    rel_counts[e.relation] = rel_counts.get(e.relation, 0) + 1
sorted_rels = sorted(rel_counts.items(), key=lambda x: -x[1])
print(f"  Total types: {len(rel_counts)}")
print(f"  Top 20:")
for rel, cnt in sorted_rels[:20]:
    print(f"    {cnt:4d}x  {rel}")

# ── Sample nodes ────────────────────────────────────────────────────────
print(f"\n[3] Sample Nodes (first 20):")
labels = list(dict.fromkeys(n.label for n in store._nodes.values()))
for i, lbl in enumerate(labels[:20]):
    print(f"    [{i}] {lbl}")

# ── Sample edges ────────────────────────────────────────────────────────
print(f"\n[4] Sample Edges (first 15):")
for e in store._edges_raw[:15]:
    src = store._id_to_label.get(e.source, "?")
    tgt = store._id_to_label.get(e.target, "?")
    print(f"    {src} --[{e.relation}]--> {tgt}")

# ── Sample 2-hop walks ─────────────────────────────────────────────────
print(f"\n[5] Sample 2-hop walks (first 10):")
walk_count = 0
for e1 in store._edges_raw:
    for e2 in store._edges_raw:
        if e2.source == e1.target:
            src_lbl = store._id_to_label.get(e1.source, "?")
            mid_lbl = store._id_to_label.get(e1.target, "?")
            tgt_lbl = store._id_to_label.get(e2.target, "?")
            print(f"    {src_lbl} --[{e1.relation}]--> {mid_lbl} --[{e2.relation}]--> {tgt_lbl}")
            walk_count += 1
            if walk_count >= 10:
                break
    if walk_count >= 10:
        break

# ── Consistency ─────────────────────────────────────────────────────────
print(f"\n[6] Consistency:")
self_loops = sum(1 for e in store._edges_raw if e.source == e.target)
print(f"  Self-loops:       {self_loops}")
print(f"  Relation types:   {len(rel_counts)}")
print(f"  Edges/nodes:      {edges/nodes:.2f}")
print(f"  Singletons (1x):  {sum(1 for c in rel_counts.values() if c == 1)}")
print(f"  Top rel %:        {sorted_rels[0][1]/edges*100:.1f}%" if sorted_rels else "  N/A")

# ── Embedding quality ───────────────────────────────────────────────────
print(f"\n[7] Embedding Quality:")
if embs > 1:
    import numpy as np
    embs_list = list(store._embeddings.values())
    if embs_list and len(embs_list) > 1:
        embs_arr = np.array(embs_list)
        norms = np.linalg.norm(embs_arr, axis=1)
        print(f"  Embed dim:  {embs_arr.shape[1]}")
        print(f"  Mean norm:  {norms.mean():.4f}")
        print(f"  Std norm:   {norms.std():.4f}")
        sims = embs_arr @ embs_arr.T
        np.fill_diagonal(sims, -1)
        print(f"  Max cosine: {sims.max():.4f}")
        print(f"  Min cosine: {sims.min():.4f}")

print("\n" + "=" * 70)
print("QUALITY CHECK COMPLETE")
print("=" * 70)
