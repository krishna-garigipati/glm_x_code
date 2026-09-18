"""Debug why food_bio_medium fbm09 (snow) crashes the walker embedding lookup,
while other questions work. Replicates glmx_ask steps 1-3 and inspects state."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

DB = str(ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_medium.db")
QS = {
    "fbm09(crash)": "What property does snow have?",
    "fbm06(ok)": "What is the wing a part of?",
    "fbm01(ok)": "What is a mammal?",
}


def main():
    p = GLMXPipeline()
    p.graph_store = SQLiteGraphStore.load_state(DB)
    p.load_models()

    for name, q in QS.items():
        q_emb = p.sbert.encode(q, normalize_embeddings=True)
        seed_sub = p.graph_store.get_subgraph_by_embedding_similarity(q_emb, top_k=20)
        target = seed_sub.seed_nodes[0] if seed_sub.seed_nodes else None
        resonated, _ = p.tier1.resonate(q_emb, p.graph_store, seed_sub.seed_nodes)
        ne = resonated.node_embeddings
        print(f"\n=== {name}: {q}")
        print(f"  target(seed0)=            {target} ({p.graph_store.get_label(target)})")
        print(f"  n seed_nodes=             {len(resonated.seed_nodes)}")
        print(f"  n resonated.nodes=        {len(resonated.nodes)}")
        print(f"  resonated.node_embeddings is None? {ne is None}")
        if ne is not None:
            print(f"  n embeddings=             {len(ne)}")
        print(f"  target in resonated.nodes?        {target in resonated.nodes}")
        print(f"  target in node_embeddings?        {(ne is not None and target in ne)}")
        print(f"  get_embedding(target) None?        {p.graph_store.get_embedding(target) is None}")
        print(f"  get_node(target).embedding None?   {p.graph_store.get_node(target).embedding is None}")

        # which seeds are missing embeddings? (the walker would crash on any that wins)
        if ne is not None:
            missing = [nid for nid in resonated.seed_nodes if nid not in ne]
            print(f"  seeds missing embeddings: {[(n, p.graph_store.get_label(n)) for n in missing]}")
        else:
            print(f"  seeds missing embeddings: n/a (node_embeddings is None -> glmx_ask fetches per-node)")


if __name__ == "__main__":
    main()