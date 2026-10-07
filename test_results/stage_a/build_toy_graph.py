"""Stage A - Toy Graph (contract section 16).

Purpose: "Prove the full pipeline works end-to-end" on a small clean graph.

Design constraints, taken directly from the contract:
  * graph_building_rules: only the 16 canonical relations; clear unambiguous
    facts; strength and confidence in 0.8-1.0; explicit direction pairs;
    short multi-hop paths supported.
  * node_creation_guidelines: clear entity/concept names, no ambiguous labels,
    specific entities plus broader categories so multi-hop has somewhere to go.

Deliberate property: for every (anchor, relation) pair that Stage A asks about,
there is EXACTLY ONE outgoing edge. The earlier tester graphs failed questions
like gg11 (glacier -> iceberg and glacier -> ice cap) and nws08 (storm had four
equal part_of targets) purely because the DATA was ambiguous, not because the
code was wrong. A toy graph whose failures are unambiguous is the point of this
stage, so ambiguity is designed out.

Multi-hop chains (the mechanism this stage most needs to prove):
    fin  -part_of-> fish  -is_a->     animal
    dog  -is_a->    mammal -is_a->   animal
    gill -part_of-> fish  -is_a->    animal
    oak  -is_a->    tree  -is_a->    plant
    rain -causes->  flood  -is_a->   disaster   (causes hop, then is_a hop)

Usage:
    python test_results/stage_a/build_toy_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

# --------------------------------------------------------------------------
# Nodes: 52 labels, grouped by domain for readability of the edge table.
# --------------------------------------------------------------------------
CONCEPTS = [
    # --- animal taxonomy (is_a backbone, 2 hops deep) ---
    "animal", "mammal", "bird", "fish", "reptile", "insect",
    "dog", "cat", "robin", "eagle", "salmon",
    # --- animal anatomy (part_of) ---
    "fin", "gill", "tail", "wing",
    # --- plants (is_a 2 hops deep) ---
    "plant", "tree", "flower",
    "oak", "rose",
    # --- weather (causes, part_of, associated_with, temporal) ---
    "storm", "cloud", "rain", "flood", "thunder", "lightning",
    "hurricane",
    # --- physical science ---
    "water", "liquid", "ice", "sunlight", "photosynthesis", "carbon dioxide",
    # --- geography (is_a, part_of) ---
    "mountain", "peak", "valley", "river", "glacier", "iceberg",
    # --- food / lexicon (has_property, synonym, antonym, example_of, supports,
    #     contradicts, linguistic_maps) ---
    "food", "fruit", "vegetable", "apple", "carrot", "sweet", "honey",
    "cold", "hot", "h2o", "doctor", "physician", "fossils", "evolution",
    "hacienda", "farm", "transparent",
    # --- seasons (precedes / follows direction pair) ---
    "spring", "summer", "autumn", "winter",
]

# --------------------------------------------------------------------------
# Edges. (source, target, relation, strength, confidence)
# Every strength/confidence is within the 0.8-1.0 band the contract requires.
# --------------------------------------------------------------------------
EDGES = [
    # ---- animal is_a backbone (enables dog->mammal->animal 2-hop) ----
    ("mammal", "animal", "is_a", 0.95, 0.95),
    ("bird", "animal", "is_a", 0.95, 0.95),
    ("fish", "animal", "is_a", 0.95, 0.95),
    ("reptile", "animal", "is_a", 0.95, 0.95),
    ("insect", "animal", "is_a", 0.95, 0.95),
    ("dog", "mammal", "is_a", 0.98, 0.98),
    ("cat", "mammal", "is_a", 0.98, 0.98),
    ("robin", "bird", "is_a", 0.98, 0.98),
    ("eagle", "bird", "is_a", 0.98, 0.98),
    ("salmon", "fish", "is_a", 0.98, 0.98),

    # ---- anatomy: part_of (walked in BOTH directions via mirroring) ----
    ("fin", "fish", "part_of", 0.95, 0.95),
    ("gill", "fish", "part_of", 0.95, 0.95),
    ("tail", "dog", "part_of", 0.90, 0.90),
    ("wing", "bird", "part_of", 0.95, 0.95),
    ("trunk", "tree", "part_of", 0.95, 0.95),

    # Second-relation edges on shared category nodes. These exist so several
    # 2-hop chains alternate between two DIFFERENT relations. Consecutive
    # repeats are collapsed by contract section 8 (collapse_consecutive_
    # repeats), so a chain like is_a,is_a would collapse to one hop and could
    # never be a multi-hop test.
    ("fish", "water", "associated_with", 0.90, 0.90),
    ("bird", "sky", "associated_with", 0.90, 0.90),

    # ---- plant is_a backbone (oak->tree->plant 2-hop) ----
    ("tree", "plant", "is_a", 0.95, 0.95),
    ("flower", "plant", "is_a", 0.95, 0.95),
    ("oak", "tree", "is_a", 0.98, 0.98),
    ("rose", "flower", "is_a", 0.98, 0.98),

    # ---- weather: causes direction ----
    ("lightning", "thunder", "causes", 0.95, 0.95),
    ("rain", "flood", "causes", 0.95, 0.95),
    ("rain", "cloud", "associated_with", 0.90, 0.90),
    ("hurricane", "cloud", "causes", 0.90, 0.90),
    ("thunder", "storm", "associated_with", 0.90, 0.90),
    ("cloud", "storm", "part_of", 0.90, 0.90),
    ("autumn", "harvest", "temporal_coincident", 0.85, 0.85),
    # precedes / follows DIRECTION PAIR (season ordering, unambiguous)
    ("spring", "summer", "precedes", 0.95, 0.95),
    ("summer", "autumn", "precedes", 0.95, 0.95),
    ("autumn", "winter", "precedes", 0.95, 0.95),
    ("winter", "spring", "precedes", 0.95, 0.95),

    # ---- physical science ----
    ("water", "liquid", "is_a", 0.98, 0.98),
    ("sunlight", "photosynthesis", "causes", 0.95, 0.95),
    ("photosynthesis", "carbon dioxide", "associated_with", 0.85, 0.85),
    ("ice", "water", "part_of", 0.90, 0.90),
    ("water", "transparent", "has_property", 0.95, 0.95),
    ("ice", "cold", "has_property", 0.98, 0.98),

    # ---- geography (peak->mountain, river->valley->mountain 2-hop) ----
    ("peak", "mountain", "part_of", 0.95, 0.95),
    ("river", "valley", "part_of", 0.95, 0.95),
    ("valley", "mountain", "part_of", 0.90, 0.90),
    ("glacier", "ice", "is_a", 0.90, 0.90),
    ("iceberg", "ice", "example_of", 0.90, 0.90),

    # ---- food: has_property, synonym, antonym, example_of, supports ----
    ("apple", "fruit", "is_a", 0.98, 0.98),
    ("carrot", "vegetable", "is_a", 0.98, 0.98),
    ("fruit", "food", "is_a", 0.95, 0.95),
    ("vegetable", "food", "is_a", 0.95, 0.95),
    ("apple", "sweet", "has_property", 0.95, 0.95),
    ("honey", "sweet", "has_property", 0.98, 0.98),
    ("cold", "hot", "antonym", 0.98, 0.98),
    ("water", "h2o", "synonym", 0.95, 0.95),
    ("doctor", "physician", "synonym", 0.98, 0.98),
    ("robin", "insect", "contradicts", 0.90, 0.90),
    ("fossils", "evolution", "supports", 0.90, 0.90),
    ("hacienda", "farm", "linguistic_maps", 0.95, 0.95),
    # linguistic_maps is symmetric ("farm" and "hacienda" denote the same thing),
    # but the contract permits mirroring for only four declared pairs, so the
    # reverse is stored as a real edge rather than synthesised at walk time.
    ("farm", "hacienda", "linguistic_maps", 0.95, 0.95),
    ("river", "sea", "spatial_near", 0.90, 0.90),
    ("peak", "sky", "spatial_near", 0.90, 0.90),
]

# Nodes referenced by the edge table but kept out of the inline list so the
# domain grouping above stays readable.
EXTRA_CONCEPTS = ["harvest", "sea", "sky", "trunk"]


def build() -> None:
    labels = list(CONCEPTS)
    for extra in EXTRA_CONCEPTS:
        if extra not in labels:
            labels.append(extra)

    edges = []
    for src, tgt, rel, s, c in EDGES:
        edges.append({"source": src, "target": tgt, "relation": rel,
                      "strength": s, "confidence": c})

    print(f"Building toy graph: {len(labels)} nodes, {len(edges)} edges")

    # Real SBERT embeddings for every node label (the one thing a toy graph
    # must not fake: §6 uses embeddings for seed selection).
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    enc = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {label: np.asarray(enc[i], dtype=np.float32) for i, label in enumerate(labels)}

    out = Path(__file__).resolve().parent / "toy_graph.db"
    for stale in (out, out.with_suffix(".db-wal"), out.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {label: i + 1 for i, label in enumerate(labels)}
    store = SQLiteGraphStore(db_path=str(out))
    # add_dataset expects id-space edges; resolve labels to ids.
    resolved = []
    for e in edges:
        src_id = concepts.get(e["source"])
        tgt_id = concepts.get(e["target"])
        if src_id is None or tgt_id is None:
            print(f"WARN: missing node in edge {e['source']}->{e['target']}")
            continue
        resolved.append({
            "source": src_id,
            "target": tgt_id,
            "relation": e["relation"],
            "strength": e["strength"],
            "confidence": e["confidence"],
        })
    store.add_dataset(
        concepts=concepts,
        edges=resolved,
        id_to_label={i: l for l, i in concepts.items()},
        embeddings=embeddings,
        # CRITICAL: add_dataset() silently merges any two nodes whose embedding
        # cosine is >= 0.92, deleting the higher-id node and RETARGETING its
        # edges. With bge-small, "dog" and "animal" score 0.9248, so an
        # unprotected build dropped the `dog` node outright and rewrote the fact
        # `tail part_of dog` into the false fact `tail part_of animal` (Stage A
        # oh18 answered "tail is part of animal"). A toy graph whose whole point
        # is that its facts are exactly auditable cannot tolerate a silent
        # semantic merge, so every label is protected and the journal is asserted
        # empty below.
        protected_labels=list(labels),
    )
    store.set_metadata("dataset_name", "toy_graph_stage_a")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 section 16")
    store.save_state(str(out))

    # Fail loudly rather than ship a graph whose facts were rewritten.
    import json as _json
    journal = _json.loads(store.get_metadata("merge_journal") or "[]")
    if journal:
        raise SystemExit(
            "REFUSING TO FREEZE: add_dataset merged {} node pair(s) despite "
            "protected_labels: {}. The graph would no longer match the edge "
            "table above.".format(len(journal), journal)
        )
    if store.get_node_count() != len(labels):
        raise SystemExit(
            "REFUSING TO FREEZE: built {} nodes, expected {}.".format(
                store.get_node_count(), len(labels))
        )

    print(f"Wrote {out}")
    print(f"  nodes={store.get_node_count()} edges={store.get_edge_count()}")
    rels = sorted(store.get_all_relations())
    print(f"  relations ({len(rels)}): {rels}")

    # Sanity: flag any (source, relation) pair with more than one target, which
    # would reintroduce the gg11-style ambiguity this graph is designed to avoid.
    seen = {}
    for e in store.get_all_edges():
        seen.setdefault((store.get_label(e.source), getattr(e, "relation", getattr(e, "relation_type", ""))), []).append(
            store.get_label(e.target))
    ambiguous = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"  ambiguous (source, relation) pairs: {len(ambiguous)}")
    for k, v in list(ambiguous.items())[:12]:
        print(f"    {k[0]} -{k[1]}-> {v}")


if __name__ == "__main__":
    build()