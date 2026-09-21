"""Build health_science_small.db: tester-c's Health & Science toy graph.

Small target (10-100 nodes). Assigned relations (Section 13.2, tester-C):
sk_provided = supports, contradicts, spatial_near, linguistic_maps
(+ 2 relations from A/B lists -> causes from A, part_of from B).

Follows tester-b's food_bio_small pattern exactly:
- SQLiteGraphStore.add_dataset WITHOUT embeddings (avoids auto-merge at cos>=0.92)
- embeddings set directly on the store, then save_state -> load_state

Usage:
    python test_results/tester-c/datasets/build_health_science_small.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CONCEPTS = [
    # --- supports ---
    "exercise", "health", "sunlight", "vitamin_d", "protein", "muscle",
    "sleep", "recovery",
    # --- contradicts ---
    "smoking", "sugar", "healthy_teeth", "fast_food", "diet",
    # --- spatial_near ---
    "heart", "lungs", "liver", "stomach", "brain", "skull",
    "kidney", "bladder",
    # --- linguistic_maps ---
    "doctor", "physician", "cure", "treat", "illness", "disease",
    "medicine", "drug",
    # --- causes ---
    "virus", "fever", "stress", "insomnia", "lung_cancer", "tooth_decay",
    # --- part_of (system taxonomies) ---
    "circulatory_system", "respiratory_system", "digestive_system",
    "neuron", "nervous_system",
]

EDGES = [
    # supports (4)
    {"source": "exercise", "target": "health",         "relation": "supports",      "strength": 0.95, "confidence": 0.95},
    {"source": "sunlight", "target": "vitamin_d",      "relation": "supports",      "strength": 0.9,  "confidence": 0.9},
    {"source": "protein",  "target": "muscle",         "relation": "supports",      "strength": 0.9,  "confidence": 0.9},
    {"source": "sleep",    "target": "recovery",       "relation": "supports",      "strength": 0.9,  "confidence": 0.9},
    # contradicts (4)
    {"source": "smoking",  "target": "health",         "relation": "contradicts",   "strength": 0.95, "confidence": 0.95},
    {"source": "sugar",    "target": "healthy_teeth",  "relation": "contradicts",   "strength": 0.9,  "confidence": 0.9},
    {"source": "fast_food","target": "diet",           "relation": "contradicts",   "strength": 0.9,  "confidence": 0.9},
    # spatial_near (4)
    {"source": "heart",    "target": "lungs",          "relation": "spatial_near",  "strength": 0.95, "confidence": 0.95},
    {"source": "liver",    "target": "stomach",        "relation": "spatial_near",  "strength": 0.9,  "confidence": 0.9},
    {"source": "brain",    "target": "skull",          "relation": "spatial_near",  "strength": 0.95, "confidence": 0.95},
    {"source": "kidney",   "target": "bladder",        "relation": "spatial_near",  "strength": 0.9,  "confidence": 0.9},
    # linguistic_maps (4)
    {"source": "doctor",   "target": "physician",      "relation": "linguistic_maps","strength": 0.95,"confidence": 0.95},
    {"source": "cure",     "target": "treat",          "relation": "linguistic_maps","strength": 0.9, "confidence": 0.9},
    {"source": "illness",  "target": "disease",        "relation": "linguistic_maps","strength": 0.95,"confidence": 0.95},
    {"source": "medicine", "target": "drug",           "relation": "linguistic_maps","strength": 0.9, "confidence": 0.9},
    # causes (4) - 'causes' from tester list A
    {"source": "virus",    "target": "fever",          "relation": "causes",        "strength": 0.95, "confidence": 0.95},
    {"source": "stress",   "target": "insomnia",       "relation": "causes",        "strength": 0.9,  "confidence": 0.9},
    {"source": "smoking",  "target": "lung_cancer",    "relation": "causes",        "strength": 0.95, "confidence": 0.95},
    {"source": "sugar",    "target": "tooth_decay",    "relation": "causes",        "strength": 0.9,  "confidence": 0.9},
    # part_of (4) - 'part_of' from tester list B
    {"source": "heart",    "target": "circulatory_system",  "relation": "part_of",  "strength": 0.95, "confidence": 0.95},
    {"source": "lungs",    "target": "respiratory_system",  "relation": "part_of",  "strength": 0.95, "confidence": 0.95},
    {"source": "stomach",  "target": "digestive_system",    "relation": "part_of",  "strength": 0.95, "confidence": 0.95},
    {"source": "neuron",   "target": "nervous_system",      "relation": "part_of",  "strength": 0.9,  "confidence": 0.9},
]


OUT = Path(__file__).resolve().parent / "health_science_small.db"


def main() -> None:
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")

    for stale in (OUT, OUT.with_suffix(".db-wal"), OUT.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {label: i + 1 for i, label in enumerate(CONCEPTS)}
    labels = list(concepts.keys())
    raw = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {label: np.asarray(raw[i], dtype=np.float32) for i, label in enumerate(labels)}

    resolved = []
    for e in EDGES:
        resolved.append({
            "source": concepts[e["source"]],
            "target": concepts[e["target"]],
            "relation": e["relation"],
            "strength": e["strength"],
            "confidence": e["confidence"],
        })

    store = SQLiteGraphStore(db_path=str(OUT))
    store.add_dataset(concepts=concepts, edges=resolved, id_to_label=concepts)
    for label, nid in concepts.items():
        store._embeddings[nid] = embeddings[label]
    store.set_metadata("dataset_name", "health_science_small")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    loaded = SQLiteGraphStore.load_state(str(OUT))
    relation_set = sorted(loaded.get_all_relations())
    n_nodes = loaded.get_node_count()
    n_edges = loaded.get_edge_count()
    print(f"Wrote health_science_small graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relation_set)}): {relation_set}")

    unresolved = []
    for e in EDGES:
        for label in (e["source"], e["target"]):
            if label not in loaded._label_to_id:
                unresolved.append(label)
    print(f"B4 label lookup: {len(unresolved)} unresolved edge labels -> "
          f"{'FAIL ' + str(set(unresolved)) if unresolved else 'OK'}")

    canonical = {"is_a", "example_of", "has_property", "causes", "caused_by", "supports",
                 "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
                 "follows", "precedes", "temporal_coincident", "spatial_near",
                 "associated_with"}
    extra = set(relation_set) - canonical
    print(f"B7 relation set: subset of 16 canonical -> {('OK' if not extra else 'FAIL ' + str(extra))}")
    assigned = {"supports", "contradicts", "spatial_near", "linguistic_maps", "causes", "part_of"}
    missing = assigned - set(relation_set)
    print(f"Assigned relations (4 unique + causes(A) + part_of(B)): "
          f"{'ALL PRESENT' if not missing else 'MISSING ' + str(missing)}")


if __name__ == "__main__":
    main()