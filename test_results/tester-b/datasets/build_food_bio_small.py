"""Build the food_bio_small.db dataset: tester-b's Food & Biology toy graph.

Small target (10-100 nodes). Assigned relations (Section 13.2): is_a, part_of,
has_property, example_of, synonym, antonym. `causes` added on purpose to satisfy
Section 5.4 (causal relation band). Edge cases (Section 13.3, tester-B):
  * missing relation  -> isolated node `photosynthesis` (honest fallback test)
  * duplicate labels  -> nodes `rice` and `Rice` (case-insensitive collision,
                         must stay distinct in `_label_to_id`, no silent merge)

Follows the lead's harness pattern exactly (test_results/lead/datasets/build_toy_eval.py):
  - SQLiteGraphStore.add_dataset WITHOUT an embeddings dict (avoids the
    add_dataset auto-merge at embedding cos >= 0.92, which would silently merge
    `rice`/`Rice` since case variants embed near-identically)
  - embeddings set directly on the store, then save_state -> load_state

Usage:
    python test_results/tester-b/datasets/build_food_bio_small.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CONCEPTS = [
    # --- taxonomy (is_a) ---
    "animal", "mammal", "dog", "cat", "bird", "robin", "eagle", "penguin",
    "fish", "salmon", "trout", "tree", "plant", "flower",
    "food", "fruit", "apple", "banana", "vegetable", "carrot", "tomato",
    "grain", "wheat", "rice", "Rice", "bread", "meat", "egg", "milk",
    # --- part_of ---
    "yolk", "petal", "leaf",
    # --- has_property ---
    "honey", "sweet", "lemon", "sour", "ice", "cold", "water", "liquid",
    "chili", "spicy",
    # --- synonym / antonym ---
    "happy", "glad", "small", "little", "big", "hot",
    # --- causes (added for Section 5.4) ---
    "sugar", "tooth decay",
    # --- isolated node for missing-relation edge case ---
    "photosynthesis",
]

EDGES = [
    # is_a taxonomy
    {"source": "dog",    "target": "mammal",      "relation": "is_a",         "strength": 0.95, "confidence": 0.95},
    {"source": "cat",    "target": "mammal",      "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "mammal", "target": "animal",      "relation": "is_a",         "strength": 0.95, "confidence": 0.95},
    {"source": "robin",  "target": "bird",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "eagle",  "target": "bird",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "penguin","target": "bird",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "bird",   "target": "animal",      "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "salmon", "target": "fish",        "relation": "is_a",         "strength": 0.95, "confidence": 0.95},
    {"source": "trout",  "target": "fish",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "fish",   "target": "animal",      "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "apple",  "target": "fruit",       "relation": "is_a",         "strength": 0.95, "confidence": 0.95},
    {"source": "banana", "target": "fruit",       "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "carrot", "target": "vegetable",   "relation": "is_a",         "strength": 0.95, "confidence": 0.95},
    {"source": "tomato", "target": "vegetable",   "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "fruit",  "target": "food",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "vegetable", "target": "food",     "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "wheat",  "target": "grain",       "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "rice",   "target": "grain",       "relation": "is_a",         "strength": 0.95, "confidence": 0.95},
    {"source": "Rice",   "target": "grain",       "relation": "is_a",         "strength": 0.95, "confidence": 0.95},
    {"source": "grain",  "target": "food",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "bread",  "target": "food",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "meat",   "target": "food",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    {"source": "egg",    "target": "food",        "relation": "is_a",         "strength": 0.9,  "confidence": 0.9},
    # part_of
    {"source": "yolk",   "target": "egg",         "relation": "part_of",      "strength": 0.95, "confidence": 0.95},
    {"source": "petal",  "target": "flower",      "relation": "part_of",      "strength": 0.9,  "confidence": 0.9},
    {"source": "leaf",   "target": "tree",        "relation": "part_of",      "strength": 0.9,  "confidence": 0.9},
    # has_property
    {"source": "honey",  "target": "sweet",       "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "lemon",  "target": "sour",        "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "ice",    "target": "cold",        "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "water",  "target": "liquid",      "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "chili",  "target": "spicy",       "relation": "has_property", "strength": 0.9,  "confidence": 0.9},
    # example_of
    {"source": "robin",  "target": "bird",        "relation": "example_of",   "strength": 0.9,  "confidence": 0.9},
    {"source": "salmon", "target": "fish",        "relation": "example_of",   "strength": 0.9,  "confidence": 0.9},
    # synonym
    {"source": "happy",  "target": "glad",        "relation": "synonym",      "strength": 0.95, "confidence": 0.95},
    {"source": "small",  "target": "little",      "relation": "synonym",      "strength": 0.95, "confidence": 0.95},
    # antonym
    {"source": "hot",    "target": "cold",        "relation": "antonym",      "strength": 0.98, "confidence": 0.98},
    {"source": "sweet",  "target": "sour",        "relation": "antonym",      "strength": 0.95, "confidence": 0.95},
    {"source": "big",    "target": "small",       "relation": "antonym",      "strength": 0.95, "confidence": 0.95},
    # causes (extra relation, Section 5.4)
    {"source": "sugar",  "target": "tooth decay", "relation": "causes",       "strength": 0.9,  "confidence": 0.9},
]


def _edges_by_id(concepts):
    resolved = []
    for e in EDGES:
        src = concepts.get(e["source"])
        tgt = concepts.get(e["target"])
        if src is not None and tgt is not None:
            resolved.append({
                "source": src,
                "target": tgt,
                "relation": e["relation"],
                "strength": e["strength"],
                "confidence": e["confidence"],
            })
    return resolved


OUT = Path(__file__).resolve().parent / "food_bio_small.db"


def main() -> None:
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")

    for stale in (OUT, OUT.with_suffix(".db-wal"), OUT.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {label: i + 1 for i, label in enumerate(CONCEPTS)}
    labels = list(concepts.keys())
    raw = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {label: np.asarray(raw[i], dtype=np.float32) for i, label in enumerate(labels)}

    store = SQLiteGraphStore(db_path=str(OUT))
    edges = _edges_by_id(concepts)
    # NOTE: embeddings deliberately NOT passed to add_dataset (mirrors the lead's
    # build_toy_eval.py). The add_dataset auto-merge (cos >= 0.92) would otherwise
    # silently collapse the case-colliding duplicate labels `rice`/`Rice`.
    store.add_dataset(concepts=concepts, edges=edges, id_to_label=concepts)
    for label, nid in concepts.items():
        store._embeddings[nid] = embeddings[label]
    store.set_metadata("dataset_name", "food_bio_small")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    # Reload as the pipeline does (load_state populates _label_to_id).
    loaded = SQLiteGraphStore.load_state(str(OUT))
    relation_set = sorted(loaded.get_all_relations())
    n_nodes = loaded.get_node_count()
    n_edges = loaded.get_edge_count()

    print(f"Wrote food_bio_small graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relation_set)}): {relation_set}")

    # Edge-case verifications (Section 13.3 / 5.6.4)
    rice_id, rice_upper_id = loaded._label_to_id.get("rice"), loaded._label_to_id.get("Rice")
    dup_ok = rice_id is not None and rice_upper_id is not None and rice_id != rice_upper_id
    print(f"Duplicate labels preserved: rice={rice_id}, Rice={rice_upper_id} -> distinct={dup_ok}")
    print(f"cos(rice, Rice nodes) = {float(np.dot(embeddings['rice'], embeddings['Rice'])):.4f} "
          f"(would have been auto-merged if embeddings passed to add_dataset)")

    # B4: _label_to_id resolves every label used by edges (case-correct), no KeyError.
    unresolved = []
    for e in EDGES:
        for label in (e["source"], e["target"]):
            if label not in loaded._label_to_id:
                unresolved.append(label)
    print(f"B4 label lookup: {len(unresolved)} unresolved edge labels -> "
          f"{'FAIL ' + str(set(unresolved)) if unresolved else 'OK'}")

    # B7: relation set is a subset of the 16 canonical relations.
    canonical = {"is_a", "example_of", "has_property", "causes", "caused_by", "supports",
                 "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
                 "follows", "precedes", "temporal_coincident", "spatial_near",
                 "associated_with"}
    extra = set(relation_set) - canonical
    print(f"B7 relation set: subset of 16 canonical -> {('OK' if not extra else 'FAIL ' + str(extra))}")
    print(f"  relation count={len(relation_set)} (Section 5.4 requires >= 4)")


if __name__ == "__main__":
    main()