"""Build the food_bio_medium.db dataset: tester-b's Food & Biology medium graph.

Medium target (100-1,000 nodes; Section 5.5). Contacts 9 relations (Section 5.4):
is_a, part_of, has_property, example_of, synonym, antonym (assigned, 13.2) plus
causes, caused_by, associated_with (added on purpose). Absent (7): follows, precedes,
contradicts, supports, spatial_near, temporal_coincident, linguistic_maps — used by
the missing-relation edge-case golden (fbm19, verified: clause best-matches the
absent `spatial_near` before any present relation).

Edge cases (Section 13.3, tester-B): missing relation (isolated node `plankton`,
honest fallback) and duplicate labels (`corn` / `Corn`, no silent merge).

IMPORTANT (duplicate-label hazard, same as Small): embeddings are NOT passed to
add_dataset (avoids the add_dataset auto-merge at embedding cos >= 0.92 that would
silently collapse corn/Corn). Embeddings are set directly on the store afterwards
(lead's build_toy_eval.py pattern), then save_state -> load_state.

Usage:
    python test_results/tester-b/datasets/build_food_bio_medium.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CONCEPTS = [
    # --- taxonomy: animals ---
    "animal", "mammal",
    "dog", "cat", "cow", "pig", "sheep", "horse", "goat", "wolf", "fox", "bear",
    "lion", "tiger", "elephant", "deer", "rabbit", "squirrel",
    "bird",
    "robin", "eagle", "penguin", "sparrow", "parrot", "owl", "pigeon", "crow",
    "duck", "swan", "goose",
    "reptile", "snake", "lizard", "turtle", "crocodile",
    "amphibian", "frog", "toad", "salamander", "newt",
    "fish", "salmon", "trout", "tuna", "cod", "bass", "perch",
    "insect", "bee", "ant", "butterfly", "beetle", "mosquito", "wasp", "fly",
    # --- taxonomy: plants ---
    "plant", "tree",
    "oak", "pine", "maple", "willow", "birch", "cedar",
    "flower",
    "rose", "tulip", "daisy", "sunflower", "orchid", "lotus",
    "grass", "shrub", "seaweed", "kelp", "algae",
    # --- taxonomy: food ---
    "food", "fruit",
    "apple", "banana", "orange", "grape", "mango", "strawberry", "lemon", "cherry",
    "pear", "peach",
    "vegetable",
    "carrot", "potato", "tomato", "onion", "garlic", "broccoli", "spinach",
    "cucumber", "lettuce", "pea", "bean",
    "grain", "wheat", "barley", "oats", "rye", "millet", "corn", "Corn",
    "dairy", "milk", "cheese", "yogurt", "butter",
    "bread", "meat", "egg", "sugar", "honey", "salt", "pepper",
    # --- parts ---
    "yolk", "shell", "petal", "stem", "root", "leaf", "trunk", "branch",
    "fin", "gill", "wing", "beak", "feather",
    # --- properties / colors ---
    "sweet", "sour", "spicy", "chili", "cold", "hot", "liquid", "salty", "white",
    "yellow", "green", "red", "heat", "blood",
    # --- synonym pairs ---
    "happy", "glad", "small", "little", "big", "large", "angry", "upset",
    "quick", "fast", "tired", "sleepy", "sad", "unhappy", "begin", "start",
    "end", "finish", "nice", "kind",
    # --- antonym pairs (extras beyond hot/cold, sweet/sour, big/small) ---
    "dark", "light", "slow", "up", "down", "wet", "dry", "loud", "quiet",
    "clean", "dirty", "empty", "full", "fresh", "stale",
    # --- causal / biological ---
    "tooth decay", "fire", "smoke", "smoking", "lung cancer", "rain", "flood",
    "stress", "headache", "sunlight", "photosynthesis",
    # --- associated_with ---
    "water", "ice", "snow",
    # --- isolated node for missing-relation edge case ---
    "plankton",
]

EDGES = [
    # is_a taxonomy
    {"source": "mammal",  "target": "animal", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "bird",    "target": "animal", "relation": "is_a", "strength": 0.9,  "confidence": 0.9},
    {"source": "reptile", "target": "animal", "relation": "is_a", "strength": 0.9,  "confidence": 0.9},
    {"source": "amphibian", "target": "animal", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "fish",    "target": "animal", "relation": "is_a", "strength": 0.9,  "confidence": 0.9},
    {"source": "insect",  "target": "animal", "relation": "is_a", "strength": 0.9,  "confidence": 0.9},
] + [
    {"source": c, "target": "mammal", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["dog", "cat", "cow", "pig", "sheep", "horse", "goat", "wolf", "fox",
              "bear", "lion", "tiger", "elephant", "deer", "rabbit", "squirrel"]
] + [
    {"source": c, "target": "bird", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["robin", "eagle", "penguin", "sparrow", "parrot", "owl", "pigeon",
              "crow", "duck", "swan", "goose"]
] + [
    {"source": c, "target": "reptile", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["snake", "lizard", "turtle", "crocodile"]
] + [
    {"source": c, "target": "amphibian", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["frog", "toad", "salamander", "newt"]
] + [
    {"source": c, "target": "fish", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["salmon", "trout", "tuna", "cod", "bass", "perch"]
] + [
    {"source": c, "target": "insect", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["bee", "ant", "butterfly", "beetle", "mosquito", "wasp", "fly"]
] + [
    {"source": c, "target": "plant", "relation": "is_a", "strength": 0.95, "confidence": 0.95}
    for c in ["tree", "flower", "grass", "shrub", "seaweed", "algae"]
] + [
    {"source": "kelp", "target": "seaweed", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
] + [
    {"source": c, "target": "tree", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["oak", "pine", "maple", "willow", "birch", "cedar"]
] + [
    {"source": c, "target": "flower", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["rose", "tulip", "daisy", "sunflower", "orchid", "lotus"]
] + [
    {"source": "fruit", "target": "food", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "vegetable", "target": "food", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "grain", "target": "food", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "dairy", "target": "food", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
] + [
    {"source": c, "target": "fruit", "relation": "is_a", "strength": 0.95, "confidence": 0.95}
    for c in ["apple", "banana", "orange", "grape", "mango", "strawberry", "lemon",
              "cherry", "pear", "peach"]
] + [
    {"source": c, "target": "vegetable", "relation": "is_a", "strength": 0.95, "confidence": 0.95}
    for c in ["carrot", "potato", "tomato", "onion", "garlic", "broccoli", "spinach",
              "cucumber", "lettuce", "pea", "bean"]
] + [
    {"source": c, "target": "grain", "relation": "is_a", "strength": 0.95, "confidence": 0.95}
    for c in ["wheat", "barley", "oats", "rye", "millet", "corn", "Corn"]
] + [
    {"source": c, "target": "dairy", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["milk", "cheese", "yogurt", "butter"]
] + [
    {"source": c, "target": "food", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["bread", "meat", "egg", "sugar", "honey", "salt"]
] + [
    # part_of
    {"source": "yolk", "target": "egg", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "shell", "target": "egg", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "petal", "target": "flower", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "stem", "target": "flower", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "root", "target": "plant", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "leaf", "target": "tree", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "trunk", "target": "tree", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "branch", "target": "tree", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "fin", "target": "fish", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "gill", "target": "fish", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "wing", "target": "bird", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "beak", "target": "bird", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "feather", "target": "bird", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    # has_property
    {"source": "honey", "target": "sweet", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "sugar", "target": "sweet", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "lemon", "target": "sour", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "chili", "target": "spicy", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "pepper", "target": "spicy", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "ice", "target": "cold", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "snow", "target": "cold", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "water", "target": "liquid", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "milk", "target": "white", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "salt", "target": "salty", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "grass", "target": "green", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "banana", "target": "yellow", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "blood", "target": "red", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "fire", "target": "hot", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    # example_of (kept minimal so the walker has a single clear example target)
    {"source": "robin", "target": "bird", "relation": "example_of", "strength": 0.95, "confidence": 0.95},
    {"source": "salmon", "target": "fish", "relation": "example_of", "strength": 0.95, "confidence": 0.95},
    # synonym
    {"source": "happy", "target": "glad", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "small", "target": "little", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "big", "target": "large", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "angry", "target": "upset", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "quick", "target": "fast", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "tired", "target": "sleepy", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "sad", "target": "unhappy", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "begin", "target": "start", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "end", "target": "finish", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "nice", "target": "kind", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    # antonym
    {"source": "hot", "target": "cold", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "sweet", "target": "sour", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "big", "target": "small", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "dark", "target": "light", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "fast", "target": "slow", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "happy", "target": "sad", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "up", "target": "down", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "wet", "target": "dry", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "loud", "target": "quiet", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "clean", "target": "dirty", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "empty", "target": "full", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "fresh", "target": "stale", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    # causes (Section 5.4 causal band)
    {"source": "sugar", "target": "tooth decay", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "fire", "target": "smoke", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "fire", "target": "heat", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "smoking", "target": "lung cancer", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "rain", "target": "flood", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "stress", "target": "headache", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "sunlight", "target": "photosynthesis", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    # caused_by (added on purpose)
    {"source": "tooth decay", "target": "sugar", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "smoke", "target": "fire", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "heat", "target": "fire", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "flood", "target": "rain", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "lung cancer", "target": "smoking", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    # associated_with (added on purpose; also binds water node)
    {"source": "water", "target": "rain", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "bee", "target": "flower", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "grass", "target": "cow", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
]


def _edges_by_id(concepts):
    resolved = []
    for e in EDGES:
        src, tgt = concepts.get(e["source"]), concepts.get(e["target"])
        if src is not None and tgt is not None:
            resolved.append({
                "source": src,
                "target": tgt,
                "relation": e["relation"],
                "strength": e["strength"],
                "confidence": e["confidence"],
            })
    return resolved


OUT = Path(__file__).resolve().parent / "food_bio_medium.db"


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
    edges = _edges_by_id(concepts)  # silently drops edges with missing endpoints (should not happen)
    store.add_dataset(concepts=concepts, edges=edges, id_to_label=concepts)
    for label, nid in concepts.items():
        store._embeddings[nid] = embeddings[label]
    store.set_metadata("dataset_name", "food_bio_medium")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    loaded = SQLiteGraphStore.load_state(str(OUT))
    relation_set = sorted(loaded.get_all_relations())
    n_nodes = loaded.get_node_count()
    n_edges = loaded.get_edge_count()

    print(f"Wrote food_bio_medium graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relation_set)}): {relation_set}")
    assert 100 <= n_nodes < 1000, "must be Medium size (100-1,000)"
    assert len(relation_set) >= 4, "Section 5.4: >= 4 relations"
    assert {"is_a", "antonym"} & set(relation_set), "need is_a or antonym"
    assert {"causes", "caused_by"} & set(relation_set), "need a causal relation"
    assert {"part_of", "has_property"} & set(relation_set), "need a partitive relation"

    # Edge cases (Section 13.3)
    corn_id, corn_upper_id = loaded._label_to_id.get("corn"), loaded._label_to_id.get("Corn")
    dup_ok = corn_id is not None and corn_upper_id is not None and corn_id != corn_upper_id
    print(f"Duplicate labels preserved: corn={corn_id}, Corn={corn_upper_id} -> distinct={dup_ok}")
    print(f"cos(corn, Corn) = {float(np.dot(embeddings['corn'], embeddings['Corn'])):.4f}")

    print(f"plankton isolated -> edges touching plankton: "
          f"{[e for e in edges if 'plankton' in (e['source'], e['target'])]}")

    # B4 / B7
    unresolved = [lbl for e in EDGES for lbl in (e["source"], e["target"])
                  if lbl not in loaded._label_to_id]
    canonical = {"is_a", "example_of", "has_property", "causes", "caused_by", "supports",
                 "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
                 "follows", "precedes", "temporal_coincident", "spatial_near",
                 "associated_with"}
    extra = set(relation_set) - canonical
    print(f"B4 label lookup: {len(unresolved)} unresolved -> "
          f"{'FAIL ' + str(set(unresolved)) if unresolved else 'OK'}")
    print(f"B7 relation set subset-of-16 -> {'OK' if not extra else 'FAIL ' + str(extra)}")


if __name__ == "__main__":
    main()