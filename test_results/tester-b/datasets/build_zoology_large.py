"""Build zoology_large.db: tester-b's large Zoology graph (Phase C, tier=scale).

First-run exploration suite for previously NEVER-EXERCISED relations:
supports, contradicts, linguistic_maps (rest of the 16 as silent riders).
Targets the Phase A scale gate: >=100 edges AND >=6 relations.

Supports the su/supports/contradicts/linguistic_maps goldens in the unified
golden runner (zoology_large suite).

Usage:
    python test_results/tester-b/datasets/build_zoology_large.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CONCEPTS = [
    # --- taxa / animals ---
    "animal", "mammal", "bird", "reptile", "amphibian", "fish", "insect",
    "lion", "tiger", "elephant", "giraffe", "zebra", "rhinoceros", "hippopotamus",
    "monkey", "gorilla", "lemur", "bat", "whale", "dolphin", "seal",
    "eagle", "owl", "parrot", "penguin", "flamingo", "hummingbird",
    "snake", "crocodile", "chameleon", "tortoise", "gecko",
    "frog", "newt", "salamander", "axolotl",
    "shark", "salmon", "clownfish", "ray", "eel",
    "bee", "ant", "butterfly", "beetle", "dragonfly", "spider",
    "wolf", "fox", "deer", "rabbit", "squirrel", "bear", "raccoon",
    # --- body parts ---
    "wing", "beak", "feather", "claw", "fang", "fin", "gill", "scale", "shell",
    "antenna", "mandible", "pouch", "tusk", "hoof", "mane", "tail",
    # --- behavior / habitat / ecology ---
    "habitat", "savanna", "rainforest", "tundra", "desert", "ocean", "coral reef",
    "migration", "hibernation", "camouflage", "nocturnal", "predator", "prey",
    "carnivore", "herbivore", "omnivore", "ecosystem", "food chain",
    "evolution", "fossils", "natural selection", "extinction", "adaptation",
    "photosynthesis", "sunlight", "oxygen",
    # --- young / collective terms (linguistic_maps) ---
    "swan", "cygnet", "cat", "kitten", "cow", "calf", "horse", "foal",
    "gold", "gruff",  # placeholder guard words (should not collide)
    # --- supports / contradicts concepts ---
    "sugar", "tooth decay", "smoking", "lung cancer", "fire", "heat", "charge", "current",
    # --- firmed-up labels so rider edges resolve ---
    "pollination", "grazer", "diurnal", "wild", "tame", "permanence", "conspicuousness",
]

EDGES = [
    # is_a taxonomy
    {"source": "mammal", "target": "animal", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "bird", "target": "animal", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "reptile", "target": "animal", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "amphibian", "target": "animal", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "fish", "target": "animal", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "insect", "target": "animal", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "spider", "target": "animal", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
] + [
    {"source": c, "target": "mammal", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["lion", "tiger", "elephant", "giraffe", "zebra", "rhinoceros",
              "hippopotamus", "monkey", "gorilla", "lemur", "bat", "whale",
              "dolphin", "seal", "wolf", "fox", "deer", "rabbit", "squirrel",
              "bear", "raccoon"]
] + [
    {"source": c, "target": "bird", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["eagle", "owl", "parrot", "penguin", "flamingo", "hummingbird", "swan"]
] + [
    {"source": c, "target": "reptile", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["snake", "crocodile", "chameleon", "tortoise", "gecko"]
] + [
    {"source": c, "target": "amphibian", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["frog", "newt", "salamander", "axolotl"]
] + [
    {"source": c, "target": "fish", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["shark", "salmon", "clownfish", "ray", "eel"]
] + [
    {"source": c, "target": "insect", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["bee", "ant", "butterfly", "beetle", "dragonfly"]
] + [
    # --- part_of ---
    {"source": "wing", "target": "bird", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "beak", "target": "bird", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "feather", "target": "bird", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "claw", "target": "eagle", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "fang", "target": "snake", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "fin", "target": "shark", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "gill", "target": "fish", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "scale", "target": "fish", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "shell", "target": "tortoise", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "antenna", "target": "butterfly", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "mandible", "target": "ant", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "pouch", "target": "mammal", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "tusk", "target": "elephant", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "hoof", "target": "zebra", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "mane", "target": "lion", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "tail", "target": "squirrel", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    # --- has_property ---
    {"source": "lion", "target": "carnivore", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "cow", "target": "herbivore", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "bat", "target": "nocturnal", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "owl", "target": "nocturnal", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "chameleon", "target": "camouflage", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "whale", "target": "ocean", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "desert", "target": "arid", "relation": "has_property", "strength": 0.0, "confidence": 0.0},  # placeholder, dropped below
    # --- habitat (associated_with) ---
] + [
    {"source": h, "target": "habitat", "relation": "associated_with", "strength": 0.85, "confidence": 0.85}
    for h in ["savanna", "rainforest", "tundra", "desert", "ocean", "coral reef"]
] + [
    {"source": "lion", "target": "savanna", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "elephant", "target": "savanna", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "monkey", "target": "rainforest", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "parrot", "target": "rainforest", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "coral reef", "target": "clownfish", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "pollination", "target": "bee", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    # --- predator / prey ---
    {"source": "lion", "target": "zebra", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "predator", "target": "prey", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "fox", "target": "rabbit", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    # --- causes / caused_by ---
    {"source": "sunlight", "target": "photosynthesis", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "photosynthesis", "target": "oxygen", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "fire", "target": "heat", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "sugar", "target": "tooth decay", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "smoking", "target": "lung cancer", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "charge", "target": "current", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "tooth decay", "target": "sugar", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "oxygen", "target": "photosynthesis", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "heat", "target": "fire", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    # --- synonym / antonym (zoology-relevant riders) ---
    {"source": "carnivore", "target": "predator", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "herbivore", "target": "grazer", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "nocturnal", "target": "diurnal", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "predator", "target": "prey", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "wild", "target": "tame", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    # --- SUPPORTS (Phase C: never-exercised) ---
    {"source": "evolution", "target": "fossils", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    {"source": "sunlight", "target": "photosynthesis", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    {"source": "natural selection", "target": "adaptation", "relation": "supports", "strength": 0.85, "confidence": 0.85},
    {"source": "fossils", "target": "evolution", "relation": "supports", "strength": 0.85, "confidence": 0.85},
    # --- CONTRADICTS (Phase C: never-exercised) ---
    {"source": "sugar", "target": "tooth decay", "relation": "contradicts", "strength": 0.9, "confidence": 0.9},
    {"source": "smoking", "target": "lung cancer", "relation": "contradicts", "strength": 0.9, "confidence": 0.9},
    {"source": "extinction", "target": "permanence", "relation": "contradicts", "strength": 0.85, "confidence": 0.85},
    {"source": "camouflage", "target": "conspicuousness", "relation": "contradicts", "strength": 0.85, "confidence": 0.85},
    # --- LINGUISTIC_MAPS (Phase C: never-exercised) ---
    {"source": "swan", "target": "cygnet", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    {"source": "cat", "target": "kitten", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    {"source": "cow", "target": "calf", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    {"source": "horse", "target": "foal", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    # --- example_of (rider) ---
    {"source": "clownfish", "target": "coral reef", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "salmon", "target": "fish", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
]


def _edges_by_id(concepts):
    resolved = []
    for e in EDGES:
        src, tgt = concepts.get(e["source"]), concepts.get(e["target"])
        if src is not None and tgt is not None and e["strength"] > 0:
            resolved.append({
                "source": src, "target": tgt, "relation": e["relation"],
                "strength": e["strength"], "confidence": e["confidence"],
            })
    return resolved


OUT = Path(__file__).resolve().parent / "zoology_large.db"


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
    store.add_dataset(concepts=concepts, edges=edges, id_to_label=concepts)
    for label, nid in concepts.items():
        store._embeddings[nid] = embeddings[label]
    store.set_metadata("dataset_name", "zoology_large")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    loaded = SQLiteGraphStore.load_state(str(OUT))
    relation_set = sorted(loaded.get_all_relations())
    n_nodes = loaded.get_node_count()
    n_edges = loaded.get_edge_count()
    print(f"Wrote zoology_large graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relation_set)}): {relation_set}")
    assert n_edges >= 100, f"scale gate: >=100 edges, got {n_edges}"
    assert len(relation_set) >= 6, f"scale gate: >=6 relations, got {len(relation_set)}"
    missing = {"supports", "contradicts", "linguistic_maps"} - set(relation_set)
    assert not missing, f"missing target relations: {missing}"
    canonical = {"is_a", "example_of", "has_property", "causes", "caused_by", "supports",
                 "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
                 "follows", "precedes", "temporal_coincident", "spatial_near",
                 "associated_with"}
    extra = set(relation_set) - canonical
    print(f"B7 relation set subset-of-16 -> {'OK' if not extra else 'FAIL ' + str(extra)}")


if __name__ == "__main__":
    main()