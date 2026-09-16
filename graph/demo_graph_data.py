"""Bundled deterministic demo knowledge graph (DEVIATION 9, PoC).

Zero-download, fully offline graph covering all 16 canonical relations so the
pipeline (resonance -> extractor -> walker -> decoder) can run end-to-end
without the ConceptNet parquet shards.

Used as the fallback graph source in scripts/glmx_ask.py when
model_training/dataset_conceptnet is not present.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from graph.graph_component_implementation.dict_graph_store import DictGraphStore

DEMO_CONCEPTS: List[str] = [
    "hot", "cold", "warm", "cool",
    "big", "small", "large", "little",
    "happy", "sad", "glad",
    "fast", "slow", "quick",
    "light", "dark",
    "day", "night",
    "dog", "cat", "animal", "mammal",
    "salmon", "fish",
    "apple", "fruit",
    "car", "vehicle", "automobile",
    "tree", "plant", "oak",
    "wheel", "engine", "page", "book", "leaf", "room", "house",
    "fire", "ice", "water", "liquid", "wet",
    "smoking", "unhealthy",
    "sun", "bright", "smoke", "rain", "flood", "heat",
    "lung_cancer",
    "ocean", "cloud",
    "spring", "summer", "autumn", "winter",
    "myth", "fact",
    "evidence", "hypothesis",
    "exercise", "health",
    "bone",
    "robin", "bird",
    "thunder", "lightning",
    "park", "lake",
    "kitchen", "dining room",
    "nerd", "geek",
]

# (source_label, target_label, relation)
DEMO_EDGES: List[Tuple[str, str, str]] = [
    # ---- antonym ----
    ("hot", "cold", "antonym"),
    ("big", "small", "antonym"),
    ("happy", "sad", "antonym"),
    ("fast", "slow", "antonym"),
    ("light", "dark", "antonym"),
    ("day", "night", "antonym"),
    # ---- synonym ----
    ("small", "little", "synonym"),
    ("big", "large", "synonym"),
    ("happy", "glad", "synonym"),
    ("fast", "quick", "synonym"),
    # ---- is_a ----
    ("dog", "animal", "is_a"),
    ("cat", "animal", "is_a"),
    ("dog", "mammal", "is_a"),
    ("salmon", "fish", "is_a"),
    ("apple", "fruit", "is_a"),
    ("car", "vehicle", "is_a"),
    ("oak", "tree", "is_a"),
    ("tree", "plant", "is_a"),
    # ---- has_property ----
    ("fire", "hot", "has_property"),
    ("ice", "cold", "has_property"),
    ("water", "liquid", "has_property"),
    ("water", "wet", "has_property"),
    ("smoking", "unhealthy", "has_property"),
    ("sun", "bright", "has_property"),
    ("car", "fast", "has_property"),
    # ---- causes ----
    ("smoking", "lung_cancer", "causes"),
    ("fire", "smoke", "causes"),
    ("rain", "flood", "causes"),
    ("fire", "heat", "causes"),
    # ---- caused_by ----
    ("lung_cancer", "smoking", "caused_by"),
    ("flood", "rain", "caused_by"),
    # ---- follows / precedes ----
    ("spring", "summer", "follows"),
    ("summer", "autumn", "follows"),
    ("autumn", "winter", "follows"),
    ("summer", "spring", "precedes"),
    ("winter", "autumn", "precedes"),
    # ---- contradicts ----
    ("myth", "fact", "contradicts"),
    # ---- supports ----
    ("evidence", "hypothesis", "supports"),
    ("exercise", "health", "supports"),
    # ---- associated_with ----
    ("water", "rain", "associated_with"),
    ("water", "ocean", "associated_with"),
    ("smoke", "fire", "associated_with"),
    ("rain", "cloud", "associated_with"),
    ("dog", "bone", "associated_with"),
    # ---- example_of ----
    ("robin", "bird", "example_of"),
    ("oak", "tree", "example_of"),
    # ---- part_of ----
    ("wheel", "car", "part_of"),
    ("engine", "car", "part_of"),
    ("page", "book", "part_of"),
    ("leaf", "tree", "part_of"),
    ("room", "house", "part_of"),
    # ---- temporal_coincident ----
    ("thunder", "lightning", "temporal_coincident"),
    # ---- spatial_near ----
    ("park", "lake", "spatial_near"),
    ("kitchen", "dining room", "spatial_near"),
    # ---- linguistic_maps ----
    ("nerd", "geek", "linguistic_maps"),
    ("automobile", "car", "linguistic_maps"),
]

ALL_16_RELATIONS = [
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near", "linguistic_maps",
]


def build_demo_store() -> DictGraphStore:
    """Build a DictGraphStore seeded with the bundled demo graph."""
    concepts: Dict[str, int] = {
        label: idx for idx, label in enumerate(DEMO_CONCEPTS, start=1)
    }
    id_to_label: Dict[int, str] = {v: k for k, v in concepts.items()}
    edges = [
        {
            "source": concepts[src],
            "target": concepts[tgt],
            "relation": rel,
            "strength": 0.9,
            "confidence": 0.8,
        }
        for src, tgt, rel in DEMO_EDGES
    ]
    store = DictGraphStore()
    store.add_dataset(concepts, edges, id_to_label)
    return store