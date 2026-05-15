from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from ...types import Node
from .toy_graph_store import ToyGraphStore


def _int8_emb(*values: int) -> np.ndarray:
    arr = np.array(values[:32], dtype=np.int8)
    if len(arr) < 32:
        arr = np.pad(arr, (0, 32 - len(arr)))
    return arr[:32]


def _float32_query(values: List[float]) -> np.ndarray:
    arr = np.array(values[:384], dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return np.ascontiguousarray(arr)


def build_animal_kingdom_graph() -> ToyGraphStore:
    g = ToyGraphStore()

    nodes: Dict[str, int] = {}
    nid = [0]

    def add(label: str, ntype: str = "Concept", emb_seed: int = 0) -> int:
        rid = nid[0]
        nodes[label] = rid
        rng = np.random.RandomState(emb_seed + rid)
        emb = rng.randint(-128, 128, size=(32,)).astype(np.int8)
        g.add_node(Node(
            id=rid, label=label, node_type=ntype,
            embedding=emb, activation=0.01,
            use_count=0, create_time=time.time(),
        ))
        nid[0] += 1
        return rid

    def link(src_label: str, tgt_label: str, rel: str, strength: float = 0.7, conf: float = 0.8) -> None:
        g.add_edge(nodes[src_label], nodes[tgt_label], rel, strength=strength, confidence=conf)

    add("animal", "Concept", 1)
    add("mammal", "Concept", 2)
    add("bird", "Concept", 3)
    add("fish", "Concept", 4)
    add("reptile", "Concept", 5)
    add("dog", "Entity", 6)
    add("cat", "Entity", 7)
    add("eagle", "Entity", 8)
    add("sparrow", "Entity", 9)
    add("salmon", "Entity", 10)
    add("shark", "Entity", 11)
    add("snake", "Entity", 12)
    add("lizard", "Entity", 13)
    add("whale", "Entity", 14)
    add("bat", "Entity", 15)
    add("penguin", "Entity", 16)
    add("frog", "Entity", 17)
    add("turtle", "Entity", 18)
    add("has_backbone", "Property", 19)
    add("has_wings", "Property", 20)
    add("has_fur", "Property", 21)
    add("lives_in_water", "Property", 22)
    add("lives_on_land", "Property", 23)
    add("can_fly", "Property", 24)
    add("is_warm_blooded", "Property", 25)
    add("is_cold_blooded", "Property", 26)
    add("lays_eggs", "Property", 27)
    add("gives_birth", "Property", 28)
    add("carnivore", "Concept", 29)
    add("herbivore", "Concept", 30)
    add("omnivore", "Concept", 31)
    add("predator", "Concept", 32)
    add("prey", "Concept", 33)

    link("mammal", "animal", "is_a", strength=0.95, conf=0.99)
    link("bird", "animal", "is_a", strength=0.95, conf=0.99)
    link("fish", "animal", "is_a", strength=0.95, conf=0.99)
    link("reptile", "animal", "is_a", strength=0.95, conf=0.99)
    link("dog", "mammal", "is_a", strength=0.95, conf=0.99)
    link("cat", "mammal", "is_a", strength=0.95, conf=0.99)
    link("whale", "mammal", "is_a", strength=0.95, conf=0.99)
    link("bat", "mammal", "is_a", strength=0.95, conf=0.99)
    link("eagle", "bird", "is_a", strength=0.95, conf=0.99)
    link("sparrow", "bird", "is_a", strength=0.95, conf=0.99)
    link("penguin", "bird", "is_a", strength=0.95, conf=0.99)
    link("salmon", "fish", "is_a", strength=0.95, conf=0.99)
    link("shark", "fish", "is_a", strength=0.95, conf=0.99)
    link("snake", "reptile", "is_a", strength=0.95, conf=0.99)
    link("lizard", "reptile", "is_a", strength=0.95, conf=0.99)
    link("turtle", "reptile", "is_a", strength=0.95, conf=0.99)
    link("frog", "animal", "is_a", strength=0.9, conf=0.8)
    link("frog", "has_backbone", "has_property", strength=0.6, conf=0.7)

    link("dog", "has_fur", "has_property", strength=0.9, conf=0.95)
    link("cat", "has_fur", "has_property", strength=0.9, conf=0.95)
    link("bat", "has_fur", "has_property", strength=0.7, conf=0.8)
    link("whale", "has_fur", "has_property", strength=0.1, conf=0.3)
    link("eagle", "has_wings", "has_property", strength=0.95, conf=0.99)
    link("sparrow", "has_wings", "has_property", strength=0.95, conf=0.99)
    link("bat", "has_wings", "has_property", strength=0.9, conf=0.9)
    link("penguin", "has_wings", "has_property", strength=0.8, conf=0.9)
    link("eagle", "can_fly", "has_property", strength=0.95, conf=0.99)
    link("sparrow", "can_fly", "has_property", strength=0.95, conf=0.99)
    link("bat", "can_fly", "has_property", strength=0.9, conf=0.95)
    link("penguin", "can_fly", "has_property", strength=0.05, conf=0.1)
    link("salmon", "lives_in_water", "has_property", strength=0.95, conf=0.99)
    link("shark", "lives_in_water", "has_property", strength=0.95, conf=0.99)
    link("whale", "lives_in_water", "has_property", strength=0.95, conf=0.99)
    link("frog", "lives_in_water", "has_property", strength=0.5, conf=0.7)
    link("dog", "lives_on_land", "has_property", strength=0.95, conf=0.99)
    link("cat", "lives_on_land", "has_property", strength=0.95, conf=0.99)
    link("snake", "lives_on_land", "has_property", strength=0.8, conf=0.9)
    link("turtle", "lives_on_land", "has_property", strength=0.5, conf=0.7)
    link("turtle", "lives_in_water", "has_property", strength=0.5, conf=0.7)

    link("mammal", "is_warm_blooded", "has_property", strength=0.95, conf=0.99)
    link("bird", "is_warm_blooded", "has_property", strength=0.95, conf=0.99)
    link("fish", "is_cold_blooded", "has_property", strength=0.95, conf=0.99)
    link("reptile", "is_cold_blooded", "has_property", strength=0.95, conf=0.99)
    link("frog", "is_cold_blooded", "has_property", strength=0.8, conf=0.8)

    link("mammal", "gives_birth", "has_property", strength=0.95, conf=0.95)
    link("bird", "lays_eggs", "has_property", strength=0.95, conf=0.99)
    link("fish", "lays_eggs", "has_property", strength=0.85, conf=0.9)
    link("reptile", "lays_eggs", "has_property", strength=0.95, conf=0.99)
    link("shark", "gives_birth", "has_property", strength=0.6, conf=0.7)

    link("dog", "carnivore", "is_a", strength=0.7, conf=0.8)
    link("cat", "carnivore", "is_a", strength=0.8, conf=0.9)
    link("eagle", "carnivore", "is_a", strength=0.9, conf=0.9)
    link("shark", "carnivore", "is_a", strength=0.95, conf=0.95)
    link("snake", "carnivore", "is_a", strength=0.8, conf=0.8)

    link("carnivore", "predator", "is_a", strength=0.9, conf=0.9)
    link("shark", "predator", "is_a", strength=0.8, conf=0.8)

    link("dog", "cat", "associated_with", strength=0.3, conf=0.5)
    link("cat", "bird", "contradicts", strength=0.6, conf=0.7)
    link("eagle", "snake", "causes", strength=0.4, conf=0.6)
    link("eagle", "salmon", "causes", strength=0.3, conf=0.5)
    link("dog", "salmon", "contradicts", strength=0.05, conf=0.3)
    link("whale", "fish", "contradicts", strength=0.7, conf=0.8)
    link("penguin", "whale", "spatial_near", strength=0.4, conf=0.6)

    link("mammal", "has_backbone", "has_property", strength=0.95, conf=0.99)
    link("bird", "has_backbone", "has_property", strength=0.95, conf=0.99)
    link("fish", "has_backbone", "has_property", strength=0.95, conf=0.99)
    link("reptile", "has_backbone", "has_property", strength=0.95, conf=0.99)

    return g


def animal_dataset_seeds() -> List[int]:
    return [0]


def animal_query_embedding() -> np.ndarray:
    rng = np.random.RandomState(42)
    emb = rng.randn(384).astype(np.float32)
    emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
    return np.ascontiguousarray(emb)
