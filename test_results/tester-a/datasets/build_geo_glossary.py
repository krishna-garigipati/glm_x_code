"""Build geo_glossary.db: tester-a's large Geography-Glossary graph (Phase C, tier=scale).

Explores previously NEVER-EXERCISED relations temporal_coincident, spatial_near
and (with zoology_large) the linguistic_maps term/language mapping relation.
Targets the Phase A scale gate: >=100 edges AND >=6 relations.

Supports the gg* goldens in the unified golden runner (geo_glossary suite).

Usage:
    python test_results/tester-a/datasets/build_geo_glossary.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CONCEPTS = [
    # --- landforms ---
    "mountain", "valley", "plateau", "plain", "hill", "desert", "delta", "peak",
    "ridge", "slope", "basin", "canyon", "gorge", "cliff", "island",
    "archipelago", "coast", "peninsula", "glacier", "volcano", "crater", "dune",
    "oasis", "mesa", "ice cap", "landform", "tundra", "rainforest",
    # --- water bodies ---
    "ocean", "sea", "lake", "river", "stream", "brook", "strait", "gulf", "bay",
    "channel", "waterfall", "estuary", "loch", "fjord", "lagoon", "waterway",
    "inlet", "shore", "source", "mouth",
    # --- regions / culture / language ---
    "mediterranean", "europe", "asia", "sahara", "amazon", "tibet", "himalaya",
    "alps", "andes", "siberia", "equator", "tropics", "arctic",
    "farm", "hacienda", "field", "campo", "vineyard", "plantation",
    "city", "town", "village", "rural", "urban", "suburb",
    # --- environment / properties ---
    "wet season", "floods", "drought", "arid", "humid", "cold", "dry", "wet",
    "hot", "fertile", "barren", "climate", "erosion", "glacier melt",
    # --- geography objects ---
    "sharjah", "nile", "vesuvius", "iceberg", "fault line", "groundwater",
    "wildlife", "habitat",
    # --- firmed-up labels for less common geography terms ---
    "tide", "wind", "flume", "atoll", "monsoon", "emirate", "port",
    "geographic feature", "lava",
]


EDGES = [
    # --- LINGUISTIC_MAPS (Phase C: never-exercised) ---
    {"source": "farm", "target": "hacienda", "relation": "linguistic_maps", "strength": 0.85, "confidence": 0.85},
    {"source": "lake", "target": "loch", "relation": "linguistic_maps", "strength": 0.85, "confidence": 0.85},
    {"source": "field", "target": "campo", "relation": "linguistic_maps", "strength": 0.85, "confidence": 0.85},
    {"source": "river", "target": "flume", "relation": "linguistic_maps", "strength": 0.8, "confidence": 0.8},
    # --- SPATIAL_NEAR (Phase C: never-exercised) ---
    {"source": "mediterranean", "target": "europe", "relation": "spatial_near", "strength": 0.9, "confidence": 0.9},
    {"source": "himalaya", "target": "tibet", "relation": "spatial_near", "strength": 0.9, "confidence": 0.9},
    {"source": "sahara", "target": "tropics", "relation": "spatial_near", "strength": 0.8, "confidence": 0.8},
    {"source": "alps", "target": "europe", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    # --- TEMPORAL_COINCIDENT (Phase C: never-exercised) ---
    {"source": "wet season", "target": "floods", "relation": "temporal_coincident", "strength": 0.85, "confidence": 0.85},
    {"source": "drought", "target": "barren", "relation": "temporal_coincident", "strength": 0.8, "confidence": 0.8},
    {"source": "monsoon", "target": "floods", "relation": "temporal_coincident", "strength": 0.85, "confidence": 0.85},
    # --- is_a ---
    {"source": "delta", "target": "landform", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "island", "target": "landform", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "plateau", "target": "landform", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "mesa", "target": "landform", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "canyon", "target": "landform", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "peninsula", "target": "landform", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "atoll", "target": "island", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "strait", "target": "waterway", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "estuary", "target": "waterway", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "fjord", "target": "inlet", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "nile", "target": "river", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "amazon", "target": "river", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "sahara", "target": "desert", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "himalaya", "target": "mountain", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "loch", "target": "lake", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "tropics", "target": "climate", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "arctic", "target": "climate", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    # --- part_of (anchor is the source/whole; see fbm06 direction convention) ---
    {"source": "mountain", "target": "peak", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "basin", "target": "plain", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "crater", "target": "volcano", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "dune", "target": "desert", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "oasis", "target": "desert", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "source", "target": "river", "relation": "part_of", "strength": 0.8, "confidence": 0.8},
    {"source": "mouth", "target": "river", "relation": "part_of", "strength": 0.8, "confidence": 0.8},
    {"source": "lagoon", "target": "atoll", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    # --- has_property ---
    {"source": "sahara", "target": "arid", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "tundra", "target": "cold", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "rainforest", "target": "humid", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "volcano", "target": "hot", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "delta", "target": "fertile", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "drought", "target": "dry", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "desert", "target": "arid", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    # --- example_of ---
    {"source": "vesuvius", "target": "volcano", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "nile", "target": "river", "relation": "example_of", "strength": 0.85, "confidence": 0.85},
    {"source": "sahara", "target": "desert", "relation": "example_of", "strength": 0.85, "confidence": 0.85},
    {"source": "sharjah", "target": "emirate", "relation": "example_of", "strength": 0.8, "confidence": 0.8},
    # --- antonym / synonym ---
    {"source": "urban", "target": "rural", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "wet", "target": "dry", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "hot", "target": "cold", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "fertile", "target": "barren", "relation": "antonym", "strength": 0.85, "confidence": 0.85},
    {"source": "brook", "target": "stream", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "coast", "target": "shore", "relation": "synonym", "strength": 0.85, "confidence": 0.85},
    # --- associated_with ---
    {"source": "glacier", "target": "iceberg", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "river", "target": "delta", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "estuary", "target": "tide", "relation": "associated_with", "strength": 0.8, "confidence": 0.8},
    {"source": "dune", "target": "wind", "relation": "associated_with", "strength": 0.8, "confidence": 0.8},
    {"source": "oasis", "target": "groundwater", "relation": "associated_with", "strength": 0.8, "confidence": 0.8},
    {"source": "canyon", "target": "erosion", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
]


EDGES += [
    # --- spatial_near volume ---
    {"source": c, "target": "port", "relation": "spatial_near", "strength": 0.8, "confidence": 0.8}
    for c in ["coast", "lagoon", "bay"]
]


EDGES += [
    # Concepts that already carry a more specific is_a parent (delta/island/
    # plateau/mesa/canyon/peninsula -> landform, strait/estuary -> waterway,
    # fjord -> inlet, loch -> lake) are excluded here to avoid is_a ambiguity.
    {"source": c, "target": "geographic feature", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["mountain", "valley", "plain", "hill", "gorge", "cliff", "glacier",
              "volcano", "crater", "dune", "oasis", "ice cap", "gulf", "bay",
              "channel", "waterfall", "lagoon", "inlet", "shore"]
]


EDGES += [
    {"source": "lagoon", "target": "island", "relation": "associated_with", "strength": 0.8, "confidence": 0.8},
    {"source": "volcano", "target": "lava", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "glacier", "target": "ice cap", "relation": "associated_with", "strength": 0.8, "confidence": 0.8},
    {"source": "fjord", "target": "sea", "relation": "associated_with", "strength": 0.8, "confidence": 0.8},
    {"source": "gulf", "target": "sea", "relation": "associated_with", "strength": 0.8, "confidence": 0.8},
    {"source": "waterfall", "target": "river", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "estuary", "target": "river", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "lagoon", "target": "sea", "relation": "spatial_near", "strength": 0.8, "confidence": 0.8},
    {"source": "gorge", "target": "river", "relation": "associated_with", "strength": 0.8, "confidence": 0.8},
    {"source": "valley", "target": "river", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "lagoon", "target": "atoll", "relation": "spatial_near", "strength": 0.8, "confidence": 0.8},
    {"source": "tibet", "target": "asia", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "siberia", "target": "asia", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "amazon", "target": "rainforest", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "equator", "target": "tropics", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "glacier", "target": "cold", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "oasis", "target": "fertile", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "ice cap", "target": "glacier", "relation": "spatial_near", "strength": 0.8, "confidence": 0.8},
    {"source": "canyon", "target": "gorge", "relation": "synonym", "strength": 0.85, "confidence": 0.85},
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


OUT = Path(__file__).resolve().parent / "geo_glossary.db"


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
    store.set_metadata("dataset_name", "geo_glossary")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    loaded = SQLiteGraphStore.load_state(str(OUT))
    relation_set = sorted(loaded.get_all_relations())
    n_nodes = loaded.get_node_count()
    n_edges = loaded.get_edge_count()
    print(f"Wrote geo_glossary graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relation_set)}): {relation_set}")
    assert n_edges >= 100, f"scale gate: >=100 edges, got {n_edges}"
    assert len(relation_set) >= 6, f"scale gate: >=6 relations, got {len(relation_set)}"
    missing = {"temporal_coincident", "spatial_near", "linguistic_maps"} - set(relation_set)
    assert not missing, f"missing target relations: {missing}"
    canonical = {"is_a", "example_of", "has_property", "causes", "caused_by", "supports",
                 "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
                 "follows", "precedes", "temporal_coincident", "spatial_near",
                 "associated_with"}
    extra = set(relation_set) - canonical
    print(f"relation set subset-of-16 -> {'OK' if not extra else 'FAIL ' + str(extra)}")


if __name__ == "__main__":
    main()