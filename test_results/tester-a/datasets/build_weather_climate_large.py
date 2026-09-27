"""Build weather_climate_large.db: tester-a's large Weather-Climate graph (Phase C, tier=scale).

Explores previously NEVER-EXERCISED relations temporal_coincident & spatial_near,
plus follows/precedes season sequencing and the standard causal rider set.
Targets the Phase A scale gate: >=100 edges AND >=6 relations.

Supports the wc* goldens in the unified golden runner (weather_climate_large suite).

Usage:
    python test_results/tester-a/datasets/build_weather_climate_large.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CONCEPTS = [
    # --- weather phenomena ---
    "rain", "cloud", "thunder", "lightning", "snow", "hail", "fog", "frost",
    "wind", "breeze", "storm", "hurricane", "typhoon", "tornado", "cyclone",
    "monsoon", "drizzle", "blizzard", "dew", "mist", "shower", "thunderstorm",
    "sun", "sunlight",
    # --- periods / seasons ---
    "spring", "summer", "autumn", "winter", "dawn", "dusk", "sunrise", "sunset",
    "midday", "midnight", "noon", "equinox", "solstice", "harvest", "blossom",
    # --- climate zones / places ---
    "tropics", "equator", "pole", "arctic", "desert", "coast", "ocean", "inland",
    "valley", "mountain", "plateau", "peninsula", "gulf", "marsh", "frost line",
    "fault line", "volcano", "climate",
    # --- properties ---
    "humid", "arid", "frigid", "mild", "dry", "wet", "cold", "hot", "clear",
    "overcast", "warm", "cool",
    # --- atmosphere / processes ---
    "humidity", "precipitation", "evaporation", "condensation", "pressure",
    "front", "jet stream", "erosion", "flooding", "weather system", "season",
    "white", "violent", "powerful", "dense", "eye", "morning",
    "weather phenomenon", "atmosphere",
]

EDGES = [
    # --- TEMPORAL_COINCIDENT (Phase C: never-exercised) ---
    {"source": "harvest", "target": "autumn", "relation": "temporal_coincident", "strength": 0.9, "confidence": 0.9},
    {"source": "monsoon", "target": "flooding", "relation": "temporal_coincident", "strength": 0.9, "confidence": 0.9},
    {"source": "blossom", "target": "spring", "relation": "temporal_coincident", "strength": 0.9, "confidence": 0.9},
    {"source": "sunset", "target": "dusk", "relation": "temporal_coincident", "strength": 0.9, "confidence": 0.9},
    # --- SPATIAL_NEAR (Phase C: never-exercised) ---
    {"source": "equator", "target": "tropics", "relation": "spatial_near", "strength": 0.9, "confidence": 0.9},
    {"source": "coast", "target": "ocean", "relation": "spatial_near", "strength": 0.9, "confidence": 0.9},
    {"source": "valley", "target": "mountain", "relation": "spatial_near", "strength": 0.9, "confidence": 0.9},
    {"source": "volcano", "target": "fault line", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    # --- follows / precedes (seasonal + diurnal chains) ---
    # Single canonical progression direction (precedes); the walker's INVERSE
    # mirror supplies the complementary follows edges without ambiguity.
    {"source": "spring", "target": "summer", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    {"source": "summer", "target": "autumn", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    {"source": "autumn", "target": "winter", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    {"source": "dawn", "target": "sunrise", "relation": "precedes", "strength": 0.9, "confidence": 0.9},
    # --- causes / caused_by ---
    {"source": "lightning", "target": "thunder", "relation": "causes", "strength": 0.95, "confidence": 0.95},
    {"source": "sun", "target": "evaporation", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "evaporation", "target": "condensation", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "condensation", "target": "precipitation", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "wind", "target": "erosion", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "cold", "target": "frost", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "pressure", "target": "wind", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "thunder", "target": "lightning", "relation": "caused_by", "strength": 0.95, "confidence": 0.95},
    {"source": "erosion", "target": "wind", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "frost", "target": "cold", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    # --- is_a ---
    {"source": "cyclone", "target": "storm", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "hurricane", "target": "cyclone", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "typhoon", "target": "cyclone", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "tornado", "target": "storm", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "blizzard", "target": "storm", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "monsoon", "target": "storm", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "thunderstorm", "target": "storm", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "rain", "target": "precipitation", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "snow", "target": "precipitation", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "hail", "target": "precipitation", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "drizzle", "target": "precipitation", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "dew", "target": "precipitation", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "mist", "target": "precipitation", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    # --- part_of ---
    {"source": "hurricane", "target": "eye", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "lightning", "target": "storm", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "thunder", "target": "storm", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "rain", "target": "storm", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "wind", "target": "storm", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "hail", "target": "storm", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "front", "target": "weather system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    # --- has_property ---
    {"source": "coast", "target": "humid", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "desert", "target": "arid", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "pole", "target": "frigid", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "snow", "target": "white", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "hail", "target": "cold", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "summer", "target": "warm", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "winter", "target": "cold", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "tornado", "target": "violent", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "hurricane", "target": "powerful", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "fog", "target": "dense", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    # --- associated_with ---
    {"source": "humidity", "target": "rain", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "pressure", "target": "wind", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "blizzard", "target": "wind", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "fog", "target": "dawn", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "dew", "target": "morning", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "jet stream", "target": "weather system", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    # --- synonym / antonym ---
    {"source": "breeze", "target": "wind", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "shower", "target": "rain", "relation": "synonym", "strength": 0.85, "confidence": 0.85},
    {"source": "wet", "target": "dry", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "hot", "target": "cold", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "clear", "target": "overcast", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "summer", "target": "winter", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "sunrise", "target": "sunset", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
]


EDGES += [
    # --- climate members is_a climate ---
    {"source": c, "target": "climate", "relation": "is_a", "strength": 0.85, "confidence": 0.85}
    for c in ["tropics", "desert", "pole", "arctic", "coast", "inland", "plateau", "marsh"]
]


EDGES += [
    # weather phenomena without a specific is_a parent (rain/snow/hail/drizzle/
    # dew/mist/shower already have precipitation as a cleaner parent)
    {"source": w, "target": "weather phenomenon", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for w in ["cloud", "fog", "frost", "dew", "mist", "shower", "sunlight",
              "midday", "midnight", "noon", "equinox", "solstice"]
]


EDGES += [
    {"source": "sun", "target": "sunlight", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "cloud", "target": "rain", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "equinox", "target": "season", "relation": "temporal_coincident", "strength": 0.85, "confidence": 0.85},
    {"source": "midday", "target": "noon", "relation": "synonym", "strength": 0.85, "confidence": 0.85},
    {"source": "arctic", "target": "pole", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "peninsula", "target": "coast", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "inland", "target": "plateau", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "marsh", "target": "wet", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
]


EDGES += [
    {"source": c, "target": "atmosphere", "relation": "part_of", "strength": 0.85, "confidence": 0.85}
    for c in ["humidity", "pressure", "front", "jet stream"]
]


EDGES += [
    {"source": s, "target": "season", "relation": "is_a", "strength": 0.95, "confidence": 0.95}
    for s in ["spring", "summer", "autumn", "winter"]
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


OUT = Path(__file__).resolve().parent / "weather_climate_large.db"


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
    store.set_metadata("dataset_name", "weather_climate_large")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    loaded = SQLiteGraphStore.load_state(str(OUT))
    relation_set = sorted(loaded.get_all_relations())
    n_nodes = loaded.get_node_count()
    n_edges = loaded.get_edge_count()
    print(f"Wrote weather_climate_large graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relation_set)}): {relation_set}")
    assert n_edges >= 100, f"scale gate: >=100 edges, got {n_edges}"
    assert len(relation_set) >= 6, f"scale gate: >=6 relations, got {len(relation_set)}"
    missing = {"temporal_coincident", "spatial_near"} - set(relation_set)
    assert not missing, f"missing target relations: {missing}"
    canonical = {"is_a", "example_of", "has_property", "causes", "caused_by", "supports",
                 "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
                 "follows", "precedes", "temporal_coincident", "spatial_near",
                 "associated_with"}
    extra = set(relation_set) - canonical
    print(f"relation set subset-of-16 -> {'OK' if not extra else 'FAIL ' + str(extra)}")


if __name__ == "__main__":
    main()