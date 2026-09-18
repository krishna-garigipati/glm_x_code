"""Build a small toy dataset (.db) with REAL SBERT embeddings.

Demonstrates the protocol-correct toy path: a prepared graph where node
embeddings come from BAAI/bge-small-en-v1.5 label encoding (the one thing
the fork's toy test was missing).
Usage:
    python test_results/lead/datasets/build_toy_sqlite.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CONCEPTS = [
    "dog", "animal", "mammal", "cat",
    "hot", "cold", "ice",
    "water", "rain", "cloud", "flood",
    "fire", "smoke",
    "car", "wheel",
    "fast", "liquid", "warm", "cool",
]

EDGES = [
    {"source": "dog", "target": "animal", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "dog", "target": "mammal", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "cat", "target": "animal", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "hot", "target": "cold", "relation": "antonym", "strength": 0.98, "confidence": 0.98},
    {"source": "warm", "target": "cool", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "water", "target": "rain", "relation": "associated_with", "strength": 0.9, "confidence": 0.85},
    {"source": "rain", "target": "cloud", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "rain", "target": "flood", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "flood", "target": "rain", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "fire", "target": "smoke", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "wheel", "target": "car", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "car", "target": "fast", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "ice", "target": "cold", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "water", "target": "liquid", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
]

# add_dataset contract: edge source/target must be the node ids (ints),
# matching the `concepts` value namespace (cf. graph/demo_graph_data.py).
def _edges_by_id(concepts: Dict[str, int]) -> List[Dict[str, Any]]:
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

OUT = Path(__file__).resolve().parent / "toy.db"


def main() -> None:
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")

    for stale in (OUT, OUT.with_name("toy_ingested.db"), OUT.with_suffix(".db-wal"), OUT.with_suffix(".db-shm")):
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
    store.set_metadata("dataset_name", "toy")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    print(f"Wrote toy graph: {store.get_node_count()} nodes, {store.get_edge_count()} edges")

    sim = float(np.dot(embeddings["hot"], embeddings["cold"]))
    print(f"cos(hot, cold) = {sim:.3f}  (real embedding check)")
    sim2 = float(np.dot(embeddings["dog"], embeddings["cat"]))
    print(f"cos(dog, cat)  = {sim2:.3f}")


if __name__ == "__main__":
    main()