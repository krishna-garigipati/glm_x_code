"""Build the nature_weather_small.db dataset: tester-a's Nature & Weather graph.

Reconciles IS-04 / IS-05: the committed 14-edge DB was a stale manual subset of the
53-triple source JSON (test_results/tester-a/datasets/nature_weather_small.json),
which meant whole relations (is_a, caused_by, antonym, has_property,
temporal_coincident) were absent and golden questions anchored on them could only
fail or answer honestly. This script makes the build deterministic and reproducible:

  source of truth = nature_weather_small.json (53 triples, 5 weather-cycle domains)
  -> SQLiteGraphStore.add_dataset WITHOUT an embeddings dict (no auto-merge at
     cos >= 0.92; labels stay exactly as authored), embeddings set post-hoc
     (lead pattern, mirrors tester-b build_food_bio_small.py)
  -> save_state -> load_state verification (B4 label lookup, B7 relation subset)

Usage:
    python test_results/tester-a/datasets/build_nature_weather_small.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CANONICAL_16 = {"is_a", "example_of", "has_property", "causes", "caused_by", "supports",
                "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
                "follows", "precedes", "temporal_coincident", "spatial_near",
                "associated_with"}

THIS = Path(__file__).resolve().parent
JSON_PATH = THIS / "nature_weather_small.json"
OUT = THIS / "nature_weather_small.db"


def main() -> None:
    triples = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    labels: list[str] = []
    for t in triples:
        for key in ("head", "tail"):
            lab = t[key].strip().lower()
            if lab not in labels:
                labels.append(lab)
    concepts = {label: i + 1 for i, label in enumerate(labels)}

    edges = []
    for t in triples:
        rel = t["relation"].strip().lower()
        src, tgt = t["head"].strip().lower(), t["tail"].strip().lower()
        if src == tgt:
            continue
        edges.append({
            "source": concepts[src],
            "target": concepts[tgt],
            "relation": rel,
            "strength": float(t.get("strength", 0.9)),
            "confidence": float(t.get("confidence", 0.9)),
        })

    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    for stale in (OUT, OUT.with_suffix(".db-wal"), OUT.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    raw = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {label: np.asarray(raw[i], dtype=np.float32) for i, label in enumerate(labels)}

    store = SQLiteGraphStore(db_path=str(OUT))
    # NOTE: embeddings deliberately not passed to add_dataset (lead/tester-b pattern)
    # to avoid the auto-merge at cos >= 0.92 collapsing near-duplicate labels.
    store.add_dataset(concepts=concepts, edges=edges, id_to_label=concepts)
    for label, nid in concepts.items():
        store._embeddings[nid] = embeddings[label]
    store.set_metadata("dataset_name", "nature_weather_small")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    loaded = SQLiteGraphStore.load_state(str(OUT))
    relation_set = sorted(loaded.get_all_relations())
    n_nodes = loaded.get_node_count()
    n_edges = loaded.get_edge_count()

    print(f"Wrote nature_weather_small graph: source_json={len(triples)} triples -> "
          f"{n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relation_set)}): {relation_set}")

    unresolved = [l for l in labels if l not in loaded._label_to_id]
    print(f"B4 label lookup: {len(unresolved)} unresolved -> {'OK' if not unresolved else str(unresolved)}")

    extra = set(relation_set) - CANONICAL_16
    print(f"B7 relation set subset of 16 canonical -> {'OK' if not extra else 'FAIL ' + str(extra)}")
    in_band = {"is_a"} <= set(relation_set) and {"antonym"} <= set(relation_set)
    print(f"Section 5.4 band (is_a + antonym present) -> {'OK' if in_band else 'FAIL'}")
    assert 10 <= n_nodes <= 1000, "small dataset must be 10-1000 nodes"


if __name__ == "__main__":
    main()