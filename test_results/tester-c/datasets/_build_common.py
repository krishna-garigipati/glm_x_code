"""Shared builder for tester-c cross-domain generalization KGs (Phase G).

Mirrors the proven pattern from build_geo_glossary.py:
  - labels BGE-embedded with BAAI/bge-small-en-v1.5
  - SQLiteGraphStore.add_dataset called WITHOUT an embeddings dict (avoids the
    add_dataset auto-merge at cos >= 0.92 collapsing distinct labels)
  - embeddings set directly on the store, then save_state -> load_state
  - reload verification: relation count, subset-of-16, edge gate

Usage (domain build script):
    from _build_common import build_db
    build_db(OUT, CONCEPTS, EDGES, dataset_name="astronomy_small")
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CANONICAL = {
    "is_a", "example_of", "has_property", "causes", "caused_by", "supports",
    "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
    "follows", "precedes", "temporal_coincident", "spatial_near",
    "associated_with",
}

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _edges_by_id(concepts: dict, edges: list) -> list:
    resolved = []
    for e in edges:
        src = concepts.get(e["source"])
        tgt = concepts.get(e["target"])
        if src is not None and tgt is not None and e["strength"] > 0:
            resolved.append({
                "source": src, "target": tgt, "relation": e["relation"],
                "strength": e["strength"], "confidence": e["confidence"],
            })
        elif src is None:
            raise KeyError(f"edge source label not in CONCEPTS: {e['source']!r}")
        elif tgt is None:
            raise KeyError(f"edge target label not in CONCEPTS: {e['target']!r}")
    return resolved


def build_db(out: Path, concepts: list, edges: list, dataset_name: str,
             min_relations: int = 6, min_edges: int = 50) -> SQLiteGraphStore:
    sbert = SentenceTransformer(EMBED_MODEL)

    for stale in (out, Path(str(out) + "-wal"), Path(str(out) + "-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {label: i + 1 for i, label in enumerate(concepts)}
    labels = list(concepts.keys())
    raw = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {label: np.asarray(raw[i], dtype=np.float32) for i, label in enumerate(labels)}

    store = SQLiteGraphStore(db_path=str(out))
    resolved = _edges_by_id(concepts, edges)
    store.add_dataset(concepts=concepts, edges=resolved, id_to_label=concepts)
    for label, nid in concepts.items():
        store._embeddings[nid] = embeddings[label]
    store.set_metadata("dataset_name", dataset_name)
    store.set_metadata("embed_model", EMBED_MODEL)
    store.save_state(str(out))

    loaded = SQLiteGraphStore.load_state(str(out))
    relset = sorted(loaded.get_all_relations())
    n_nodes, n_edges = loaded.get_node_count(), loaded.get_edge_count()
    extra = set(relset) - CANONICAL
    print(f"Wrote {dataset_name}: {n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relset)}): {relset}")
    print(f"subset-of-16 -> {'OK' if not extra else 'FAIL ' + str(extra)}")
    assert not extra, f"non-canonical relations: {extra}"
    assert len(relset) >= min_relations, f"min-relations gate: {len(relset)} < {min_relations}"
    assert n_edges >= min_edges, f"edge gate: {n_edges} < {min_edges}"
    print(f"Verification passed: >= {min_relations} relations, >= {min_edges} edges")
    return loaded