"""SQLiteGraphStore: GraphStore protocol backed by SQLite persistence.
Fixed 4-table schema (nodes, edges, embeddings, metadata).
Loads into in-memory dicts for O(1) reads at query time."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from graph.graph_component_implementation.utils import cosine_similarity
from resonance.types import GraphStore, Node, Edge, Subgraph as ResonanceSubgraph

from .dict_graph_store import EdgeRecord

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    node_type TEXT DEFAULT 'concept',
    activation REAL DEFAULT 0.5,
    use_count INTEGER DEFAULT 0,
    create_time REAL,
    sense_id INTEGER
);

CREATE TABLE IF NOT EXISTS edges (
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    strength REAL DEFAULT 0.9,
    confidence REAL DEFAULT 0.8,
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    node_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class SQLiteGraphStore(GraphStore):
    def __init__(self, db_path: Optional[str] = None):
        self._nodes: Dict[int, Node] = {}
        self._label_to_id: Dict[str, int] = {}
        self._id_to_label: Dict[int, str] = {}
        self._neighbors: Dict[int, List[Tuple[int, Edge]]] = {}
        self._embeddings: Dict[int, np.ndarray] = {}
        self._edges_raw: List[EdgeRecord] = []
        self._relation_set: set = set()
        self._next_id: int = 1
        self._db_path: Optional[str] = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._metadata: Dict[str, str] = {}

        if db_path:
            self._connect()
            self._load_from_sqlite()

    # ---- Persistence ----

    def _connect(self):
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SQL_SCHEMA)
        self._conn.commit()

    def _load_from_sqlite(self):
        self._connect()
        cursor = self._conn.cursor()

        cursor.execute("SELECT id, label, node_type, activation, use_count, create_time, sense_id FROM nodes")
        for row in cursor.fetchall():
            nid, label, node_type, activation, use_count, create_time, sense_id = row
            self._nodes[nid] = Node(
                id=nid, label=label, node_type=node_type,
                embedding=np.zeros(_EMBEDDING_DIM, dtype=np.float32),
                activation=activation, use_count=use_count,
                create_time=create_time, sense_id=sense_id,
            )
            self._id_to_label[nid] = label
            self._label_to_id[label] = nid
            if nid >= self._next_id:
                self._next_id = nid + 1

        cursor.execute("SELECT source_id, target_id, relation, strength, confidence FROM edges")
        for row in cursor.fetchall():
            src, tgt, rel, strength, confidence = row
            self._relation_set.add(rel)
            self._edges_raw.append(EdgeRecord(
                source=src, target=tgt, relation=rel,
                strength=strength, confidence=confidence,
            ))
            edge = Edge(
                source=src, target=tgt, relation_type=rel,
                strength=strength, confidence=confidence,
                last_used=time.time(), frequency=1,
            )
            self._neighbors.setdefault(src, []).append((tgt, edge))
            self._neighbors.setdefault(tgt, []).append((src, edge))

        cursor.execute("SELECT node_id, vector FROM embeddings")
        for row in cursor.fetchall():
            nid, blob = row
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            self._embeddings[nid] = vec
            if nid in self._nodes:
                old = self._nodes[nid]
                self._nodes[nid] = Node(
                    id=old.id, label=old.label, node_type=old.node_type,
                    embedding=vec, activation=old.activation,
                    use_count=old.use_count, create_time=old.create_time,
                    sense_id=old.sense_id,
                )

        cursor.execute("SELECT key, value FROM metadata")
        for row in cursor.fetchall():
            self._metadata[row[0]] = row[1]

        logger.info(
            "SQLiteGraphStore loaded from %s: %d nodes, %d edges, %d embeddings",
            self._db_path, len(self._nodes), len(self._edges_raw), len(self._embeddings),
        )

    def save_state(self, path: str) -> None:
        path = str(path)
        is_new = path != self._db_path
        if is_new:
            if self._conn:
                self._conn.close()
            self._db_path = path
            self._conn = sqlite3.connect(path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._init_schema()
        else:
            self._connect()

        t0 = time.time()
        conn = self._conn
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM metadata")

        for nid, node in self._nodes.items():
            conn.execute(
                "INSERT INTO nodes (id, label, node_type, activation, use_count, create_time, sense_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (nid, node.label, node.node_type, node.activation, node.use_count, node.create_time, node.sense_id),
            )

        for er in self._edges_raw:
            conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, relation, strength, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (er.source, er.target, er.relation, er.strength, er.confidence),
            )

        for nid, vec in self._embeddings.items():
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (node_id, vector) VALUES (?, ?)",
                (nid, vec.astype(np.float32).tobytes()),
            )

        meta = dict(self._metadata)
        meta["node_count"] = str(len(self._nodes))
        meta["edge_count"] = str(len(self._edges_raw))
        meta["embedding_dim"] = str(_EMBEDDING_DIM)
        meta["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for key, value in meta.items():
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

        conn.commit()
        elapsed = time.time() - t0
        logger.info(
            "SQLiteGraphStore saved to %s: %d nodes, %d edges, %d embeddings in %.2fs",
            self._db_path, len(self._nodes), len(self._edges_raw), len(self._embeddings), elapsed,
        )

    @classmethod
    def load_state(cls, path: str) -> "SQLiteGraphStore":
        return cls(db_path=path)

    def get_db_path(self) -> Optional[str]:
        return self._db_path

    def get_file_size(self) -> int:
        if self._db_path and os.path.exists(self._db_path):
            return os.path.getsize(self._db_path)
        return 0

    # ---- Dataset loading (same as DictGraphStore) ----

    def add_dataset(
        self,
        concepts: Dict[str, int],
        edges: List[Dict[str, Any]],
        id_to_label: Dict[int, str],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        relation_map: Optional[Dict[str, str]] = None,
    ) -> None:
        for label, nid in concepts.items():
            label_lower = label.lower().strip()
            self._id_to_label[nid] = label

            emb = None
            if embeddings and label in embeddings:
                emb = embeddings[label]
            elif embeddings and label_lower in embeddings:
                emb = embeddings[label_lower]

            if nid not in self._nodes:
                self._nodes[nid] = Node(
                    id=nid, label=label, node_type="concept",
                    embedding=emb if emb is not None else np.zeros(_EMBEDDING_DIM, dtype=np.float32),
                    activation=0.5, use_count=0, create_time=time.time(),
                )
            if nid > self._next_id:
                self._next_id = nid + 1

        for e in edges:
            src = e["source"]
            tgt = e["target"]
            rel = e.get("relation", "related_to")
            if relation_map and rel in relation_map:
                rel = relation_map[rel]
            strength = e.get("strength", 0.9)
            confidence = e.get("confidence", 0.8)

            self._relation_set.add(rel)
            edge = Edge(
                source=src, target=tgt, relation_type=rel,
                strength=strength, confidence=confidence,
                last_used=time.time(), frequency=1,
            )

            self._neighbors.setdefault(src, []).append((tgt, edge))
            self._neighbors.setdefault(tgt, []).append((src, edge))

            self._edges_raw.append(EdgeRecord(
                source=src, target=tgt, relation=rel,
                strength=strength, confidence=confidence,
            ))

        if embeddings:
            for label, emb in embeddings.items():
                nid = concepts.get(label)
                if nid is not None:
                    self._embeddings[nid] = emb

        if len(self._embeddings) > 1:
            all_nids = sorted(self._nodes.keys())
            merged = {}
            for i in range(len(all_nids)):
                for j in range(i + 1, len(all_nids)):
                    ni, nj = all_nids[i], all_nids[j]
                    ei = self._embeddings.get(ni)
                    ej = self._embeddings.get(nj)
                    if ei is not None and ej is not None:
                        sim = float(np.dot(ei, ej))
                        if sim >= 0.92:
                            keep, drop = (ni, nj) if ni < nj else (nj, ni)
                            merged[drop] = keep
            for drop_id, keep_id in merged.items():
                if drop_id in self._nodes:
                    del self._nodes[drop_id]
                if drop_id in self._embeddings:
                    del self._embeddings[drop_id]
                if drop_id in self._id_to_label:
                    del self._id_to_label[drop_id]
                for er in self._edges_raw:
                    if er.source == drop_id:
                        er.source = keep_id
                    if er.target == drop_id:
                        er.target = keep_id
                if drop_id in self._neighbors:
                    drop_neighbors = self._neighbors.pop(drop_id)
                    for tgt, edge in drop_neighbors:
                        new_src = keep_id if edge.source == drop_id else edge.source
                        new_tgt = keep_id if edge.target == drop_id else edge.target
                        new_edge = Edge(
                            source=new_src, target=new_tgt,
                            relation_type=edge.relation_type,
                            strength=edge.strength, confidence=edge.confidence,
                            last_used=edge.last_used, frequency=edge.frequency,
                        )
                        self._neighbors.setdefault(keep_id, []).append((new_tgt, new_edge))

        logger.info(
            "Added dataset: %d nodes, %d edges, %d relation types",
            len(self._nodes), len(edges), len(self._relation_set),
        )

    def add_node(self, label: str, embedding: Optional[np.ndarray] = None,
                 node_type: str = "concept") -> int:
        if embedding is not None and len(self._embeddings) > 0:
            best_nid, best_sim = None, -1.0
            for existing_nid, existing_emb in self._embeddings.items():
                sim = float(np.dot(embedding, existing_emb))
                if sim > best_sim and sim >= 0.92:
                    best_sim = sim
                    best_nid = existing_nid
            if best_nid is not None:
                return best_nid

        nid = self._next_id
        self._next_id += 1
        self._id_to_label[nid] = label
        self._nodes[nid] = Node(
            id=nid, label=label, node_type=node_type,
            embedding=embedding if embedding is not None else np.zeros(_EMBEDDING_DIM, dtype=np.float32),
            activation=0.5, use_count=0, create_time=time.time(),
        )
        if embedding is not None:
            self._embeddings[nid] = embedding
        return nid

    def add_edge(self, source: int, target: int, relation: str,
                 strength: float = 0.9, confidence: float = 0.8) -> None:
        self._relation_set.add(relation)
        edge = Edge(
            source=source, target=target, relation_type=relation,
            strength=strength, confidence=confidence,
            last_used=time.time(), frequency=1,
        )
        self._neighbors.setdefault(source, []).append((target, edge))
        self._neighbors.setdefault(target, []).append((source, edge))
        self._edges_raw.append(EdgeRecord(
            source=source, target=target, relation=relation,
            strength=strength, confidence=confidence,
        ))

    def get_all_relations(self) -> List[str]:
        return sorted(self._relation_set)

    def get_node_count(self) -> int:
        return len(self._nodes)

    def get_edge_count(self) -> int:
        return len(self._edges_raw)

    def set_metadata(self, key: str, value: str):
        self._metadata[key] = value

    def get_metadata(self, key: str, default: str = "") -> str:
        return self._metadata.get(key, default)

    # ---- GraphStore protocol ----

    def get_node(self, node_id: int) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_neighbors(self, node_id: int) -> List[Tuple[int, Edge]]:
        return self._neighbors.get(node_id, [])

    def get_all_nodes(self) -> List[Node]:
        return list(self._nodes.values())

    def get_subgraph_by_embedding_similarity(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> ResonanceSubgraph:
        scored = []
        for nid, emb in self._embeddings.items():
            sim = float(np.dot(query_embedding, emb))
            scored.append((sim, nid))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        top = scored[:top_k]
        seed_ids = [nid for _, nid in top]

        included = set(seed_ids)
        for e in self._edges_raw:
            if e.source in included or e.target in included:
                included.add(e.source)
                included.add(e.target)

        included = sorted(included)
        node_activations = {}
        for nid in included:
            sim = 0.0
            if nid in self._embeddings:
                sim = float(np.dot(query_embedding, self._embeddings[nid]))
            node_activations[nid] = max(0.01, min(1.0, sim))

        subgraph_edges: List[Tuple[int, int, str]] = []
        edge_strengths: Dict[Tuple[int, int, str], float] = {}
        edge_confidences: Dict[Tuple[int, int, str], float] = {}
        for e in self._edges_raw:
            if e.source in included and e.target in included:
                key = (e.source, e.target, e.relation)
                subgraph_edges.append(key)
                edge_strengths[key] = e.strength
                edge_confidences[key] = e.confidence
                rev_key = (e.target, e.source, e.relation)
                subgraph_edges.append(rev_key)
                edge_strengths[rev_key] = e.strength
                edge_confidences[rev_key] = e.confidence

        return ResonanceSubgraph(
            nodes=included,
            node_activations=node_activations,
            edges=subgraph_edges,
            edge_strengths=edge_strengths,
            edge_confidences=edge_confidences,
            seed_nodes=seed_ids,
            tier_used=1,
            activation_energy=0.0,
            query_embedding=query_embedding.astype(np.float32),
            timestamp=time.time(),
        )

    def get_label(self, node_id: int) -> str:
        return self._id_to_label.get(node_id, f"node_{node_id}")

    def get_embedding(self, node_id: int) -> Optional[np.ndarray]:
        return self._embeddings.get(node_id)

    def get_all_edges(self) -> List[EdgeRecord]:
        return self._edges_raw
