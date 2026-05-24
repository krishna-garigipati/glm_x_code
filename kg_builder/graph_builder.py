import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .entity_resolver import EntityResolver
from .confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(
        self,
        resolver: EntityResolver,
        scorer: ConfidenceScorer,
        min_confidence: float = 0.25,
        db_path: Optional[str] = None,
    ):
        self._resolver = resolver
        self._scorer = scorer
        self._min_confidence = min_confidence
        self._db_path = db_path

    def build(self, extracted_triples: List[Dict]) -> Dict[str, Any]:
        triple_freq: Dict[Tuple[str, str, str], int] = {}
        triple_meta: Dict[Tuple[str, str, str], Dict] = {}
        for item in extracted_triples:
            triple = item.get("triple")
            if not triple or len(triple) != 3:
                continue
            e1, rel, e2 = triple
            key = (e1.lower().strip(), rel, e2.lower().strip())
            triple_freq[key] = triple_freq.get(key, 0) + 1
            if key not in triple_meta:
                triple_meta[key] = {
                    "level": item.get("level", 1),
                    "relation_confidence": item.get("relation_confidence", 0.8),
                    "resolution_method": item.get("resolution_method", "new"),
                }
        concepts: Dict[str, int] = {}
        id_to_label: Dict[int, str] = {}
        edges: List[Dict[str, Any]] = []
        next_node_id = 1
        for (e1, rel, e2), freq in sorted(triple_freq.items(), key=lambda x: -x[1]):
            meta = triple_meta.get((e1, rel, e2), {})
            confidence = self._scorer.score(
                extraction_level=meta.get("level", 1),
                resolution_method=meta.get("resolution_method", "new"),
                frequency=freq,
                relation_confidence=meta.get("relation_confidence", 0.8),
            )
            if not self._scorer.is_acceptable(confidence):
                continue
            for entity_label in (e1, e2):
                if entity_label not in concepts:
                    concepts[entity_label] = next_node_id
                    id_to_label[next_node_id] = entity_label
                    next_node_id += 1
            edges.append({
                "source": concepts[e1],
                "target": concepts[e2],
                "relation": rel,
                "strength": float(min(1.0, freq / 5.0)),
                "confidence": float(confidence),
            })
        self._resolver.compute_all_embeddings()
        embeddings: Dict[str, np.ndarray] = {}
        for label in concepts:
            emb = self._resolver.get_embedding(label)
            if emb is not None:
                embeddings[label] = emb
        result = {
            "concepts": concepts,
            "edges": edges,
            "id_to_label": id_to_label,
            "embeddings": embeddings,
            "node_count": len(concepts),
            "edge_count": len(edges),
            "relation_types": list(set(e["relation"] for e in edges)),
        }
        if self._db_path:
            self._save_to_sqlite(result, self._db_path)
        logger.info("Graph built: %d nodes, %d edges, %d relation types",
                     result["node_count"], result["edge_count"],
                     len(result["relation_types"]))
        return result

    def _save_to_sqlite(self, graph: Dict[str, Any], db_path: str):
        import sqlite3
        import numpy as np
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS nodes (id INTEGER PRIMARY KEY, label TEXT UNIQUE)")
        cur.execute("CREATE TABLE IF NOT EXISTS edges (source INTEGER, target INTEGER, relation TEXT, strength REAL, confidence REAL)")
        cur.execute("CREATE TABLE IF NOT EXISTS embeddings (node_id INTEGER PRIMARY KEY, embedding BLOB)")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        for label, node_id in graph["concepts"].items():
            cur.execute("INSERT OR IGNORE INTO nodes (id, label) VALUES (?, ?)", (node_id, label))
        for e in graph["edges"]:
            cur.execute("INSERT OR IGNORE INTO edges (source, target, relation, strength, confidence) VALUES (?, ?, ?, ?, ?)",
                        (e["source"], e["target"], e["relation"], e["strength"], e["confidence"]))
        for label, emb in graph.get("embeddings", {}).items():
            nid = graph["concepts"].get(label)
            if nid is not None:
                cur.execute("INSERT OR REPLACE INTO embeddings (node_id, embedding) VALUES (?, ?)",
                            (nid, np.asarray(emb).tobytes()))
        conn.commit()
        conn.close()
        logger.info(f"Graph saved to SQLite: {db_path}")
