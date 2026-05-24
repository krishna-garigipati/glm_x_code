"""
Cross-graph merger using embedding-only entity/relation alignment.

Opens two SQLiteGraphStore databases, aligns entities and relations
using EntityRegistry and RelationRegistry, and produces a merged store
with no semantic duplicates.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from graph.graph_component_implementation.dict_graph_store import EdgeRecord
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
from resonance.types import Node, Edge

from .entity_registry import EntityRegistry
from .relation_registry import RelationRegistry

logger = logging.getLogger(__name__)


class CrossGraphMerger:
    """
    Merge two KG databases with semantic entity/relation alignment.

    Usage:
        merger = CrossGraphMerger(entity_registry, relation_registry)
        store = merger.merge(db1_path, db2_path, output_path)
    """

    def __init__(
        self,
        entity_registry: EntityRegistry,
        relation_registry: RelationRegistry,
    ):
        self._er = entity_registry
        self._rr = relation_registry

    def merge(
        self,
        db1_path: str,
        db2_path: str,
        output_path: str,
    ) -> SQLiteGraphStore:
        """Load two DBs, align entities/relations, save merged result."""
        logger.info("Merging: %s + %s -> %s", db1_path, db2_path, output_path)

        store1 = SQLiteGraphStore(db_path=db1_path)
        store2 = SQLiteGraphStore(db_path=db2_path)

        merged = SQLiteGraphStore()
        merged._connect()
        merged._init_schema()
        merged._conn.execute("DELETE FROM nodes")
        merged._conn.execute("DELETE FROM edges")
        merged._conn.execute("DELETE FROM embeddings")
        merged._conn.execute("DELETE FROM metadata")

        node_id_map: Dict[Tuple[int, int], int] = {}
        canonical_to_new_id: Dict[str, int] = {}
        new_id_to_canonical: Dict[int, str] = {}
        new_id_to_label: Dict[int, str] = {}
        next_nid = 1

        edge_buckets: Dict[Tuple[int, int, str], float] = defaultdict(float)
        edge_confidences: Dict[Tuple[int, int, str], float] = defaultdict(float)
        edge_counts: Dict[Tuple[int, int, str], int] = defaultdict(int)

        for store_idx, store in enumerate([store1, store2]):
            labels = [store._nodes[nid].label for nid in sorted(store._nodes.keys())]
            nids = [nid for nid in sorted(store._nodes.keys())]

            embs_list: List[np.ndarray] = []
            for nid in nids:
                emb = store._embeddings.get(nid)
                if emb is None or emb.size == 0:
                    label = store._nodes[nid].label
                    emb = store._nodes[nid].embedding
                    if emb is None or emb.size == 0:
                        emb = np.zeros(384, dtype=np.float32)
                embs_list.append(emb)

            canonical_labels, new_ids = self._er.align_batch(
                labels, np.array([e for e in embs_list])
            )

            for orig_nid, label, canonical, new_id in zip(
                nids, labels, canonical_labels, new_ids
            ):
                node_id_map[(store_idx, orig_nid)] = new_id
                if new_id not in new_id_to_canonical:
                    canonical_to_new_id[canonical] = new_id
                    new_id_to_canonical[new_id] = canonical
                    new_id_to_label[new_id] = label
                    if new_id >= next_nid:
                        next_nid = new_id + 1

        logger.info(
            "Entity alignment: %s → %d canonical IDs (store1: %d nodes, store2: %d nodes)",
            f"{store1.get_node_count()} + {store2.get_node_count()}",
            len(new_id_to_canonical),
            store1.get_node_count(),
            store2.get_node_count(),
        )

        for store_idx, store in enumerate([store1, store2]):
            er_list = store._edges_raw
            relation_texts = list(set(e.relation for e in er_list))

            if relation_texts:
                self._rr.canonicalize_batch(relation_texts)

            for e in er_list:
                src = node_id_map.get((store_idx, e.source))
                tgt = node_id_map.get((store_idx, e.target))
                if src is None or tgt is None:
                    continue
                rel_canonical = self._rr.canonicalize(e.relation)
                key = (src, tgt, rel_canonical)

                edge_counts[key] += 1
                edge_buckets[key] = max(edge_buckets[key], e.strength)
                edge_confidences[key] = max(edge_confidences[key], e.confidence)

        for nid in sorted(new_id_to_canonical.keys()):
            label = new_id_to_label.get(nid, new_id_to_canonical.get(nid, f"node_{nid}"))
            emb = self._er.get_embedding(new_id_to_canonical.get(nid, label))
            if emb is None:
                emb = np.zeros(384, dtype=np.float32)
            else:
                emb = emb.astype(np.float32)
            merged._nodes[nid] = Node(
                id=nid, label=label, node_type="concept",
                embedding=emb, activation=0.5,
                use_count=0, create_time=time.time(),
            )
            merged._id_to_label[nid] = label
            merged._label_to_id[label] = nid
            merged._embeddings[nid] = emb

        for (src, tgt, rel), strength in edge_buckets.items():
            conf = edge_confidences.get((src, tgt, rel), 0.8)
            merged._relation_set.add(rel)
            edge_rec = EdgeRecord(
                source=src, target=tgt, relation=rel,
                strength=min(1.0, float(strength)),
                confidence=float(conf),
            )
            merged._edges_raw.append(edge_rec)
            edge_obj = Edge(
                source=src, target=tgt, relation_type=rel,
                strength=min(1.0, float(strength)),
                confidence=float(conf),
                last_used=time.time(), frequency=edge_counts.get((src, tgt, rel), 1),
            )
            merged._neighbors.setdefault(src, []).append((tgt, edge_obj))
            merged._neighbors.setdefault(tgt, []).append((src, edge_obj))

        if next_nid > merged._next_id:
            merged._next_id = next_nid

        merged.set_metadata("merged_from", f"{os.path.basename(db1_path)} + {os.path.basename(db2_path)}")
        merged.set_metadata("merge_time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        merged.set_metadata("source1_nodes", str(store1.get_node_count()))
        merged.set_metadata("source2_nodes", str(store2.get_node_count()))
        merged.set_metadata("merged_nodes", str(merged.get_node_count()))
        merged.set_metadata("merged_edges", str(merged.get_edge_count()))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.save_state(output_path)

        dedup_pct = 0.0
        total_in = store1.get_node_count() + store2.get_node_count()
        if total_in > 0:
            dedup_pct = (1 - merged.get_node_count() / total_in) * 100

        logger.info(
            "Merge complete: %d nodes, %d edges (dedup %.1f%%) → %s",
            merged.get_node_count(), merged.get_edge_count(), dedup_pct, output_path,
        )
        return merged
