"""DictGraphStore: In-memory GraphStore that works with ANY dataset.
Implements the protocols expected by resonance, walker, and G2P planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time

import numpy as np

from resonance.types import GraphStore, Node, Edge, Subgraph as ResonanceSubgraph

from .utils import cosine_similarity

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384


@dataclass
class EdgeRecord:
    source: int
    target: int
    relation: str
    strength: float = 0.9
    confidence: float = 0.8


class DictGraphStore(GraphStore):
    """Universal in-memory graph store that can ingest any dataset.
    
    Usage:
        store = DictGraphStore()
        store.add_dataset(concepts_dict, edges_list, embeddings_dict, relation_map)
    """

    def __init__(self):
        self._nodes: Dict[int, Node] = {}
        self._label_to_id: Dict[str, int] = {}
        self._id_to_label: Dict[int, str] = {}
        self._neighbors: Dict[int, List[Tuple[int, Edge]]] = {}
        self._embeddings: Dict[int, np.ndarray] = {}
        self._edges_raw: List[EdgeRecord] = []
        self._relation_set: set = set()
        self._next_id: int = 1

    # ----- Dataset loading (universal) -----

    def add_dataset(
        self,
        concepts: Dict[str, int],
        edges: List[Dict[str, Any]],
        id_to_label: Dict[int, str],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        relation_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add any dataset to the graph store.
        
        Args:
            concepts: label -> node_id mapping
            edges: list of dicts with source, target, relation, strength, confidence
            id_to_label: node_id -> label mapping
            embeddings: label -> embedding vector (optional, for similarity search)
            relation_map: maps raw relation names to standard types
                          e.g. {"Antonym": "antonym", "Synonym": "synonym", "RelatedTo": "associated_with"}
        """
        for label, nid in concepts.items():
            label_lower = label.lower().strip()
            if label_lower in self._label_to_id:
                existing_id = self._label_to_id[label_lower]
                self._id_to_label[existing_id] = label
            else:
                self._label_to_id[label_lower] = nid
                self._id_to_label[nid] = label

            emb = None
            if embeddings and label in embeddings:
                emb = embeddings[label]
            elif embeddings and label_lower in embeddings:
                emb = embeddings[label_lower]

            node = Node(
                id=nid,
                label=label,
                node_type="concept",
                embedding=emb if emb is not None else np.zeros(_EMBEDDING_DIM, dtype=np.float32),
                activation=0.5,
                use_count=0,
                create_time=time.time(),
            )
            if nid not in self._nodes:
                self._nodes[nid] = node
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
                source=src,
                target=tgt,
                relation_type=rel,
                strength=strength,
                confidence=confidence,
                last_used=time.time(),
                frequency=1,
            )

            if src not in self._neighbors:
                self._neighbors[src] = []
            self._neighbors[src].append((tgt, edge))

            if tgt not in self._neighbors:
                self._neighbors[tgt] = []
            self._neighbors[tgt].append((src, edge))

            self._edges_raw.append(EdgeRecord(
                source=src, target=tgt, relation=rel,
                strength=strength, confidence=confidence,
            ))

        if embeddings:
            for label, emb in embeddings.items():
                nid = concepts.get(label)
                if nid is not None:
                    self._embeddings[nid] = emb

        logger.info(
            "Added dataset: %d nodes, %d edges, %d relation types",
            len(concepts), len(edges), len(self._relation_set),
        )

    def add_node(self, label: str, embedding: Optional[np.ndarray] = None,
                 node_type: str = "concept") -> int:
        label_lower = label.lower().strip()
        if label_lower in self._label_to_id:
            return self._label_to_id[label_lower]

        nid = self._next_id
        self._next_id += 1
        self._label_to_id[label_lower] = nid
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
        if source not in self._neighbors:
            self._neighbors[source] = []
        self._neighbors[source].append((target, edge))
        if target not in self._neighbors:
            self._neighbors[target] = []
        self._neighbors[target].append((source, edge))
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

    # ----- GraphStore protocol implementation -----

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
        scored.sort(key=lambda x: x[0], reverse=True)
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

    # ----- Persistence -----

    def save_state(self, path: str) -> None:
        """Save full graph state to a directory."""
        import json, pickle, os
        os.makedirs(path, exist_ok=True)

        nodes_data = {}
        for nid, node in self._nodes.items():
            nodes_data[str(nid)] = {
                "id": node.id, "label": node.label, "node_type": node.node_type,
                "activation": node.activation, "use_count": node.use_count,
                "create_time": node.create_time, "sense_id": node.sense_id,
            }
        with open(os.path.join(path, "nodes.json"), "w") as f:
            json.dump(nodes_data, f, indent=2)

        edges_data = []
        for e in self._edges_raw:
            edges_data.append({
                "source": e.source, "target": e.target, "relation": e.relation,
                "strength": e.strength, "confidence": e.confidence,
            })
        with open(os.path.join(path, "edges.json"), "w") as f:
            json.dump(edges_data, f, indent=2)

        label_to_id = dict(self._label_to_id)
        with open(os.path.join(path, "label_map.json"), "w") as f:
            json.dump(label_to_id, f, indent=2)

        id_to_label = {str(k): v for k, v in self._id_to_label.items()}
        with open(os.path.join(path, "id_to_label.json"), "w") as f:
            json.dump(id_to_label, f, indent=2)

        relation_set = sorted(self._relation_set)
        with open(os.path.join(path, "relation_set.json"), "w") as f:
            json.dump(relation_set, f, indent=2)

        meta = {"next_id": self._next_id}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        emb_nids = []
        emb_vectors = []
        for nid in sorted(self._embeddings.keys()):
            emb_nids.append(nid)
            emb_vectors.append(self._embeddings[nid])
        if emb_vectors:
            np.savez(os.path.join(path, "embeddings.npz"),
                     node_ids=np.array(emb_nids, dtype=np.int32),
                     vectors=np.stack(emb_vectors))

        import shutil
        if os.path.exists(path):
            for fname in ["nodes.json", "edges.json", "label_map.json",
                          "id_to_label.json", "relation_set.json", "meta.json",
                          "embeddings.npz"]:
                if not os.path.exists(os.path.join(path, fname)):
                    if fname == "embeddings.npz" and not emb_vectors:
                        continue

        logger.info("Graph state saved to %s (%d nodes, %d edges, %d embeddings)",
                     path, len(self._nodes), len(self._edges_raw), len(self._embeddings))

    @classmethod
    def load_state(cls, path: str) -> "DictGraphStore":
        """Load full graph state from a directory."""
        import json, os, pickle
        store = cls()

        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            store._next_id = meta.get("next_id", 1)

        nodes_path = os.path.join(path, "nodes.json")
        if os.path.exists(nodes_path):
            with open(nodes_path) as f:
                nodes_data = json.load(f)
            for nid_str, nd in nodes_data.items():
                nid = int(nid_str)
                emb = store._embeddings.get(nid, np.zeros(_EMBEDDING_DIM, dtype=np.float32))
                node = Node(
                    id=nd["id"], label=nd["label"], node_type=nd.get("node_type", "concept"),
                    embedding=emb, activation=nd.get("activation", 0.5),
                    use_count=nd.get("use_count", 0), create_time=nd.get("create_time", 0.0),
                    sense_id=nd.get("sense_id"),
                )
                store._nodes[nid] = node

        id_to_label_path = os.path.join(path, "id_to_label.json")
        if os.path.exists(id_to_label_path):
            with open(id_to_label_path) as f:
                idl = json.load(f)
            store._id_to_label = {int(k): v for k, v in idl.items()}

        label_map_path = os.path.join(path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path) as f:
                store._label_to_id = json.load(f)

        relation_set_path = os.path.join(path, "relation_set.json")
        if os.path.exists(relation_set_path):
            with open(relation_set_path) as f:
                store._relation_set = set(json.load(f))

        embeddings_path = os.path.join(path, "embeddings.npz")
        if os.path.exists(embeddings_path):
            data = np.load(embeddings_path)
            node_ids = data["node_ids"]
            vectors = data["vectors"]
            for nid, vec in zip(node_ids, vectors):
                store._embeddings[int(nid)] = vec
                if int(nid) in store._nodes:
                    old = store._nodes[int(nid)]
                    store._nodes[int(nid)] = Node(
                        id=old.id, label=old.label, node_type=old.node_type,
                        embedding=vec, activation=old.activation,
                        use_count=old.use_count, create_time=old.create_time,
                        sense_id=old.sense_id,
                    )

        edges_path = os.path.join(path, "edges.json")
        if os.path.exists(edges_path):
            with open(edges_path) as f:
                edges_data = json.load(f)
            for ed in edges_data:
                store._edges_raw.append(EdgeRecord(
                    source=ed["source"], target=ed["target"], relation=ed["relation"],
                    strength=ed.get("strength", 0.9), confidence=ed.get("confidence", 0.8),
                ))
                src, tgt, rel = ed["source"], ed["target"], ed["relation"]
                edge_obj = Edge(
                    source=src, target=tgt, relation_type=rel,
                    strength=ed.get("strength", 0.9), confidence=ed.get("confidence", 0.8),
                    last_used=0.0, frequency=1,
                )
                store._neighbors.setdefault(src, []).append((tgt, edge_obj))
                store._neighbors.setdefault(tgt, []).append((src, edge_obj))

        logger.info("Graph state loaded from %s (%d nodes, %d edges, %d embeddings)",
                     path, len(store._nodes), len(store._edges_raw), len(store._embeddings))
        return store
