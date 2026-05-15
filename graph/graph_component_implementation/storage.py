from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import os
import struct
import threading
import time

from .errors import ShardCorruptedError
from .models import Edge, Node
from .serializer import GraphSerializer


class ShardedDiskStore:
    def __init__(
        self,
        base_path: str,
        shard_prefix: str,
        shard_size_mb: int,
        max_shards: int,
        serializer: GraphSerializer,
        node_type_registry: Dict[str, int],
        relation_registry: Optional[Dict[str, int]] = None,
        embedding_dim: int = 32,
    ) -> None:
        self.base_path = base_path
        self.shard_prefix = shard_prefix
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
        self.max_shards = max_shards
        self.serializer = serializer
        self.node_type_registry = node_type_registry
        self.relation_registry = relation_registry or {}
        self._embedding_dim = embedding_dim
        self.node_index: Dict[int, Tuple[int, int]] = {}
        self.edge_index: Dict[str, Tuple[int, int]] = {}
        self.edge_by_source: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_target: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_relation: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_timestamp: List[Tuple[float, str]] = []
        self.node_shard_index: Dict[int, int] = {}
        self.edge_shard_index: Dict[int, int] = {}
        self.node_shard_sizes: Dict[int, int] = defaultdict(int)
        self.edge_shard_sizes: Dict[int, int] = defaultdict(int)
        self.meta_lock = threading.Lock()
        os.makedirs(self.base_path, exist_ok=True)
        self._load_meta()

    def _meta_path(self) -> str:
        return os.path.join(self.base_path, "graph_meta.bin")

    def _nodes_path(self, shard_id: int) -> str:
        return os.path.join(self.base_path, f"nodes_{self.shard_prefix}{shard_id}.bin")

    def _edges_path(self, shard_id: int) -> str:
        return os.path.join(self.base_path, f"edges_{self.shard_prefix}{shard_id}.bin")

    def _load_meta(self) -> None:
        meta_path = self._meta_path()
        if not os.path.exists(meta_path):
            return
        with open(meta_path, "rb") as file_handle:
            data = file_handle.read()
        meta = self.serializer.deserialize(data)
        self.node_index = meta.get("node_index", {})
        self.edge_index = meta.get("edge_index", {})
        self.edge_by_source = defaultdict(set, meta.get("edge_by_source", {}))
        self.edge_by_target = defaultdict(set, meta.get("edge_by_target", {}))
        self.edge_by_relation = defaultdict(set, meta.get("edge_by_relation", {}))
        self.edge_by_timestamp = meta.get("edge_by_timestamp", [])
        self.node_shard_index = meta.get("node_shard_index", {})
        self.edge_shard_index = meta.get("edge_shard_index", {})
        self.node_shard_sizes = defaultdict(int, meta.get("node_shard_sizes", {}))
        self.edge_shard_sizes = defaultdict(int, meta.get("edge_shard_sizes", {}))
        self.relation_registry = meta.get("relation_registry", self.relation_registry)

    def _persist_meta(self) -> None:
        meta = {
            "node_index": self.node_index,
            "edge_index": self.edge_index,
            "edge_by_source": dict(self.edge_by_source),
            "edge_by_target": dict(self.edge_by_target),
            "edge_by_relation": dict(self.edge_by_relation),
            "edge_by_timestamp": self.edge_by_timestamp,
            "node_shard_index": self.node_shard_index,
            "edge_shard_index": self.edge_shard_index,
            "node_shard_sizes": dict(self.node_shard_sizes),
            "edge_shard_sizes": dict(self.edge_shard_sizes),
            "relation_registry": self.relation_registry,
        }
        payload = self.serializer.serialize(meta)
        tmp = self._meta_path() + ".tmp"
        with open(tmp, "wb") as file_handle:
            file_handle.write(payload)
        os.replace(tmp, self._meta_path())

    def _assign_shard(self, key_hash: int, is_node: bool, record_size: int) -> int:
        shard_id = abs(key_hash) % self.max_shards
        shard_sizes = self.node_shard_sizes if is_node else self.edge_shard_sizes
        shard_index = self.node_shard_index if is_node else self.edge_shard_index
        if shard_id in shard_sizes and shard_sizes[shard_id] + record_size > self.shard_size_bytes:
            for candidate in range(self.max_shards):
                if shard_sizes[candidate] + record_size <= self.shard_size_bytes:
                    shard_id = candidate
                    break
            else:
                raise ShardCorruptedError("All shards are full")
        shard_index[key_hash] = shard_id
        return shard_id

    def _stable_hash(self, key: str) -> int:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _append_record(self, path: str, payload: bytes) -> int:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "ab") as file_handle:
            offset = file_handle.tell()
            file_handle.write(struct.pack("<I", len(payload)))
            file_handle.write(payload)
        return offset

    def _read_record(self, path: str, offset: int) -> bytes:
        with open(path, "rb") as file_handle:
            file_handle.seek(offset)
            raw_len = file_handle.read(4)
            if len(raw_len) != 4:
                raise ShardCorruptedError("Record length missing")
            (length,) = struct.unpack("<I", raw_len)
            payload = file_handle.read(length)
        if len(payload) != length:
            raise ShardCorruptedError("Record incomplete")
        return payload

    def add_node(self, node: Node) -> bool:
        if node.node_id in self.node_index:
            return False
        payload = self._pack_node(node)
        shard_id = self._assign_shard(node.node_id, True, len(payload) + 4)
        path = self._nodes_path(shard_id)
        offset = self._append_record(path, payload)
        self.node_index[node.node_id] = (shard_id, offset)
        self.node_shard_sizes[shard_id] += len(payload) + 4
        with self.meta_lock:
            self._persist_meta()
        return True

    def update_node(self, node: Node) -> bool:
        if node.node_id not in self.node_index:
            return False
        payload = self._pack_node(node)
        shard_id = self.node_index[node.node_id][0]
        path = self._nodes_path(shard_id)
        offset = self._append_record(path, payload)
        self.node_index[node.node_id] = (shard_id, offset)
        self.node_shard_sizes[shard_id] += len(payload) + 4
        with self.meta_lock:
            self._persist_meta()
        return True

    def get_node(self, node_id: int) -> Optional[Node]:
        item = self.node_index.get(node_id)
        if item is None:
            return None
        shard_id, offset = item
        payload = self._read_record(self._nodes_path(shard_id), offset)
        return self._unpack_node(payload)

    def add_edge(self, edge: Edge) -> bool:
        key = self.edge_key(edge.source, edge.target, edge.relation)
        payload = self._pack_edge(edge)
        if key in self.edge_index:
            return self.update_edge(edge)
        shard_id = self._assign_shard(self._stable_hash(key), False, len(payload) + 4)
        path = self._edges_path(shard_id)
        offset = self._append_record(path, payload)
        self.edge_index[key] = (shard_id, offset)
        self.edge_shard_sizes[shard_id] += len(payload) + 4
        self._update_edge_indexes(edge, key)
        with self.meta_lock:
            self._persist_meta()
        return True

    def update_edge(self, edge: Edge) -> bool:
        key = self.edge_key(edge.source, edge.target, edge.relation)
        if key not in self.edge_index:
            return False
        payload = self._pack_edge(edge)
        shard_id = self.edge_index[key][0]
        path = self._edges_path(shard_id)
        offset = self._append_record(path, payload)
        self.edge_index[key] = (shard_id, offset)
        self.edge_shard_sizes[shard_id] += len(payload) + 4
        self._update_edge_indexes(edge, key)
        with self.meta_lock:
            self._persist_meta()
        return True

    def remove_node(self, node_id: int) -> None:
        self.node_index.pop(node_id, None)
        self.node_shard_index.pop(node_id, None)
        with self.meta_lock:
            self._persist_meta()

    def remove_edge(self, source: int, target: int, relation: str) -> None:
        key = self.edge_key(source, target, relation)
        relation_id = self.relation_registry.get(relation)
        self.edge_index.pop(key, None)
        self.edge_shard_index.pop(self._stable_hash(key), None)
        self.edge_by_source.get(source, set()).discard(key)
        self.edge_by_target.get(target, set()).discard(key)
        if relation_id is not None:
            self.edge_by_relation.get(relation_id, set()).discard(key)
        self.edge_by_timestamp = [item for item in self.edge_by_timestamp if item[1] != key]
        with self.meta_lock:
            self._persist_meta()

    def reset(self) -> None:
        self.node_index.clear()
        self.edge_index.clear()
        self.edge_by_source.clear()
        self.edge_by_target.clear()
        self.edge_by_relation.clear()
        self.edge_by_timestamp.clear()
        self.node_shard_index.clear()
        self.edge_shard_index.clear()
        self.node_shard_sizes.clear()
        self.edge_shard_sizes.clear()
        self.relation_registry.clear()
        for name in os.listdir(self.base_path):
            if name.startswith("nodes_") or name.startswith("edges_") or name == "graph_meta.bin":
                os.remove(os.path.join(self.base_path, name))

    def get_edge(self, source: int, target: int, relation: str) -> Optional[Edge]:
        key = self.edge_key(source, target, relation)
        item = self.edge_index.get(key)
        if item is None:
            return None
        shard_id, offset = item
        payload = self._read_record(self._edges_path(shard_id), offset)
        return self._unpack_edge(payload)

    def iter_nodes(self) -> Iterable[Node]:
        for node_id in list(self.node_index.keys()):
            node = self.get_node(node_id)
            if node is not None:
                yield node

    def iter_edges(self) -> Iterable[Edge]:
        for key in list(self.edge_index.keys()):
            parts = key.split(":")
            if len(parts) != 3:
                continue
            relation_id = int(parts[2])
            relation = _reverse_lookup(self.relation_registry, relation_id)
            edge = self.get_edge(int(parts[0]), int(parts[1]), relation)
            if edge is not None:
                yield edge

    def edge_key(self, source: int, target: int, relation: str) -> str:
        relation_id = self.relation_registry[relation]
        return f"{source}:{target}:{relation_id}"

    def _update_edge_indexes(self, edge: Edge, key: str) -> None:
        relation_id = self.relation_registry[edge.relation]
        self.edge_by_source.get(edge.source, set()).discard(key)
        self.edge_by_target.get(edge.target, set()).discard(key)
        self.edge_by_relation.get(relation_id, set()).discard(key)
        self.edge_by_timestamp = [item for item in self.edge_by_timestamp if item[1] != key]
        self.edge_by_source[edge.source].add(key)
        self.edge_by_target[edge.target].add(key)
        self.edge_by_relation[relation_id].add(key)
        self.edge_by_timestamp.append((edge.last_used, key))
        self.edge_by_timestamp.sort(key=lambda item: item[0])

    def _pack_node(self, node: Node) -> bytes:
        label_bytes = node.label.encode("utf-8")
        node_type_id = self.node_type_registry[node.node_type]
        if node.embedding.shape != (self._embedding_dim,):
            raise ShardCorruptedError("Embedding size mismatch")
        embedding_bytes = node.embedding.astype("int8").tobytes()
        header = struct.pack("<QIB", node.node_id, len(label_bytes), node_type_id)
        tail = struct.pack("<fId", node.activation, node.use_count, node.create_time)
        return header + label_bytes + embedding_bytes + tail

    def _unpack_node(self, payload: bytes) -> Node:
        try:
            node_id, label_len, node_type_id = struct.unpack("<QIB", payload[:13])
            label_start = 13
            label_end = label_start + label_len
            label = payload[label_start:label_end].decode("utf-8")
            emb_start = label_end
            emb_end = emb_start + self._embedding_dim
            embedding = memoryview(payload[emb_start:emb_end]).tobytes()
            embedding_array = _bytes_to_int8_array(embedding)
            activation, use_count, create_time = struct.unpack("<fId", payload[emb_end:emb_end + 16])
        except Exception as exc:
            raise ShardCorruptedError("Node record corrupted") from exc
        node_type = _reverse_lookup(self.node_type_registry, node_type_id)
        return Node(
            node_id=node_id,
            label=label,
            node_type=node_type,
            embedding=embedding_array,
            activation=activation,
            use_count=use_count,
            create_time=create_time,
        )

    def _pack_edge(self, edge: Edge) -> bytes:
        relation_id = self.relation_registry[edge.relation]
        header = struct.pack("<qqB", edge.source, edge.target, relation_id)
        tail = struct.pack("<ffdI", edge.strength, edge.confidence, edge.last_used, edge.frequency)
        return header + tail

    def _unpack_edge(self, payload: bytes) -> Edge:
        try:
            source, target, relation_id = struct.unpack("<qqB", payload[:17])
            strength, confidence, last_used, frequency = struct.unpack("<ffdI", payload[17:37])
        except Exception as exc:
            raise ShardCorruptedError("Edge record corrupted") from exc
        relation = _reverse_lookup(self.relation_registry, relation_id)
        return Edge(
            source=source,
            target=target,
            relation=relation,
            strength=strength,
            confidence=confidence,
            last_used=last_used,
            frequency=frequency,
        )


def _reverse_lookup(mapping: Dict[str, int], value: int) -> str:
    for key, item in mapping.items():
        if item == value:
            return key
    return ""


def _bytes_to_int8_array(data: bytes) -> "numpy.ndarray":
    import numpy as np

    return np.frombuffer(data, dtype=np.int8).copy()


class MemoryStore:
    def __init__(self, node_type_registry: Dict[str, int], embedding_dim: int = 32) -> None:
        self.node_type_registry = node_type_registry
        self._embedding_dim = embedding_dim
        self.relation_registry: Dict[str, int] = {}
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.edge_by_source: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_target: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_relation: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_timestamp: List[Tuple[float, str]] = []

    def add_node(self, node: Node) -> bool:
        if node.node_id in self.nodes:
            return False
        self.nodes[node.node_id] = node
        return True

    def update_node(self, node: Node) -> bool:
        if node.node_id not in self.nodes:
            return False
        self.nodes[node.node_id] = node
        return True

    def get_node(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)

    def add_edge(self, edge: Edge) -> bool:
        key = self.edge_key(edge.source, edge.target, edge.relation)
        if key in self.edges:
            return self.update_edge(edge)
        self.edges[key] = edge
        self._update_edge_indexes(edge, key)
        return True

    def update_edge(self, edge: Edge) -> bool:
        key = self.edge_key(edge.source, edge.target, edge.relation)
        if key not in self.edges:
            return False
        self.edges[key] = edge
        self._update_edge_indexes(edge, key)
        return True

    def get_edge(self, source: int, target: int, relation: str) -> Optional[Edge]:
        key = self.edge_key(source, target, relation)
        return self.edges.get(key)

    def remove_node(self, node_id: int) -> None:
        self.nodes.pop(node_id, None)

    def remove_edge(self, source: int, target: int, relation: str) -> None:
        key = self.edge_key(source, target, relation)
        relation_id = self.relation_registry.get(relation)
        self.edges.pop(key, None)
        self.edge_by_source.get(source, set()).discard(key)
        self.edge_by_target.get(target, set()).discard(key)
        if relation_id is not None:
            self.edge_by_relation.get(relation_id, set()).discard(key)
        self.edge_by_timestamp = [item for item in self.edge_by_timestamp if item[1] != key]

    def reset(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.edge_by_source.clear()
        self.edge_by_target.clear()
        self.edge_by_relation.clear()
        self.edge_by_timestamp.clear()
        self.relation_registry.clear()

    def iter_nodes(self) -> Iterable[Node]:
        return list(self.nodes.values())

    def iter_edges(self) -> Iterable[Edge]:
        return list(self.edges.values())

    def edge_key(self, source: int, target: int, relation: str) -> str:
        relation_id = self.relation_registry[relation]
        return f"{source}:{target}:{relation_id}"

    def _update_edge_indexes(self, edge: Edge, key: str) -> None:
        relation_id = self.relation_registry[edge.relation]
        self.edge_by_source.get(edge.source, set()).discard(key)
        self.edge_by_target.get(edge.target, set()).discard(key)
        self.edge_by_relation.get(relation_id, set()).discard(key)
        self.edge_by_timestamp = [item for item in self.edge_by_timestamp if item[1] != key]
        self.edge_by_source[edge.source].add(key)
        self.edge_by_target[edge.target].add(key)
        self.edge_by_relation[relation_id].add(key)
        self.edge_by_timestamp.append((edge.last_used, key))
        self.edge_by_timestamp.sort(key=lambda item: item[0])


class LmdbStore:
    def __init__(
        self,
        base_path: str,
        serializer: GraphSerializer,
        node_type_registry: Dict[str, int],
        relation_registry: Optional[Dict[str, int]] = None,
        embedding_dim: int = 32,
        lmdb_map_size_gb: int = 4,
    ) -> None:
        try:
            import lmdb
        except Exception as exc:
            raise ShardCorruptedError("lmdb backend requires lmdb package") from exc
        self._lmdb = lmdb
        self.base_path = base_path
        self.serializer = serializer
        self.node_type_registry = node_type_registry
        self.relation_registry = relation_registry or {}
        self._embedding_dim = embedding_dim
        os.makedirs(self.base_path, exist_ok=True)
        self.env = lmdb.open(self.base_path, max_dbs=5, map_size=lmdb_map_size_gb * 1024 * 1024 * 1024)
        self.nodes_db = self.env.open_db(b"nodes")
        self.edges_db = self.env.open_db(b"edges")
        self.meta_db = self.env.open_db(b"meta")
        self.edge_by_source: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_target: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_relation: Dict[int, set[str]] = defaultdict(set)
        self.edge_by_timestamp: List[Tuple[float, str]] = []
        self._load_meta()

    def _load_meta(self) -> None:
        with self.env.begin(db=self.meta_db) as txn:
            raw = txn.get(b"meta")
        if not raw:
            return
        meta = self.serializer.deserialize(raw)
        self.edge_by_source = defaultdict(set, meta.get("edge_by_source", {}))
        self.edge_by_target = defaultdict(set, meta.get("edge_by_target", {}))
        self.edge_by_relation = defaultdict(set, meta.get("edge_by_relation", {}))
        self.edge_by_timestamp = meta.get("edge_by_timestamp", [])
        self.relation_registry = meta.get("relation_registry", self.relation_registry)

    def _persist_meta(self) -> None:
        meta = {
            "edge_by_source": dict(self.edge_by_source),
            "edge_by_target": dict(self.edge_by_target),
            "edge_by_relation": dict(self.edge_by_relation),
            "edge_by_timestamp": self.edge_by_timestamp,
            "relation_registry": self.relation_registry,
        }
        payload = self.serializer.serialize(meta)
        with self.env.begin(db=self.meta_db, write=True) as txn:
            txn.put(b"meta", payload)

    def add_node(self, node: Node) -> bool:
        key = str(node.node_id).encode("utf-8")
        payload = self.serializer.serialize(node)
        with self.env.begin(db=self.nodes_db, write=True) as txn:
            if txn.get(key) is not None:
                return False
            txn.put(key, payload)
        return True

    def update_node(self, node: Node) -> bool:
        key = str(node.node_id).encode("utf-8")
        payload = self.serializer.serialize(node)
        with self.env.begin(db=self.nodes_db, write=True) as txn:
            if txn.get(key) is None:
                return False
            txn.put(key, payload)
        return True

    def get_node(self, node_id: int) -> Optional[Node]:
        key = str(node_id).encode("utf-8")
        with self.env.begin(db=self.nodes_db) as txn:
            raw = txn.get(key)
        if not raw:
            return None
        return self.serializer.deserialize(raw)

    def add_edge(self, edge: Edge) -> bool:
        key = self.edge_key(edge.source, edge.target, edge.relation)
        payload = self.serializer.serialize(edge)
        with self.env.begin(db=self.edges_db, write=True) as txn:
            if txn.get(key.encode("utf-8")) is not None:
                return self.update_edge(edge)
            txn.put(key.encode("utf-8"), payload)
        self._update_edge_indexes(edge, key)
        self._persist_meta()
        return True

    def update_edge(self, edge: Edge) -> bool:
        key = self.edge_key(edge.source, edge.target, edge.relation)
        payload = self.serializer.serialize(edge)
        with self.env.begin(db=self.edges_db, write=True) as txn:
            if txn.get(key.encode("utf-8")) is None:
                return False
            txn.put(key.encode("utf-8"), payload)
        self._update_edge_indexes(edge, key)
        self._persist_meta()
        return True

    def get_edge(self, source: int, target: int, relation: str) -> Optional[Edge]:
        key = self.edge_key(source, target, relation).encode("utf-8")
        with self.env.begin(db=self.edges_db) as txn:
            raw = txn.get(key)
        if not raw:
            return None
        return self.serializer.deserialize(raw)

    def remove_node(self, node_id: int) -> None:
        key = str(node_id).encode("utf-8")
        with self.env.begin(db=self.nodes_db, write=True) as txn:
            txn.delete(key)

    def remove_edge(self, source: int, target: int, relation: str) -> None:
        key = self.edge_key(source, target, relation)
        relation_id = self.relation_registry.get(relation)
        with self.env.begin(db=self.edges_db, write=True) as txn:
            txn.delete(key.encode("utf-8"))
        self.edge_by_source.get(source, set()).discard(key)
        self.edge_by_target.get(target, set()).discard(key)
        if relation_id is not None:
            self.edge_by_relation.get(relation_id, set()).discard(key)
        self.edge_by_timestamp = [item for item in self.edge_by_timestamp if item[1] != key]
        self._persist_meta()

    def reset(self) -> None:
        with self.env.begin(write=True) as txn:
            txn.drop(self.nodes_db, delete=False)
            txn.drop(self.edges_db, delete=False)
        self.edge_by_source.clear()
        self.edge_by_target.clear()
        self.edge_by_relation.clear()
        self.edge_by_timestamp.clear()
        self.relation_registry.clear()
        self._persist_meta()

    def iter_nodes(self) -> Iterable[Node]:
        with self.env.begin(db=self.nodes_db) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield self.serializer.deserialize(value)

    def iter_edges(self) -> Iterable[Edge]:
        with self.env.begin(db=self.edges_db) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield self.serializer.deserialize(value)

    def edge_key(self, source: int, target: int, relation: str) -> str:
        relation_id = self.relation_registry[relation]
        return f"{source}:{target}:{relation_id}"

    def _update_edge_indexes(self, edge: Edge, key: str) -> None:
        relation_id = self.relation_registry[edge.relation]
        self.edge_by_source.get(edge.source, set()).discard(key)
        self.edge_by_target.get(edge.target, set()).discard(key)
        self.edge_by_relation.get(relation_id, set()).discard(key)
        self.edge_by_timestamp = [item for item in self.edge_by_timestamp if item[1] != key]
        self.edge_by_source[edge.source].add(key)
        self.edge_by_target[edge.target].add(key)
        self.edge_by_relation[relation_id].add(key)
        self.edge_by_timestamp.append((edge.last_used, key))
        self.edge_by_timestamp.sort(key=lambda item: item[0])
