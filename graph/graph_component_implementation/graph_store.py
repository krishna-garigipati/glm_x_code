from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple
import logging
import os
import threading
import time

import numpy as np

from .cache import build_cache
from .config_loader import load_config
from .errors import (
    EdgeNotFoundError,
    InvalidEmbeddingDimensionError,
    NodeNotFoundError,
)
from .models import Edge, Node, Subgraph
from .prefetch import MarkovPrefetcher
from .serializer import GraphSerializer
from .storage import LmdbStore, MemoryStore, ShardedDiskStore
from .utils import (
    cosine_similarity,
    dequantize_embedding,
    quantize_embedding,
    validate_label,
    validate_node_type,
)


class GraphStore:
    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config_graph.yaml")
        self.config = load_config(config_path)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
        storage_cfg = self.config["storage"]
        cache_cfg = self.config["cache"]
        node_cfg = self.config["node"]
        edge_cfg = self.config["edge"]
        serialization_cfg = self.config["serialization"]
        backup_cfg = self.config.get("backup", {})
        self._node_types = node_cfg["node_types"]
        self._label_max_length = node_cfg["label_max_length"]
        fmt = node_cfg.get("attributes_binary_format", {}).get("embedding", "32b")
        self._embedding_dim = node_cfg.get("embedding_dim", int(fmt.rstrip("b")))
        self._embedding_dtype = node_cfg["embedding_dtype"]
        self._embedding_logical_range = node_cfg["embedding_logical_range"]
        self._activation_dtype = node_cfg["activation_dtype"]
        self._activation_min = float(np.float32(node_cfg.get("activation_min", 0.01)))
        self._activation_max = float(np.float32(node_cfg.get("activation_max", 1.0)))
        self._use_count_max = node_cfg["use_count_max"]
        self._relation_max = edge_cfg.get("relation_max", 256)
        self._prefetch_top_k = cache_cfg.get("prefetch_top_k", 3)
        self._lmdb_map_size_gb = storage_cfg.get("lmdb_map_size_gb", 4)
        base_path = storage_cfg["base_path"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        self.serializer = GraphSerializer(
            compression=serialization_cfg["compression"],
            compression_level=serialization_cfg["compression_level"],
        )
        self._node_type_registry = {name: idx for idx, name in enumerate(self._node_types)}
        backend = storage_cfg["backend"]
        if backend == "sharded_disk":
            self.storage = ShardedDiskStore(
                base_path=base_path,
                shard_prefix=storage_cfg.get("shard_prefix", "shard_"),
                shard_size_mb=storage_cfg.get("shard_size_mb", 100),
                max_shards=storage_cfg.get("max_shards", 1000),
                serializer=self.serializer,
                node_type_registry=self._node_type_registry,
                embedding_dim=self._embedding_dim,
            )
        elif backend == "memory_only":
            self.storage = MemoryStore(self._node_type_registry, embedding_dim=self._embedding_dim)
        elif backend == "lmdb":
            self.storage = LmdbStore(
                base_path=base_path,
                serializer=self.serializer,
                node_type_registry=self._node_type_registry,
                embedding_dim=self._embedding_dim,
                lmdb_map_size_gb=self._lmdb_map_size_gb,
            )
        else:
            raise ValueError(f"Unknown storage backend: {backend}")
        self._relation_registry = self.storage.relation_registry
        self.node_cache = build_cache(cache_cfg["type"], cache_cfg["ram_limit_mb"])
        self.edge_cache = build_cache(cache_cfg["type"], cache_cfg["ram_limit_mb"])
        self.prefetcher = MarkovPrefetcher(
            order=cache_cfg["prefetch_markov_order"],
            threads=cache_cfg["prefetch_threads"],
            fetch_fn=self._prefetch_node,
            top_k=self._prefetch_top_k,
        )
        self._backup_enabled = backup_cfg.get("enabled", False)
        self._backup_interval = backup_cfg.get("interval_seconds", 3600)
        self._backup_keep = backup_cfg.get("keep_last_n", 24)
        self._backup_base_dir = base_path
        self._backup_thread = None
        if self._backup_enabled:
            self._start_backup_thread()

    def add_node(
        self,
        node_id: int,
        label: str,
        node_type: str,
        embedding: np.ndarray,
        activation: float = 0.01,
    ) -> bool:
        with self._lock:
            validate_label(label, self._label_max_length)
            validate_node_type(node_type, self._node_types)
            quantized = self._validate_embedding(embedding)
            clamped = max(self._activation_min, min(self._activation_max, float(activation)))
            node = Node(
                node_id=node_id,
                label=label,
                node_type=node_type,
                embedding=quantized,
                activation=clamped,
                use_count=0,
                create_time=time.time(),
            )
            added = self.storage.add_node(node)
            if added:
                self.node_cache.put(node_id, node)
                self.prefetcher.record(node_id)
            self._logger.debug("add_node node_id=%s added=%s", node_id, added)
            return added

    def get_node(self, node_id: int) -> Optional[Node]:
        with self._lock:
            cached = self.node_cache.get(node_id)
            if cached is not None:
                node = cached
            else:
                node = self.storage.get_node(node_id)
                if node is not None:
                    self.node_cache.put(node_id, node)
            if node is None:
                self._logger.debug("get_node node_id=%s hit=%s", node_id, False)
                return None
            if node.use_count < self._use_count_max:
                updated = Node(
                    node_id=node.node_id,
                    label=node.label,
                    node_type=node.node_type,
                    embedding=node.embedding,
                    activation=node.activation,
                    use_count=min(self._use_count_max, node.use_count + 1),
                    create_time=node.create_time,
                )
                self.storage.update_node(updated)
                self.node_cache.put(node_id, updated)
                node = updated
            self.prefetcher.record(node_id)
            self._logger.debug("get_node node_id=%s hit=%s", node_id, True)
            return node

    def update_node_embedding(self, node_id: int, embedding: np.ndarray) -> bool:
        with self._lock:
            quantized = self._validate_embedding(embedding)
            node = self.storage.get_node(node_id)
            if node is None:
                raise NodeNotFoundError()
            updated = Node(
                node_id=node.node_id,
                label=node.label,
                node_type=node.node_type,
                embedding=quantized,
                activation=node.activation,
                use_count=node.use_count,
                create_time=node.create_time,
            )
            ok = self.storage.update_node(updated)
            if ok:
                self.node_cache.put(node_id, updated)
            self._logger.debug("update_node_embedding node_id=%s updated=%s", node_id, ok)
            return ok

    def add_edge(
        self,
        source: int,
        target: int,
        relation: str,
        strength: float = 0.5,
        confidence: float = 0.5,
    ) -> bool:
        with self._lock:
            if relation not in self._relation_registry:
                if len(self._relation_registry) >= self._relation_max:
                    raise ValueError("Relation registry exhausted")
                self._relation_registry[relation] = len(self._relation_registry)
            edge = Edge(
                source=source,
                target=target,
                relation=relation,
                strength=float(strength),
                confidence=float(confidence),
                last_used=time.time(),
                frequency=1,
            )
            added = self.storage.add_edge(edge)
            if added:
                key = self.storage.edge_key(source, target, relation)
                self.edge_cache.put(key, edge)
            self._logger.debug("add_edge %s->%s %s added=%s", source, target, relation, added)
            return added

    def get_edge(self, source: int, target: int, relation: str) -> Optional[Edge]:
        with self._lock:
            key = self.storage.edge_key(source, target, relation)
            cached = self.edge_cache.get(key)
            if cached is not None:
                return cached
            edge = self.storage.get_edge(source, target, relation)
            if edge is not None:
                self.edge_cache.put(key, edge)
            self._logger.debug("get_edge %s->%s %s hit=%s", source, target, relation, edge is not None)
            return edge

    def update_edge_weights(self, updates: Dict[Tuple[int, int, str], Tuple[float, float]]) -> None:
        with self._lock:
            for (source, target, relation), (strength, confidence) in updates.items():
                edge = self.storage.get_edge(source, target, relation)
                if edge is None:
                    raise EdgeNotFoundError()
                updated = Edge(
                    source=edge.source,
                    target=edge.target,
                    relation=edge.relation,
                    strength=float(strength),
                    confidence=float(confidence),
                    last_used=time.time(),
                    frequency=edge.frequency + 1,
                )
                self.storage.update_edge(updated)
                key = self.storage.edge_key(source, target, relation)
                self.edge_cache.put(key, updated)

    def get_neighbors(
        self,
        node_id: int,
        relation_filter: Optional[List[str]] = None,
    ) -> List[Tuple[int, Edge]]:
        with self._lock:
            neighbors: List[Tuple[int, Edge]] = []
            keys = set()
            keys.update(self.storage.edge_by_source.get(node_id, set()))
            keys.update(self.storage.edge_by_target.get(node_id, set()))
            for key in keys:
                parts = key.split(":")
                if len(parts) != 3:
                    continue
                source = int(parts[0])
                target = int(parts[1])
                relation_id = int(parts[2])
                relation = self._reverse_lookup(self._relation_registry, relation_id)
                if relation_filter and relation not in relation_filter:
                    continue
                edge = self.get_edge(source, target, relation)
                if edge is None:
                    continue
                neighbor_id = target if source == node_id else source
                neighbors.append((neighbor_id, edge))
            return neighbors

    def get_subgraph_activated(self, seed_nodes: List[int], max_nodes: int = 1000) -> Subgraph:
        with self._lock:
            visited = set()
            queue = deque(seed_nodes)
            nodes: Dict[int, Node] = {}
            edges: List[Edge] = []
            while queue and len(nodes) < max_nodes:
                node_id = queue.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)
                node = self.get_node(node_id)
                if node is None:
                    continue
                if node.activation < self._activation_min:
                    continue
                nodes[node_id] = node
                for neighbor_id, edge in self.get_neighbors(node_id):
                    edges.append(edge)
                    if neighbor_id not in visited and len(nodes) < max_nodes:
                        queue.append(neighbor_id)
            return Subgraph(nodes=nodes, edges=edges)

    def get_subgraph_by_embedding_similarity(self, query_embedding: np.ndarray, top_k: int = 100) -> Subgraph:
        with self._lock:
            quantized = self._validate_embedding(query_embedding)
            logical_min, logical_max = self._embedding_logical_range
            query_float = dequantize_embedding(
                quantized,
                embedding_dim=self._embedding_dim,
                logical_min=logical_min,
                logical_max=logical_max,
            )
            scored: List[Tuple[float, Node]] = []
            for node in self.storage.iter_nodes():
                node_float = dequantize_embedding(
                    node.embedding,
                    embedding_dim=self._embedding_dim,
                    logical_min=logical_min,
                    logical_max=logical_max,
                )
                score = cosine_similarity(query_float, node_float)
                scored.append((score, node))
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = scored[:top_k]
            nodes = {node.node_id: node for _, node in selected}
            edges: List[Edge] = []
            for node_id in nodes.keys():
                for _, edge in self.get_neighbors(node_id):
                    if edge.source in nodes and edge.target in nodes:
                        edges.append(edge)
            return Subgraph(nodes=nodes, edges=edges)

    def prune(self, utility_threshold: float = 0.01) -> int:
        with self._lock:
            removed = 0
            node_ids = {node.node_id for node in self.storage.iter_nodes()}
            edges_to_remove: List[Tuple[int, int, str]] = []
            for edge in self.storage.iter_edges():
                if edge.source not in node_ids or edge.target not in node_ids:
                    edges_to_remove.append((edge.source, edge.target, edge.relation))
            for source, target, relation in edges_to_remove:
                self.storage.remove_edge(source, target, relation)
            removed_nodes: List[int] = []
            for node in list(self.storage.iter_nodes()):
                if node.activation < utility_threshold - 1e-6:
                    self.storage.remove_node(node.node_id)
                    removed += 1
                    removed_nodes.append(node.node_id)
            if removed_nodes:
                for edge in self.storage.iter_edges():
                    if edge.source in removed_nodes or edge.target in removed_nodes:
                        self.storage.remove_edge(edge.source, edge.target, edge.relation)
            if removed:
                self.node_cache.clear()
                self.edge_cache.clear()
            self._logger.debug("prune removed=%s", removed)
            return removed

    def save_checkpoint(self, filepath: str) -> bool:
        with self._lock:
            snapshot = {
                "nodes": list(self.storage.iter_nodes()),
                "edges": list(self.storage.iter_edges()),
                "relation_registry": self._relation_registry,
            }
            payload = self.serializer.serialize(snapshot)
            with open(filepath, "wb") as file_handle:
                file_handle.write(payload)
            self._logger.info("save_checkpoint filepath=%s", filepath)
            return True

    def load_checkpoint(self, filepath: str) -> bool:
        with self._lock:
            with open(filepath, "rb") as file_handle:
                payload = file_handle.read()
            snapshot = self.serializer.deserialize(payload)
            nodes = snapshot.get("nodes", [])
            edges = snapshot.get("edges", [])
            self.storage.reset()
            self._relation_registry.clear()
            self._relation_registry.update(snapshot.get("relation_registry", {}))
            for node in nodes:
                self.storage.add_node(node)
            for edge in edges:
                self.storage.add_edge(edge)
            self._logger.info("load_checkpoint filepath=%s", filepath)
            return True

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        if embedding.shape != (self._embedding_dim,):
            raise InvalidEmbeddingDimensionError()
        if self._embedding_dtype != "int8":
            raise InvalidEmbeddingDimensionError("Unsupported embedding dtype")
        logical_min, logical_max = self._embedding_logical_range
        return quantize_embedding(
            embedding,
            embedding_dim=self._embedding_dim,
            logical_min=logical_min,
            logical_max=logical_max,
        )

    def _prefetch_node(self, node_id: int) -> None:
        self.get_node(node_id)

    def close(self) -> None:
        self.prefetcher.shutdown()
        self._backup_enabled = False

    def _start_backup_thread(self) -> None:
        self._backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self._backup_thread.start()

    def _backup_loop(self) -> None:
        base_path = self._backup_base_dir
        while True:
            time.sleep(self._backup_interval)
            timestamp = int(time.time())
            backup_dir = os.path.join(base_path, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f"backup_{timestamp}.bin")
            self.save_checkpoint(backup_path)
            backups = sorted(
                [name for name in os.listdir(backup_dir) if name.startswith("backup_")]
            )
            while len(backups) > self._backup_keep:
                to_remove = backups.pop(0)
                os.remove(os.path.join(backup_dir, to_remove))

    @staticmethod
    def _reverse_lookup(mapping: Dict[str, int], value: int) -> str:
        for key, item in mapping.items():
            if item == value:
                return key
        return ""
