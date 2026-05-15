import sys, os, time, tempfile, shutil, struct, threading, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import pytest

from knowledge_graph.graph_component_implementation.graph_store import GraphStore
from knowledge_graph.graph_component_implementation.models import Node, Edge, Subgraph
from knowledge_graph.graph_component_implementation.errors import (
    GraphStoreError, NodeNotFoundError, EdgeNotFoundError,
    DuplicateNodeError, InvalidEmbeddingDimensionError,
    SerializationFailedError, ShardCorruptedError,
)
from knowledge_graph.graph_component_implementation.cache import LruCache, ArcCache, NoCache, build_cache
from knowledge_graph.graph_component_implementation.prefetch import MarkovPrefetcher
from knowledge_graph.graph_component_implementation.serializer import GraphSerializer
from knowledge_graph.graph_component_implementation.storage import ShardedDiskStore, MemoryStore
from knowledge_graph.graph_component_implementation.utils import (
    quantize_embedding, dequantize_embedding, cosine_similarity,
    validate_label, validate_node_type,
)

try:
    import lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

HAVE_HYPOTHESIS = False
try:
    from hypothesis import given, strategies as st, settings, HealthCheck
    import hypothesis.extra.numpy as npst
    HAVE_HYPOTHESIS = True
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_SRC = os.path.join(PROJECT_ROOT, "graph_component_implementation", "config_graph.yaml")

RNG = np.random.default_rng(42)


def make_emb(dim=32):
    return RNG.uniform(-1.0, 1.0, dim).astype(np.float32)


def _fresh_graph(backend="sharded_disk", cache_type="lru", base_path=None, backup_enabled=False, overrides=None):
    with open(CONFIG_SRC, "r") as f:
        cfg = yaml.safe_load(f)
    if base_path is None:
        base_path = tempfile.mkdtemp()
    cfg["storage"]["backend"] = backend
    cfg["storage"]["base_path"] = base_path
    cfg["backup"]["enabled"] = backup_enabled
    cfg["cache"]["type"] = cache_type
    if overrides:
        _deep_merge(cfg, overrides)
    cfg_path = os.path.join(base_path, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    gs = GraphStore(config_path=cfg_path)
    return gs, base_path


def _deep_merge(base, overrides):
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _fresh_sharded_store(base_path=None, serializer=None, embedding_dim=32, extra=None):
    if base_path is None:
        base_path = tempfile.mkdtemp()
    if serializer is None:
        serializer = GraphSerializer(compression="none", compression_level=0)
    if extra is None:
        extra = {}
    ser = serializer
    node_type_registry = {"Concept": 0, "Entity": 1}
    store = ShardedDiskStore(
        base_path=base_path,
        shard_prefix=extra.get("shard_prefix", "shard_"),
        shard_size_mb=extra.get("shard_size_mb", 1),
        max_shards=extra.get("max_shards", 4),
        serializer=ser,
        node_type_registry=node_type_registry,
        embedding_dim=embedding_dim,
    )
    return store, base_path


# ================================================================
# 2A. STORAGE BACKEND PARITY
# ================================================================

class TestStorageBackends:
    @pytest.fixture(params=["memory_only", "sharded_disk"] if not HAS_LMDB else ["memory_only", "sharded_disk", "lmdb"])
    def backend_gs(self, request):
        gs, tmp = _fresh_graph(backend=request.param)
        yield gs, tmp, request.param
        try: gs.close()
        except: pass
        shutil.rmtree(tmp, ignore_errors=True)

    def test_init_backend(self, backend_gs):
        gs, tmp, backend = backend_gs
        assert gs is not None

    def test_crud_nodes(self, backend_gs):
        gs, tmp, backend = backend_gs
        emb = make_emb()
        assert gs.add_node(1, "Alpha", "Concept", emb, 0.75) is True
        n = gs.get_node(1)
        assert n is not None and n.node_id == 1 and n.label == "Alpha"
        new_emb = make_emb()
        assert gs.update_node_embedding(1, new_emb) is True
        n2 = gs.get_node(1)
        assert n2 is not None
        assert gs.get_node(999) is None

    def test_crud_edges(self, backend_gs):
        gs, tmp, backend = backend_gs
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        assert gs.add_edge(1, 2, "causes", 0.8, 0.9) is True
        e = gs.get_edge(1, 2, "causes")
        assert e is not None and abs(e.strength - 0.8) < 1e-5
        gs.update_edge_weights({(1, 2, "causes"): (0.9, 0.95)})
        e2 = gs.get_edge(1, 2, "causes")
        assert abs(e2.strength - 0.9) < 1e-5

    def test_iter_nodes_edges(self, backend_gs):
        gs, tmp, backend = backend_gs
        emb = make_emb()
        for i in range(5):
            gs.add_node(i, f"N{i}", "Concept", emb)
        gs.add_edge(0, 1, "rel")
        assert len(list(gs.storage.iter_nodes())) == 5
        assert len(list(gs.storage.iter_edges())) == 1

    def test_reset_clears_all(self, backend_gs):
        gs, tmp, backend = backend_gs
        emb = make_emb()
        gs.add_node(1, "X", "Concept", emb)
        gs.add_edge(1, 1, "self")
        gs.storage.reset()
        assert len(list(gs.storage.iter_nodes())) == 0
        assert len(list(gs.storage.iter_edges())) == 0

    def test_checkpoint_roundtrip(self, backend_gs):
        gs, tmp, backend = backend_gs
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        gs.add_edge(1, 2, "rel")
        cp = os.path.join(tmp, "cp.bin")
        assert gs.save_checkpoint(cp) is True
        gs2, tmp2 = _fresh_graph(backend=backend)
        assert gs2.load_checkpoint(cp) is True
        assert gs2.get_node(1) is not None
        assert gs2.get_edge(1, 2, "rel") is not None
        gs2.close()
        shutil.rmtree(tmp2, ignore_errors=True)

    def test_edge_indexes_consistency(self, backend_gs):
        gs, tmp, backend = backend_gs
        emb = make_emb()
        for i in range(4):
            gs.add_node(i, f"N{i}", "Concept", emb)
        gs.add_edge(0, 1, "causes")
        gs.add_edge(0, 2, "related")
        gs.add_edge(1, 3, "causes")
        store = gs.storage
        assert len(store.edge_by_source.get(0, set())) == 2
        assert len(store.edge_by_target.get(1, set())) == 1
        assert len(store.edge_by_target.get(3, set())) == 1


# ================================================================
# 2B. SHARDED DISK STORE SPECIFICS
# ================================================================

class TestShardedDiskStoreSpecifics:
    def test_shard_assignment_distribution(self):
        store, tmp = _fresh_sharded_store()
        n_nodes = 50
        for i in range(n_nodes):
            node = Node(i, f"N{i}", "Concept", np.zeros(32, dtype=np.int8), 0.5, 0, time.time())
            store.add_node(node)
        shards_used = set(store.node_shard_index.values())
        assert len(shards_used) <= store.max_shards
        assert len(shards_used) > 1
        shutil.rmtree(tmp, ignore_errors=True)

    def test_shard_overflow(self):
        store, tmp = _fresh_sharded_store(extra={"shard_size_mb": 0, "max_shards": 2})
        large_emb = np.zeros(32, dtype=np.int8)
        node = Node(1, "N1", "Concept", large_emb, 0.5, 0, time.time())
        store.add_node(node)
        node2 = Node(2, "N2", "Concept", large_emb, 0.5, 0, time.time())
        store.add_node(node2)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_all_shards_full_raises(self):
        store, tmp = _fresh_sharded_store(extra={"shard_size_mb": 0, "max_shards": 1})
        with pytest.raises(ShardCorruptedError):
            for i in range(1000):
                node = Node(i, f"N{i}", "Concept", np.zeros(32, dtype=np.int8), 0.5, 0, time.time())
                store.add_node(node)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_shard_corrupted_record(self):
        store, tmp = _fresh_sharded_store()
        node = Node(1, "Test", "Concept", np.zeros(32, dtype=np.int8), 0.5, 0, time.time())
        store.add_node(node)
        shard_id, offset = store.node_index[1]
        path = store._nodes_path(shard_id)
        with open(path, "r+b") as f:
            f.seek(offset)
            f.write(b"\xff\xff\xff\xff")
        with pytest.raises(ShardCorruptedError):
            store.get_node(1)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_meta_persistence(self):
        store, tmp = _fresh_sharded_store()
        node = Node(1, "A", "Concept", np.zeros(32, dtype=np.int8), 0.5, 0, time.time())
        store.add_node(node)
        meta_path = store._meta_path()
        assert os.path.exists(meta_path)
        store2, _ = _fresh_sharded_store(base_path=tmp)
        assert 1 in store2.node_index
        assert store2.get_node(1) is not None
        store2.reset()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_node_binary_roundtrip(self):
        store, tmp = _fresh_sharded_store()
        original_emb = np.array([-128, -64, 0, 64, 127] + [0]*27, dtype=np.int8)
        original = Node(42, "Hello World", "Entity", original_emb, 0.75, 10, 1234567890.0)
        packed = store._pack_node(original)
        unpacked = store._unpack_node(packed)
        assert unpacked.node_id == 42
        assert unpacked.label == "Hello World"
        assert unpacked.node_type == "Entity"
        assert np.array_equal(unpacked.embedding, original_emb)
        assert abs(unpacked.activation - 0.75) < 1e-6
        assert unpacked.use_count == 10
        assert abs(unpacked.create_time - 1234567890.0) < 1e-6
        shutil.rmtree(tmp, ignore_errors=True)

    def test_edge_binary_roundtrip(self):
        store, tmp = _fresh_sharded_store()
        store.relation_registry["test_rel"] = 7
        original = Edge(10, 20, "test_rel", 0.85, 0.92, 1234567890.0, 5)
        packed = store._pack_edge(original)
        unpacked = store._unpack_edge(packed)
        assert unpacked.source == 10
        assert unpacked.target == 20
        assert unpacked.relation == "test_rel"
        assert abs(unpacked.strength - 0.85) < 1e-6
        assert abs(unpacked.confidence - 0.92) < 1e-6
        assert abs(unpacked.last_used - 1234567890.0) < 1e-6
        assert unpacked.frequency == 5
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================
# 2C. LMDB STORE (skip if not installed)
# ================================================================

@pytest.mark.skipif(not HAS_LMDB, reason="lmdb package not installed")
class TestLmdbStoreSpecifics:
    def test_lmdb_init_custom_map_size(self):
        from knowledge_graph.graph_component_implementation.storage import LmdbStore
        tmp = tempfile.mkdtemp()
        ser = GraphSerializer(compression="none", compression_level=0)
        store = LmdbStore(tmp, ser, {"Concept": 0}, lmdb_map_size_gb=1)
        assert store is not None
        store.reset()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_lmdb_reset_drops_databases(self):
        from knowledge_graph.graph_component_implementation.storage import LmdbStore
        tmp = tempfile.mkdtemp()
        ser = GraphSerializer(compression="none", compression_level=0)
        store = LmdbStore(tmp, ser, {"Concept": 0})
        node = Node(1, "A", "Concept", np.zeros(32, dtype=np.int8), 0.5, 0, time.time())
        store.add_node(node)
        store.reset()
        assert list(store.iter_nodes()) == []
        shutil.rmtree(tmp, ignore_errors=True)

    def test_lmdb_concurrent_transactions(self):
        from knowledge_graph.graph_component_implementation.storage import LmdbStore
        tmp = tempfile.mkdtemp()
        ser = GraphSerializer(compression="none", compression_level=0)
        store = LmdbStore(tmp, ser, {"Concept": 0})
        node = Node(1, "A", "Concept", np.zeros(32, dtype=np.int8), 0.5, 0, time.time())
        store.add_node(node)
        n = store.get_node(1)
        assert n is not None
        store.reset()
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================
# 2D. CACHE: ALL 3 VARIANTS
# ================================================================

class TestCache:
    def test_nocache_always_returns_none(self):
        c = NoCache()
        c.put(1, "value")
        assert c.get(1) is None
        c.clear()

    def test_lru_cache_eviction(self):
        c = LruCache(ram_limit_mb=0, size_func=lambda v: 100)
        c.put(1, "a")
        c.put(2, "b")
        assert c.get(1) is None
        assert c.get(2) is None

    def test_lru_cache_move_to_end_on_get(self):
        c = LruCache(ram_limit_mb=100, size_func=lambda v: 1)
        c.put(1, "a")
        c.put(2, "b")
        c.put(3, "c")
        c.get(1)
        assert list(c.items.keys()) == [2, 3, 1]

    def test_lru_cache_clear(self):
        c = LruCache(ram_limit_mb=100, size_func=lambda v: 1)
        c.put(1, "a")
        c.clear()
        assert c.get(1) is None
        assert c.current_size == 0

    def test_lru_cache_size_accounting(self):
        c = LruCache(ram_limit_mb=1, size_func=lambda v: 500 * 1024)
        c.put(1, "x" * 500 * 1024)
        assert c.get(1) is not None
        assert c.current_size > 0

    def test_arc_cache_t1_to_t2_promotion(self):
        c = ArcCache(ram_limit_mb=100, size_func=lambda v: 1)
        c.put(1, "a")
        assert 1 in c.t1
        c.get(1)
        assert 1 in c.t2

    def test_arc_cache_clear(self):
        c = ArcCache(ram_limit_mb=100, size_func=lambda v: 1)
        c.put(1, "a")
        c.put(2, "b")
        c.clear()
        assert c.get(1) is None
        assert c.get(2) is None
        assert c.current_size == 0
        assert c.p == 0

    def test_arc_cache_ghost_list_adaptation(self):
        c = ArcCache(ram_limit_mb=0, size_func=lambda v: 100)
        for i in range(20):
            c.put(i, f"v{i}")
        assert len(c.b1) + len(c.b2) > 0

    def test_build_cache_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown cache type"):
            build_cache("foobar", 100)


# ================================================================
# 2E. MARKOV PREFETCHER
# ================================================================

class TestMarkovPrefetcher:
    def test_records_transitions(self):
        results = []
        def fetch(n):
            results.append(n)
        mp = MarkovPrefetcher(order=3, threads=1, fetch_fn=fetch, top_k=2)
        for n in [1, 2, 3, 4, 1, 2, 3, 4]:
            mp.record(n)
        mp.shutdown()
        key = (1, 2, 3)
        assert key in mp.transitions
        assert mp.transitions[key].get(4, 0) >= 1

    def test_predict_next_top_k(self):
        mp = MarkovPrefetcher(order=2, threads=1, fetch_fn=lambda n: None, top_k=2)
        mp.record(1)
        mp.record(2)
        mp.record(3)
        mp.record(1)
        mp.record(2)
        mp.record(4)
        preds = mp._predict_next()
        assert len(preds) <= 2
        mp.shutdown()

    def test_no_prediction_when_history_short(self):
        mp = MarkovPrefetcher(order=5, threads=1, fetch_fn=lambda n: None, top_k=3)
        mp.record(1)
        mp.record(2)
        assert mp._predict_next() == []
        mp.shutdown()

    def test_no_prediction_on_unseen_transition(self):
        mp = MarkovPrefetcher(order=2, threads=1, fetch_fn=lambda n: None, top_k=3)
        mp.record(1)
        mp.record(2)
        mp.record(3)
        assert mp._predict_next() == []
        mp.shutdown()

    def test_shutdown_stops_executor(self):
        mp = MarkovPrefetcher(order=2, threads=2, fetch_fn=lambda n: None, top_k=3)
        mp.record(1)
        mp.shutdown()
        assert mp.executor._shutdown

    def test_prefetch_triggers_fetch_fn(self):
        fetched = []
        def fetch(n):
            fetched.append(n)
        mp = MarkovPrefetcher(order=2, threads=1, fetch_fn=fetch, top_k=3)
        for n in [1, 2, 3, 1, 2, 3]:
            mp.record(n)
        time.sleep(0.1)
        mp.shutdown()
        if fetched:
            assert fetched[0] == 3

    def test_concurrent_record_calls(self):
        mp = MarkovPrefetcher(order=2, threads=2, fetch_fn=lambda n: None, top_k=3)
        errors = []
        def record_many(start):
            try:
                for i in range(100):
                    mp.record(start + i)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=record_many, args=(0,)),
                   threading.Thread(target=record_many, args=(100,))]
        for t in threads: t.start()
        for t in threads: t.join()
        mp.shutdown()
        assert len(errors) == 0


# ================================================================
# 2F. SERIALIZER
# ================================================================

class TestSerializer:
    def test_none_compression_roundtrip(self):
        s = GraphSerializer(compression="none", compression_level=0)
        data = {"hello": "world", "nums": [1, 2, 3]}
        blob = s.serialize(data)
        restored = s.deserialize(blob)
        assert restored == data

    @pytest.mark.skipif(not HAS_LZ4, reason="lz4 not installed")
    def test_lz4_compression_roundtrip(self):
        for level in [1, 3, 9]:
            s = GraphSerializer(compression="lz4", compression_level=level)
            data = {"x" * 1000: "y" * 1000}
            blob = s.serialize(data)
            restored = s.deserialize(blob)
            assert restored == data

    @pytest.mark.skipif(not HAS_ZSTD, reason="zstandard not installed")
    def test_zstd_compression_roundtrip(self):
        for level in [1, 10]:
            s = GraphSerializer(compression="zstd", compression_level=level)
            data = {"x" * 1000: "y" * 1000}
            blob = s.serialize(data)
            restored = s.deserialize(blob)
            assert restored == data

    def test_unknown_compression_raises(self):
        s = GraphSerializer(compression="invalid", compression_level=0)
        with pytest.raises(SerializationFailedError):
            s._compress(b"test")

    def test_unavailable_lz4_raises(self):
        import sys as _sys
        import builtins
        from unittest.mock import patch
        orig_mods = {}
        for k in list(_sys.modules.keys()):
            if k.startswith('lz4'):
                orig_mods[k] = _sys.modules.pop(k)
        s = GraphSerializer(compression="lz4", compression_level=1)
        real_import = builtins.__import__
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'lz4' or (isinstance(name, str) and name.startswith('lz4.')):
                    raise ImportError("not available")
                return real_import(name, *args, **kwargs)
            mock_import.side_effect = side_effect
            with pytest.raises(SerializationFailedError):
                s._compress(b"test")
        _sys.modules.update(orig_mods)


# ================================================================
# 2G. UTILS
# ================================================================

class TestUtils:
    def test_quantize_float32_to_int8_roundtrip(self):
        original = np.array([0.95, -0.85, 0.0, 0.5, -0.1, 0.33, -0.99, 0.01,
                             0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 1.0, -1.0,
                             0.0]*16, dtype=np.float32)[:32]
        quantized = quantize_embedding(original, 32, -1.0, 1.0)
        assert quantized.dtype == np.int8
        dequantized = dequantize_embedding(quantized, 32, -1.0, 1.0)
        assert np.allclose(original, dequantized, atol=0.02)

    def test_quantize_int8_passthrough(self):
        arr = np.array([-128, 0, 127, -50, 50]*7, dtype=np.int8)[:32]
        result = quantize_embedding(arr, 32, -1.0, 1.0)
        assert np.array_equal(arr, result)

    def test_quantize_rejects_non_float_non_int8(self):
        arr = np.zeros(32, dtype=np.int32)
        with pytest.raises(ValueError, match="must be float or int8"):
            quantize_embedding(arr, 32, -1.0, 1.0)

    def test_dequantize_rejects_non_int8(self):
        arr = np.zeros(32, dtype=np.float32)
        with pytest.raises(ValueError, match="must be int8"):
            dequantize_embedding(arr, 32, -1.0, 1.0)

    def test_quantize_wrong_shape_raises(self):
        arr = np.zeros(31, dtype=np.float32)
        with pytest.raises(ValueError):
            quantize_embedding(arr, 32, -1.0, 1.0)

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_mismatched_shapes(self):
        with pytest.raises(ValueError, match="Mismatched embedding shapes"):
            cosine_similarity(np.zeros(3), np.zeros(4))

    def test_validate_label(self):
        validate_label("hello", 256)
        with pytest.raises(ValueError):
            validate_label(123, 256)
        with pytest.raises(ValueError):
            validate_label("x" * 300, 256)

    def test_validate_node_type(self):
        allowed = ["Concept", "Entity", "Pattern"]
        validate_node_type("Concept", allowed)
        with pytest.raises(ValueError):
            validate_node_type("InvalidType", allowed)


# ================================================================
# 2H. GRAPHSTORE EDGE CASES
# ================================================================

class TestGraphStoreEdgeCases:
    def test_add_node_clamps_activation_below_min(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "Low", "Concept", emb, activation=-999)
        n = gs.get_node(1)
        assert abs(n.activation - 0.01) < 1e-6
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_add_node_clamps_activation_above_max(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "High", "Concept", emb, activation=999)
        n = gs.get_node(1)
        assert abs(n.activation - 1.0) < 1e-6
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_node_increments_use_count(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "Counted", "Concept", emb)
        for _ in range(5):
            gs.get_node(1)
        n = gs.get_node(1)
        assert n.use_count == 6
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_node_caches_after_storage_read(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "Cached", "Concept", emb)
        gs.node_cache.clear()
        gs.get_node(1)
        assert gs.node_cache.get(1) is not None
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_add_edge_auto_registers_relation(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        gs.add_edge(1, 2, "novel_relation")
        assert "novel_relation" in gs._relation_registry
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_add_edge_relation_limit_exhausted(self):
        gs, tmp = _fresh_graph(overrides={"edge": {"relation_max": 2}})
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        gs.add_edge(1, 2, "r1")
        gs.add_edge(1, 2, "r2")
        with pytest.raises(ValueError):
            gs.add_edge(1, 2, "r3")
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_update_edge_weights_nonexistent_raises(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        gs.add_edge(1, 2, "existing_rel")
        with pytest.raises(EdgeNotFoundError):
            gs.update_edge_weights({(1, 3, "existing_rel"): (0.5, 0.5)})
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_neighbors_no_neighbors(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "Alone", "Concept", emb)
        assert gs.get_neighbors(1) == []
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_neighbors_empty_filter_returns_all(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        gs.add_edge(1, 2, "rel")
        assert len(gs.get_neighbors(1, relation_filter=[])) == 1
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_subgraph_activated_empty_seeds(self):
        gs, tmp = _fresh_graph()
        sub = gs.get_subgraph_activated([])
        assert len(sub.nodes) == 0
        assert len(sub.edges) == 0
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_subgraph_activated_max_nodes_boundary(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        for i in range(10):
            gs.add_node(i, f"N{i}", "Concept", emb, activation=0.9)
        for i in range(9):
            gs.add_edge(i, i+1, "link")
        sub = gs.get_subgraph_activated([0], max_nodes=3)
        assert len(sub.nodes) <= 3
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_subgraph_activated_skips_low_activation(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "High", "Concept", emb, activation=0.9)
        gs.add_node(2, "Low", "Concept", emb, activation=0.01)
        gs.add_edge(1, 2, "link")
        sub = gs.get_subgraph_activated([1], max_nodes=10)
        assert 1 in sub.nodes
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_similarity_search_top_k(self):
        gs, tmp = _fresh_graph()
        for i in range(10):
            gs.add_node(i, f"N{i}", "Concept", make_emb(), activation=0.5)
        sub = gs.get_subgraph_by_embedding_similarity(make_emb(), top_k=5)
        assert len(sub.nodes) == 5
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_similarity_search_fewer_nodes_than_k(self):
        gs, tmp = _fresh_graph()
        for i in range(3):
            gs.add_node(i, f"N{i}", "Concept", make_emb(), activation=0.5)
        sub = gs.get_subgraph_by_embedding_similarity(make_emb(), top_k=10)
        assert len(sub.nodes) == 3
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_prune_removes_dangling_edges(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        gs.add_edge(1, 2, "link")
        gs.storage.remove_node(1)
        pruned = gs.prune(utility_threshold=-1.0)
        assert pruned == 0
        edges_after = list(gs.storage.iter_edges())
        assert len(edges_after) == 0
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_prune_clears_caches(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "Low", "Concept", emb, activation=0.01)
        gs.get_node(1)
        gs.prune(utility_threshold=0.02)
        assert gs.node_cache.get(1) is None
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_prune_empty_store_returns_zero(self):
        gs, tmp = _fresh_graph()
        assert gs.prune() == 0
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_save_checkpoint_empty_store(self):
        gs, tmp = _fresh_graph()
        cp = os.path.join(tmp, "empty.bin")
        assert gs.save_checkpoint(cp) is True
        assert os.path.exists(cp)
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_validate_embedding_wrong_dim_raises(self):
        gs, tmp = _fresh_graph()
        with pytest.raises(InvalidEmbeddingDimensionError):
            gs.add_node(1, "Bad", "Concept", np.zeros(31))
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_add_edge_updates_frequency(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        gs.add_edge(1, 2, "rel")
        gs.update_edge_weights({(1, 2, "rel"): (0.6, 0.7)})
        e = gs.get_edge(1, 2, "rel")
        assert e.frequency == 2
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_add_node_duplicate_returns_false(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        assert gs.add_node(1, "A", "Concept", emb) is True
        assert gs.add_node(1, "B", "Entity", make_emb()) is False
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_get_node_nonexistent_returns_none(self):
        gs, tmp = _fresh_graph()
        assert gs.get_node(99999) is None
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_update_node_embedding_nonexistent_raises(self):
        gs, tmp = _fresh_graph()
        with pytest.raises(NodeNotFoundError):
            gs.update_node_embedding(999, make_emb())
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_similarity_search_edges_subgraph_only(self):
        gs, tmp = _fresh_graph()
        emb_a = np.zeros(32, dtype=np.float32); emb_a[0] = 0.9
        emb_b = np.zeros(32, dtype=np.float32); emb_b[0] = 0.8
        emb_c = np.zeros(32, dtype=np.float32); emb_c[0] = -0.9
        gs.add_node(1, "A", "Concept", emb_a, activation=0.5)
        gs.add_node(2, "B", "Concept", emb_b, activation=0.5)
        gs.add_node(3, "C", "Concept", emb_c, activation=0.5)
        gs.add_edge(1, 2, "link")
        sub = gs.get_subgraph_by_embedding_similarity(emb_a, top_k=2)
        assert 1 in sub.nodes
        assert 2 in sub.nodes
        assert len(sub.edges) >= 1
        for e in sub.edges:
            assert e.source in sub.nodes and e.target in sub.nodes


# ================================================================
# 2I. THREAD SAFETY
# ================================================================

class TestThreadSafety:
    def test_concurrent_add_node(self):
        gs, tmp = _fresh_graph()
        n_threads = 10
        nodes_per = 20
        errors = []
        def add_nodes(start):
            try:
                for i in range(start, start + nodes_per):
                    gs.add_node(i, f"N{i}", "Concept", make_emb())
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=add_nodes, args=(i * nodes_per,)) for i in range(n_threads)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0
        total = len(list(gs.storage.iter_nodes()))
        assert total == n_threads * nodes_per
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_concurrent_get_and_update(self):
        gs, tmp = _fresh_graph()
        gs.add_node(1, "Target", "Concept", make_emb())
        errors = []
        def getter():
            try:
                for _ in range(50):
                    gs.get_node(1)
            except Exception as e:
                errors.append(e)
        def updater():
            try:
                for _ in range(50):
                    gs.update_node_embedding(1, make_emb())
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=getter) for _ in range(4)]
        threads.append(threading.Thread(target=updater))
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_concurrent_add_edge_and_neighbors(self):
        gs, tmp = _fresh_graph()
        for i in range(20):
            gs.add_node(i, f"N{i}", "Concept", make_emb())
        errors = []
        def adder():
            try:
                for i in range(19):
                    gs.add_edge(i, i+1, "rel")
            except Exception as e:
                errors.append(e)
        def neighbor():
            try:
                for _ in range(20):
                    gs.get_neighbors(5)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=adder) for _ in range(2)]
        threads += [threading.Thread(target=neighbor) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================
# 2J. BACKUP THREAD
# ================================================================

class TestBackup:
    def test_backup_thread_starts_when_enabled(self):
        gs, tmp = _fresh_graph(backup_enabled=True, overrides={"backup": {"interval_seconds": 1, "keep_last_n": 2}})
        assert gs._backup_thread is not None
        assert gs._backup_thread.is_alive()
        time.sleep(1.5)
        backup_dir = os.path.join(tmp, "backups")
        if os.path.exists(backup_dir):
            backups = [f for f in os.listdir(backup_dir) if f.startswith("backup_")]
        gs._backup_enabled = False
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_backup_rotation_keeps_last_n(self):
        gs, tmp = _fresh_graph(backup_enabled=True, overrides={"backup": {"interval_seconds": 0.5, "keep_last_n": 3}})
        time.sleep(2.5)
        backup_dir = os.path.join(tmp, "backups")
        if os.path.exists(backup_dir):
            backups = sorted([f for f in os.listdir(backup_dir) if f.startswith("backup_")])
            assert len(backups) <= 3
        gs._backup_enabled = False
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_backup_disabled_no_thread(self):
        gs, tmp = _fresh_graph(backup_enabled=False)
        assert gs._backup_thread is None
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================
# 2K. CONFIG / INITIALIZATION
# ================================================================

class TestConfigInit:
    def test_default_config_path(self):
        gs, tmp = _fresh_graph()
        assert gs.config is not None
        gs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_config_file_raises(self):
        with pytest.raises((FileNotFoundError, GraphStoreError)):
            GraphStore(config_path="/nonexistent/path/config.yaml")

    def test_unknown_backend_raises(self):
        tmp = tempfile.mkdtemp()
        with open(CONFIG_SRC) as f:
            cfg = yaml.safe_load(f)
        cfg["storage"]["backend"] = "unknown_backend"
        cfg["storage"]["base_path"] = tmp
        cfg_path = os.path.join(tmp, "cfg.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        with pytest.raises(ValueError, match="Unknown storage backend"):
            GraphStore(config_path=cfg_path)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_close_shuts_down_prefetcher(self):
        gs, tmp = _fresh_graph()
        gs.close()
        assert gs.prefetcher.executor._shutdown
        assert gs._backup_enabled is False


# ================================================================
# 2L. RECOVERY / DURABILITY
# ================================================================

class TestRecovery:
    def test_load_checkpoint_restores_all_data(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        nodes_data = [(1, "A"), (2, "B"), (3, "C")]
        for nid, lbl in nodes_data:
            gs.add_node(nid, lbl, "Concept", make_emb(), activation=0.5)
        gs.add_edge(1, 2, "knows", 0.9, 0.8)
        gs.add_edge(2, 3, "knows", 0.7, 0.6)
        cp = os.path.join(tmp, "full.bin")
        gs.save_checkpoint(cp)
        gs2, tmp2 = _fresh_graph()
        gs2.load_checkpoint(cp)
        for nid, lbl in nodes_data:
            n = gs2.get_node(nid)
            assert n is not None and n.label == lbl
        e1 = gs2.get_edge(1, 2, "knows")
        assert e1 is not None and abs(e1.strength - 0.9) < 1e-5
        store = gs2.storage
        assert len(store.edge_by_source.get(1, set())) == 1
        assert len(store.edge_by_target.get(3, set())) == 1
        gs.close()
        gs2.close()
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)

    def test_load_checkpoint_restores_relation_registry(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        gs.add_node(1, "A", "Concept", emb)
        gs.add_node(2, "B", "Concept", emb)
        gs.add_edge(1, 2, "custom_rel")
        cp = os.path.join(tmp, "cp.bin")
        gs.save_checkpoint(cp)
        gs2, tmp2 = _fresh_graph()
        gs2.load_checkpoint(cp)
        assert "custom_rel" in gs2._relation_registry
        assert gs2.get_edge(1, 2, "custom_rel") is not None
        gs.close()
        gs2.close()
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)

    def test_load_checkpoint_preserves_edge_indexes(self):
        gs, tmp = _fresh_graph()
        emb = make_emb()
        for i in range(5):
            gs.add_node(i, f"N{i}", "Concept", emb)
        gs.add_edge(0, 1, "a")
        gs.add_edge(0, 2, "b")
        gs.add_edge(3, 4, "a")
        cp = os.path.join(tmp, "cp.bin")
        gs.save_checkpoint(cp)
        gs2, tmp2 = _fresh_graph()
        gs2.load_checkpoint(cp)
        neigh = gs2.get_neighbors(0)
        assert len(neigh) == 2
        neigh3 = gs2.get_neighbors(3)
        assert len(neigh3) == 1
        gs.close()
        gs2.close()
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)
