"""
Universal incremental KG builder — dataset-agnostic.

Accepts data in any supported format (.json/.jsonl/.csv/.parquet/.txt),
auto-detects format, and builds a KG incrementally with persistent
entity/relation alignment.

Usage:
    python -m model_training.universal.build_incremental ^
        --data path/to/data.jsonl --out-db output.db ^
        --entity-reg entity_reg.pkl --relation-reg relation_reg.pkl ^
        --start 0 --end 1000
"""
import sys, json, time, os, logging, pickle, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_loader.base import load_data
from graph.alignment import EntityRegistry, RelationRegistry

for noisy in ["sentence_transformers", "transformers", "httpx", "urllib3",
               "huggingface_hub", "filelock", "PIL"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

BASE = Path(__file__).resolve().parent.parent.parent
BATCH_CHECKPOINT = 200


def setup_logging(run_label: str):
    log_file = BASE / "model_training" / "universal" / f"build_{run_label}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger("build_incr")


def main():
    parser = argparse.ArgumentParser(description="Universal incremental KG builder")
    parser.add_argument("--data", required=True, help="Input data path (.json/.jsonl/.csv/.parquet/.txt)")
    parser.add_argument("--out-db", required=True, help="Output SQLite DB path")
    parser.add_argument("--entity-reg", default=None, help="Entity registry path (auto if omitted)")
    parser.add_argument("--relation-reg", default=None, help="Relation registry path (auto if omitted)")
    parser.add_argument("--start", type=int, default=0, help="Start index (document/row)")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--format", default=None, help="Force data format (auto-detected by extension if omitted)")
    args = parser.parse_args()

    out_db = Path(args.out_db)
    entity_reg_path = Path(args.entity_reg) if args.entity_reg else out_db.with_suffix(".entity_registry.pkl")
    relation_reg_path = Path(args.relation_reg) if args.relation_reg else out_db.with_suffix(".relation_registry.pkl")

    logger = setup_logging(f"{args.start}-{args.end}")
    print(f"UNIVERSAL INCREMENTAL KG BUILD — {args.data}")
    print(f"  Range:   {args.start} – {args.end}")
    print(f"  DB:      {out_db}")
    print(f"  Format:  {args.format or 'auto'}")
    print(f"  Registry: {entity_reg_path.parent}")

    from sentence_transformers import SentenceTransformer
    shared_sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")

    entity_registry = EntityRegistry(sbert_model=shared_sbert, registry_path=str(entity_reg_path))
    relation_registry = RelationRegistry(sbert_model=shared_sbert, registry_path=str(relation_reg_path))
    logger.info("Registries: ER=%d canonicals, RR=%d clusters",
                entity_registry.size, relation_registry.num_clusters)

    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
    if out_db.exists():
        store = SQLiteGraphStore(db_path=str(out_db))
        logger.info("Existing DB: %d nodes, %d edges", store.get_node_count(), store.get_edge_count())
        if entity_registry.size == 0:
            nids = sorted(store._nodes.keys())
            labels = [store._nodes[n].label for n in nids]
            embs_list = []
            for nid in nids:
                e = store._embeddings.get(nid)
                embs_list.append(e if e is not None and e.size > 0 else np.zeros(384, dtype=np.float32))
            entity_registry.align_batch(labels, np.array(embs_list))
            rels = list(store._relation_set)
            if rels:
                relation_registry.canonicalize_batch(rels)
            logger.info("  Pre-populated: ER=%d, RR=%d", entity_registry.size, relation_registry.num_clusters)
    else:
        store = SQLiteGraphStore(db_path=str(out_db))
        logger.info("No existing DB — creating new")

    total_triples = 0
    T_GLOBAL = time.time()
    dataset_name = Path(args.data).stem

    from kg_builder import KGBuilderPipeline, KGBuilderConfig
    from kg_builder.cross_sentence_linker import CrossSentenceLinker
    from tqdm import tqdm

    config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False, embed_merge_threshold=0.92)
    pipeline = KGBuilderPipeline(config)
    pipeline.initialize()
    dp, te, er, gb = pipeline.doc_processor, pipeline.triple_extractor, pipeline.entity_resolver, pipeline.graph_builder
    te._sbert = None
    er._sbert = shared_sbert

    checkpoint_dir = out_db.parent / f".checkpoints_{out_db.stem}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batch_start = args.start
    while batch_start < args.end:
        batch_end = min(batch_start + BATCH_CHECKPOINT, args.end)
        batch_tag = f"{batch_start}-{batch_end}"

        logger.info("── Batch %s ──", batch_tag)

        data = load_data(args.data, format=args.format, start=batch_start, end=batch_end)

        has_documents = len(data.documents) > 0
        has_sentences = len(data.sentences) > 0
        has_triples = len(data.triples) > 0

        if has_triples and not has_documents and not has_sentences:
            logger.info("Loaded %d pre-extracted triples", len(data.triples))
            all_triple_results = [
                {"triple": (s, r, o), "level": 1, "relation_confidence": 1.0,
                 "resolution_method": "exact", "doc_index": batch_start + i,
                 "raw_connector": r}
                for i, (s, r, o) in enumerate(data.triples)
            ]
            gd = gb.build(all_triple_results)
            _add_to_store(store, gd, entity_registry, relation_registry, shared_sbert, out_db,
                          entity_reg_path, relation_reg_path, dataset_name, batch_end - 1)
            total_triples += len(all_triple_results)
            mb = os.path.getsize(str(out_db)) / (1024 * 1024)
            print(f"  DB: {store.get_node_count()} nodes, {store.get_edge_count()} edges ({mb:.1f} MB)")
            batch_start = batch_end
            continue

        documents = data.documents
        if not documents:
            logger.warning("No documents in range %s", batch_tag)
            batch_start = batch_end
            continue

        logger.info("  Documents: %d", len(documents))

        t0 = time.time()
        all_sentences_list = dp.process_batch(documents, batch_size=32)
        t_spacy = time.time() - t0
        print(f"  spaCy: {t_spacy:.1f}s ({t_spacy/len(documents):.2f}s/doc)")

        start_from = 0
        all_triple_results = []
        checkpoint_path = checkpoint_dir / f"chk_{batch_tag}.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                saved = pickle.load(f)
            all_triple_results = saved["triples"]
            start_from = saved["doc_index"] + 1
            er._canonical_to_id = saved["canonical_to_id"]
            er._id_to_canonical = {int(k): v for k, v in saved["id_to_canonical"].items()}
            er._embeddings = saved["embeddings"]
            er._surface_forms = saved["surface_forms"]
            er._entity_freq = saved["entity_freq"]
            er._next_id = saved["next_id"]
            print(f"  Resumed doc {start_from-1} ({len(all_triple_results)} triples)")

        t_start = time.time()
        for doc_idx in tqdm(range(start_from, len(documents)), desc="Documents", unit="doc",
                            ncols=80, initial=start_from, total=len(documents)):
            sentences = all_sentences_list[doc_idx]
            linker = CrossSentenceLinker()
            raw_triple_data = []
            for sent in sentences:
                if len(sent.get("entities", [])) == 0 and len(sent.get("text", "")) < 10:
                    continue
                try:
                    extracted = te.extract_or_escalate(sent)
                except Exception:
                    continue
                for item in extracted:
                    t = item["triple"]
                    raw_conn = item.get("raw_connector", t[1])
                    raw_triple_data.append((t[0], t[1], t[2], item.get("rel_confidence", 0.5), raw_conn, sent.get("text", "")))

            if raw_triple_data:
                linked = linker.link([(d[0], d[1], d[2], d[3], d[4]) for d in raw_triple_data])
                original_keys = {(d[0].lower().strip(), d[1], d[2].lower().strip()) for d in raw_triple_data}
                all_entities = []
                for item in linked:
                    all_entities.append(item[0])
                    all_entities.append(item[2])
                resolved = er.resolve_batch(all_entities)
                unique_resolved = list(set(resolved))
                unique_canonical, _ = entity_registry.align_batch(unique_resolved)
                resolved_to_canonical = dict(zip(unique_resolved, unique_canonical))
                for idx, item in enumerate(linked):
                    e1, rel, e2, rel_conf, raw_conn = item
                    re1 = resolved_to_canonical.get(resolved[idx * 2], resolved[idx * 2])
                    re2 = resolved_to_canonical.get(resolved[idx * 2 + 1], resolved[idx * 2 + 1])
                    if re1 == re2:
                        continue
                    key = (e1.lower().strip(), rel, e2.lower().strip())
                    level = 1 if key in original_keys else 2
                    all_triple_results.append({
                        "triple": (re1, raw_conn, re2), "level": level,
                        "relation_confidence": rel_conf, "resolution_method": "embedding",
                        "doc_index": doc_idx + batch_start,
                        "raw_connector": raw_conn,
                    })

            if (doc_idx + 1) % BATCH_CHECKPOINT == 0 or doc_idx == len(documents) - 1:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "triples": all_triple_results, "doc_index": doc_idx,
                        "canonical_to_id": er._canonical_to_id,
                        "id_to_canonical": {str(k): v for k, v in er._id_to_canonical.items()},
                        "embeddings": er._embeddings, "surface_forms": er._surface_forms,
                        "entity_freq": er._entity_freq, "next_id": er._next_id,
                    }, f)
                entity_registry.save(str(entity_reg_path))
                relation_registry.save(str(relation_reg_path))
                tqdm.write(f"  CHECKPOINT doc {doc_idx} ({len(all_triple_results)} triples)")

        t_extract = time.time() - t_start
        checkpoint_path.unlink(missing_ok=True)

        unique_raw = len(set(t["triple"][1] for t in all_triple_results))
        all_rels = [t["triple"][1] for t in all_triple_results]
        canonical_rels, _ = relation_registry.canonicalize_batch(all_rels)
        for t, cr in zip(all_triple_results, canonical_rels):
            t["triple"] = (t["triple"][0], cr, t["triple"][2])
        print(f"  Relations: {unique_raw} → {len(set(canonical_rels))}")

        gd = gb.build(all_triple_results)
        print(f"  Graph: {gd['node_count']} nodes, {gd['edge_count']} edges")

        _add_to_store(store, gd, entity_registry, relation_registry, shared_sbert, out_db,
                      entity_reg_path, relation_reg_path, dataset_name, batch_end - 1)

        total_triples += len(all_triple_results)
        mb = os.path.getsize(str(out_db)) / (1024 * 1024)
        print(f"  DB: {store.get_node_count()} nodes, {store.get_edge_count()} edges ({mb:.1f} MB)")

        batch_start = batch_end

    elapsed = time.time() - T_GLOBAL
    print(f"\n{'='*60}")
    store = SQLiteGraphStore(db_path=str(out_db))
    print(f"BUILD COMPLETE — items {args.start}–{args.end}")
    print(f"  Final DB: {store.get_node_count()} nodes, {store.get_edge_count()} edges "
          f"({os.path.getsize(str(out_db))/(1024*1024):.1f} MB)")
    print(f"  Registry: {entity_registry.size} canonicals, {relation_registry.num_clusters} relation clusters")
    print(f"  Time:     {elapsed:.0f}s")
    print(f"{'='*60}")


def _add_to_store(store, gd, entity_registry, relation_registry, shared_sbert,
                   out_db, entity_reg_path, relation_reg_path, dataset_name, last_idx):
    from resonance.types import Node as RNode
    from graph.graph_component_implementation.dict_graph_store import EdgeRecord
    from resonance.types import Edge

    existing_label_to_id = {node.label: nid for nid, node in store._nodes.items()}
    next_available_id = store._next_id
    local_to_store_id = {}
    new_node_labels = []

    for label in gd["concepts"]:
        if label in existing_label_to_id:
            local_to_store_id[gd["concepts"][label]] = existing_label_to_id[label]
        else:
            new_nid = next_available_id
            local_to_store_id[gd["concepts"][label]] = new_nid
            existing_label_to_id[label] = new_nid
            new_node_labels.append(label)
            next_available_id += 1
            emb = gd.get("embeddings", {}).get(label)
            if emb is not None:
                emb = emb.astype(np.float32)
            store._nodes[new_nid] = RNode(
                id=new_nid, label=label, node_type="concept",
                embedding=emb if emb is not None else np.zeros(384, dtype=np.float32),
                activation=0.5, use_count=0, create_time=time.time(),
            )
            store._id_to_label[new_nid] = label
            store._label_to_id[label] = new_nid

    for edge in gd["edges"]:
        src = local_to_store_id.get(edge["source"])
        tgt = local_to_store_id.get(edge["target"])
        if src is not None and tgt is not None and src != tgt:
            rel = edge["relation"]
            strength = edge["strength"]
            confidence = edge["confidence"]
            store._relation_set.add(rel)
            edge_obj = Edge(
                source=src, target=tgt, relation_type=rel,
                strength=strength, confidence=confidence,
                last_used=time.time(), frequency=1,
            )
            store._neighbors.setdefault(src, []).append((tgt, edge_obj))
            store._neighbors.setdefault(tgt, []).append((src, edge_obj))
            store._edges_raw.append(EdgeRecord(
                source=src, target=tgt, relation=rel,
                strength=strength, confidence=confidence,
            ))

    if store._next_id < next_available_id:
        store._next_id = next_available_id

    new_nids = [existing_label_to_id[lbl] for lbl in new_node_labels
                if existing_label_to_id[lbl] not in store._embeddings]
    if new_nids:
        new_labels = [store._nodes[nid].label for nid in new_nids]
        new_embs = shared_sbert.encode(new_labels, normalize_embeddings=True)
        for nid, emb in zip(new_nids, new_embs):
            store._embeddings[nid] = emb.astype(np.float32)
            node = store._nodes[nid]
            store._nodes[nid] = RNode(
                id=node.id, label=node.label, node_type=node.node_type,
                embedding=emb.astype(np.float32), activation=node.activation,
                use_count=node.use_count, create_time=node.create_time,
            )

    store.set_metadata("dataset_name", dataset_name)
    store.set_metadata("last_index", str(last_idx))
    store.save_state(str(out_db))
    entity_registry.save(str(entity_reg_path))
    relation_registry.save(str(relation_reg_path))


if __name__ == "__main__":
    main()
