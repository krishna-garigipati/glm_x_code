"""
Universal parallel KG builder — dataset-agnostic.

Splits a data range across N worker subprocesses, each building an
independent chunk DB via the universal incremental builder, then
sequentially merges all chunk DBs into the main DB.

Usage:
    python -m model_training.universal.build_parallel ^
        --data path/to/data.jsonl --out-db output.db ^
        --start 0 --end 5000 --workers 2
"""
import sys, os, time, shutil, subprocess, argparse, logging, json, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_loader.base import DataLoader

BASE = Path(__file__).resolve().parent.parent.parent
TEMP_DIR = BASE / "model_training" / "universal" / "parallel_tmp"
CHUNK_SIZE = 1000


def setup_logging():
    log_file = BASE / "model_training" / "universal" / "build_parallel.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger("parallel_build")


def main():
    parser = argparse.ArgumentParser(description="Universal parallel KG builder")
    parser.add_argument("--data", required=True, help="Input data path")
    parser.add_argument("--out-db", required=True, help="Output SQLite DB path")
    parser.add_argument("--entity-reg", default=None, help="Entity registry path (auto if omitted)")
    parser.add_argument("--relation-reg", default=None, help="Relation registry path (auto if omitted)")
    parser.add_argument("--start", type=int, required=True, help="Start index")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--format", default=None, help="Force data format (auto-detected by extension if omitted)")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent workers")
    args = parser.parse_args()

    logger = setup_logging()

    OUT_DB = Path(args.out_db)
    ENTITY_REG_PATH = Path(args.entity_reg) if args.entity_reg else OUT_DB.with_suffix(".entity_registry.pkl")
    RELATION_REG_PATH = Path(args.relation_reg) if args.relation_reg else OUT_DB.with_suffix(".relation_registry.pkl")

    print(f"UNIVERSAL PARALLEL KG BUILD — {args.data}")
    print(f"  Range:   {args.start} – {args.end}")
    print(f"  Workers: {args.workers}")
    print(f"  Main DB: {OUT_DB}")
    print(f"  Registry: {ENTITY_REG_PATH.parent}")

    ranges = []
    for s in range(args.start, args.end, CHUNK_SIZE):
        e = min(s + CHUNK_SIZE, args.end)
        ranges.append((s, e))
    print(f"Split into {len(ranges)} worker chunks")

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    T_GLOBAL = time.time()
    procs = []
    pending = list(ranges)
    active = []
    results = []

    def launch_worker(s, e):
        wdir = TEMP_DIR / f"w_{s}-{e}"
        wdir.mkdir(parents=True, exist_ok=True)
        out_db = wdir / f"chunk_{s}-{e}.db"
        out_er = wdir / "entity_registry.pkl"
        out_rr = wdir / "relation_registry.pkl"

        if ENTITY_REG_PATH.exists():
            shutil.copy2(str(ENTITY_REG_PATH), str(out_er))
        if RELATION_REG_PATH.exists():
            shutil.copy2(str(RELATION_REG_PATH), str(out_rr))

        cmd = [
            sys.executable, "-m", "model_training.universal.build_incremental",
            "--data", args.data,
            "--out-db", str(out_db),
            "--entity-reg", str(out_er),
            "--relation-reg", str(out_rr),
            "--start", str(s), "--end", str(e),
        ]
        if args.format:
            cmd.extend(["--format", args.format])

        log_path = wdir / "worker.log"
        log_fh = open(str(log_path), "w", encoding="utf-8")
        env = os.environ.copy()
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["TOKENIZERS_PARALLELISM"] = "false"
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, cwd=str(BASE), env=env)
        procs.append((p, s, e, wdir, out_db, out_er, out_rr, log_fh))
        active.append((p, s, e, wdir, out_db, out_er, out_rr, log_fh))
        logger.info("  Launched worker %d-%d (PID=%d, %d active, %d pending)",
                     s, e, p.pid, len(active), len(pending))

    for _ in range(min(args.workers, len(pending))):
        launch_worker(*pending.pop(0))

    last_progress = {}
    while active:
        still_active = []
        for item in active:
            p, s, e, wdir, out_db, out_er, out_rr, log_fh = item
            ret = p.poll()
            if ret is None:
                still_active.append(item)
                continue
            log_fh.close()
            if ret == 0:
                results.append((s, e, str(wdir), str(out_db), str(out_er), str(out_rr)))
                logger.info("Worker %d-%d DONE (exit=0, %d/%d finished)", s, e, len(results), len(ranges))
            else:
                log_text = Path(str(wdir / "worker.log")).read_text(encoding="utf-8")[-2000:]
                logger.error("Worker %d-%d FAILED (exit=%d). Tail:\n%s", s, e, ret, log_text)
            if pending:
                launch_worker(*pending.pop(0))
        active = still_active
        if active:
            lines = []
            for p, s, e, wdir, out_db, out_er, out_rr, log_fh in active:
                try:
                    text = Path(str(wdir / "worker.log")).read_text(encoding="utf-8", errors="replace")
                    for line in reversed(text.splitlines()):
                        if "Documents:" in line or "Graph:" in line or "DB:" in line or "Relations:" in line:
                            prog = line.strip()
                            break
                    else:
                        prog = "(processing...)"
                except Exception:
                    prog = "(reading...)"
                if prog != last_progress.get((s, e)):
                    last_progress[(s, e)] = prog
                    lines.append(f"  Worker {s}-{e}: {prog}")
            if lines:
                logger.info("├─ Progress ──────────────────────────────")
                for l in lines:
                    logger.info(l)
                logger.info("└─────────────────────────────────────────")
            time.sleep(15)

    t_workers = time.time() - T_GLOBAL
    logger.info("All workers finished: %d success, %d failed in %.0fs",
                len(results), len(ranges) - len(results), t_workers)

    if not results:
        logger.error("No workers succeeded — aborting")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"MERGING {len(results)} worker results into main DB...")
    print(f"{'='*60}")

    T_MERGE = time.time()
    shared_sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")

    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
    from resonance.types import Node, Edge
    from graph.graph_component_implementation.dict_graph_store import EdgeRecord
    from graph.alignment import EntityRegistry, RelationRegistry
    import numpy as np

    if OUT_DB.exists():
        main_store = SQLiteGraphStore(db_path=str(OUT_DB))
        logger.info("Main DB: %d nodes, %d edges", main_store.get_node_count(), main_store.get_edge_count())
    else:
        main_store = SQLiteGraphStore(db_path=str(OUT_DB))

    main_er = EntityRegistry(sbert_model=shared_sbert, registry_path=str(ENTITY_REG_PATH))
    main_rr = RelationRegistry(sbert_model=shared_sbert, registry_path=str(RELATION_REG_PATH))

    dataset_name = Path(args.data).stem
    for s, e, wdir, chunk_db, chunk_er, chunk_rr in results:
        logger.info("Merging worker %d-%d into main DB...", s, e)
        try:
            chunk_store = SQLiteGraphStore(db_path=chunk_db)
            logger.info("  Chunk: %d nodes, %d edges", chunk_store.get_node_count(), chunk_store.get_edge_count())

            if os.path.exists(chunk_er):
                tmp_er = EntityRegistry(sbert_model=shared_sbert, registry_path=chunk_er)
                for label in tmp_er.get_all_canonical_labels():
                    if main_er.get_canonical_id(label) is None:
                        emb = tmp_er.get_embedding(label)
                        if emb is not None:
                            main_er._canonical_label_to_emb[label] = emb
                            nid = main_er._next_id
                            main_er._canonical_label_to_id[label] = nid
                            main_er._id_to_canonical_label[nid] = label
                            main_er._next_id += 1
                logger.info("  ER: absorbed %d canonicals (now %d)", tmp_er.size, main_er.size)

            existing_labels = {node.label: nid for nid, node in main_store._nodes.items()}
            next_nid = main_store._next_id

            for chunk_nid in sorted(chunk_store._nodes.keys()):
                label = chunk_store._id_to_label[chunk_nid]
                if label in existing_labels:
                    continue
                store_nid = next_nid
                next_nid += 1
                existing_labels[label] = store_nid
                emb = chunk_store._embeddings.get(chunk_nid)
                if emb is None or emb.size == 0:
                    emb = np.zeros(384, dtype=np.float32)
                main_store._nodes[store_nid] = Node(
                    id=store_nid, label=label, node_type="concept",
                    embedding=emb.astype(np.float32), activation=0.5,
                    use_count=0, create_time=time.time(),
                )
                main_store._id_to_label[store_nid] = label
                main_store._label_to_id[label] = store_nid
                main_store._embeddings[store_nid] = emb.astype(np.float32)

            for edge in chunk_store._edges_raw:
                src_label = chunk_store._id_to_label.get(edge.source)
                tgt_label = chunk_store._id_to_label.get(edge.target)
                src = existing_labels.get(src_label)
                tgt = existing_labels.get(tgt_label)
                if src is not None and tgt is not None and src != tgt:
                    main_store._relation_set.add(edge.relation)
                    main_store._edges_raw.append(EdgeRecord(
                        source=src, target=tgt, relation=edge.relation,
                        strength=edge.strength, confidence=edge.confidence,
                    ))
                    eobj = Edge(
                        source=src, target=tgt, relation_type=edge.relation,
                        strength=edge.strength, confidence=edge.confidence,
                        last_used=time.time(), frequency=1,
                    )
                    main_store._neighbors.setdefault(src, []).append((tgt, eobj))
                    main_store._neighbors.setdefault(tgt, []).append((src, eobj))

            if main_store._next_id < next_nid:
                main_store._next_id = next_nid

            if os.path.exists(chunk_rr):
                tmp_rr = RelationRegistry(sbert_model=shared_sbert, registry_path=chunk_rr)
                for cid in tmp_rr._centroids:
                    if cid not in main_rr._centroids:
                        main_rr._centroids[cid] = tmp_rr._centroids[cid]
                        main_rr._members[cid] = tmp_rr._members.get(cid, [])
                        main_rr._member_counts[cid] = tmp_rr._member_counts.get(cid, 0)
                        for text in set(main_rr._members[cid]):
                            main_rr._canonical_to_cluster[text] = cid
                next_num = max(int(k.split("_")[1]) for k in main_rr._centroids) + 1
                main_rr._next_cluster_id = max(main_rr._next_cluster_id, next_num)
                logger.info("  RR: absorbed %d clusters (now %d)", tmp_rr.num_clusters, main_rr.num_clusters)

            chunk_store._conn.close()
        except Exception as ex:
            logger.error("  Failed to merge worker %d-%d: %s", s, e, ex)
            import traceback
            logger.error(traceback.format_exc())

    missing_emb = [nid for nid in main_store._nodes if nid not in main_store._embeddings]
    if missing_emb:
        labels = [main_store._nodes[nid].label for nid in missing_emb]
        embs = shared_sbert.encode(labels, normalize_embeddings=True)
        for nid, emb in zip(missing_emb, embs):
            main_store._embeddings[nid] = emb.astype(np.float32)

    main_store.set_metadata("dataset_name", dataset_name)
    main_store.set_metadata("last_index", str(args.end - 1))
    main_store.set_metadata("parallel_build", f"{args.start}-{args.end}")
    main_store.save_state(str(OUT_DB))
    main_er.save(str(ENTITY_REG_PATH))
    main_rr.save(str(RELATION_REG_PATH))

    t_merge = time.time() - T_MERGE

    main_store._conn.close()
    gc.collect()
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    mb = os.path.getsize(str(OUT_DB)) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"PARALLEL BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  Data:     {args.data}")
    print(f"  Range:    {args.start} – {args.end}")
    print(f"  Workers:  {len(results)}/{len(ranges)} succeeded")
    store_final = SQLiteGraphStore(db_path=str(OUT_DB))
    print(f"  DB:       {store_final.get_node_count()} nodes, {store_final.get_edge_count()} edges ({mb:.1f} MB)")
    print(f"  Registry: ER={main_er.size}, RR={main_rr.num_clusters}")
    print(f"  Workers:  {t_workers:.0f}s")
    print(f"  Merge:    {t_merge:.0f}s")
    print(f"  Total:    {time.time() - T_GLOBAL:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
