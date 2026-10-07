"""Canonical CONTENT hash of the showcase graph DB, independent of SQLite framing.

Hashes exactly what the evaluation rests on: node id->label, and every
(source_id, target_id, relation, strength, confidence) tuple. Also reports the
volatile columns separately so framing-only drift can be told apart from content
drift.

Why this exists alongside `frozen_graph_digest.txt`: that file holds
`finalize_and_hash()`, which hashes the .db *file*. The file hash changes on
every rebuild because `saved_at` / `create_time` are wall-clock stamps. This
hash excludes them, so "the graph content did not change across a rebuild" is a
measurable statement rather than an assertion.

    python -B graph_content_hash.py showcase_eval.db

Expected: 0d6edf6eb6d78ac45230abc484f37f214ffb26bccbbe460babe6a92155680c48
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


def content_hash(db: Path) -> tuple[str, dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()

    labels = sorted(
        (int(i), str(l)) for i, l in cur.execute("SELECT id, label FROM nodes"))
    edges = sorted(
        (int(a), int(b), str(r), float(s), float(c))
        for a, b, r, s, c in cur.execute(
            "SELECT source_id, target_id, relation, strength, confidence FROM edges"))

    meta = dict(cur.execute("SELECT key, value FROM metadata"))
    volatile = {}
    try:
        ct = [r[0] for r in cur.execute("SELECT create_time FROM nodes")]
        volatile["create_time_distinct"] = len(set(ct))
        volatile["create_time_sample"] = sorted(set(ct))[:3]
    except Exception as exc:
        volatile["create_time_error"] = str(exc)
    try:
        act = [r[0] for r in cur.execute("SELECT activation FROM nodes")]
        volatile["activation_sum"] = round(sum(act), 6)
        uc = [r[0] for r in cur.execute("SELECT use_count FROM nodes")]
        volatile["use_count_sum"] = int(sum(uc))
    except Exception as exc:
        volatile["runtime_error"] = str(exc)
    con.close()

    h = hashlib.sha256()
    h.update(json.dumps(labels, ensure_ascii=False).encode())
    h.update(json.dumps(edges, ensure_ascii=False).encode())
    info = {
        "n_labels": len(labels),
        "n_edges": len(edges),
        "metadata_keys": sorted(meta),
        "dataset_name": meta.get("dataset_name"),
        **volatile,
    }
    return h.hexdigest(), info


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d, info = content_hash(Path(p))
        print(f"{d}  {p}")
        print(f"    {json.dumps(info, ensure_ascii=False)}")
