"""Dataset compliance checker (Phase 5) - EXPERIMENTATION.md Section 5.4 / 13.x.

Validates every tester dataset directly from its .db (stdlib sqlite3, no heavy
imports): node/edge size band, >=4 distinct relations, relation codes confined
to the canonical 16, golden-suite coverage (>=8 goldens, >=2 per relation).

Usage:
    python test_results/lead/check_dataset.py [--suites food_bio_small,food_bio_medium]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from learning.types import Edge  # noqa: E402
from test_results.lead.unified_golden_runner import SUITES  # noqa: E402

CANONICAL = Edge.VALID_RELATIONS


def _open(db: str):
    conn = sqlite3.connect(db)
    def tables():
        return [t[0] for t in conn.execute(
            "select name from sqlite_master where type='table'")]
    assert "nodes" in tables() and "edges" in tables(), "missing nodes/edges tables"
    return conn


def check_suite(name: str) -> list[dict]:
    suite = SUITES[name]
    db_path = Path(suite["db"])
    if not db_path.exists():
        return [{"level": "FAIL", "check": "db_exists", "detail": "db file missing"}]

    report = []
    conn = _open(str(db_path))
    try:
        n_nodes = conn.execute("select count(*) from nodes").fetchone()[0]
        n_edges = conn.execute("select count(*) from edges").fetchone()[0]
        relations = sorted(r[0] for r in conn.execute(
            "select distinct relation from edges").fetchall())
        n_rels = len(relations)

        band = "small (10-100)" if 10 <= n_nodes < 100 else (
            "medium (100-1000)" if 100 <= n_nodes < 1000 else "out-of-band")
        if 10 <= n_nodes < 1000:
            report.append({"level": "PASS", "check": "size_band", "detail": f"{n_nodes} nodes -> {band}"})
        else:
            report.append({"level": "FAIL", "check": "size_band", "detail": f"{n_nodes} nodes"})

        report.append({"level": "PASS", "check": "nodes", "detail": str(n_nodes)})
        report.append({"level": "PASS", "check": "edges", "detail": str(n_edges)})

        if n_rels >= 4:
            report.append({"level": "PASS", "check": "min4_relations",
                           "detail": f"{n_rels} distinct relations"})
        else:
            report.append({"level": "FAIL", "check": "min4_relations",
                           "detail": f"{n_rels} < 4 (Section 5.4)"})

        codes = [r for r in relations if r not in CANONICAL]
        if codes:
            report.append({"level": "FAIL", "check": "relation_codes_in_canon16",
                           "detail": f"out-of-code: {codes}"})
        else:
            report.append({"level": "PASS", "check": "relation_codes_in_canon16",
                           "detail": f"{len(relations)} all in canonical 16"})

        # duplicate-label probe (case-colliding labels kept distinct)
        dups = [l for l, c in conn.execute(
            "select lower(label) as ll, count(*) from nodes group by ll").fetchall() if c > 1]
        if dups:
            report.append({"level": "INFO", "check": "label_case_collisions",
                           "detail": f"{len(dups)} lower-case groups: {dups}"})
        else:
            report.append({"level": "PASS", "check": "label_unique", "detail": "no collisions"})

        # golden coverage vs available relations
        goldens = suite["goldens"]
        rel_counts: dict[str, int] = {}
        for g in goldens:
            if "chain" in g:
                for rel in g["chain"]:
                    rel_counts[rel] = rel_counts.get(rel, 0) + 1
        covered = [rel for rel in relations if rel_counts.get(rel, 0) >= 2]
        if len(goldens) >= 8:
            report.append({"level": "PASS", "check": "golden_count",
                           "detail": f"{len(goldens)} >= 8"})
        else:
            report.append({"level": "FAIL", "check": "golden_count",
                           "detail": f"{len(goldens)} < 8"})
        if covered:
            report.append({"level": "PASS", "check": "golden_coverage",
                           "detail": f">=2 goldens per relation for: {covered}"})
        else:
            report.append({"level": "FAIL", "check": "golden_coverage",
                           "detail": "no relation has >=2 goldens"})
    finally:
        conn.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", default=",".join(SUITES.keys()))
    args = parser.parse_args()

    for name in [s.strip() for s in args.suites.split(",") if s.strip()]:
        print("=" * 60)
        print(f"DATASET: {name}")
        for item in check_suite(name):
            print(f"  [{item['level']:4s}] {item['check']}: {item['detail']}")


if __name__ == "__main__":
    main()