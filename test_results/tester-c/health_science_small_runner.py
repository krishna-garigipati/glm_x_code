"""Level-1 evaluation of the health_science_small graph through the full pipeline.

Mirrors test_results/tester-b/food_bio_runner.py for tester-c's Health & Science
small dataset. Runs the pre-registered golden set (health_science_small_golden.md)
through scripts/glmx_ask in-process and scores each question on:

    hops     : len(walk_path_edges) == expected hop count        (P2 check)
    object   : gold answer node appears after the seed in path   (object selection)
    chain    : relation_chain == expected chain                  (Section 13.5)

Goldens are pre-registered/frozen (health_science_small_golden.md written before
this run). Per-question full JSON is written as
test_results/tester-c/health_science_small_<id>.json (Section 5.7 naming).

Usage:
    python test_results/tester-c/health_science_small_runner.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

GOLDEN = [
    {"id": "hsc01", "q": "What supports health?",                "gold": "exercise",      "chain": ["supports"],      "hops": 1},
    {"id": "hsc02", "q": "What is supported by sunlight?",       "gold": "vitamin_d",     "chain": ["supports"],      "hops": 1},
    {"id": "hsc03", "q": "What contradicts health?",             "gold": "smoking",       "chain": ["contradicts"],   "hops": 1},
    {"id": "hsc04", "q": "What does sugar contradict?",          "gold": "healthy_teeth", "chain": ["contradicts"],   "hops": 1},
    {"id": "hsc05", "q": "What is near the heart?",              "gold": "lungs",         "chain": ["spatial_near"],  "hops": 1},
    {"id": "hsc06", "q": "What is located near the liver?",      "gold": "stomach",       "chain": ["spatial_near"],  "hops": 1},
    {"id": "hsc07", "q": "What is the term for cure?",           "gold": "treat",         "chain": ["linguistic_maps"], "hops": 1},
    {"id": "hsc08", "q": "What do you call a doctor?",           "gold": "physician",     "chain": ["linguistic_maps"], "hops": 1},
    {"id": "hsc09", "q": "What leads to fever?",                 "gold": "virus",         "chain": ["causes"],        "hops": 1},
    {"id": "hsc10", "q": "What causes insomnia?",                "gold": "stress",        "chain": ["causes"],        "hops": 1},
    {"id": "hsc11", "q": "What is the heart a part of?",         "gold": "circulatory_system", "chain": ["part_of"],  "hops": 1},
    {"id": "hsc12", "q": "What is the stomach a part of?",       "gold": "digestive_system",   "chain": ["part_of"],  "hops": 1},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "test_results" / "tester-c" / "datasets" / "health_science_small.db"))
    parser.add_argument("--out", default=str(ROOT / "test_results" / "tester-c" / "health_science_small_results.txt"))
    parser.add_argument("--json-dir", default=str(ROOT / "test_results" / "tester-c"))
    args = parser.parse_args()

    pipeline = GLMXPipeline()
    pipeline.graph_store = SQLiteGraphStore.load_state(args.db)
    pipeline.load_models()

    lines = []
    lines.append("=" * 78)
    lines.append("HEALTH_SCIENCE_SMALL (Level 1: full pipeline on health_science_small.db)")
    lines.append("=" * 78)

    summary_ok = 0
    summary_total = 0
    rows = []
    for g in GOLDEN:
        r = pipeline.ask(g["q"])
        path = list(r["walk_path_labels"])
        edges = list(r["walk_path_edges"])
        chain = list(r["relation_chain"])
        hops_ok = len(edges) == g["hops"]
        object_ok = g["gold"] in path[1:] or g["gold"] in str(r["answer"])
        chain_ok = list(chain) == list(g["chain"])
        ok = hops_ok and object_ok

        summary_total += 1
        if ok:
            summary_ok += 1
        rows.append({
            "id": g["id"], "q": g["q"], "gold": g["gold"],
            "hops_expect": g["hops"], "hops": len(edges), "hops_ok": hops_ok,
            "chain_expect": g["chain"], "chain": chain, "chain_ok": chain_ok,
            "object_ok": object_ok, "ok": ok, "path": " -> ".join(path),
            "answer": str(r["answer"]),
            "heuristic_used": bool(r["heuristic_used"]),
            "template_matched": bool(r["template_matched"]),
            "conf": float(r["confidence"]),
            "walk_conf": float(r["walk_confidence"]),
            "full": r,
        })

    for row in rows:
        mark = "PASS" if row["ok"] else "FAIL"
        lines.append(f"\n[{mark}] {row['id']} Q: {row['q']}")
        lines.append(f"    gold={row['gold']!r} hops_expected={row['hops_expect']} | "
                     f"hops={row['hops']} ({'ok' if row['hops_ok'] else 'X'}) | "
                     f"object={'ok' if row['object_ok'] else 'X'} | "
                     f"chain_expected={row['chain_expect']} chain={row['chain']} "
                     f"({'ok' if row['chain_ok'] else 'X'})")
        lines.append(f"    heuristic_used={row['heuristic_used']} template_matched={row['template_matched']} "
                     f"conf={row['conf']:.4f} walk_conf={row['walk_conf']:.4f}")
        lines.append(f"    path: {row['path']}")
        lines.append(f"    answer: {row['answer']}")

        json_dir = Path(args.json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)
        result = dict(row["full"])
        result["_golden_id"] = row["id"]
        result["_golden_gold"] = row["gold"]
        result["_golden_chain"] = row["chain_expect"]
        result["_golden_hops"] = row["hops_expect"]
        result["_pass"] = row["ok"]
        (json_dir / f"health_science_small_{row['id']}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )

    lines.append("\n" + "=" * 78)
    lines.append(f"SUMMARY: {summary_ok}/{summary_total} questions passed")
    lines.append("=" * 78)

    text = "\n".join(lines)
    print(text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved result to {out_path}")


if __name__ == "__main__":
    main()