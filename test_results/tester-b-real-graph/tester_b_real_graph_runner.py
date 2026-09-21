"""Level-1 evaluation of the tester-b-real-graph DB through the full GLM-X pipeline.

The graph was built from a plain-text corpus by kg_builder (no curated relation
insertion): test_results/tester-b-real-graph/datasets/food_biology_source.txt ->
routes/test:build_real_graph.py -> tester_b_real_graph.db. This runner executes the
pre-registered golden set (tester_b_real_graph_golden.md) via scripts/glmx_ask and
scores each question on:

    hops     : len(walk_path_edges) == expected hop count        (P2 check)
    object   : gold answer node appears after the seed in path   (object selection)
    chain    : relation_chain == expected chain                  (Section 13.5; logged)

Per-question full JSON is written to test_results/tester-b-real-graph/
tester_b_real_graph_<id>.json.

Usage:
    python test_results/tester-b-real-graph/tester_b_real_graph_runner.py
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
    {"id": "trg01", "q": "What is a salmon?",                "gold": "a fish",              "chain": ["is_a"],              "hops": 1},
    {"id": "trg02", "q": "What property does gold have?",    "gold": "high density",       "chain": ["has_property"],      "hops": 1},
    {"id": "trg03", "q": "What does sugar cause?",           "gold": "tooth decay",        "chain": ["causes"],            "hops": 1},
    {"id": "trg04", "q": "What is lung cancer caused by?",   "gold": "smoking",            "chain": ["caused_by"],         "hops": 1},
    {"id": "trg05", "q": "What follows night?",               "gold": "day",                "chain": ["follows"],           "hops": 1},
    {"id": "trg06", "q": "What precedes summer?",             "gold": "spring",             "chain": ["precedes"],          "hops": 1},
    {"id": "trg07", "q": "What does science contradict?",     "gold": "superstition",       "chain": ["contradicts"],       "hops": 1},
    {"id": "trg08", "q": "What does calcium support?",        "gold": "bone health",        "chain": ["supports"],          "hops": 1},
    {"id": "trg09", "q": "What is smoking associated with?",  "gold": "lung cancer",        "chain": ["associated_with"],   "hops": 1},
    {"id": "trg10", "q": "What is an example of a fish?",     "gold": "salmon",             "chain": ["example_of"],        "hops": 1},
    {"id": "trg11", "q": "What is part of the circulatory system?", "gold": "the heart",    "chain": ["part_of"],           "hops": 1},
    {"id": "trg12", "q": "What is a synonym of insomnia?",    "gold": "sleeplessness",      "chain": ["synonym"],           "hops": 1},
    {"id": "trg13", "q": "What is the opposite of day?",      "gold": "night",              "chain": ["antonym"],           "hops": 1},
    {"id": "trg14", "q": "What does night coincide with?",    "gold": "dusk",               "chain": ["temporal_coincident"], "hops": 1},
    {"id": "trg15", "q": "What is near the heart?",           "gold": "the lungs",          "chain": ["spatial_near"],      "hops": 1},
    {"id": "trg16", "q": "What does casa translate to?",      "gold": "house",              "chain": ["linguistic_maps"],   "hops": 1},
    {"id": "trg17", "q": "What leads to tooth decay?",        "gold": "sugar",              "chain": ["causes"],            "hops": 1},
    {"id": "trg18", "q": "What causes lung cancer?",          "gold": "smoking",            "chain": ["causes"],            "hops": 1},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "test_results" / "tester-b-real-graph" / "tester_b_real_graph.db"))
    parser.add_argument("--out", default=str(ROOT / "test_results" / "tester-b-real-graph" / "tester_b_real_graph_results.txt"))
    parser.add_argument("--json-dir", default=str(ROOT / "test_results" / "tester-b-real-graph"))
    args = parser.parse_args()

    pipeline = GLMXPipeline()
    pipeline.graph_store = SQLiteGraphStore.load_state(args.db)
    pipeline.load_models()

    lines = []
    lines.append("=" * 78)
    lines.append("TESTER-B-REAL-GRAPH (Level 1: full pipeline on tester_b_real_graph.db)")
    lines.append("Graph built by kg_builder from food_biology_source.txt; relations normalised to canonical 16.")
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
        (json_dir / f"tester_b_real_graph_{row['id']}.json").write_text(
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