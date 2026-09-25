"""Re-run tester-a's frozen golden set against the (rebuilt) nature_weather_small.db.

Regenerates test_results/tester-a/nature_weather_small_results.json in the exact
schema used on 2026-09-21 (header from live DB stats + graded per-question rows).
Golden questions are NOT edited (frozen in nature_weather_small_golden.md); only the
DB and the generated results change. Run after
    python test_results/tester-a/datasets/build_nature_weather_small.py

Usage:
    python test_results/tester-a/nature_weather_small_runner.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

GOLDEN = [
    {"q": "What causes flood?",             "chain": ["causes"],          "nodes": ["rain"],                                        "hops": 1},
    {"q": "What causes rain?",              "chain": ["caused_by"],       "nodes": ["cloud", "storm"],                              "hops": 1},
    {"q": "What comes after summer?",       "chain": ["follows"],         "nodes": ["autumn"],                                       "hops": 1},
    {"q": "What comes before summer?",      "chain": ["precedes"],        "nodes": ["spring"],                                       "hops": 1},
    {"q": "What is the opposite of sun?",   "chain": ["antonym"],         "nodes": ["cloud"],                                        "hops": 1},
    {"q": "What is associated with rain?",  "chain": ["associated_with"], "nodes": ["cloud"],                                        "hops": 1},
    {"q": "What is rain?",                  "chain": ["is_a"],            "nodes": ["water"],                                        "hops": 1},
    {"q": "What is part of a storm?",       "chain": ["part_of"],         "nodes": ["lightning"],                                    "hops": 1},
    {"q": "What does lightning cause?",     "chain": ["causes"],          "nodes": ["fire"],                                         "hops": 1},
    {"q": "What comes before winter?",      "chain": ["precedes"],        "nodes": ["autumn"],                                       "hops": 1},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "test_results" / "tester-a" / "datasets" / "nature_weather_small.db"))
    parser.add_argument("--out", default=str(ROOT / "test_results" / "tester-a" / "nature_weather_small_results.json"))
    args = parser.parse_args()

    store = SQLiteGraphStore.load_state(args.db)
    pipeline = GLMXPipeline()
    pipeline.graph_store = store
    pipeline.load_models()

    summary_ok = summary_total = chain_exact = 0
    results = []
    for num, g in enumerate(GOLDEN, start=1):
        r = pipeline.ask(g["q"])
        path = list(r["walk_path_labels"])
        edges = list(r["walk_path_edges"])
        chain = list(r["relation_chain"])
        hops_ok = len(edges) == g["hops"]
        node_hit = any(n in path[1:] or n in str(r["answer"]).lower() for n in g["nodes"])
        exact = list(chain) == list(g["chain"])
        ok = hops_ok and node_hit
        summary_total += 1
        summary_ok += int(ok)
        chain_exact += int(exact)
        results.append({
            "num": num,
            "question": g["q"],
            "expected_chain": g["chain"],
            "expected_answer_nodes": g["nodes"],
            "extracted_chain": chain,
            "chain_exact": bool(exact),
            "hops": len(edges),
            "hops_expected": g["hops"],
            "hops_ok": bool(hops_ok),
            "object_ok": bool(node_hit),
            "pass": bool(ok),
            "answer": str(r["answer"]),
            "walk_path_labels": [str(x) for x in path],
            "template_matched": bool(r["template_matched"]),
            "heuristic_used": bool(r.get("heuristic_used", r.get("heuristic_fallback_used", False))),
            "honest_by_entity": bool(r.get("honest_by_entity", False)),
            "honest_by_relation": bool(r.get("honest_by_relation", False)),
            "honest_no_relation": bool(r.get("honest_no_relation", False)),
            "confidence": float(r.get("confidence", 0.0)),
            "walk_confidence": float(r.get("walk_confidence", 0.0)),
            "entity_top_sim": float(r.get("entity_top_sim", 0.0)),
        })

    relation_set = sorted(store.get_all_relations())
    doc = {
        "dataset": "nature_weather_small.db",
        "domain": "Nature & Weather (Tester A)",
        "nodes": store.get_node_count(),
        "edges": store.get_edge_count(),
        "relations_in_graph": relation_set,
        "run": "2026-09-25 on dharani (IS-04/05 reconcile: DB rebuilt from 53-triple JSON)",
        "grading": "pass = hops==expected AND expected answer node appears in walk path or answer",
        "summary": {
            "total": summary_total,
            "passed": summary_ok,
            "failed": summary_total - summary_ok,
            "chain_exact": chain_exact,
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"Saved {args.out}")
    print(f"SUMMARY: {summary_ok}/{summary_total} passed | chain_exact={chain_exact}")
    for res in results:
        mark = "PASS" if res["pass"] else "FAIL"
        print(f"[{mark}] #{res['num']} {res['question']} | chain={'/'.join(res['extracted_chain'])} "
              f"(exact={res['chain_exact']}) | hops={res['hops']} object={res['object_ok']} "
              f"| path={' -> '.join(res['walk_path_labels'])} | {res['answer']}")


if __name__ == "__main__":
    main()