"""Level-1 evaluation of the toy_eval graph through the full pipeline.

Runs the golden question set through scripts/glmx_ask (in-process, exactly
mirrors `python scripts/glmx_ask.py --db toy_eval.db -q ...`) and scores each
question on two criteria:

    hops     : len(walk_path_edges) == expected hop count       (P2 check)
    object   : gold answer node appears after the seed in path  (object selection)

Usage:
    python test_results/lead/toy_eval_runner.py --out <result.txt>
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

GOLDEN = [
    {"q": "What is a dog?",                    "gold": "animal",  "hops": 1},
    {"q": "What is the opposite of hot?",       "gold": "cold",    "hops": 1},
    {"q": "What does rain cause?",              "gold": "flood",   "hops": 1},
    {"q": "What property does water have?",     "gold": "liquid",  "hops": 1},
    {"q": "What is ice?",                       "gold": "cold",    "hops": 1},
    {"q": "Tell me something related to water", "gold": "rain",    "hops": 1},
    {"q": "What does fire cause?",              "gold": "smoke",   "hops": 1},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "test_results" / "lead" / "datasets" / "toy_eval.db"))
    parser.add_argument("--out", required=True, help="Result text file path")
    args = parser.parse_args()

    pipeline = GLMXPipeline()
    pipeline.graph_store = SQLiteGraphStore.load_state(args.db)
    pipeline.load_models()

    lines = []
    lines.append("=" * 78)
    lines.append("TOY EVAL  (Level 1: full pipeline on toy_eval.db)")
    lines.append("=" * 78)

    summary_ok = 0
    summary_total = 0
    rows = []
    for g in GOLDEN:
        r = pipeline.ask(g["q"])
        path = list(r["walk_path_labels"])
        edges = list(r["walk_path_edges"])
        hops_ok = len(edges) == g["hops"]
        object_ok = g["gold"] in path[1:] or g["gold"] in str(r["answer"])
        ok = hops_ok and object_ok
        summary_total += 1
        if ok:
            summary_ok += 1
        rows.append({
            "q": g["q"], "gold": g["gold"], "hops_expect": g["hops"],
            "hops": len(edges), "hops_ok": hops_ok, "object_ok": object_ok,
            "ok": ok, "path": " -> ".join(path),
            "answer": str(r["answer"]),
            "chain": list(r["relation_chain"]),
            "conf": float(r["walk_confidence"]),
        })

    for row in rows:
        mark = "PASS" if row["ok"] else "FAIL"
        lines.append(f"\n[{mark}] Q: {row['q']}")
        lines.append(f"    gold={row['gold']!r} hops_expected={row['hops_expect']} | "
                     f"hops={row['hops']} ({'ok' if row['hops_ok'] else 'X'}) | "
                     f"object={row['gold'] in row['path']} ({'ok' if row['object_ok'] else 'X'})")
        lines.append(f"    chain={row['chain']} walk_confidence={row['conf']:.4f}")
        lines.append(f"    path: {row['path']}")
        lines.append(f"    answer: {row['answer']}")

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