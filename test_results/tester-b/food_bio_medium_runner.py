"""Level-1 evaluation of the food_bio_medium graph through the full pipeline.

Mirrors food_bio_runner.py for the Medium Food & Biology dataset. Runs the
pre-registered golden set (food_bio_medium_golden.md, FROZEN) through
scripts/glmx_ask (in-process, equivalent to `--db food_bio_medium.db -q ...`)
and scores hops + object (+ recorded chain). Tester-B edge cases:

    fbm19 missing relation -> must emit the exact Section 6.4 honest-fallback
          sentence (heuristic_fallback_used=True, empty walk). Design-time verified:
          clause best-matches the absent `spatial_near`, isolated node `plankton`.
    fbm20 duplicate labels -> `corn` / `Corn` distinct nodes, no silent merge.

Per-question full JSON -> test_results/tester-b/food_bio_medium_<id>.json.

Usage:
    python test_results/tester-b/food_bio_medium_runner.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

FALLBACK_SENTENCE = "I don't have a relation in my knowledge graph that fully answers this question."

GOLDEN = [
    {"id": "fbm01", "q": "What is a mammal?",                    "gold": "animal", "chain": ["is_a"],            "hops": 1},
    {"id": "fbm02", "q": "What is a penguin?",                  "gold": "bird",   "chain": ["is_a"],            "hops": 1},
    {"id": "fbm03", "q": "What is a salmon?",                   "gold": "fish",   "chain": ["is_a"],            "hops": 1},
    {"id": "fbm04", "q": "What is the yolk a part of?",         "gold": "egg",    "chain": ["part_of"],         "hops": 1},
    {"id": "fbm05", "q": "What is the petal a part of?",        "gold": "flower", "chain": ["part_of"],         "hops": 1},
    {"id": "fbm06", "q": "What is the wing a part of?",         "gold": "bird",   "chain": ["part_of"],         "hops": 1},
    {"id": "fbm07", "q": "What property does honey have?",      "gold": "sweet",  "chain": ["has_property"],    "hops": 1},
    {"id": "fbm08", "q": "What property does chili have?",      "gold": "spicy",  "chain": ["has_property"],    "hops": 1},
    {"id": "fbm09", "q": "What property does snow have?",       "gold": "cold",   "chain": ["has_property"],    "hops": 1},
    {"id": "fbm10", "q": "Give me an example of a bird",        "gold": "robin",  "chain": ["example_of"],      "hops": 1},
    {"id": "fbm11", "q": "Give me an example of a fish",        "gold": "salmon", "chain": ["example_of"],      "hops": 1},
    {"id": "fbm12", "q": "What is another word for big?",       "gold": "large",  "chain": ["synonym"],         "hops": 1},
    {"id": "fbm13", "q": "What is another word for quick?",     "gold": "fast",   "chain": ["synonym"],         "hops": 1},
    {"id": "fbm14", "q": "What is the opposite of dark?",       "gold": "light",  "chain": ["antonym"],         "hops": 1},
    {"id": "fbm15", "q": "What is the opposite of wet?",        "gold": "dry",    "chain": ["antonym"],         "hops": 1},
    # causal band + added relations
    {"id": "fbm16", "q": "What leads to tooth decay?",          "gold": "sugar",  "chain": ["causes"],          "hops": 1},
    {"id": "fbm17", "q": "What is tooth decay caused by?",      "gold": "sugar",  "chain": ["caused_by"],       "hops": 1},
    {"id": "fbm18", "q": "Tell me something associated with water", "gold": "rain", "chain": ["associated_with"], "hops": 1},
    # edge cases (Section 13.3)
    {"id": "fbm19", "q": "What organism lives close to plankton?",
     "gold": FALLBACK_SENTENCE, "chain": ["has_property"], "hops": 0,
     "edge_case": "missing_relation"},
    {"id": "fbm20", "q": "What is corn?",                       "gold": "grain",  "chain": ["is_a"],            "hops": 1,
     "edge_case": "duplicate_labels"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_medium.db"))
    parser.add_argument("--out", default=str(ROOT / "test_results" / "tester-b" / "food_bio_medium_results.txt"))
    parser.add_argument("--json-dir", default=str(ROOT / "test_results" / "tester-b"))
    args = parser.parse_args()

    pipeline = GLMXPipeline()
    pipeline.graph_store = SQLiteGraphStore.load_state(args.db)
    pipeline.load_models()

    lines = []
    lines.append("=" * 78)
    lines.append("FOOD_BIO_MEDIUM (Level 1: full pipeline on food_bio_medium.db)")
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

        if g.get("edge_case") == "missing_relation":
            answer_text = str(r["answer"]).strip()
            honest_gate = bool(r.get("honest_no_relation")) or bool(r.get("honest_by_relation"))
            ok = (
                honest_gate
                and len(edges) == 0
                and FALLBACK_SENTENCE in answer_text
            )
            note = "honest-fallback (honesty gate fired + empty walk + fallback sentence contained)"
        elif g.get("edge_case") == "duplicate_labels":
            store = pipeline.graph_store
            ids = (store._label_to_id.get("corn"), store._label_to_id.get("Corn"))
            dup_ok = None not in ids and ids[0] != ids[1]
            ok = hops_ok and object_ok and dup_ok
            note = f"duplicate labels distinct ids corn={ids[0]} Corn={ids[1]}"
        else:
            ok = hops_ok and object_ok
            note = ""

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
            "note": note,
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
        if row["note"]:
            lines.append(f"    edge-case: {row['note']}")

        json_dir = Path(args.json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)
        result = dict(row["full"])
        result.update({
            "_golden_id": row["id"], "_golden_gold": row["gold"],
            "_golden_chain": row["chain_expect"], "_golden_hops": row["hops_expect"],
            "_pass": row["ok"],
        })
        (json_dir / f"food_bio_medium_{row['id']}.json").write_text(
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