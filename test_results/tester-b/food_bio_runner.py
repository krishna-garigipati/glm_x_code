"""Level-1 evaluation of the food_bio_small graph through the full pipeline.

Mirrors test_results/lead/toy_eval_runner.py for tester-b's Food & Biology small
dataset. Runs the pre-registered golden set through scripts/glmx_ask (in-process,
exactly equivalent to `python scripts/glmx_ask.py --db food_bio_small.db -q ...`)
and scores each question on:

    hops     : len(walk_path_edges) == expected hop count        (P2 check)
    object   : gold answer node appears after the seed in path   (object selection)
    chain    : relation_chain == expected chain                  (Section 13.5)

Edge-case goldens (Section 13.3, tester-B) get their own checks:
    fbs14 missing relation -> must emit the exact Section 6.4 honest-fallback
          sentence (heuristic_fallback_used=True, empty walk path), never a
          fabricated chain.
    fbs15 duplicate labels -> answer must resolve through a graph where `rice`
          and `Rice` are distinct nodes (no silent merge corruption).

Per-question full JSON is written to test_results/tester-b/food_bio_small_<id>.json
(Section 5.7 naming: <dataset>_<question_id>.json).

Usage:
    python test_results/tester-b/food_bio_runner.py
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
    {"id": "fbs01", "q": "What is a salmon?",                   "gold": "fish",   "chain": ["is_a"],         "hops": 1},
    {"id": "fbs02", "q": "What is a robin?",                    "gold": "bird",   "chain": ["is_a"],         "hops": 1},
    {"id": "fbs03", "q": "What is the yolk a part of?",         "gold": "egg",    "chain": ["part_of"],      "hops": 1},
    {"id": "fbs04", "q": "What is the petal a part of?",        "gold": "flower", "chain": ["part_of"],      "hops": 1},
    {"id": "fbs05", "q": "What property does honey have?",      "gold": "sweet",  "chain": ["has_property"], "hops": 1},
    {"id": "fbs06", "q": "What property does lemon have?",      "gold": "sour",   "chain": ["has_property"], "hops": 1},
    {"id": "fbs07", "q": "Give me an example of a bird",        "gold": "robin",  "chain": ["example_of"],   "hops": 1},
    {"id": "fbs08", "q": "Give me an example of a fish",        "gold": "salmon", "chain": ["example_of"],   "hops": 1},
    {"id": "fbs09", "q": "What is another word for happy?",     "gold": "glad",   "chain": ["synonym"],      "hops": 1},
    {"id": "fbs10", "q": "What is another word for small?",     "gold": "little", "chain": ["synonym"],      "hops": 1},
    {"id": "fbs11", "q": "What is the opposite of hot?",        "gold": "cold",   "chain": ["antonym"],      "hops": 1},
    {"id": "fbs12", "q": "What is the opposite of sweet?",      "gold": "sour",   "chain": ["antonym"],      "hops": 1},
    # causal golden (extra relation `causes`, Section 5.4)
    {"id": "fbs13", "q": "What leads to tooth decay?",          "gold": "sugar",  "chain": ["causes"],       "hops": 1},
    # edge cases (Section 13.3)
    {"id": "fbs14", "q": "What follows photosynthesis?",        "gold": FALLBACK_SENTENCE,
     "chain": ["has_property"], "hops": 0, "edge_case": "missing_relation"},
    {"id": "fbs15", "q": "What is rice?",                       "gold": "grain",  "chain": ["is_a"],         "hops": 1,
     "edge_case": "duplicate_labels"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_small.db"))
    parser.add_argument("--out", default=str(ROOT / "test_results" / "tester-b" / "food_bio_small_results.txt"))
    parser.add_argument("--json-dir", default=str(ROOT / "test_results" / "tester-b"))
    args = parser.parse_args()

    pipeline = GLMXPipeline()
    pipeline.graph_store = SQLiteGraphStore.load_state(args.db)
    pipeline.load_models()

    lines = []
    lines.append("=" * 78)
    lines.append("FOOD_BIO_SMALL (Level 1: full pipeline on food_bio_small.db)")
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
            # honest fallback: heuristic fallback fired AND empty walk path AND exact sentence
            ok = (
                bool(r["heuristic_used"])
                and len(edges) == 0
                and str(r["answer"]).strip() == FALLBACK_SENTENCE
            )
            note = "honest-fallback (fallback fired + empty walk + exact sentence)"
        elif g.get("edge_case") == "duplicate_labels":
            store = pipeline.graph_store
            ids = (store._label_to_id.get("rice"), store._label_to_id.get("Rice"))
            dup_ok = None not in ids and ids[0] != ids[1]
            ok = hops_ok and object_ok and dup_ok
            note = f"duplicate labels distinct ids rice={ids[0]} Rice={ids[1]}"
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

        # Section 5.7: per-question full JSON result
        json_dir = Path(args.json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)
        result = dict(row["full"])
        result["_golden_id"] = row["id"]
        result["_golden_gold"] = row["gold"]
        result["_golden_chain"] = row["chain_expect"]
        result["_golden_hops"] = row["hops_expect"]
        result["_pass"] = row["ok"]
        (json_dir / f"food_bio_small_{row['id']}.json").write_text(
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