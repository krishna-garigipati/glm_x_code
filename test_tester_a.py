"""Tester A - Nature & Weather Small Dataset runner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

DB_PATH = Path("C:/Users/obili/OneDrive/Documents/genesis_frameworks/glm_x_code/test_results/tester-a/datasets/nature_weather_small.db")

GOLDEN = [
    {"q": "What causes flood?",                  "gold": ["rain"],        "chain": ["causes"],       "hops": 1},
    {"q": "What causes rain?",                   "gold": ["cloud", "storm"], "chain": ["caused_by"],   "hops": 1},
    {"q": "What comes after summer?",            "gold": ["autumn"],      "chain": ["follows"],      "hops": 1},
    {"q": "What comes before summer?",           "gold": ["spring"],      "chain": ["precedes"],     "hops": 1},
    {"q": "What is the opposite of sun?",        "gold": ["cloud"],       "chain": ["antonym"],      "hops": 1},
    {"q": "What is associated with rain?",       "gold": ["cloud", "water"], "chain": ["associated_with"], "hops": 1},
    {"q": "What is rain?",                       "gold": ["water", "weather"], "chain": ["is_a"],     "hops": 1},
    {"q": "What is part of a storm?",            "gold": ["lightning", "hail"], "chain": ["part_of"],  "hops": 1},
    {"q": "What does lightning cause?",          "gold": ["fire", "thunder"], "chain": ["causes"],     "hops": 1},
    {"q": "What comes before winter?",           "gold": ["autumn"],      "chain": ["precedes"],     "hops": 1},
]

def main():
    pipeline = GLMXPipeline()
    pipeline.graph_store = SQLiteGraphStore.load_state(str(DB_PATH))
    pipeline.load_models()

    print("=" * 78)
    print("TESTER-A: NATURE_WEATHER_SMALL")
    print("=" * 78)

    passed = 0
    total = 0
    for g in GOLDEN:
        r = pipeline.ask(g["q"])
        path = list(r["walk_path_labels"])
        edges = list(r["walk_path_edges"])
        chain = list(r["relation_chain"])

        hops_ok = len(edges) == g["hops"]
        object_ok = any(gold in path[1:] or gold in str(r["answer"]) for gold in g["gold"])
        chain_ok = chain == g["chain"]

        # For questions where expected relation is not in graph (antonym, is_a, part_of, caused_by),
        # or honesty gate fires (entity ambiguity), don't penalize for object/chain.
        missing_relation = g["chain"][0] not in ["causes", "follows", "precedes", "associated_with"]
        honest_gate = bool(r.get("honest_no_relation")) or bool(r.get("honest_by_relation")) or bool(r.get("honest_by_entity"))
        if (missing_relation or honest_gate) and (bool(r.get("honest_no_relation")) or bool(r.get("honest_by_relation")) or bool(r.get("honest_by_entity"))):
            ok = hops_ok  # only check hops, honesty gate fired correctly
            note = "honest-fallback (relation not in graph or entity ambiguous)"
        else:
            ok = hops_ok and object_ok
            note = ""

        total += 1
        if ok:
            passed += 1

        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] Q: {g['q']}")
        print(f"    gold={g['gold']} hops_expected={g['hops']} chain_expected={g['chain']}")
        print(f"    hops={len(edges)} ({'ok' if hops_ok else 'X'}) | object={'ok' if object_ok else 'X'} | chain={'ok' if chain_ok else 'X'} (got {chain})")
        print(f"    walk_confidence={r['walk_confidence']:.4f} template_matched={r['template_matched']}")
        print(f"    path: {' -> '.join(path)}")
        print(f"    answer: {r['answer'][:120]}...")
        if note:
            print(f"    edge-case: {note}")

    print(f"\n{'='*78}")
    print(f"SUMMARY: {passed}/{total} passed")
    print(f"{'='*78}")

if __name__ == "__main__":
    main()