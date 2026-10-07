"""Instrument the live pipeline to measure which mechanisms actually fire.

Behaviour-neutral. Counts, per question, how the answer was reached and whether
any hop was backed by a real stored edge or by a CONTRACT-LEGAL mirror.

Note: the former activation-rescue counters were removed together with the
rescue itself. There is now no path that hands the walker an edge it did not
earn from resonance ranking, so "answers unbacked by a stored edge" is the only
integrity signal that matters here.
"""
import json
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

STAGE = Path(__file__).resolve().parent
DB = STAGE / "relation_coverage.db"
QSET = STAGE / "relation_coverage_questions_frozen.json"

random.seed(0)
np.random.seed(0)

pipe = GLMXPipeline()
pipe._seed = 0
pipe._no_learning = True
pipe._measure = True
pipe.graph_store = SQLiteGraphStore.load_state(str(DB))
pipe.load_models()

# stored triples + their declared inverse pairs
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
_lab = {i: (l or "").strip().lower() for i, l in con.execute("SELECT id,label FROM nodes")}
STORED = {(_lab.get(s, ""), r, _lab.get(t, "")) for s, t, r in
          con.execute("SELECT source_id,target_id,relation FROM edges")}
con.close()

INVERSE = {"causes": "caused_by", "caused_by": "causes", "precedes": "follows",
           "follows": "precedes", "part_of": "has_part", "has_part": "part_of"}

rows = []
for q in json.loads(QSET.read_text(encoding="utf-8"))["questions"]:
    res = pipe.ask(q["q"])
    labels = [str(x) for x in (res.get("walk_path_labels") or [])]
    edges = [str(x) for x in (res.get("walk_path_edges") or [])]
    kinds = []
    for i, e in enumerate(edges):
        if i + 1 >= len(labels):
            continue
        a, b = labels[i].lower(), labels[i + 1].lower()
        if (a, e, b) in STORED:
            kinds.append("stored")
        elif INVERSE.get(e) and (b, INVERSE[e], a) in STORED:
            kinds.append("declared-inverse")
        elif (b, e, a) in STORED:
            kinds.append("ILLEGAL-same-label-mirror")
        else:
            kinds.append("ILLEGAL-fabricated")
    rows.append({
        "id": q["id"], "cat": q["cat"], "q": q["q"],
        "chain": res.get("relation_chain"),
        "anchor": (res.get("selected_anchor") or {}).get("label"),
        "path": labels, "edges": edges, "hop_kinds": kinds,
        "heuristic": bool(res.get("heuristic_used")),
        "honest": bool(res.get("honest_no_relation")),
        "answer": res.get("answer"),
    })

out = STAGE / "audit_mechanisms.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

n = len(rows)
illegal = [r for r in rows if any(k.startswith("ILLEGAL") for k in r["hop_kinds"])]
print("questions                     :", n)
print("heuristic fallback fired      :", sum(1 for r in rows if r["heuristic"]))
print("honest refusals               :", sum(1 for r in rows if r["honest"]))
print("hops: stored                  :", sum(1 for r in rows for k in r["hop_kinds"] if k == "stored"))
print("hops: declared-inverse        :", sum(1 for r in rows for k in r["hop_kinds"] if k == "declared-inverse"))
print("hops: ILLEGAL same-label      :", sum(1 for r in rows for k in r["hop_kinds"] if k == "ILLEGAL-same-label-mirror"))
print("hops: ILLEGAL fabricated      :", sum(1 for r in rows for k in r["hop_kinds"] if k == "ILLEGAL-fabricated"))
print("questions with an ILLEGAL hop :", len(illegal))
for r in illegal:
    print("   ", r["id"], r["hop_kinds"])
print("wrote", out)