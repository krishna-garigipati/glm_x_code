"""Contract section 12 determinism: same graph + same question -> identical answer."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
a = json.loads((HERE / "_run1.json").read_text(encoding="utf-8"))
b = json.loads((HERE / "stage_a_results.json").read_text(encoding="utf-8"))

print("run1:", a["n_questions"], "rows | run2:", b["n_questions"], "rows")

keys = ["answer", "relation_chain", "path_edges", "path_labels",
        "selected_anchor", "honest_no_relation", "heuristic_used",
        "heuristic_disclosed", "invariant_ok", "passed"]

ra = {r["id"]: r for r in a["rows"]}
rb = {r["id"]: r for r in b["rows"]}

if set(ra) != set(rb):
    print("ROW SET DIFFERS:", set(ra) ^ set(rb))
else:
    print("row ids identical:", len(ra))

diffs = []
for qid in sorted(ra):
    for k in keys:
        if ra[qid].get(k) != rb[qid].get(k):
            diffs.append((qid, k, ra[qid].get(k), rb[qid].get(k)))

print("\ndiffering fields:", len(diffs))
for qid, k, x, y in diffs[:40]:
    print("  {} . {}\n      run1: {}\n      run2: {}".format(qid, k, x, y))

print("\nDETERMINISTIC" if not diffs else "\nNON-DETERMINISTIC")

# Timing is expected to vary; confirm it is the only moving part.
t1 = sum(r.get("time_seconds") or 0 for r in a["rows"])
t2 = sum(r.get("time_seconds") or 0 for r in b["rows"])
print("wall time run1={:.1f}s run2={:.1f}s (expected to differ; not an answer field)".format(t1, t2))